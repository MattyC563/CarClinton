"""Race strategy optimiser for the sprint (25 min), feature (40 min) and endurance (60 min) races.

For each race it tries every combination of 0..max_stops pit stops (stop laps, compound per
stint, fuel loaded per stint) and picks the fastest. No-stop is always evaluated.

Lap time model  = tyre curve from the random forest (fuel-corrected)
                  + fuel weight (same seconds_per_kg as lap_analysis.py)
                  + LICO time cost, if lift-and-coast is used.

LICO rule: if the car would have to stop with <= 2 laps to go, the stop is dropped and the
fuel saving needed to reach the flag is reported instead (total litres, litres/lap, %).

Most of the numbers that aren't in your logs (pit loss, refuel rate, LICO cost) are
PLACEHOLDERS in StrategyParams - replace them with your series' real values.
"""
import itertools
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

RACES = {"Sprint": 25 * 60, "Feature": 40 * 60, "Endurance": 60 * 60}   # seconds


@dataclass
class StrategyParams:
    # --- must match what you used in lap_analysis.py ---
    seconds_per_kg: float = 0.03
    fuel_density_kg_per_l: float = 0.745
    # --- pit stops (PLACEHOLDERS) ---
    pit_lane_loss_s: float = 20.0        # time lost driving through the pit lane
    tyre_change_s: float = 10.0          # stationary time for a tyre change
    refuel_rate_l_per_s: float = 2.0     # stop lasts max(tyre change, litres / rate)
    # --- fuel ---
    fuel_margin_l: float = 0.5           # reserve left at the end of every stint
    tank_l: Optional[float] = None       # None = read "Max Fuel" from the logs
    burn_l_per_lap: Optional[float] = None   # None = average from the logs
    # --- lift and coast ---
    lico_s_per_l: float = 3.0            # PLACEHOLDER: seconds lost per litre saved
    lico_max_laps_to_go: int = 2         # only consider LICO if the stop would be this late
    # --- search ---
    max_stops: int = 3
    min_stint_laps: int = 3
    track_temp: Optional[float] = None   # None = baseline from the logs
    compounds: Optional[Tuple[str, ...]] = None   # None = every compound in the data


def fmt_race_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h else f"{int(m)}:{s:04.1f}"


class Planner:
    def __init__(self, result, data, params=None):
        self.P = params or StrategyParams()
        P = self.P
        temp = result["baseline_track_temp"] if P.track_temp is None else P.track_temp
        self.curves = self._build_curves(result, temp, P.compounds)
        if not self.curves:
            raise ValueError("No tyre curves available for the requested compounds.")

        self.b = P.burn_l_per_lap or float(data["FuelUsed_l"].mean())
        self.C = P.tank_l or float(data["MaxFuel_l"].max())
        if not (self.b > 0 and self.C > 0):
            raise ValueError("Couldn't work out fuel burn / tank size - set burn_l_per_lap and tank_l.")
        self.range_laps = math.floor((self.C - P.fuel_margin_l) / self.b + 1e-9)
        self._cache = {}

    # ------------------------------------------------------------------ tyres
    @staticmethod
    def _build_curves(result, temp, compounds):
        curves = {}
        for comp, num in result["tyres_to_num"].items():
            if compounds and comp not in compounds:
                continue
            if comp not in result["life_range"].index:
                continue
            lo, hi = (int(x) for x in result["life_range"].loc[comp])
            life = np.arange(lo, hi + 1)
            X = pd.DataFrame({"TyreLife": life, "Compound": num, "TrackTemp": temp})
            pred = result["model"].predict(X)
            if len(life) > 1:
                # tyres only get slower: stops the optimiser exploiting noise in the forest
                pred = IsotonicRegression(increasing=True).fit_transform(life, pred)
                slope = max(float(np.polyfit(life, pred, 1)[0]), 0.0)
            else:
                slope = 0.0
            curves[comp] = {"curve": pred, "slope": slope}
        return curves

    def _tyre_time(self, comp, j):
        """Fuel-corrected lap time on lap j (1-based) of a stint on fresh tyres."""
        c = self.curves[comp]
        n = len(c["curve"])
        if j <= n:
            return c["curve"][j - 1]
        return c["curve"][-1] + c["slope"] * (j - n)   # beyond the data: straight-line extrapolation

    # ------------------------------------------------------------------ stints
    def _stint(self, comp, laps, sigma):
        key = (comp, laps, round(sigma, 6))
        if key not in self._cache:
            P = self.P
            burn = self.b - sigma
            start = laps * burn + P.fuel_margin_l
            j = np.arange(1, laps + 1)
            fuel_avg_l = start - burn * (j - 0.5)
            base = np.array([self._tyre_time(comp, x) for x in j])
            t = base + P.seconds_per_kg * P.fuel_density_kg_per_l * fuel_avg_l + P.lico_s_per_l * sigma
            self._cache[key] = (t, float(t.sum()))
        return self._cache[key]

    def _fuel_plan(self, laps):
        """LICO saving (L/lap) needed per stint, or None if the stint can't be done."""
        P = self.P
        sig = []
        for i, l in enumerate(laps):
            if l <= self.range_laps:
                sig.append(0.0)
            elif i == len(laps) - 1 and l - self.range_laps <= P.lico_max_laps_to_go:
                sig.append(self.b - (self.C - P.fuel_margin_l) / l)   # final stint only
            else:
                return None
        return sig

    def _fuel_starts(self, laps, sig):
        return [l * (self.b - s) + self.P.fuel_margin_l for l, s in zip(laps, sig)]

    def _stop_time(self, litres_added):
        P = self.P
        return P.pit_lane_loss_s + max(P.tyre_change_s, litres_added / P.refuel_rate_l_per_s)

    def _splits(self, N, k):
        P = self.P
        min_last = max(P.min_stint_laps, P.lico_max_laps_to_go + 1)   # no stops with <=2 laps to go
        for stops in itertools.combinations(range(1, N), k):
            bounds = (0,) + stops + (N,)
            laps = [bounds[i + 1] - bounds[i] for i in range(k + 1)]
            if any(l < P.min_stint_laps for l in laps[:-1]) or laps[-1] < min_last:
                continue
            yield stops, laps

    # ------------------------------------------------------------------ search
    def search(self, N):
        """Best strategy for each number of stops (only feasible ones)."""
        best = {}
        for k in range(self.P.max_stops + 1):
            for stops, laps in self._splits(N, k):
                sig = self._fuel_plan(laps)
                if sig is None:
                    continue
                starts = self._fuel_starts(laps, sig)
                stop_t = sum(self._stop_time(starts[i + 1] - self.P.fuel_margin_l) for i in range(k))
                for comps in itertools.product(self.curves, repeat=k + 1):
                    total = stop_t + sum(self._stint(c, l, s)[1] for c, l, s in zip(comps, laps, sig))
                    if k not in best or total < best[k][0]:
                        best[k] = (total, stops, laps, sig, comps)
        return {k: self._detail(N, *v[1:]) for k, v in best.items()}

    def _detail(self, N, stops, laps, sig, comps):
        starts = self._fuel_starts(laps, sig)
        lap_times, stop_times = [], []
        for i, (c, l, s) in enumerate(zip(comps, laps, sig)):
            t = self._stint(c, l, s)[0].copy()
            if i < len(laps) - 1:
                st = self._stop_time(starts[i + 1] - self.P.fuel_margin_l)
                stop_times.append(st)
                t[-1] += st                      # the stop happens at the end of the stint's last lap
            lap_times.extend(t)
        cum = np.cumsum(lap_times)
        extrap = any(l > len(self.curves[c]["curve"]) for c, l in zip(comps, laps))
        return {
            "stops": len(stops), "stop_laps": stops, "compounds": comps, "stint_laps": laps,
            "fuel_start_l": starts, "sigma": sig, "stop_times": stop_times,
            "cum": cum, "total": float(cum[-1]), "extrapolated": extrap,
            "laps_to_go_without_lico": laps[-1] - self.range_laps if sig[-1] > 0 else None,
        }

    # ------------------------------------------------------------------ one race
    def plan_race(self, name, duration_s):
        P = self.P
        # first guess of the lap count from a typical lap, then refine
        comp0 = min(self.curves, key=lambda c: self.curves[c]["curve"][0])
        t0 = self.curves[comp0]["curve"][0] + P.seconds_per_kg * P.fuel_density_kg_per_l * self.C / 2
        N, seen, solved = max(1, math.ceil(duration_s / t0)), [], {}

        def solve(n):
            if n not in solved:
                by_k = self.search(n)
                if not by_k:
                    raise RuntimeError(f"{name}: no feasible strategy within {P.max_stops} stops "
                                       f"(fuel lasts {self.range_laps} laps per stint).")
                solved[n] = (by_k, min(by_k.values(), key=lambda d: d["total"]))
            return solved[n]

        while True:
            by_k, best = solve(N)
            cum = best["cum"]
            # timed race: it ends on the first line crossing after the clock runs out
            if N > 1 and cum[N - 2] >= duration_s:
                nxt = N - 1
            elif cum[-1] < duration_s:
                nxt = N + 1
            else:
                break
            if nxt in seen or nxt == N:
                # flip-flopping between two lap counts: take the longer distance
                N = max(N, nxt)
                by_k, best = solve(N)
                break
            seen.append(N)
            N = nxt
        return {"name": name, "duration_s": duration_s, "laps": N, "by_stops": by_k,
                "best": best, "planner": self}


def plan_all_races(result, data, params=None, races=None):
    planner = Planner(result, data, params)
    return [planner.plan_race(n, d) for n, d in (races or RACES).items()]


# ---------------------------------------------------------------------- reporting
def _describe(d, N, P):
    parts = []
    for i, (c, l) in enumerate(zip(d["compounds"], d["stint_laps"])):
        parts.append(f"{c} x{l}")
    return " -> ".join(parts)


def strategy_table(plan):
    pl, N = plan["planner"], plan["laps"]
    best_total = plan["best"]["total"]
    rows = []
    for k in range(pl.P.max_stops + 1):
        d = plan["by_stops"].get(k)
        if d is None:
            rows.append({"Stops": k, "Stop laps": "-", "Stints": "not feasible", "Start fuel/stint (L)": "-",
                         "Race time": "-", "vs best (s)": "-", "LICO": "-", "Extrapolated": "-"})
            continue
        lico = "-"
        if d["sigma"][-1] > 0:
            s = d["sigma"][-1]
            lico = (f"save {s * d['stint_laps'][-1]:.1f} L ({s:.2f} L/lap, {100 * s / pl.b:.1f}%)"
                    f" in final stint")
        rows.append({
            "Stops": k,
            "Stop laps": ", ".join(map(str, d["stop_laps"])) or "-",
            "Stints": _describe(d, N, pl.P),
            "Start fuel/stint (L)": " / ".join(f"{f:.1f}" for f in d["fuel_start_l"]),
            "Race time": fmt_race_time(d["total"]),
            "vs best (s)": f"{d['total'] - best_total:+.1f}",
            "LICO": lico,
            "Extrapolated": "yes" if d["extrapolated"] else "-",
        })
    return pd.DataFrame(rows)


def format_plan(plan):
    pl, N, best = plan["planner"], plan["laps"], plan["best"]
    P = pl.P
    out = [f"\n=== {plan['name']} ({plan['duration_s'] // 60:.0f} min) ===",
           f"Race distance: {N} laps  |  fuel need {N * pl.b:.1f} L at {pl.b:.2f} L/lap "
           f"vs {pl.C:.0f} L tank (lasts {pl.range_laps} laps with {P.fuel_margin_l} L reserve)"]
    out.append(f"OPTIMAL: {best['stops']} stop(s)"
               + (f" on lap {', '.join(map(str, best['stop_laps']))}" if best["stop_laps"] else "")
               + f"  |  {_describe(best, N, P)}  |  {fmt_race_time(best['total'])}")

    # LICO callouts, for any strategy that needs it
    for k, d in sorted(plan["by_stops"].items()):
        if d["sigma"][-1] > 0:
            s, laps_f = d["sigma"][-1], d["stint_laps"][-1]
            out.append(f"  LICO ({k}-stop): without it the driver would have to stop with "
                       f"{d['laps_to_go_without_lico']} lap(s) to go. To reach the flag, save "
                       f"{s * laps_f:.1f} L over the final {laps_f} laps "
                       f"({s:.2f} L/lap, {100 * s / pl.b:.1f}% of normal burn).")
    if 0 not in plan["by_stops"]:
        to_go = N - pl.range_laps
        out.append(f"  No-stop not possible: fuel lasts {pl.range_laps} of {N} laps; a stop would come "
                   f"with {to_go} laps to go (too many for LICO, "
                   f"{N * pl.b + P.fuel_margin_l - pl.C:.1f} L short).")
    out.append("")
    out.append(strategy_table(plan).to_string(index=False))
    if any(d["extrapolated"] for d in plan["by_stops"].values()):
        out.append("  ! 'Extrapolated' = a stint is longer than any stint in your data, so tyre wear "
                   "on the extra laps is a straight-line guess (real tyres often fall off a cliff).")
    return "\n".join(out)