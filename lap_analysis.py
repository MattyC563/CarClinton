"""Lap times and fuel-adjusted lap times from a MoTeC CSV DataFrame.

    from motec_csv_to_df import load_motec_csv
    from lap_analysis import lap_table, plot_laps

    df = load_motec_csv("DylanLongRun.csv")
    laps = lap_table(df)
    print(laps)
    plot_laps(laps)

Fuel adjustment: every lap is corrected to what it would have been at a
reference fuel load (default: empty tank), so the only thing left changing the
lap time is (mostly) tyre wear:

    adjusted = raw_time - seconds_per_kg * (avg_fuel_kg - reference_fuel_kg)
"""
import numpy as np
import pandas as pd


def lap_table(
    df,
    fuel_channel="Fuel Level",         # litres in your file
    seconds_per_kg=0.03,               # lap time gained per kg of fuel burnt - TUNE THIS
    fuel_density_kg_per_l=0.745,       # petrol
    reference_fuel_l=0.0,              # correct every lap to this fuel level
    exclude_first_lap=True,            # out lap / standing start
    mistake_threshold_s=1.0,           # slower than the wear trend by more than this = mistake
):
    beacons = df.attrs.get("beacons")
    if not beacons:
        raise ValueError("No beacon markers found in df.attrs['beacons'].")

    edges = [df.index[0]] + list(beacons)
    rows = []
    for i in range(len(edges) - 1):
        seg = df.loc[edges[i]:edges[i + 1]]
        fuel_avg_l = seg[fuel_channel].mean()
        rows.append({
            "lap": i + 1,
            "lap_time": edges[i + 1] - edges[i],
            "fuel_start_l": seg[fuel_channel].iloc[0],
            "fuel_end_l": seg[fuel_channel].iloc[-1],
            "fuel_avg_l": fuel_avg_l,
            "in_pit": bool(seg["In Pit"].max()) if "In Pit" in seg else False,
            "invalid": bool(seg["Lap Invalidated"].max()) if "Lap Invalidated" in seg else False,
        })
    laps = pd.DataFrame(rows).set_index("lap")

    # --- fuel correction -------------------------------------------------
    extra_l = laps["fuel_avg_l"] - reference_fuel_l
    laps["fuel_correction"] = seconds_per_kg * fuel_density_kg_per_l * extra_l
    laps["fuel_adj_time"] = laps["lap_time"] - laps["fuel_correction"]

    # --- which laps count as "clean"? -----------------------------------
    laps["clean"] = ~(laps["in_pit"] | laps["invalid"])
    if exclude_first_lap:
        laps.loc[laps.index[0], "clean"] = False

    # --- flag mistakes: laps well above the (robust) wear trend ----------
    keep = laps["clean"].copy()
    for _ in range(3):  # refit a few times, ignoring flagged laps
        if keep.sum() < 3:
            break
        slope, icpt = np.polyfit(laps.index[keep], laps.loc[keep, "fuel_adj_time"], 1)
        resid = laps["fuel_adj_time"] - (slope * laps.index + icpt)
        keep = laps["clean"] & (resid < mistake_threshold_s)
    laps["mistake"] = laps["clean"] & ~keep
    laps["used"] = keep

    # tyre-wear trend in seconds per lap, from clean fuel-adjusted laps
    if keep.sum() >= 2:
        laps.attrs["wear_s_per_lap"] = np.polyfit(laps.index[keep], laps.loc[keep, "fuel_adj_time"], 1)[0]
    return laps


def fmt_time(t):
    """135.303 -> '2:15.303'"""
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:06.3f}"


def plot_laps(laps):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(laps.index, laps["lap_time"], "o--", color="grey", label="Raw lap time")
    used = laps[laps["used"]]
    ax.plot(used.index, used["fuel_adj_time"], "o-", label="Fuel-adjusted")
    bad = laps[~laps["used"]]
    ax.plot(bad.index, bad["fuel_adj_time"], "x", color="red", ms=10, label="Excluded (out lap / mistake)")
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap time (s)")
    ax.set_xticks(laps.index)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    from motec_csv_to_df import load_motec_csv

    df = load_motec_csv(sys.argv[1])
    laps = lap_table(df)
    out = laps.copy()
    out["lap_time"] = out["lap_time"].map(fmt_time)
    out["fuel_adj_time"] = out["fuel_adj_time"].map(fmt_time)
    print(out[["lap_time", "fuel_avg_l", "fuel_correction", "fuel_adj_time", "clean", "mistake"]].round(2))
    if "wear_s_per_lap" in laps.attrs:
        print(f"\nTyre wear trend: {laps.attrs['wear_s_per_lap']:+.3f} s/lap")
    plot_laps(laps)
