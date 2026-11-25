"""Random-forest tyre degradation model.

Same idea as your stint_analysis(): learn fuel-corrected lap time from
TyreLife, Compound and TrackTemp, then read the degradation curve off the model.

    from motec_csv_to_df import load_motec_csv
    from lap_analysis import lap_table
    from tyre_degradation import stint_dataframe, stint_analysis, degradation_table, plot_degradation

    df = load_motec_csv("DylanLongRun.csv")
    laps = lap_table(df)
    data = stint_dataframe(df, laps, stint=1)             # add more stints/files with pd.concat
    result = stint_analysis(data, {"SM": 0})
    table = degradation_table(result)
    print(table)
    print(result["wear_rate_s_per_lap"])
    plot_degradation(table)
"""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# 1) Turn a MoTeC session into rows the model can use
# ---------------------------------------------------------------------------
def stint_dataframe(df, laps, stint=1, compound=None, start_tyre_life=0,
                    track_temp_channel="Road Temp"):
    """One row per clean lap: Stint, TyreLife, Compound, TrackTemp, FuelCorrectedLapTimes.

    compound: e.g. "SM". If None it's read from the log's Comment ("Tires: Semislicks (SM)").
    start_tyre_life: laps already on the tyres before this log started (0 = new set).
    """
    if compound is None:
        m = re.search(r"\(([^)]+)\)", df.attrs["metadata"].get("Comment", ""))
        compound = m.group(1) if m else "UNKNOWN"

    beacons = [df.index[0]] + list(df.attrs["beacons"])
    temps = [df.loc[beacons[i]:beacons[i + 1], track_temp_channel].mean()
             for i in range(len(beacons) - 1)]

    out = pd.DataFrame({
        "Stint": stint,
        "TyreLife": laps.index + start_tyre_life,   # keeps the out lap in the count
        "Compound": compound,
        "TrackTemp": temps,
        "FuelCorrectedLapTimes": laps["fuel_adj_time"].values,
        "FuelUsed_l": (laps["fuel_start_l"] - laps["fuel_end_l"]).values,   # litres burnt per lap
        "MaxFuel_l": df["Max Fuel"].max() if "Max Fuel" in df else float("nan"),
        "Used": laps["used"].values,                # clean, non-mistake laps
    })
    return out[out["Used"]].drop(columns="Used").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) The model (same shape as your stint_analysis)
# ---------------------------------------------------------------------------
def stint_analysis(data, tyres_to_num, n_estimators=100, min_samples_leaf=1):
    X = data[["TyreLife", "Compound", "TrackTemp"]].copy()
    X["Compound"] = X["Compound"].map(tyres_to_num)
    if X["Compound"].isna().any():
        raise ValueError(f"Compound missing from tyres_to_num: "
                         f"{sorted(set(data['Compound']) - set(tyres_to_num))}")
    y = data["FuelCorrectedLapTimes"].copy()

    model = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=42
    ).fit(X, y)

    return {
        "model": model,
        "baseline_track_temp": data["TrackTemp"].mean(),
        "tyres_to_num": tyres_to_num,
        # observed tyre-life range per compound: don't read the curve outside this
        "life_range": data.groupby("Compound")["TyreLife"].agg(["min", "max"]),
    }


# ---------------------------------------------------------------------------
# 3) Degradation curves
# ---------------------------------------------------------------------------
def degradation_table(result, track_temp=None):
    """Predicted lap time and degradation per lap of tyre life, for every compound.

    Track temp is held at the baseline so the curve shows tyre life only.
    Degradation is measured relative to the freshest observed lap.
    """
    temp = result["baseline_track_temp"] if track_temp is None else track_temp
    frames = []
    for compound, num in result["tyres_to_num"].items():
        if compound not in result["life_range"].index:
            continue
        lo, hi = result["life_range"].loc[compound]
        life = np.arange(lo, hi + 1)
        X = pd.DataFrame({"TyreLife": life, "Compound": num, "TrackTemp": temp})
        pred = result["model"].predict(X)
        frames.append(pd.DataFrame({
            "Compound": compound,
            "TyreLife": life,
            "PredLapTime": pred,
            "DegVsFresh": pred - pred[0],
            "DegPerLap": np.r_[np.nan, np.diff(pred)],
        }))
    table = pd.concat(frames, ignore_index=True)

    # The forest's curve is a staircase, so also give one smooth number per compound:
    # the straight-line slope (s/lap) of the predicted curve.
    result["wear_rate_s_per_lap"] = {
        c: np.polyfit(g["TyreLife"], g["PredLapTime"], 1)[0] if len(g) > 1 else np.nan
        for c, g in table.groupby("Compound")
    }
    return table


def plot_degradation(table, data=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for compound, g in table.groupby("Compound"):
        ax.plot(g["TyreLife"], g["DegVsFresh"], "o-", label=f"{compound} (model)")
        if data is not None:
            d = data[data["Compound"] == compound]
            first = g["PredLapTime"].iloc[0]
            ax.plot(d["TyreLife"], d["FuelCorrectedLapTimes"] - first, "x", alpha=0.5,
                    label=f"{compound} (actual laps)")
    ax.set_xlabel("Tyre life (laps)")
    ax.set_ylabel("Time lost vs freshest lap (s)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    from motec_csv_to_df import load_motec_csv
    from lap_analysis import lap_table

    df = load_motec_csv(sys.argv[1])
    laps = lap_table(df)
    data = stint_dataframe(df, laps)
    result = stint_analysis(data, {c: i for i, c in enumerate(sorted(data["Compound"].unique()))})
    table = degradation_table(result)
    print(table.round(3))
    print("\nWear rate (s/lap):", {k: round(v, 3) for k, v in result["wear_rate_s_per_lap"].items()})
    plot_degradation(table, data)