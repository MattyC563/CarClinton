"""Process every MoTeC CSV in a folder, merge the data by tyre compound, fit the model.

    python main.py                       # uses "Data Examples"
    python main.py "Data Examples" --save --no-plot
    python main.py "Data Examples" --seconds-per-kg 0.035

Needs motec_csv_to_df.py, lap_analysis.py and tyre_degradation.py in the same folder.
"""
import argparse
from pathlib import Path

import pandas as pd

from lap_analysis import fmt_time, lap_table
from motec_csv_to_df import load_motec_csv
from strategy import StrategyParams, format_plan, plan_all_races, strategy_table
from tyre_degradation import degradation_table, plot_degradation, stint_analysis, stint_dataframe


def process_folder(folder, seconds_per_kg=0.03):
    """Returns (stints, summary).

    stints:  one DataFrame of model-ready rows per file
    summary: one row per file (compound, laps, best lap, wear rate...)
    """
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No .csv files found in '{folder}'")

    stints, summary = [], []
    for stint_no, path in enumerate(files, start=1):
        try:
            df = load_motec_csv(path)
            laps = lap_table(df, seconds_per_kg=seconds_per_kg)
            data = stint_dataframe(df, laps, stint=stint_no)
        except Exception as e:  # keep going if one file is odd
            print(f"  SKIPPED {path.name}: {e}")
            continue
        if data.empty:
            print(f"  SKIPPED {path.name}: no clean laps")
            continue

        data["File"] = path.name
        stints.append(data)
        summary.append({
            "File": path.name,
            "Stint": stint_no,
            "Compound": data["Compound"].iloc[0],
            "Laps": len(laps),
            "CleanLaps": len(data),
            "BestLap": fmt_time(data["FuelCorrectedLapTimes"].min()),
            "AvgTrackTemp": round(data["TrackTemp"].mean(), 1),
            "WearLinear_s_per_lap": round(laps.attrs.get("wear_s_per_lap", float("nan")), 3),
        })
        print(f"  OK      {path.name}  ({data['Compound'].iloc[0]}, {len(data)} clean laps)")

    if not stints:
        raise RuntimeError("No files could be processed.")
    return stints, pd.DataFrame(summary)


def merge_by_compound(stints):
    """Combine every stint that used the same compound into one dataset per compound."""
    all_data = pd.concat(stints, ignore_index=True)
    return {c: g.reset_index(drop=True) for c, g in all_data.groupby("Compound")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="Data Examples")
    ap.add_argument("--seconds-per-kg", type=float, default=0.03)
    ap.add_argument("--save", action="store_true", help="write merged_<compound>.csv and strategy_summary.csv")
    ap.add_argument("--pit-lane-loss", type=float, default=20.0, help="seconds lost in the pit lane")
    ap.add_argument("--tyre-change", type=float, default=10.0, help="stationary tyre change time (s)")
    ap.add_argument("--refuel-rate", type=float, default=2.0, help="litres per second")
    ap.add_argument("--lico-cost", type=float, default=3.0, help="seconds lost per litre saved by lift and coast")
    ap.add_argument("--fuel-margin", type=float, default=0.5, help="reserve litres at the end of each stint")
    ap.add_argument("--max-stops", type=int, default=3)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print(f"Reading '{args.folder}':")
    stints, summary = process_folder(args.folder, args.seconds_per_kg)
    print("\nPer-file summary:")
    print(summary.to_string(index=False))

    by_compound = merge_by_compound(stints)
    print("\nMerged by compound:")
    for c, g in by_compound.items():
        print(f"  {c}: {g['Stint'].nunique()} stint(s), {len(g)} laps")
        if args.save:
            g.to_csv(f"merged_{c}.csv", index=False)

    all_data = pd.concat(by_compound.values(), ignore_index=True)
    tyres_to_num = {c: i for i, c in enumerate(sorted(by_compound))}
    result = stint_analysis(all_data, tyres_to_num)
    table = degradation_table(result)

    print("\nWear rate (s/lap) from model:")
    for c, rate in result["wear_rate_s_per_lap"].items():
        print(f"  {c}: {rate:+.3f}")
    print()
    print(table.round(3).to_string(index=False))

    # ---- race strategy for the sprint / feature / endurance ----
    params = StrategyParams(
        seconds_per_kg=args.seconds_per_kg, pit_lane_loss_s=args.pit_lane_loss,
        tyre_change_s=args.tyre_change, refuel_rate_l_per_s=args.refuel_rate,
        lico_s_per_l=args.lico_cost, fuel_margin_l=args.fuel_margin, max_stops=args.max_stops,
    )
    plans = plan_all_races(result, all_data, params)
    print("\n" + "#" * 70 + "\nRACE STRATEGY\n" + "#" * 70)
    for plan in plans:
        print(format_plan(plan))
    if args.save:
        pd.concat([strategy_table(p).assign(Race=p["name"], Laps=p["laps"]) for p in plans]) \
            .to_csv("strategy_summary.csv", index=False)

    if not args.no_plot:
        plot_degradation(table, all_data)