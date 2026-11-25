"""Load a MoTeC i2 CSV export (from an .ld log) into a pandas DataFrame.

Typical i2 CSV layout:
    "Format","MoTeC CSV File",...      <- metadata preamble (venue, driver, sample rate...)
    ...
    "Time","Speed","RPM",...           <- channel names
    "s","km/h","rpm",...               <- units
    (maybe a blank row)
    0.000,123.4,8000,...               <- data

Usage:
    python motec_csv_to_df.py session.csv
or:
    from motec_csv_to_df import load_motec_csv
    df = load_motec_csv("session.csv")
    df.attrs["units"]     # {"Speed": "km/h", ...}
    df.attrs["metadata"]  # {"Venue": "...", "Sample Rate": "20", ...}
"""
import csv
import sys
from pathlib import Path

import pandas as pd


def _dedupe(names):
    """Make column names unique: Speed, Speed -> Speed, Speed.1"""
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def load_motec_csv(path, time_index=True, encoding="utf-8-sig"):
    path = Path(path)
    metadata, names, units, header_idx = {}, None, None, None

    # 1) Scan only the preamble (cheap even for huge files)
    with open(path, newline="", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            first = row[0].strip() if row else ""
            if first.lower() == "time" and len(row) > 2:  # channel-name row
                names = [c.strip() for c in row]
                units = [c.strip() for c in next(reader, [])]
                header_idx = i
                break
            if first and len(row) >= 2 and row[1].strip():
                metadata[first] = row[1].strip()
            # some exports put a second key/value pair on the right (cols 4-5)
            if len(row) >= 6 and row[4].strip() and row[5].strip():
                metadata[row[4].strip()] = row[5].strip()

    if names is None:
        raise ValueError("Couldn't find a 'Time' header row - is this a MoTeC CSV export?")

    units += [""] * (len(names) - len(units))  # pad if units row is short

    # 2) Read the data block
    names_u = _dedupe(names)
    df = pd.read_csv(
        path,
        skiprows=header_idx + 2,  # preamble + names + units
        header=None,
        names=names_u,
        usecols=range(len(names_u)),
        encoding=encoding,
        encoding_errors="replace",
        low_memory=False,
    )

    # 3) Clean up: drop unnamed columns, convert to numbers where possible
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # keep original if it was genuinely text (e.g. a string channel)
        if converted.notna().any() or df[col].isna().all():
            df[col] = converted

    time_col = names_u[0]
    df = df.dropna(subset=[time_col]).reset_index(drop=True)  # removes blank/extra rows

    df.attrs["units"] = dict(zip(names_u, units))
    df.attrs["metadata"] = metadata
    # beacon markers (lap split times in seconds) as a list of floats
    df.attrs["beacons"] = [float(x) for x in metadata.get("Beacon Markers", "").split()]

    if time_index:
        df = df.set_index(time_col)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python motec_csv_to_df.py <file.csv>")
    df = load_motec_csv(sys.argv[1])
    print(df.attrs["metadata"])
    print(df.head())
    print(f"\n{len(df)} rows x {len(df.columns)} channels")