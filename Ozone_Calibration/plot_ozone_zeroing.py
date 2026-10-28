#!/usr/bin/env python3
"""Calculate the October 16, 2026 VBUS ozone zeroing-calibration averages.

The Campbell logger timestamps are UTC. Calibration windows below are entered
in Utah local time (America/Denver), which was MDT on October 16, 2026.
Only records from the calibration date and explicitly selected raw files are
retained. The tall-tower file ends before the bottom calibration window does,
so that mean intentionally uses only the available records.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


CALIBRATION_DATE = date(2026, 10, 16)
MOUNTAIN_TIME = ZoneInfo("America/Denver")
MAX_VALID_MV = 2500.0
DEFAULT_TALL_TOWER_DIRECTORY = Path(
    "/Users/tomw/Documents/VBUS/LEMSv3/Data/FleetBlock_Data/TallTower"
)
DEFAULT_POP_UP_DIRECTORY = Path(
    "/Users/tomw/Documents/VBUS/LEMSv3/Data/FleetBlock_Data/PopUp/Campbell"
)
DEFAULT_TALL_TOWER_FILE = DEFAULT_TALL_TOWER_DIRECTORY / "CSV_Ozone_0p5Hz40.dat"
DEFAULT_POP_UP_FILE = DEFAULT_POP_UP_DIRECTORY / "CSV_PopUp_Ozone_0p5Hz27.dat"


@dataclass(frozen=True)
class CalibrationWindow:
    name: str
    data_file: Path
    value_column: int
    start: time
    end: time
    color: str


def parse_timestamp(row: list[str]) -> datetime:
    """Parse a converted Campbell timestamp as a naive UTC datetime."""
    year = int(float(row[0]))
    day_of_year = int(float(row[1]))
    hhmm = int(float(row[2]))
    seconds = float(row[3])
    return datetime.strptime(f"{year} {day_of_year:03d}", "%Y %j").replace(
        hour=hhmm // 100, minute=hhmm % 100, second=int(seconds),
        microsecond=round((seconds % 1) * 1_000_000),
    )


def read_calibration_records(
    path: Path, value_column: int
) -> list[tuple[datetime, float, Path]]:
    """Read valid mV records for CALIBRATION_DATE from one raw data file."""
    records: list[tuple[datetime, float, Path]] = []
    if not path.is_file():
        raise FileNotFoundError(f"Calibration input file not found: {path}")
    with path.open(newline="") as data_file:
        for row in csv.reader(data_file):
            if len(row) <= value_column:
                continue
            try:
                timestamp_utc = parse_timestamp(row)
                value_mv = float(row[value_column])
            except (ValueError, IndexError):
                continue
            if timestamp_utc.date() != CALIBRATION_DATE:
                continue
            if not np.isfinite(value_mv) or abs(value_mv) > MAX_VALID_MV:
                continue
            local_timestamp = timestamp_utc.replace(tzinfo=timezone.utc).astimezone(
                MOUNTAIN_TIME
            ).replace(tzinfo=None)
            records.append((local_timestamp, value_mv, path))
    return records


def window_bounds(window: CalibrationWindow) -> tuple[datetime, datetime]:
    return (
        datetime.combine(CALIBRATION_DATE, window.start),
        datetime.combine(CALIBRATION_DATE, window.end),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tall-tower-file",
        type=Path,
        default=DEFAULT_TALL_TOWER_FILE,
        help="Tall-tower calibration file (default: CSV_Ozone_0p5Hz40.dat).",
    )
    parser.add_argument(
        "--pop-up-file",
        type=Path,
        default=DEFAULT_POP_UP_FILE,
        help="Pop-up calibration file (default: CSV_PopUp_Ozone_0p5Hz27.dat).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for the calibration PNG.",
    )
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Also print every selected timestamp, mV value, and source file.",
    )
    args = parser.parse_args()

    windows = [
        CalibrationWindow(
            "Tall tower top (Ozone 2)",
            args.tall_tower_file,
            5,
            time(11, 45),
            time(12, 5),
            "tab:blue",
        ),
        CalibrationWindow(
            "Tall tower bottom (Ozone 1)",
            args.tall_tower_file,
            4,
            time(12, 30),
            time(13, 30),
            "tab:orange",
        ),
        CalibrationWindow(
            "Pop-up tower (Ozone)",
            args.pop_up_file,
            4,
            time(13, 50),
            time(14, 7),
            "tab:green",
        ),
    ]

    series: list[tuple[CalibrationWindow, list[tuple[datetime, float, Path]]]] = []
    for window in windows:
        records = read_calibration_records(window.data_file, window.value_column)
        start, end = window_bounds(window)
        selected = [record for record in records if start <= record[0] <= end]
        if not selected:
            raise ValueError(
                f"No valid {window.name} records between {start:%H:%M} and {end:%H:%M} "
                f"America/Denver on {CALIBRATION_DATE}."
            )
        selected_values = [value for _, value, _ in selected]
        selected_files = sorted({path for _, _, path in selected})
        print(
            f"{window.name}: {np.mean(selected_values):.2f} mV "
            f"(n={len(selected_values):,}; {start:%H:%M}–{end:%H:%M} America/Denver)"
        )
        print("  Files used: " + ", ".join(path.name for path in selected_files))
        if args.print_values:
            for timestamp, value, path in selected:
                print(f"    {timestamp:%Y-%m-%d %H:%M:%S}  {value:10.4f} mV  {path.name}")
        series.append((window, records))

    args.output_directory.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 6))
    for window, records in series:
        timestamps = [timestamp for timestamp, _, _ in records]
        values = [value for _, value, _ in records]
        start, end = window_bounds(window)
        selected_values = [
            value for timestamp, value, _ in records if start <= timestamp <= end
        ]
        mean_mv = np.mean(selected_values)
        axis.plot(timestamps, values, linewidth=0.65, color=window.color, label=window.name)
        axis.axvspan(start, end, color=window.color, alpha=0.14)
        axis.hlines(mean_mv, start, end, colors=window.color, linestyles="--", linewidth=1.5)
        axis.annotate(
            f"{mean_mv:.2f} mV\n(n={len(selected_values):,})",
            (start + (end - start) / 2, mean_mv),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=window.color,
            fontsize="small",
        )

    axis.set_title("Ozone Sensor Voltage — October 16, 2026 Calibration")
    axis.set_xlabel("Utah local time (America/Denver)")
    axis.set_ylabel("Voltage (mV)")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path = args.output_directory / "ozone_zeroing_2026-10-16.png"
    figure.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
