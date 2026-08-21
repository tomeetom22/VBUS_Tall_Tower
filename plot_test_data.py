#!/usr/bin/env python3
"""Plot the Campbell Scientific test-data files in ``Test_Data``.

The four-part timestamp at the start of each row is interpreted as
year, day-of-year, HHMM, and seconds into the minute.  Campbell's -7999
missing-value flag is replaced with NaN before plotting.

All fast 20 Hz files are combined into one figure.  The slower sensor
families are plotted in separate figures (Ozone, GPS, and CNR4), with the
available files for each family overlaid in that figure.  Figures are saved
as PNG files; no plot windows are opened.

Usage:
    python plot_test_data.py
    python plot_test_data.py path/to/Test_Data
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


MISSING_VALUE = -7999.0
TIME_COLUMNS = 4

# The Campbell files do not contain headers.  These names mirror the
# Sample() order in Combined_CR3000_AllSensors.cr3.  The Ozone test files in
# Test_Data contain four values, so their exact source table is not fully
# represented by the current CRBasic source; those columns remain explicit
# channel names rather than being given misleading sensor names.
COLUMN_LABELS = {
    "Fast": [
        "CSAT1_Ux (m/s)", "CSAT1_Uy (m/s)", "CSAT1_Uz (m/s)", "CSAT1_Ts (deg C)",
        "CSAT2_Ux (m/s)", "CSAT2_Uy (m/s)", "CSAT2_Uz (m/s)", "CSAT2_Ts (deg C)",
        "CSAT3_Ux (m/s)", "CSAT3_Uy (m/s)", "CSAT3_Uz (m/s)", "CSAT3_Ts (deg C)",
        "IRGA_Ux (m/s)", "IRGA_Uy (m/s)", "IRGA_Uz (m/s)", "IRGA_Ts (deg C)",
        "IRGA_SonicDiag", "CO2_Density (mg/m^3)", "CO2 (ppm)",
        "H2O_Density (g/m^3)",
        "Relative Humidity (%)",
        "IRGA_GasDiag", "IRGA_AirTemp (deg C)", "IRGA_AirPressure (kPa)",
        "CO2_Signal", "H2O_Signal", "CO2_Density_FastTemp (mg/m^3)",
        "BattVolt (V)", "LoggerTemp (deg C)",
    ],
    "CNR4": [
        "SW_Up_mV", "SW_Down_mV", "LW_Up_mV", "LW_Down_mV",
        "SW_Up (W/m^2)", "SW_Down (W/m^2)", "LW_Up (W/m^2)",
        "LW_Down (W/m^2)", "CNR4_T_C (deg C)", "SW_Net (W/m^2)",
        "LW_Net (W/m^2)", "NetRadiation (W/m^2)", "Albedo",
    ],
    "GPS": [
        "GPSData_1", "GPSData_2", "GPSData_3", "GPSData_4",
        "GPS_SpeedKnots (knots)", "GPS_CourseDeg (degrees)", "GPSData_7",
        "GPS_FixQuality", "GPS_Satellites (count)", "GPS_Altitude_m (m)",
        "GPS_PPS_us (us)", "GPSData_12", "GPS_Ready", "GPSData_14", "GPSData_15",
    ],
    "Ozone": [
        "Ozone_channel_1", "Ozone_channel_2", "Ozone_channel_3", "Ozone_channel_4",
    ],
}


def read_dat_file(path: Path) -> tuple[np.ndarray, list[datetime]]:
    """Read one comma-separated Campbell data file."""
    rows: list[list[float]] = []
    timestamps: list[datetime] = []

    with path.open(newline="") as data_file:
        for line_number, row in enumerate(csv.reader(data_file), start=1):
            if not row or all(not field.strip() for field in row):
                continue
            try:
                year = int(float(row[0]))
                day_of_year = int(float(row[1]))
                hhmm = int(float(row[2]))
                seconds = float(row[3])
                values = [float(value) for value in row[TIME_COLUMNS:]]
                timestamp = datetime(year, 1, 1) + timedelta(
                    days=day_of_year - 1,
                    hours=hhmm // 100,
                    minutes=hhmm % 100,
                    seconds=seconds,
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Could not parse {path}:{line_number}") from exc

            rows.append(values)
            timestamps.append(timestamp)

    if not rows:
        raise ValueError(f"No data found in {path}")

    width = max(len(row) for row in rows)
    data = np.full((len(rows), width), np.nan)
    for row_index, row in enumerate(rows):
        data[row_index, : len(row)] = row
    data[data == MISSING_VALUE] = np.nan
    return data, timestamps


def format_axes(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.3)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M:%S"))
    axis.set_xlabel("UTC time")


def labels_for(sensor_name: str, column_count: int) -> list[str]:
    labels = COLUMN_LABELS[sensor_name]
    return labels[:column_count] + [
        f"Column {index + 1}" for index in range(len(labels), column_count)
    ]


def add_derived_products(data: np.ndarray) -> np.ndarray:
    """Insert CO2 ppm and RH, calculated from IRGASON measurements.

    H2O density is in g/m^3 and air temperature is in degrees C.  Vapor
    pressure is calculated with the ideal gas law, and saturation vapor
    pressure uses the Buck equation over liquid water.  CO2 density is in
    mg/m^3, air pressure is in kPa, and CO2 ppm is calculated as a molar
    mixing ratio using the ideal gas law.
    """
    co2_density = data[:, 17]
    h2o_density = data[:, 18]
    air_temperature_c = data[:, 20]
    air_pressure_kpa = data[:, 21]
    temperature_k = air_temperature_c + 273.15

    co2_ppm = (
        co2_density * 8.314462618 * temperature_k
        / (44.01 * air_pressure_kpa)
    )

    vapor_pressure_hpa = h2o_density * 461.5 * temperature_k / 100000.0
    saturation_pressure_hpa = 6.1121 * np.exp(
        (18.678 - air_temperature_c / 234.5)
        * (air_temperature_c / (257.14 + air_temperature_c))
    )
    relative_humidity = 100.0 * vapor_pressure_hpa / saturation_pressure_hpa

    # Keep the derived series missing wherever either source measurement is
    # missing or physically invalid.
    invalid = (
        ~np.isfinite(co2_density)
        | ~np.isfinite(h2o_density)
        | ~np.isfinite(air_temperature_c)
        | ~np.isfinite(air_pressure_kpa)
        | (temperature_k <= 0)
        | (air_pressure_kpa <= 0)
    )
    co2_ppm[invalid] = np.nan
    relative_humidity[invalid] = np.nan

    return np.column_stack(
        (data[:, :18], co2_ppm, data[:, 18:19], relative_humidity, data[:, 19:])
    )


def plot_combined_fast(files: list[Path], output_directory: Path) -> None:
    """Plot all 20 Hz files together in one figure."""
    datasets = [
        (add_derived_products(data), timestamps)
        for path in files
        for data, timestamps in [read_dat_file(path)]
    ]
    column_count = max(data.shape[1] for data, _ in datasets)
    figure, axes = plt.subplots(
        column_count,
        1,
        sharex=False,
        figsize=(14, max(3, 2.2 * column_count)),
        squeeze=False,
    )
    axes = axes[:, 0]
    labels = labels_for("Fast", column_count)

    for column_index, axis in enumerate(axes):
        for path, (data, timestamps) in zip(files, datasets):
            if column_index < data.shape[1]:
                axis.plot(
                    timestamps,
                    data[:, column_index],
                    linewidth=0.7,
                    label=path.stem,
                )
        axis.set_ylabel(labels[column_index])
        format_axes(axis)
        if column_index == 0:
            axis.legend(loc="upper right", fontsize="small")

    figure.suptitle("Combined 20 Hz test data")
    figure.tight_layout()
    figure.savefig(output_directory / "combined_fast_20hz.png", dpi=150, bbox_inches="tight")


def plot_slow_family(sensor_name: str, files: list[Path], output_directory: Path) -> None:
    """Plot all files for one slow sensor family in one figure."""
    datasets = [read_dat_file(path) for path in files]
    column_count = max(data.shape[1] for data, _ in datasets)
    figure, axes = plt.subplots(
        column_count,
        1,
        sharex=False,
        figsize=(14, max(3, 2.5 * column_count)),
        squeeze=False,
    )
    axes = axes[:, 0]
    labels = labels_for(sensor_name, column_count)

    for column_index, axis in enumerate(axes):
        for path, (data, timestamps) in zip(files, datasets):
            if column_index < data.shape[1]:
                axis.plot(
                    timestamps,
                    data[:, column_index],
                    linewidth=1.0,
                    label=path.stem,
                )
        axis.set_ylabel(labels[column_index])
        format_axes(axis)
        if column_index == 0:
            axis.legend(loc="upper right", fontsize="small")

    figure.suptitle(f"{sensor_name} test data")
    figure.tight_layout()
    figure.savefig(output_directory / f"{sensor_name.lower()}_test_data.png", dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_directory",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "Test_Data",
        help="directory containing the .dat files (default: Test_Data)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="directory for PNG output (default: plots)",
    )
    args = parser.parse_args()

    data_directory = args.data_directory
    fast_files = sorted(data_directory.glob("CSV_Fast_20Hz*.dat"))
    slow_patterns = {
        "Ozone": "CSV_Ozone_0p5Hz*.dat",
        "GPS": "CSV_GPS_0p1Hz*.dat",
        "CNR4": "CSV_CNR4_0p1Hz*.dat",
    }

    if not fast_files and not any(data_directory.glob(pattern) for pattern in slow_patterns.values()):
        parser.error(f"No expected .dat files found in {data_directory}")

    args.output_directory.mkdir(parents=True, exist_ok=True)

    if fast_files:
        plot_combined_fast(fast_files, args.output_directory)

    for sensor_name, pattern in slow_patterns.items():
        files = sorted(data_directory.glob(pattern))
        if files:
            plot_slow_family(sensor_name, files, args.output_directory)

    plt.close("all")


if __name__ == "__main__":
    main()
