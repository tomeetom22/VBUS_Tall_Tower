#!/usr/bin/env python3
"""Plot summarized VBUS tall-tower data from Converted_TT.

The converted Campbell files are headerless and can be large. This program
streams them one row at a time, retaining only one-minute means. It then
plots trailing 30-minute means rather than raw samples. For the fast sonic
data, it also plots U_30_minute_mean - U_1_minute_mean.

Input data are read only. PNG files are written to plots.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


MISSING_VALUE = -7999.0
TIME_COLUMNS = 4
ROLLING_WINDOW = timedelta(minutes=30)
MAX_CO2_PPM = 5000.0
SITE_LATITUDE_DEG = 40.7507
SITE_LONGITUDE_DEG = -111.9014
MOUNTAIN_TIME = ZoneInfo("America/Denver")

RAW_COLUMN_LABELS = {
    "Fast": [
        "CSAT1_Ux (m/s)", "CSAT1_Uy (m/s)", "CSAT1_Uz (m/s)", "CSAT1_Ts (deg C)",
        "CSAT2_Ux (m/s)", "CSAT2_Uy (m/s)", "CSAT2_Uz (m/s)", "CSAT2_Ts (deg C)",
        "CSAT3_Ux (m/s)", "CSAT3_Uy (m/s)", "CSAT3_Uz (m/s)", "CSAT3_Ts (deg C)",
        "IRGA_Ux (m/s)", "IRGA_Uy (m/s)", "IRGA_Uz (m/s)", "IRGA_Ts (deg C)",
        "IRGA_SonicDiag", "CO2_Density (mg/m^3)", "H2O_Density (g/m^3)",
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
    "Ozone": ["Ozone1_mV", "Ozone2_mV", "Ozone1 (ppbv)", "Ozone2 (ppbv)"],
}


def iter_dat_rows(path: Path):
    """Yield a timestamp and numeric values from a headerless Campbell file."""
    with path.open(newline="") as data_file:
        for line_number, row in enumerate(csv.reader(data_file), start=1):
            if not row or all(not field.strip() for field in row):
                continue
            try:
                year = int(float(row[0]))
                day_of_year = int(float(row[1]))
                hhmm = int(float(row[2]))
                seconds = float(row[3])
                timestamp = datetime(year, 1, 1) + timedelta(
                    days=day_of_year - 1,
                    hours=hhmm // 100,
                    minutes=hhmm % 100,
                    seconds=seconds,
                )
                values = np.asarray(row[TIME_COLUMNS:], dtype=float)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Could not parse {path}:{line_number}") from exc

            values[values == MISSING_VALUE] = np.nan
            yield timestamp, values


def expand_bin_width(
    totals: np.ndarray, counts: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a minute-bin accumulator if a later file has more columns."""
    if width <= len(totals):
        return totals, counts
    return (
        np.pad(totals, (0, width - len(totals)), constant_values=0.0),
        np.pad(counts, (0, width - len(counts)), constant_values=0),
    )


def aggregate_to_minutes(files: list[Path]) -> tuple[list[datetime], np.ndarray]:
    """Stream files and return one-minute means, without retaining raw rows."""
    minute_bins: dict[datetime, tuple[np.ndarray, np.ndarray]] = {}

    for path in files:
        for timestamp, values in iter_dat_rows(path):
            minute = timestamp.replace(second=0, microsecond=0)
            if minute not in minute_bins:
                totals = np.zeros(len(values), dtype=float)
                counts = np.zeros(len(values), dtype=np.int64)
            else:
                totals, counts = minute_bins[minute]
                totals, counts = expand_bin_width(totals, counts, len(values))

            valid = np.isfinite(values)
            totals[: len(values)][valid] += values[valid]
            counts[: len(values)][valid] += 1
            minute_bins[minute] = totals, counts

    if not minute_bins:
        raise ValueError("No data found")

    timestamps = sorted(minute_bins)
    width = max(len(minute_bins[timestamp][0]) for timestamp in timestamps)
    data = np.full((len(timestamps), width), np.nan)
    for row_index, timestamp in enumerate(timestamps):
        totals, counts = minute_bins[timestamp]
        valid = counts > 0
        data[row_index, : len(totals)][valid] = totals[valid] / counts[valid]
    return timestamps, data


def rolling_mean(
    timestamps: list[datetime], data: np.ndarray, window: timedelta
) -> np.ndarray:
    """Calculate trailing time-based means, ignoring missing values."""
    sums = np.zeros(data.shape[1], dtype=float)
    counts = np.zeros(data.shape[1], dtype=np.int64)
    active: deque[tuple[datetime, np.ndarray]] = deque()
    means = np.full(data.shape, np.nan)

    for row_index, (timestamp, values) in enumerate(zip(timestamps, data)):
        active.append((timestamp, values))
        valid = np.isfinite(values)
        sums[valid] += values[valid]
        counts[valid] += 1

        cutoff = timestamp - window
        while active and active[0][0] <= cutoff:
            _, expired = active.popleft()
            valid = np.isfinite(expired)
            sums[valid] -= expired[valid]
            counts[valid] -= 1

        valid = counts > 0
        means[row_index, valid] = sums[valid] / counts[valid]

    return means


def add_fast_derived_products(data: np.ndarray) -> np.ndarray:
    """Add CO2 ppm and relative humidity to fast-table minute means."""
    if data.shape[1] < 22:
        return data

    co2_density = data[:, 17]
    h2o_density = data[:, 18]
    air_temperature_c = data[:, 20]
    air_pressure_kpa = data[:, 21]
    temperature_k = air_temperature_c + 273.15

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        co2_ppm = co2_density * 8.314462618 * temperature_k / (44.01 * air_pressure_kpa)
        vapor_pressure_hpa = h2o_density * 461.5 * temperature_k / 100000.0
        saturation_pressure_hpa = 6.1121 * np.exp(
            (18.678 - air_temperature_c / 234.5)
            * (air_temperature_c / (257.14 + air_temperature_c))
        )
        relative_humidity = 100.0 * vapor_pressure_hpa / saturation_pressure_hpa

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
    co2_ppm[(co2_ppm <= 0.0) | (co2_ppm > MAX_CO2_PPM)] = np.nan
    relative_humidity = np.clip(relative_humidity, 0.0, 100.0)
    return np.column_stack((data[:, :18], co2_ppm, data[:, 18:19], relative_humidity, data[:, 19:]))


def labels_for(sensor_name: str, column_count: int) -> list[str]:
    labels = RAW_COLUMN_LABELS[sensor_name]
    if sensor_name == "Fast" and column_count >= 29:
        labels = labels[:18] + ["CO2 (ppm)", labels[18], "Relative Humidity (%)"] + labels[19:]
    return labels[:column_count] + [
        f"Column {index + 1}" for index in range(len(labels), column_count)
    ]


def sunrise_sunset_utc(local_day: date) -> tuple[datetime, datetime]:
    """Return NOAA-approximate sunrise and sunset as naive UTC datetimes."""
    day_number = local_day.timetuple().tm_yday
    gamma = 2.0 * math.pi / 365.0 * (day_number - 1)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    solar_declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    latitude = math.radians(SITE_LATITUDE_DEG)
    zenith = math.radians(90.833)
    hour_angle = math.acos(
        max(
            -1.0,
            min(
                1.0,
                math.cos(zenith) / (math.cos(latitude) * math.cos(solar_declination))
                - math.tan(latitude) * math.tan(solar_declination),
            ),
        )
    )
    solar_noon_minutes = 720.0 - 4.0 * SITE_LONGITUDE_DEG - equation_of_time
    offset_minutes = 4.0 * math.degrees(hour_angle)
    midnight_utc = datetime.combine(local_day, time.min, tzinfo=timezone.utc)
    sunrise = midnight_utc + timedelta(minutes=solar_noon_minutes - offset_minutes)
    sunset = midnight_utc + timedelta(minutes=solar_noon_minutes + offset_minutes)
    return sunrise.replace(tzinfo=None), sunset.replace(tzinfo=None)


def add_night_shading(axis: plt.Axes, timestamps: list[datetime]) -> None:
    """Shade nighttime using site sunrise/sunset, while retaining UTC x-axes."""
    if not timestamps:
        return
    start_local = timestamps[0].replace(tzinfo=timezone.utc).astimezone(MOUNTAIN_TIME).date()
    end_local = timestamps[-1].replace(tzinfo=timezone.utc).astimezone(MOUNTAIN_TIME).date()
    local_day = start_local - timedelta(days=2)
    final_day = end_local + timedelta(days=1)
    label_added = False
    while local_day <= final_day:
        _, sunset = sunrise_sunset_utc(local_day)
        next_sunrise, _ = sunrise_sunset_utc(local_day + timedelta(days=1))
        axis.axvspan(
            sunset,
            next_sunrise,
            color="#405a73",
            alpha=0.12,
            linewidth=0,
            label="Night (America/Denver)" if not label_added else None,
            zorder=0,
        )
        label_added = True
        local_day += timedelta(days=1)


def format_axes(axis: plt.Axes, timestamps: list[datetime]) -> None:
    add_night_shading(axis, timestamps)
    axis.grid(True, alpha=0.3)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axis.set_xlabel("UTC time (night shading uses America/Denver)")
    if len(timestamps) > 1:
        axis.set_xlim(timestamps[0], timestamps[-1])


def plot_cnr4_rolling_means(
    timestamps: list[datetime], rolling_data: np.ndarray, output_directory: Path
) -> None:
    """Save only the requested 30-minute CNR4 radiation and albedo means."""
    columns = [5, 11, 12]  # SW down, net radiation, albedo.
    columns = [column for column in columns if column < rolling_data.shape[1]]
    figure, axes = plt.subplots(
        len(columns),
        1,
        sharex=True,
        figsize=(14, max(3, 2.5 * len(columns))),
        squeeze=False,
    )
    labels = labels_for("CNR4", rolling_data.shape[1])
    for axis, column_index in zip(axes[:, 0], columns):
        axis.plot(timestamps, rolling_data[:, column_index], linewidth=0.8)
        axis.set_ylabel(labels[column_index])
        format_axes(axis, timestamps)

    figure.suptitle("CNR4: trailing 30-minute rolling means (SW down, net radiation, albedo)")
    figure.tight_layout()
    figure.savefig(
        output_directory / "cnr4_30min_rolling_mean.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_fast_sonic_rolling_means(
    timestamps: list[datetime], rolling_data: np.ndarray, output_directory: Path
) -> None:
    """Overlay matching sonic variables; diagnostic fields are intentionally omitted."""
    groups = {
        "Ux (m/s)": [0, 4, 8, 12],
        "Uy (m/s)": [1, 5, 9, 13],
        "Uz (m/s)": [2, 6, 10, 14],
        "Ts (deg C)": [3, 7, 11, 15],
    }
    sensor_names = ["CSAT1", "CSAT2", "CSAT3", "IRGA"]
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 12), squeeze=False)
    for axis, (label, indices) in zip(axes[:, 0], groups.items()):
        for sensor_name, column_index in zip(sensor_names, indices):
            if column_index < rolling_data.shape[1]:
                axis.plot(timestamps, rolling_data[:, column_index], linewidth=0.8, label=sensor_name)
        axis.set_ylabel(label)
        format_axes(axis, timestamps)
        axis.legend(loc="upper right", ncols=2, fontsize="small")

    figure.suptitle("Sonic and IRGASON: trailing 30-minute rolling means")
    figure.tight_layout()
    figure.savefig(
        output_directory / "fast_sonics_30min_rolling_mean.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_fast_co2_rh_rolling_means(
    timestamps: list[datetime], rolling_data: np.ndarray, output_directory: Path
) -> None:
    """Plot only the requested CO2 ppm and relative-humidity rolling means."""
    columns = [18, 20]
    labels = labels_for("Fast", rolling_data.shape[1])
    columns = [column for column in columns if column < rolling_data.shape[1]]
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 7), squeeze=False)
    for axis, column_index in zip(axes[:, 0], columns):
        axis.plot(timestamps, rolling_data[:, column_index], linewidth=0.8)
        axis.set_ylabel(labels[column_index])
        format_axes(axis, timestamps)

    figure.suptitle("CO2 and relative humidity: trailing 30-minute rolling means")
    figure.tight_layout()
    figure.savefig(
        output_directory / "fast_co2_rh_30min_rolling_mean.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_ozone_rolling_means(
    timestamps: list[datetime], rolling_data: np.ndarray, output_directory: Path
) -> None:
    """Overlay ozone ppbv values and intentionally omit the raw millivolt channels."""
    figure, axis = plt.subplots(figsize=(14, 5))
    for column_index, label in ((2, "Ozone1 (ppbv)"), (3, "Ozone2 (ppbv)")):
        if column_index < rolling_data.shape[1]:
            axis.plot(timestamps, rolling_data[:, column_index], linewidth=0.9, label=label)
    format_axes(axis, timestamps)
    axis.set_ylabel("Ozone (ppbv)")
    axis.legend(loc="upper right")
    figure.suptitle("Ozone: trailing 30-minute rolling means")
    figure.tight_layout()
    figure.savefig(
        output_directory / "ozone_30min_rolling_mean.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_velocity_deviations(
    timestamps: list[datetime],
    minute_data: np.ndarray,
    rolling_data: np.ndarray,
    output_directory: Path,
) -> None:
    """Plot U_30min - U_1min at one-minute intervals for all sonic channels."""
    component_indices = {
        "Ux": [0, 4, 8, 12],
        "Uy": [1, 5, 9, 13],
        "Uz": [2, 6, 10, 14],
    }
    sensor_names = ["CSAT1", "CSAT2", "CSAT3", "IRGA"]
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 10), squeeze=False)

    for axis, (component, indices) in zip(axes[:, 0], component_indices.items()):
        for sensor_name, column_index in zip(sensor_names, indices):
            if column_index >= minute_data.shape[1]:
                continue
            deviation = rolling_data[:, column_index] - minute_data[:, column_index]
            axis.plot(timestamps, deviation, linewidth=0.8, label=sensor_name)
        axis.set_ylabel(f"Ubar_30min - U_1min\n{component} (m/s)")
        axis.legend(loc="upper right", ncols=2, fontsize="small")
        format_axes(axis, timestamps)

    figure.suptitle("Turbulent velocity component at one-minute intervals")
    figure.tight_layout()
    figure.savefig(
        output_directory / "fast_velocity_deviation_1min.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_irga_wind_rose(minute_data: np.ndarray, output_directory: Path) -> None:
    """Plot a meteorological wind-from rose from the top IRGASON.

    The sonic is oriented with +Ux toward true west (270 degrees). With the
    right-handed sonic coordinate system and +Uz upward, +Uy points south.
    The resulting direction is converted from velocity-toward to the usual
    meteorological wind-from convention.
    """
    if minute_data.shape[1] < 14:
        return

    u_component = minute_data[:, 12]
    v_component = minute_data[:, 13]
    speed_all = np.hypot(u_component, v_component)
    finite_speed = np.isfinite(speed_all)
    valid = finite_speed & (speed_all >= 0.1)
    if not np.any(valid):
        return

    # +Ux is west and +Uy is south: 90 - atan2(v, u) is the wind-from azimuth.
    direction_from = (90.0 - np.degrees(np.arctan2(v_component[valid], u_component[valid]))) % 360.0
    speed = speed_all[valid]
    sector_count = 16
    sector_width = 360.0 / sector_count
    sector_index = (np.floor((direction_from + sector_width / 2.0) / sector_width).astype(int)
                    % sector_count)
    speed_edges = np.array([0.1, 1.0, 3.0, 5.0, 8.0, np.inf])
    speed_labels = ["0.1–1", "1–3", "3–5", "5–8", "≥8 m/s"]
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, len(speed_labels)))
    total_records = np.count_nonzero(finite_speed)

    figure, axis = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    angles = np.deg2rad(np.arange(sector_count) * sector_width)
    bottoms = np.zeros(sector_count)
    for lower, upper, label, color in zip(
        speed_edges[:-1], speed_edges[1:], speed_labels, colors
    ):
        values = np.zeros(sector_count)
        in_speed_bin = (speed >= lower) & (speed < upper)
        for sector in range(sector_count):
            values[sector] = np.count_nonzero(in_speed_bin & (sector_index == sector))
        values = 100.0 * values / total_records
        axis.bar(
            angles,
            values,
            width=np.deg2rad(sector_width * 0.9),
            bottom=bottoms,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            align="center",
            label=label,
        )
        bottoms += values

    calm_percent = 100.0 * np.count_nonzero(finite_speed & (speed_all < 0.1)) / total_records
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_thetagrids(np.arange(0, 360, 45), labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    axis.set_ylabel("Occurrence (%)", labelpad=28)
    axis.set_title(
        "Top IRGASON wind rose: one-minute mean wind-from direction\n"
        f"Calm (<0.1 m/s): {calm_percent:.1f}%",
        va="bottom",
    )
    axis.legend(title="Wind speed", loc="lower left", bbox_to_anchor=(1.05, 0.0))
    figure.tight_layout()
    figure.savefig(output_directory / "irga_wind_rose.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_directory",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "Converted_TT",
        help="directory containing converted .dat files (default: Converted_TT)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).parent / "plots",
        help="directory for PNG output (default: plots)",
    )
    args = parser.parse_args()

    families = {
        "Fast": "CSV_Fast_20Hz*.dat",
        "CNR4": "CSV_CNR4_0p1Hz*.dat",
        "Ozone": "CSV_Ozone_0p5Hz*.dat",
    }
    files_by_family = {
        name: sorted(args.data_directory.glob(pattern))
        for name, pattern in families.items()
    }
    if not any(files_by_family.values()):
        parser.error(f"No expected .dat files found in {args.data_directory}")

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for sensor_name, files in files_by_family.items():
        if not files:
            continue
        print(f"Processing {sensor_name} ({len(files)} files)...", flush=True)
        timestamps, minute_data = aggregate_to_minutes(files)
        if sensor_name == "Fast":
            minute_data = add_fast_derived_products(minute_data)
        if sensor_name == "CNR4" and minute_data.shape[1] > 12:
            valid_albedo = (minute_data[:, 12] > 0.0) & (minute_data[:, 12] <= 1.0)
            minute_data[~valid_albedo, 12] = np.nan
        rolling_data = rolling_mean(timestamps, minute_data, ROLLING_WINDOW)
        if sensor_name == "Fast":
            plot_fast_sonic_rolling_means(timestamps, rolling_data, args.output_directory)
            plot_fast_co2_rh_rolling_means(timestamps, rolling_data, args.output_directory)
            plot_velocity_deviations(timestamps, minute_data, rolling_data, args.output_directory)
            plot_irga_wind_rose(minute_data, args.output_directory)
        elif sensor_name == "CNR4":
            if minute_data.shape[1] > 12:
                rolling_data[~valid_albedo, 12] = np.nan
            plot_cnr4_rolling_means(timestamps, rolling_data, args.output_directory)
        elif sensor_name == "Ozone":
            plot_ozone_rolling_means(timestamps, rolling_data, args.output_directory)
        print(f"Wrote {sensor_name} plots from {len(timestamps)} one-minute records.", flush=True)


if __name__ == "__main__":
    main()
