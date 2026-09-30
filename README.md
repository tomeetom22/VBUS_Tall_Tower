# VBUS Tall Tower CR3000 Codes

CRBasic programs for the VBUS tall-tower sensor system.

## Combined program

`Combined_CR3000_AllSensors.cr3` combines:

- Three CSAT3B sonic anemometers on the shared SDM bus
- One IRGASON
- One CNR4 net radiometer
- Two analog ozone sensors
- One Garmin GPS16X-HVS for UTC clock synchronization

## Addresses and channels

| Instrument | Connection | Address or channels |
|---|---|---|
| CSAT3B #1 | SDM | Address 1 |
| CSAT3B #2 | SDM | Address 4 |
| CSAT3B #3 | SDM | Address 5 |
| IRGASON | SDM | Address 9 |
| CNR4 radiation | Differential analog | Channels 1–4 |
| CNR4 Pt-100 | Differential analog/current excitation | Channel 8, Ix1/IXR |
| Ozone #1 | Differential analog | Channel 5 |
| Ozone #2 | Differential analog | Channel 6 |
| Garmin GPS | Com1 | C1/C2 |

All SDM devices share SDM-C1, SDM-C2, and SDM-C3. Each SDM device must have a unique address.

## Sample rates

- CSAT3Bs: 20 Hz
- IRGASON: 20 Hz
- CNR4: 0.1 Hz, once every 10 seconds
- Ozone sensors: 0.5 Hz, once every 2 seconds
- GPS clock update and GPS table: 0.1 Hz

The main scan is 50 ms. Lower-rate analog measurements are placed in a 2-second slow sequence to reduce processing load on the 20 Hz scan.

## Calibration values

Replace the CNR4 sensitivities with the values from the CNR4 calibration certificate if they differ from the current constants in the program.

Set the ozone offsets individually:

```crbasic
Const Ozone1_mV_Offset = 0
Const Ozone2_mV_Offset = 0
```

Ozone conversion currently uses:

```crbasic
Ozone1 = (Ozone1_mV - Ozone1_mV_Offset) * 0.1
```

The same structure is used for ozone sensor 2.

## Data storage

The combined program writes separate binary TOB3 files to the CR3000 card:

- `Fast_20Hz`
- `CNR4_0p1Hz`
- `Ozone_0p5Hz`
- `GPS_0p1Hz`

Files are split by record count using `TableFile()` and use option `-1`, which retains newer files and removes the oldest files only when the card becomes full.

## Tall-tower converted-data headers

The files in `Converted_TT/` do not include header rows. The following
comma-delimited headers match the field order in the converted files and the
`Sample()` statements in `Combined_CR3000_AllSensors.cr3`.

All four files begin with the converted logger timestamp fields:

```text
Year,Day_of_Year,HHMM,Seconds
```

`CSV_Fast_20Hz*.dat`:

```text
Year,Day_of_Year,HHMM,Seconds,CSAT1_Ux,CSAT1_Uy,CSAT1_Uz,CSAT1_Ts,CSAT2_Ux,CSAT2_Uy,CSAT2_Uz,CSAT2_Ts,CSAT3_Ux,CSAT3_Uy,CSAT3_Uz,CSAT3_Ts,IRGA_Ux,IRGA_Uy,IRGA_Uz,IRGA_Ts,IRGA_SonicDiag,CO2_Density,H2O_Density,IRGA_GasDiag,IRGA_AirTemp,IRGA_AirPressure,CO2_Signal,H2O_Signal,CO2_Density_FastTemp,BattVolt,LoggerTemp
```

`CSV_CNR4_0p1Hz*.dat`:

```text
Year,Day_of_Year,HHMM,Seconds,SW_Up_mV,SW_Down_mV,LW_Up_mV,LW_Down_mV,SW_Up,SW_Down,LW_Up,LW_Down,CNR4_T_C,SW_Net,LW_Net,NetRadiation,Albedo
```

`CSV_Ozone_0p5Hz*.dat`:

```text
Year,Day_of_Year,HHMM,Seconds,Ozone1_mV,Ozone2_mV,Ozone1,Ozone2
```

`CSV_GPS_0p1Hz*.dat`:

```text
Year,Day_of_Year,HHMM,Seconds,Latitude_Deg,Latitude_Min,Longitude_Deg,Longitude_Min,GPS_SpeedKnots,GPS_CourseDeg,MagneticVariation_Deg,GPS_FixQuality,GPS_Satellites,GPS_Altitude_m,GPS_PPS_us,TimeSinceValidGPRMC_s,GPS_Ready,MaxClockAdjustment_ms,ClockChangeCount
```

## Important assumptions

- The CNR4 temperature element is Pt-100, not the thermistor option.
- The Garmin is connected to CR3000 `Com1` using C1/C2 and operates at 38,400 baud.
- GPS time offset is zero, so logger timestamps are maintained in UTC.
- CNR4 longwave correction is not currently applied; the current code records uncorrected longwave net radiation because the CNR4 body-temperature correction is not included in the simplified calculation.
- Compile the program in CRBasic Editor before loading it, and check `Status.SkippedScan` after connecting all instruments.
