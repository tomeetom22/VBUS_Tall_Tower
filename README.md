# VBUS Tall Tower CR3000 Codes

CRBasic programs for the VBUS tall-tower sensor system.

## Combined program

`Combined_CR3000_AllSensors.cr3` combines:

- Three CSAT3B sonic anemometers on the shared SDM bus
- One IRGASON
- One CNR4 net radiometer
- Three analog ozone sensors
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
| Ozone #3 | Differential analog | Channel 7 |
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
Const Ozone3_mV_Offset = 0
```

Ozone conversion currently uses:

```crbasic
Ozone1 = (Ozone1_mV - Ozone1_mV_Offset) * 0.1
```

The same structure is used for ozone sensors 2 and 3.

## Data storage

The combined program writes separate binary TOB3 files to the CR3000 card:

- `Fast_20Hz`
- `CNR4_0p1Hz`
- `Ozone_0p5Hz`
- `GPS_0p1Hz`

Files are split by record count using `TableFile()` and use option `-1`, which retains newer files and removes the oldest files only when the card becomes full.

## Important assumptions

- The CNR4 temperature element is Pt-100, not the thermistor option.
- The Garmin is connected to CR3000 `Com1` using C1/C2 and operates at 38,400 baud.
- GPS time offset is zero, so logger timestamps are maintained in UTC.
- CNR4 longwave correction is not currently applied; the current code records uncorrected longwave net radiation because the CNR4 body-temperature correction is not included in the simplified calculation.
- Compile the program in CRBasic Editor before loading it, and check `Status.SkippedScan` after connecting all instruments.