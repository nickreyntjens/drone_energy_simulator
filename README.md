# Simulation Suite Specification Report

## 1. Introduction

This simulation suite models the operation and performance of an insect‑hunting drone over multiple days. The simulation is designed to incorporate realistic features, including:

- **Drone Behavior:**  
  The drone uses continuous kinematics and a laser weapon to hunt insects while consuming energy and managing its battery. Its performance is affected by acceleration, speed, and a requirement that it must slow down (below a maximum speed) in order to effectively shoot.

- **Field Environment:**  
  Insects are distributed in a field according to a spatial Poisson process whose local density decays exponentially from the field edges. In addition, insects may sometimes be hidden (with a 5% chance) when the drone arrives at their location—thus they remain unengaged and are carried over to the next day.

- **Day/Night Cycle:**  
  The drone is active only during the daytime (from 09:00 to 18:00) when sunlight is sufficient; outside this window, it docks at a charging station.  
  If the drone is fully charged and the active period has ended, it remains docked until the next day’s hunting period begins.

- **Daily Inflows:**  
  Each day, new insects are added to the field (the “daily inflow”), while any insects that were not killed (including hidden ones) carry over to the next day.

- **Output Metrics:**  
  Throughout the simulation, key performance metrics (such as insects killed, flight time, recharge count, energy used, and battery depreciation) are tracked.

The end goal is to perform multi‑day simulations and then use scenario sweeps (varying parameters like drone speed, insect density decay, daily insect inflow, etc.) to analyze cost per hectare and other cost metrics (e.g., battery depreciation cost per hectare).

---

Note that when equiping a sprays drone with one or more an optical units, the spray drone may at times use its laser, or at times use its liquid to combat the targeted pests. For spray drones to do dynamic flow rates, and determine where to spray, an insect density map of the field must be made anyway, so when can assume that the locations of the insects are known in the simulation.

## 2. Tools Overview

### A. One-Day Simulation (`drone_simulation_one_day.py`)

**Purpose:**  
This module simulates one day of drone operation. It reads an input file containing:
- The current day’s settings (environment, drone parameters, and state carried over from previous days), and  
- The insect population (represented as a list of insect objects, each with a position and a “hidden” flag).

It then simulates the active (hunting) period, including:
- The drone’s hunting actions (engaging insects with proper deceleration before shooting),  
- Recharging events (when the battery falls below a set threshold), and  
- The day/night idle period (when the drone remains docked at the charging station).

**GUI/No-GUI Option:**  
- The one‑day simulation module can run **with a GUI** to visually verify the drone’s behavior (showing a plot, a slider to scrub through time, and real‑time performance metrics).  
- Alternatively, it can run **without a GUI** (batch mode) to facilitate automated simulations and scenario sweeps where visual output is not required.

**Output:**  
The one‑day simulation produces a JSON output file (e.g., `one_day_output.json`) that includes a daily summary of metrics (insects killed, flight time, recharge count, energy used, battery depreciation, etc.) and an updated “state” (the remaining insect population and cumulative performance metrics) that will serve as the input for the next day.

---

### B. Multi‑Day Runner (`multi_day_runner.py`)

**Purpose:**  
The multi‑day runner acts as a driver that calls the one‑day simulation repeatedly for a specified number of days (default is 20). It performs the following tasks:
1. Reads an initial configuration file (e.g., `multi_day_config.json`) that defines global simulation parameters and the initial state.
2. For each day:
   - Generates an input file for the day (using the previous day’s output as the new “state” and adding daily insect inflow).
   - Runs the one‑day simulation (either in GUI mode for visual verification during development or in no‑GUI mode for batch processing).
   - Stores the output in a dedicated file (e.g., `day_01_output.json`, `day_02_output.json`, …).
3. Aggregates all daily outputs into a final CSV file (e.g., `daily_results.csv`) that summarizes key metrics per day.

---

### C. Scenario Runner (`scenario_runner.py`)

**Purpose:**  
The scenario runner automates multi‑day simulations across different sets of parameters. It:
1. Reads a list of scenarios (provided as a CSV or JSON file where each scenario specifies parameter values like max speed, poisson decay, daily insect inflow, maximum speed when shooting, etc.).
2. For each scenario, invokes the multi‑day runner and collects the results.
3. Produces a final aggregated CSV file (e.g., `scenario_results.csv`) where each row represents one scenario’s overall summary (e.g., average insects killed, total flight time, total battery depreciation, cost per hectare, etc.), which can be imported into Google Sheets for further analysis and visualization.

---

## 3. Input/Output Formats

### A. One-Day Simulation

#### Input File (`one_day_input.json`)
```json
{
  "day": 1,
  "environment": {
    "field_width": 1000.0,
    "field_height": 1000.0,
    "poisson_decay": 0.05,
    "active_start": "09:00",
    "active_end": "18:00",
    "initial_insect_count": 1000,
    "daily_inflow": 200
  },
  "drone": {
    "max_acc": 1.0,
    "max_speed": 5.0,
    "energy_consumption": 300.0,
    "battery_mAh": 6700.0,
    "num_cells": 3,
    "laser_shot_energy": 1.0,
    "low_battery_threshold_fraction": 0.33,
    "max_speed_when_shooting_kmh": 3.0
  },
  "state": {
    "insect_population": [
      {"position": [123.0, 456.0], "hidden": false},
      {"position": [789.0, 654.0], "hidden": false}
    ],
    "cumulative_battery_depreciation": 0.0,
    "cumulative_flight_time_sec": 0.0,
    "cumulative_energy_used": 0.0,
    "cumulative_recharge_count": 0,
    "cumulative_recharge_time_sec": 0.0
  }
}
```

#### Output File (`one_day_output.json`)
```json
{
  "day": 1,
  "results": {
    "insects_killed": 350,
    "flight_time_sec": 16200,
    "recharge_count": 3,
    "total_recharge_time_sec": 5400,
    "energy_used": 2450000,
    "battery_depreciation": 4.0
  },
  "state": {
    "insect_population": [
      {"position": [x3, y3], "hidden": false},
      {"position": [x7, y7], "hidden": false}
    ],
    "cumulative_battery_depreciation": 4.0,
    "cumulative_flight_time_sec": 16200,
    "cumulative_energy_used": 2450000,
    "cumulative_recharge_count": 3,
    "cumulative_recharge_time_sec": 5400
  }
}
```
*Note:* Insects that are hidden (5% chance) remain in the population with their `"hidden"` flag set. At the start of each day, all insects are reset to `"hidden": false`.

---

### B. Multi-Day Runner

#### Input File (`multi_day_config.json`)
```json
{
  "num_days": 20,
  "initial_insect_count": 1000,
  "daily_inflow_count": 200,
  "environment": {
    "field_width": 1000.0,
    "field_height": 1000.0,
    "poisson_decay": 0.05,
    "docking_station": [500.0, 500.0]
  },
  "drone": {
    "max_acc": 1.0,
    "max_speed": 5.0,
    "energy_consumption": 300.0,
    "battery_mAh": 6700.0,
    "num_cells": 3,
    "low_battery_threshold_fraction": 0.33,
    "laser_shot_energy": 1.0,
    "laser_shot_time": 1.0,
    "max_speed_when_shooting_kmh": 3.0
  }
}
```

#### Output Files
- **Daily Files:**  
  Each day produces a file (e.g., `day_01_output.json`, `day_02_output.json`, …) with the one‑day output format.
- **Aggregate Output:**  
  After all days are processed, the multi‑day runner aggregates key metrics from each day into a CSV file (`daily_results.csv`) with columns such as:
  
  ```
  Day,Insects Killed,Flight Time (h),Recharges,Recharge Time (h),Energy Used (J),Battery Depreciation (USD)
  1,350,4.5,3,1.5,2450000,4.0
  2,320,4.3,2,1.0,2300000,3.0
  …
  ```

---

### C. Scenario Runner

#### Input Scenarios File (CSV Example: `scenarios.csv`)
```
scenario_id,max_speed,poisson_decay,daily_inflow_count,max_speed_when_shooting_kmh
A,5.0,0.05,200,3.0
B,5.0,0.04,250,3.5
C,4.5,0.05,200,3.0
```

#### Output File (`scenario_results.csv`)
This CSV aggregates each scenario’s overall simulation metrics, with rows such as:
```
Scenario ID,Avg. Insects Killed,Total Flight Time (h),Total Recharges,Total Battery Depreciation (USD),Total Energy Used (J),Cost per Hectare (USD)
A,320,85.3,2.8,15.2,23000000,12.5
B,340,80.1,3.0,17.0,21000000,13.0
C,315,83.0,2.7,14.0,22500000,12.0
```

---

## 4. Default Parameter Values

Below are the default values currently used in auto mode:

**Environment:**
- **Field Width:** 1000.0 m  
- **Field Height:** 1000.0 m  
- **Poisson Decay:** 0.05  
- **Docking Station Location:** [500.0, 500.0]  
- **Active Period:** 09:00 to 18:00

**Insect Population:**
- **Initial Insect Count:** 1000  
- **Daily Inflow Count:** 200

**Drone Parameters:**
- **Max Acceleration:** 1.0 m/s²  
- **Max Speed:** 5.0 m/s  
- **Energy Consumption:** 300.0 W (300 J/s)  
- **Battery:**  
  - **Capacity:** 6700 mAh, 3‑cell LiPo → ~89244 Joules  
  - **Low Battery Threshold:** 33% of capacity (≈29748 J)  
- **Laser Shot Energy:** 1.0 J  
- **Laser Shot Time:** 1.0 second  
- **Max Speed When Shooting:** 3 km/h (≈0.83 m/s)

**Time Parameters:**
- **Time Step (dt):** 0.1 s  
- **Recharge Time:** 1800.0 s (30 minutes per recharge)

**Multi-Day:**
- **Number of Days:** 20

**Solar Panel:**
- **Energy per Hour:** 720000 J/h

**Battery Cost:**
- **Battery Cost:** 200 USD  
- **Battery Maximum Cycles:** 500

---

## 5. Overall Workflow Summary

1. **One-Day Simulation:**  
   - **Input:** Reads a JSON file (`one_day_input.json`) with day-specific settings, environmental parameters, drone parameters, and state (including the current insect population and cumulative metrics).  
   - **Operation:** Simulates the active (hunting) period (09:00–18:00), handling events such as hunting, recharging, and a “hidden” insect mechanism (5% chance that an insect is inaccessible and thus carried over). It can run with a GUI for visual verification of the drone’s path, speed, recharging events, etc., or run headless (no GUI) for batch processing in multi‑day or scenario simulations.  
   - **Output:** Writes a JSON summary (`one_day_output.json`) that reports daily metrics (insects killed, flight time, recharge count, energy used, battery depreciation) and an updated state (remaining insect population, cumulative data).

2. **Multi-Day Runner:**  
   - **Input:** Reads a configuration file (e.g., `multi_day_config.json`) that specifies the number of days, initial insect count, daily inflow, environmental and drone parameters.  
   - **Operation:** Iterates over the number of days. For each day, it uses the output “state” from the previous day as the input for the next day's simulation and adds the new daily insect inflow.  
   - **Output:** Writes daily output files (e.g., `day_01_output.json`, …) and produces an aggregated CSV file (`daily_results.csv`) summarizing each day’s performance metrics.

3. **Scenario Runner:**  
   - **Input:** Reads a list of scenarios (CSV or JSON) where each scenario specifies different values for key parameters (e.g., max drone speed, Poisson decay, daily insect inflow, max speed when shooting, etc.).  
   - **Operation:** For each scenario, it runs a multi‑day simulation (via the multi‑day runner) and collects a summary of the overall performance.  
   - **Output:** Produces a final CSV file (`scenario_results.csv`) where each row aggregates the metrics for a scenario (e.g., average insects killed, total flight time, total battery depreciation, cost per hectare). This file can be imported into Google Sheets for further analysis and graphing.

---

## 6. Final Remarks

This specification report defines the simulation suite’s modular design, detailing three main components:
- **One-Day Simulation:** Runs a single day’s events, with both GUI and non-GUI modes.
- **Multi-Day Runner:** Chains one-day simulations together, carrying state forward and aggregating daily output.
- **Scenario Runner:** Automates multi‑day simulations over varying parameters for sensitivity analysis.

The input and output file formats (JSON for simulation detail and CSV for aggregated results) are clearly defined along with default parameter values. This modular, file-based approach facilitates interactive development, debugging, and later batch processing or scenario analysis.

Feel free to use this report as the definitive specification when generating code or collaborating further. Let me know if you need additional details or modifications!
