import json
import subprocess
import os
import csv
import argparse
import numpy as np

def generate_fixed_insects(count, field_width, field_height, decay_rate, seed=None):
    """
    Generate exactly 'count' insect objects according to a spatial acceptance criterion.
    Each insect is represented as a dictionary with keys 'position' and 'hidden'.
    
    The acceptance probability decays exponentially with the distance to the nearest field edge.
    (i.e. insects are more likely to appear near field boundaries).
    
    Parameters:
        count (int): Number of insects to generate.
        field_width (float): Width of the field.
        field_height (float): Height of the field.
        decay_rate (float): Rate at which the acceptance probability decays.
        seed (int, optional): Seed for reproducibility.
    
    Returns:
        List[dict]: A list of insect objects.
    """
    if seed is not None:
        np.random.seed(seed)
    insects = []
    while len(insects) < count:
        x = np.random.uniform(0, field_width)
        y = np.random.uniform(0, field_height)
        # Calculate the distance to the nearest border.
        d_border = min(x, field_width - x, y, field_height - y)
        # The acceptance probability decays from 1 at the border to a lower probability in the center.
        p_accept = np.exp(-decay_rate * d_border)
        if np.random.rand() < p_accept:
            insects.append({"position": [x, y], "hidden": False})
    return insects

def main():
    parser = argparse.ArgumentParser(description="Multi-day Runner for Drone Insect-Hunting Simulation")
    parser.add_argument("--config", type=str, default="multi_day_config.json",
                        help="Path to multi-day configuration JSON file (default: multi_day_config.json)")
    parser.add_argument("--gui", action="store_true",
                        help="Run one-day simulation with GUI (default is headless)")
    parser.add_argument("--one-day-script", type=str, default="drone_simulation_one_day.py",
                        help="Path to the one-day simulation script (default: drone_simulation_one_day.py)")
    args = parser.parse_args()

    # Load the multi-day configuration file.
    with open(args.config, "r") as f:
        config = json.load(f)

    # Extract global simulation parameters.
    num_days = config.get("num_days", 20)
    initial_insect_count = config.get("initial_insect_count", 1000)
    daily_inflow_count = config.get("daily_inflow_count", 200)

    # Environment parameters (field dimensions, poisson decay, docking station, etc.).
    environment = config["environment"]
    field_width = environment["field_width"]
    field_height = environment["field_height"]
    decay_rate = environment["poisson_decay"]
    # Use the docking station location defined in the config.
    docking_station = environment["docking_station"]

    # Drone parameters.
    drone_params = config["drone"]

    # Create the initial simulation state.
    # The state holds the insect population plus cumulative performance metrics.
    state = {
        "insect_population": generate_fixed_insects(initial_insect_count, field_width, field_height, decay_rate, seed=42),
        "cumulative_battery_depreciation": 0.0,
        "cumulative_flight_time_sec": 0.0,
        "cumulative_energy_used": 0.0,
        "cumulative_recharge_count": 0,
        "cumulative_recharge_time_sec": 0.0
    }

    daily_results = []  # List to store key metrics for each day.

    # Run simulation for each day.
    for day in range(1, num_days + 1):
        print(f"---------- Starting simulation for Day {day} ----------")
        
        # Record the starting insect population count for the day.
        starting_population = len(state["insect_population"])

        # Prepare the input data for the one-day simulation.
        day_input = {
            "day": day,
            "environment": environment,
            "drone": drone_params,
            "state": state
        }
        input_filename = f"day_{day:02d}_input.json"
        output_filename = f"day_{day:02d}_output.json"
        with open(input_filename, "w") as f:
            json.dump(day_input, f, indent=4)
        print(f"Day {day} input saved to {input_filename}")
        print(f"Starting insect population: {starting_population}")

        # Build the command to call the one-day simulation script.
        command = ["python", args.one_day_script,
                   "--input", input_filename,
                   "--output", output_filename,
                   "--slim-output"]  # Flag to reduce output size.
        if not args.gui:
            command.append("--headless")

        # Call the one-day simulation script.
        print("Running one-day simulation...")
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Simulation for Day {day} failed with return code {result.returncode}. Exiting.")
            break

        # Load the one-day simulation output.
        with open(output_filename, "r") as f:
            day_output = json.load(f)

        # Extract key performance metrics from the output.
        results = day_output.get("results", {})
        daily_results.append({
            "Day": day,
            "Starting Population": starting_population,
            "Insects Killed": results.get("insects_killed", 0),
            "Flight Time (h)": round(results.get("flight_time_sec", 0) / 3600.0, 2),
            "Recharges": results.get("recharge_count", 0),
            "Recharge Time (h)": round(results.get("total_recharge_time_sec", 0) / 3600.0, 2),
            "Energy Used (J)": results.get("energy_used", 0),
            "Battery Depreciation (USD)": round(results.get("battery_depreciation", 0), 2)
        })

        # Update the state for the next day.
        state = day_output.get("state", state)

        # Add daily insect inflow to the insect population.
        daily_inflow = generate_fixed_insects(daily_inflow_count, field_width, field_height, decay_rate)
        for insect in daily_inflow:
            insect["hidden"] = False
        state["insect_population"].extend(daily_inflow)

        print(f"Day {day} simulation complete. {results.get('insects_killed', 0)} insects killed.")

    # After all days are simulated, aggregate daily outputs into a CSV file.
    csv_filename = "daily_results.csv"
    fieldnames = [
        "Day",
        "Starting Population",
        "Insects Killed",
        "Flight Time (h)",
        "Recharges",
        "Recharge Time (h)",
        "Energy Used (J)",
        "Battery Depreciation (USD)"
    ]
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in daily_results:
            writer.writerow(row)
    print(f"\nAggregated daily results saved to {csv_filename}")

if __name__ == "__main__":
    main()

