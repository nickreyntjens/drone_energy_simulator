import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math

# ----------------------- Helper Functions -----------------------

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def time_to_seconds(t_str):
    """Converts a time string 'HH:MM' to seconds past midnight."""
    h, m = map(int, t_str.split(":"))
    return h * 3600 + m * 60

def generate_insects(field_width, field_height, base_rate, decay_rate, seed):
    """
    Generate insect positions over the field.
    Each insect is represented as a dictionary with keys 'position' and 'hidden'.
    """
    np.random.seed(seed)
    area = field_width * field_height
    n_candidates = np.random.poisson(lam=base_rate * area)
    insects = []
    for _ in range(n_candidates):
        x = np.random.uniform(0, field_width)
        y = np.random.uniform(0, field_height)
        d_border = min(x, field_width - x, y, field_height - y)
        p_accept = np.exp(-decay_rate * d_border)
        if np.random.rand() < p_accept:
            insects.append({"position": [x, y], "hidden": False})
    return insects

def nearest_neighbor_tsp(start, targets):
    """
    Compute a TSP route using a simple nearest neighbor heuristic.
    'targets' is a list of insect dictionaries.
    Returns a list of insect dictionaries in the chosen route order.
    """
    route = []
    available = targets.copy()
    current = np.array(start)
    while available:
        distances = [distance(current, np.array(t["position"])) for t in available]
        idx = np.argmin(distances)
        route.append(available.pop(idx))
        current = np.array(route[-1]["position"])
    return route

def nearest_neighbor_tsp_fast(start, targets):
    """
    Compute a TSP route using a vectorized nearest neighbor heuristic.
    """
    route = []
    available = targets.copy()
    current = np.array(start)
    if len(available) == 0:
        return route
    positions = np.array([t["position"] for t in available])
    while positions.shape[0] > 0:
        dists = np.linalg.norm(positions - current, axis=1)
        idx = np.argmin(dists)
        route.append(available[idx])
        available.pop(idx)
        positions = np.delete(positions, idx, axis=0)
        current = np.array(route[-1]["position"])
    return route

# ----------------------- Drone Class -----------------------

class Drone:
    def __init__(self, position, max_acc, max_speed, battery_capacity, energy_consumption,
                 laser_shot_energy, low_battery_threshold):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.max_acc = max_acc
        self.max_speed = max_speed
        self.battery_capacity = battery_capacity  # in Joules
        self.battery = battery_capacity           # starts fully charged
        self.energy_consumption = energy_consumption  # in J/s
        self.laser_shot_energy = laser_shot_energy      # in Joules per shot
        self.low_battery_threshold = low_battery_threshold
        self.total_energy_used = 0.0  # cumulative energy usage (J)
        self.total_time = 0.0         # total simulation time (seconds)
        self.path = [self.position.copy()]
        self.log = []               # log simulation events
        self.total_recharge_time = 0.0  # seconds spent recharging
        self.recharge_count = 0         # count of recharges
        self.insects_killed_count = 0   # count of insects successfully shot

    def update(self, acceleration, dt):
        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        self.position += self.velocity * dt
        self.path.append(self.position.copy())
        # Consume flight energy:
        energy_used = self.energy_consumption * dt
        self.battery -= energy_used
        self.total_energy_used += energy_used
        self.total_time += dt

    def apply_acceleration_towards(self, target, dt):
        direction = target - self.position
        dist = np.linalg.norm(direction)
        if dist == 0:
            return np.array([0.0, 0.0])
        desired_velocity = (direction / dist) * self.max_speed
        required_acc = (desired_velocity - self.velocity) / dt
        acc_norm = np.linalg.norm(required_acc)
        if acc_norm > self.max_acc:
            required_acc = (required_acc / acc_norm) * self.max_acc
        return required_acc

# ----------------------- Simulation Function -----------------------

def simulate(drone, insects, charging_station, field_width, field_height,
             dt, laser_shot_time, recharge_time, engagement_range,
             max_speed_when_shooting, active_period_end,
             use_fast_tsp=False, hidden_probability=0.05):
    """
    Simulate one day of the drone mission using the nearest-neighbor TSP approach.
      - Recalculates the TSP route using NN (or fast variant).
      - Diverts to the charging station when battery is low.
      - When within engagement range, an insect is either shot or marked hidden
        (with probability 'hidden_probability').
      - The simulation stops once the active period (in seconds) is reached.

    Returns:
        simulation_history: List of simulation log entries.
        log: List of event messages.
        remaining_insects: List of insect objects that were not killed.
    """
    DOCKING_TOLERANCE = 1.0  # meters

    simulation_history = []
    remaining_insects = insects.copy()
    route = []

    print("Starting simulation...")

    while remaining_insects:
        # End simulation if active period is over.
        if drone.total_time >= active_period_end:
            drone.log.append(f"Active period ended at t={drone.total_time:.2f}s")
            print("Active period ended.")
            break

        if not route:
            if use_fast_tsp:
                route = nearest_neighbor_tsp_fast(drone.position, remaining_insects)
            else:
                route = nearest_neighbor_tsp(drone.position, remaining_insects)
            drone.log.append(f"Recalculated route with {len(route)} targets at t={drone.total_time:.2f}s")
            print(f"Recalculating route: {len(route)} targets remaining.")

        target = route[0]
        target_pos = np.array(target["position"])

        # If battery is low, override target with the charging station.
        if drone.battery < drone.low_battery_threshold:
            target = {"position": charging_station.tolist(), "hidden": False}
            target_pos = charging_station
            drone.log.append(f"Low battery at t={drone.total_time:.2f}s, returning to charging station")
            print(f"Low battery ({drone.battery:.2f} J). Returning to charging station.")
            route = []  # Force route recalculation after recharge

        reached = False
        while not reached:
            if drone.total_time >= active_period_end:
                drone.log.append(f"Active period ended during flight at t={drone.total_time:.2f}s")
                print("Active period ended during flight.")
                reached = True
                break

            acc = drone.apply_acceleration_towards(target_pos, dt)
            current_speed = np.linalg.norm(drone.velocity) * 3.6  # km/h conversion
            current_acc = np.linalg.norm(acc)
            drone.update(acc, dt)
            simulation_history.append({
                'time': drone.total_time,
                'position': drone.position.copy().tolist(),
                'battery': drone.battery,
                'energy_used': drone.total_energy_used,
                'event': None,
                'recharge_count': drone.recharge_count,
                'recharge_time': drone.total_recharge_time,
                'insects_killed': drone.insects_killed_count,
                'speed': current_speed,
                'acceleration': current_acc
            })

            # Keep drone within field boundaries.
            drone.position[0] = min(max(drone.position[0], 0), field_width)
            drone.position[1] = min(max(drone.position[1], 0), field_height)

            # If heading to charging station:
            if np.array_equal(target_pos, charging_station):
                if distance(drone.position, charging_station) <= DOCKING_TOLERANCE:
                    reached = True
                    drone.log.append(f"Docked for recharge at t={drone.total_time:.2f}s")
                    print("Docked at charging station. Recharging...")
                    recharge_time_elapsed = 0.0
                    while recharge_time_elapsed < recharge_time and drone.total_time < active_period_end:
                        simulation_history.append({
                            'time': drone.total_time,
                            'position': drone.position.copy().tolist(),
                            'battery': drone.battery,
                            'energy_used': drone.total_energy_used,
                            'event': 'recharging',
                            'recharge_count': drone.recharge_count,
                            'recharge_time': drone.total_recharge_time,
                            'insects_killed': drone.insects_killed_count,
                            'speed': 0.0,
                            'acceleration': 0.0
                        })
                        drone.total_time += dt
                        recharge_time_elapsed += dt
                    drone.total_recharge_time += recharge_time_elapsed
                    drone.recharge_count += 1
                    drone.battery = drone.battery_capacity
                    drone.log.append(f"Recharged at t={drone.total_time:.2f}s")
                    print("Recharging complete.")
                    break

            else:
                # When within engagement range:
                if distance(drone.position, target_pos) <= engagement_range:
                    if not target["hidden"]:
                        if np.random.rand() < hidden_probability:
                            target["hidden"] = True
                            drone.log.append(f"Insect at {target['position']} became hidden at t={drone.total_time:.2f}s")
                            print(f"Insect at {target['position']} became hidden at t={drone.total_time:.2f}s.")
                            reached = True
                            route.pop(0)
                            # Do not remove from remaining_insects so it carries over.
                            break
                        else:
                            while np.linalg.norm(drone.velocity) > max_speed_when_shooting:
                                deceleration = -drone.velocity
                                drone.update(deceleration, dt)
                                simulation_history.append({
                                    'time': drone.total_time,
                                    'position': drone.position.copy().tolist(),
                                    'battery': drone.battery,
                                    'energy_used': drone.total_energy_used,
                                    'event': 'waiting_for_deceleration',
                                    'recharge_count': drone.recharge_count,
                                    'recharge_time': drone.total_recharge_time,
                                    'insects_killed': drone.insects_killed_count,
                                    'speed': np.linalg.norm(drone.velocity)*3.6,
                                    'acceleration': np.linalg.norm(deceleration)
                                })
                            reached = True
                            drone.log.append(f"Insect at {target['position']} shot at t={drone.total_time:.2f}s")
                            print(f"Insect at {target['position']} engaged at t={drone.total_time:.2f}s.")
                            simulation_history.append({
                                'time': drone.total_time,
                                'position': drone.position.copy().tolist(),
                                'battery': drone.battery,
                                'energy_used': drone.total_energy_used,
                                'event': 'insect_shot',
                                'recharge_count': drone.recharge_count,
                                'recharge_time': drone.total_recharge_time,
                                'insects_killed': drone.insects_killed_count,
                                'speed': np.linalg.norm(drone.velocity)*3.6,
                                'acceleration': current_acc
                            })
                            drone.insects_killed_count += 1
                            shot_delay = 0.0
                            while shot_delay < laser_shot_time and drone.total_time < active_period_end:
                                drone.total_time += dt
                                shot_delay += dt
                                simulation_history.append({
                                    'time': drone.total_time,
                                    'position': drone.position.copy().tolist(),
                                    'battery': drone.battery,
                                    'energy_used': drone.total_energy_used,
                                    'event': 'shooting',
                                    'recharge_count': drone.recharge_count,
                                    'recharge_time': drone.total_recharge_time,
                                    'insects_killed': drone.insects_killed_count,
                                    'speed': np.linalg.norm(drone.velocity)*3.6,
                                    'acceleration': 0.0
                                })
                            drone.battery -= drone.laser_shot_energy
                            drone.total_energy_used += drone.laser_shot_energy
                            # Remove the insect from remaining_insects.
                            for idx, insect in enumerate(remaining_insects):
                                pos = np.array(insect["position"])
                                if np.allclose(pos, target_pos, atol=engagement_range):
                                    del remaining_insects[idx]
                                    break
                            route.pop(0)
                            break
                    else:
                        reached = True
                        drone.log.append(f"Insect at {target['position']} skipped (hidden) at t={drone.total_time:.2f}s")
                        print(f"Insect at {target['position']} is hidden, skipped at t={drone.total_time:.2f}s.")
                        route.pop(0)
                        break

            if drone.battery <= 0:
                print("Drone battery depleted unexpectedly!")
                drone.log.append("Drone battery depleted unexpectedly!")
                return simulation_history, drone.log, remaining_insects

    print("Simulation complete.")
    return simulation_history, drone.log, remaining_insects

# ----------------------- Visualization -----------------------

def visualize_simulation(simulation_history, insects, charging_station, drone_path,
                           field_width, field_height, extra_info):
    """
    Create a 2D plot with a time slider and an output panel.
    Displays key simulation metrics.
    """
    if not simulation_history:
        print("No simulation history to display.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.05, right=0.70, bottom=0.25)
    ax.set_title("Drone Simulation")
    ax.set_xlim(0, field_width)
    ax.set_ylim(0, field_height)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    charging_station_marker, = ax.plot(charging_station[0], charging_station[1],
                                       marker='s', markersize=10, color='green', label='Charging Station')
    insect_x = [insect["position"][0] for insect in insects]
    insect_y = [insect["position"][1] for insect in insects]
    ax.scatter(insect_x, insect_y, color='red', label='Insects')
    drone_path_line, = ax.plot([], [], 'b-', label='Drone Path')
    drone_marker, = ax.plot([], [], 'bo', markersize=8, label='Drone')

    times = [entry['time'] for entry in simulation_history]
    t_min = times[0] / 3600.0
    t_max = times[-1] / 3600.0
    ax_slider = plt.axes([0.1, 0.1, 0.55, 0.05])
    time_slider = Slider(ax_slider, 'Flight Time (h)', t_min, t_max, valinit=t_min)

    ax_text = plt.axes([0.75, 0.25, 0.20, 0.65])
    ax_text.axis('off')
    text_out = ax_text.text(0.05, 0.95, '', transform=ax_text.transAxes,
                            verticalalignment='top', fontsize=10, family='monospace')

    def update(val):
        t_hours = time_slider.val
        t_sec = t_hours * 3600.0
        idx = 0
        for i, entry in enumerate(simulation_history):
            if entry['time'] >= t_sec:
                idx = i
                break
        state = simulation_history[idx]
        pos = state['position']
        drone_marker.set_data(pos[0], pos[1])
        path = np.array([entry['position'] for entry in simulation_history[:idx+1]])
        drone_path_line.set_data(path[:,0], path[:,1])
        flight_time_sec = state['time'] - state['recharge_time']
        flight_time_hours = flight_time_sec / 3600.0
        energy_used = state['energy_used']
        solar_panel_hours = energy_used / extra_info['solar_panel_energy_per_hour']
        recharge_time_hours = state['recharge_time'] / 3600.0
        recharge_count = state['recharge_count']
        battery_dep = (extra_info['battery_cost'] / extra_info['battery_max_cycles']) * recharge_count
        insects_killed = state['insects_killed']
        speed_kmh = state['speed']
        acceleration = state['acceleration']

        text_str = (
            f"Total Flight Time: {flight_time_hours:.2f} h\n"
            f"Energy Used: {energy_used:.2f} J\n"
            f"Solar Panel Hours: {solar_panel_hours:.2f} h\n"
            f"Recharge Time: {recharge_time_hours:.2f} h\n"
            f"Recharges: {recharge_count}\n"
            f"Battery Depreciation: ${battery_dep:.2f} USD\n"
            f"Insects Killed: {insects_killed}\n"
            f"Drone Speed: {speed_kmh:.2f} km/h\n"
            f"Drone Acceleration: {acceleration:.2f} m/s²"
        )
        text_out.set_text(text_str)
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    ax.legend()
    plt.show()

# ----------------------- Main Simulation -----------------------

def main():
    parser = argparse.ArgumentParser(description="Drone Insect-Hunting Simulation")
    parser.add_argument("--input", type=str, help="JSON input file for one-day simulation")
    parser.add_argument("--output", type=str, default="one_day_output.json", help="Output JSON file name")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--fast-tsp", action="store_true", help="Use faster TSP calculation (vectorized NN)")
    parser.add_argument("--load", type=str, help="Load simulation state from a JSON file")
    parser.add_argument("--save", type=str, help="Save simulation state to a JSON file after simulation")
    parser.add_argument("--slim-output", action="store_true",
                        help="Omit large fields (simulation history, logs) to reduce file size")
    parser.add_argument("--auto", action="store_true",
                        help="Run simulation using default parameters (auto mode)")
    args = parser.parse_args()

    # Load simulation state from load or input file, or use auto mode (default values)
    if args.load:
        with open(args.load, "r") as f:
            saved_state = json.load(f)
        day = saved_state.get("day", 1)
        environment = saved_state["environment"]
        drone_params = saved_state["drone"]
        state = saved_state["state"]
        insects = state.get("insect_population", [])
        for insect in insects:
            insect["hidden"] = False

    elif args.input:
        with open(args.input, "r") as f:
            input_data = json.load(f)
        day = input_data.get("day", 1)
        environment = input_data["environment"]
        drone_params = input_data["drone"]
        state = input_data.get("state", {
            "insect_population": [],
            "cumulative_battery_depreciation": 0.0,
            "cumulative_flight_time_sec": 0.0,
            "cumulative_energy_used": 0.0,
            "cumulative_recharge_count": 0,
            "cumulative_recharge_time_sec": 0.0
        })
        insects = state.get("insect_population", [])
        for insect in insects:
            insect["hidden"] = False

    elif args.auto:
        # Use default parameters in auto mode.
        day = 1
        environment = {
            "field_width": 1000.0,
            "field_height": 1000.0,
            "poisson_decay": 0.05,
            "active_start": "09:00",
            "active_end": "18:00",
            "initial_insect_count": 1000,
            "daily_inflow": 200,
            "docking_station": [500.0, 500.0],
            "hidden_probability": 0.05
        }
        drone_params = {
            "max_acc": 1.0,
            "max_speed": 5.0,
            "energy_consumption": 300.0,
            "battery_mAh": 6700.0,
            "num_cells": 3,
            "laser_shot_energy": 1.0,
            "low_battery_threshold_fraction": 0.33,
            "max_speed_when_shooting_kmh": 3.0,
            "battery_cost": 200.0,
            "battery_max_cycles": 500.0,
            "laser_shot_time": 1.0
        }
        state = {
            "insect_population": [],
            "cumulative_battery_depreciation": 0.0,
            "cumulative_flight_time_sec": 0.0,
            "cumulative_energy_used": 0.0,
            "cumulative_recharge_count": 0,
            "cumulative_recharge_time_sec": 0.0
        }
        if not state["insect_population"]:
            state["insect_population"] = generate_insects(
                environment["field_width"],
                environment["field_height"],
                environment["poisson_decay"],
                environment["poisson_decay"],
                seed=42)
        insects = state["insect_population"]

    else:
        # Otherwise, use interactive prompts.
        print("----- Drone Simulation Setup -----")
        field_width = float(input("Enter field width in meters (e.g., 1000): ") or 1000)
        field_height = float(input("Enter field height in meters (e.g., 1000): ") or 1000)
        base_rate = float(input("Enter base insect rate (e.g., 0.05): ") or 0.05)
        decay_rate = float(input("Enter decay rate (e.g., 0.05): ") or 0.05)
        active_start = input("Active start (HH:MM, e.g. 09:00): ") or "09:00"
        active_end = input("Active end (HH:MM, e.g. 18:00): ") or "18:00"
        docking_x = float(input("Enter docking station X (e.g., 500): ") or 500)
        docking_y = float(input("Enter docking station Y (e.g., 500): ") or 500)
        hidden_probability = float(input("Hidden insect probability (0.05 for 5%): ") or 0.05)
        max_acc = float(input("Drone max acceleration (m/s^2, e.g., 1.0): ") or 1.0)
        max_speed = float(input("Drone max speed (m/s, e.g., 5.0): ") or 5.0)
        energy_consumption = float(input("Drone energy consumption (W, e.g., 300): ") or 300)
        battery_mah = float(input("Battery capacity (mAh, e.g., 6700): ") or 6700)
        num_cells = int(input("Number of LiPo cells (e.g., 3): ") or 3)
        laser_shot_energy = float(input("Laser shot energy (J, e.g., 1.0): ") or 1.0)
        low_battery_fraction = float(input("Low battery threshold fraction (0.33): ") or 0.33)
        max_speed_shooting_kmh = float(input("Max speed shooting (km/h, e.g., 3.0): ") or 3.0)

        day = 1
        environment = {
            "field_width": field_width,
            "field_height": field_height,
            "poisson_decay": decay_rate,
            "active_start": active_start,
            "active_end": active_end,
            "initial_insect_count": 1000,
            "daily_inflow": 200,
            "docking_station": [docking_x, docking_y],
            "hidden_probability": hidden_probability
        }
        drone_params = {
            "max_acc": max_acc,
            "max_speed": max_speed,
            "energy_consumption": energy_consumption,
            "battery_mAh": battery_mah,
            "num_cells": num_cells,
            "laser_shot_energy": laser_shot_energy,
            "low_battery_threshold_fraction": low_battery_fraction,
            "max_speed_when_shooting_kmh": max_speed_shooting_kmh,
            "battery_cost": 200.0,
            "battery_max_cycles": 500.0,
            "laser_shot_time": 1.0
        }
        state = {
            "insect_population": [],
            "cumulative_battery_depreciation": 0.0,
            "cumulative_flight_time_sec": 0.0,
            "cumulative_energy_used": 0.0,
            "cumulative_recharge_count": 0,
            "cumulative_recharge_time_sec": 0.0
        }
        if not state["insect_population"]:
            state["insect_population"] = generate_insects(field_width, field_height, base_rate, decay_rate, seed=42)
        insects = state["insect_population"]

    # Extract environment & drone parameters.
    field_width = environment["field_width"]
    field_height = environment["field_height"]
    charging_station = np.array(environment["docking_station"])
    active_start_sec = time_to_seconds(environment.get("active_start", "09:00"))
    active_end_sec = time_to_seconds(environment.get("active_end", "18:00"))
    active_period = active_end_sec - active_start_sec
    HIDDEN_PROBABILITY = environment.get("hidden_probability", 0.05)

    MAX_ACCEL = drone_params["max_acc"]
    MAX_SPEED = drone_params["max_speed"]
    DRONE_ENERGY_CONSUMPTION = drone_params["energy_consumption"]
    BATTERY_MAH = drone_params["battery_mAh"]
    NUM_CELLS = drone_params["num_cells"]
    BATTERY_VOLTAGE = NUM_CELLS * 3.7
    BATTERY_CAPACITY = (BATTERY_MAH / 1000) * BATTERY_VOLTAGE * 3600
    LOW_BATTERY_THRESHOLD = BATTERY_CAPACITY * drone_params["low_battery_threshold_fraction"]
    LASER_SHOT_ENERGY = drone_params["laser_shot_energy"]
    LASER_SHOT_TIME = drone_params.get("laser_shot_time", 1.0)
    ENGAGEMENT_RANGE = 2.0
    max_speed_shooting_kmh = drone_params["max_speed_when_shooting_kmh"]
    MAX_SPEED_WHEN_SHOOTING = max_speed_shooting_kmh / 3.6

    BATTERY_COST = drone_params.get("battery_cost", 200.0)
    BATTERY_MAX_CYCLES = drone_params.get("battery_max_cycles", 500.0)
    DT = 0.1
    RECHARGE_TIME = 1800.0

    drone = Drone(
        position=charging_station,
        max_acc=MAX_ACCEL,
        max_speed=MAX_SPEED,
        battery_capacity=BATTERY_CAPACITY,
        energy_consumption=DRONE_ENERGY_CONSUMPTION,
        laser_shot_energy=LASER_SHOT_ENERGY,
        low_battery_threshold=LOW_BATTERY_THRESHOLD
    )

    simulation_history, log, remaining_insects = simulate(
        drone, insects, charging_station, field_width, field_height,
        DT, LASER_SHOT_TIME, RECHARGE_TIME,
        ENGAGEMENT_RANGE, MAX_SPEED_WHEN_SHOOTING,
        active_period,
        use_fast_tsp=args.fast_tsp,
        hidden_probability=HIDDEN_PROBABILITY
    )

    flight_time_sec = drone.total_time - drone.total_recharge_time
    energy_used = drone.total_energy_used
    recharge_count = drone.recharge_count
    battery_depreciation = (BATTERY_COST / BATTERY_MAX_CYCLES) * recharge_count

    print("\n--- Simulation Log ---")
    for entry in log:
        print(entry)
    print(f"\nTotal simulation time: {drone.total_time/3600:.2f} hours")
    print(f"Total energy used: {energy_used:.2f} Joules")
    print(f"Total recharge time: {drone.total_recharge_time/3600:.2f} hours")
    print(f"Number of recharges: {recharge_count}")
    print(f"Battery Depreciation: ${battery_depreciation:.2f} USD")
    print(f"Insects Killed: {drone.insects_killed_count}")

    results = {
        "insects_killed": drone.insects_killed_count,
        "flight_time_sec": flight_time_sec,
        "recharge_count": recharge_count,
        "total_recharge_time_sec": drone.total_recharge_time,
        "energy_used": energy_used,
        "battery_depreciation": battery_depreciation
    }
    # Here we update the state to use only the remaining insects.
    new_state = {
        "insect_population": remaining_insects,
        "cumulative_battery_depreciation": state.get("cumulative_battery_depreciation", 0.0) + battery_depreciation,
        "cumulative_flight_time_sec": state.get("cumulative_flight_time_sec", 0.0) + flight_time_sec,
        "cumulative_energy_used": state.get("cumulative_energy_used", 0.0) + energy_used,
        "cumulative_recharge_count": state.get("cumulative_recharge_count", 0) + recharge_count,
        "cumulative_recharge_time_sec": state.get("cumulative_recharge_time_sec", 0.0) + drone.total_recharge_time
    }
    drone_path_list = [p.tolist() for p in drone.path]
    output_data = {
        "day": day,
        "results": results,
        "state": new_state,
        "environment": environment,
        "drone": drone_params
    }
    if not args.slim_output:
        output_data["simulation_history"] = simulation_history
        output_data["drone_path"] = drone_path_list
        output_data["log"] = log

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Simulation output saved to {args.output}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"Simulation state saved to {args.save}")

    if not args.headless:
        extra_info = {
            "battery_cost": BATTERY_COST,
            "battery_max_cycles": BATTERY_MAX_CYCLES,
            "solar_panel_energy_per_hour": 720000.0
        }
        visualize_simulation(simulation_history, insects, charging_station, drone.path,
                             field_width, field_height, extra_info)

if __name__ == '__main__':
    main()

