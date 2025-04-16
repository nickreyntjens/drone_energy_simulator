import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math

# Uncomment to set an interactive backend if needed:
# import matplotlib
# matplotlib.use('TkAgg')

#plt.ion()  # Turn on interactive mode

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
    Each insect is represented as a dictionary with key 'position'.
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
            insects.append({"position": [x, y]})
    return insects

def nearest_neighbor_tsp(start, targets):
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

def get_insect_subset_by_grid(insects, field_width, field_height, nrows, ncols):
    row_height = field_height / nrows
    col_width = field_width / ncols
    cell_counts = {}
    cell_insects = {}
    for insect in insects:
        x, y = insect["position"]
        row = int(y // row_height)
        col = int(x // col_width)
        row = min(row, nrows - 1)
        col = min(col, ncols - 1)
        key = (row, col)
        cell_counts[key] = cell_counts.get(key, 0) + 1
        cell_insects.setdefault(key, []).append(insect)
    best_cell = max(cell_counts, key=cell_counts.get)
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr = best_cell[0] + dr
            nc = best_cell[1] + dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                neighbors.append((nr, nc))
    subset = []
    for cell in neighbors:
        subset.extend(cell_insects.get(cell, []))
    return subset

def recalc_route(current_position, remaining_insects, field_width, field_height, use_fast_tsp=False):
    """Recalculate the TSP route for the remaining insects."""
    if len(remaining_insects) > 1000:
        if len(remaining_insects) > 10000:
            nrows, ncols = 10, 10
        else:
            nrows, ncols = 2, 5
        subset = get_insect_subset_by_grid(remaining_insects, field_width, field_height, nrows, ncols)
        if not subset:
            subset = remaining_insects
        if use_fast_tsp:
            return nearest_neighbor_tsp_fast(current_position, subset)
        else:
            return nearest_neighbor_tsp(current_position, subset)
    else:
        if use_fast_tsp:
            return nearest_neighbor_tsp_fast(current_position, remaining_insects)
        else:
            return nearest_neighbor_tsp(current_position, remaining_insects)

def fly_towards(drone, target_pos, dt, threshold, simulation_history):
    """
    Update the drone by flying one time-step toward target_pos.
    Append a simulation record. Return True if within threshold of target_pos.
    """
    acc = drone.apply_acceleration_towards(target_pos, dt)
    current_speed = np.linalg.norm(drone.velocity) * 3.6
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
    return distance(drone.position, target_pos) <= threshold

# ----------------------- Drone Class -----------------------

class Drone:
    def __init__(self, position, max_acc, max_speed, battery_capacity, energy_consumption,
                 laser_shot_energy, low_battery_threshold):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.max_acc = max_acc
        self.max_speed = max_speed
        self.battery_capacity = battery_capacity   # in Joules
        self.battery = battery_capacity            # fully charged initially
        self.energy_consumption = energy_consumption  # J/s
        self.laser_shot_energy = laser_shot_energy    # energy per shot (J)
        self.low_battery_threshold = low_battery_threshold
        self.total_energy_used = 0.0
        self.total_time = 0.0
        self.path = [self.position.copy()]
        self.log = []
        self.total_recharge_time = 0.0
        self.recharge_count = 0
        self.insects_killed_count = 0

    def update(self, acceleration, dt):
        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        self.position += self.velocity * dt
        self.path.append(self.position.copy())
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

# ----------------------- Simulation Function (State Machine) -----------------------

def simulate(drone, insects, charging_station, field_width, field_height,
             dt, lock_time, recharge_time, engagement_range,
             max_speed_when_shooting, active_period, use_fast_tsp=False):
    """
    Simulate one day of the drone mission using a state-machine approach.
    States:
      - "target_insect": go towards next insect target.
      - "goto_recharge": drone flies to the charging station.
      - "recharge": drone waits to recharge.
      
    fly_towards() updates the drone’s position toward a target until a threshold (engagement or docking)
    is reached. The route is recalculated only when needed.
    """
    DOCKING_TOLERANCE = 1.0  # meters
    simulation_history = []
    remaining_insects = insects.copy()
    state = "target_insect"  # initial state: target insects
    route = []
    
    while (remaining_insects or state in ("goto_recharge", "recharge")) and drone.total_time < active_period:
        
        # Transition to recharge state if battery is low.
        if state == "target_insect" and drone.battery < drone.low_battery_threshold:
            print(f"Low battery ({drone.battery:.2f} J). Switching to recharge mode.")
            state = "goto_recharge"
            route = []  # drop insect route
        
        if state == "target_insect":
            # Recalculate route if needed.
            if not route:
                if remaining_insects:
                    route = recalc_route(drone.position, remaining_insects, field_width, field_height, use_fast_tsp)
                    drone.log.append(f"Recalculated route with {len(route)} targets at t={drone.total_time:.2f}s")
                    print(f"Recalculating route: {len(route)} targets selected.")
                else:
                    break  # no insects left
            # Set the target as the first insect in the route.
            target = route[0]
            target_pos = np.array(target["position"])
            # Fly toward the insect until within engagement range.
            reached = fly_towards(drone, target_pos, dt, engagement_range, simulation_history)
            if reached:
                # Check which insects are within engagement range.
                engaged = [insect for insect in remaining_insects
                           if distance(drone.position, np.array(insect["position"])) <= engagement_range]
                if engaged:
                    num_lasers = drone_params.get("num_lasers", 1)
                    targets_to_lock = engaged[:num_lasers]
                    lock_delay = 0.0
                    while lock_delay < lock_time and drone.total_time < active_period:
                        drone.total_time += dt
                        lock_delay += dt
                        simulation_history.append({
                            'time': drone.total_time,
                            'position': drone.position.copy().tolist(),
                            'battery': drone.battery,
                            'energy_used': drone.total_energy_used,
                            'event': 'locking',
                            'recharge_count': drone.recharge_count,
                            'recharge_time': drone.total_recharge_time,
                            'insects_killed': drone.insects_killed_count,
                            'speed': np.linalg.norm(drone.velocity) * 3.6,
                            'acceleration': 0.0
                        })
                    total_laser_energy = drone.laser_shot_energy * len(targets_to_lock)
                    drone.battery -= total_laser_energy
                    drone.total_energy_used += total_laser_energy
                    drone.insects_killed_count += len(targets_to_lock)
                    drone.log.append(f"Shooting insects at {[t['position'] for t in targets_to_lock]} at t={drone.total_time:.2f}s")
                    print(f"Shooting {len(targets_to_lock)} targets concurrently at t={drone.total_time:.2f}s.")
                    for t in targets_to_lock:
                        if t in remaining_insects:
                            remaining_insects.remove(t)
                        if t in route:
                            route.remove(t)
                    continue
                else:
                    # Remove the current target if engagement did not occur.
                    if route:
                        route.pop(0)
                    continue

        elif state == "goto_recharge":
            # Fly toward charging station.
            reached = fly_towards(drone, charging_station, dt, DOCKING_TOLERANCE, simulation_history)
            if reached:
                print("Drone docked. Initiating recharge.")
                state = "recharge"
            continue

        elif state == "recharge":
            recharge_elapsed = 0.0
            while recharge_elapsed < recharge_time and drone.total_time < active_period:
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
                recharge_elapsed += dt
            drone.total_recharge_time += recharge_elapsed
            drone.recharge_count += 1
            drone.battery = drone.battery_capacity
            print("Recharging complete. Resuming insect-hunting.")
            state = "target_insect"
            route = []  # Force recalculation of route.
            continue

        # Continue flying toward the current target.
        reached = False
        while not reached and drone.total_time < active_period:
            acc = drone.apply_acceleration_towards(target_pos, dt)
            current_speed = np.linalg.norm(drone.velocity) * 3.6
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
            # Clamp position within the field.
            drone.position[0] = min(max(drone.position[0], 0), field_width)
            drone.position[1] = min(max(drone.position[1], 0), field_height)
            if distance(drone.position, target_pos) <= engagement_range:
                reached = True
                break
            if drone.battery <= 0:
                print("Drone battery depleted unexpectedly!")
                drone.log.append("Drone battery depleted unexpectedly!")
                return simulation_history, drone.log, remaining_insects

    print("Simulation complete.")
    return simulation_history, drone.log, remaining_insects

# ----------------------- Global Configuration -----------------------

insect_params = {
    'field_width': 1000.0,
    'field_height': 1000.0,
    'base_rate': 0.05,
    'decay_rate': 0.05,
    'seed': 42
}

drone_params = {
    'max_acc': 1.0,
    'max_speed': 5.0,
    'energy_consumption': 300.0,
    'battery_mAh': 6700.0,
    'num_cells': 3,
    'laser_shot_energy': 1.0,
    'low_battery_threshold_fraction': 0.33,
    'max_speed_when_shooting_kmh': 3.0,
    'lock_time': 1.0,           # seconds to lock on target
    'engagement_range': 2.0,    # engagement distance (m)
    'num_lasers': 2             # number of lasers operating concurrently
}

sim_params = {
    "dt": 0.1,
    "recharge_time": 1800.0
}

environment = {
    "docking_station": [500.0, 500.0],
    "active_start": "09:00",
    "active_end": "18:00"
}

current_insects = []
simulation_history = []

# ----------------------- Main GUI and Visualization -----------------------

fig = plt.figure(figsize=(12, 8))
ax_sim = fig.add_axes([0.05, 0.30, 0.65, 0.65])
ax_sim.set_title("Drone Simulation")
ax_sim.set_xlim(0, insect_params['field_width'])
ax_sim.set_ylim(0, insect_params['field_height'])
ax_sim.set_xlabel("X (m)")
ax_sim.set_ylabel("Y (m)")

charging_station = np.array(environment["docking_station"])
charging_station_marker, = ax_sim.plot(charging_station[0], charging_station[1],
                                       marker='s', markersize=10, color='green', label='Charging Station')
insect_scatter = ax_sim.scatter([], [], color='red', label='Insects')
drone_path_line, = ax_sim.plot([], [], 'b-', label='Drone Path')
drone_marker, = ax_sim.plot([], [], 'bo', markersize=8, label='Drone')
ax_sim.legend()

ax_text = fig.add_axes([0.73, 0.30, 0.25, 0.65])
ax_text.axis('off')
# Initial placeholder text; will be updated while scrubbing.
text_out = ax_text.text(0.05, 0.95, "Simulation info will appear here.",
                        transform=ax_text.transAxes, verticalalignment='top',
                        fontsize=9, family='monospace')

ax_slider = fig.add_axes([0.05, 0.15, 0.65, 0.05])
time_slider = Slider(ax_slider, 'Flight Time (h)', 0, 1, valinit=0)
ax_slider.set_visible(False)

insect_count_text = fig.text(0.05, 0.97, "Insect Count: 0", fontsize=12, color='blue')

# ----------------------- Update Simulation View Callback -----------------------

def update_simulation_view(val):
    """
    Update the drone marker, flight path, and info box based on the slider's time value.
    Only the records up to the slider's current time are drawn.
    """
    if not simulation_history:
        return
    current_time_seconds = time_slider.val * 3600.0
    idx = min(range(len(simulation_history)), key=lambda i: abs(simulation_history[i]['time'] - current_time_seconds))
    current_record = simulation_history[idx]
    # Update drone marker position.
    pos = current_record['position']
    drone_marker.set_data(pos[0], pos[1])
    # Update the flight path up to the current record.
    path_up_to = np.array([rec['position'] for rec in simulation_history[:idx+1]])
    if path_up_to.size:
        drone_path_line.set_data(path_up_to[:, 0], path_up_to[:, 1])
    # Update the information box with details at this time.
    info_str = (
        f"Time: {current_record['time']:.2f} s\n"
        f"Battery: {current_record['battery']:.2f} J\n"
        f"Insects killed: {current_record['insects_killed']}\n"
        f"Recharges: {current_record['recharge_count']}\n"
        f"Speed: {current_record['speed']:.2f} km/h"
    )
    text_out.set_text(info_str)
    fig.canvas.draw_idle()

# ----------------------- Run Simulation Callback -----------------------

def run_simulation_callback(event):
    global current_insects, simulation_history, time_slider

    if not current_insects:
        current_insects = generate_insects(
            insect_params['field_width'],
            insect_params['field_height'],
            insect_params['base_rate'],
            insect_params['decay_rate'],
            insect_params['seed']
        )
        ix = [i["position"][0] for i in current_insects]
        iy = [i["position"][1] for i in current_insects]
        insect_scatter.set_offsets(np.column_stack((ix, iy)))
        ax_sim.set_xlim(0, insect_params['field_width'])
        ax_sim.set_ylim(0, insect_params['field_height'])
        insect_count_text.set_text(f"Insect Count: {len(current_insects)}")
        fig.canvas.draw_idle()
    
    BATTERY_VOLTAGE = drone_params['num_cells'] * 3.7
    BATTERY_CAPACITY = (drone_params['battery_mAh'] / 1000) * BATTERY_VOLTAGE * 3600
    LOW_BATTERY_THRESHOLD = BATTERY_CAPACITY * drone_params['low_battery_threshold_fraction']
    
    drone = Drone(
        position=charging_station,
        max_acc=drone_params['max_acc'],
        max_speed=drone_params['max_speed'],
        battery_capacity=BATTERY_CAPACITY,
        energy_consumption=drone_params['energy_consumption'],
        laser_shot_energy=drone_params['laser_shot_energy'],
        low_battery_threshold=LOW_BATTERY_THRESHOLD
    )
    
    active_start_sec = time_to_seconds(environment["active_start"])
    active_end_sec = time_to_seconds(environment["active_end"])
    active_period = active_end_sec - active_start_sec
    
    simulation_history, log, remaining_insects = simulate(
        drone, current_insects, np.array(environment["docking_station"]),
        insect_params['field_width'], insect_params['field_height'],
        sim_params["dt"],
        drone_params['lock_time'],
        sim_params["recharge_time"],
        drone_params['engagement_range'],
        drone_params['max_speed_when_shooting_kmh'] / 3.6,
        active_period,
        use_fast_tsp=False
    )
    
    if simulation_history:
        t_min = simulation_history[0]['time'] / 3600.0
        t_max = simulation_history[-1]['time'] / 3600.0
        time_slider.ax.clear()
        time_slider = Slider(time_slider.ax, 'Flight Time (h)', t_min, t_max, valinit=t_min)
        time_slider.on_changed(update_simulation_view)
        ax_slider.set_visible(True)
    
    # Draw the complete drone path (will be updated by the slider).
    drone_path_arr = np.array(drone.path)
    if drone_path_arr.size:
        drone_path_line.set_data(drone_path_arr[:, 0], drone_path_arr[:, 1])
    fig.canvas.draw_idle()

# ----------------------- Main Button Panel -----------------------

ax_btn_insects = fig.add_axes([0.05, 0.05, 0.13, 0.05])
btn_insects = Button(ax_btn_insects, "Gen Insects")

ax_btn_drone = fig.add_axes([0.19, 0.05, 0.13, 0.05])
btn_drone = Button(ax_btn_drone, "Config Drone")

ax_btn_sim = fig.add_axes([0.33, 0.05, 0.13, 0.05])
btn_sim = Button(ax_btn_sim, "Sim Params")

ax_btn_run = fig.add_axes([0.47, 0.05, 0.13, 0.05])
btn_run = Button(ax_btn_run, "Run Sim")

ax_btn_help = fig.add_axes([0.61, 0.05, 0.13, 0.05])
btn_help = Button(ax_btn_help, "Help")

ax_btn_config = fig.add_axes([0.75, 0.05, 0.13, 0.05])
btn_config = Button(ax_btn_config, "Config Report")

# ----------------------- Configuration Windows -----------------------
# (These windows remain mostly unchanged.)

def sim_params_config_window(event):
    fig_sim_params = plt.figure("Simulation Parameters", figsize=(6, 5))
    ax1 = fig_sim_params.add_axes([0.15, 0.75, 0.7, 0.1])  # dt
    ax2 = fig_sim_params.add_axes([0.15, 0.55, 0.7, 0.1])  # recharge time
    ax3 = fig_sim_params.add_axes([0.15, 0.35, 0.7, 0.1])  # docking station x
    ax4 = fig_sim_params.add_axes([0.15, 0.15, 0.7, 0.1])  # docking station y

    slider_dt = Slider(ax1, 'dt (s)', 0.01, 1.0, valinit=sim_params["dt"])
    slider_recharge = Slider(ax2, 'Recharge Time (s)', 600, 3600, valinit=sim_params["recharge_time"])
    slider_dock_x = Slider(ax3, 'Docking X', 0, insect_params["field_width"], 
                             valinit=environment["docking_station"][0])
    slider_dock_y = Slider(ax4, 'Docking Y', 0, insect_params["field_height"], 
                             valinit=environment["docking_station"][1])
    
    ax_apply = fig_sim_params.add_axes([0.65, 0.02, 0.3, 0.1])
    btn_apply = Button(ax_apply, "Apply")
    
    def apply_sim_params(event):
        print("Sim Params Apply button clicked.")
        sim_params["dt"] = slider_dt.val
        sim_params["recharge_time"] = slider_recharge.val
        environment["docking_station"] = [slider_dock_x.val, slider_dock_y.val]
        print("Updated sim_params:", sim_params)
        print("Updated docking_station:", environment["docking_station"])
        plt.close(fig_sim_params)
        fig.canvas.draw()
    
    btn_apply.on_clicked(apply_sim_params)
    plt.show(block=True)

def insects_config_window(event):
    fig_insect = plt.figure("Insect Generation Settings", figsize=(6, 4))
    ax1 = fig_insect.add_axes([0.15, 0.75, 0.7, 0.12])
    ax2 = fig_insect.add_axes([0.15, 0.55, 0.7, 0.12])
    ax3 = fig_insect.add_axes([0.15, 0.35, 0.7, 0.12])
    ax4 = fig_insect.add_axes([0.15, 0.15, 0.7, 0.12])
    
    slider_field_width = Slider(ax1, 'Field Width', 500, 2000, valinit=insect_params['field_width'])
    slider_field_height = Slider(ax2, 'Field Height', 500, 2000, valinit=insect_params['field_height'])
    slider_base_rate = Slider(ax3, 'Base Rate', 0.01, 0.1, valinit=insect_params['base_rate'])
    slider_decay_rate = Slider(ax4, 'Decay Rate', 0.01, 0.1, valinit=insect_params['decay_rate'])
    
    ax_apply = fig_insect.add_axes([0.65, 0.02, 0.3, 0.1])
    btn_apply = Button(ax_apply, "Apply")
    
    def apply_insect_settings(event):
        global current_insects
        print("Insect Config Apply button clicked.")
        insect_params['field_width'] = slider_field_width.val
        insect_params['field_height'] = slider_field_height.val
        insect_params['base_rate'] = slider_base_rate.val
        insect_params['decay_rate'] = slider_decay_rate.val
        current_insects = generate_insects(
            insect_params['field_width'],
            insect_params['field_height'],
            insect_params['base_rate'],
            insect_params['decay_rate'],
            insect_params['seed']
        )
        ax_sim.set_xlim(0, insect_params['field_width'])
        ax_sim.set_ylim(0, insect_params['field_height'])
        if current_insects:
            ix = [i["position"][0] for i in current_insects]
            iy = [i["position"][1] for i in current_insects]
        else:
            ix, iy = [], []
        insect_scatter.set_offsets(np.column_stack((ix, iy)))
        insect_count_text.set_text(f"Insect Count: {len(current_insects)}")
        print("Generated", len(current_insects), "insects.")
        plt.close(fig_insect)
        fig.canvas.draw()
    
    btn_apply.on_clicked(apply_insect_settings)
    plt.show(block=True)

def drone_config_window(event):
    fig_drone = plt.figure("Drone Configuration Settings", figsize=(7, 9))
    ax1 = fig_drone.add_axes([0.15, 0.88, 0.7, 0.05])
    ax2 = fig_drone.add_axes([0.15, 0.78, 0.7, 0.05])
    ax3 = fig_drone.add_axes([0.15, 0.68, 0.7, 0.05])
    ax4 = fig_drone.add_axes([0.15, 0.58, 0.7, 0.05])
    ax5 = fig_drone.add_axes([0.15, 0.48, 0.7, 0.05])
    ax6 = fig_drone.add_axes([0.15, 0.38, 0.7, 0.05])
    ax7 = fig_drone.add_axes([0.15, 0.28, 0.7, 0.05])
    ax8 = fig_drone.add_axes([0.15, 0.18, 0.7, 0.05])
    ax9 = fig_drone.add_axes([0.15, 0.08, 0.7, 0.05])
    
    slider_max_acc = Slider(ax1, 'Max Acc', 0.5, 2.0, valinit=drone_params['max_acc'])
    slider_max_speed = Slider(ax2, 'Max Speed', 3.0, 10.0, valinit=drone_params['max_speed'])
    slider_energy = Slider(ax3, 'Energy Cons.', 30, 1000, valinit=drone_params['energy_consumption'])
    slider_battery = Slider(ax4, 'Battery (mAh)', 3000, 100000, valinit=drone_params['battery_mAh'])
    slider_shot_energy = Slider(ax5, 'Laser Energy', 0.5, 5.0, valinit=drone_params['laser_shot_energy'])
    slider_low_batt = Slider(ax6, 'Low Batt Frac', 0.1, 0.5, valinit=drone_params['low_battery_threshold_fraction'])
    slider_lock_time = Slider(ax7, 'Lock Time (s)', 0.5, 5.0, valinit=drone_params['lock_time'])
    slider_engagement = Slider(ax8, 'Eng. Range (m)', 1.0, 20.0, valinit=drone_params['engagement_range'])
    slider_num_lasers = Slider(ax9, 'Num Lasers', 1, 10, valinit=drone_params['num_lasers'], valfmt='%d')
    
    ax_apply = fig_drone.add_axes([0.70, 0.94, 0.25, 0.05])
    btn_apply = Button(ax_apply, "Apply")
    
    def apply_drone_settings(event):
        print("Drone Config Apply button clicked.")
        drone_params['max_acc'] = slider_max_acc.val
        drone_params['max_speed'] = slider_max_speed.val
        drone_params['energy_consumption'] = slider_energy.val
        drone_params['battery_mAh'] = slider_battery.val
        drone_params['laser_shot_energy'] = slider_shot_energy.val
        drone_params['low_battery_threshold_fraction'] = slider_low_batt.val
        drone_params['lock_time'] = slider_lock_time.val
        drone_params['engagement_range'] = slider_engagement.val
        drone_params['num_lasers'] = int(slider_num_lasers.val)
        print("Updated drone_params:", drone_params)
        plt.close(fig_drone)
        fig.canvas.draw()
    
    btn_apply.on_clicked(apply_drone_settings)
    plt.show(block=True)

def show_config_report(event):
    config_fig = plt.figure("Configuration Report", figsize=(8, 6))
    config_ax = config_fig.add_subplot(111)
    config_ax.axis('off')
    config_dict = {
        "Insect Params": insect_params, 
        "Drone Params": drone_params, 
        "Simulation Params": sim_params, 
        "Environment": environment
    }
    config_text_str = json.dumps(config_dict, indent=2)
    config_ax.text(0, 1, config_text_str, verticalalignment='top', fontsize=10, family='monospace')
    plt.show(block=False)

def show_help_window(event):
    help_fig = plt.figure("Help", figsize=(8, 6))
    help_ax = help_fig.add_subplot(111)
    help_ax.axis('off')
    help_text = (
        "Help - Parameter Explanations:\n\n"
        "Insect Parameters:\n"
        "  field_width: Width (m) of the simulation field.\n"
        "  field_height: Height (m) of the simulation field.\n"
        "  base_rate: Base insect density.\n"
        "  decay_rate: Reduction in insect probability near field borders.\n\n"
        "Drone Parameters:\n"
        "  max_acc: Max acceleration (m/s²).\n"
        "  max_speed: Max speed (m/s).\n"
        "  energy_consumption: Energy consumption (J/s).\n"
        "  battery_mAh: Battery capacity (mAh).\n"
        "  laser_shot_energy: Energy per shot (J).\n"
        "  low_battery_threshold_fraction: Fraction below which drone returns to charge.\n"
        "  max_speed_when_shooting_kmh: Speed threshold for shooting (km/h).\n"
        "  lock_time: Time to lock onto a target (s).\n"
        "  engagement_range: Engagement distance (m).\n"
        "  num_lasers: Number of lasers operating concurrently.\n\n"
        "Simulation Parameters:\n"
        "  dt: Simulation time step (s).\n"
        "  recharge_time: Duration of recharge (s).\n\n"
        "Environment:\n"
        "  docking_station: Coordinates of the recharge station.\n"
        "  active_start / active_end: Simulation active time window."
    )
    help_ax.text(0, 1, help_text, verticalalignment='top', fontsize=10, family='monospace')
    plt.show(block=False)

# ----------------------- Update Simulation View Callback -----------------------

def update_simulation_view(val):
    """
    Update the drone marker, draw only the flight path flown until this time,
    and display simulation details (time, battery, insects killed, recharges, speed)
    corresponding to the current timeline position.
    """
    if not simulation_history:
        return
    current_time_seconds = time_slider.val * 3600.0
    idx = min(range(len(simulation_history)), key=lambda i: abs(simulation_history[i]['time'] - current_time_seconds))
    current_record = simulation_history[idx]
    # Update drone marker
    pos = current_record['position']
    drone_marker.set_data(pos[0], pos[1])
    # Draw path only up to the current simulation record
    path_up_to = np.array([rec['position'] for rec in simulation_history[:idx+1]])
    if path_up_to.size:
        drone_path_line.set_data(path_up_to[:, 0], path_up_to[:, 1])
    # Update the info text box with simulation details at this time
    info_str = (
        f"Time: {current_record['time']:.2f} s\n"
        f"Battery: {current_record['battery']:.2f} J\n"
        f"Insects killed: {current_record['insects_killed']}\n"
        f"Recharges: {current_record['recharge_count']}\n"
        f"Speed: {current_record['speed']:.2f} km/h"
    )
    text_out.set_text(info_str)
    fig.canvas.draw_idle()

# ----------------------- Run Simulation Callback -----------------------

def run_simulation_callback(event):
    global current_insects, simulation_history, time_slider

    if not current_insects:
        current_insects = generate_insects(
            insect_params['field_width'],
            insect_params['field_height'],
            insect_params['base_rate'],
            insect_params['decay_rate'],
            insect_params['seed']
        )
        ix = [i["position"][0] for i in current_insects]
        iy = [i["position"][1] for i in current_insects]
        insect_scatter.set_offsets(np.column_stack((ix, iy)))
        ax_sim.set_xlim(0, insect_params['field_width'])
        ax_sim.set_ylim(0, insect_params['field_height'])
        insect_count_text.set_text(f"Insect Count: {len(current_insects)}")
        fig.canvas.draw_idle()
    
    BATTERY_VOLTAGE = drone_params['num_cells'] * 3.7
    BATTERY_CAPACITY = (drone_params['battery_mAh'] / 1000) * BATTERY_VOLTAGE * 3600 # In Joules 
    LOW_BATTERY_THRESHOLD = BATTERY_CAPACITY * drone_params['low_battery_threshold_fraction']
    
    drone = Drone(
        position=charging_station,
        max_acc=drone_params['max_acc'],
        max_speed=drone_params['max_speed'],
        battery_capacity=BATTERY_CAPACITY,
        energy_consumption=drone_params['energy_consumption'],
        laser_shot_energy=drone_params['laser_shot_energy'],
        low_battery_threshold=LOW_BATTERY_THRESHOLD
    )
    
    active_start_sec = time_to_seconds(environment["active_start"])
    active_end_sec = time_to_seconds(environment["active_end"])
    active_period = active_end_sec - active_start_sec
    
    simulation_history, log, remaining_insects = simulate(
        drone, current_insects, np.array(environment["docking_station"]),
        insect_params['field_width'], insect_params['field_height'],
        sim_params["dt"],
        drone_params['lock_time'],
        sim_params["recharge_time"],
        drone_params['engagement_range'],
        drone_params['max_speed_when_shooting_kmh'] / 3.6,
        active_period,
        use_fast_tsp=False
    )
    
    if simulation_history:
        t_min = simulation_history[0]['time'] / 3600.0
        t_max = simulation_history[-1]['time'] / 3600.0
        time_slider.ax.clear()
        time_slider = Slider(time_slider.ax, 'Flight Time (h)', t_min, t_max, valinit=t_min)
        time_slider.on_changed(update_simulation_view)
        ax_slider.set_visible(True)
    
    drone_path_arr = np.array(drone.path)
    if drone_path_arr.size:
        drone_path_line.set_data(drone_path_arr[:, 0], drone_path_arr[:, 1])
    fig.canvas.draw_idle()

# ----------------------- Main Button Panel -----------------------

ax_btn_insects = fig.add_axes([0.05, 0.05, 0.13, 0.05])
btn_insects = Button(ax_btn_insects, "Gen Insects")

ax_btn_drone = fig.add_axes([0.19, 0.05, 0.13, 0.05])
btn_drone = Button(ax_btn_drone, "Config Drone")

ax_btn_sim = fig.add_axes([0.33, 0.05, 0.13, 0.05])
btn_sim = Button(ax_btn_sim, "Sim Params")

ax_btn_run = fig.add_axes([0.47, 0.05, 0.13, 0.05])
btn_run = Button(ax_btn_run, "Run Sim")

ax_btn_help = fig.add_axes([0.61, 0.05, 0.13, 0.05])
btn_help = Button(ax_btn_help, "Help")

ax_btn_config = fig.add_axes([0.75, 0.05, 0.13, 0.05])
btn_config = Button(ax_btn_config, "Config Report")

# Bind button callbacks
btn_insects.on_clicked(insects_config_window)
btn_drone.on_clicked(drone_config_window)
btn_sim.on_clicked(sim_params_config_window)
btn_run.on_clicked(run_simulation_callback)
btn_help.on_clicked(show_help_window)
btn_config.on_clicked(show_config_report)

plt.show()

