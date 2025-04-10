import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math

# ----------------------- Helper Functions -----------------------

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_effective_area(field_width, field_height, decay_rate, grid_points=100):
    xs = np.linspace(0, field_width, grid_points)
    ys = np.linspace(0, field_height, grid_points)
    xv, yv = np.meshgrid(xs, ys)
    d_border = np.minimum(np.minimum(xv, field_width - xv), np.minimum(yv, field_height - yv))
    f = np.exp(-decay_rate * d_border)
    effective_area = f.mean() * field_width * field_height
    return effective_area

def generate_insects(field_width, field_height, base_rate, decay_rate, seed):
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
            insects.append(np.array([x, y]))
    return insects

def nearest_neighbor_tsp(start, targets):
    route = []
    available = targets.copy()
    current = start
    while available:
        distances = [np.linalg.norm(target - current) for target in available]
        idx = np.argmin(distances)
        route.append(available.pop(idx))
        current = route[-1]
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
        self.battery = battery_capacity
        self.energy_consumption = energy_consumption  # in J/s
        self.laser_shot_energy = laser_shot_energy      # in Joules per shot
        self.low_battery_threshold = low_battery_threshold
        self.total_energy_used = 0.0
        self.total_time = 0.0  # total simulation time in seconds (includes flight & recharging)
        self.path = [self.position.copy()]
        self.log = []  # log simulation events
        # Recharge statistics:
        self.total_recharge_time = 0.0  # seconds spent recharging
        self.recharge_count = 0

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

# ----------------------- Multiday Simulation -----------------------

def simulate_multiday(drone, field_width, field_height, dt, laser_shot_time, recharge_time,
                      engagement_range, max_speed_when_shooting, num_days, decay_rate,
                      base_rate_initial, base_rate_daily, docking_station, RANDOM_SEED):
    simulation_history = []
    insects_killed_count = 0
    current_time = 0.0
    # Generate initial insect population.
    insect_population = generate_insects(field_width, field_height, base_rate_initial, decay_rate, RANDOM_SEED)
    
    # Set drone to docking station at start.
    drone.position = docking_station.copy()
    
    for day in range(num_days):
        day_start = day * 86400.0
        active_start = day_start + 9 * 3600.0  # 9 AM
        active_end = day_start + 18 * 3600.0   # 6 PM
        day_end = (day + 1) * 86400.0
        
        # For subsequent days, add daily insect inflow.
        if day > 0:
            new_insects = generate_insects(field_width, field_height, base_rate_daily, decay_rate, RANDOM_SEED + day)
            insect_population.extend(new_insects)
        
        # Idle until active_start.
        if current_time < active_start:
            while current_time < active_start:
                current_time += dt
                simulation_history.append({
                    'time': current_time,
                    'position': docking_station.copy(),
                    'battery': drone.battery,
                    'energy_used': drone.total_energy_used,
                    'event': 'idle',
                    'recharge_count': drone.recharge_count,
                    'recharge_time': drone.total_recharge_time,
                    'insects_killed': insects_killed_count,
                    'speed': 0.0,
                    'acceleration': 0.0,
                    'day': day + 1
                })
            drone.position = docking_station.copy()
            drone.battery = drone.battery_capacity
        
        current_time = active_start
        drone.total_time = active_start  # reset simulation time for active period
        
        # ACTIVE PERIOD: 9 AM to 6 PM.
        while current_time < active_end:
            if not insect_population:
                current_time += dt
                simulation_history.append({
                    'time': current_time,
                    'position': drone.position.copy(),
                    'battery': drone.battery,
                    'energy_used': drone.total_energy_used,
                    'event': 'no_insects',
                    'recharge_count': drone.recharge_count,
                    'recharge_time': drone.total_recharge_time,
                    'insects_killed': insects_killed_count,
                    'speed': 0.0,
                    'acceleration': 0.0,
                    'day': day + 1
                })
                continue

            # Compute route if needed.
            route = nearest_neighbor_tsp(drone.position, insect_population) if not drone.log or not route else route

            if not route:
                break

            target = route[0]

            # If battery low, set target to docking station.
            if drone.battery < drone.low_battery_threshold:
                target = docking_station

            reached = False
            while not reached and current_time < active_end:
                acc = drone.apply_acceleration_towards(target, dt)
                current_speed = np.linalg.norm(drone.velocity) * 3.6  # km/h
                current_acc = np.linalg.norm(acc)
                drone.update(acc, dt)
                current_time = drone.total_time
                simulation_history.append({
                    'time': current_time,
                    'position': drone.position.copy(),
                    'battery': drone.battery,
                    'energy_used': drone.total_energy_used,
                    'event': None,
                    'recharge_count': drone.recharge_count,
                    'recharge_time': drone.total_recharge_time,
                    'insects_killed': insects_killed_count,
                    'speed': current_speed,
                    'acceleration': current_acc,
                    'day': day + 1
                })
                drone.position[0] = min(max(drone.position[0], 0), field_width)
                drone.position[1] = min(max(drone.position[1], 0), field_height)

                # If going to docking station.
                if np.array_equal(target, docking_station):
                    if distance(drone.position, docking_station) <= 1.0:
                        reached = True
                        drone.log.append(f"Docked for recharge at t={drone.total_time:.2f}s (Day {day+1})")
                        recharge_time_elapsed = 0.0
                        while recharge_time_elapsed < recharge_time and current_time < active_end:
                            simulation_history.append({
                                'time': drone.total_time,
                                'position': drone.position.copy(),
                                'battery': drone.battery,
                                'energy_used': drone.total_energy_used,
                                'event': 'recharging',
                                'recharge_count': drone.recharge_count,
                                'recharge_time': drone.total_recharge_time,
                                'insects_killed': insects_killed_count,
                                'speed': 0.0,
                                'acceleration': 0.0,
                                'day': day + 1
                            })
                            drone.total_time += dt
                            recharge_time_elapsed += dt
                            current_time = drone.total_time
                        drone.total_recharge_time += recharge_time_elapsed
                        drone.recharge_count += 1
                        drone.battery = drone.battery_capacity
                        drone.log.append(f"Recharged at t={drone.total_time:.2f}s (Day {day+1})")
                        # NEW: If after recharging, current_time is already past active_end, break out.
                        if drone.total_time >= active_end:
                            reached = True
                            break
                        break
                else:
                    if distance(drone.position, target) <= engagement_range:
                        # Wait until speed is below max_speed_when_shooting.
                        while np.linalg.norm(drone.velocity) > max_speed_when_shooting:
                            deceleration = -drone.velocity
                            drone.update(deceleration, dt)
                            current_time = drone.total_time
                            simulation_history.append({
                                'time': current_time,
                                'position': drone.position.copy(),
                                'battery': drone.battery,
                                'energy_used': drone.total_energy_used,
                                'event': 'waiting_for_deceleration',
                                'recharge_count': drone.recharge_count,
                                'recharge_time': drone.total_recharge_time,
                                'insects_killed': insects_killed_count,
                                'speed': np.linalg.norm(drone.velocity) * 3.6,
                                'acceleration': np.linalg.norm(deceleration),
                                'day': day + 1
                            })
                        reached = True
                        drone.log.append(f"Insect at {target} shot at t={drone.total_time:.2f}s (Day {day+1})")
                        simulation_history.append({
                            'time': drone.total_time,
                            'position': drone.position.copy(),
                            'battery': drone.battery,
                            'energy_used': drone.total_energy_used,
                            'event': 'insect_shot',
                            'recharge_count': drone.recharge_count,
                            'recharge_time': drone.total_recharge_time,
                            'insects_killed': insects_killed_count,
                            'speed': np.linalg.norm(drone.velocity) * 3.6,
                            'acceleration': current_acc,
                            'day': day + 1
                        })
                        insects_killed_count += 1
                        shot_delay = 0.0
                        while shot_delay < laser_shot_time and current_time < active_end:
                            drone.total_time += dt
                            shot_delay += dt
                            current_time = drone.total_time
                            simulation_history.append({
                                'time': current_time,
                                'position': drone.position.copy(),
                                'battery': drone.battery,
                                'energy_used': drone.total_energy_used,
                                'event': 'shooting',
                                'recharge_count': drone.recharge_count,
                                'recharge_time': drone.total_recharge_time,
                                'insects_killed': insects_killed_count,
                                'speed': np.linalg.norm(drone.velocity) * 3.6,
                                'acceleration': 0.0,
                                'day': day + 1
                            })
                        drone.battery -= drone.laser_shot_energy
                        drone.total_energy_used += drone.laser_shot_energy
                        for idx, insect in enumerate(insect_population):
                            if np.allclose(insect, target, atol=engagement_range):
                                del insect_population[idx]
                                break
                        route.pop(0)
                        break
                if drone.battery <= 0:
                    print("Drone battery depleted unexpectedly!")
                    return simulation_history, drone.log
        # End of active period: force drone to dock.
        drone.position = docking_station.copy()
        drone.battery = drone.battery_capacity
        while current_time < day_end:
            current_time += dt
            simulation_history.append({
                'time': current_time,
                'position': docking_station.copy(),
                'battery': drone.battery,
                'energy_used': drone.total_energy_used,
                'event': 'idle',
                'recharge_count': drone.recharge_count,
                'recharge_time': drone.total_recharge_time,
                'insects_killed': insects_killed_count,
                'speed': 0.0,
                'acceleration': 0.0,
                'day': day + 1
            })
            drone.position = docking_station.copy()

    return simulation_history, drone.log

# ----------------------- Visualization -----------------------

def visualize_simulation(simulation_history, field_width, field_height, extra_info):
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

    docking_station = extra_info["docking_station"]
    charging_station_marker, = ax.plot(docking_station[0], docking_station[1],
                                       marker='s', markersize=10, color='green', label='Docking Station')
    drone_path_line, = ax.plot([], [], 'b-', label='Drone Path')
    drone_marker, = ax.plot([], [], 'bo', markersize=8, label='Drone')

    times = [entry['time'] for entry in simulation_history]
    t_min = times[0] / 3600.0
    t_max = times[-1] / 3600.0
    ax_slider = plt.axes([0.1, 0.1, 0.55, 0.05])
    time_slider = Slider(ax_slider, 'Time (h)', t_min, t_max, valinit=t_min)

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
        current_day = state.get('day', 1)
        idle_per_day = 15 * 3600.0  # 15 hours idle per day
        flight_time_sec = state['time'] - state['recharge_time'] - ((current_day - 1) * idle_per_day)
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
            f"Day: {current_day}\n"
            f"Flight Time: {flight_time_hours:.2f} h\n"
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

if __name__ == '__main__':
    # Common parameters
    FIELD_WIDTH = 1000.0
    FIELD_HEIGHT = 1000.0
    DECAY_RATE_DEFAULT = 0.05

    MAX_ACCEL = 1.0
    MAX_SPEED = 5.0
    DRONE_ENERGY_CONSUMPTION = 300.0  # 300 W
    BATTERY_MAH = 6700.0
    NUM_CELLS = 3
    BATTERY_VOLTAGE = NUM_CELLS * 3.7
    BATTERY_CAPACITY = (BATTERY_MAH / 1000) * BATTERY_VOLTAGE * 3600
    LOW_BATTERY_THRESHOLD = BATTERY_CAPACITY / 3.0
    RECHARGE_TIME = 1800.0  # 30 minutes
    LASER_SHOT_TIME = 1.0
    LASER_SHOT_ENERGY = 1.0
    ENGAGEMENT_RANGE = 2.0
    # New parameter: max speed when shooting (set in km/h, converted to m/s)
    MAX_SPEED_WHEN_SHOOTING_DEFAULT = 3.0 / 3.6

    CHARGING_STATION = np.array([500.0, 500.0])

    NUM_DAYS_DEFAULT = 5
    INITIAL_INSECT_COUNT_DEFAULT = 5000
    DAILY_INFLOW_COUNT_DEFAULT = 1000
    RANDOM_SEED = 42

    SOLAR_PANEL_ENERGY_PER_HOUR = 720000.0

    BATTERY_COST = 200.0
    BATTERY_MAX_CYCLES = 500.0

    # Mode selection
    if '-auto' in sys.argv:
        num_days = NUM_DAYS_DEFAULT
        initial_insect_count = INITIAL_INSECT_COUNT_DEFAULT
        daily_inflow_count = DAILY_INFLOW_COUNT_DEFAULT
        max_speed_when_shooting = MAX_SPEED_WHEN_SHOOTING_DEFAULT
    else:
        num_days = int(input("Enter number of days (e.g., 20): ") or NUM_DAYS_DEFAULT)
        initial_insect_count = int(input("Enter initial insect count (e.g., 1000): ") or INITIAL_INSECT_COUNT_DEFAULT)
        daily_inflow_count = int(input("Enter daily insect inflow count (e.g., 200): ") or DAILY_INFLOW_COUNT_DEFAULT)
        max_speed_shooting_kmh = float(input("Enter maximum speed when shooting (km/h, e.g., 3): ") or 3)
        max_speed_when_shooting = max_speed_shooting_kmh / 3.6

    effective_area = compute_effective_area(FIELD_WIDTH, FIELD_HEIGHT, DECAY_RATE_DEFAULT, grid_points=100)
    base_rate_initial = initial_insect_count / effective_area
    base_rate_daily = daily_inflow_count / effective_area

    drone = Drone(position=CHARGING_STATION.copy(),
                  max_acc=MAX_ACCEL,
                  max_speed=MAX_SPEED,
                  battery_capacity=BATTERY_CAPACITY,
                  energy_consumption=DRONE_ENERGY_CONSUMPTION,
                  laser_shot_energy=LASER_SHOT_ENERGY,
                  low_battery_threshold=LOW_BATTERY_THRESHOLD)

    simulation_history, log = simulate_multiday(
        drone, FIELD_WIDTH, FIELD_HEIGHT, dt=1, laser_shot_time=LASER_SHOT_TIME,
        recharge_time=RECHARGE_TIME, engagement_range=ENGAGEMENT_RANGE,
        max_speed_when_shooting=max_speed_when_shooting, num_days=num_days,
        decay_rate=DECAY_RATE_DEFAULT, base_rate_initial=base_rate_initial,
        base_rate_daily=base_rate_daily, docking_station=CHARGING_STATION,
        RANDOM_SEED=RANDOM_SEED
    )
    battery_depreciation_cost = (BATTERY_COST / BATTERY_MAX_CYCLES) * drone.recharge_count

    print("\n--- Simulation Log ---")
    for entry in log:
        print(entry)
    total_simulation_hours = drone.total_time / 3600.0
    print(f"\nTotal simulation time: {total_simulation_hours:.2f} hours")
    print(f"Total energy used: {drone.total_energy_used:.2f} Joules")
    print(f"Total recharge time: {drone.total_recharge_time/3600:.2f} hours")
    print(f"Number of recharges: {drone.recharge_count}")
    print(f"Battery Depreciation: ${battery_depreciation_cost:.2f} USD")

    extra_info = {
        "battery_cost": BATTERY_COST,
        "battery_max_cycles": BATTERY_MAX_CYCLES,
        "solar_panel_energy_per_hour": SOLAR_PANEL_ENERGY_PER_HOUR,
        "docking_station": CHARGING_STATION
    }
    visualize_simulation(simulation_history, FIELD_WIDTH, FIELD_HEIGHT, extra_info)

