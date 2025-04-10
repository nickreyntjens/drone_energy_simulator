import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math

# ----------------------- Helper Functions -----------------------

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def generate_insects(field_width, field_height, base_rate, decay_rate, seed):
    """
    Generate insect positions over the field.
    The local insect density is given by: rate = base_rate * exp(-decay_rate * d),
    where d is the distance to the closest border.
    A number of candidate insects is sampled from a Poisson distribution with parameter (base_rate * area),
    and each candidate is accepted with probability exp(-decay_rate * d).
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
            insects.append(np.array([x, y]))
    return insects

def nearest_neighbor_tsp(start, targets):
    """
    Compute a TSP route using a simple nearest neighbor heuristic.
    Starting at 'start', repeatedly select the nearest remaining target.
    """
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
        self.battery = battery_capacity           # starts fully charged
        self.energy_consumption = energy_consumption  # in J/s
        self.laser_shot_energy = laser_shot_energy      # in Joules per shot
        self.low_battery_threshold = low_battery_threshold
        self.total_energy_used = 0.0  # cumulative energy usage (J)
        self.total_time = 0.0         # total simulation time (seconds; includes flight and recharging)
        self.path = [self.position.copy()]
        self.log = []               # log simulation events
        # Recharge statistics:
        self.total_recharge_time = 0.0  # seconds spent recharging
        self.recharge_count = 0         # count of recharges

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
             dt, laser_shot_time, recharge_time, engagement_range, max_speed_when_shooting):
    """
    Simulate the drone mission:
      - Compute a TSP route among remaining insect targets.
      - Divert to the charging station when battery is low.
      - Engage (shoot) an insect when within engagement range.
      When within engagement range, if the drone's speed is above max_speed_when_shooting,
      it will decelerate until its speed is low enough.
    At each time step, record state including:
      - Current recharge stats,
      - Number of insects killed,
      - Drone speed (km/h) and acceleration (m/s²).
    """
    DOCKING_TOLERANCE = 1.0  # m

    simulation_history = []  # List of state records
    remaining_insects = insects.copy()
    route = []
    insects_killed_count = 0  # Number of insects killed

    print("Starting simulation...")

    while remaining_insects:
        if not route:
            route = nearest_neighbor_tsp(drone.position, remaining_insects)
            drone.log.append(f"Recalculated route with {len(route)} targets at t={drone.total_time:.2f}s")
            print(f"Recalculating route: {len(route)} targets remaining.")

        target = route[0]

        # If battery is low, override target with charging station.
        if drone.battery < drone.low_battery_threshold:
            target = charging_station
            drone.log.append(f"Low battery at t={drone.total_time:.2f}s, returning to charging station")
            print(f"Low battery ({drone.battery:.2f} J). Returning to charging station.")
            route = []  # Force route recomputation after recharge

        reached = False
        while not reached:
            acc = drone.apply_acceleration_towards(target, dt)
            current_speed = np.linalg.norm(drone.velocity) * 3.6  # convert m/s to km/h
            current_acc = np.linalg.norm(acc)
            drone.update(acc, dt)
            simulation_history.append({
                'time': drone.total_time,
                'position': drone.position.copy(),
                'battery': drone.battery,
                'energy_used': drone.total_energy_used,
                'event': None,
                'recharge_count': drone.recharge_count,
                'recharge_time': drone.total_recharge_time,
                'insects_killed': insects_killed_count,
                'speed': current_speed,
                'acceleration': current_acc
            })

            # Constrain drone position within field boundaries.
            drone.position[0] = min(max(drone.position[0], 0), field_width)
            drone.position[1] = min(max(drone.position[1], 0), field_height)

            # If heading to charging station:
            if np.array_equal(target, charging_station):
                if distance(drone.position, charging_station) <= DOCKING_TOLERANCE:
                    reached = True
                    drone.log.append(f"Docked for recharge at t={drone.total_time:.2f}s")
                    print("Docked at charging station. Recharging...")
                    recharge_time_elapsed = 0.0
                    while recharge_time_elapsed < recharge_time:
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
                # If drone is within engagement range to shoot insect:
                if distance(drone.position, target) <= engagement_range:
                    # Wait until drone's speed is below max_speed_when_shooting.
                    while np.linalg.norm(drone.velocity) > max_speed_when_shooting:
                        deceleration = -drone.velocity  # simple full deceleration
                        drone.update(deceleration, dt)
                        simulation_history.append({
                            'time': drone.total_time,
                            'position': drone.position.copy(),
                            'battery': drone.battery,
                            'energy_used': drone.total_energy_used,
                            'event': 'waiting_for_deceleration',
                            'recharge_count': drone.recharge_count,
                            'recharge_time': drone.total_recharge_time,
                            'insects_killed': insects_killed_count,
                            'speed': np.linalg.norm(drone.velocity)*3.6,
                            'acceleration': np.linalg.norm(deceleration)
                        })
                    reached = True
                    drone.log.append(f"Insect at {target} shot at t={drone.total_time:.2f}s")
                    print(f"Insect at {target} engaged at t={drone.total_time:.2f}s.")
                    simulation_history.append({
                        'time': drone.total_time,
                        'position': drone.position.copy(),
                        'battery': drone.battery,
                        'energy_used': drone.total_energy_used,
                        'event': 'insect_shot',
                        'recharge_count': drone.recharge_count,
                        'recharge_time': drone.total_recharge_time,
                        'insects_killed': insects_killed_count,
                        'speed': np.linalg.norm(drone.velocity)*3.6,
                        'acceleration': current_acc
                    })
                    insects_killed_count += 1
                    shot_delay = 0.0
                    while shot_delay < laser_shot_time:
                        drone.total_time += dt
                        shot_delay += dt
                        simulation_history.append({
                            'time': drone.total_time,
                            'position': drone.position.copy(),
                            'battery': drone.battery,
                            'energy_used': drone.total_energy_used,
                            'event': 'shooting',
                            'recharge_count': drone.recharge_count,
                            'recharge_time': drone.total_recharge_time,
                            'insects_killed': insects_killed_count,
                            'speed': np.linalg.norm(drone.velocity)*3.6,
                            'acceleration': 0.0
                        })
                    drone.battery -= drone.laser_shot_energy
                    drone.total_energy_used += drone.laser_shot_energy
                    for idx, insect in enumerate(remaining_insects):
                        if np.allclose(insect, target, atol=engagement_range):
                            del remaining_insects[idx]
                            break
                    route.pop(0)
                    break

            if drone.battery <= 0:
                print("Drone battery depleted unexpectedly!")
                return simulation_history, drone.log

    print("Simulation complete.")
    return simulation_history, drone.log

# ----------------------- Visualization -----------------------

def visualize_simulation(simulation_history, insects, charging_station, drone_path,
                           field_width, field_height, extra_info):
    """
    Create a 2D plot with a time slider and an output panel.
    The output panel displays:
      - Total Flight Time (in hours; simulation time minus recharge time)
      - Energy Used (J) and equivalent "1 m² solar panel hours"
      - Total Recharge Time (in hours)
      - Number of Recharges
      - Battery Depreciation Cost (in USD)
      - Number of Insects Killed
      - Drone Speed (km/h)
      - Drone Acceleration (m/s²)
    The slider's domain is in flight hours.
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

    # Plot the charging station and insect positions.
    charging_station_marker, = ax.plot(charging_station[0], charging_station[1],
                                       marker='s', markersize=10, color='green', label='Charging Station')
    insect_x = [insect[0] for insect in insects]
    insect_y = [insect[1] for insect in insects]
    insects_scatter = ax.scatter(insect_x, insect_y, color='red', label='Insects')
    drone_path_line, = ax.plot([], [], 'b-', label='Drone Path')
    drone_marker, = ax.plot([], [], 'bo', markersize=8, label='Drone')

    # Create a slider for time scrubbing (domain in flight hours).
    times = [entry['time'] for entry in simulation_history]
    t_min = times[0] / 3600.0
    t_max = times[-1] / 3600.0
    ax_slider = plt.axes([0.1, 0.1, 0.55, 0.05])
    time_slider = Slider(ax_slider, 'Flight Time (h)', t_min, t_max, valinit=t_min)

    # Create an output text panel.
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
        # Compute total flight time (simulation time minus recharge time), in hours.
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
            f"Equivalent Solar Panel Hours: {solar_panel_hours:.2f} h\n"
            f"Total Recharge Time: {recharge_time_hours:.2f} h\n"
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
    if '-auto' in sys.argv:
        print("Auto mode: using default simulation parameters.")
        FIELD_WIDTH = 1000.0
        FIELD_HEIGHT = 1000.0

        BASE_RATE = 0.05
        DECAY_RATE = 0.01
        RANDOM_SEED = 42

        MAX_ACCEL = 1.0
        MAX_SPEED = 5.0
        DRONE_ENERGY_CONSUMPTION = 300.0  # 300 W (300 J/s)

        # Battery: 6700 mAh with a given number of LiPo cells.
        BATTERY_MAH = 6700.0
        NUM_CELLS = 3  # Typical range: 2s to 6s; default 3s.
        BATTERY_VOLTAGE = NUM_CELLS * 3.7  # Nominal voltage in Volts
        BATTERY_CAPACITY = (BATTERY_MAH / 1000) * BATTERY_VOLTAGE * 3600  # in Joules
        LOW_BATTERY_THRESHOLD = BATTERY_CAPACITY / 3.0
        RECHARGE_TIME = 1800.0  # 1800 s = 30 minutes per recharge

        LASER_SHOT_TIME = 1.0
        LASER_SHOT_ENERGY = 1.0

        ENGAGEMENT_RANGE = 2.0

        # New parameter: max speed when shooting.
        # Default is now 3 km/h, converted to m/s:
        MAX_SPEED_WHEN_SHOOTING = 3.0 / 3.6

        CHARGING_STATION_X = 500.0
        CHARGING_STATION_Y = 500.0

        BATTERY_COST = 200.0
        BATTERY_MAX_CYCLES = 500.0

        SOLAR_PANEL_ENERGY_PER_HOUR = 720000.0  # Joules per hour (~200 W/m²)
    else:
        print("----- Drone Simulation Setup -----")
        FIELD_WIDTH = float(input("Enter field width in meters (e.g., 1000): ") or 1000)
        FIELD_HEIGHT = float(input("Enter field height in meters (e.g., 1000): ") or 1000)

        BASE_RATE = float(input("Enter base insect rate (insects/m² near border, e.g., 0.05): ") or 0.05)
        DECAY_RATE = float(input("Enter decay rate for insect density (e.g., 0.05): ") or 0.05)
        RANDOM_SEED = int(input("Enter random seed (e.g., 42): ") or 42)

        MAX_ACCEL = float(input("Enter drone maximum acceleration (m/s², e.g., 1.0): ") or 1.0)
        MAX_SPEED = float(input("Enter drone maximum speed (m/s, e.g., 5.0): ") or 5.0)
        DRONE_ENERGY_CONSUMPTION = float(input("Enter drone energy consumption per second (W, e.g., 300): ") or 300)
        
        BATTERY_MAH = float(input("Enter battery capacity in mAh (e.g., 6700): ") or 6700)
        NUM_CELLS = int(input("Enter number of LiPo cells (e.g., 3): ") or 3)
        BATTERY_VOLTAGE = NUM_CELLS * 3.7
        BATTERY_CAPACITY = (BATTERY_MAH / 1000) * BATTERY_VOLTAGE * 3600  # in Joules
        
        LOW_BATTERY_THRESHOLD = BATTERY_CAPACITY / 3.0
        RECHARGE_TIME = float(input("Enter recharge time in seconds (e.g., 1800 for 30 min): ") or 1800)

        LASER_SHOT_TIME = float(input("Enter laser shot time in seconds (e.g., 1): ") or 1)
        LASER_SHOT_ENERGY = float(input("Enter laser shot energy in Joules (between 0.1 and 5, e.g., 1): ") or 1)

        ENGAGEMENT_RANGE = float(input("Enter insect engagement range in meters (e.g., 2): ") or 2)

        # New parameter: max speed when shooting. Enter in km/h (will be converted to m/s).
        max_speed_shooting_kmh = float(input("Enter maximum speed when shooting (km/h, e.g., 3): ") or 3)
        MAX_SPEED_WHEN_SHOOTING = max_speed_shooting_kmh / 3.6

        CHARGING_STATION_X = float(input("Enter charging station X coordinate (e.g., 500): ") or 500)
        CHARGING_STATION_Y = float(input("Enter charging station Y coordinate (e.g., 500): ") or 500)

        BATTERY_COST = float(input("Enter battery cost (e.g., 200): ") or 200)
        BATTERY_MAX_CYCLES = float(input("Enter battery maximum number of cycles (e.g., 500): ") or 500)

        SOLAR_PANEL_ENERGY_PER_HOUR = float(input("Enter solar panel energy per hour in J (e.g., 720000): ") or 720000)

    CHARGING_STATION = np.array([CHARGING_STATION_X, CHARGING_STATION_Y])
    DT = 0.1  # Time step in seconds

    # Generate insect field.
    insects = generate_insects(FIELD_WIDTH, FIELD_HEIGHT, BASE_RATE, DECAY_RATE, RANDOM_SEED)
    print(f"Generated {len(insects)} insects.")
    if len(insects) == 0:
        print("No insects generated. Please try different insect parameters.")
        exit()

    # Instantiate the drone (starting at the charging station).
    drone = Drone(position=CHARGING_STATION,
                  max_acc=MAX_ACCEL,
                  max_speed=MAX_SPEED,
                  battery_capacity=BATTERY_CAPACITY,
                  energy_consumption=DRONE_ENERGY_CONSUMPTION,
                  laser_shot_energy=LASER_SHOT_ENERGY,
                  low_battery_threshold=LOW_BATTERY_THRESHOLD)

    simulation_history, log = simulate(drone, insects, CHARGING_STATION,
                                       FIELD_WIDTH, FIELD_HEIGHT, DT,
                                       LASER_SHOT_TIME, RECHARGE_TIME, ENGAGEMENT_RANGE,
                                       MAX_SPEED_WHEN_SHOOTING)
    battery_depreciation_cost = (BATTERY_COST / BATTERY_MAX_CYCLES) * drone.recharge_count

    print("\n--- Simulation Log ---")
    for entry in log:
        print(entry)
    print(f"\nTotal simulation time: {drone.total_time/3600:.2f} hours")
    print(f"Total energy used: {drone.total_energy_used:.2f} Joules")
    print(f"Total recharge time: {drone.total_recharge_time/3600:.2f} hours")
    print(f"Number of recharges: {drone.recharge_count}")
    print(f"Battery Depreciation: ${battery_depreciation_cost:.2f} USD")

    extra_info = {
        "battery_cost": BATTERY_COST,
        "battery_max_cycles": BATTERY_MAX_CYCLES,
        "solar_panel_energy_per_hour": SOLAR_PANEL_ENERGY_PER_HOUR
    }
    visualize_simulation(simulation_history, insects, CHARGING_STATION, drone.path,
                           FIELD_WIDTH, FIELD_HEIGHT, extra_info)

