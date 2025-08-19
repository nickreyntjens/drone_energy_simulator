import numpy as np
import math
import pathlib
import sys

# Ensure repository root is on the path for importing drone module
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from drone import Drone

# Reference parameters for a small quadcopter
MASS = 1.0  # kg
drag_coefficient = 1.0
frontal_area = 0.1  # m^2
HOVER_POWER = 150.0  # W

def compute_expected_energy(acc, initial_speed, dt):
    speed = initial_speed + np.linalg.norm(acc) * dt
    rho = 1.225
    drag_force = 0.5 * rho * drag_coefficient * frontal_area * speed**2
    horiz_thrust = MASS * np.linalg.norm(acc) + drag_force
    total_thrust = math.sqrt((MASS * 9.81)**2 + horiz_thrust**2)
    thrust_ratio = total_thrust / (MASS * 9.81)
    power = HOVER_POWER * thrust_ratio + drag_force * speed
    delta_ke = 0.5 * MASS * (speed**2 - initial_speed**2)
    if delta_ke > 0:
        power += delta_ke / dt
    return power * dt, speed

def test_hover_energy():
    d = Drone(position=[0,0], max_acc=5, max_speed=10,
              battery_capacity=1e6, energy_consumption=HOVER_POWER,
              laser_shot_energy=0, low_battery_threshold=0,
              mass=MASS, drag_coefficient=drag_coefficient,
              frontal_area=frontal_area)
    d.update(np.array([0.0,0.0]), 1.0)
    assert math.isclose(d.total_energy_used, HOVER_POWER, rel_tol=1e-3)

def test_acceleration_energy_matches_reference():
    d = Drone(position=[0,0], max_acc=5, max_speed=10,
              battery_capacity=1e6, energy_consumption=HOVER_POWER,
              laser_shot_energy=0, low_battery_threshold=0,
              mass=MASS, drag_coefficient=drag_coefficient,
              frontal_area=frontal_area)
    acc = np.array([1.0,0.0])
    expected_energy, speed = compute_expected_energy(acc, 0.0, 1.0)
    d.update(acc, 1.0)
    assert math.isclose(d.total_energy_used, expected_energy, rel_tol=1e-3)
    assert math.isclose(np.linalg.norm(d.velocity), speed, rel_tol=1e-3)

def test_constant_speed_transit_vs_hover():
    d = Drone(position=[0,0], max_acc=5, max_speed=10,
              battery_capacity=1e6, energy_consumption=HOVER_POWER,
              laser_shot_energy=0, low_battery_threshold=0,
              mass=MASS, drag_coefficient=drag_coefficient,
              frontal_area=frontal_area)
    # set initial velocity to 5 m/s
    d.velocity = np.array([5.0,0.0])
    d.prev_speed = 5.0
    expected_energy, _ = compute_expected_energy(np.array([0.0,0.0]), 5.0, 1.0)
    d.update(np.array([0.0,0.0]),1.0)
    assert math.isclose(d.total_energy_used, expected_energy, rel_tol=1e-3)
    # compare to hover energy to ensure transit uses more power
    assert d.total_energy_used > HOVER_POWER
