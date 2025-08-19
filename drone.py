import numpy as np
import math

class Drone:
    def __init__(self, position, max_acc, max_speed, battery_capacity,
                 energy_consumption, laser_shot_energy, low_battery_threshold,
                 mass=1.0, drag_coefficient=0.0, frontal_area=0.0,
                 air_density=1.225):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.max_acc = max_acc
        self.max_speed = max_speed
        self.battery_capacity = battery_capacity   # Joules
        self.battery = battery_capacity            # fully charged initially
        # Base power draw required to hover (W)
        self.energy_consumption = energy_consumption
        self.laser_shot_energy = laser_shot_energy
        self.low_battery_threshold = low_battery_threshold
        self.total_energy_used = 0.0
        self.total_time = 0.0
        self.path = [self.position.copy()]
        self.log = []
        self.total_recharge_time = 0.0
        self.recharge_count = 0
        self.insects_killed_count = 0
        # Physics properties
        self.mass = mass
        self.drag_coefficient = drag_coefficient
        self.frontal_area = frontal_area
        self.air_density = air_density
        self.prev_speed = 0.0

    def update(self, acceleration, dt):
        # Update velocity with acceleration
        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
            speed = self.max_speed
        # Update position
        self.position += self.velocity * dt
        self.path.append(self.position.copy())

        g = 9.81
        # Aerodynamic drag opposite to motion
        drag_force = 0.5 * self.air_density * self.drag_coefficient * self.frontal_area * speed**2
        # Thrust required horizontally must counter acceleration and drag
        horizontal_thrust = self.mass * np.linalg.norm(acceleration) + drag_force
        total_thrust = math.sqrt((self.mass * g)**2 + horizontal_thrust**2)

        thrust_ratio = total_thrust / (self.mass * g)
        # Base hover power scaled by thrust demand
        power = self.energy_consumption * thrust_ratio
        # Power to overcome drag
        power += drag_force * speed
        # Additional power for kinetic energy gain
        delta_ke = 0.5 * self.mass * (speed**2 - self.prev_speed**2)
        if delta_ke > 0:
            power += delta_ke / dt
        self.prev_speed = speed

        energy_used = power * dt
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
