import numpy as np
from .catalog import KnownCatalog
from .planner import plan_batch_route

class KillerDrone:
    def __init__(self, speed_mps=8.0, fov_m=12.0, engage_range_m=10.0, max_shots_hz=15,
                 lock_delay_s=0.08, max_target_speed=6.0, kill_energy_J=0.5, laser_eff=0.2,
                 replan_every_s=8.0):
        self.p = np.array([0.0, 0.0], dtype=float)
        self.speed = speed_mps
        self.fov = fov_m
        self.engage_range = engage_range_m
        self._dt_min_shot = 1.0 / max_shots_hz
        self.lock_delay_s = lock_delay_s
        self.max_target_speed = max_target_speed
        self.kill_energy_J = kill_energy_J
        self.laser_eff = laser_eff
        self.replan_every_s = replan_every_s
        self._last_shot_s = -1e9
        self._last_plan_s = -1e9
        self.route = []

    def maybe_replan(self, catalog: KnownCatalog, now_s: float):
        if now_s - self._last_plan_s < self.replan_every_s:
            return
        pts = catalog.fresh_points(now_s)
        self.route = plan_batch_route((float(self.p[0]), float(self.p[1])), pts)
        self._last_plan_s = now_s

    def step_move(self, dt: float):
        if not self.route:
            return
        tx, ty = self.route[0]
        to = np.array([tx, ty]) - self.p
        d = np.linalg.norm(to)
        step = self.speed * dt
        if d <= step:
            self.p[:] = [tx, ty]
            self.route.pop(0)
        else:
            self.p += to / max(d, 1e-9) * step

    def try_shoot(self, insects, now_s: float, rng=np.random):
        if now_s - self._last_shot_s < self._dt_min_shot:
            return 0, 0.0, None
        best = None; best_d = 1e9
        for ins in insects:
            if not getattr(ins, "alive", True): continue
            v = ins.v if hasattr(ins, "v") else np.array([ins.vx, ins.vy])
            speed = float(np.linalg.norm(v))
            if speed > self.max_target_speed: continue
            p = ins.p if hasattr(ins, "p") else np.array([ins.x, ins.y])
            d = float(np.linalg.norm(p - self.p))
            if d > self.fov or d > self.engage_range: continue
            if d < best_d: best_d, best = d, ins
        if best is None: return 0, 0.0, None
        p_lock = min(1.0, self._dt_min_shot / max(self.lock_delay_s, 1e-3))
        if rng.random() < p_lock:
            best.alive = False
            best.death_t = now_s
            self._last_shot_s = now_s
            return 1, self.kill_energy_J, best.id
        return 0, 0.0, None
