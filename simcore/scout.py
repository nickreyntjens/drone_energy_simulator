import numpy as np
from typing import Iterable
from .catalog import KnownCatalog

class ScoutDrone:
    def __init__(self, speed_mps=10.0, lane_sep_m=20.0, scan_interval_s=30.0, detection_prob=0.9):
        self.p = np.array([0.0, 0.0], dtype=float)
        self._dir = 1
        self._lane = 0
        self.speed = speed_mps
        self.lane_sep = lane_sep_m
        self.scan_interval_s = scan_interval_s
        self.detection_prob = detection_prob
        self._last_scan = 0.0

    def step(self, dt: float, field_size):
        # lawnmower x-sweep
        self.p[0] += self._dir * self.speed * dt
        if self.p[0] > field_size[0]-10: self.p[0]=field_size[0]-10; self._dir=-1; self._lane+=1
        if self.p[0] < 10:                self.p[0]=10;                self._dir= 1; self._lane+=1
        if self._lane > 0:
            self.p[1] = min(field_size[1]-10, 10 + self._lane*self.lane_sep)
            if self.p[1] >= field_size[1]-10: self._lane = 0

    def maybe_full_field_scan(self, insects: Iterable, now_s: float, catalog: KnownCatalog, rng=np.random):
        if now_s - self._last_scan < self.scan_interval_s:
            return 0
        self._last_scan = now_s
        hits = 0
        for ins in insects:
            if not getattr(ins, "alive", True): 
                continue
            if rng.random() <= self.detection_prob:
                pos = (float(ins.p[0]), float(ins.p[1])) if hasattr(ins, "p") else (float(ins.x), float(ins.y))
                catalog.observe(ins.id, pos, now_s)
                hits += 1
        return hits
