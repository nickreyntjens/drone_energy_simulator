import numpy as np
from dataclasses import dataclass
from .catalog import KnownCatalog
from .scout import ScoutDrone
from .killer import KillerDrone

@dataclass
class Insect:
    id: int
    p: np.ndarray
    v: np.ndarray
    species: int = 0
    alive: bool = True
    death_t: float | None = None

class Engine:
    def __init__(self, field=(220.0,160.0), seed=42):
        self.field = np.array(field, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.insects: list[Insect] = []
        self.next_id = 1
        self.catalog = KnownCatalog(ttl_s=120)
        self.scout = ScoutDrone()
        self.killer = KillerDrone()
        self.kpis = {"kills":0, "ipm":0.0, "kills_per_Wh":0.0, "energy_per_kill_J":0.0}

    def spawn_from_edges(self, dt, density_base_per1000m2_min=25.0, wind=0.0):
        perim = 2*(self.field[0]+self.field[1])
        lam = (density_base_per1000m2_min/1000.0) * perim * 3 * (dt/60.0)
        k = self.rng.poisson(max(lam, 0.0))
        for _ in range(k):
            side = int(self.rng.integers(0,4))
            if side==0: p=[0, self.rng.random()*self.field[1]]; v=[1+self.rng.random()*1.5+0.4*wind, self.rng.normal()*0.5]
            elif side==1: p=[self.field[0], self.rng.random()*self.field[1]]; v=[-(1+self.rng.random()*1.5)+0.4*wind, self.rng.normal()*0.5]
            elif side==2: p=[self.rng.random()*self.field[0], 0]; v=[self.rng.normal()*0.5+0.4*wind, 1+self.rng.random()*1.5]
            else: p=[self.rng.random()*self.field[0], self.field[1]]; v=[self.rng.normal()*0.5+0.4*wind, -(1+self.rng.random()*1.5)]
            self.insects.append(Insect(self.next_id, np.array(p,float), np.array(v,float))); self.next_id+=1

    def step(self, dt: float, params: dict):
        self.t += dt
        self.spawn_from_edges(dt, params.get("density_base",25.0), params.get("wind",0.0))
        # move insects
        alive_keep = []
        for ins in self.insects:
            if not ins.alive and ins.death_t and self.t - ins.death_t > 3.0:
                continue
            ins.p += ins.v*dt
            ins.v += self.rng.normal(0,0.2,2)*dt
            alive_keep.append(ins)
        self.insects = alive_keep

        # scout
        self.scout.step(dt, self.field)
        self.scout.maybe_full_field_scan(self.insects, self.t, self.catalog, rng=self.rng)
        self.catalog.expire(self.t)

        # killer
        self.killer.maybe_replan(self.catalog, self.t)
        self.killer.step_move(dt)
        shots, laser_J, killed_id = self.killer.try_shoot(self.insects, self.t, rng=self.rng)
        if shots:
            self.kpis["kills"] += shots
            if killed_id is not None:
                self.catalog.mark_engaged(killed_id)

        return shots
