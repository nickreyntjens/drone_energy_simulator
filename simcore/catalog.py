from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class KnownTarget:
    pos: Tuple[float, float]
    last_seen_s: float
    engaged: bool = False

class KnownCatalog:
    def __init__(self, ttl_s: float = 120.0):
        self.ttl_s = ttl_s
        self._items: Dict[int, KnownTarget] = {}

    def observe(self, insect_id: int, pos, now_s: float):
        kt = self._items.get(insect_id)
        if kt:
            kt.pos = pos
            kt.last_seen_s = now_s
        else:
            self._items[insect_id] = KnownTarget(pos, now_s)

    def expire(self, now_s: float):
        drop = [i for i, kt in self._items.items() if now_s - kt.last_seen_s > self.ttl_s]
        for i in drop:
            self._items.pop(i, None)

    def mark_engaged(self, insect_id: int):
        if insect_id in self._items:
            self._items[insect_id].engaged = True

    def fresh_points(self, now_s: float):
        return [kt.pos for kt in self._items.values()
                if now_s - kt.last_seen_s <= self.ttl_s and not kt.engaged]

    def __len__(self):
        return len(self._items)
