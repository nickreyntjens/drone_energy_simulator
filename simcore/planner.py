import numpy as np
from typing import Iterable, Tuple, List

def plan_batch_route(from_xy: Tuple[float,float], points: Iterable[Tuple[float,float]], cell_m=20.0, horizon=40) -> List[Tuple[float,float]]:
    """Pick densest grid cell (+ neighbors) then greedy NN through that batch."""
    pts = list(points)
    if not pts:
        return []
    # density grid
    cells = {}
    for (x, y) in pts:
        key = (int(x//cell_m), int(y//cell_m))
        cells[key] = cells.get(key, 0) + 1
    (cx, cy), _ = max(cells.items(), key=lambda kv: kv[1])
    neigh = {(cx,cy),(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)}
    batch = [(x,y) for (x,y) in pts if (int(x//cell_m), int(y//cell_m)) in neigh]

    cur = np.array(from_xy, dtype=float)
    remaining = [np.array(p, dtype=float) for p in batch]
    route = []
    for _ in range(min(horizon, len(remaining))):
        dists = [np.linalg.norm(p - cur) for p in remaining]
        i = int(np.argmin(dists))
        nxt = remaining.pop(i)
        route.append((float(nxt[0]), float(nxt[1])))
        cur = nxt
        if not remaining: break
    return route
