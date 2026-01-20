"""Headless runner that saves frames for CI artifacts."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simcore.engine import Engine

os.makedirs("artifacts", exist_ok=True)

eng = Engine(seed=42)
params = {"density_base": 25.0, "wind": 2.0}
for i in range(300):  # ~10s at 0.033
    eng.step(0.033, params)
    if i % 15 == 0:  # ~20 frames
        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        ax.set_xlim(0, eng.field[0]); ax.set_ylim(0, eng.field[1])
        ax.set_title(f"t={eng.t:.1f}s  known={len(eng.catalog)}  kills={eng.kpis['kills']}")
        xs_alive = [ins.p[0] for ins in eng.insects if ins.alive]
        ys_alive = [ins.p[1] for ins in eng.insects if ins.alive]
        xs_dead  = [ins.p[0] for ins in eng.insects if not ins.alive]
        ys_dead  = [ins.p[1] for ins in eng.insects if not ins.alive]
        ax.plot(xs_alive, ys_alive, ".", ms=2, color="tab:green")
        ax.plot(xs_dead,  ys_dead,  ".", ms=2, color="tab:red")
        ax.plot(eng.scout.p[0], eng.scout.p[1], "s", ms=6, color="orange", label="scout")
        ax.plot(eng.killer.p[0], eng.killer.p[1], "^", ms=6, color="cyan", label="killer")
        if eng.killer.route:
            rx, ry = zip(*([ (eng.killer.p[0],eng.killer.p[1]) ] + eng.killer.route))
            ax.plot(rx, ry, "-", lw=1.5, color="deepskyblue", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)
        fig.savefig(f"artifacts/frame_{i:04d}.png", bbox_inches="tight")
        plt.close(fig)

print({"kills": eng.kpis["kills"], "known": len(eng.catalog)})
