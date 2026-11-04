"""Plot the three ALL-16 trajectories (book centre XY) with the caddy regions."""
import json
import sys
from pathlib import Path

K1 = Path("/home/lzx/cl_binding/.server/k1_screen_20261101_01")
sys.path.insert(0, str(K1 / "code"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import k1_main as KM

VID = Path("/home/lzx/cl_binding/video/2026-11-04_K7_整层全换_三条指令")
LABEL = {"A": "instruction LEFT", "B": "instruction FRONT", "N": "instruction RIGHT"}


def main():
    ctx = KM.Ctx("scene2")
    g = ctx.geometry
    data = json.loads((VID / "paths.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)
    for ax, key in zip(axes, ("A", "B", "N")):
        obj = np.asarray(data["paths"][key]["obj"])
        for name, site in g.items():
            rot = np.asarray(site["rotation"])
            half = np.asarray(site["half_size"])
            ex = [sum(abs(rot[i][j]) * half[j] for j in range(3)) for i in (0, 1)]
            ax.add_patch(Rectangle((site["centre"][0] - ex[0], site["centre"][1] - ex[1]),
                                   2 * ex[0], 2 * ex[1], fill=False, lw=1.0, ec="0.4"))
            ax.text(site["centre"][0], site["centre"][1], name, ha="center", va="center", fontsize=7, color="0.4")
        t = np.arange(len(obj))
        sc = ax.scatter(obj[:, 0], obj[:, 1], c=t, cmap="viridis", s=6)
        ax.plot(obj[0, 0], obj[0, 1], "o", ms=8, mfc="none", mec="k")
        ax.plot(obj[-1, 0], obj[-1, 1], "s", ms=7, mfc="none", mec="r")
        first = data["instruction"][key]["first"]
        succ = data["instruction"][key]["success"]
        ax.set_title("%s\nfirst target: %s  |  left@%s front@%s" % (
            LABEL[key], (first["targets"] if first else "none"), succ["left"], succ["front"]), fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(sc, ax=axes, label="step")
    fig.suptitle("FT4 with all 16 expert layers from R0 (whole-layer swap), init 4: book-centre path", fontsize=11)
    fig.savefig(VID / "paths_three_instructions.png", dpi=150, bbox_inches="tight")
    print("saved", (VID / "paths_three_instructions.png"))


if __name__ == "__main__":
    main()
