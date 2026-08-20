"""Plot |E| focal profile from Sim4Life MQS output (figure8 coil + 3-shell head).

Reads the staggered-grid (Yee) EM E field snapshots from an Output.h5 file,
interpolates components to voxel centers, computes |E|, and produces a
two-panel figure: (a) |E| heatmap on a vertical slice through the focus,
(b) |E| decay along the depth (z) axis through the focus.

Dependencies: h5py, numpy, matplotlib only (no research_agent imports).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELD_GROUP = "f168b80e-4537-414d-ba7c-9683e5bf1c34"
# The field group itself stores no axes; the grid axes (axis_x/y/z with
# shapes 121/112/113, matching the staggered field) live under this mesh.
MESH_GROUP = "ec91132d-e31f-4d4f-bc02-a5aeb7465744"
DEFAULT_H5 = (
    "artifacts/b4_head_figure8.smash_Results/"
    "6a2d4140-2880-48c5-8390-38fef2fcd6c6_Output.h5"
)
OUT_PNG = Path("docs/images/b4_efield_focal_profile.png")


def load_abs_components(h5: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load |Ex|, |Ey|, |Ez| from staggered snapshots (last dim = real/imag)."""
    base = (
        f"FieldGroups/{FIELD_GROUP}/AllFields/EM E(x,y,z,f0)/_Object/Snapshots/0"
    )
    comps = []
    for i in range(3):
        raw = np.asarray(h5[f"{base}/comp{i}"])
        comps.append(np.sqrt(raw[..., 0] ** 2 + raw[..., 1] ** 2))
    return tuple(comps)  # type: ignore[return-value]


def _avg2(a: np.ndarray, axis: int) -> np.ndarray:
    """NaN-aware average of adjacent cells along `axis` (NaN = outside grid)."""
    lo = np.take(a, range(a.shape[axis] - 1), axis=axis)
    hi = np.take(a, range(1, a.shape[axis]), axis=axis)
    both_nan = np.isnan(lo) & np.isnan(hi)
    out = np.nanmean(np.stack([lo, hi]), axis=0)
    out[both_nan] = np.nan
    return out


def to_voxel_centers(
    ex: np.ndarray, ey: np.ndarray, ez: np.ndarray
) -> np.ndarray:
    """Average staggered components onto the (120,111,112) voxel grid."""
    exc = _avg2(_avg2(ex, 1), 2)
    eyc = _avg2(_avg2(ey, 0), 2)
    ezc = _avg2(_avg2(ez, 0), 1)
    # Treat voxels where any component is outside the grid as outside.
    outside = np.isnan(exc) | np.isnan(eyc) | np.isnan(ezc)
    emag = np.sqrt(np.nan_to_num(exc) ** 2
                   + np.nan_to_num(eyc) ** 2
                   + np.nan_to_num(ezc) ** 2)
    emag[outside] = np.nan
    return emag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5, help="Path to Output.h5")
    ap.add_argument("--out", default=str(OUT_PNG), help="Output PNG path")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as h5:
        ex, ey, ez = load_abs_components(h5)
        axes = {
            k: np.asarray(h5[f"Meshes/{MESH_GROUP}/axis_{k}"]) * 1e3  # m -> mm
            for k in "xyz"
        }

    emag = to_voxel_centers(ex, ey, ez)
    print(f"|E| voxel grid shape: {emag.shape}")

    # Voxel-center coordinates (mm)
    centers = {k: 0.5 * (axes[k][:-1] + axes[k][1:]) for k in "xyz"}
    xc, yc, zc = centers["x"], centers["y"], centers["z"]

    fi = np.unravel_index(np.nanargmax(emag), emag.shape)
    fval = emag[fi]
    fpos = (xc[fi[0]], yc[fi[1]], zc[fi[2]])
    print(f"Focus voxel index (ix,iy,iz) = {fi}")
    print(f"Focus position (x,y,z) = ({fpos[0]:.2f}, {fpos[1]:.2f}, {fpos[2]:.2f}) mm")
    print(f"Focus |E|/I = {fval:.3e} V/m per A  (reference ~1.93e-2 V/m/A)")
    print(f"Equivalent focal field at 5 kA: {fval * 5000:.1f} V/m (reference ~96 V/m)")

    # (a) vertical x-z slice through the focus (fixed y = y_focus); z vertical
    sl = emag[:, fi[1], :]  # (nx, nz)
    # (b) depth profile along z through the focus (fixed x, y), measured as
    # distance below the head surface (top of the valid region at this x, y)
    depth_line = emag[fi[0], fi[1], :]  # (nz,)
    valid_z = ~np.isnan(depth_line)
    z_surface = zc[valid_z].max()
    depth_mm = z_surface - zc  # 0 at the head surface, increasing downward

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    pcm = ax1.pcolormesh(xc, zc, sl.T, shading="auto", cmap="inferno")
    ax1.plot(fpos[0], fpos[2], "c*", markersize=15, markeredgecolor="white",
             label=f"focus (x={fpos[0]:.1f}, z={fpos[2]:.1f}) mm")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("z [mm]")
    ax1.set_aspect("equal")
    ax1.set_title("(a) |E| slice through focus (y = %.1f mm)" % fpos[1])
    ax1.legend(loc="upper right", fontsize=8)
    fig.colorbar(pcm, ax=ax1, label="|E| / I [V/m per A]")

    ax2.plot(depth_mm[valid_z], depth_line[valid_z], "b.-", lw=1.2)
    ax2.axvline(z_surface - fpos[2], color="r", ls="--", lw=1,
                label=f"focus, depth = {z_surface - fpos[2]:.1f} mm")
    ax2.set_xlabel("Depth below head surface [mm]")
    ax2.set_ylabel("|E| / I [V/m per A]")
    ax2.set_title("(b) Depth decay through focus (x = %.1f, y = %.1f mm)"
                  % (fpos[0], fpos[1]))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
