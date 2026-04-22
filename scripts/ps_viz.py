#!/usr/bin/env python3
"""
Visualize multiple per-vertex scalar attributes on an OBJ mesh using Polyscope.
Each attribute file should contain one scalar per vertex (same order as OBJ).
"""

import argparse
from pathlib import Path
import numpy as np
import polyscope as ps
import igl


def read_vertex_scalars_txt(path: Path) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for tok in s.split():
                vals.append(float(tok))
    if len(vals) == 0:
        raise ValueError(f"No scalars found in {path}")
    return np.asarray(vals, dtype=np.float64)


def normalize_array(x: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, required=True, help="Path to OBJ mesh")
    ap.add_argument(
        "--attrs",
        type=Path,
        nargs="+",
        required=True,
        help="List of attribute txt files",
    )
    ap.add_argument(
        "--names",
        nargs="*",
        help="Optional names for the attributes (same order as --attrs)",
    )
    ap.add_argument("--normalize", action="store_true")    
    args = ap.parse_args()

    V, F = igl.read_triangle_mesh(str(args.mesh))
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)

    if args.names and len(args.names) != len(args.attrs):
        raise ValueError("Number of names must match number of attribute files")

    ps.init()    

    m = ps.register_surface_mesh(args.mesh.stem, V, F)

    for i, attr_path in enumerate(args.attrs):
        if not attr_path.exists():
            raise FileNotFoundError(attr_path)

        scalars = read_vertex_scalars_txt(attr_path)

        if scalars.shape[0] != V.shape[0]:
            raise ValueError(
                f"{attr_path}: expected {V.shape[0]} scalars, got {scalars.shape[0]}"
            )

        if args.normalize:
            scalars = normalize_array(scalars)

        name = (
            args.names[i]
            if args.names
            else attr_path.stem
        )

        m.add_scalar_quantity(
            name,
            scalars,
            defined_on="vertices",
            enabled=(i == 0),
        )
        
    ps.show()


if __name__ == "__main__":
    main()
