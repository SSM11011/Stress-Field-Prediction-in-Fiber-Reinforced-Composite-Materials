# -*- coding: utf-8 -*-
"""
Extract element centroid coordinates and von Mises stress from ABAQUS .odb files.

Must be run with the Abaqus Python interpreter (which includes odbAccess):
    abaqus python data/extract_stress_coords.py <sample_id> <n_fibers>
    abaqus python data/extract_stress_coords.py all <n_fibers>

Reads:  job_nf{n_fibers}_{sample_id}.odb  (from working directory)
Writes:
    outputs/stress_maps/nf{n_fibers}/stress_{sample_id}.npy  — von Mises values (Pa)
    outputs/stress_maps/nf{n_fibers}/coords_{sample_id}.npy  — element centroids (x, y)
"""

from odbAccess import *
import numpy as np
import sys
import os


def extract_sample(sample_id, n_fibers):
    odb_name = "job_nf%d_%d.odb" % (n_fibers, sample_id)
    if not os.path.exists(odb_name):
        print("ODB not found: %s — skipping." % odb_name)
        return False

    out_dir = os.path.join("outputs", "stress_maps", "nf%d" % n_fibers)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Extracting sample %d (nf=%d) ..." % (sample_id, n_fibers))
    odb = openOdb(odb_name, readOnly=True)

    step  = odb.steps["Load"]
    frame = step.frames[-1]

    # --- von Mises stress scalar field ---
    mises = frame.fieldOutputs["S"].getScalarField(invariant=MISES)

    # --- Compute element centroids from node coordinates ---
    instance = odb.rootAssembly.instances["INST"]

    node_coords = {}
    for node in instance.nodes:
        node_coords[node.label] = (node.coordinates[0], node.coordinates[1])

    label_to_centroid = {}
    for elem in instance.elements:
        cx = cy = 0.0
        conn = elem.connectivity
        n = len(conn)
        for nid in conn:
            cx += node_coords[nid][0]
            cy += node_coords[nid][1]
        label_to_centroid[elem.label] = (cx / n, cy / n)

    # --- Match stress values to centroids ---
    stress_vals = []
    centroids   = []
    for v in mises.values:
        lbl = v.elementLabel
        if lbl in label_to_centroid:
            stress_vals.append(v.data)
            centroids.append(label_to_centroid[lbl])

    odb.close()

    if len(stress_vals) == 0:
        print("  WARNING: no stress values extracted for sample %d." % sample_id)
        return False

    stress_arr = np.array(stress_vals, dtype=np.float64)
    coords_arr = np.array(centroids,   dtype=np.float64)  # (N_elem, 2)

    np.save(os.path.join(out_dir, "stress_%d.npy" % sample_id), stress_arr)
    np.save(os.path.join(out_dir, "coords_%d.npy" % sample_id), coords_arr)

    print("  -> %d elements | stress range: %.3e – %.3e Pa" % (
        len(stress_arr), stress_arr.min(), stress_arr.max()))
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: abaqus python data/extract_stress_coords.py <sample_id|all> <n_fibers>")
        sys.exit(1)

    arg      = sys.argv[-2]
    n_fibers = int(sys.argv[-1])

    if arg == "all":
        import glob as _glob
        # Discover all matching ODB files; no hardcoded upper bound
        odb_pattern = "job_nf%d_*.odb" % n_fibers
        odb_files = sorted(
            _glob.glob(odb_pattern),
            key=lambda p: int(os.path.splitext(p)[0].split("_")[-1])
        )
        if not odb_files:
            print("No ODB files found matching: %s" % odb_pattern)
            sys.exit(0)
        sample_ids = [
            int(os.path.splitext(p)[0].split("_")[-1]) for p in odb_files
        ]
        print("Found %d ODB file(s) to process." % len(sample_ids))
        n_done = 0
        for sid in sample_ids:
            if extract_sample(sid, n_fibers):
                n_done += 1
        print("Done. Extracted %d/%d samples." % (n_done, len(sample_ids)))
    else:
        extract_sample(int(arg), n_fibers)
        print("Done.")
