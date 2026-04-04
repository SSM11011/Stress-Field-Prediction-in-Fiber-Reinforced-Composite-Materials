# -*- coding: utf-8 -*-
"""
ABAQUS 2D plane-strain FEM simulation for fiber-matrix composite RVE.

Arguments passed after '--' by abaqus cae:
    argv[-2] = sample_id   (integer)
    argv[-1] = n_fibers    (integer, for radius computation)

Run via:
    abaqus cae noGUI=data/abaqus_simulation.py -- <sample_id> <n_fibers>

Reads:  outputs/microstructures/nf{n_fibers}/fibers_{sample_id}.npy
Writes: job_{n_fibers}_{sample_id}.odb  (in current working directory)
"""

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import math
import numpy as np
import sys
import os

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
sample_id = int(sys.argv[-2])
n_fibers  = int(sys.argv[-1])

# Fiber radius consistent with 30% VF
VF_TARGET = 0.30
RADIUS = math.sqrt(VF_TARGET / (n_fibers * math.pi))

fibers_path = os.path.join(
    "outputs", "microstructures", "nf%d" % n_fibers,
    "fibers_%d.npy" % sample_id
)
fibers = np.load(fibers_path)

# ---------------------------------------------------------------------------
# Model geometry
# ---------------------------------------------------------------------------
model_name = "Composite_nf%d_%d" % (n_fibers, sample_id)
model = mdb.Model(name=model_name)

s = model.ConstrainedSketch(name="sketch", sheetSize=2.0)
s.rectangle(point1=(0, 0), point2=(1, 1))
for (x, y) in fibers:
    s.CircleByCenterPerimeter(
        center=(float(x), float(y)),
        point1=(float(x) + RADIUS, float(y))
    )

part = model.Part(name="Part",
                  dimensionality=TWO_D_PLANAR,
                  type=DEFORMABLE_BODY)
part.BaseShell(sketch=s)

# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
model.Material(name="Matrix")
model.materials["Matrix"].Elastic(table=((3.0e9, 0.35),))  # Epoxy

model.Material(name="Fiber")
model.materials["Fiber"].Elastic(table=((70.0e9, 0.20),))  # Glass / Al-like

model.HomogeneousSolidSection(name="MatrixSec", material="Matrix")
model.HomogeneousSolidSection(name="FiberSec",  material="Fiber")

# ---------------------------------------------------------------------------
# Section assignment: identify fiber faces by centroid proximity
# ---------------------------------------------------------------------------
for i in range(len(part.faces)):
    face = part.faces[i]
    pt = face.pointOn[0]
    is_fiber = any(
        math.hypot(pt[0] - float(fx), pt[1] - float(fy)) < RADIUS * 0.99
        for fx, fy in fibers
    )
    region = regionToolset.Region(faces=part.faces[i:i+1])
    sec_name = "FiberSec" if is_fiber else "MatrixSec"
    part.SectionAssignment(region=region, sectionName=sec_name)

# ---------------------------------------------------------------------------
# Assembly and step
# ---------------------------------------------------------------------------
assembly = model.rootAssembly
instance  = assembly.Instance(name="Inst", part=part, dependent=ON)

model.StaticStep(name="Load", previous="Initial")

# ---------------------------------------------------------------------------
# Boundary conditions (uniaxial tension in x)
# ---------------------------------------------------------------------------
left_edges = instance.edges.getByBoundingBox(
    xMin=-0.001, xMax=0.001, yMin=-0.001, yMax=1.001)
model.DisplacementBC(name="FixLeft", createStepName="Initial",
                     region=regionToolset.Region(edges=left_edges), u1=SET)

bl_verts = instance.vertices.getByBoundingBox(
    xMin=-0.001, xMax=0.001, yMin=-0.001, yMax=0.001)
model.DisplacementBC(name="FixCorner", createStepName="Initial",
                     region=regionToolset.Region(vertices=bl_verts), u2=SET)

right_edges = instance.edges.getByBoundingBox(
    xMin=0.999, xMax=1.001, yMin=-0.001, yMax=1.001)
model.DisplacementBC(name="Pull", createStepName="Load",
                     region=regionToolset.Region(edges=right_edges), u1=0.01)

# ---------------------------------------------------------------------------
# Mesh (plane-strain quadrilateral elements, CPE4R reduced integration)
# ---------------------------------------------------------------------------
elemType1 = mesh.ElemType(elemCode=CPE4R, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=CPE3,  elemLibrary=STANDARD)
part.setElementType(regions=(part.faces[:],), elemTypes=(elemType1, elemType2))

mesh_size = min(RADIUS * 0.5, 0.015)   # finer mesh near smaller fibers
part.seedPart(size=mesh_size)
part.generateMesh()

# ---------------------------------------------------------------------------
# Submit job
# ---------------------------------------------------------------------------
job_name = "job_nf%d_%d" % (n_fibers, sample_id)
mdb.Job(name=job_name, model=model_name)
mdb.jobs[job_name].submit()
mdb.jobs[job_name].waitForCompletion()

print("Simulation complete: %s" % job_name)
