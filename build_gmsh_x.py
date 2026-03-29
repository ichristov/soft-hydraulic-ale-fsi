"""
Build a 2D triangular FSI mesh (two superposed rectangles) with Gmsh,
respecting the fluid/solid interface at y = IFACE, and return a DOLFINx
mesh with tagged subdomains and boundary facets.

FEniCSx migration (Phase 1):
  - Removed meshio dependency.
  - Added gmsh physical groups for cell domains and boundary facets
    (required by dolfinx.io.gmsh.model_to_mesh).
  - dolfinx.io.gmsh.model_to_mesh() is called before gmsh.finalize().
  - Returns (mesh, cell_tags, facet_tags) instead of an xdmf file path.
  - FLUID_FLAG/SOLID_FLAG changed from 0/1 to 1/2 because gmsh physical
    group tags must be >= 1.

Author:
-------
Ivan C. Christov, Purdue University
(with input from Claude)
March 2026
"""

from mpi4py import MPI

# ---------------------------------------------------------------------------
# Tag constants — import these in the simulation notebooks instead of
# redefining FLUID_FLAG / SOLID_FLAG / INFLOW / OUTFLOW there.
# NOTE: FLUID_FLAG was 0 and SOLID_FLAG was 1 in the legacy FEniCS code.
#       They are now 1 and 2 because gmsh physical group tags must be >= 1.
# ---------------------------------------------------------------------------
FLUID_FLAG  = 1   # cell tag: fluid subdomain
SOLID_FLAG  = 2   # cell tag: solid subdomain

INTERFACE_TAG = 10  # facet tag: fluid-solid interface  (y = IFACE, interior edge)
                    # Needed for steady monolithic solver (dS measure)
WALL_B_TAG    = 11  # facet tag: bottom no-slip wall   (y = YMIN)
WALL_T_TAG    = 12  # facet tag: top no-slip wall       (y = YMAX, solid exterior)
INFLOW_TAG    = 13  # facet tag: fluid inlet            (x = XMIN, y <= IFACE)
OUTFLOW_TAG   = 14  # facet tag: fluid outlet           (x = XMAX, y <= IFACE)
WALL_L_TAG    = 15  # facet tag: solid left wall        (x = XMIN, y >= IFACE)
WALL_R_TAG    = 16  # facet tag: solid right wall       (x = XMAX, y >= IFACE)


def build_gmsh_x(N=100,
                 domain_size=[0.0, 1.0, 0.0, 1.0, 0.5],
                 basename="fsi_rect",
                 inlet_refinement=None,
                 inlet_refinement_solid_only=False,
                 interface_refinement=None):
    """
    Build a 2D triangular FSI mesh with Gmsh and return a DOLFINx mesh.

    Parameters
    ----------
    N : int
        Approximate number of elements along the longest domain side.
    domain_size : list of 5 floats
        [XMIN, XMAX, YMIN, YMAX, IFACE] where IFACE is the fluid/solid
        interface height.
    basename : str
        Base filename for the saved .msh file (kept for debugging/inspection).
        Pass None or "" to skip writing the .msh file.
    inlet_refinement : float < 1 or None
        Refinement factor at the inlet (fraction of h, e.g. 0.2 = 5x finer).
        None disables inlet refinement.
    inlet_refinement_solid_only : bool
        If True, inlet refinement is restricted to the solid domain (y >= IFACE).
    interface_refinement : float < 1 or None
        Refinement factor near the fluid-solid interface.
        None disables interface refinement.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
    cell_tags : dolfinx.mesh.MeshTags
        Cell markers: FLUID_FLAG for fluid cells, SOLID_FLAG for solid cells.
    facet_tags : dolfinx.mesh.MeshTags
        Facet markers: WALL_B_TAG, WALL_T_TAG, INFLOW_TAG, OUTFLOW_TAG,
        WALL_L_TAG, WALL_R_TAG.
    """
    import gmsh
    import numpy as np
    from dolfinx.io import gmsh as gmshio

    XMIN, XMAX, YMIN, YMAX, IFACE = domain_size

    h = max(XMAX - XMIN, YMAX - YMIN) / N
    DX = XMAX - XMIN
    DY = YMAX - YMIN
    DY_IFACE = IFACE

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("fsi_domain")
    occ = gmsh.model.occ

    # Big rectangle: Ω = [XMIN, XMAX] x [YMIN, YMAX]
    r1 = occ.addRectangle(XMIN, YMIN, 0.0, DX, DY)
    # Fluid sub-rectangle: Ω_f = [XMIN, XMAX] x [YMIN, IFACE]
    r2 = occ.addRectangle(XMIN, YMIN, 0.0, DX, DY_IFACE)

    # Fragment creates two non-overlapping surfaces sharing the interface edge
    entities, _ = occ.fragment([(2, r1)], [(2, r2)])
    occ.synchronize()

    # -----------------------------------------------------------------------
    # Physical groups for cell domains
    # gmsh.model_to_mesh() uses physical groups to build MeshTags; without
    # them cell_tags / facet_tags will be empty.
    # -----------------------------------------------------------------------
    fluid_surfaces, solid_surfaces = [], []
    for dim, tag in gmsh.model.getEntities(2):
        bbox = gmsh.model.getBoundingBox(dim, tag)
        ymax_s = bbox[4]
        if ymax_s <= IFACE + 1e-6:
            fluid_surfaces.append(tag)
        else:
            solid_surfaces.append(tag)

    gmsh.model.addPhysicalGroup(2, fluid_surfaces, FLUID_FLAG)
    gmsh.model.setPhysicalName(2, FLUID_FLAG, "fluid")
    gmsh.model.addPhysicalGroup(2, solid_surfaces, SOLID_FLAG)
    gmsh.model.setPhysicalName(2, SOLID_FLAG, "solid")

    # -----------------------------------------------------------------------
    # Physical groups for boundary facets
    # Classify curves by their bounding box position.
    # The fluid-solid interface curve is interior — it is intentionally
    # left untagged so it does not appear in facet_tags.
    # -----------------------------------------------------------------------
    wall_b_curves, wall_t_curves = [], []
    inflow_curves, wall_l_curves = [], []
    outflow_curves, wall_r_curves = [], []
    interface_curves = []

    for dim, tag in gmsh.model.getEntities(1):
        bbox = gmsh.model.getBoundingBox(dim, tag)
        xmin_c, ymin_c, _, xmax_c, ymax_c, _ = bbox

        if abs(ymin_c - YMIN) < 1e-6 and abs(ymax_c - YMIN) < 1e-6:
            wall_b_curves.append(tag)
        elif abs(ymin_c - YMAX) < 1e-6 and abs(ymax_c - YMAX) < 1e-6:
            wall_t_curves.append(tag)
        elif abs(xmin_c - XMIN) < 1e-6 and abs(xmax_c - XMIN) < 1e-6:
            if ymax_c <= IFACE + 1e-6:
                inflow_curves.append(tag)   # fluid inlet
            else:
                wall_l_curves.append(tag)   # solid left wall
        elif abs(xmin_c - XMAX) < 1e-6 and abs(xmax_c - XMAX) < 1e-6:
            if ymax_c <= IFACE + 1e-6:
                outflow_curves.append(tag)  # fluid outlet
            else:
                wall_r_curves.append(tag)   # solid right wall
        elif (abs(ymin_c - IFACE) < 1e-6 and abs(ymax_c - IFACE) < 1e-6
              and xmax_c - xmin_c > 1e-6):
            interface_curves.append(tag)    # fluid-solid interface (interior)

    for phys_tag, curves, name in [
        (INTERFACE_TAG, interface_curves, "interface"),
        (WALL_B_TAG,    wall_b_curves,    "wall_b"),
        (WALL_T_TAG,    wall_t_curves,    "wall_t"),
        (INFLOW_TAG,    inflow_curves,    "inflow"),
        (OUTFLOW_TAG,   outflow_curves,   "outflow"),
        (WALL_L_TAG,    wall_l_curves,    "wall_l"),
        (WALL_R_TAG,    wall_r_curves,    "wall_r"),
    ]:
        if curves:
            gmsh.model.addPhysicalGroup(1, curves, phys_tag)
            gmsh.model.setPhysicalName(1, phys_tag, name)
        else:
            print(f"WARNING: No curves found for boundary '{name}' (tag {phys_tag})!")

    # -----------------------------------------------------------------------
    # Mesh size / refinement fields  (logic unchanged from legacy code)
    # -----------------------------------------------------------------------
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    active_fields = []

    # Fluid-solid interface refinement
    if interface_refinement is not None and 0 < interface_refinement < 1:
        interface_curves = []
        for dim, tag in gmsh.model.getEntities(1):
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, xmax = bbox[0], bbox[3]
            ymin, ymax = bbox[1], bbox[4]
            if (abs(ymin - IFACE) < 1e-6 and
                    abs(ymax - IFACE) < 1e-6 and
                    xmax - xmin > 1e-6):
                interface_curves.append(tag)
                print(f"Info    : Found interface curve {tag}: "
                      f"x = [{xmin:.6f}, {xmax:.6f}], y = {ymin:.6f}")

        if interface_curves:
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", interface_curves)
            gmsh.model.mesh.field.setNumber(f_dist, "Sampling", N)

            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h * interface_refinement)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 0.2 * DY)
            active_fields.append(f_thresh)
        else:
            print("WARNING: No interface curves found!")

    # Inlet refinement
    if inlet_refinement is not None and 0 < inlet_refinement < 1:
        inlet_curves = []
        for dim, tag in gmsh.model.getEntities(1):
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin_c, ymin_c, _, xmax_c, ymax_c, _ = bbox
            if inlet_refinement_solid_only:
                if abs(xmin_c - XMIN) < 1e-6 and abs(xmax_c - XMIN) < 1e-6 \
                        and ymin_c >= IFACE - 1e-6:
                    inlet_curves.append(tag)
            else:
                if abs(xmin_c - XMIN) < 1e-6 and abs(xmax_c - XMIN) < 1e-6:
                    inlet_curves.append(tag)

        if inlet_curves:
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", inlet_curves)

            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h * inlet_refinement)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 0.3 * DX)

            if inlet_refinement_solid_only:
                f_box = gmsh.model.mesh.field.add("Box")
                gmsh.model.mesh.field.setNumber(f_box, "VIn",       h * inlet_refinement)
                gmsh.model.mesh.field.setNumber(f_box, "VOut",      h * 10)
                gmsh.model.mesh.field.setNumber(f_box, "XMin",      XMIN)
                gmsh.model.mesh.field.setNumber(f_box, "XMax",      0.3 * XMAX)
                gmsh.model.mesh.field.setNumber(f_box, "YMin",      IFACE)
                gmsh.model.mesh.field.setNumber(f_box, "YMax",      YMAX)
                gmsh.model.mesh.field.setNumber(f_box, "Thickness", 0.01)

                f_max = gmsh.model.mesh.field.add("Max")
                gmsh.model.mesh.field.setNumbers(f_max, "FieldsList", [f_thresh, f_box])
                active_fields.append(f_max)
            else:
                active_fields.append(f_thresh)

    # Combine fields
    if active_fields:
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        if len(active_fields) == 1:
            gmsh.model.mesh.field.setAsBackgroundMesh(active_fields[0])
        else:
            f_min = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", active_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # -----------------------------------------------------------------------
    # Generate mesh
    # -----------------------------------------------------------------------
    gmsh.model.mesh.generate(2)

    if basename:
        gmsh.write(f"{basename}.msh")

    # Convert to DOLFINx — must be called BEFORE gmsh.finalize()
    # In dolfinx 0.10, model_to_mesh returns a MeshData namedtuple:
    # (mesh, cell_tags, facet_tags, ridge_tags, peak_tags, physical_groups)
    mesh_data = gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
    )
    gmsh.finalize()

    return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
