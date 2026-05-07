"""
Build Gmsh-generated FSI meshes for DOLFINx.

This module provides:
    - 2D channel-style fluid+solid meshes split by a horizontal interface.
    - 3D tapered cylindrical fluid cores with annular solid walls.
    - 3D rectangular fluid channels with a top elastic plate.

Both builders return a DOLFINx mesh plus cell/facet MeshTags suitable for
boundary-condition and variational-form assembly in ALE-FSI notebooks.

Author:
-------
Ivan C. Christov, Purdue University
(with input from Claude and GitHub Copilot)
March 2026, May 2026
"""

from mpi4py import MPI

# ---------------------------------------------------------------------------
# Tag constants — import these in simulation notebooks instead of redefining.
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

# 3D tapered-cylinder facet tags
FLUID_INLET_3D_TAG        = 30  # z = 0,    r <= a0
FLUID_OUTLET_3D_TAG       = 31  # z = L,    r <= aL
INTERFACE_3D_TAG          = 32  # r = a(z), shared fluid-solid surface
RIGID_OUTER_WALL_3D_TAG   = 33  # r = b(z), exterior rigid wall
SOLID_INLET_RING_3D_TAG   = 34  # z = 0,    a0 <= r <= b0
SOLID_OUTLET_RING_3D_TAG  = 35  # z = L,    aL <= r <= bL

# 3D rectangular-channel + elastic-top-plate facet tags
CHANNEL_FLUID_INLET_3D_TAG       = 40  # z = 0 on fluid volume
CHANNEL_FLUID_OUTLET_3D_TAG      = 41  # z = L on fluid volume
CHANNEL_INTERFACE_3D_TAG         = 42  # y = fluid_height, shared fluid-solid surface
CHANNEL_RIGID_BOTTOM_3D_TAG      = 43  # y = 0 on fluid volume
CHANNEL_RIGID_LEFT_3D_TAG        = 44  # x = xmin on fluid volume
CHANNEL_RIGID_RIGHT_3D_TAG       = 45  # x = xmax on fluid volume
CHANNEL_SOLID_CLAMP_LEFT_3D_TAG  = 46  # x = xmin on solid volume
CHANNEL_SOLID_CLAMP_RIGHT_3D_TAG = 47  # x = xmax on solid volume
CHANNEL_SOLID_TOP_3D_TAG         = 48  # y = fluid_height + solid_thickness
CHANNEL_SOLID_INLET_3D_TAG       = 49  # z = 0 on solid volume
CHANNEL_SOLID_OUTLET_3D_TAG      = 50  # z = L on solid volume


def _validate_3d_tapered_inputs(N,
                                length,
                                r_inner_0,
                                r_inner_l,
                                r_outer_0,
                                r_outer_l,
                                interface_refinement,
                                inlet_refinement,
                                outlet_refinement):
    """Validate geometry and optional local-refinement controls for 3D meshes."""
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if length <= 0:
        raise ValueError("length must be > 0.")
    for name, value in [
        ("r_inner_0", r_inner_0),
        ("r_inner_l", r_inner_l),
        ("r_outer_0", r_outer_0),
        ("r_outer_l", r_outer_l),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")
    if r_outer_0 <= r_inner_0 or r_outer_l <= r_inner_l:
        raise ValueError("Outer radius must exceed inner radius at z=0 and z=L.")

    for name, value in [
        ("interface_refinement", interface_refinement),
        ("inlet_refinement", inlet_refinement),
        ("outlet_refinement", outlet_refinement),
    ]:
        if value is not None and not (0.0 < value < 1.0):
            raise ValueError(f"{name} must be in (0, 1) or None.")


def _classify_3d_surface_groups(gmsh,
                                fluid_only_surfs,
                                solid_only_surfs,
                                interface_surfs,
                                z0,
                                z1,
                                tol,
                                length):
    """Split 3D boundary surfaces into inlet/outlet/interface/outer-ring groups."""
    # Cap/lateral split uses z-span: caps are nearly planar at constant z.
    cap_tol = max(10.0 * tol, 1e-4 * length)

    def _surface_bbox_info(surfs):
        info = {}
        for s in surfs:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s)
            info[s] = {
                "zmid": 0.5 * (zmin + zmax),
                "zspan": zmax - zmin,
            }
        return info

    def _split_caps_and_lateral(surfs):
        info = _surface_bbox_info(surfs)
        caps = [s for s in surfs if info[s]["zspan"] <= cap_tol]
        lateral = [s for s in surfs if info[s]["zspan"] > cap_tol]
        return caps, lateral, info

    fluid_caps, fluid_lateral, fluid_info = _split_caps_and_lateral(fluid_only_surfs)
    solid_caps, solid_lateral, solid_info = _split_caps_and_lateral(solid_only_surfs)

    if fluid_lateral:
        raise RuntimeError(
            f"Unexpected lateral surfaces in fluid-only boundary: {fluid_lateral}."
        )

    fluid_inlet = [s for s in fluid_caps
                   if abs(fluid_info[s]["zmid"] - z0) <= abs(fluid_info[s]["zmid"] - z1)]
    fluid_outlet = [s for s in fluid_caps if s not in fluid_inlet]

    solid_inlet_ring = [s for s in solid_caps
                        if abs(solid_info[s]["zmid"] - z0) <= abs(solid_info[s]["zmid"] - z1)]
    solid_outlet_ring = [s for s in solid_caps if s not in solid_inlet_ring]
    rigid_outer_wall = solid_lateral

    groups = {
        "fluid_inlet": fluid_inlet,
        "fluid_outlet": fluid_outlet,
        "interface": interface_surfs,
        "rigid_outer_wall": rigid_outer_wall,
        "solid_inlet_ring": solid_inlet_ring,
        "solid_outlet_ring": solid_outlet_ring,
    }
    missing = [name for name, surfs in groups.items() if not surfs]
    if missing:
        raise RuntimeError(f"Failed to classify required 3D boundary groups: {missing}.")

    return groups


def _apply_3d_refinement_fields(gmsh,
                                h,
                                length,
                                groups,
                                interface_refinement,
                                inlet_refinement,
                                outlet_refinement):
    """Apply optional local size fields around selected 3D boundary groups."""
    active_fields = []

    def _add_surface_refinement(surface_tags, factor, dist_max_frac):
        if factor is None or not surface_tags:
            return

        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", surface_tags)

        f_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h * factor)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", dist_max_frac * length)
        active_fields.append(f_thresh)

    inlet_surfs = groups["fluid_inlet"] + groups["solid_inlet_ring"]
    outlet_surfs = groups["fluid_outlet"] + groups["solid_outlet_ring"]

    _add_surface_refinement(groups["interface"], interface_refinement, 0.2)
    _add_surface_refinement(inlet_surfs, inlet_refinement, 0.2)
    _add_surface_refinement(outlet_surfs, outlet_refinement, 0.2)

    if active_fields:
        # Prevent global boundary-size extension from diluting local controls.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        if len(active_fields) == 1:
            gmsh.model.mesh.field.setAsBackgroundMesh(active_fields[0])
        else:
            f_min = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", active_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_min)


def _augment_and_validate_3d_facet_tags(dmesh,
                                        np,
                                        mesh_data,
                                        z0,
                                        z1,
                                        length,
                                        r_inner_0,
                                        r_inner_l,
                                        r_outer_0,
                                        r_outer_l,
                                        tol):
    """Geometrically retag 3D caps/rings and enforce mandatory facet tags."""
    mesh = mesh_data.mesh
    fdim = mesh.topology.dim - 1
    z_tol = max(10.0 * tol, 1e-4 * length)
    r_tol = max(10.0 * tol, 1e-4 * max(r_outer_0, r_outer_l))

    def _radial(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)

    def _is_inlet_fluid(x):
        r = _radial(x)
        return np.logical_and(np.abs(x[2] - z0) <= z_tol, r <= r_inner_0 + r_tol)

    def _is_inlet_solid_ring(x):
        r = _radial(x)
        return np.logical_and.reduce((
            np.abs(x[2] - z0) <= z_tol,
            r >= r_inner_0 - r_tol,
            r <= r_outer_0 + r_tol,
        ))

    def _is_outlet_fluid(x):
        r = _radial(x)
        return np.logical_and(np.abs(x[2] - z1) <= z_tol, r <= r_inner_l + r_tol)

    def _is_outlet_solid_ring(x):
        r = _radial(x)
        return np.logical_and.reduce((
            np.abs(x[2] - z1) <= z_tol,
            r >= r_inner_l - r_tol,
            r <= r_outer_l + r_tol,
        ))

    inlet_fluid_facets = dmesh.locate_entities_boundary(mesh, fdim, _is_inlet_fluid)
    inlet_solid_ring_facets = dmesh.locate_entities_boundary(mesh, fdim, _is_inlet_solid_ring)
    outlet_fluid_facets = dmesh.locate_entities_boundary(mesh, fdim, _is_outlet_fluid)
    outlet_solid_ring_facets = dmesh.locate_entities_boundary(mesh, fdim, _is_outlet_solid_ring)

    mandatory_cap_groups = {
        "fluid_inlet": inlet_fluid_facets,
        "fluid_outlet": outlet_fluid_facets,
        "solid_inlet_ring": inlet_solid_ring_facets,
        "solid_outlet_ring": outlet_solid_ring_facets,
    }
    missing_mandatory_caps = [
        name for name, facets in mandatory_cap_groups.items() if len(facets) == 0
    ]
    if missing_mandatory_caps:
        raise RuntimeError(
            "Mandatory 3D cap groups are empty after geometric classification: "
            f"{missing_mandatory_caps}."
        )

    # Start from gmsh physical tags, then override end caps/rings geometrically.
    facet_to_tag = {
        int(i): int(v) for i, v in zip(mesh_data.facet_tags.indices, mesh_data.facet_tags.values)
    }

    for f in inlet_fluid_facets:
        facet_to_tag[int(f)] = FLUID_INLET_3D_TAG
    for f in outlet_fluid_facets:
        facet_to_tag[int(f)] = FLUID_OUTLET_3D_TAG
    for f in inlet_solid_ring_facets:
        if facet_to_tag.get(int(f)) != FLUID_INLET_3D_TAG:
            facet_to_tag[int(f)] = SOLID_INLET_RING_3D_TAG
    for f in outlet_solid_ring_facets:
        if facet_to_tag.get(int(f)) != FLUID_OUTLET_3D_TAG:
            facet_to_tag[int(f)] = SOLID_OUTLET_RING_3D_TAG

    facet_indices = np.array(sorted(facet_to_tag.keys()), dtype=np.int32)
    facet_values = np.array([facet_to_tag[i] for i in facet_indices], dtype=np.int32)

    mandatory_facet_tags = {
        FLUID_INLET_3D_TAG,
        FLUID_OUTLET_3D_TAG,
        INTERFACE_3D_TAG,
        RIGID_OUTER_WALL_3D_TAG,
        SOLID_INLET_RING_3D_TAG,
        SOLID_OUTLET_RING_3D_TAG,
    }
    present_facet_tags = set(np.unique(facet_values).tolist())
    missing_mandatory_tags = sorted(mandatory_facet_tags - present_facet_tags)
    if missing_mandatory_tags:
        raise RuntimeError(
            "Mandatory 3D facet tags missing after tag assembly: "
            f"{missing_mandatory_tags}."
        )

    facet_tags = dmesh.meshtags(mesh, fdim, facet_indices, facet_values)
    return mesh, facet_tags


def build_2D_mesh_with_gmsh(N=100,
                            domain_size=None,
                            basename="fsi_rect",
                            inlet_refinement=None,
                            inlet_refinement_solid_only=False,
                            interface_refinement=None,
                            outlet_refinement=None):
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
    outlet_refinement : float < 1 or None
        Refinement factor at the outlet (fraction of h, e.g. 0.2 = 5x finer).
        None disables outlet refinement.

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
    from dolfinx.io import gmsh as gmshio

    if domain_size is None:
        domain_size = [0.0, 1.0, 0.0, 1.0, 0.5]

    for name, value in [
        ("interface_refinement", interface_refinement),
        ("inlet_refinement", inlet_refinement),
        ("outlet_refinement", outlet_refinement),
    ]:
        if value is not None and not (0.0 < value < 1.0):
            raise ValueError(f"{name} must be in (0, 1) or None.")

    XMIN, XMAX, YMIN, YMAX, IFACE = domain_size

    h = max(XMAX - XMIN, YMAX - YMIN) / N
    DX = XMAX - XMIN
    DY = YMAX - YMIN
    DY_IFACE = IFACE

    gmsh.initialize()
    try:
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

        # -------------------------------------------------------------------
        # Physical groups for cell domains
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # Physical groups for boundary facets
        # -------------------------------------------------------------------
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
                    inflow_curves.append(tag)
                else:
                    wall_l_curves.append(tag)
            elif abs(xmin_c - XMAX) < 1e-6 and abs(xmax_c - XMAX) < 1e-6:
                if ymax_c <= IFACE + 1e-6:
                    outflow_curves.append(tag)
                else:
                    wall_r_curves.append(tag)
            elif (abs(ymin_c - IFACE) < 1e-6 and abs(ymax_c - IFACE) < 1e-6
                  and xmax_c - xmin_c > 1e-6):
                interface_curves.append(tag)

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

        # Optional local refinements are blended with a Min field.
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        active_fields = []

        if interface_refinement is not None:
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

        if inlet_refinement is not None:
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
                    # Extra box filter keeps inlet refinement confined to solid region.
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

        if outlet_refinement is not None:
            # 2D symmetry with inlet control: refine near x = XMAX outlet curves.
            outlet_curves_ref = []
            for dim, tag in gmsh.model.getEntities(1):
                bbox = gmsh.model.getBoundingBox(dim, tag)
                xmin_c, _, _, xmax_c, _, _ = bbox
                if abs(xmin_c - XMAX) < 1e-6 and abs(xmax_c - XMAX) < 1e-6:
                    outlet_curves_ref.append(tag)

            if outlet_curves_ref:
                f_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", outlet_curves_ref)

                f_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h * outlet_refinement)
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 0.3 * DX)
                active_fields.append(f_thresh)

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

        gmsh.model.mesh.generate(2)

        if basename:
            gmsh.write(f"{basename}.msh")

        mesh_data = gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2
        )

        return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    finally:
        gmsh.finalize()


def build_3D_tapered_cylinder_mesh_with_gmsh(
        N=80,
        length=1.0,
        r_inner_0=0.25,
        r_inner_l=0.10,
        r_outer_0=0.45,
        r_outer_l=0.20,
    interface_refinement=None,
    inlet_refinement=None,
    outlet_refinement=None,
        basename="fsi_tapered_cylinder"):
    """
    Build a conformal 3D tapered cylinder FSI mesh with Gmsh.

    Geometry:
      - Fluid core:  0 <= r <= a(z)
      - Solid wall:  a(z) <= r <= b(z)
      with linear tapers a(z), b(z) from z=0 to z=length.

        Special cases:
            - If a(0) == a(L), the fluid core is generated as a straight cylinder.
            - If b(0) == b(L), the outer wall is generated as a straight cylinder.
            These are handled natively (no caller-side perturbation needed).

    Parameters
    ----------
    N : int
        Approximate number of elements along the largest geometric scale.
    length : float
        Axial length L of the domain.
    r_inner_0, r_inner_l : float
        Fluid radii a(0), a(L).
    r_outer_0, r_outer_l : float
        Outer wall radii b(0), b(L), must satisfy b > a at both ends.
    interface_refinement : float < 1 or None
        Refinement factor near the fluid-solid interface surface.
        None disables this refinement.
    inlet_refinement : float < 1 or None
        Refinement factor near inlet cap surfaces (fluid inlet + solid ring).
        None disables this refinement.
    outlet_refinement : float < 1 or None
        Refinement factor near outlet cap surfaces (fluid outlet + solid ring).
        None disables this refinement.
    basename : str
        Base filename for the saved .msh file (debugging/inspection).
        Pass None or "" to skip writing the .msh file.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
    cell_tags : dolfinx.mesh.MeshTags
        Cell markers: FLUID_FLAG for fluid cells, SOLID_FLAG for solid cells.
    facet_tags : dolfinx.mesh.MeshTags
        Facet markers use *_3D_TAG constants defined in this module.
    """
    import gmsh
    import numpy as np
    from dolfinx import mesh as dmesh
    from dolfinx.io import gmsh as gmshio

    _validate_3d_tapered_inputs(
        N,
        length,
        r_inner_0,
        r_inner_l,
        r_outer_0,
        r_outer_l,
        interface_refinement,
        inlet_refinement,
        outlet_refinement,
    )

    h = max(length, 2.0 * r_outer_0, 2.0 * r_outer_l) / N
    z0, z1 = 0.0, float(length)
    tol = 1e-7 * max(1.0, length, r_outer_0, r_outer_l)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("fsi_tapered_cylinder")
        occ = gmsh.model.occ

        # Stage 1: Build inner/outer solids and cut to form annular solid wall.
        # OpenCASCADE addCone raises when r0 == r1, so use cylinders for zero taper.
        def _add_axisymmetric_solid(r0, r1):
            if abs(r1 - r0) <= tol:
                return occ.addCylinder(0.0, 0.0, z0, 0.0, 0.0, z1 - z0, r0)
            return occ.addCone(0.0, 0.0, z0, 0.0, 0.0, z1 - z0, r0, r1)

        inner = _add_axisymmetric_solid(r_inner_0, r_inner_l)
        outer = _add_axisymmetric_solid(r_outer_0, r_outer_l)
        solid_cut, _ = occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=False)
        occ.synchronize()

        if not solid_cut:
            raise RuntimeError("Failed to construct tapered annular solid volume.")

        # Stage 2: Fragment so fluid/solid share a conformal interface.
        fragment_in = [(3, inner)] + solid_cut
        occ.fragment(fragment_in, [])
        occ.synchronize()

        volumes = [tag for dim, tag in gmsh.model.getEntities(3)]
        if len(volumes) != 2:
            raise RuntimeError(f"Expected 2 volumes (fluid+solid), found {len(volumes)}.")

        # Stage 3: Identify fluid vs solid volume from boundary-surface counts.
        vol_boundary = {}
        for v in volumes:
            bnd = gmsh.model.getBoundary([(3, v)], oriented=False, recursive=False)
            vol_boundary[v] = [s for dim, s in bnd if dim == 2]

        fluid_vol = None
        solid_vol = None
        for v in volumes:
            n_surf = len(vol_boundary[v])
            if n_surf == 3:
                fluid_vol = v
            elif n_surf == 4:
                solid_vol = v

        if fluid_vol is None or solid_vol is None:
            raise RuntimeError("Could not robustly identify fluid/solid volumes from topology.")

        gmsh.model.addPhysicalGroup(3, [fluid_vol], FLUID_FLAG)
        gmsh.model.setPhysicalName(3, FLUID_FLAG, "fluid")
        gmsh.model.addPhysicalGroup(3, [solid_vol], SOLID_FLAG)
        gmsh.model.setPhysicalName(3, SOLID_FLAG, "solid")

        fluid_surfs = set(vol_boundary[fluid_vol])
        solid_surfs = set(vol_boundary[solid_vol])

        interface_surfs = list(fluid_surfs & solid_surfs)
        fluid_only_surfs = list(fluid_surfs - solid_surfs)
        solid_only_surfs = list(solid_surfs - fluid_surfs)

        if len(interface_surfs) != 1:
            raise RuntimeError(
                f"Expected 1 fluid-solid interface surface, found {len(interface_surfs)}."
            )

        # Stage 4: Classify boundary surfaces into mandatory physical groups.
        groups = _classify_3d_surface_groups(
            gmsh,
            fluid_only_surfs,
            solid_only_surfs,
            interface_surfs,
            z0,
            z1,
            tol,
            length,
        )

        for phys_tag, surfs, name in [
            (FLUID_INLET_3D_TAG,       groups["fluid_inlet"],       "fluid_inlet"),
            (FLUID_OUTLET_3D_TAG,      groups["fluid_outlet"],      "fluid_outlet"),
            (INTERFACE_3D_TAG,         groups["interface"],         "interface"),
            (RIGID_OUTER_WALL_3D_TAG,  groups["rigid_outer_wall"],  "rigid_outer_wall"),
            (SOLID_INLET_RING_3D_TAG,  groups["solid_inlet_ring"],  "solid_inlet_ring"),
            (SOLID_OUTLET_RING_3D_TAG, groups["solid_outlet_ring"], "solid_outlet_ring"),
        ]:
            gmsh.model.addPhysicalGroup(2, surfs, phys_tag)
            gmsh.model.setPhysicalName(2, phys_tag, name)

        # Stage 5: Apply base size + optional local refinement fields and mesh.
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        _apply_3d_refinement_fields(
            gmsh,
            h,
            length,
            groups,
            interface_refinement,
            inlet_refinement,
            outlet_refinement,
        )
        gmsh.model.mesh.generate(3)

        if basename:
            gmsh.write(f"{basename}.msh")

        mesh_data = gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3
        )

        # Stage 6: Convert to DOLFINx and enforce robust cap/ring facet tagging.
        mesh, facet_tags = _augment_and_validate_3d_facet_tags(
            dmesh,
            np,
            mesh_data,
            z0,
            z1,
            length,
            r_inner_0,
            r_inner_l,
            r_outer_0,
            r_outer_l,
            tol,
        )

        return mesh, mesh_data.cell_tags, facet_tags
    finally:
        gmsh.finalize()


def build_3D_channel_mesh_with_gmsh(
        N=40,
        length=1.0,
        width=0.5,
        fluid_height=0.25,
        solid_thickness=0.10,
        interface_refinement=None,
        inlet_refinement=None,
        outlet_refinement=None,
        basename="fsi_channel_3d"):
    """
    Build a conformal 3D rectangular fluid channel with a top elastic plate.

    Geometry:
      - Fluid block: xmin <= x <= xmax, 0 <= y <= fluid_height, 0 <= z <= length
      - Solid block: xmin <= x <= xmax, fluid_height <= y <= fluid_height+solid_thickness,
                     0 <= z <= length

    The model has exactly one fluid-solid interface surface (y = fluid_height).

    Parameters
    ----------
    N : int
        Approximate number of elements along the largest geometric scale.
    length : float
        Channel length along z.
    width : float
        Channel width along x.
    fluid_height : float
        Undeformed channel height for the fluid region.
    solid_thickness : float
        Thickness of the top elastic plate.
    interface_refinement : float < 1 or None
        Refinement factor near the fluid-solid interface surface.
    inlet_refinement : float < 1 or None
        Refinement factor near the inlet (z=0) surfaces.
    outlet_refinement : float < 1 or None
        Refinement factor near the outlet (z=length) surfaces.
    basename : str
        Base filename for the saved .msh file. Pass None or "" to skip writing.

    Returns
    -------
    mesh : dolfinx.mesh.Mesh
    cell_tags : dolfinx.mesh.MeshTags
    facet_tags : dolfinx.mesh.MeshTags
    """
    import gmsh
    from dolfinx.io import gmsh as gmshio

    if N <= 0:
        raise ValueError("N must be a positive integer.")
    for name, value in [
        ("length", length),
        ("width", width),
        ("fluid_height", fluid_height),
        ("solid_thickness", solid_thickness),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")

    for name, value in [
        ("interface_refinement", interface_refinement),
        ("inlet_refinement", inlet_refinement),
        ("outlet_refinement", outlet_refinement),
    ]:
        if value is not None and not (0.0 < value < 1.0):
            raise ValueError(f"{name} must be in (0, 1) or None.")

    h = max(length, width, fluid_height + solid_thickness) / N
    xmin, xmax = -0.5 * width, 0.5 * width
    y0, y_if, y_top = 0.0, fluid_height, fluid_height + solid_thickness
    z0, z1 = 0.0, length
    geom_scale = max(1.0, length, width, fluid_height, solid_thickness)
    tol = 1e-7 * geom_scale
    pos_tol = max(1e-4 * geom_scale, 10.0 * tol)
    span_tol = max(1e-6 * geom_scale, 10.0 * tol)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("fsi_channel_3d")
        occ = gmsh.model.occ

        fluid = occ.addBox(xmin, y0, z0, width, fluid_height, length)
        solid = occ.addBox(xmin, y_if, z0, width, solid_thickness, length)

        # Fragment enforces a conformal interface mesh between fluid and solid.
        occ.fragment([(3, fluid)], [(3, solid)])
        occ.synchronize()

        volumes = [tag for dim, tag in gmsh.model.getEntities(3)]
        if len(volumes) != 2:
            raise RuntimeError(f"Expected 2 volumes (fluid+solid), found {len(volumes)}.")

        fluid_vol = None
        solid_vol = None
        vol_boundary = {}
        for v in volumes:
            bb = gmsh.model.getBoundingBox(3, v)
            ymid = 0.5 * (bb[1] + bb[4])
            bnd = gmsh.model.getBoundary([(3, v)], oriented=False, recursive=False)
            vol_boundary[v] = [s for dim, s in bnd if dim == 2]
            if ymid <= y_if + 10.0 * tol:
                fluid_vol = v
            else:
                solid_vol = v

        if fluid_vol is None or solid_vol is None:
            raise RuntimeError("Could not identify fluid/solid volumes in 3D channel model.")

        gmsh.model.addPhysicalGroup(3, [fluid_vol], FLUID_FLAG)
        gmsh.model.setPhysicalName(3, FLUID_FLAG, "fluid")
        gmsh.model.addPhysicalGroup(3, [solid_vol], SOLID_FLAG)
        gmsh.model.setPhysicalName(3, SOLID_FLAG, "solid")

        fluid_surfs = set(vol_boundary[fluid_vol])
        solid_surfs = set(vol_boundary[solid_vol])

        interface_surfs = list(fluid_surfs & solid_surfs)
        fluid_only_surfs = list(fluid_surfs - solid_surfs)
        solid_only_surfs = list(solid_surfs - fluid_surfs)

        if len(interface_surfs) != 1:
            raise RuntimeError(
                f"Expected 1 fluid-solid interface surface, found {len(interface_surfs)}."
            )

        fluid_inlet = []
        fluid_outlet = []
        rigid_bottom = []
        rigid_left = []
        rigid_right = []

        for s in fluid_only_surfs:
            bb = gmsh.model.getBoundingBox(2, s)
            x0, yb, zb, x1, yt, zt = bb
            xmid = 0.5 * (x0 + x1)
            ymid = 0.5 * (yb + yt)
            zmid = 0.5 * (zb + zt)
            xspan = x1 - x0
            yspan = yt - yb
            zspan = zt - zb

            if zspan <= span_tol and abs(zmid - z0) <= pos_tol:
                fluid_inlet.append(s)
            elif zspan <= span_tol and abs(zmid - z1) <= pos_tol:
                fluid_outlet.append(s)
            elif yspan <= span_tol and abs(ymid - y0) <= pos_tol:
                rigid_bottom.append(s)
            elif xspan <= span_tol and abs(xmid - xmin) <= pos_tol:
                rigid_left.append(s)
            elif xspan <= span_tol and abs(xmid - xmax) <= pos_tol:
                rigid_right.append(s)

        solid_clamp_left = []
        solid_clamp_right = []
        solid_top = []
        solid_inlet = []
        solid_outlet = []

        for s in solid_only_surfs:
            bb = gmsh.model.getBoundingBox(2, s)
            x0, yb, zb, x1, yt, zt = bb
            xmid = 0.5 * (x0 + x1)
            ymid = 0.5 * (yb + yt)
            zmid = 0.5 * (zb + zt)
            xspan = x1 - x0
            yspan = yt - yb
            zspan = zt - zb

            if xspan <= span_tol and abs(xmid - xmin) <= pos_tol:
                solid_clamp_left.append(s)
            elif xspan <= span_tol and abs(xmid - xmax) <= pos_tol:
                solid_clamp_right.append(s)
            elif yspan <= span_tol and abs(ymid - y_top) <= pos_tol:
                solid_top.append(s)
            elif zspan <= span_tol and abs(zmid - z0) <= pos_tol:
                solid_inlet.append(s)
            elif zspan <= span_tol and abs(zmid - z1) <= pos_tol:
                solid_outlet.append(s)

        groups = {
            "fluid_inlet": fluid_inlet,
            "fluid_outlet": fluid_outlet,
            "interface": interface_surfs,
            "rigid_bottom": rigid_bottom,
            "rigid_left": rigid_left,
            "rigid_right": rigid_right,
            "solid_clamp_left": solid_clamp_left,
            "solid_clamp_right": solid_clamp_right,
            "solid_top": solid_top,
            "solid_inlet": solid_inlet,
            "solid_outlet": solid_outlet,
        }
        missing = [name for name, surfs in groups.items() if not surfs]
        if missing:
            raise RuntimeError(
                "Failed to classify required 3D channel boundary groups: "
                f"{missing}."
            )

        for phys_tag, surfs, name in [
            (CHANNEL_FLUID_INLET_3D_TAG,       groups["fluid_inlet"],       "channel_fluid_inlet"),
            (CHANNEL_FLUID_OUTLET_3D_TAG,      groups["fluid_outlet"],      "channel_fluid_outlet"),
            (CHANNEL_INTERFACE_3D_TAG,         groups["interface"],         "channel_interface"),
            (CHANNEL_RIGID_BOTTOM_3D_TAG,      groups["rigid_bottom"],      "channel_rigid_bottom"),
            (CHANNEL_RIGID_LEFT_3D_TAG,        groups["rigid_left"],        "channel_rigid_left"),
            (CHANNEL_RIGID_RIGHT_3D_TAG,       groups["rigid_right"],       "channel_rigid_right"),
            (CHANNEL_SOLID_CLAMP_LEFT_3D_TAG,  groups["solid_clamp_left"],  "channel_solid_clamp_left"),
            (CHANNEL_SOLID_CLAMP_RIGHT_3D_TAG, groups["solid_clamp_right"], "channel_solid_clamp_right"),
            (CHANNEL_SOLID_TOP_3D_TAG,         groups["solid_top"],         "channel_solid_top"),
            (CHANNEL_SOLID_INLET_3D_TAG,       groups["solid_inlet"],       "channel_solid_inlet"),
            (CHANNEL_SOLID_OUTLET_3D_TAG,      groups["solid_outlet"],      "channel_solid_outlet"),
        ]:
            gmsh.model.addPhysicalGroup(2, surfs, phys_tag)
            gmsh.model.setPhysicalName(2, phys_tag, name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        active_fields = []

        def _add_surface_refinement(surface_tags, factor, dist_max_frac):
            if factor is None or not surface_tags:
                return

            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "FacesList", surface_tags)

            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", h * factor)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", dist_max_frac * length)
            active_fields.append(f_thresh)

        inlet_surfs = groups["fluid_inlet"] + groups["solid_inlet"]
        outlet_surfs = groups["fluid_outlet"] + groups["solid_outlet"]

        _add_surface_refinement(groups["interface"], interface_refinement, 0.2)
        _add_surface_refinement(inlet_surfs, inlet_refinement, 0.2)
        _add_surface_refinement(outlet_surfs, outlet_refinement, 0.2)

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

        gmsh.model.mesh.generate(3)

        if basename:
            gmsh.write(f"{basename}.msh")

        mesh_data = gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3
        )

        return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
    finally:
        gmsh.finalize()
