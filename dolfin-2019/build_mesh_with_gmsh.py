def build_mesh_with_gmsh(N=100,
                         domain_size=[0.0, 1.0, 0.0, 1.0, 0.5],
                         basename="fsi_rect",
                         inlet_refinement=None,
                         inlet_refinement_solid_only=False,
                         interface_refinement=None):
    """
    Build a 2D triangular FSI mesh (two superposed rectangles) with Gmsh,
    respecting the fluid/solid interface at y = IFACE,
    and convert to XDMF format for use in dolfin.

    Author:
    -------
    Ivan C. Christov, Purdue University
    (with input from Copilot)
    November 2025

    Parameters:
    -----------
    N : int
        Approximate number of elements along the longest domain side.
    domain_size : list of 5 floats to define rectangles
        Domain size: [XMIN, XMAX, YMIN, YMAX, IFACE]
    basename : str
        Base filename for output
    inlet_refinement : float < 1 or None
        Refinement factor at inlet (fraction of h, e.g., 0.2 means 5x finer).
        If None, no refinement is applied at inlet.
    inlet_refinement_solid_only : bool
        If True, inlet refinement only applies to solid domain (y >= SOLID_BOTTOM).
        If False, inlet refinement applies to entire domain height.
    interface_refinement : float < 1 or None
        Refinement factor near fluid-solid interface (fraction of h, e.g., 0.5 means 2x finer).
        If None, no refinement is applied at interface.
    """

    import gmsh
    import meshio
    import numpy as np

    XMIN, XMAX, YMIN, YMAX, IFACE = domain_size

    h = max(XMAX-XMIN,YMAX-YMIN)/N
    DX = XMAX - XMIN
    DY = YMAX - YMIN
    DY_IFACE = IFACE

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("fsi_domain")
    occ = gmsh.model.occ

    # Big rectangle: Ω = [0, OMEGA_W] x [0, OMEGA_H]
    r1 = occ.addRectangle(XMIN, YMIN, 0.0, DX, DY)
    # Fluid domain: Ω_f = [0, OMEGA_W] x [0, SOLID_BOTTOM]
    r2 = occ.addRectangle(XMIN, YMIN, 0.0, DX, DY_IFACE)

    # Resolve overlap
    entities, _ = occ.fragment([(2, r1)], [(2, r2)])

    occ.synchronize()

    # Uniform element size on all points (base mesh size)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    # List to collect all field indices for combining
    active_fields = []

    # --- FLUID-SOLID INTERFACE REFINEMENT ---
    if (interface_refinement is not None and
        interface_refinement > 0 and
        interface_refinement < 1):
        # Find the interface curve (horizontal line at y = SOLID_BOTTOM)
        interface_curves = []
        for dim, tag in gmsh.model.getEntities(1):
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, xmax = bbox[0], bbox[3]
            ymin, ymax = bbox[1], bbox[4]

            # Check if curve is at the interface (y ≈ SOLID_BOTTOM)
            if (abs(ymin - IFACE) < 1e-6 and
                abs(ymax - IFACE) < 1e-6 and
                xmax - xmin > 1e-6):
                interface_curves.append(tag)
                print(f"Info    : Found interface curve {tag}: x = [{xmin:.6f}, {xmax:.6f}], y = {ymin:.6f}")

        if interface_curves:
            # Create distance field from interface
            field_interface_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_interface_dist, "CurvesList", interface_curves)
            # More sampling points along curves
            gmsh.model.mesh.field.setNumber(field_interface_dist, "Sampling", N)

            # Create threshold field for interface refinement
            field_interface_threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(field_interface_threshold, "InField", field_interface_dist)
            gmsh.model.mesh.field.setNumber(field_interface_threshold, "SizeMin", h * interface_refinement)
            gmsh.model.mesh.field.setNumber(field_interface_threshold, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(field_interface_threshold, "DistMin", 0.0)
            # Transition distance set to 20% around interface
            gmsh.model.mesh.field.setNumber(field_interface_threshold, "DistMax", 0.2 * DY)

            active_fields.append(field_interface_threshold)
        else:
            print("WARNING: No interface curves found!")

    # --- INLET REFINEMENT (optional) ---
    if inlet_refinement is not None and inlet_refinement > 0 and inlet_refinement < 1:
        # Find the inlet boundary (left edge at x=0)
        inlet_curves = []
        for dim, tag in gmsh.model.getEntities(1):  # Get all curves (1D entities)
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, xmax = bbox[0], bbox[3]

            if inlet_refinement_solid_only:
                # Only include inlet curves in the solid region (y >= SOLID_BOTTOM)
                ymin, ymax = bbox[1], bbox[4]
                if abs(xmin) < 1e-6 and abs(xmax) < 1e-6 and ymin >= IFACE - 1e-6:
                    inlet_curves.append(tag)
            else:
                # Include all inlet curves (entire left boundary)
                if abs(xmin) < 1e-6 and abs(xmax) < 1e-6:
                    inlet_curves.append(tag)

        if inlet_curves:  # Only proceed if inlet curves were found
            # Create distance field from inlet
            field_distance = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_distance, "CurvesList", inlet_curves)

            # Create threshold field for smooth refinement
            field_threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_distance)
            # Fine mesh at inlet
            gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", h * inlet_refinement)
            # Coarse mesh far away
            gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", h)
            # Start refinement immediately
            gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0.0)
            # Transition distance set to 30% of domain width
            gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", 0.3 * DX)

            # If solid-only refinement, restrict inlet field with box field
            if inlet_refinement_solid_only:
                # Create a box field for the solid domain
                field_box = gmsh.model.mesh.field.add("Box")
                # Inside solid - allow fine mesh
                gmsh.model.mesh.field.setNumber(field_box, "VIn", h * inlet_refinement)
                # Outside solid - coarse mesh (this blocks inlet refinement in fluid)
                # Large value to ensure Max works
                gmsh.model.mesh.field.setNumber(field_box, "VOut", h * 10)
                gmsh.model.mesh.field.setNumber(field_box, "XMin", XMIN)
                gmsh.model.mesh.field.setNumber(field_box, "XMax", 0.3 * XMAX)
                gmsh.model.mesh.field.setNumber(field_box, "YMin", IFACE)
                gmsh.model.mesh.field.setNumber(field_box, "YMax", YMAX)
                # Sharp transition
                gmsh.model.mesh.field.setNumber(field_box, "Thickness", 0.01)

                # Take MAX of threshold and box: in fluid (VOut), mesh is coarse;
                # in solid (VIn), use inlet refinement
                field_inlet_restricted = gmsh.model.mesh.field.add("Max")
                gmsh.model.mesh.field.setNumbers(field_inlet_restricted, "FieldsList",
                                                 [field_threshold, field_box])
                active_fields.append(field_inlet_restricted)
            else:
                # No restriction - inlet refinement applies everywhere
                active_fields.append(field_threshold)

    # --- COMBINE ALL ACTIVE FIELDS ---
    if len(active_fields) > 0:
        # Control how algorithm respects the background mesh
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        if len(active_fields) == 1:
            # Single field - set directly as background mesh
            gmsh.model.mesh.field.setAsBackgroundMesh(active_fields[0])
        else:
            # Multiple fields - combine with Min field (takes finest mesh everywhere)
            field_combined = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(field_combined, "FieldsList", active_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_combined)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Save to msh file
    msh_file = f"{basename}.msh"
    gmsh.write(msh_file)
    gmsh.finalize()

    # Convert to XDMF (triangles only)
    msh = meshio.read(msh_file)
    tri_cells = [c for c in msh.cells if c.type == "triangle"]
    tri_data = np.concatenate([c.data for c in tri_cells])
    meshio.write_points_cells(
        f"{basename}.xdmf",
        points=msh.points[:, :2], #drop z coords; using msh.points results in mesh.geometry().dim() = 3
        cells=[("triangle", tri_data)],
    )
    return f"{basename}.xdmf"
