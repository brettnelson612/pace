# 0001: Bounds, fill, lattices and the Reactor

Status: accepted (September 2026)

## Context

PACE's model has to be a complete input to OpenMC, the MOOSE Reactor module and heat transfer, THM and Cardinal. Before this change it had no way to say what occupies the space between components (the coolant around a pin), where the modeled system ends, or what happens at that edge. Lattices stored one entry per slot and could not be placed anywhere.

An earlier idea was a single "fill" for the whole model: one coolant region equal to the system boundary minus every solid. It fails in every coupled tool:

- **OpenMC and Cardinal.** Cardinal maps MOOSE elements to OpenMC cells by element centroid, and each cell instance receives one temperature and one density. One global coolant cell gets one density for the whole reactor, so moderator feedback loses its spatial shape.
- **THM.** Each `FlowChannel1Phase` needs its own flow area and hydraulic diameter, which only a per-pin coolant region can supply.
- **MOOSE Reactor module.** `PinMeshGenerator` is built from rings, then a background region, then ducts; `AssemblyMeshGenerator` adds an assembly background. The fill is already per level in MOOSE's own vocabulary.

## Decisions

### Composites own space: `bounds` and `fill`

Every `CComponent` has `bounds` (a `GRectanglePrism`, `GHexPrism` or `GCylinder`) and an optional `fill` material.

- **Fill occupies whatever part of the bounds no member occupies.** Nobody authors a fill geometry; translators derive it as bounds minus the union of members. It therefore can't overlap its neighbours or leave gaps next to them.
- **Holes count as fill.** A gap that should be helium must be modeled as helium (as a member, or as the fill of a nested composite bounded by the clad's inner cylinder), or it silently becomes the surrounding fill.
- **Fill is always a single material.** Components are *placed* (`PComponent`, lattice slots), never used as a fill. A placed child's size comes from its own bounds, so the parent never restates it.
- **`fill=None`** means the members are expected to occupy the bounds exactly (e.g. concentric layers tiling a cylinder); leftover space is a translation-time error.
- **Bounds is never a physical wall.** A bound sits either on a repetition line (a pin cell's pitch box, drawn through open water) or on a wall's inner face, when the coolant on each side must be tracked separately (an SFR duct: gap sodium in the assembly's fill, bundle sodium in a nested composite's fill). Walls, ducts and cladding are members.
- **Every composite has bounds.** The bounds-less "grouping" form was dropped: its main use case (a reusable rod) is naturally bounded by its clad, and one form keeps every translator a direct mapping. Grouping for convenience belongs in the Workshop UI, not the stored model.
- The minimum member count is 1 (a guide-tube cell is one tube plus fill).

In OpenMC terms, a composite maps to a universe (one cell per member, plus the fill as the unbounded outermost cell) *and* the parent cell that clips it (region = the child's bounds at the placement's pose). The translator moves the bounds from child to parent. PACE's "fill" is deliberately not OpenMC's "fill" (which means any contents of a cell); OpenMC's meaning appears only inside the OpenMC translator.

### Boundary conditions exist only on the Reactor

"Boundary" had two meanings, now separated: **bounds** (owned space, at every level) and **boundary conditions** (physics at the edge of the modeled system, only on the `Reactor`). Interior surfaces are transmission surfaces in OpenMC and internal interfaces in MOOSE.

All boundary conditions are keyed by `BoundsFace`, a named face of the Reactor's bounds shape (`x_min` ... `z_max` for a box, `side_0` ... `side_5` plus `z_min`/`z_max` for a hex prism, `radial`/`z_min`/`z_max` for a cylinder):

- `neutron_bcs`: `vacuum`, `reflective`, `periodic` (faces must be paired), `white`. Required on exactly the faces of the bounds shape.
- `thermal_bcs`: `adiabatic` for now (also the symmetry condition, and the default for unlisted faces).
- `flow_bcs`: a `FlowInlet` (total mass flow, inlet temperature) and a `FlowOutlet` (pressure), or none. Coolant enters and leaves through faces, so inlet temperature, flow and pressure are boundary conditions rather than operating state.

### The Reactor

`Reactor` (versioned, in the registry) holds `description`, `bounds`, an optional `fill`, the `root` placement, the three condition maps, and an `OperatingState`:

- `power_w`: total thermal power, needed only by coupled runs (Cardinal tally normalization).
- `initial_temperatures_k`: a uniform starting temperature per material. OpenMC selects cross sections by temperature, so even standalone runs need these (the VERA problem 1 variants differ only in fuel temperature).

The Reactor's `fill` covers only space inside its bounds that the root doesn't occupy (usually none; needed for things like bypass flow around a core). It never reaches coolant inside the root's nested composites.

Soluble boron stays in the water material's composition for now; it moves to operating state only if parametric studies need to vary it.

### Lattices

`Lattice` is abstract, with `RectLattice` and `HexLattice`:

- **Addresses.** Rect: `(row, col)`, row 0 on top (matches OpenMC `universes` and the Reactor module `pattern`). Hex: `(ring, index)`, ring 0 the center, ring k has 6k slots, index 0 the top slot proceeding clockwise (OpenMC's within-ring convention; its translator only reverses ring order). `HexOrientation.FLAT_TOP` / `POINTY_TOP` map to OpenMC's `'y'` / `'x'`.
- **Placements, not a full grid.** `placements: list[LatticePlacement]`, one entry per placed component listing every address it occupies. Compact (a 17x17 assembly is three entries), easy to query, and empty slots are implicit. `RectLattice.from_grid()` / `to_grid()` convert for authoring and review. Validation: every address in range, none occupied twice, each ref in one placement.
- **Removed:** `outer`, `lower_left_m`, the per-slot `universes` list, `LatticeCell`, and `GNull`. Position comes from whoever places the lattice; empty slots are simply unlisted.
- **Fill is split per slot.** Each slot's surrounding fill becomes its own cell instance (OpenMC), mesh region (Reactor module background) and THM channel — never one region spanning the whole lattice, which would recreate the single-global-fill problem. Translators also wrap each lattice in a universe before placing it, to avoid the cell-instance collision Cardinal warns about.
- An occupant's bounds must fit inside its slot; the lattice fill covers the rest.

### ComponentRef

Placements go through `ComponentRef(kind, id)`, with `ComponentKind` limited to what can be placed (`lcomponent`, `ccomponent`, `lattice`). Family names are unique only within one kind, so a bare id can't say which registry to look in. It replaces the `component` + `component_type` pair on `PComponent`, and is shared by `PComponent`, lattice placements and `Reactor.root`.

### Translation lives in adapters

All `to_open_mc()` / `to_moose()` stubs are removed from the domain model. Solver adapters consume a `ResolvedReactor` (a `Reactor` plus the fully hydrated `ResolvedModel` of its root) and a run spec.

## Regions vs resolution, and run state

- **The model defines regions** (what distinct things exist, and where). **The run spec sets resolution** (axial layers, fuel rings). Every translator takes resolution from the same run-spec values so Cardinal's element-to-cell mapping stays aligned.
- **Models are uniform within each region.** Spatially varying values (temperature and density fields, burned compositions per pin, layer and ring) are run state, stored as snapshots keyed by region address, never in the registry. A GT-derived version (`gt_run_id` set) is a deliberate, labeled promotion of run state (e.g. averaging a snapshot into one material), not the default outcome of a run. This matches Serpent (`div` depletion zones, restart files), OpenMC (`diff_burnable_mats`, `depletion_results.h5`, `prev_results`) and ARMI (per-time-node database state).

## Deferred

- Geometric containment checks (members inside bounds; occupants inside slots; exact tiling when `fill=None`).
- A fluid equation of state on `Material` (milestone 2).
- Region addresses and state snapshots (milestone 3, with GTRun persistence).
- Whether `GAddition` / `GSubtraction` remain necessary.
- `GHexPrism` has no orientation field yet; hex bounds faces assume one.
