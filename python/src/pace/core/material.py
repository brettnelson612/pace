"""
core/material.py

Material domain objects: Material (the versioned, polymorphic class
holding actual composition data — MIsotopic, MMixture, MVoid — plus
validate()), mirroring geometry.py's structure intentionally — kept in
sync by design, not by accident. Translation to solver inputs lives in
the solver adapters, not here.

No separate bare "Material" identity class/table — a version's family
is just a `family_name` string carried on the version itself. A
version's id is one opaque string built from family_name +
version_label (e.g. "uranium3.2-1", "uranium3.2-2" — see
Material.build_id()).

family_name is NOT unique per row — every version in a family shares
it. Rejecting a duplicate family_name is enforced only at "create a
brand-new family" (v1, derived_from=None) time, and that's application
logic (RegistryService), not a database constraint.

What "material" means physically here: a material is a description of
what substance occupies a region of space — which isotopes are
present, in what relative amounts, and at what reference density —
plus how its properties respond to conditions (conductivity and
specific heat as functions of temperature). It is uniform across every
region it fills. It says nothing about shape (that's Geometry) or about
the actual temperature/density at any point during a run: spatially
varying values are run state (initial values in OperatingConditions;
converged fields in run output), never stored here.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Self

from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import MaterialID
from pace.core.pace_object import PaceObject
from pace.core.versioned import Versioned


class PercentType(StrEnum):
    """How a composition's component weights are expressed.

    Values:
        - ao: atomic percent. The number of atoms of a specific
              nuclide/element divided by the total number of atoms in
              the material. The natural unit for chemical/
              stoichiometric ratios — e.g. UO2's 1:2 uranium-to-oxygen
              atom ratio is exact and unambiguous in "ao" terms.
        - wo: weight percent. The mass of a specific nuclide/element
              divided by the total mass of the material. Natural for
              compositions given by measured or spec'd mass fractions —
              e.g. a structural alloy's constituent metals by weight.

    On MIsotopic the weights are RELATIVE, not absolute percentages
    that must sum to 100: {"U": 1.0, "O": 2.0} under "ao" means a 1:2
    atom ratio. On MMixture the same vocabulary carries a stricter
    constraint (true fractions summing to 1) — see that class. This
    mirrors OpenMC reusing percent_type across add_element()/
    add_nuclide() and mix_materials().
    """

    AO = "ao"
    WO = "wo"


class DensityUnit(StrEnum):
    """Unit of a material's reference density.

    Values:
        - g/cm3: grams per cubic centimeter.
        - kg/m3: kilograms per cubic meter.
        - atom/b-cm: atoms per barn-centimeter — atomic number density
              in units directly compatible with microscopic cross
              sections (a barn is 10^-24 cm^2), so a macroscopic cross
              section is N * sigma with no conversion. Use only when
              atom densities are already in this form.
    """

    G_PER_CM3 = "g/cm3"
    KG_PER_M3 = "kg/m3"
    ATOM_PER_B_CM = "atom/b-cm"


class MaterialType(StrEnum):
    """Discriminator tag for Material subclasses.

    Values:
        - isotopic: a direct composition of nuclides/elements
              (MIsotopic) — "this material is made of these isotopes,
              in these amounts."
        - mixture: a weighted combination of other, already-defined
              Materials (MMixture) — "this material is X% of material
              A plus Y% of material B."
        - void: no material at all (MVoid) — physically distinct from
              a sparse/low-density material: zero cross sections, not
              small ones. See MVoid's docstring for why this can't be
              approximated with MIsotopic instead.
    """

    ISOTOPIC = "isotopic"
    MIXTURE = "mixture"
    VOID = "void"


# material types registered here upon definition — see
# Material.__init_subclass__
MATERIAL_TYPE_TO_CLASS: dict[str, type[Material]] = {}


@dataclass(frozen=True, kw_only=True)
class ThermalScatteringLibrary(PaceObject):
    """One S(alpha,beta) bound-thermal-scattering table entry —
    required for any moderating material (water, graphite, zirconium
    hydride, ...) to be translated correctly, since bound-nuclide
    scattering physics differs from free-gas scattering.

    `name`:     follows OpenMC's GND naming convention (e.g. "c_H_in_H2O").
    `nuclide`:  which nuclide's population this covers (e.g. "C0", "Be9")
    `fraction`: is the atom fraction of the material this table covers

    Most materials need exactly one entry at fraction 1.0; some
    (e.g. BeO, where both Be and O need separate tables) need one
    entry per bound nuclide.

    Example — light water:
        ThermalScatteringLibrary(name="c_H_in_H2O")
    """

    name: str
    nuclide: str
    fraction: float = 1.0

    def validate(self) -> None:
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")

    def to_dict(self) -> dict:
        return {"name": self.name, "nuclide": self.nuclide, "fraction": self.fraction}

    @classmethod
    def from_dict(cls, data: dict) -> ThermalScatteringLibrary:
        return cls(
            name=data["name"], nuclide=data["name"], fraction=data.get("fraction", 1.0)
        )


@dataclass(frozen=True, kw_only=True)
class TabulatedProperty(PaceObject):
    """A temperature-dependent material property as (temperature_k,
    value) points for interpolation — thermal conductivity, specific
    heat, thermal expansion, elastic modulus. Distinct from a
    Material's own simulation_job_id/user_edit lineage: this describes how ONE
    physical property varies with temperature, not how the material's
    composition evolved over time.

    Units are whatever the specific field on Material documents (e.g.
    W/(m*K) for thermal_conductivity) — not enforced here.
    """

    points: tuple[tuple[float, float], ...]

    def validate(self) -> None:
        if len(self.points) < 1:
            raise ValueError("TabulatedProperty needs at least one point")
        temperatures = [t for t, _ in self.points]
        if temperatures != sorted(temperatures):
            raise ValueError("points must be sorted by increasing temperature_k")

    def to_dict(self) -> dict:
        return {"points": [[t, v] for t, v in self.points]}

    @classmethod
    def from_dict(cls, data: dict) -> TabulatedProperty:
        return cls(points=tuple((t, v) for t, v in data["points"]))


@dataclass(frozen=True, kw_only=True)
class MaterialComponentEntry(PaceObject):
    """
    One entry in an MIsotopic's `components` dict — describes a single
    nuclide or element's contribution to a material.

    Mirrors OpenMC's per-element/per-nuclide component shape: a bare
    percent for an exact nuclide, or percent + enrichment fields for
    element-level shorthand.

    Two physically distinct forms, both represented by this one class:

    1. Exact nuclide, no enrichment fields —
       e.g. {"percent": 3.2} under the key "U235". This states exactly
       how much of a specific isotope is present. No ambiguity, no
       expansion needed. This is the ONLY form a simulation-derived (v2+)
       composition can take — depletion output is always exact
       per-isotope densities (see Material and
       MIsotopic._validate_composition).

    2. Element + all three enrichment fields set together —
       e.g. {"percent": 1.0, "enrichment": 3.2, "enrichment_target":
       "U235", "enrichment_type": "wo"} under the key "U". This
       describes enriching one isotope of a naturally-occurring
       element relative to the rest of that element's natural
       isotopes. Physically: natural uranium is ~0.72% U235; the
       example above says "this uranium has been enriched to 3.2 wo%
       U235," i.e. standard LWR reactor-grade fuel. This general
       enrichment procedure only makes physical sense for elements
       composed of exactly two naturally-occurring isotopes (e.g. U,
       Li, B) — OpenMC itself only supports it for that case.

    Fields left as bare None (no enrichment) mean "add this element at
    its natural isotopic abundance" — e.g. {"percent": 2.0} under "O"
    means natural oxygen (~99.76% O16, ~0.04% O17, ~0.20% O18),
    expanded by OpenMC internally.

    Note: this class inherits PaceObject (unlike GPose, which is a bare
    value object with no validation hook) specifically so its
    all-or-none enrichment-field invariant gets validated automatically
    via PaceObject's __post_init__ -> self.validate() wiring, rather
    than relying on every call site to remember to check it.
    """

    percent: float = field(metadata={"constraint": Constraint.POSITIVE})
    enrichment: float | None = field(
        metadata={"constraint": Constraint.POSITIVE, "range": (0, 100)}, default=None
    )
    enrichment_target: str | None = None
    enrichment_type: PercentType | None = None

    def validate(self):
        validate_fields(self)
        if self.enrichment_type is not None and not isinstance(
            self.enrichment_type, PercentType
        ):
            raise TypeError(
                f"enrichment_type must be a PercentType, got {self.enrichment_type!r}"
            )
        self._validate_enrichment_fields()

    def to_dict(self) -> dict:
        return {
            "percent": self.percent,
            "enrichment": self.enrichment,
            "enrichment_target": self.enrichment_target,
            "enrichment_type": (
                self.enrichment_type.value if self.enrichment_type is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> MaterialComponentEntry:
        return cls(
            percent=data["percent"],
            enrichment=data.get("enrichment"),
            enrichment_target=data.get("enrichment_target"),
            enrichment_type=(
                PercentType(data["enrichment_type"])
                if data.get("enrichment_type") is not None
                else None
            ),
        )

    def _validate_enrichment_fields(self):
        """Enforce the all-or-none enrichment invariant.

        Either all three of enrichment/enrichment_target/enrichment_type
        are set (a fully-specified enrichment entry), or none of them
        are (a bare nuclide or natural-abundance element entry). No
        partial combination is valid — e.g. this rejects OpenMC's own
        "enrichment with no target defaults to U235" shortcut, since
        PACE requires the target to always be explicit for clarity.
        """
        field_checks = [
            self.enrichment is None,
            self.enrichment_target is None,
            self.enrichment_type is None,
        ]

        if any(field_checks) and not all(field_checks):
            raise ValueError(
                "MaterialComponentEntry initialization must involve either "
                "all or none of the enrichment fields."
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class Material(Versioned[MaterialID]):
    """
    Shared base for concrete composition types (MIsotopic, MMixture,
    MVoid).

    Identity/lineage (id/family_name/version_label/derived_from/
    simulation_job_id/user_edit, build_id(), create(), the three-state
    lineage rule) are inherited from Versioned — see that class's
    docstring. This class adds composition-related concerns: the
    type-tag dispatch mechanism, thermal-scattering tables, and
    MOOSE-relevant thermomechanical properties.

    Notably absent from this class: temperature. Composition and
    temperature are physically orthogonal in OpenMC's own model — the
    same nuclide inventory behaves differently at different
    temperatures only because of Doppler broadening of cross sections,
    not because the material itself has changed. Temperature is
    therefore supplied at translation time (from OperatingConditions,
    or from a run's fields), never stored as a field here.

    density_value on concrete subclasses is the REFERENCE density the
    composition is specified at. For solids it is also the value the
    solvers use (thermal expansion is neglected in v1). For fluids, the
    local density during a coupled run comes from the fluid's equation
    of state at the local temperature and pressure, so the stored value
    is a starting point, not a constant.
    """

    material_type: ClassVar[MaterialType]

    # S(alpha,beta) bound-thermal-scattering tables — required for any
    # moderating material (water, graphite, ...) to translate
    # correctly to OpenMC. Empty for non-moderating materials (fuel,
    # cladding, structural alloys) and always empty for MVoid.
    thermal_scattering: tuple[ThermalScatteringLibrary, ...] = ()

    # MOOSE-relevant thermomechanical properties, each optional since
    # not every material needs every property (a coolant doesn't need
    # elastic_modulus; a pure neutronics-only material may need none
    # of these at all). Units: thermal_conductivity in W/(m*K),
    # specific_heat in J/(kg*K), thermal_expansion_coefficient in 1/K,
    # elastic_modulus in Pa — each as a function of temperature_k.
    thermal_conductivity: TabulatedProperty | None = None
    specific_heat: TabulatedProperty | None = None
    thermal_expansion_coefficient: TabulatedProperty | None = None
    elastic_modulus: TabulatedProperty | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "material_type" in cls.__dict__:
            MATERIAL_TYPE_TO_CLASS[cls.material_type.value] = cls

    @abstractmethod
    def _validate_composition(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Self: ...

    def validate(self) -> None:
        """Check field-level constraints, then identity/lineage (via
        Versioned), then composition, then thermal-scattering-table
        sanity."""
        validate_fields(self)
        super().validate()
        self._validate_composition()
        self._validate_thermal_scattering()

    def _validate_thermal_scattering(self) -> None:
        names = [entry.name for entry in self.thermal_scattering]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate thermal_scattering table name(s) in {names!r}")

    def to_dict(self) -> dict:
        try:
            type_value = self.material_type.value
        except AttributeError:
            raise ValueError(
                f"{type(self).__name__} does not define material_type — "
                "set it as a ClassVar on the subclass"
            ) from None
        return {
            "type": type_value,
            **super().to_dict(),
            "thermal_scattering": [
                entry.to_dict() for entry in self.thermal_scattering
            ],
            "thermal_conductivity": (
                self.thermal_conductivity.to_dict()
                if self.thermal_conductivity is not None
                else None
            ),
            "specific_heat": (
                self.specific_heat.to_dict() if self.specific_heat is not None else None
            ),
            "thermal_expansion_coefficient": (
                self.thermal_expansion_coefficient.to_dict()
                if self.thermal_expansion_coefficient is not None
                else None
            ),
            "elastic_modulus": (
                self.elastic_modulus.to_dict()
                if self.elastic_modulus is not None
                else None
            ),
        }

    @classmethod
    def _material_fields_from_dict(cls, data: dict) -> dict:
        """Shared parsing for every field Material adds on top of
        Versioned — every concrete subclass's from_dict() threads this
        through via **cls._material_fields_from_dict(data) rather than
        repeating the same parsing block three times."""
        return {
            **cls._base_fields_from_dict(data),
            "thermal_scattering": tuple(
                ThermalScatteringLibrary.from_dict(entry)
                for entry in data.get("thermal_scattering", [])
            ),
            "thermal_conductivity": (
                TabulatedProperty.from_dict(data["thermal_conductivity"])
                if data.get("thermal_conductivity") is not None
                else None
            ),
            "specific_heat": (
                TabulatedProperty.from_dict(data["specific_heat"])
                if data.get("specific_heat") is not None
                else None
            ),
            "thermal_expansion_coefficient": (
                TabulatedProperty.from_dict(data["thermal_expansion_coefficient"])
                if data.get("thermal_expansion_coefficient") is not None
                else None
            ),
            "elastic_modulus": (
                TabulatedProperty.from_dict(data["elastic_modulus"])
                if data.get("elastic_modulus") is not None
                else None
            ),
        }


@dataclass(frozen=True, kw_only=True, eq=False)
class MIsotopic(Material):
    """
    Direct nuclide/element composition — the common case, and the only
    form simulation-derived versions can take.

    Physically: this is "what is this material actually made of" —
    a set of nuclides/elements (see MaterialComponentEntry for the two
    forms an entry can take), a total density, and which convention
    (atomic or weight percent) the component weights use.

    Example — 3.2 wo% enriched UO2 fuel, atom-percent stoichiometry,
    density in g/cm3:
        MIsotopic.create(
            family_name="uranium3.2_uo2",
            version_label="1",
            components={
                "U": MaterialComponentEntry(
                    percent=1.0,
                    enrichment=3.2,
                    enrichment_target="U235",
                    enrichment_type=PercentType.WO,
                ),
                "O": MaterialComponentEntry(percent=2.0),
            },
            percent_type=PercentType.AO,
            density_value=10.3,
            density_unit=DensityUnit.G_PER_CM3,
        )
    Reading this: one uranium atom for every two oxygen atoms (the
    UO2 stoichiometry, "ao"), with the uranium enriched to 3.2 wo%
    U235 — i.e. standard reactor-grade fuel — at a total density of
    10.3 g/cm3.
    """

    material_type: ClassVar[MaterialType] = MaterialType.ISOTOPIC
    components: dict[str, MaterialComponentEntry]
    percent_type: PercentType
    density_value: float = field(metadata={"constraint": Constraint.POSITIVE})
    density_unit: DensityUnit

    def _validate_composition(self) -> None:
        """Enforce the two composition-level physical rules:

        Rules enforced:
            - a simulation-derived (v2+) version's components must all be
                  exact nuclide fractions — depletion (the Bateman
                  equation) produces isotope-by-isotope densities, never
                  element-level enrichment shorthand, so any enrichment
                  field on a v2+ entry indicates an inconsistent/
                  incorrectly-constructed version.
            - an enrichment-format entry's key must be a bare element
                  symbol (no digits), never a specific nuclide — you
                  enrich an element's isotope mix, you don't "enrich" an
                  already-exact isotope.
        """
        if not isinstance(self.percent_type, PercentType):
            raise TypeError(
                f"percent_type must be a PercentType, got {self.percent_type!r}"
            )
        if not isinstance(self.density_unit, DensityUnit):
            raise TypeError(
                f"density_unit must be a DensityUnit, got {self.density_unit!r}"
            )
        if not self.components:
            raise ValueError("MIsotopic requires at least one component")

        for name, entry in self.components.items():
            if entry.enrichment is not None:
                if self.simulation_job_id is not None:
                    raise ValueError(
                        f"component '{name}' uses enrichment format, which "
                        "is not valid on a simulation-derived (v2+) Material "
                        "— depletion output must be exact nuclide fractions"
                    )
                if any(char.isdigit() for char in name):
                    raise ValueError(
                        f"component '{name}' is a nuclide but has an enrichment-format "
                        "entry. Only bare elements keys can map to enrichment format entries"
                    )

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "percent_type": self.percent_type.value,
            "density_value": self.density_value,
            "density_unit": self.density_unit.value,
            "components": {
                name: entry.to_dict() for name, entry in self.components.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> MIsotopic:
        return cls(
            **cls._material_fields_from_dict(data),
            percent_type=PercentType(data["percent_type"]),
            density_value=data["density_value"],
            density_unit=DensityUnit(data["density_unit"]),
            components={
                name: MaterialComponentEntry.from_dict(v)
                for name, v in data["components"].items()
            },
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class MMixture(Material):
    """
    A material defined as a weighted mix of other, already-defined
    Materials — e.g. homogenizing several distinct materials (fuel,
    cladding, coolant, structural filler) into one effective material
    for a simplified/coarser model region.

    Mixing is its own Material subclass storing references + fractions
    — mirrors GAddition storing list[tuple[GeometryID, GPose]] rather
    than pre-flattening geometry at construction time. The actual
    nuclide-level combination happens in the solver adapter, not here —
    this class only records the recipe.

    percent_type here shares OpenMC's mix_materials() vocabulary
    ("ao"/"wo"), but carries a DIFFERENT constraint than MIsotopic's
    percent_type: mixing combines whole, already-normalized materials
    into a shared volume, so each fraction must be a true proportion
    of the resulting mixture — bounded (0, 1) and summing to exactly
    1 — rather than an unbounded relative weight. This mirrors OpenMC's
    own mix_materials(), which hard-errors on ao/wo fractions that
    don't sum to 1. OpenMC's "vo" (volume-fraction) mixing is not
    supported.

    Example — a 70/30 atom-fraction mix of two previously-defined
    material versions:
        MMixture.create(
            family_name="fuel_filler_blend",
            version_label="1",
            components=[
                (MaterialID("fuel_v1-1"), 0.7),
                (MaterialID("filler_v1-1"), 0.3),
            ],
            percent_type=PercentType.AO,
        )
    Note: the ids inside `components` reference OTHER, already-existing
    Materials — unaffected by this class's own identity scheme.
    """

    material_type: ClassVar[MaterialType] = MaterialType.MIXTURE
    components: list[tuple[MaterialID, float]]
    percent_type: PercentType

    def _validate_composition(self) -> None:
        """Enforce the mixture-fraction physical rules:

        Rules enforced:
            - at least 2 constituent materials — a "mixture" of one
                  material isn't a mixture.
            - every fraction strictly between 0 and 1
            - fractions sum to exactly 1 (within floating-point
                  tolerance) — mixing combines 100% of a shared volume;
                  no void/remainder concept is supported yet.
        """
        if not isinstance(self.percent_type, PercentType):
            raise TypeError(
                f"percent_type must be a PercentType, got {self.percent_type!r}"
            )
        if len(self.components) < 2:
            raise ValueError("MMixture requires at least 2 constituent materials")

        for material_id, fraction in self.components:
            if not (0 < fraction < 1):
                raise ValueError(
                    f"mix fraction for {material_id!r} must be in (0, 1), "
                    f"got {fraction}"
                )

        total = sum([fraction for _, fraction in self.components])
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"mix fractions must sum to 1, got {total}")

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "percent_type": self.percent_type.value,
            "components": [
                [material_id, fraction] for material_id, fraction in self.components
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MMixture:
        return cls(
            **cls._material_fields_from_dict(data),
            percent_type=PercentType(data["percent_type"]),
            components=[
                (MaterialID(material_id), fraction)
                for material_id, fraction in data["components"]
            ],
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class MVoid(Material):
    """
    No material at all — an empty region, e.g. a fuel rod's gas
    plenum, a gap deliberately left unfilled, or any space a solver
    should treat as having zero interaction probability.

    Physically distinct from a sparse/low-density MIsotopic: void
    means zero cross sections, not small ones — a particle passes
    through completely unimpeded, which is a different physical
    statement than "very unlikely to interact." This can't be
    approximated by giving MIsotopic a near-zero density: its
    components dict requires at least one entry and density_value is
    constrained strictly positive (Constraint.POSITIVE), by design —
    relaxing either to let a "fake void" through would corrupt the
    guarantee every other consumer of MIsotopic currently relies on
    (that a real MIsotopic always has genuine, physical mass density).

    No fields — there is nothing to compose or configure.

    Example:
        MVoid.create(family_name="plenum_void", version_label="1")
    """

    material_type: ClassVar[MaterialType] = MaterialType.VOID

    def _validate_composition(self) -> None:
        # nothing to validate — void has no composition
        pass

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> MVoid:
        return cls(
            **cls._material_fields_from_dict(data),
        )


def material_from_dict(data: dict) -> Material:
    """Reconstruct the correct concrete Material subclass from a dict
    produced by any subclass's to_dict() — dispatches on the "type"
    key rather than requiring the caller to already know which
    concrete class they're deserializing. Material.from_dict() is
    abstract, so it can never be called directly on the base class;
    this is the supported way to deserialize a Material of unknown
    concrete type."""
    cls = MATERIAL_TYPE_TO_CLASS.get(data["type"])
    if cls is None:
        raise ValueError(f"unknown material type: {data['type']!r}")
    return cls.from_dict(data)
