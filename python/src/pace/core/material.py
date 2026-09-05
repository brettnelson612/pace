"""
core/material.py

Material modeling objects for PACE, mirroring the Geometry/GeometryVersion
split: Material is bare identity (registry "materials"); MaterialVersion is
the versioned, polymorphic class holding actual composition data.

What "material" means physically here: a material is a description of what
substance occupies a region of space — which isotopes are present, in what
relative amounts, and at what density. It says nothing about shape (that's
Geometry) or temperature (deliberately excluded — see MaterialVersion).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import GTRunID, MaterialID, MaterialVersionID
from pace.core.pace_object import PaceObject

"""
Percentage Types:
    - ao: atomic percent. The number of atoms of a specific
          nuclide/element divided by the total number of atoms in the
          material. This is the natural unit for expressing chemical/
          stoichiometric ratios — e.g. UO2's 1:2 uranium-to-oxygen atom
          ratio is exact and unambiguous in "ao" terms.
    - wo: weight percent. The mass of a specific nuclide/element
          divided by the total mass of the material. Natural for
          compositions given by measured or spec'd mass fractions —
          e.g. a structural alloy's constituent metals by weight.

Note: percentages here (on MIsotopic) are RELATIVE WEIGHTS, not
absolute percentages that must sum to 100. OpenMC normalizes them
internally against the material's separately-specified density.
Example: {"U": 1.0, "O": 2.0} under "ao" means a 1:2 atom ratio
(UO2's stoichiometry), not "1% U, 2% O with 97% unaccounted for."

percent_type carries a DIFFERENT constraint on MMixture (see that
class) despite sharing the same name and vocabulary — this mirrors
OpenMC's own choice to reuse percent_type identically across
add_element()/add_nuclide() and mix_materials(), even though the
accompanying values are constrained differently in each case.
"""
PercentType = Literal["ao", "wo"]

"""
Density Units:
    - g/cm3: grams per cubic centimeter. The common general-purpose
          density unit.
    - kg/m3: kilograms per cubic meter.
    - atom/b-cm: atoms per barn-centimeter — atomic number density
          expressed in units directly compatible with microscopic
          cross sections (a barn is 10^-24 cm^2, the standard unit
          nuclear cross sections are quoted in). Expressing density
          this way means the macroscopic cross section is just
          N * sigma with no unit conversion — a convenience for the
          transport-physics side of the pipeline, not a general
          density unit. Only worth using if you already have
          atom-density numbers in this form (e.g. from a cross-section
          library); otherwise prefer g/cm3 with a components dict.
"""
DensityUnit = Literal["g/cm3", "kg/m3", "atom/b-cm"]


class MaterialType(str, Enum):
    """Discriminator tag for MaterialVersion subclasses (not yet wired
    into any dispatch/deserialization logic — mirrors GeometryType's
    current status).

    Values:
        - isotopic: a direct composition of nuclides/elements
              (MIsotopic) — "this material is made of these isotopes,
              in these amounts."
        - mixture: a weighted combination of other, already-defined
              MaterialVersions (MMixture) — "this material is X% of
              material A plus Y% of material B."
    """

    ISOTOPIC = "isotopic"
    MIXTURE = "mixture"


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
       expansion needed. This is the ONLY form a GT-run-derived (v2+)
       composition can take — depletion output is always exact
       per-isotope densities (see MaterialVersion and
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

    Note: this class inherits PaceObject (unlike GPos, which is a bare
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
        self._validate_enrichment_fields()

    def to_dict(self) -> dict:
        return {
            "percent": self.percent,
            "enrichment": self.enrichment,
            "enrichment_target": self.enrichment_target,
            "enrichment_type": self.enrichment_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MaterialComponentEntry:
        return cls(
            percent=data["percent"],
            enrichment=data.get("enrichment"),
            enrichment_target=data.get("enrichment_target"),
            enrichment_type=data.get("enrichment_type"),
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


@dataclass(frozen=True, kw_only=True)
class Material(PaceObject):
    """Bare identity for a material — no composition data.

    Mirrors Geometry: a Material is just an ID plus the list of
    MaterialVersions that have ever been recorded under it. What the
    material actually consists of physically lives entirely on its
    MaterialVersions (see MIsotopic, MMixture) — this class exists so
    a material (e.g. "the fuel") can be referenced stably across many
    versions of its composition over time (e.g. as it depletes).
    """

    id: MaterialID
    version_ids: list[MaterialVersionID] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.id, "version_ids": list(self.version_ids)}

    @classmethod
    def from_dict(cls, data: dict) -> Material:
        return cls(id=data["id"], version_ids=list(data.get("version_ids", [])))


@dataclass(frozen=True, kw_only=True)
class MaterialVersion(PaceObject):
    """
    Shared base for concrete composition types (MIsotopic, MMixture).

    Notably absent from this class: temperature. Composition and
    temperature are physically orthogonal in OpenMC's own model — the
    same nuclide inventory behaves differently at different
    temperatures only because of Doppler broadening of cross sections,
    not because the material itself has changed. Temperature is
    therefore passed as an argument to to_open_mc()/to_moose() at
    solver-translation time, never stored as a field here.

    Versioning rule (identical to GeometryVersion): v1 is always
    user-authored (derived_from and gt_run_id both None) — a
    composition a person specified directly. v2+ can only be created
    as the recorded output of a GT run (both fields set together) —
    physically, this means composition change via depletion: OpenMC's
    depletion module solves the Bateman equations to evolve a nuclide
    inventory forward under a flux/power history, and that evolved
    inventory becomes a new MaterialVersion linked back to the GT run
    that produced it. One of derived_from/gt_run_id set without the
    other is invalid — there's no physical mechanism that produces a
    "half-derived" version.
    """

    id: MaterialVersionID
    material_id: MaterialID
    derived_from: MaterialVersionID | None = None
    gt_run_id: GTRunID | None = None

    @abstractmethod
    def _validate_composition(self) -> None:
        pass

    @abstractmethod
    def to_open_mc(self, temperature_k: float | None = None):
        """Temperature is passed at call time, never stored on the version."""

    @abstractmethod
    def to_moose(self, temperature_k: float | None = None):
        pass

    def validate(self) -> None:
        validate_fields(self)
        self._validate_lineage()
        self._validate_composition()

    def _validate_lineage(self) -> None:
        """Check that derived_from and gt_run_id are set together or
        both None — see the versioning rule described in this class's
        docstring for the physical reasoning."""
        has_predecessor = self.derived_from is not None
        has_gt_run = self.gt_run_id is not None
        if has_predecessor != has_gt_run:
            raise ValueError(
                "derived_from and gt_run_id must be set together or both "
                "None (got derived_from="
                f"{self.derived_from!r}, gt_run_id={self.gt_run_id!r})"
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class MIsotopic(MaterialVersion):
    """
    Direct nuclide/element composition — the common case, and the only
    form v2+ (depletion-derived) versions can take.

    Physically: this is "what is this material actually made of" —
    a set of nuclides/elements (see MaterialComponentEntry for the two
    forms an entry can take), a total density, and which convention
    (atomic or weight percent) the component weights use.

    Example — 3.2 wo% enriched UO2 fuel, atom-percent stoichiometry,
    density in g/cm3:
        MIsotopic(
            id=...,
            material_id=...,
            components={
                "U": MaterialComponentEntry(
                    percent=1.0,
                    enrichment=3.2,
                    enrichment_target="U235",
                    enrichment_type="wo",
                ),
                "O": MaterialComponentEntry(percent=2.0),
            },
            percent_type="ao",
            density_value=10.3,
            density_unit="g/cm3",
        )
    Reading this: one uranium atom for every two oxygen atoms (the
    UO2 stoichiometry, "ao"), with the uranium enriched to 3.2 wo%
    U235 — i.e. standard reactor-grade fuel — at a total density of
    10.3 g/cm3.

    eq=False / identity-based equality: `components` is a dict
    (unhashable, and not meaningfully comparable by value for a
    frozen-dataclass default __eq__ the way scalar-only
    GeometryVersion subclasses are) — same reasoning as
    GAddition/GSubtraction using identity-based __eq__/__hash__ rather
    than the field-based default.
    """

    components: dict[str, MaterialComponentEntry]
    percent_type: PercentType
    density_value: float = field(metadata={"constraint": Constraint.POSITIVE})
    density_unit: DensityUnit

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def _validate_composition(self) -> None:
        """Enforce the two composition-level physical rules:

        Rules enforced:
            - a GT-run-derived (v2+) version's components must all be
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
        if not self.components:
            raise ValueError("MIsotopic requires at least one component")

        for name, entry in self.components.items():
            if entry.enrichment is not None:
                if self.gt_run_id is not None:
                    raise ValueError(
                        f"component '{name}' uses enrichment format, which "
                        "is not valid on a GT-run-derived (v2+) MaterialVersion "
                        "— depletion output must be exact nuclide fractions"
                    )
                if any(char.isdigit() for char in name):
                    raise ValueError(
                        f"component '{name}' is a nuclide but has an enrichment-format "
                        "entry. Only bare elements keys can map to enrichment format entries"
                    )

    def to_open_mc(self, temperature_k: float | None = None):
        # TODO: build an openmc.Material, call add_components(self.components,
        # percent_type=self.percent_type), set_density(self.density_unit,
        # self.density_value), and apply temperature_k if provided.
        raise NotImplementedError

    def to_moose(self, temperature_k: float | None = None):
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "type": MaterialType.ISOTOPIC.value,
            "id": self.id,
            "material_id": self.material_id,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "percent_type": self.percent_type,
            "density_value": self.density_value,
            "density_unit": self.density_unit,
            "components": {
                name: entry.to_dict() for name, entry in self.components.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> MIsotopic:
        return cls(
            id=data["id"],
            material_id=data["material_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            percent_type=data["percent_type"],
            density_value=data["density_value"],
            density_unit=data["density_unit"],
            components={
                name: MaterialComponentEntry.from_dict(v)
                for name, v in data["components"].items()
            },
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class MMixture(MaterialVersion):
    """
    A material defined as a weighted mix of other, already-defined
    MaterialVersions — e.g. homogenizing several distinct materials
    (fuel, cladding, coolant, structural filler) into one effective
    material for a simplified/coarser model region.

    Answers the earlier "mix_materials()" design question: rather than
    a method on Material or a standalone helper function, mixing is
    its own MaterialVersion subclass storing references + fractions —
    mirrors GAddition storing list[tuple[GeometryVersionID, GPos]]
    rather than pre-flattening geometry at construction time. The
    actual nuclide-level combination happens in to_open_mc(), not
    here — this class only records the recipe.

    percent_type here shares OpenMC's mix_materials() vocabulary
    ("ao"/"wo"), but carries a DIFFERENT constraint than MIsotopic's
    percent_type: mixing combines whole, already-normalized materials
    into a shared volume, so each fraction must be a true proportion
    of the resulting mixture — bounded (0, 1) and summing to exactly
    1 — rather than an unbounded relative weight. This mirrors OpenMC's
    own mix_materials(), which hard-errors on ao/wo fractions that
    don't sum to 1 (OpenMC additionally supports "vo"/volume-fraction
    mixing with an implicit void remainder — deliberately not
    supported here yet, to avoid introducing a void-material concept
    before it's needed).

    Example — a 70/30 atom-fraction mix of two previously-defined
    material versions:
        MMixture(
            id=...,
            material_id=...,
            components=[
                (MaterialVersionID("fuel_v1"), 0.7),
                (MaterialVersionID("filler_v1"), 0.3),
            ],
            percent_type="ao",
        )

    eq=False for the same reason as MIsotopic: `components` holds a
    list, not meaningfully comparable via the frozen-dataclass
    default.
    """

    components: list[tuple[MaterialVersionID, float]]
    percent_type: PercentType

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def _validate_composition(self) -> None:
        """Enforce the mixture-fraction physical rules:

        Rules enforced:
            - at least 2 constituent materials — a "mixture" of one
                  material isn't a mixture.
            - every fraction strictly between 0 and 1 — a fraction of
                  0 or 1 means that constituent isn't really part of a
                  mix.
            - fractions sum to exactly 1 (within floating-point
                  tolerance) — mixing combines 100% of a shared volume;
                  no void/remainder concept is supported yet.
        """
        if len(self.components) < 2:
            raise ValueError("MMixture requires at least 2 constituent materials")

        for material_version_id, fraction in self.components:
            if not (0 < fraction < 1):
                raise ValueError(
                    f"mix fraction for {material_version_id!r} must be in (0, 1), "
                    f"got {fraction}"
                )

        total = sum([fraction for _, fraction in self.components])
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"mix fractions must sum to 1, got {total}")

    def to_open_mc(self, temperature_k: float | None = None):
        # TODO: resolve each constituent MaterialVersion, build/mix the
        # corresponding openmc.Material objects per self.percent_type, apply
        # temperature_k.
        raise NotImplementedError

    def to_moose(self, temperature_k: float | None = None):
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {
            "type": MaterialType.MIXTURE.value,
            "id": self.id,
            "material_id": self.material_id,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "percent_type": self.percent_type,
            "components": [[mvid, fraction] for mvid, fraction in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MMixture:
        return cls(
            id=data["id"],
            material_id=data["material_id"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            percent_type=data["percent_type"],
            components=[(mvid, fraction) for mvid, fraction in data["components"]],
        )
