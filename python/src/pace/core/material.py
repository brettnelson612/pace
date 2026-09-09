"""
core/material.py

Material domain objects, mirroring geometry.py's structure: no separate
bare "Material" identity class/table — a version's family is just a
`family_name` string carried on the version itself. A version's id is
one opaque string built from family_name + version_label (e.g.
"uranium3.2-1", "uranium3.2-6month_depletion" — see
MaterialVersion.build_id()).

family_name is NOT unique per row — every version in a family shares
it. Rejecting a duplicate family_name is enforced only at "create a
brand-new family" (v1, derived_from=None) time, and that's application
logic (ComponentService), not a database constraint.

What "material" means physically here: a material is a description of
what substance occupies a region of space — which isotopes are
present, in what relative amounts, and at what density. It says
nothing about shape (that's Geometry) or temperature (deliberately
excluded — see MaterialVersion).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pace.core.constraints import Constraint, validate_fields
from pace.core.ids import GTRunID, MaterialVersionID
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
    """Discriminator tag for MaterialVersion subclasses.

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
class MaterialVersion(PaceObject):
    """
    Shared base for concrete composition types (MIsotopic, MMixture).

    Identity: id is a single opaque string built from family_name +
    version_label via build_id() — e.g. family_name="uranium3.2",
    version_label="1" -> id="uranium3.2-1". _validate_id() confirms
    the two stay consistent, catching an accidentally mismatched/
    copy-pasted id at construction time.

    Notably absent from this class: temperature. Composition and
    temperature are physically orthogonal in OpenMC's own model — the
    same nuclide inventory behaves differently at different
    temperatures only because of Doppler broadening of cross sections,
    not because the material itself has changed. Temperature is
    therefore passed as an argument to to_open_mc()/to_moose() at
    solver-translation time, never stored as a field here.

    Versioning rule: a version is either v1 (user-authored from
    scratch) or a derived version, and if derived, it has exactly one
    recorded cause:
        - derived_from is None: version 1. gt_run_id must be None and
              user_edit must be False.
        - derived_from is set: a derived version. Exactly one of
              gt_run_id (this version is the recorded output of a GT
              run — physically, depletion: OpenMC's depletion module
              solving the Bateman equations to evolve a nuclide
              inventory forward under a flux/power history) or
              user_edit (a person directly edited a predecessor
              version's composition) must also be set — never neither,
              and never both.
    """

    id: MaterialVersionID
    family_name: str
    version_label: str
    derived_from: MaterialVersionID | None = None
    gt_run_id: GTRunID | None = None
    user_edit: bool = False

    @staticmethod
    def build_id(family_name: str, version_label: str) -> MaterialVersionID:
        """The one canonical way an id is constructed from a
        family_name + version_label pair. Used both when constructing
        a new version and by _validate_id() to confirm an existing
        id actually matches its own family_name/version_label."""
        return MaterialVersionID(f"{family_name}-{version_label}")

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
        self._validate_id()
        self._validate_lineage()
        self._validate_composition()

    def _validate_id(self) -> None:
        expected = self.build_id(self.family_name, self.version_label)
        if self.id != expected:
            raise ValueError(
                f"id {self.id!r} does not match family_name/version_label "
                f"(expected {expected!r})"
            )

    def _validate_lineage(self) -> None:
        """Check that derived_from and exactly one of gt_run_id/user_edit
        are set together, or all three are unset — see the versioning
        rule described in this class's docstring for the physical
        reasoning."""
        has_predecessor = self.derived_from is not None
        has_gt_run = self.gt_run_id is not None

        if not has_predecessor:
            if has_gt_run or self.user_edit:
                raise ValueError(
                    "gt_run_id must be unset and user_edit must be False "
                    "when derived_from is unset (version 1 has no "
                    "derivation cause)"
                )
        elif has_gt_run == self.user_edit:
            # both set, or both unset — either way, invalid
            raise ValueError(
                "when derived_from is set, exactly one of gt_run_id or "
                "user_edit must also be set (got gt_run_id="
                f"{self.gt_run_id!r}, user_edit={self.user_edit!r})"
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
            id=MaterialVersion.build_id("uranium3.2_uo2", "1"),
            family_name="uranium3.2_uo2",
            version_label="1",
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

    eq=False / id-based equality: `components` is a dict (unhashable,
    and not meaningfully comparable by value for a frozen-dataclass
    default __eq__ the way scalar-only GeometryVersion subclasses
    are) — same reasoning, and same pattern (isinstance check +
    self.id == other.id, hash(self.id)), as GAddition/GSubtraction.
    """

    components: dict[str, MaterialComponentEntry]
    percent_type: PercentType
    density_value: float = field(metadata={"constraint": Constraint.POSITIVE})
    density_unit: DensityUnit

    def __eq__(self, value: object) -> bool:
        return isinstance(value, MIsotopic) and self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

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
            "family_name": self.family_name,
            "version_label": self.version_label,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "user_edit": self.user_edit,
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
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
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
    mirrors GAddition storing list[tuple[GeometryVersionID, GPose]]
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
            id=MaterialVersion.build_id("fuel_filler_blend", "1"),
            family_name="fuel_filler_blend",
            version_label="1",
            components=[
                (MaterialVersionID("fuel_v1-1"), 0.7),
                (MaterialVersionID("filler_v1-1"), 0.3),
            ],
            percent_type="ao",
        )
    Note: the ids inside `components` reference OTHER, already-existing
    MaterialVersions — unaffected by this class's own identity scheme.

    eq=False for the same reason as MIsotopic: `components` holds a
    list, not meaningfully comparable via the frozen-dataclass
    default.
    """

    components: list[tuple[MaterialVersionID, float]]
    percent_type: PercentType

    def __eq__(self, value: object) -> bool:
        return isinstance(value, MMixture) and self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

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
            "family_name": self.family_name,
            "version_label": self.version_label,
            "derived_from": self.derived_from,
            "gt_run_id": self.gt_run_id,
            "user_edit": self.user_edit,
            "percent_type": self.percent_type,
            "components": [[mvid, fraction] for mvid, fraction in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MMixture:
        return cls(
            id=data["id"],
            family_name=data["family_name"],
            version_label=data["version_label"],
            derived_from=data["derived_from"],
            gt_run_id=data["gt_run_id"],
            user_edit=data["user_edit"],
            percent_type=data["percent_type"],
            components=[(mvid, fraction) for mvid, fraction in data["components"]],
        )


# =============================================================================
# Type dispatch — maps MaterialType values to their concrete class, and
# back. Lives here, not in the persistence layer, for the same reason as
# geometry.py's GEOMETRY_TYPE_TO_CLASS/CLASS_TO_GEOMETRY_TYPE: it's a fact
# about this module's own class hierarchy, not about how anything gets
# stored.
#
# Must be kept in sync by hand whenever a new MaterialVersion subclass is
# added — nothing enforces that automatically.
# =============================================================================
MATERIAL_TYPE_TO_CLASS: dict[str, type[MaterialVersion]] = {
    MaterialType.ISOTOPIC.value: MIsotopic,
    MaterialType.MIXTURE.value: MMixture,
}

# Inverse lookup, keyed on exact type (not isinstance) — avoids any
# ambiguity if the class hierarchy ever grows a subclass of a subclass.
CLASS_TO_MATERIAL_TYPE: dict[type[MaterialVersion], str] = {
    cls: type_value for type_value, cls in MATERIAL_TYPE_TO_CLASS.items()
}
