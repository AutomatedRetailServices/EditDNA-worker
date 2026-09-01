"""General semantic-atom importance classification -- D-031.

RAW 33402023395 exposed a real CoverageLedger over-conservatism: a discarded
clip's incidental year ("...en 2023.") was flagged as a "lost critical
atom" and blocked Selection Freeze, even though the Human Gold oracle
itself does not preserve that year in its own equivalent delivery -- the
audience-facing idea (endoscopy -> gastritis diagnosis -> medication) is
fully intact without it. That is evidence CoverageLedger's OLD rule --
"any missing number/negation atom blocks, unconditionally" -- was too
blunt, not evidence Selection damaged the story.

This module generalizes that rule from

    UNIQUE ATOM LOST = BLOCK

to

    CRITICAL SEMANTIC ATOM LOST = BLOCK
    CONTEXTUAL ATOM LOST = WARN / OBSERVE (does not block by itself)
    UNCERTAIN ATOM LOST = still blocks (WHEN UNCERTAIN, KEEP)

No Video00 fact, phrase, or literal value ("2023" included) is hardcoded
anywhere below -- every rule here is a general marker-vocabulary or
structural check that would fire identically on any subject matter.

## Classes

CRITICAL: a negation (always -- flips a claim's truth value outright);
a percentage, price, measurement, or dose (a unit/currency/percent marker
present in the clip's own text); or a number whose clause contains
correction language ("instead of", "en vez de", "actually", "corrijo", ...
-- exactly the shape of the canonical directive's own example, "diagnosed
in 2023 instead of 2022").

CONTEXTUAL: a bare, plausible-year-shaped number (1900-2099) appearing in
an ordinary temporal-aside clause ("during", "en", "durante", "temporada",
...) with no correction, unit, currency, percent, or dose marker anywhere
in the same clip's text -- the canonical directive's own "during one
period in 2023 I had stomach problems" example.

UNCERTAIN: anything that does not match a CRITICAL or CONTEXTUAL rule
above -- e.g. a bare quantity with no unit/currency/percent context, or a
number this deterministic layer cannot confidently place. `blocks_freeze`
treats UNCERTAIN exactly like CRITICAL: "WHEN UNCERTAIN, KEEP" means an
atom this layer cannot confidently clear as safe-to-lose stays in the
blocking lane, never silently downgraded to a warning. The bounded
`SemanticAtomImportanceArbiter` below is where a real implementation would
resolve UNCERTAIN with the smallest necessary context (the clip's own
sentence plus the final KEEP text) -- no implementation exists yet in this
codebase (same honest-gap pattern as `CausalOrderArbiter` in
`causal_order_validator.py`); every caller defaults it to `None`, which
leaves UNCERTAIN atoms blocking, exactly as intended.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

CRITICAL = "CRITICAL"
CONTEXTUAL = "CONTEXTUAL"
UNCERTAIN = "UNCERTAIN"

_VALID_IMPORTANCE = frozenset({CRITICAL, CONTEXTUAL, UNCERTAIN})

# General (English + Spanish) marker vocabulary. Matched as a plain
# substring of the clip's own text (casefolded) -- deliberately coarse
# (whole-clause presence, not token-adjacency) to stay robust and simple,
# matching this codebase's existing lexical-marker style (e.g.
# causal_order_validator.py's connector lexicon). No Video00-specific
# fact, disease, product, or phrase appears in any list below.
_PERCENT_MARKERS = ("%", "por ciento", "porciento", "percent")
_CURRENCY_MARKERS = (
    "$", "usd", "dolar", "dólar", "dolares", "dólares",
    "peso", "pesos", "euro", "euros", "precio", "price", "cost", "costo",
)
_MEASUREMENT_UNIT_MARKERS = (
    "cm", "centimetro", "centímetro", "centimetros", "centímetros",
    "mm", "milimetro", "milímetro", "milimetros", "milímetros",
    "kg", "kilo", "kilos", "kilogramo", "kilogramos",
    "lb", "lbs", "libra", "libras", "pound", "pounds",
    "mg", "miligramo", "miligramos",
    "ml", "mililitro", "mililitros",
    "oz", "onza", "onzas", "ounce", "ounces",
    "pulgada", "pulgadas", "inch", "inches",
    "metro", "metros", "meter", "meters",
    "litro", "litros", "liter", "liters",
    "grado", "grados", "degree", "degrees",
)
_DOSE_MARKERS = (
    "pastilla", "pastillas", "pill", "pills", "dosis", "dose",
    "tableta", "tabletas", "capsula", "cápsula", "capsulas", "cápsulas",
    "veces al dia", "veces al día", "times a day", "times daily",
)
_CORRECTION_MARKERS = (
    "instead of", "en vez de", "en lugar de", "no fue", "actually",
    "en realidad", "realmente fue", "changed from", "cambio de",
    "cambió de", "corrijo", "correccion", "corrección", "correction",
    "me equivoque", "me equivoqué", "wait no", "espera no", "digo,",
)
# A general chronology-relation marker: the clause explicitly relates this
# atom's time to another event ("before/after that happened", "since/until
# X") -- unlike a bare temporal aside, removing the atom here can change
# WHICH event came first, i.e. required chronology, per the canonical
# directive's own "chronology when chronology affects meaning" criterion.
# Deliberately multi-word phrases, not bare "before"/"after"/"since" --
# those single words are common enough as ordinary filler (e.g. "before it
# worked") to false-positive on sentences with no real chronology relation.
_CHRONOLOGY_RELATION_MARKERS = (
    "before that", "after that", "since then", "until then",
    "before this", "after this", "before i", "after i", "before we", "after we",
    "antes de eso", "despues de eso", "después de eso", "desde entonces",
    "hasta entonces", "antes de que", "despues de que", "después de que",
    "once i", "una vez que",
)
_TEMPORAL_ASIDE_MARKERS = (
    "durante", "temporada", "during", "period", " en ", "in ", "por ",
    "around", "sobre", "hacia",
)

_YEAR_MIN, _YEAR_MAX = 1900, 2099


def _clause_has_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = f" {(text or '').casefold()} "
    return any(marker in lowered for marker in markers)


def _looks_like_year(token: str) -> bool:
    if not token.isdigit() or len(token) != 4:
        return False
    return _YEAR_MIN <= int(token) <= _YEAR_MAX


@dataclass(frozen=True)
class AtomImportance:
    atom: str
    atom_type: str  # "number" | "negation"
    importance: str  # CRITICAL | CONTEXTUAL | UNCERTAIN
    evidence: str
    resolved_by: str  # "deterministic" | "semantic_arbiter"


class SemanticAtomImportanceArbiter(Protocol):
    """Bounded arbiter for exactly one narrow question per atom: "would
    removing this atom materially change the speaker's audience-facing
    claim, factual meaning, causal meaning, or required chronology?" Given
    the atom, its own source sentence, and the final KEEP text as minimal
    context -- mirrors this codebase's other bounded arbiters (text/context
    only, no clip identity). Returns (importance in {CRITICAL, CONTEXTUAL,
    UNCERTAIN}, confidence 0..1, short general reason)."""

    def classify_atom(self, atom_text: str, source_sentence: str, kept_text: str) -> tuple[str, float, str]: ...


def classify_negation_atom(atom: str) -> AtomImportance:
    """A negation always flips a claim's truth value -- never contextual."""
    return AtomImportance(atom, "negation", CRITICAL, "negation_changes_truth_value", "deterministic")


def classify_number_atom(atom: str, source_text: str) -> AtomImportance:
    """Deterministic classification for one missing numeric atom, using
    only its own clip's text as context (see module docstring for the
    exact rules and why each is general, not Video00-specific)."""
    if _clause_has_any(source_text, _PERCENT_MARKERS):
        return AtomImportance(atom, "number", CRITICAL, "percentage", "deterministic")
    if _clause_has_any(source_text, _CURRENCY_MARKERS):
        return AtomImportance(atom, "number", CRITICAL, "price", "deterministic")
    if _clause_has_any(source_text, _MEASUREMENT_UNIT_MARKERS):
        return AtomImportance(atom, "number", CRITICAL, "measurement", "deterministic")
    if _clause_has_any(source_text, _DOSE_MARKERS):
        return AtomImportance(atom, "number", CRITICAL, "dose_or_quantity", "deterministic")
    if _clause_has_any(source_text, _CORRECTION_MARKERS):
        return AtomImportance(atom, "number", CRITICAL, "correction_language_present", "deterministic")
    if _clause_has_any(source_text, _CHRONOLOGY_RELATION_MARKERS):
        return AtomImportance(atom, "number", CRITICAL, "chronology_relation_language_present", "deterministic")
    if _looks_like_year(atom) and _clause_has_any(source_text, _TEMPORAL_ASIDE_MARKERS):
        return AtomImportance(atom, "number", CONTEXTUAL, "incidental_year_in_ordinary_temporal_aside", "deterministic")
    return AtomImportance(atom, "number", UNCERTAIN, "no_deterministic_signal_found", "deterministic")


def resolve_uncertain_with_arbiter(
    classifications: list[AtomImportance],
    *,
    source_text: str,
    kept_text: str,
    arbiter: SemanticAtomImportanceArbiter | None,
) -> list[AtomImportance]:
    """Escalate only UNCERTAIN atoms to the bounded arbiter. A CRITICAL or
    CONTEXTUAL deterministic verdict is never second-guessed. No arbiter,
    an arbiter exception, or a malformed verdict all leave the atom exactly
    as UNCERTAIN -- fails open toward the SAFE (blocking) side, never
    toward silently clearing an atom as contextual without evidence."""
    if arbiter is None:
        return classifications
    resolved: list[AtomImportance] = []
    for item in classifications:
        if item.importance != UNCERTAIN:
            resolved.append(item)
            continue
        try:
            verdict, _confidence, reason = arbiter.classify_atom(item.atom, source_text, kept_text)
        except Exception:
            resolved.append(item)
            continue
        verdict = str(verdict or "").upper()
        if verdict not in _VALID_IMPORTANCE:
            resolved.append(item)
            continue
        resolved.append(AtomImportance(
            atom=item.atom, atom_type=item.atom_type, importance=verdict,
            evidence=f"semantic_arbiter:{str(reason)[:160]}", resolved_by="semantic_arbiter",
        ))
    return resolved


def blocks_freeze(importance: str) -> bool:
    """CRITICAL and UNCERTAIN both block Selection Freeze -- "WHEN
    UNCERTAIN, KEEP" means an atom this layer cannot confidently clear as
    safe-to-lose is never treated as merely a warning. Only a confidently
    CONTEXTUAL verdict does not block by itself."""
    return importance in (CRITICAL, UNCERTAIN)
