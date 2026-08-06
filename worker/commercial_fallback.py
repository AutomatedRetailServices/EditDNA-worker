"""Deterministic English/Spanish production and commercial fallback rules."""
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from worker.text_normalization import NormalizedText, normalize_match_text, normalized_text

EXACT_STANDALONE_FILLER = {
    "is that good",
    "is that funny",
    "am i saying it right",
    "that one good",
    "why can't i remember",
    "thanks",
    "thank you",
    "thank you guys",
    "okay thanks",
    "all right thank you",
    "alright thank you",
}

END_OF_TAKE_FILLER = {
    "thanks that's it",
    "thank you we're done",
}

EXACT_STANDALONE_PRODUCTION_DIRECTIONS = {
    "hold on",
    "wait a second",
    "wait one second",
    "no restart",
    "wait no",
    "restart",
    "do that again",
    "cut that",
    "corten", "corta", "otra vez", "empecemos de nuevo", "repetimos",
}

LEADING_PRODUCTION_DIRECTIONS = (
    "wait let me",
    "wait i need to",
    "wait no",
    "hold on let me",
    "no restart",
    "let me redo",
    "let me do that again",
    "do that again",
    "redo that",
    "restart that",
    "restart this",
    "let me start again",
    "cut that",
    "espera déjame", "espera necesito", "no otra vez", "vamos de nuevo",
    "déjame repetir", "empecemos de nuevo",
)

SAFE_MULTIWORD_META_PHRASES = ()

TAIL_DEPENDENT_ENDINGS = [
    "as well",
    "too",
    "either",
]

TAIL_DEPENDENT_STARTS = [
    "and",
    "so",
    "but",
    "because",
]


def _normalized_words(text: str) -> List[str]:
    return list(normalized_text(text).tokens)


def has_phrase(n: NormalizedText, phrase: str) -> bool:
    phrase_compact = normalized_text(phrase).compact
    return bool(phrase_compact) and f" {phrase_compact} " in f" {n.compact} "


def starts_phrase(n: NormalizedText, phrase: str) -> bool:
    phrase_compact = normalized_text(phrase).compact
    return bool(phrase_compact) and (n.compact == phrase_compact or n.compact.startswith(phrase_compact + " "))


def has_any_phrase(n: NormalizedText, phrases: Sequence[str]) -> bool:
    return any(has_phrase(n, phrase) for phrase in phrases)


# Heuristic precedence:
# A. invalid/tail/microfragment (handled by boundary/tail validators)
# B. verified filler or production meta
# C. explicit viewer-directed CTA
# D. explicit problem cues
# E. explicit proof cues
# F. explicit benefit cues
# G. explicit product enumeration/features
# H. hook/story/other fallback


def is_camera_rolling_slate(text: str) -> bool:
    n = normalized_text(text)
    if n.compact in {
        "camera rolling",
        "okay camera rolling",
        "camera is rolling",
        "the camera is rolling",
        "rolling",
        "and rolling",
        "we're rolling",
        "cámara grabando",
        "la cámara está grabando",
        "estamos grabando",
    }:
        return True
    return starts_phrase(n, "camera rolling take") or starts_phrase(n, "okay camera rolling take")


TAKE_SLATE_DISCOURSE_PREFIXES = (("all", "right"), ("okay",), ("so",))
TAKE_SLATE_INTRO_PREFIXES = (("this", "is"), ("that's",), ("we're", "on"))
TAKE_SLATE_NUMBERS = {"two", "three", "2", "3"}


def _strip_token_prefixes(
    tokens: tuple[str, ...], prefixes: Sequence[tuple[str, ...]],
) -> tuple[str, ...]:
    for prefix in prefixes:
        if tokens[:len(prefix)] == prefix:
            return tokens[len(prefix):]
    return tokens


def is_clear_recording_direction(n: NormalizedText) -> bool:
    """Recognize restart/redo/interruption commands shared by slate rules."""
    if not n.tokens:
        return True
    if n.compact in EXACT_STANDALONE_PRODUCTION_DIRECTIONS or n.compact == "wait":
        return True
    if any(starts_phrase(n, phrase) for phrase in LEADING_PRODUCTION_DIRECTIONS):
        return True
    if is_camera_rolling_slate(n.compact):
        return True
    if n.compact in {
        "start over", "let's start over", "we're starting over", "i'm starting over",
        "let me restart", "let me redo that", "no redo that", "no start over",
        "let's go", "let us go",
    } or starts_phrase(n, "start over from"):
        return True
    if len(n.tokens) > 1 and n.tokens[0] in {"wait", "no"}:
        return is_clear_recording_direction(normalized_text(" ".join(n.tokens[1:])))
    return False


def is_compound_take_slate(n: NormalizedText) -> bool:
    """Distinguish recording takes from dosage, quantity, and usage language."""
    tokens = _strip_token_prefixes(n.tokens, TAKE_SLATE_DISCOURSE_PREFIXES)
    tokens = _strip_token_prefixes(tokens, TAKE_SLATE_INTRO_PREFIXES)
    if not tokens or tokens[0] != "take":
        return False
    if len(tokens) >= 3 and tokens[1] == "number" and tokens[2] in TAKE_SLATE_NUMBERS:
        remainder = tokens[3:]
    elif len(tokens) >= 2 and tokens[1] in TAKE_SLATE_NUMBERS:
        remainder = tokens[2:]
    else:
        return False
    return is_clear_recording_direction(normalized_text(" ".join(remainder)))


def production_meta_rule(text: str, include_camera_rolling: bool = True) -> Optional[str]:
    n = normalized_text(text)
    if not n.compact:
        return None
    start_over_commands = {
        "start over",
        "let's start over",
        "okay start over",
        "no start over",
        "can we start over",
        "i need to start over",
    }
    if n.compact in start_over_commands or starts_phrase(n, "start over from"):
        return "production_meta_phrase"
    if include_camera_rolling and is_camera_rolling_slate(text):
        return "production_meta_phrase"
    if is_compound_take_slate(n) or n.compact == "i'm starting again":
        return "production_meta_phrase"
    if n.compact in {"toma dos", "toma tres", "toma número dos", "toma número tres"}:
        return "production_meta_phrase"
    return None


def filler_rule(text: str) -> Optional[str]:
    n = normalized_text(text)
    if not n.tokens:
        return "empty"
    if len(n.tokens) == 1 and n.tokens[0] in {"and", "uh", "um", "hmm", "like", "wait", "okay", "alright"}:
        return "standalone_meta_token"
    if n.compact in EXACT_STANDALONE_FILLER:
        return "standalone_meta_token"
    if n.compact in END_OF_TAKE_FILLER:
        return "end_of_take_filler"
    if n.compact in EXACT_STANDALONE_PRODUCTION_DIRECTIONS:
        return "standalone_production_direction"
    if any(starts_phrase(n, pat) for pat in LEADING_PRODUCTION_DIRECTIONS):
        return "restart_or_interruption_language"
    meta_reason = production_meta_rule(text)
    if meta_reason:
        return meta_reason
    if SAFE_MULTIWORD_META_PHRASES and has_any_phrase(n, SAFE_MULTIWORD_META_PHRASES):
        return "production_meta_phrase"
    return None


def looks_like_filler(text: str) -> bool:
    return filler_rule(text) is not None


def looks_like_dependent_tail(text: str) -> bool:
    n = normalized_text(text)
    t = n.text
    if not n.tokens or len(n.tokens) > 4:
        return False
    if any(t.endswith(suf) for suf in TAIL_DEPENDENT_ENDINGS):
        return True
    if len(n.tokens) == 1 and n.tokens[0] in TAIL_DEPENDENT_STARTS:
        return True
    return n.tokens[0] in TAIL_DEPENDENT_STARTS and not t.endswith((".", "?", "!"))


SAFE_EXPLICIT_CTA_PHRASES = (
    "buy now", "shop now", "order now", "tap the link", "click the link",
    "click below", "get yours", "add to cart", "check it out below",
    "check it out", "check these out", "check them out", "drop it down below",
)

# Link-only instructions are CTAs without requiring a purchase verb. They are
# matched as complete normalized utterances (after one optional discourse word)
# so descriptive or historical mentions of a link cannot trigger this rule.
EXPLICIT_LINK_CTA_PHRASES = {
    "link below", "link in bio", "check the link", "check the link below",
    "the link is below", "the link is in my bio", "tap the link",
    "click the link", "click the link below",
}
LINK_CTA_LEADING_DISCOURSE = {"so", "okay", "and", "please"}

IMPERATIVE_ACTION_VERBS = {"buy", "shop", "order", "grab", "get", "check", "tap", "click", "drop", "pick", "add"}
CTA_MODIFIERS = {"now", "today", "below", "link", "yours", "available", "cart", "set", "collection"}
CTA_PRODUCT_OBJECTS = {
    "it", "this", "that", "these", "them", "one", "some", "yours", "product", "products",
    "set", "collection", "shade", "shades", "item", "items", "gloss", "glosses",
}
CTA_PRODUCT_NOUNS = {
    "product", "products", "set", "collection", "shade", "shades", "item", "items",
    "gloss", "glosses",
}
CTA_OBJECT_DETERMINERS = {"the", "a", "an", "your", "our"}
CTA_LEADING_DISCOURSE = {"so", "okay", "well", "and", "please"}


@dataclass(frozen=True)
class CTAActionFrame:
    action: str
    action_index: int
    frame_type: str
    clause_tokens: tuple[str, ...]


def _action_target(tokens: Sequence[str], action_index: int) -> Optional[str]:
    target_index = action_index + 1
    if target_index < len(tokens) and tokens[target_index] in CTA_OBJECT_DETERMINERS:
        target_index += 1
    return tokens[target_index] if target_index < len(tokens) else None


def _starts_explicit_cta(n: NormalizedText, phrase: str) -> bool:
    """Match an explicit CTA at command position, not inside reported speech."""
    phrase_tokens = normalized_text(phrase).tokens
    if n.tokens[:len(phrase_tokens)] == phrase_tokens:
        return True
    return bool(
        n.tokens
        and n.tokens[0] in CTA_LEADING_DISCOURSE
        and n.tokens[1:1 + len(phrase_tokens)] == phrase_tokens
    )


def _cta_clauses(text: str) -> List[tuple[NormalizedText, str]]:
    """Split punctuation-delimited clauses without borrowing prefix evidence."""
    normalized = normalize_match_text(text)
    clauses: List[tuple[NormalizedText, str]] = []
    boundary = ""
    for part in re.split(r"([.!?;,:\n]+)", normalized):
        if not part:
            continue
        if re.fullmatch(r"[.!?;,:\n]+", part):
            boundary = part
            continue
        clause = normalized_text(part)
        if clause.tokens:
            clauses.append((clause, boundary))
        boundary = ""
    return clauses


def _strip_cta_discourse(tokens: tuple[str, ...]) -> tuple[str, ...]:
    while tokens and tokens[0] in CTA_LEADING_DISCOURSE:
        tokens = tokens[1:]
    return tokens


def _is_reported_or_narrated_action(
    tokens: tuple[str, ...], action_index: int, previous_clause: Optional[NormalizedText],
) -> str:
    before = tokens[:action_index]
    subject_context = _strip_cta_discourse(before)
    reporting = {"said", "says", "told", "asked", "reminded", "heard", "wrote"}
    if any(token in reporting for token in before):
        return "reported_speech"
    if previous_clause and any(token in reporting for token in previous_clause.tokens):
        return "reported_speech"
    if any(token in before for token in {"did", "decided", "usually", "yesterday", "went", "used", "was"}):
        return "historical_action"
    if subject_context and subject_context[0] in {"i", "we", "my", "our"}:
        return "first_person_narration"
    if subject_context and subject_context[0] in {"he", "she", "they", "his", "her", "their"}:
        return "third_person_narration"
    return "unrelated_prefix"


def cta_action_frames(text: str) -> List[CTAActionFrame]:
    """Bind each CTA action to an immediate command/modal frame in its clause."""
    frames: List[CTAActionFrame] = []
    clauses = _cta_clauses(text)
    for clause_index, (clause, boundary_before) in enumerate(clauses):
        tokens = clause.tokens
        command_tokens = _strip_cta_discourse(tokens)
        command_offset = len(tokens) - len(command_tokens)
        carries_reported_speech = bool(boundary_before and set(boundary_before) <= {",", ":"})
        previous_clause = clauses[clause_index - 1][0] if clause_index and carries_reported_speech else None
        for action_index, action in enumerate(tokens):
            if action not in IMPERATIVE_ACTION_VERBS:
                continue
            local_index = action_index - command_offset
            narrative_frame = _is_reported_or_narrated_action(
                tokens, action_index, previous_clause
            )
            frame_type: Optional[str] = None
            if narrative_frame == "reported_speech":
                frame_type = narrative_frame
            elif local_index == 0:
                frame_type = "imperative_action"
            elif local_index >= 0:
                before = command_tokens[:local_index]
                if before in (("you", "can"), ("you", "could"), ("you", "should"), ("you", "must")):
                    frame_type = "viewer_directed_modal"
                elif before == ("go",):
                    frame_type = "viewer_directed_go"
                elif before == ("make", "sure", "you"):
                    frame_type = "viewer_directed_reminder"
                elif before == ("don't", "forget", "to"):
                    frame_type = "viewer_directed_reminder"
                elif before == ("go", "ahead", "and"):
                    frame_type = "viewer_directed_go"
            if frame_type is None:
                frame_type = narrative_frame
            frames.append(CTAActionFrame(action, action_index, frame_type, tokens))
    return frames


def is_explicit_link_cta(n: NormalizedText) -> bool:
    tokens = n.tokens
    if tokens and tokens[0] in LINK_CTA_LEADING_DISCOURSE:
        tokens = tokens[1:]
    return " ".join(tokens) in EXPLICIT_LINK_CTA_PHRASES


def cta_action_rule(text: str) -> Optional[str]:
    """Return a CTA rule only when an ambiguous action has viewer intent.

    Action words such as ``order``, ``shop``, and ``check`` are also ordinary
    nouns or narration verbs. Token position alone is therefore never enough:
    imperative uses need a product object or CTA modifier, while non-imperative
    uses need an explicit viewer-directed prefix or link/purchase instruction.
    """
    n = normalized_text(text)
    if not n.tokens:
        return None

    if is_explicit_link_cta(n):
        return "explicit_link_instruction"
    if any(starts_phrase(n, phrase) for phrase in (
        "compra ahora", "ordena ahora", "pide ahora", "haz clic en el enlace",
        "toca el enlace", "añádelo al carrito", "consigue el tuyo",
    )):
        return "safe_explicit_cta_phrase_es"

    for frame in cta_action_frames(text):
        if frame.frame_type not in {
            "imperative_action", "viewer_directed_modal",
            "viewer_directed_go", "viewer_directed_reminder",
        }:
            continue
        clause = NormalizedText("", "", frame.clause_tokens, " ".join(frame.clause_tokens))
        clause_token_set = set(frame.clause_tokens)
        action = frame.action
        action_index = frame.action_index
        next_token = frame.clause_tokens[action_index + 1] if action_index + 1 < len(frame.clause_tokens) else None
        action_target = _action_target(frame.clause_tokens, action_index)
        has_modifier = any(token in CTA_MODIFIERS for token in frame.clause_tokens[action_index + 1:])
        has_product_object = action_target in CTA_PRODUCT_OBJECTS

        if any(_starts_explicit_cta(clause, phrase) for phrase in SAFE_EXPLICIT_CTA_PHRASES):
            return "safe_explicit_cta_phrase"
        if not (next_token in CTA_MODIFIERS or has_product_object):
            continue
        if action in {"tap", "click"} and not ({"link", "below"} & clause_token_set):
            continue
        if action == "check" and not (has_modifier or has_any_phrase(clause, ("check it out", "check these out", "check them out"))):
            continue
        if action == "drop" and not ({"below", "link"} & clause_token_set):
            continue
        if action == "add" and "cart" not in clause_token_set:
            continue
        if action in {"order", "grab", "get", "pick"} and not (
            has_modifier or action_target in CTA_PRODUCT_NOUNS
        ):
            continue
        return frame.frame_type

    return None


def has_cta_action_context(text: str) -> bool:
    return cta_action_rule(text) is not None


def has_product_enumeration_evidence(text: str) -> bool:
    n = normalized_text(text)
    product_terms = {
        "stocking", "santa", "hat", "tree", "snowman", "shade", "shades", "color", "colors",
        "design", "designs", "variant", "variants", "set", "pack", "gloss", "glosses", "item", "items",
        "scent", "scents", "flavor", "flavors", "size", "sizes", "option", "options", "ornament", "ornaments",
        "capsule", "capsules", "tablet", "tablets", "scoop", "scoops", "gummy", "gummies",
    }
    list_structure = "," in n.text and "and" in n.tokens
    product_context = has_any_phrase(n, (
        "it includes", "included are", "includes", "included", "comes with", "you get",
        "set of", "set comes with", "variants", "designs", "colors", "shades",
    ))
    quantity_context = any(token.isdigit() for token in n.tokens) or any(
        token in n.tokens for token in {"one", "two", "three", "four", "five", "six", "set", "pack"}
    )
    product_term_count = sum(1 for token in n.tokens if token in product_terms)
    return product_term_count >= 2 and (list_structure or product_context or quantity_context)


def classify_slot_rule(text: str) -> tuple[str, str]:
    n = normalized_text(text)

    meta_reason = production_meta_rule(text)
    if meta_reason:
        return "OTHER", meta_reason

    if has_cta_action_context(text):
        return "CTA", "direct_purchase_or_action_language"

    # Explicit commercial-function evidence is evaluated before generic
    # grammatical constructions ("it's a", "this is a", "these are").
    if has_any_phrase(n, (
        "tired of", "sick of", "problem", "problems", "struggle",
        "keep giving you", "frustrated", "failed alternative", "es un problema",
        "me cuesta", "tengo problemas", "lucho por",
    )):
        return "PROBLEM", "problem_language"

    if has_any_phrase(n, (
        "i think they're really good", "i get so many compliments", "before and after",
        "five stars", "measurable result", "resultados comprobados", "cinco estrellas",
    )):
        return "PROOF", "proof_or_testimonial_language"

    if has_any_phrase(n, (
        "because i found", "i've been using", "i've tried", "honestly", "for me",
        "let me tell you", "when i", "at first", "the first time", "my experience", "i discovered",
    )):
        return "STORY", "personal_story_language"

    if has_any_phrase(n, (
        "so you can", "you can", "you'll", "you will", "feel", "helps you", "so freaking",
        "elevates any outfit", "feel fresh", "confident", "so cute", "they are all",
        "te ayuda", "para que puedas", "te sentirás", "hidrata tu piel",
    )):
        return "BENEFITS", "positive_appeal_or_outcome_language"

    if has_any_phrase(n, (
        "each gummy", "packed with", "ingredients", "it has", "it comes with", "comes with",
        "this bag", "these probiotics", "slippery m", "prebiotic",
        "probiotic", "flavored", "you get", "set of", "comes in", "lip glosses", "these are",
        "included", "includes", "variants", "designs", "colors", "shades", "contiene",
        "viene con", "incluye", "tiene una fórmula", "tonos", "colores",
    )) or has_product_enumeration_evidence(text):
        return "FEATURES", "product_details_quantity_variants_or_enumeration"

    if "?" in n.text or starts_phrase(n, "if") or starts_phrase(n, "hey") or starts_phrase(n, "listen") or starts_phrase(n, "stop scrolling") or starts_phrase(n, "ladies") or starts_phrase(n, "guys") or has_any_phrase(n, (
        "i found the perfect", "perfect gift", "wait until you see", "looking for the perfect",
    )):
        return "HOOK", "opening_promise_or_product_discovery"

    if has_any_phrase(n, (
        "it's actually", "it's a", "this is a", "these are",
        "es un", "esta es una", "este es un",
    )):
        return "FEATURES", "generic_product_construction"

    if looks_like_filler(text):
        return "OTHER", "filler_or_too_short"
    return "OTHER", "unclassified_product_context"


def classify_slot(text: str) -> str:
    return classify_slot_rule(text)[0]
