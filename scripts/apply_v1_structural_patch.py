"""One-shot structural recovery patch for PR #23.

The replacements are intentionally exact and fail loudly if the expected recovery
baseline changes. This keeps the large legacy pipeline edit auditable.
"""
from pathlib import Path

PATH = Path("worker/pipeline.py")
text = PATH.read_text(encoding="utf-8")

replacements = [
    (
        '        keep = not is_filler and not is_tail\n',
        '        # Clean Cut deletion is limited to explicit production/meta filler.\n'
        '        # Linguistic dependency is useful context for ranking/composition but\n'
        '        # must not delete otherwise valid speech.\n'
        '        keep = not is_filler\n',
    ),
    (
        '        if not keep:\n'
        '            if is_tail:\n'
        '                reason = "Dependent tail without full context (cola tipo \'available as well\')."\n'
        '            else:\n'
        '                reason = "Marked as filler / meta (redo, wait, etc.)."\n',
        '        if not keep:\n'
        '            reason = "Explicit production/meta speech removed by Clean Cut."\n'
        '        elif is_tail:\n'
        '            reason = "Dependent context retained for semantic/composer evaluation."\n',
    ),
    (
        '        if info.primary_slot.value == "OTHER":\n'
        '            # Preserve the heuristic/public slot and keep state, but ensure a validated\n'
        '            # non-sales judgment cannot enter a sales composer timeline.\n'
        '            c["meta"]["semantic_v2"]["application_status"] = "excluded_other"\n'
        '            c["meta"]["semantic_v2"]["application_reason"] = "validated_other_excluded_from_sales_composer"\n'
        '            c["meta"]["semantic_v2"]["excluded_from_composer"] = True\n'
        '            continue\n',
        '        if info.primary_slot.value == "OTHER":\n'
        '            # OTHER is a semantic label, never deletion authority. Preserve valid\n'
        '            # speech and let the editable draft/composer decide presentation.\n'
        '            c["slot"] = "OTHER"\n'
        '            c["meta"]["slot"] = "OTHER"\n'
        '            c["meta"]["semantic_v2"]["application_status"] = "applied_other_preserved"\n'
        '            c["meta"]["semantic_v2"]["application_reason"] = "semantic_label_not_deletion_authority"\n'
        '            c["meta"]["semantic_v2"]["applied"] = True\n'
        '            continue\n',
    ),
    (
        '    if not meta.get("keep", True) or meta.get("semantic_v2", {}).get("excluded_from_composer", False):\n'
        '        return False\n',
        '    if not meta.get("keep", True):\n'
        '        return False\n',
    ),
    (
        '    visual = safe_float(clip.get("visual_score", 0.0))\n'
        '    return not (visual > 0.0 and visual < 0.58)\n',
        '    # Visual quality is a ranking signal. A low embedding score is not enough\n'
        '    # to delete intelligible, valid speech from an editable draft.\n'
        '    return True\n',
    ),
    (
        '    keepable = [\n'
        '        c\n'
        '        for c in clips\n'
        '        if _composer_hard_eligible(c)\n'
        '    ]\n',
        '    # Composer selection is intentionally isolated from Clean Cut state. Legacy\n'
        '    # scoring/dedupe functions may mutate meta.keep, so operate on deep copies\n'
        '    # and retain every valid original clip for alternates/restore.\n'
        '    keepable = [\n'
        '        copy.deepcopy(c)\n'
        '        for c in clips\n'
        '        if _composer_hard_eligible(c)\n'
        '    ]\n',
    ),
    (
        '            if c["meta"].get("keep", True)\n'
        '            and not c["meta"].get("semantic_v2", {}).get("excluded_from_composer", False)\n'
        '            and safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC\n',
        '            if c["meta"].get("keep", True)\n'
        '            and safe_float(c.get("semantic_score", 0.0)) >= COMPOSER_MIN_SEMANTIC\n',
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected exactly one match, found {count}: {old[:90]!r}")
    text = text.replace(old, new, 1)

PATH.write_text(text, encoding="utf-8")
print(f"Applied {len(replacements)} exact structural replacements to {PATH}")
