"""Wire the stable editable Flow B draft into the pipeline result."""
from pathlib import Path

path = Path("worker/pipeline.py")
text = path.read_text(encoding="utf-8")

import_line = "from worker.editable_draft import build_editable_draft\n"
if import_line not in text:
    anchor = "from worker.diagnostics import (\n"
    if text.count(anchor) != 1:
        raise SystemExit("Unexpected pipeline import baseline")
    text = text.replace(anchor, import_line + anchor, 1)

if '"editable_draft":editable_draft' not in text:
    anchor = "        result={\"ok\":True,\"session_id\":session_id,\"input_local\":input_local,\n"
    if text.count(anchor) != 1:
        raise SystemExit("Unexpected pipeline result baseline")
    insert = (
        "        editable_draft = build_editable_draft(\n"
        "            clips, used, mode=mode,\n"
        "            clean_cut_discard_diagnostics=clean_cut_discard_diagnostics,\n"
        "        )\n"
    )
    text = text.replace(anchor, insert + anchor, 1)

    result_anchor = "      \"input_durations_sec\":durations,\"clips\":clips,\"slots\":slots,\"composer\":composer,\n"
    if text.count(result_anchor) != 1:
        raise SystemExit("Unexpected pipeline result fields baseline")
    text = text.replace(
        result_anchor,
        "      \"input_durations_sec\":durations,\"clips\":clips,\"slots\":slots,\"composer\":composer,\n"
        "      \"editable_draft\":editable_draft,\n",
        1,
    )

path.write_text(text, encoding="utf-8")
print("Editable draft wired into worker/pipeline.py")
