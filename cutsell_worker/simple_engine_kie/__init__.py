"""A SEPARATE copy of the simple engine's model call for kie.ai (PO 16 Nov 2026).

PO: "no es adaptar el codigo.. es duplicar el codigo y hacer una adaptacion a esa duplicacion; si no sirve no se dana
el codigo original y todo sigue desde el api de anthropic".

Nothing in cutsell_worker/simple_engine/ or the adapter is changed. The engine already takes its model call as an
argument (`llm=` in simple_engine_adapter.process_with_simple_engine), so this package only provides another call,
`call_kie`, with the same signature, plus a side-by-side test (`compare.py`). Production keeps calling Anthropic; this
copy is used only by the comparison test until the PO decides otherwise.
"""
