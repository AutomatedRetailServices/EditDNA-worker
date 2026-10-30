# Deploy 30 Oct 2026 (PO OK "continua", 19:32 ET)

Production = d19a866 on `feat/simple-engine-v1` (Modal worker deploy run 41145109513, Render cutsell-api
deploy dep-dbiiicp42hec73aj87sg). Rollback, NOT applied: branch `rollback/antes-30oct` (= b5eb3e1).

## What went live
1. **Little speech** (`fix/short-speech-cut`): a video with a few words is cut where the speech ends; only a
   video with no words at all is "no speech". `diagnostics.little_speech` -> app notice "Almost no voice".
2. **Engine 1.2.5**: `finish_sentences` (a kept part takes back the one word that closes its sentence) and
   the unheard-word rule in `refine` (>= 100 ms of sound over the floor in a mid-sentence pause <= 1.5 s keeps
   the pause). Found on job 92d11b5b ("...la mira", "una cosa [exotica]"). Gold G02 added (27 videos / 611).
3. **Clean up voice v2** (`feat/clean-voice-ai`): MDX-Net Kim_Vocal_2 voice isolation (onnxruntime, model
   baked in the Modal image with sha256 check) instead of afftdn; copies under clean/v2/. ~2.5x audio length
   on the worker's 2 CPUs. Without the model: v1 filter, recorded as method "filter".
4. **Captions / text**: 24 more fonts (33 total, OFL/Apache, licences in cutsell_worker/fonts) and validated
   `caption_style` / text overlay `font` + `style` (cutsell_worker/text_style.py).

## Checks
Full suite: 10186 passed, the same 44 old benchmark-test failures as production, none new.
Live after deploy: healthz OK; 3 real cuts (perf-cut-timing run 41145364657) all finished with a draft.

## App
TestFlight build 68 = `build/clon-and-style` 32e1f58: Clon, bubbly style, promoting sheet by brand,
"Almost no voice" notice, editor rows (Edit / Audio / Text), Fonts / Style sheet, Duplicate.
