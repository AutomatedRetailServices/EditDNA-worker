import unittest
from unittest.mock import patch

from benchmark_clip_sanitizer import sanitize_benchmark_result
import tasks


class BenchmarkClipSanitizerTests(unittest.TestCase):
    def test_preserves_valid_short_utterance(self):
        result = {
            "clips": [{
                "id": "cta", "start": 0.0, "end": 0.25,
                "text": "Buy now!", "meta": {},
            }]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual([clip["id"] for clip in cleaned["clips"]], ["cta"])

    def test_drops_short_incomplete_fragments(self):
        result = {
            "clips": [
                {"id": "c1", "start": 59.62, "end": 59.74, "text": "I found the perfect ...", "meta": {}},
                {"id": "c2", "start": 59.74, "end": 59.78, "text": "I found the perfect ...", "meta": {}},
                {"id": "c3", "start": 60.0, "end": 61.0, "text": "Complete thought.", "meta": {}},
            ]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual([clip["id"] for clip in cleaned["clips"]], ["c3"])

    def test_trims_only_incomplete_tail_after_cta(self):
        result = {
            "clips": [{
                "id": "cta", "start": 56.36, "end": 59.62,
                "text": "Go check these out. I found the perfect ...",
                "words": [
                    {"start": 56.36, "end": 56.6, "word": " Go"},
                    {"start": 56.6, "end": 57.0, "word": " check"},
                    {"start": 57.0, "end": 57.4, "word": " these"},
                    {"start": 57.4, "end": 58.48, "word": " out."},
                    {"start": 58.94, "end": 59.14, "word": " I"},
                    {"start": 59.14, "end": 59.62, "word": " found"},
                ],
                "meta": {},
            }]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        clip = cleaned["clips"][0]
        self.assertEqual(clip["text"], "Go check these out.")
        self.assertEqual(clip["end"], 58.48)
        self.assertEqual(len(clip["words"]), 4)

    def test_preserves_two_complete_sentences(self):
        result = {
            "clips": [{
                "id": "two", "start": 0.0, "end": 2.0,
                "text": "First sentence. Second sentence.",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": " First"},
                    {"start": 0.5, "end": 1.0, "word": " sentence."},
                    {"start": 1.0, "end": 1.5, "word": " Second"},
                    {"start": 1.5, "end": 2.0, "word": " sentence."},
                ],
                "meta": {},
            }]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual(cleaned["clips"][0]["text"], "First sentence. Second sentence.")

    def test_drops_only_contiguous_duplicate(self):
        result = {
            "clips": [
                {"id": "a", "start": 0.0, "end": 1.0, "text": "Same phrase.", "meta": {}},
                {"id": "b", "start": 1.1, "end": 2.1, "text": "Same phrase.", "meta": {}},
                {"id": "c", "start": 10.0, "end": 11.0, "text": "Same phrase.", "meta": {}},
            ]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual([clip["id"] for clip in cleaned["clips"]], ["a", "c"])


class BenchmarkSemanticV2SwitchTests(unittest.TestCase):
    def test_false_flag_disables_global_llm_for_entire_pipeline_call(self):
        import worker.pipeline as pipeline_module

        original = pipeline_module.EDITDNA_USE_LLM
        pipeline_module.EDITDNA_USE_LLM = True
        observed = {}

        def fake_run_pipeline(**kwargs):
            observed["global_enabled_during_call"] = pipeline_module.EDITDNA_USE_LLM
            observed["request_flag"] = kwargs["use_semantic_v2"]
            return {"ok": True}

        try:
            with patch.object(pipeline_module, "run_pipeline", side_effect=fake_run_pipeline):
                result = tasks.run_benchmark_pipeline(
                    use_semantic_v2=False,
                    session_id="test",
                    local_files=["video.mp4"],
                )
            self.assertEqual(result, {"ok": True})
            self.assertFalse(observed["global_enabled_during_call"])
            self.assertFalse(observed["request_flag"])
            self.assertTrue(pipeline_module.EDITDNA_USE_LLM)
        finally:
            pipeline_module.EDITDNA_USE_LLM = original

    def test_true_flag_preserves_global_setting_and_forces_v2_request(self):
        import worker.pipeline as pipeline_module

        original = pipeline_module.EDITDNA_USE_LLM
        pipeline_module.EDITDNA_USE_LLM = False
        observed = {}

        def fake_run_pipeline(**kwargs):
            observed["global_enabled_during_call"] = pipeline_module.EDITDNA_USE_LLM
            observed["request_flag"] = kwargs["use_semantic_v2"]
            return {"ok": True}

        try:
            with patch.object(pipeline_module, "run_pipeline", side_effect=fake_run_pipeline):
                tasks.run_benchmark_pipeline(
                    use_semantic_v2=True,
                    session_id="test",
                    local_files=["video.mp4"],
                )
            self.assertFalse(observed["global_enabled_during_call"])
            self.assertTrue(observed["request_flag"])
            self.assertFalse(pipeline_module.EDITDNA_USE_LLM)
        finally:
            pipeline_module.EDITDNA_USE_LLM = original


if __name__ == "__main__":
    unittest.main()
