import sys
import types
import unittest
from unittest.mock import patch

from benchmark_clip_sanitizer import sanitize_benchmark_result

if "rq" not in sys.modules:
    rq_module = types.ModuleType("rq")
    exceptions_module = types.ModuleType("rq.exceptions")

    class StopRequested(Exception):
        pass

    exceptions_module.StopRequested = StopRequested
    rq_module.exceptions = exceptions_module
    sys.modules["rq"] = rq_module
    sys.modules["rq.exceptions"] = exceptions_module

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

    def test_preserves_brief_utterances_without_ascii_punctuation(self):
        result = {
            "clips": [
                {"id": "yes", "start": 0.0, "end": 0.20, "text": "Yes", "meta": {}},
                {"id": "buy", "start": 1.0, "end": 1.25, "text": "Buy now", "meta": {}},
                {"id": "zh", "start": 2.0, "end": 2.25, "text": "好！", "meta": {}},
            ]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual([clip["id"] for clip in cleaned["clips"]], ["yes", "buy", "zh"])

    def test_preserves_unicode_transcripts(self):
        result = {
            "clips": [
                {"id": "es", "start": 0.0, "end": 1.0, "text": "¡Mañana será mejor!", "meta": {}},
                {"id": "ru", "start": 2.0, "end": 3.0, "text": "Привет мир!", "meta": {}},
                {"id": "zh", "start": 4.0, "end": 5.0, "text": "你好世界！", "meta": {}},
            ]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual([clip["id"] for clip in cleaned["clips"]], ["es", "ru", "zh"])

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

    def test_skips_abbreviation_boundary_before_incomplete_tail(self):
        result = {
            "clips": [{
                "id": "abbr", "start": 0.0, "end": 3.0,
                "text": "Call now. Meet Dr. Smith ...",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": " Call"},
                    {"start": 0.4, "end": 0.9, "word": " now."},
                    {"start": 1.0, "end": 1.4, "word": " Meet"},
                    {"start": 1.4, "end": 1.7, "word": " Dr."},
                    {"start": 1.7, "end": 2.2, "word": " Smith"},
                ],
                "meta": {},
            }]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual(cleaned["clips"][0]["text"], "Call now.")
        self.assertEqual(cleaned["clips"][0]["end"], 0.9)

    def test_skips_business_suffix_boundary_before_lowercase_incomplete_tail(self):
        for suffix in ("Inc.", "Ltd.", "Corp.", "Co.", "LLC"):
            with self.subTest(suffix=suffix):
                text = f"Call now. From Acme {suffix} to ..."
                result = {
                    "clips": [{
                        "id": "business", "start": 0.0, "end": 3.0,
                        "text": text,
                        "words": [
                            {"start": 0.0, "end": 0.4, "word": " Call"},
                            {"start": 0.4, "end": 0.9, "word": " now."},
                            {"start": 1.0, "end": 1.4, "word": " From"},
                            {"start": 1.4, "end": 1.8, "word": " Acme"},
                            {"start": 1.8, "end": 2.2, "word": f" {suffix}"},
                            {"start": 2.2, "end": 2.6, "word": " to"},
                            {"start": 2.6, "end": 3.0, "word": " ..."},
                        ],
                        "meta": {},
                    }]
                }
                cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
                self.assertEqual(cleaned["clips"][0]["text"], "Call now.")
                self.assertEqual(cleaned["clips"][0]["end"], 0.9)

    def test_business_suffix_boundary_before_ascii_ellipsis_tail(self):
        for suffix in ("Inc.", "Ltd.", "Corp.", "Co."):
            with self.subTest(suffix=suffix):
                result = {
                    "clips": [{
                        "id": suffix, "start": 0.0, "end": 1.25,
                        "text": f"Call now. From Acme {suffix} ...",
                        "words": [
                            {"start": 0.0, "end": 0.4, "word": " Call"},
                            {"start": 0.4, "end": 0.9, "word": " now."},
                            {"start": 1.0, "end": 1.1, "word": " From"},
                            {"start": 1.1, "end": 1.2, "word": " Acme"},
                            {"start": 1.2, "end": 1.25, "word": f" {suffix}"},
                        ],
                        "meta": {},
                    }]
                }
                cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
                self.assertEqual(cleaned["clips"][0]["text"], "Call now.")
                self.assertEqual(cleaned["clips"][0]["end"], 0.9)

    def test_business_suffix_boundary_before_unicode_ellipsis_tail(self):
        for suffix in ("Inc.", "Ltd.", "Corp.", "Co."):
            with self.subTest(suffix=suffix):
                result = {
                    "clips": [{
                        "id": suffix, "start": 0.0, "end": 1.25,
                        "text": f"Call now. From Acme {suffix} …",
                        "words": [
                            {"start": 0.0, "end": 0.4, "word": " Call"},
                            {"start": 0.4, "end": 0.9, "word": " now."},
                            {"start": 1.0, "end": 1.1, "word": " From"},
                            {"start": 1.1, "end": 1.2, "word": " Acme"},
                            {"start": 1.2, "end": 1.25, "word": f" {suffix}"},
                        ],
                        "meta": {},
                    }]
                }
                cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
                self.assertEqual(cleaned["clips"][0]["text"], "Call now.")
                self.assertEqual(cleaned["clips"][0]["end"], 0.9)

    def test_business_suffix_allows_genuine_new_sentence_boundary(self):
        result = {
            "clips": [{
                "id": "business", "start": 0.0, "end": 2.0,
                "text": "Acme Inc. This is complete.",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": " Acme"},
                    {"start": 0.4, "end": 0.8, "word": " Inc."},
                    {"start": 0.8, "end": 1.2, "word": " This"},
                    {"start": 1.2, "end": 1.5, "word": " is"},
                    {"start": 1.5, "end": 2.0, "word": " complete."},
                ],
                "meta": {},
            }]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual(cleaned["clips"][0]["text"], "Acme Inc. This is complete.")

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

    def test_preserves_punctuation_and_case_differences(self):
        result = {
            "clips": [
                {"id": "question", "start": 0.0, "end": 1.0, "text": "Really?", "meta": {}},
                {"id": "exclamation", "start": 1.1, "end": 2.1, "text": "Really!", "meta": {}},
                {"id": "case", "start": 2.2, "end": 3.2, "text": "REALLY!", "meta": {}},
            ]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual(
            [clip["id"] for clip in cleaned["clips"]],
            ["question", "exclamation", "case"],
        )

    def test_drops_entire_contiguous_duplicate_run(self):
        result = {
            "clips": [
                {"id": "a", "start": 0.0, "end": 1.0, "text": "Same phrase.", "meta": {}},
                {"id": "b", "start": 1.1, "end": 2.1, "text": "Same phrase.", "meta": {}},
                {"id": "c", "start": 2.2, "end": 3.2, "text": "Same phrase.", "meta": {}},
            ]
        }
        cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)
        self.assertEqual([clip["id"] for clip in cleaned["clips"]], ["a"])


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
