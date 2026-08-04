import unittest

from benchmark_clip_sanitizer import sanitize_benchmark_result


class BenchmarkClipSanitizerTests(unittest.TestCase):
    def test_business_suffixes_do_not_hide_previous_sentence_boundary(self):
        cases = ("Inc.", "Ltd.", "LLC", "Corp.", "Co.")
        for suffix in cases:
            with self.subTest(suffix=suffix):
                text = f"Call now. From Acme {suffix} to ..."
                result = {
                    "clips": [{
                        "id": suffix, "start": 0.0, "end": 3.0, "text": text,
                        "words": [
                            {"start": 0.0, "end": 0.4, "word": " Call"},
                            {"start": 0.4, "end": 0.8, "word": " now."},
                            {"start": 0.8, "end": 1.2, "word": " From"},
                            {"start": 1.2, "end": 1.6, "word": " Acme"},
                            {"start": 1.6, "end": 2.0, "word": f" {suffix}"},
                            {"start": 2.0, "end": 2.4, "word": " to"},
                            {"start": 2.4, "end": 3.0, "word": " ..."},
                        ],
                        "meta": {},
                    }]
                }

                cleaned = sanitize_benchmark_result(result, use_semantic_v2=False)

                self.assertEqual(cleaned["clips"][0]["text"], "Call now.")
                self.assertEqual(cleaned["clips"][0]["end"], 0.8)
                self.assertEqual(len(cleaned["clips"][0]["words"]), 2)

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


if __name__ == "__main__":
    unittest.main()
