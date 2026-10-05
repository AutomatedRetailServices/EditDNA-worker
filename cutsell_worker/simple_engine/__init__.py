"""CutSell simple engine v1.0.

One small, calibrated editorial engine that replaces the legacy V2 decision stack
when CUTSELL_ENGINE=simple. See docs/CUTSELL_SIMPLE_ENGINE_V1.md.

    from cutsell_worker.simple_engine import process
    out = process(words, duration, audio_path, video_path=None)
"""
from .engine import VERSION, caption_groups, captions, process

__all__ = ["VERSION", "process", "captions", "caption_groups"]
