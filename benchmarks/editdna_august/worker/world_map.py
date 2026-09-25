def map_words_to_clips(asr_words, clips):
    """
    Une palabras del ASR con cada clip usando alineación temporal.
    """
    clip_word_map = {c["id"]: [] for c in clips}

    for w in asr_words:
        w_start = w["start"]
        w_end = w["end"]

        for c in clips:
            if w_start >= c["start"] - 0.05 and w_end <= c["end"] + 0.05:
                clip_word_map[c["id"]].append(w)
                break

    return clip_word_map
