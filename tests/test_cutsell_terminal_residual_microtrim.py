from cutsell_worker.speech_visual_microtrim import _is_terminal_delivery, _minimum_quiet_ratio


def test_terminal_delivery_allows_moderate_performance_residual():
    assert _is_terminal_delivery("perfectamente.") is True
    assert _is_terminal_delivery('done!”') is True
    assert _minimum_quiet_ratio(0.50, terminal_delivery=True) == 0.35
    assert _minimum_quiet_ratio(0.80, terminal_delivery=True) == 0.45
    assert _minimum_quiet_ratio(1.10, terminal_delivery=True) == 0.55


def test_incomplete_delivery_keeps_strict_acoustic_guard():
    assert _is_terminal_delivery("funcionando") is False
    assert _is_terminal_delivery("porque") is False
    assert _minimum_quiet_ratio(0.50, terminal_delivery=False) == 0.68
    assert _minimum_quiet_ratio(0.80, terminal_delivery=False) == 0.76
    assert _minimum_quiet_ratio(1.10, terminal_delivery=False) == 0.82


def test_terminal_policy_never_weakens_asr_word_guards():
    # This helper only changes the acoustic threshold. The detector still requires
    # cut_start > left_word_end and cut_end < right_word_start, so lexical words are
    # never made eligible by the terminal residual policy itself.
    assert _minimum_quiet_ratio(0.50, terminal_delivery=True) < _minimum_quiet_ratio(
        0.50, terminal_delivery=False
    )
