from types import SimpleNamespace

from cutsell_worker.semantic_best_take_integrity import _prefer_information_rich_tied_winner


def take(clip_id, text, *, complete=True, duration=8.0):
    return SimpleNamespace(
        clip_id=clip_id,
        text=text,
        complete_idea=complete,
        duration_sec=duration,
    )


def test_prefers_richer_take_when_hybrid_marks_both_as_tied_winners():
    short = take(
        "short",
        "Al terminar mi contrato hablé con mi ginecóloga y le pedí todos los test que ella pudiera imaginar o indicar.",
        duration=7.4,
    )
    rich = take(
        "rich",
        "Al terminar mi contrato cambié de ginecóloga y le pedí todos los test que ella pudiera imaginar e indicar. Ahí me mandó a hacer sonografías.",
        duration=11.6,
    )
    decisions = {"short": ("winner", 0.95), "rich": ("winner", 0.95)}
    assert _prefer_information_rich_tied_winner([short, rich], decisions, "short") == "rich"


def test_does_not_drop_critical_number_for_richer_peer():
    local = take("local", "El riesgo fue de 5 a 10% y por eso hice el estudio completo.")
    peer = take("peer", "El riesgo fue importante y por eso hice el estudio completo con una evaluación adicional y sonografías.", duration=12.0)
    decisions = {"local": ("winner", 0.95), "peer": ("winner", 0.95)}
    assert _prefer_information_rich_tied_winner([local, peer], decisions, "local") is None


def test_does_not_treat_different_ideas_as_information_growth():
    local = take("local", "Me hice una sonografía de tiroides y apareció un nódulo sospechoso.")
    peer = take("peer", "Después tuve gastritis, problemas de digestión, dolor de estómago y me hicieron una endoscopía durante varios meses.", duration=13.0)
    decisions = {"local": ("winner", 0.95), "peer": ("winner", 0.95)}
    assert _prefer_information_rich_tied_winner([local, peer], decisions, "local") is None


def test_requires_complete_peer():
    local = take("local", "Le pedí a mi ginecóloga todos los test que pudiera indicar.")
    peer = take("peer", "Le pedí a mi ginecóloga todos los test que pudiera indicar y entonces me mandó", complete=False, duration=10.0)
    decisions = {"local": ("winner", 0.95), "peer": ("winner", 0.95)}
    assert _prefer_information_rich_tied_winner([local, peer], decisions, "local") is None
