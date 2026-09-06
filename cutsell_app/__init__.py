"""CutSell clean mobile API package."""

from cutsell_worker.commercial_observability import initialize_observability

OBSERVABILITY_STATUS = initialize_observability(service="cutsell-api")
