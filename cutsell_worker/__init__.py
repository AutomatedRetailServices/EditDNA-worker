"""CutSell clean worker package.

Mobile-first Flow B engine. Modules are intentionally small and communicate
through versioned typed contracts defined in ``cutsell_worker.contracts``.
"""

from .commercial_observability import initialize_observability
from .clean_cut_retry import install_clean_cut_contract_recovery

install_clean_cut_contract_recovery()

__version__ = "0.1.0"
OBSERVABILITY_STATUS = initialize_observability(service="cutsell-worker")
