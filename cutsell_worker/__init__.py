"""CutSell clean worker package.

Mobile-first Flow B engine. Modules are intentionally small and communicate
through versioned typed contracts defined in ``cutsell_worker.contracts``.
"""

from .commercial_observability import initialize_observability
from .clean_cut_retry import install_clean_cut_contract_recovery
from .recording_process_context import install_recording_process_context_cleanup
from .recording_suffix_trim import install_visual_self_critique_suffix_trim
from .dangling_delivery import install_dangling_delivery_cleanup
from .internal_self_correction import install_internal_self_correction_trim
from .recording_breaks import install_recording_break_cleanup
from .restart_questions import install_short_restart_question_cleanup
from .frustrated_restart import install_soft_frustration_restart_cleanup

install_clean_cut_contract_recovery()
install_recording_process_context_cleanup()
install_visual_self_critique_suffix_trim()
install_dangling_delivery_cleanup()
install_internal_self_correction_trim()
install_recording_break_cleanup()
install_short_restart_question_cleanup()
install_soft_frustration_restart_cleanup()

__version__ = "0.1.0"
OBSERVABILITY_STATUS = initialize_observability(service="cutsell-worker")
