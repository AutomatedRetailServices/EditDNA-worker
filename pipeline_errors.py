"""Lightweight classified exceptions shared by tasks and the media pipeline."""


class PipelineError(RuntimeError):
    """Base class for classified pipeline failures."""


class SelectionError(PipelineError):
    """A deterministic failure caused by an invalid or empty selection."""


class MissingSelectedClipsError(SelectionError):
    """Selected clip IDs did not resolve uniquely against analyzed clips."""


class UploadError(PipelineError):
    """The rendered output could not be uploaded."""


class JobCanceledError(PipelineError):
    """A cooperative render-job cancellation request was observed."""
