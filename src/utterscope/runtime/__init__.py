"""Runtime helpers shared by CLI and pipeline."""

from utterscope.runtime.silence import enable_quiet_mode, silence_third_party

__all__ = ["enable_quiet_mode", "silence_third_party"]
