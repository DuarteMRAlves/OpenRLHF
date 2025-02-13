# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for vLLM."""
import logging
import math
import sys
import time

from typing import Optional

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("openrlhf")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger


def make_progress_logger(
    total_steps: int,
    *,
    log_every_percent: Optional[int] = None,
    desc: str = "Progress",
    log_time: bool = False,
):
    if log_every_percent is not None:
        log_steps = [
            math.ceil(total_steps * p / 100) 
            for p in range(log_every_percent, 100 + log_every_percent, log_every_percent)
        ]
    
    if log_time:
        start_time = time.time()

    def _print_step(step, force: bool = False):
        return log_every_percent is None or step in log_steps or force

    def _log_fn(step, msg: str = None, force=False):
        if _print_step(step, force):
            perc = (step / total_steps) * 100
            full_msg = f"{desc}: {step}/{total_steps} ({perc:.0f}%)"
            if log_time:
                full_msg += f" {get_time_info(start_time, total_steps, step)}"
            if msg is not None:
                full_msg += f" - {msg}"
            print(full_msg, flush=True)

    return _log_fn

def get_time_info(start_time, total_steps, step):
    if step == 0:
        return "[00:00<NA]"
    elapsed = time.time() - start_time

    elapsed_hh, elapsed_mm, elapsed_ss = seconds_to_hms(elapsed)

    # Calculate elapsed time.
    elapsed = time.time() - start_time

    # Calculate average time per step.
    avg_time_per_step = elapsed / step

    # Determine the number of steps remaining.
    steps_left = total_steps - step

    # Estimate the remaining time in seconds.
    remaining_seconds = avg_time_per_step * steps_left

    # Convert remaining seconds into hours, minutes, and seconds.
    left_hh, left_mm, left_ss = seconds_to_hms(remaining_seconds)

    if left_hh == 0 and elapsed_hh == 0:
        return f"[{elapsed_mm:02d}:{elapsed_ss:02d}<{left_mm:02d}:{left_ss:02d}]"

    return f"[{elapsed_hh:02d}:{elapsed_mm:02d}:{elapsed_ss:02d}<{left_hh:02d}:{left_mm:02d}:{left_ss:02d}]"


def seconds_to_hms(seconds):
    """
    Convert seconds to hours, minutes, and seconds.
    """
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

def estimate_time_left_hms(start_time, total_steps, current_step):
    """
    Estimate the remaining time to complete the process, returning hours, minutes, and seconds.
    
    Parameters:
    - start_time: The timestamp when the process started (at step 0).
    - total_steps: The total number of steps in the process.
    - current_step: The current step number (should be > 0 for a valid estimate).

    Returns:
    - A tuple (hours, minutes, seconds) representing the estimated time remaining,
      or None if not enough steps have been completed.
    """
    # Cannot compute an estimate if no steps have been completed.
    if current_step <= 0:
        return None

    # Calculate elapsed time.
    elapsed = time.time() - start_time

    # Calculate average time per step.
    avg_time_per_step = elapsed / current_step

    # Determine the number of steps remaining.
    steps_left = total_steps - current_step

    # Estimate the remaining time in seconds.
    remaining_seconds = avg_time_per_step * steps_left

    # Convert remaining seconds into hours, minutes, and seconds.
    return seconds_to_hms(remaining_seconds)