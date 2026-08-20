"""Scoped suppression helpers for noisy third-party tool initialization."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import io
import logging
import warnings
from typing import Iterator


@contextmanager
def suppress_vendor_init_output() -> Iterator[None]:
    """Silence known vendor initialization chatter for local OCR/layout stacks."""
    with ExitStack() as stack:
        stream = io.StringIO()
        stack.enter_context(redirect_stdout(stream))
        stack.enter_context(redirect_stderr(stream))
        stack.enter_context(_suppress_info_logging())
        stack.enter_context(warnings.catch_warnings())
        warnings.filterwarnings(
            "ignore",
            message=r"No ccache found\..*",
            category=UserWarning,
        )
        yield


@contextmanager
def _suppress_info_logging() -> Iterator[None]:
    previous_disable = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(previous_disable)
