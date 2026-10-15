"""Run geomeTRIC without letting it reconfigure the application's logging.

``geometric.optimize.run_optimizer`` -- which PySCF's geomeTRIC interface also
goes through -- calls ``logging.config.fileConfig`` on every optimization.  That
closes and removes the handlers the application installed and reconfigures the
root logger from geomeTRIC's (or PySCF's) log.ini: the pipeline's log format is
replaced, every optimization step is printed, and with a quiet log.ini the
pipeline's own log goes silent.  Both refinement drivers that optimize with
geomeTRIC run it inside ``geometric_quiet()``.
"""

from __future__ import annotations

import logging
import logging.config
from collections.abc import Iterator
from contextlib import contextmanager

# geomeTRIC's loggers.  Every module logs through geometric.nifty's logger, which
# sets its own level to INFO (so a level on the "geometric" parent has no
# effect); the other two are its fallbacks outside the package.
_GEOMETRIC_LOGGERS = ("geometric.nifty", "NiftyLogger", "MoleculeLogger")


@contextmanager
def geometric_quiet() -> Iterator[None]:
    """Skip geomeTRIC's fileConfig call and hold its loggers at WARNING.

    Warnings and errors from geomeTRIC still reach the application's handlers;
    its banner and per-step INFO output do not.
    """
    loggers = [logging.getLogger(name) for name in _GEOMETRIC_LOGGERS]
    file_config, levels = logging.config.fileConfig, [lg.level for lg in loggers]
    logging.config.fileConfig = lambda *args, **kwargs: None
    for lg in loggers:
        lg.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logging.config.fileConfig = file_config
        for lg, level in zip(loggers, levels, strict=True):
            lg.setLevel(level)
