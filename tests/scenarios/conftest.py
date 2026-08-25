import os
import shutil

import pytest

from dragen.utilities.InputInfo import RveInfo

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUTPUT_DATA_DIR = os.path.join(REPO_ROOT, 'OutputData')


@pytest.fixture(autouse=True)
def _cleanup_output_dir():
    """Delete whatever OutputData/<date>_<NNN> directory a scenario creates during the test.

    Case_*.py scripts set root='./' and also reference their input files (ExampleInput/...) via
    paths relative to the repo root, so the working directory can't simply be redirected into a
    tmp_path without breaking those reads. Instead, snapshot OutputData/ before the test and
    remove whatever appeared afterward (tests/scenarios/assertions.py has already read what it
    needs from RveInfo.store_path by then), keeping runs from littering the working tree without
    touching any Case_*.py file.

    dragen/run.py's setup_logging()/initializations() add a new TimedRotatingFileHandler to the
    shared RveInfo.LOGGER/RESULT_LOG loggers on every Run(), never closing the previous one. On
    Windows an open handle to a file blocks deleting the directory that contains it, so without
    closing handlers first, rmtree silently (ignore_errors=True) leaves a locked result-logs file
    behind -- and the *next* scenario's os.makedirs() sees that leftover directory already exists
    (dragen/run.py checks `if not os.path.isdir(...)` before creating it) and writes straight into
    it, silently merging multiple scenarios' output into one directory.
    """
    before = set(os.listdir(OUTPUT_DATA_DIR)) if os.path.isdir(OUTPUT_DATA_DIR) else set()
    logger_handlers_before = list(RveInfo.LOGGER.handlers)
    result_log_handlers_before = list(RveInfo.RESULT_LOG.handlers)
    yield
    for logger, handlers_before in (
        (RveInfo.LOGGER, logger_handlers_before),
        (RveInfo.RESULT_LOG, result_log_handlers_before),
    ):
        for handler in list(logger.handlers):
            if handler not in handlers_before:
                handler.close()
                logger.removeHandler(handler)

    after = set(os.listdir(OUTPUT_DATA_DIR)) if os.path.isdir(OUTPUT_DATA_DIR) else set()
    for name in after - before:
        shutil.rmtree(os.path.join(OUTPUT_DATA_DIR, name), ignore_errors=True)
