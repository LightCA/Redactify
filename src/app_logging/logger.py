import os
import logging

from pathlib import Path

from app_logging.threaded_file_handler import ThreadedFileHandler

_initialized: bool = False


def init_logger():
    global _initialized
    if _initialized:
        return

    log_dir = Path(os.environ["WORKING_DIRECTORY"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    handler = ThreadedFileHandler(log_dir / "redactify.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    _initialized = True
