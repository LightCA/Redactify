import logging

from pathlib import Path

from text.threaded_file_writer import ThreadedFileWriterRegistry


class ThreadedFileHandler(logging.Handler):
    def __init__(self, file_path: Path, level=logging.NOTSET):
        super().__init__(level)
        self.writer = ThreadedFileWriterRegistry.get_writer(file_path)
        self.writer.print = True

    def emit(self, record: logging.LogRecord):
        try:
            self.writer.write_line(self.format(record))
        except Exception:
            self.handleError(record)
