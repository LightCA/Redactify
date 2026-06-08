import os

from queue import Queue
from threading import Thread, Lock, Event
from typing import TextIO
from pathlib import Path

_STOP = object()


class ThreadedFileWriter:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.print = False

        self._is_empty = False
        self._queue: Queue = Queue()
        self._trigger = Event()
        self._running: bool = False
        self._thread: Thread
        self._file: TextIO
        self._run_lock = Lock()

    def _run(self):
        while True:
            try:
                self._trigger.wait()
                self._trigger.clear()
                while not self._queue.empty():
                    data = self._queue.get(timeout=0.1)
                    if data is _STOP:
                        self._file.flush()
                        return
                    if self._is_empty:
                        if data[:1] == "\n":
                            data = data[1:]
                        self._is_empty = False
                    self._file.write(data)
                    if self.print:
                        print(data)
                self._file.flush()
            except Exception as ex:
                print(ex)

    def write(self, text: str):
        self._queue.put(text)
        self._trigger.set()

    def write_line(self, text: str):
        self._queue.put("\n" + text)
        self._trigger.set()

    def is_empty(self):
        return os.path.getsize(self.file_path) == 0

    def start(self):
        with self._run_lock:
            if not self._running:
                self._running = True
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                self._file = open(self.file_path, mode="a", encoding="utf-8")
                self._is_empty = self.is_empty()
                self._thread = Thread(target=self._run, daemon=True)
                self._thread.start()

    def stop(self):
        with self._run_lock:
            if self._running:
                self._queue.put(_STOP)
                self._trigger.set()
                self._thread.join()
                self._file.close()
                self._running = False


class ThreadedFileWriterRegistry:
    _writers: dict[Path, ThreadedFileWriter] = {}
    _lock = Lock()

    @classmethod
    def get_writer(cls, file_path: Path) -> ThreadedFileWriter:
        with cls._lock:
            writer = cls._writers.get(file_path)
            if writer is None:
                writer = ThreadedFileWriter(file_path)
                cls._writers[file_path] = writer

        writer.start()
        return writer

    @classmethod
    def stop_writer(cls, file_path: Path) -> bool:
        with cls._lock:
            writer = cls._writers.get(file_path)

        if writer is None:
            return False

        writer.stop()
        return True

    @classmethod
    def stop_all_writers(cls):
        with cls._lock:
            writers = list(cls._writers.values())

        for writer in writers:
            writer.stop()
