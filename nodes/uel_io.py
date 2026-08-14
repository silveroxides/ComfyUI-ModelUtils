import os
import uuid
from collections import deque
from contextlib import contextmanager

from unifiedefficientloader import IncrementalSafetensorsWriter


class AsyncTensorCursor:
    """Ordered, bounded tensor access over one UEL asynchronous stream."""

    def __init__(self, handler, keys, *, pin_memory: bool):
        self.handler = handler
        self.expected = tuple(keys)
        self.position = 0
        self.pending = deque()
        self.stream = None
        if self.expected:
            self.stream = handler.async_stream(
                list(self.expected),
                batch_size=1,
                prefetch_batches=1,
                pin_memory=pin_memory,
            )

    def take(self, expected_key):
        if self.position >= len(self.expected):
            raise RuntimeError(f"Unexpected tensor request: {expected_key}")
        planned = self.expected[self.position]
        if planned != expected_key:
            raise RuntimeError(
                f"UEL stream order mismatch: expected {planned}, requested {expected_key}"
            )
        if not self.pending:
            self.pending.extend(next(self.stream))
        key, tensor = self.pending.popleft()
        if key != expected_key:
            raise RuntimeError(f"UEL yielded {key} while {expected_key} was expected")
        self.position += 1
        return tensor

    def release(self, key):
        self.handler.mark_processed(key)

    def finish(self):
        if self.position != len(self.expected) or self.pending:
            raise RuntimeError("UEL stream ended before all planned tensors were consumed")
        if self.stream is not None:
            try:
                next(self.stream)
            except StopIteration:
                return
            raise RuntimeError("UEL stream yielded unplanned tensors")

    def close(self):
        if self.stream is not None:
            close = getattr(self.stream, "close", None)
            if close is not None:
                close()
            self.stream = None
        self.pending.clear()


@contextmanager
def atomic_uel_writer(output_path: str, metadata: dict | None = None):
    """Write incrementally and expose the destination only after finalization."""

    directory = os.path.dirname(output_path) or "."
    os.makedirs(directory, exist_ok=True)
    basename = os.path.basename(output_path)
    temporary = os.path.join(directory, f".{basename}.{uuid.uuid4().hex}.tmp")
    try:
        with IncrementalSafetensorsWriter(
            temporary,
            metadata=metadata or {},
            max_workers=1,
        ) as writer:
            yield writer
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def stream_work_units(handlers, work_units, *, pin_memory: bool):
    """Yield bounded logical units from one ordered stream per input handler."""

    planned = {source: [] for source in handlers}
    for _, entries in work_units:
        for source, keys in entries.items():
            planned[source].extend(keys)
    cursors = {
        source: AsyncTensorCursor(handler, planned[source], pin_memory=pin_memory)
        for source, handler in handlers.items()
    }
    try:
        for logical_key, entries in work_units:
            loaded = {}
            consumed = []
            try:
                for source, keys in entries.items():
                    cursor = cursors[source]
                    for key in keys:
                        loaded[(source, key)] = cursor.take(key)
                        consumed.append((source, key))
                yield logical_key, loaded
            finally:
                loaded.clear()
                for source, key in consumed:
                    cursors[source].release(key)
        for cursor in cursors.values():
            cursor.finish()
    finally:
        for cursor in cursors.values():
            cursor.close()
