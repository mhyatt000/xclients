from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Thread
from typing import cast, Generic, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@dataclass(frozen=True)
class LatestResult(Generic[TOut]):
    seq: int
    value: TOut | None = None
    error: BaseException | None = None


class LatestWorker(Generic[TIn, TOut]):
    def __init__(self, fn: Callable[[TIn], TOut], *, name: str = "latest-worker") -> None:
        self._fn = fn
        self._cv = Condition()
        self._pending: TIn | None = None
        self._has_pending = False
        self._pending_seq = 0
        self._result: LatestResult[TOut] | None = None
        self._closed = False
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, value: TIn) -> int:
        with self._cv:
            if self._closed:
                raise RuntimeError("LatestWorker is closed")
            self._pending_seq += 1
            self._pending = value
            self._has_pending = True
            self._cv.notify()
            return self._pending_seq

    def latest(self) -> LatestResult[TOut] | None:
        with self._cv:
            return self._result

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()
        self._thread.join()

    def __enter__(self) -> LatestWorker[TIn, TOut]:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._has_pending and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                value = self._pending
                seq = self._pending_seq
                self._pending = None
                self._has_pending = False

            try:
                result = LatestResult(seq=seq, value=self._fn(cast(TIn, value)))
            except BaseException as exc:
                result = LatestResult[TOut](seq=seq, error=exc)

            with self._cv:
                self._result = result
