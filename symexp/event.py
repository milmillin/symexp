from typing import Generic, ParamSpec, Callable

_P = ParamSpec("_P")


class Event(Generic[_P]):
    def __init__(self):
        self._callbacks: list[Callable[_P, None]] = []

    def invoke(self, *args: _P.args, **kwargs: _P.kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)

    def register(self, callback: Callable[_P, None]):
        self._callbacks.append(callback)

    def deregister(self, callback: Callable[_P, None]):
        self._callbacks.remove(callback)
