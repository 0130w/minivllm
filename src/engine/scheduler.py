from collections import deque

from src.engine.sequence import Sequence


class Scheduler:
    def __init__(self):
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        raise NotImplementedError

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self):
        raise NotImplementedError
