from collections import deque

from src.config import Config
from src.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, conf: Config):
        self.max_num_seqs = conf.max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        # TODO:
        scheduled_seqs = []

        # prefill stage
        while len(scheduled_seqs) < self.max_num_seqs and self.waiting:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs

        # decode stage
        while len(scheduled_seqs) < self.max_num_seqs and self.running:
            seq = self.running.popleft()
            # TODO: 抢占
            seq.is_prefill = False
            scheduled_seqs.append(seq)

        return scheduled_seqs

    def postprocess(self):
        raise NotImplementedError
