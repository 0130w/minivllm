from transformers import AutoTokenizer
from src.engine.scheduler import Scheduler
from src.engine.sequence import Sequence


class LLMEngine:
    def __init__(self, model_path, conf):
        self.scheduler = Scheduler()
        self.tokenizer = AutoTokenizer.from_pretrained(
            conf.model, use_fast=True
        )  # specify use_fast=True to use rust tokenizer

    def add_request(self, prompt):
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(token_ids)
        self.scheduler.add(seq)

    def is_finished(self):
        pass

    def step(self):
        pass

    def generate(self, prompts: list[str]) -> list[str]:
        for prompt in prompts:
            self.add_request(prompt)

        outputs = {}
        while not self.is_finished():
            output = self.step()
        raise NotImplementedError
