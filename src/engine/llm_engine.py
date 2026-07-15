from transformers import AutoTokenizer
from dataclasses import fields
from src.config import Config
from src.engine.scheduler import Scheduler
from src.engine.sequence import Sequence


class LLMEngine:
    def __init__(self, model: str, **kwargs):
        conf_fields = {field.name for field in fields(Config)}
        conf_kwargs = {k: v for k, v in kwargs.items() if k in conf_fields}
        conf = Config(model, **conf_kwargs)
        self.scheduler = Scheduler(conf)
        self.tokenizer = AutoTokenizer.from_pretrained(
            conf.model, use_fast=True
        )  # specify use_fast=True to use rust tokenizer
        Sequence.block_size = conf.kvcache_block_size

    def add_request(self, prompt: str):
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(token_ids)
        self.scheduler.add(seq)

    def is_finished(self):
        self.scheduler.is_finished()

    def step(self):
        seqs = self.scheduler.schedule()

    def generate(self, prompts: list[str]) -> list[str]:
        for prompt in prompts:
            self.add_request(prompt)

        outputs = {}
        while not self.is_finished():
            output = self.step()
        raise NotImplementedError
