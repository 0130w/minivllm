from dataclasses import dataclass


@dataclass(slots=True)
class Config:
    model: str
    max_num_seqs: int = 512
    kvcache_block_size: int = 256
