import torch
from dataclasses import dataclass


@dataclass(slots=True)
class Context:
    is_prefill: bool = True


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(context: Context):
    global _CONTEXT
    _CONTEXT = context


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
