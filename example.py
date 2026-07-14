from transformers import AutoTokenizer
from pathlib import Path
from src.llm import LLM


def main():
    path = Path("~/huggingface/Qwen3-0.6B/").expanduser()  # model path
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path)
    # TODO: sampling params
    messages = ["introduce yourself", "list all prime numbers within 100"]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": message},
            ],
            tokenize=False,  # return concatenated str if False, TokenID if True
            add_generation_prompt=True,  # add <|im_start|>assistant if True
        )
        for message in messages
    ]
    outputs = llm.generate(prompts)
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")  # var!r is equivalent to repr(var)
        print(f"Output: {output!r}")


if __name__ == "__main__":
    main()
