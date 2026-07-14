class LLMEngine:
    def __init__(self, model_path):
        pass

    def add_request(self, prompt):
        pass

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
