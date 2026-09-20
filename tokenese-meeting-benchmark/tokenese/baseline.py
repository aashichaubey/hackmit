import time
from .prompts import answer_prompt
from .tokens import count_tokens

class DeletionBaseline:
    def __init__(self):
        from llmlingua import PromptCompressor
        self.compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", device_map="cpu", use_llmlingua2=True)
        self.cache = {}

    def compress(self, notes: str, question: str, target_tokens: int, model: str):
        start = time.monotonic()
        choices = []
        for rate in (0.3, 0.5, 0.7):
            key = (notes, rate)
            if key not in self.cache:
                self.cache[key] = self.compressor.compress_prompt(notes, rate=rate)["compressed_prompt"]
            context = self.cache[key]
            tokens = count_tokens(answer_prompt(context, question), model)
            choices.append((abs(tokens - target_tokens), context, tokens, rate))
        _, context, tokens, rate = min(choices)
        return {"context": context, "prompt_tokens": tokens, "rate": rate, "seconds": time.monotonic() - start}
