import os
import time
from .prompts import answer_prompt
from .tokens import count_tokens

class BearBaseline:
    def __init__(self):
        from thetokencompany import TheTokenCompany
        key = os.getenv("TOKEN_COMPANY_API_KEY")
        if not key:
            raise ValueError("TOKEN_COMPANY_API_KEY is required for Bear-2")
        self.client = TheTokenCompany(api_key=key)
        self.cache = {}

    def compress(self, notes: str, question: str, target_tokens: int, model: str):
        start = time.monotonic()
        choices = []
        trial_input_tokens = 0
        trial_output_tokens = 0
        for aggressiveness in (0.15, 0.3, 0.5):
            key = (notes, aggressiveness)
            if key not in self.cache:
                self.cache[key] = self.client.compress(notes, model="bear-2", aggressiveness=aggressiveness)
                trial_input_tokens += self.cache[key].input_tokens
                trial_output_tokens += self.cache[key].output_tokens
            response = self.cache[key]
            context = response.output
            tokens = count_tokens(answer_prompt(context, question), model)
            choices.append((abs(tokens - target_tokens), context, tokens, aggressiveness, response))
        _, context, tokens, aggressiveness, response = min(choices, key=lambda item: item[0])
        return {"context": context, "prompt_tokens": tokens, "aggressiveness": aggressiveness, "seconds": time.monotonic() - start, "bear_input_tokens": response.input_tokens, "bear_output_tokens": response.output_tokens, "bear_trial_input_tokens": trial_input_tokens, "bear_trial_output_tokens": trial_output_tokens}
