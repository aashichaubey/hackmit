import tiktoken
from . import MODEL

def count_tokens(text: str, model: str = MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
