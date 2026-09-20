import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

class CachedResponses:
    def __init__(self, client, path):
        self.client = client
        self.path = Path(path)
        self.values = json.loads(self.path.read_text()) if self.path.exists() else {}

    def parse(self, *, model, input, text_format):
        key = hashlib.sha256(json.dumps([model, input, text_format.model_json_schema()], sort_keys=True).encode()).hexdigest()
        if key not in self.values:
            response = self.client.responses.parse(model=model, input=input, text_format=text_format)
            if response.output_parsed is None or response.usage is None:
                return response
            self.values[key] = {"parsed": response.output_parsed.model_dump(), "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.values, indent=2) + "\n")
        saved = self.values[key]
        return SimpleNamespace(output_parsed=text_format.model_validate(saved["parsed"]), usage=SimpleNamespace(input_tokens=saved["input_tokens"], output_tokens=saved["output_tokens"]))

class CachedClient:
    def __init__(self, client, path):
        self.responses = CachedResponses(client, path)
