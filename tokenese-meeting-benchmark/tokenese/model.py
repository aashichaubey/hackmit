from .facts import Answer, MeetingFacts, Usage
from .prompts import extraction_prompt

def parse_response(client, prompt, model, shape):
    response = client.responses.parse(model=model, input=prompt, text_format=shape)
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("No structured output returned")
    usage = response.usage
    if usage is None:
        raise ValueError("No API token usage returned")
    return parsed, Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)

def extract_facts(client, notes: str, model: str):
    if not notes.strip():
        raise ValueError("Meeting notes are blank")
    facts, usage = parse_response(client, extraction_prompt(notes), model, MeetingFacts)
    if any(fact.source_quote not in notes for fact in facts.facts):
        raise ValueError("An extracted source quote is absent from the meeting notes")
    return facts, usage

def answer_question(client, prompt: str, model: str):
    return parse_response(client, prompt, model, Answer)
