from typing import Literal
from pydantic import BaseModel, Field, field_validator

Kind = Literal["proposal", "decision", "action", "agreement", "disagreement"]

class Fact(BaseModel):
    kind: Kind
    text: str = Field(min_length=1)
    person: str = ""
    deadline: str = ""
    source_quote: str = Field(min_length=1)

    @field_validator("text", "person", "deadline")
    @classmethod
    def safe_field(cls, value):
        if "|" in value or "\n" in value or "\r" in value:
            raise ValueError("Fact fields cannot contain | or line breaks")
        return value

class MeetingFacts(BaseModel):
    facts: list[Fact]

class Answer(BaseModel):
    found: bool
    answer: str

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
