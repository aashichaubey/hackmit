EXTRACTION_INSTRUCTION = "Extract only explicit meeting proposals, decisions, actions, agreements, and disagreements. In text, keep only the proposal subject, decision subject, action task, or agreement topic; put people and deadlines in their separate fields. Copy a short source quote for each fact. Do not turn a proposal into a decision. Keep people, task text, and dates literal. Omit uncertain facts."
ANSWER_INSTRUCTION = "Answer the question using only the meeting facts below. Treat the person assigned an action as its owner. For who/when questions, return only the name or date. For yes/no questions, use exactly 'yes' or 'no' when explicit. If the answer is absent or uncertain, set found to false and answer to 'not found'. Do not infer a decision from a proposal."

def extraction_prompt(notes: str) -> str:
    return f"{EXTRACTION_INSTRUCTION}\n<meeting>\n{notes}\n</meeting>"

def answer_prompt(context: str, question: str, legend: str = "") -> str:
    grammar = f"\nGrammar: {legend}" if legend else ""
    return f"{ANSWER_INSTRUCTION}{grammar}\n<meeting_facts>\n{context}\n</meeting_facts>\nQuestion: {question}"
