"""Transparent synthetic development stressors, never public-source test evidence."""
from .evo_data import DATA, validate_contrast_pair
from .evo_search import save


def make_contrasts():
    pairs = []
    for i in range(20):
        variant = i % 5
        person, other = (("Alice", "Bo") if i < 10 else ("Zeynep", "Łukasz"))
        task = "ship the API" if i % 2 == 0 else "publish the report"
        fact = {"kind":"action", "person":person, "text":task, "deadline":"next Friday"}
        if variant == 0:
            field, value, category = "person", other, "speaker_assignee"
            question, answers = f"Who is assigned to {task}?", ([person], [other])
            def sentence(f): return f"Morgan: {f['person']} will {f['text']} by {f['deadline']}."
        elif variant == 1:
            field, value, category = "deadline", "this Friday", "date_modifier"
            question, answers = f"When must {person} {task}?", (["next Friday"], ["this Friday"])
            def sentence(f): return f"{f['person']} will {f['text']} by {f['deadline']}."
        elif variant == 2:
            fact.update(kind="proposal", deadline="")
            field, value, category = "kind", "decision", "proposal"
            question, answers = f"Was the plan to {task} decided?", (["not found"], ["yes"])
            def sentence(f): return f"{f['person']} {'proposed' if f['kind']=='proposal' else 'decided'} to {f['text']}."
        elif variant == 3:
            fact.update(kind="agreement", text="the launch plan", deadline="")
            field, value, category = "kind", "disagreement", "negation"
            question, answers = f"Did {person} agree with the launch plan?", (["yes"], ["no"])
            def sentence(f): return f"{f['person']} {'agreed' if f['kind']=='agreement' else 'disagreed'} with {f['text']}."
        else:
            field, value, category = "text", "not " + task, "negation"
            question, answers = f"Will {person} {task}?", (["yes"], ["no"])
            def sentence(f): return f"{f['person']} will {f['text']} by {f['deadline']}."
        altered = {**fact, field:value}
        sides = []
        for side, f in enumerate((fact, altered)):
            notes = sentence(f)
            sides.append({"notes":notes, "facts":[{**f, "source_quote":notes}], "question":question,
                          "answers":answers[side], "expected_found":answers[side] != ["not found"]})
        pair = {"id":f"contrast-{i+1:02}", "split":"dev", "review_status":"deterministic_checked",
                "field":field, "category":category, "before":sides[0], "after":sides[1]}
        validate_contrast_pair(pair)
        pairs.append(pair)
    return pairs


def contrast_cases(pairs):
    cases = []
    for pair in pairs:
        validate_contrast_pair(pair)
        for side, other in (("before","after"),("after","before")):
            source = pair[side]
            case_id = pair['id'] + '-' + side
            cases.append({"id":case_id,"split":"dev","notes":source['notes'],"facts":source['facts'],
                "provenance":{"source_id":pair['id'],"series_id":"synthetic-development-contrasts",
                    "url":"local:tokenese/evo_contrasts.py", "annotation_status":"deterministic_checked", "scope":"synthetic diagnostics, not public-source evidence"},
                "questions":[{"id":case_id+'-q', "question":source['question'],"answers":source['answers'],
                    "expected_found":source['expected_found'],"category":pair['category'],"contrast_id":pair['id']+'-'+other+'-q',
                    "evidence":[{"start":0,"end":len(source['notes']),"quote":source['notes']}]}]})
    return cases


if __name__ == '__main__':
    save(DATA / 'counterexamples.json', {'split':'dev','pairs':make_contrasts()})
