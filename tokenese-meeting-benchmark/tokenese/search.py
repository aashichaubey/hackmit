from .encode import profiles, render_english, render_tokenese, tokenese_legend
from .prompts import answer_prompt
from .tokens import count_tokens
from .facts import MeetingFacts

def qualify_candidate(result):
    return not result.get("critical_errors") and result.get("answer_accuracy", 0) >= result.get("english_accuracy", 1) - 1 / 24

def screen_profiles(cases, model):
    results = []
    for name, profile in profiles().items():
        savings = []
        hard_failures = 0
        for case in cases:
            facts = MeetingFacts(facts=case["facts"])
            lines = render_tokenese(facts, name).splitlines()
            for fact, line in zip(facts.facts, lines):
                parts = line.split(profile["separator"])
                if parts[0] != profile["labels"][fact.kind] or any(value and value not in parts for value in (fact.person, fact.text)) or (fact.deadline and profile["deadline_marker"] + fact.deadline not in parts):
                    hard_failures += 1
            for question in case["questions"]:
                english = count_tokens(answer_prompt(render_english(facts), question["question"]), model)
                encoded = count_tokens(answer_prompt(render_tokenese(facts, name), question["question"], tokenese_legend(name)), model)
                savings.append(english - encoded)
        results.append({"profile": name, "version": profile["version"], "hard_failures": hard_failures, "mean_token_savings": sum(savings) / len(savings), "cheaper_questions": sum(x > 0 for x in savings), "total_questions": len(savings)})
    return results

def search_candidates(cases, model):
    screened = screen_profiles(cases, model)
    survivors = [item["profile"] for item in screened if item["hard_failures"] == 0 and item["cheaper_questions"] > 0]
    return {"screened": screened, "survivors": survivors, "status": "awaiting answer-quality validation" if survivors else "no token-cheaper candidate"}
