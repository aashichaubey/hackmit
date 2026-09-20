# Annotation review packet — development only

Please assess the draft labels against the source, without referring to model answers. This is a selected diagnostic sample, not an independent qualification corpus. Full excerpts are included because absence/unknown labels require checking the whole source.

For each item, record: reviewer identity, accept/correct/uncertain, corrected label if needed, and source-based rationale. Changes belong in a new dataset version; original frozen results must remain intact.

## dev-01:fact-0:deadline

The deadline matches a section heading but is absent from the quoted fact. Check whether it is a meeting date rather than a task deadline.

Draft annotation:
```json
{
  "kind": "action",
  "text": "Łukasz helping coordinate a core sprint there",
  "person": "Łukasz",
  "deadline": "2023-06-05",
  "source_quote": "Łukasz helping coordinate a core sprint there."
}
```

Reviewer: ___  Decision: ___  Corrected label: ___

Rationale: ___

## dev-02:holistic_questions:dev-02-q04:alias

A shorter name occurs in the source but is absent from accepted answers. Check that it uniquely identifies the same person.

Draft annotation:
```json
{
  "id": "dev-02-q04",
  "question": "Who is assigned to evaluate how to require code reviews for feature PRs?",
  "answers": [
    "Łukasz, the Developer-in-Residence",
    "The DiR"
  ],
  "expected_found": true,
  "category": "action",
  "evidence": [
    {
      "start": 2990,
      "end": 3073,
      "quote": "The SC asked the DiR to evaluate how we could require code reviews for feature PRs."
    }
  ]
}
```

Candidate variant(s), **not accepted automatically**: Łukasz

Reviewer: ___  Decision: ___  Corrected label: ___

Rationale: ___

## dev-10:holistic_questions:dev-10-q05:yesno

The fixed instruction requires yes/no for explicit yes/no questions, but no such answer is accepted. Check question intent before editing.

Draft annotation:
```json
{
  "id": "dev-10-q05",
  "question": "Was there a decision made about the Code of Conduct violation by August 7, 2024?",
  "answers": [
    "Conclusion of the Code of Conduct violation"
  ],
  "expected_found": true,
  "category": "agreement",
  "evidence": [
    {
      "start": 61,
      "end": 104,
      "quote": "Conclusion of the Code of Conduct violation"
    }
  ]
}
```

Reviewer: ___  Decision: ___  Corrected label: ___

Rationale: ___

## Full source: dev-01

[Public source](https://github.com/python/steering-council/blob/11c75e79836dcfe21a0f3feb15a8f3215c18e8be/updates/2023-06-steering-council-update.md)

## 2023-06-05

- Three Steering Council members were available to meet today, Emily and Pablo were traveling. - The SC met with Łukasz, the Developer-in-Residence, and discussed: - The new Deputy Developer-in-Residence job description draft. - EuroPython coming up and Łukasz helping coordinate a core sprint there. - Discussed all our PEP 703 (Making the Global Interpreter Lock Optional in CPython) thoughts. - Discussed OpenSSL 3.x support backporting needs for older releases. - Discussed 3.12 and 3.13 release schedule planning with Thomas (release manager hat). - Discussed ongoing 3.12beta1 early findings as projects try it out. - Briefly discussed support for core developer mentorship with Deb.

## 2023-06-12

- The Steering Council discussed PEP 703 (Making the Global Interpreter Lock Optional in CPython) extensively, with a focus on establishing a process and criteria for evaluating and deciding on the PEP.

## 2023-06-19

- The Steering Council discussed PEP 703 (Making the Global Interpreter Lock Optional in CPython) and decided to find out if there’s consensus among Core Devs for the idea and the specific PEP, through a poll on discuss.python.org. - The SC decided to schedule office hours a half hour before the weekly meeting, as an experiment. - The SC approved Petr’s proposed changes to PEP 387 (Backwards Compatibility Policy).

## 2023-06-26

- The Steering Council met with Seth Larson, the new Security Developer-in-Residence. One of Seth’s tasks is to improve the process behind the Python Security Response Team (PSRT), and to investigate the PSRT or PSF becoming a MITRE Sub-CNA, so it has authority over the reporting and ranking of CVEs. Seth will be working mostly with the PSRT team, only planning to have email discussions or ad-hoc meetings with the SC for now. - The SC met with Łukasz, the Developer-in-Residence, and discussed: - Łukasz handling the macOS installer build as well as the Windows build for 3.12 beta 4. - His keynote at PyCon Colombia. - A new EdgeDB release that should alleviate CLA bot inefficiencies. - EuroPython, his planned talk, the Core Dev panel, and the sprints there. - The Deputy Developer-in-Residence job opening, which will be posted this week. - The SC discussed json.AttrDict, which was added without review or discussion, with previous proposals having been rejected, and agreed with the rollback of it for now, at least until it can be discussed on discuss.python.org and a consensus is reached. - The SC briefly discussed other open issues involving Python 3.12, leaving them to the Release Team to handle. - The SC officially invited Russel Keith-Magee to the Core Dev Sprint in Brno in October.

## Full source: dev-02

[Public source](https://github.com/python/steering-council/blob/11c75e79836dcfe21a0f3feb15a8f3215c18e8be/updates/2023-07-steering-council-update.md)

## 2023-07-03

- The Steering Council checked in on ongoing conduct issues and the steps to handle them. - The SC discussed an email that suggested an update to PEP 12 (Sample reStructuredText PEP Template) to formalize including acceptance and rejection rationale to PEPs once they have a decision. The SC thinks this is a good idea and will work on drafting this change. - The SC discussed dropping the python-committers mailing list. An email will be sent to gauge interest and usage of the mailing list.

## 2023-07-10

- The Steering Council met with Łukasz, the Developer-in-Residence, and discussed: - The Deputy Developer-in-Residence job, which has been posted and started receiving applications. - The 3.12 final beta release, which has a few important bug fix PRs that are open. - EuroPython, which is happening next week. Łukasz will be giving a talk and running a panel along with sprints - The SC continued to discuss support for core developer mentorship with Deb. The distinction between supporting mentorship for new core developers and improving the process of becoming a core developer was raised – we are specifically looking for how to support existing core developers in mentoring contributors. The SC decided to pause the discussion to solidify what we are looking for in this arrangement. - The SC discussed PEP 703 (Making the Global Interpreter Lock Optional in CPython). - The poll results are in and indicate that 87% of respondents think we should be actively looking to make Python free-threaded and 63% want to accept and support the maintenance cost of PEP 703. - Proceeding with caution, the SC decided to formalize a statement on our intention to accept the PEP, along with requirements and expectations for doing so. - The SC discussed two C API changes in 3.12 that did not follow PEP 387 (Backwards Compatibility Policy) to decide how they should be handled, namely: - Type Object’s `tp_dict` can now be `NULL`. This has limited impact and is likely acceptable. - `PyLongObject` implementation and `ob_digits`. The SC decided to roll back changes only in 3.12, not main (3.13) for now. Overall, this change is okay. - The SC checked in on requiring PR reviews; a poll has been opened.

## 2023-07-17

- The Steering Council checked in on ongoing conduct issues and the steps to handle them. - The SC received feedback from the community on the lack of use for the python-committers email list and decided to close it. - The SC continued to discuss PEP 703 (Making the Global Interpreter Lock Optional in CPython), with a focus on establishing a hard list of requirements for acceptance to be included in a forthcoming pre-announcement.

## 2023-07-24

- The Steering Council met with Łukasz, the Developer-in-Residence, and discussed: - How EuroPython went (both the conference and the sprints). - dtrace and how to test the probes so they keep functioning. - Miss Islington needs porting to GitHub Actions before we can switch over to requiring 2FA. - The SC asked the DiR to evaluate how we could require code reviews for feature PRs. - The SC discussed support for core developer mentorship, namely the budget around a proposal to do an inventory of the current mentorship program. - The SC gave our approval to the Security Developer-in-Residence for their CNA proposal. - The SC discussed updating non-borrowing C APIs to match borrowing ones. - The SC continued to discuss PEP 703 (Making the Global Interpreter Lock Optional in CPython).

## 2023-07-31

- The Steering Council confirmed that the python-committers email list has been closed. - The SC discussed the state of PEP 713 (Callable Modules) and PEP 702 (Marking deprecations using the type system). - The SC continued to discuss the hard requirements around PEP 703 (Making the Global Interpreter Lock Optional in CPython), now that the pre-acceptance has been pre-announced. - The SC discussed how to handle the interview process for the Deputy Developer-in-Residence position.

## Full source: dev-10

[Public source](https://github.com/python/steering-council/blob/11c75e79836dcfe21a0f3feb15a8f3215c18e8be/updates/2024-08-steering-council-update.md)

# Steering Council Updates for August 2024

## 2024-08-07

- Conclusion of the Code of Conduct violation
- Discussion on [PEP 667 - Consistent views of namespaces](https://peps.python.org/pep-0667/) semantic changes

## 2024-08-14

- Deliberation of the decision regarding the Code of Conduct violation
- Discussion on how to make the Steering Council more effective by suggesting a different body parallel with the Steering Council to deal with social matters
- Highlight on the need to make clarification on the following different bodies: Code of Conduct Working Group, Steering Council and Python Software Foundation
- Discussion on the resignation of a Core Python Developer
- Mentoring survey: still to be released
- Discussion on [PEP 741 - Python Configuration C API](https://peps.python.org/pep-0741/)

## 2024-08-21

- Discussion on [PEP 741 - Python Configuration C API](https://peps.python.org/pep-0741/)
- Discussion on having overlapping Steering Council terms for continuity’s sake
- Discussion on mentorship, its role and how the Steering Council can improve mentorship for some at-risk packages/libraries

## 2024-08-28

- Cleared out pending tasks and re-evaluated the priority tasks
- Accepted [PEP 741 - Python Configuration C API](https://peps.python.org/pep-0741/)
- Tabled discussion of [PEP 12 – Sample reStructuredText PEP Template](https://peps.python.org/pep-0012/) changes (regrouping rejection rationale, and adding a discussion of  publishing new modules to PyPI) for the Core Dev sprint
- Discussion on publishing to PyPI, specifically:
  - Getting approval
  - Process or flow of release
- Re-review of [PEP 667 - Consistent views of namespaces](https://peps.python.org/pep-0667/)
- Discussion on more explicit authority for the Release Manager’s role going forward
- Highlight on the ongoing [PEP 13 – Python Language Governance](https://peps.python.org/pep-0013/) discussion on potential changes
- Discussion on whether there should be a PEP for PyREPL
- Discussion on how to know when something needs a PEP or not and update [PEP 2 – Procedure for Adding New Modules](https://peps.python.org/pep-0002/)
