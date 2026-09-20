"""Build the deterministic, balanced benchmark_v1 dataset."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "benchmark_v1.json"


def qa(question: str, answer: str) -> dict:
    return {"question": question, "expected_answer": answer}


ACADEMIC = [
    ("Dr. Alvarez", "Monday", "ocean circulation", "thermohaline circulation is driven by density differences caused by temperature and salinity", "surface currents are mainly driven by wind", "Fridtjof Nansen", "chapter 9", "Thursday at 15:00", "the Coriolis effect changes direction but does not initiate the flow"),
    ("Professor Mensah", "Tuesday", "behavioral economics", "loss aversion means losses often feel more significant than equal-sized gains", "a sunk cost should not affect a forward-looking decision", "Daniel Kahneman and Amos Tversky", "pages 112–138", "Friday at 10:30", "the classroom example assumed participants understood the probabilities"),
    ("Dr. Iqbal", "Wednesday", "database systems", "a transaction should satisfy atomicity, consistency, isolation, and durability", "an index speeds reads but can add write overhead", "Edgar F. Codd", "chapter 12", "Monday at 14:00", "serializable isolation was discussed as the strongest standard level"),
    ("Professor Sato", "Thursday", "ecology", "keystone species have effects disproportionately large relative to their abundance", "primary succession begins without established soil while secondary succession does not", "Robert Paine", "lab worksheet 4", "Tuesday at 16:15", "the island example excluded newly introduced invasive species"),
    ("Dr. Bennett", "Friday", "constitutional law", "judicial review allows courts to assess whether laws conflict with a constitution", "binding precedent differs from persuasive authority", "John Marshall", "case brief 6", "Wednesday at 13:30", "the hypothetical assumed federal rather than state jurisdiction"),
]


def academic(i, row):
    teacher, day, topic, central, contrast, person, assignment, office, caveat = row
    context = f"{teacher}'s {day} lecture surveyed {topic} and began by connecting the topic to the previous unit. Students first reviewed the terminology on the opening slide and compared two examples from the assigned reading. The main proposition was that {central}. {teacher} returned to that point after a diagram and explained how it affects interpretation of later examples. A related distinction was that {contrast}. The distinction matters because similar-looking cases can require different explanations. During the historical section, the lecture associated a foundational contribution with {person}, while noting that later researchers refined the original account. A short demonstration followed, and students recorded observations before discussing sources of measurement or reasoning error. The instructor emphasized a methodological caution: {caveat}. That caution limits the conclusion but does not invalidate the main example. Questions near the end covered terminology, the contrast between the two cases, and how assumptions change an analysis. The required follow-up is {assignment}, due before the next class. The normal help session was rescheduled, so office hours will be {office}. Slides and a practice problem are posted on the course site, and the optional reading provides additional background rather than replacing the required assignment."
    questions = [qa("What was the lecture's main proposition?", central), qa("What related distinction was emphasized?", contrast), qa("Which person was associated with the foundational contribution?", person), qa("When are the rescheduled office hours?", office), qa("What methodological caution limited the example?", caveat)]
    return context, questions


MEETINGS = [
    ("Atlas redesign", "Mira", "April 18", "the accessibility audit", "Noah", "May 2", "$8,000", "vendor photography requires legal approval", "weekly on Wednesdays"),
    ("community garden launch", "Elena", "June 6", "the irrigation plan", "Marcus", "June 20", "$3,500", "soil delivery depends on the weather", "every other Monday"),
    ("library catalog migration", "Priya", "August 14", "the duplicate-record review", "Jon", "September 1", "$12,000", "rare-book records cannot be auto-merged", "Tuesdays at 09:30"),
    ("winter conference", "Sam", "October 9", "the speaker confirmations", "Aisha", "October 23", "$6,200", "travel bookings wait until contracts are signed", "Fridays at 11:00"),
    ("mobile app pilot", "Leo", "January 12", "the consent-flow prototype", "Rina", "January 26", "$9,500", "pilot analytics must exclude personal identifiers", "Thursdays at 14:00"),
]


def meeting(i, row):
    project, lead, review_date, deliverable, owner, due, budget, dependency, cadence = row
    context = f"The working group met for forty-five minutes to review the {project}. {lead} facilitated and began with decisions carried over from the previous session. The team agreed that the next formal review will occur on {review_date}, with a short written update circulated one day earlier. The highest-priority deliverable is {deliverable}. {owner} owns that work and committed to completing it by {due}. Other members will provide source material, but ownership remains with {owner}. Finance confirmed a remaining discretionary budget of {budget}; spending above that amount requires steering-committee approval. The group chose to reserve part of the budget for testing rather than use all of it on launch materials. One dependency could change the schedule: {dependency}. If that dependency slips, the owner should flag it in the shared tracker rather than silently moving the due date. The communications discussion produced two decisions: use plain-language status labels and send external announcements only after the formal review. The team rejected a proposal to add a second approval meeting because it would duplicate the steering review. Progress meetings will recur {cadence}. Notes will be stored with the project files, and unresolved risks will remain visible until an owner closes them with evidence. The meeting ended with a reminder that dates in the tracker are authoritative when they conflict with informal chat messages."
    return context, [qa("Who facilitated the meeting?", lead), qa("When is the next formal review?", review_date), qa("Who owns the highest-priority deliverable?", owner), qa("What is the remaining discretionary budget?", budget), qa("What dependency could change the schedule?", dependency)]


POLICIES = [
    ("equipment lending", "staff and registered students", "14 days", "one 7-day renewal", "$15 per day", "medical equipment", "written department approval"),
    ("remote-work", "full-time employees", "three days per week", "manager approval", "within 24 hours", "positions requiring daily laboratory access", "a documented accommodation"),
    ("visitor access", "sponsored visitors", "08:00–18:00", "an active temporary badge", "30 minutes", "restricted archives", "escort by an authorized archivist"),
    ("training reimbursement", "employees past probation", "$1,200 per year", "preapproval from a supervisor", "60 days", "recreational courses", "direct job relevance documented by the director"),
    ("data retention", "project teams", "three years", "encrypted institutional storage", "72 hours", "legal-hold records", "written release from counsel"),
]


def policy(i, row):
    name, eligible, limit, requirement, penalty, exception_item, exception_condition = row
    context = f"The organization adopted this {name} policy to make routine decisions consistent across departments. It applies to {eligible}, provided all required records are current. The standard limit is {limit}. Requests beyond that limit are not automatically denied, but they must be reviewed by the designated administrator before any commitment is made. The ordinary requirement is {requirement}; verbal permission does not substitute for the recorded approval. Missing a required return, response, or reporting deadline triggers a consequence of {penalty}. Staff should document the event in the central system so that later reviewers can distinguish an approved exception from an oversight. The policy does not apply in the usual way to {exception_item}. That case is permitted only with {exception_condition}. Emergencies may justify immediate protective action, but the responsible person must still create a record afterward. Supervisors may clarify procedures but cannot waive eligibility rules on their own. Appeals go to the operations director within ten business days and must identify the disputed decision. Annual audits sample both approved and rejected requests. Personal convenience alone is not a sufficient reason for an exception. When two versions of the policy appear to conflict, the version in the controlled handbook governs. Questions should be sent to operations rather than resolved through informal precedent."
    return context, [qa("Who is eligible under the policy?", eligible), qa("What is the standard limit?", limit), qa("What is the ordinary requirement?", requirement), qa("What consequence follows a missed deadline?", penalty), qa(f"Under what condition is the exception for {exception_item} permitted?", exception_condition)]


SCHEDULES = [
    ("innovation summit", "May 11", "08:15", "Room 4-270", "11:40", "Kresge Auditorium", "15:10", "Lobby 7", "17:30", "every 20 minutes"),
    ("public history forum", "June 22", "09:00", "Archive Hall", "12:15", "River Room", "14:45", "Gallery B", "18:00", "every 30 minutes"),
    ("robotics open house", "July 8", "10:20", "Maker Bay 2", "13:00", "Demo Arena", "16:35", "Lab 5", "19:10", "every 15 minutes"),
    ("climate workshop", "September 16", "08:40", "Earth Sciences 102", "11:15", "Green Auditorium", "15:30", "Map Room", "17:50", "every 25 minutes"),
    ("writers festival", "November 4", "09:30", "Library Theater", "12:45", "Courtyard Tent", "16:00", "Studio C", "20:00", "every 40 minutes"),
]


def schedule(i, row):
    event, date, start, room, keynote, keyloc, session, sessloc, close, shuttle = row
    context = f"The {event} takes place on {date}. Registration opens at {start} in {room}, where attendees can collect badges and revised maps. The welcome remarks begin twenty minutes later in the same room. Two morning sessions then run in parallel; participants may switch rooms only during the scheduled break to avoid disrupting speakers. The featured keynote starts at {keynote} in {keyloc}, and doors close five minutes after it begins. Lunch follows in the central dining area, with dietary requests identified on badge records rather than at the serving line. An afternoon hands-on session begins at {session} in {sessloc}. Because equipment must remain calibrated, late arrivals to that session may observe but cannot join the first exercise. A poster walk and moderated discussion follow. The closing program begins at {close}; awards are announced near its end. Shuttle service from the main entrance runs {shuttle} beginning thirty minutes before registration and continues until one hour after closing. Lost badges are replaced at registration after identity verification, while other lost property goes to venue security. The online program is authoritative for room changes, but the date and listed keynote time will not change. Volunteers should arrive forty-five minutes before registration."
    return context, [qa("On what date does the event take place?", date), qa("Where does registration open?", room), qa("When does the keynote start?", keynote), qa("Where is the afternoon hands-on session?", sessloc), qa("How frequently does the shuttle run?", shuttle)]


TECHNICAL = [
    ("Orchid API", "HTTPS on port 443", "X-Orchid-Key", "120 requests per minute", "429", "24 hours", "/v2/jobs/{id}"),
    ("Beacon sensor", "Bluetooth Low Energy", "Beacon-Token", "10 readings per second", "E17", "30 days", "/device/status"),
    ("Cedar backup agent", "TLS 1.3", "Cedar-Auth", "4 concurrent uploads", "B04", "14 days", "/api/backups/{id}"),
    ("Nimbus queue", "AMQP on port 5671", "Nimbus-Key", "500 messages per second", "Q09", "7 days", "/queues/{name}/metrics"),
    ("Quartz renderer", "gRPC on port 7443", "Quartz-Credential", "8 jobs per worker", "R12", "48 hours", "/render/jobs/{id}"),
]


def technical(i, row):
    system, protocol, header, rate, error, retention, endpoint = row
    context = f"The {system} service is intended for internal applications that need predictable automated access. Clients connect using {protocol}. Every request must include the credential in the {header} header; credentials in query strings are rejected because URLs may be logged. The default service limit is {rate} per project. Bursts above the limit receive error {error}, and clients should retry with exponential backoff rather than immediately repeat the request. Successful create operations return an identifier that can be checked at {endpoint}. Status responses distinguish queued, running, completed, and failed work. Clients should treat unknown fields as forward-compatible additions and must not depend on field ordering. Completed operational records are retained for {retention}, after which the service may remove them without another notice. Deleting a client-side reference does not cancel server-side work. Cancellation is accepted only while a job is queued or running, and a completed job cannot be reopened. Timestamps use UTC in ISO 8601 form. Text is UTF-8, numeric sizes are bytes, and an omitted optional value differs from an explicit zero. Client logs should retain request identifiers so support can trace failures without receiving credential values. The service team publishes maintenance windows at least two business days ahead when possible. Sandbox credentials cannot access production records."
    return context, [qa("Which protocol does the service use?", protocol), qa("Which header carries the credential?", header), qa("What is the default service limit?", rate), qa("Which error indicates the limit was exceeded?", error), qa("How long are completed operational records retained?", retention)]


FACTUAL = [
    ("Lake Baikal", "Siberia", "about 1,642 meters", "roughly 20 percent", "the Angara River", "winter ice", "a UNESCO World Heritage Site in 1996"),
    ("the Atacama Desert", "northern Chile", "some stations record years without measurable rain", "salt flats and volcanic terrain", "the Pacific subtropical high", "large daily temperature changes", "major astronomical observatories"),
    ("the Great Barrier Reef", "off northeastern Australia", "more than 2,300 kilometers", "thousands of individual reefs", "coral bleaching", "warm shallow seas", "World Heritage listing in 1981"),
    ("the Danube River", "central and southeastern Europe", "about 2,850 kilometers", "ten countries", "the Black Sea", "seasonal flooding", "the Danube Delta Biosphere Reserve"),
    ("Mauna Loa", "the island of Hawaiʻi", "4,169 meters above sea level", "a broad shield volcano", "the Pacific Plate hotspot", "long fluid lava flows", "systematic atmospheric CO2 measurements nearby"),
]


def factual(i, row):
    subject, location, measure, distinction, outlet, feature, designation = row
    context = f"{subject} is located in {location} and is widely studied for both physical and ecological reasons. A commonly cited measurement is {measure}, although exact values can vary slightly with the method and reference point used. The feature is also notable for {distinction}. Its present form reflects long-term geological and climatic processes rather than a single event. Water, wind, ice, or volcanic activity continues to shape the surrounding landscape. A particularly important connected feature is {outlet}. Seasonal observations often focus on {feature}, which affects access, habitats, or measurement conditions. Scientists use satellite data together with field instruments because neither source alone captures every relevant scale. Repeated measurements use documented reference points so that changes can be compared across years. Local communities have long histories connected to the region, and modern conservation decisions therefore involve cultural as well as environmental considerations. The site is associated with {designation}. That status draws attention but does not by itself eliminate pollution, climate, tourism, or development pressures. Researchers caution against treating one annual measurement as a complete trend. Educational displays often simplify the system, while technical reports separate measured values from estimates. Visitors are asked to follow local access guidance and avoid disturbing monitored areas."
    return context, [qa("Where is the subject located?", location), qa("What commonly cited measurement is given?", measure), qa("What is it also notable for?", distinction), qa("What connected feature is identified?", outlet), qa("What designation or recognized association is mentioned?", designation)]


PROCEDURES = [
    ("calibrate the water-quality probe", "rinse the sensor with deionized water", "pH 7 buffer", "wait until the reading changes by less than 0.02 for 30 seconds", "record the slope", "repeat with pH 4 buffer", "do not wipe the glass bulb"),
    ("archive a completed project", "confirm all deliverables are final", "create a read-only snapshot", "verify its checksum against the manifest", "record the archive identifier", "move working drafts to the temporary folder", "do not delete legal-hold material"),
    ("prepare the laser cutter", "inspect the exhaust filter", "place material flat on the bed", "run the focus gauge at the material surface", "record the material preset", "perform a low-power outline test", "never cut PVC"),
    ("receive a refrigerated shipment", "inspect the temperature indicator", "compare the packing list with the order", "photograph any damaged seal before opening", "record the arrival temperature", "transfer contents to labeled cold storage", "do not accept an unlabeled specimen"),
    ("publish the monthly report", "lock the reporting period", "reconcile totals with the ledger", "have a second analyst verify exceptions", "record the approval timestamp", "export both PDF and accessible HTML", "do not include customer identifiers"),
]


def procedure(i, row):
    task, first, second, check, record, nextstep, warning = row
    context = f"Use the following controlled procedure to {task}. Before starting, confirm that the work area is available and that the current procedure revision is displayed. The first operational step is to {first}. If that inspection or confirmation fails, stop and resolve the issue rather than continuing with incomplete information. Next, {second}. Keep unrelated items away from the work surface so they cannot be confused with the materials in use. Then {check}. This verification is the acceptance checkpoint; a result outside the stated condition requires repeating the preceding preparation step. Once the checkpoint passes, {record} in the designated log. The log entry should identify the operator and use the current date. After recording, {nextstep}. Confirm the result is readable and associated with the correct project or sample. The most important prohibition is: {warning}. This restriction applies even when the schedule is tight. If equipment, packaging, or source records appear damaged, isolate the item and notify the responsible coordinator. Do not improvise a repair unless the procedure explicitly authorizes it. At completion, restore shared tools, remove temporary materials, and mark the task complete. A second person is required only when the acceptance checkpoint or local safety notice says so."
    return context, [qa("What is the first operational step?", first), qa("What is the next preparation step?", second), qa("What is the acceptance checkpoint?", check), qa("What must be recorded?", record), qa("What action is explicitly prohibited?", warning)]


PROJECTS = [
    ("Harbor dashboard", "green", "72 percent", "data ingestion", "June 28", "schema changes from the vendor", "Nina", "$24,000"),
    ("Maple renovation", "amber", "48 percent", "electrical rough-in", "August 9", "permit review taking longer than planned", "Owen", "$61,500"),
    ("Civic survey", "green", "83 percent", "response cleaning", "March 15", "lower participation from two districts", "Fatima", "$18,200"),
    ("Orion migration", "red", "39 percent", "identity integration", "November 3", "legacy accounts missing stable identifiers", "Chen", "$47,000"),
    ("Studio curriculum", "amber", "65 percent", "instructor review", "January 19", "two modules need accessibility revisions", "Grace", "$13,750"),
]


def project(i, row):
    project, status, complete, milestone, due, risk, owner, budget = row
    context = f"The latest status report covers the {project}. Overall status is {status}, and the team estimates {complete} of planned work is complete. Progress since the previous report includes closing several review comments and updating the shared schedule to reflect actual completion dates. The next major milestone is {milestone}, due {due}. Work feeding that milestone is underway, but reviewers should not treat draft outputs as final. The leading risk is {risk}. The risk remains open because its trigger has not been eliminated, although the team has documented a mitigation and a fallback. {owner} is accountable for coordinating the mitigation and reporting any change in severity. The approved remaining budget is {budget}; committed amounts are tracked separately from invoices already paid. No scope change has been approved this period. Two low-priority requests were moved to the backlog so the milestone team can protect the due date. Quality checks found minor documentation gaps but no critical defect in completed work. Stakeholders will receive another update after the milestone review. If the leading risk occurs before then, the project owner must issue an interim notice within one business day. The report distinguishes schedule confidence from percent complete: a high completion percentage does not automatically mean the due date is safe."
    return context, [qa("What is the project's overall status?", status), qa("What percentage of planned work is complete?", complete), qa("What is the next major milestone?", milestone), qa("What is the leading risk?", risk), qa("Who owns coordination of the mitigation?", owner)]


BUSINESS = [
    ("North region", "$1.84 million", "12 percent", "Aster line", "38 percent", "Cedar account", "$210,000", "net 30"),
    ("South region", "$2.15 million", "8 percent", "Beacon line", "42 percent", "Harbor account", "$185,000", "net 45"),
    ("West region", "$1.62 million", "15 percent", "Cobalt line", "35 percent", "Juniper account", "$240,000", "net 30"),
    ("Central region", "$2.48 million", "5 percent", "Delta line", "31 percent", "Maple account", "$195,000", "net 60"),
    ("East region", "$1.97 million", "10 percent", "Ember line", "40 percent", "Orchid account", "$225,000", "net 45"),
]


def business(i, row):
    region, revenue, growth, product, margin, account, receivable, terms = row
    context = f"The quarterly business review summarizes the {region}. Recognized revenue was {revenue}, representing {growth} growth from the comparable quarter. The strongest product family was the {product}, supported by renewals as well as new orders. Gross margin for the region was {margin}; freight and temporary staffing were the largest deviations from plan. The largest named customer relationship in the report is the {account}. Its open receivable is {receivable} under {terms} payment terms, and the amount is not yet past due. Sales leadership expects normal collection but asked finance to monitor any change in the customer's purchasing schedule. New pipeline is spread across several industries, reducing dependence on a single pending deal. One proposed discount was declined because it would have reduced margin below the regional floor. Inventory availability improved during the quarter, although two specialty components still have longer lead times. The forecast assumes no major currency movement and no unplanned price increase from the primary carrier. Operating expenses remained within the approved envelope. Regional leaders will review collections and inventory again at the next monthly close. The review separates bookings from recognized revenue and excludes unsigned letters of intent. Managers agreed to revisit hiring only if two consecutive monthly forecasts remain above plan."
    return context, [qa("What recognized revenue did the region report?", revenue), qa("What was the year-over-year growth?", growth), qa("Which product family was strongest?", product), qa("What gross margin was reported?", margin), qa("What is the open receivable for the largest named account?", receivable)]


NARRATIVES = [
    ("Rosa", "the field station", "a blue sample case", "the north storage room", "Jamal", "the replacement bridge opened at 16:00", "the paper log remained in the main office"),
    ("Evan", "the community theater", "a brass key ring", "the costume workshop", "Maya", "the evening rehearsal moved to 19:30", "the lighting plan stayed with the stage manager"),
    ("Leila", "the coastal archive", "a sealed map tube", "shelf C-14", "Tom", "the ferry resumed service at 11:20", "the digital copy remained under review"),
    ("Marco", "the mountain lodge", "a red emergency radio", "the reception cabinet", "Inez", "the east trail reopened on Saturday", "the spare batteries stayed in the guide room"),
    ("Anika", "the neighborhood clinic", "a gray document pouch", "the records annex", "Paul", "the mobile unit departed at 07:45", "the signed receipt remained at reception"),
]


def narrative(i, row):
    person, place, item, destination, recipient, event, retained = row
    context = f"{person} arrived at {place} shortly after the morning briefing. The team had already reviewed the weather, checked the day's visitor list, and posted a revised notice near the entrance. {person} was carrying {item}, which had been signed out the previous afternoon for a scheduled task. After confirming the label with the coordinator, {person} placed it in {destination}. {recipient} witnessed the transfer and entered the time in the local register. The group then discussed a separate logistics update: {event}. That change affected travel plans but did not alter the location of the transferred item. During lunch, two visitors asked whether the item would be available for public viewing; the coordinator explained that access required an appointment. Later, a delivery arrived with routine supplies and was moved to a different room. Before leaving, {person} checked the register again and noticed that {retained}. This detail mattered because another staff member had expected it to travel with the item. The coordinator sent a short clarification and assigned a follow-up for the next workday. Nothing was reported missing or damaged. At closing, doors were checked in sequence, and the person on duty kept the register rather than sending it with the evening courier."
    return context, [qa("Who arrived after the morning briefing?", person), qa("What item was being carried?", item), qa("Where was the item placed?", destination), qa("Who witnessed the transfer?", recipient), qa("What separate logistics update was discussed?", event)]


CATEGORIES = [
    ("lecture_academic_notes", ACADEMIC, academic), ("meeting_notes", MEETINGS, meeting),
    ("policies_handbooks", POLICIES, policy), ("schedules_events", SCHEDULES, schedule),
    ("technical_documentation", TECHNICAL, technical), ("factual_passages", FACTUAL, factual),
    ("procedural_instructions", PROCEDURES, procedure), ("project_status_reports", PROJECTS, project),
    ("structured_business", BUSINESS, business), ("mixed_narrative_factual", NARRATIVES, narrative),
]


def build() -> list[dict]:
    items = []
    for category, rows, builder in CATEGORIES:
        for index, row in enumerate(rows, 1):
            context, questions = builder(index, row)
            items.append({"id": f"{category}-{index:02d}", "category": category, "context": context, "questions": questions})
    return items


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    items = build()
    OUTPUT.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} contexts and {sum(len(item['questions']) for item in items)} QA pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
