"""
Prompt templates for the Meetolog AI extraction pipeline.

Implements a modular, template-driven prompt engineering system across
five specialised extraction domains:

1. **Summarization** — Chunk and merge prompts for hierarchical
   Map-Reduce processing of long transcripts.
2. **Artifact Extraction** — Master prompt for comprehensive extraction
   of all Agile artifact categories with few-shot examples.
3. **Task Detection** — Focused prompt for explicit and inferred task
   extraction with granular few-shot guidance.
4. **Decision Detection** — Chain-of-Thought prompt for distinguishing
   confirmed decisions from discarded proposals.
5. **Meeting Summarization** — Structured summary for stakeholders.

Prompt Design Principles
------------------------
- **Role Prompting**: Each prompt assigns a domain-expert persona to
  anchor the LLM's behaviour.
- **Instructions over Constraints**: Affirmative directives (extract X,
  classify Y) rather than prohibitions (do not Z).
- **Few-Shot Examples**: Every extraction prompt includes a realistic
  input/output pair to lock formatting and quality.
- **Chain of Thought (CoT)**: Decision and idea extraction instruct
  step-by-step evaluation before final output.
- **Schema Enforcement**: Output schemas are specified inline; the
  validation layer in ``models.artifacts`` enforces them at runtime.

Public API
----------
``CHUNK_SUMMARIZATION_PROMPT``
    Template for per-chunk summarization (Map phase).
``MERGE_SUMMARIZATION_PROMPT``
    Template for merging sequential summaries (Reduce phase).
``RAG_AUGMENTED_EXTRACTION_CONTEXT``
    Template for injecting RAG context alongside condensed summaries.
``build_extraction_prompt(transcript)``
    Assemble the master artifact extraction prompt.
``build_task_detection_prompt(transcript)``
    Assemble the focused task detection prompt.
``build_decision_detection_prompt(transcript)``
    Assemble the Chain-of-Thought decision detection prompt.
``build_summarization_prompt(transcript)``
    Assemble the meeting-level summarization prompt.
"""

# ======================================================================
# DOMAIN 1: HIERARCHICAL SUMMARIZATION
# ======================================================================

CHUNK_SUMMARIZATION_PROMPT = """\
You are a senior technical meeting analyst specialising in Agile software \
delivery. Below is segment {chunk_index} of {total_chunks} from a longer \
meeting transcript.

TRANSCRIPT SEGMENT:
---
{chunk_text}
---

Produce a dense, information-preserving summary of this segment.

1. Retain every concrete detail:
   - Task assignments (who, what, when)
   - Decisions reached (what was agreed, by whom, why)
   - Blockers or impediments (what is blocked, who owns resolution)
   - Feature requests and user-story language ("As a …, I want …")
   - Action items and follow-ups
   - Participant names and roles
2. Preserve exact names, dates, numbers, priorities, and technical terms verbatim.
3. Organise in chronological order of discussion.
4. Remove filler words, pleasantries, and repetition — but never drop an \
actionable item, decision, or blocker.

Return the summary as plain text. Do not wrap in JSON or markdown headings.\
"""

MERGE_SUMMARIZATION_PROMPT = """\
You are a senior technical meeting analyst performing the merge step in a \
Hierarchical Summarization pipeline. Below are {num_summaries} sequential \
summaries covering consecutive sections of the same meeting. Overlapping \
context windows may cause some items to appear more than once.

SUMMARIES:
---
{combined_summaries}
---

Merge these into a single, unified summary of the entire meeting.

1. Retain every unique detail across all summaries:
   - Task assignments, decisions, blockers, user stories, action items, \
participant names and roles.
2. Deduplicate items that appear in multiple summaries — keep the most \
complete version of each.
3. Preserve exact names, dates, numbers, priorities, and technical terms.
4. Organise chronologically.
5. Compress filler and repetition — never drop an actionable item, \
decision, or blocker.

Return the merged summary as plain text. Do not wrap in JSON or markdown headings.\
"""

# ======================================================================
# DOMAIN 2: RAG-AUGMENTED EXTRACTION CONTEXT
# ======================================================================

RAG_AUGMENTED_EXTRACTION_CONTEXT = """\
## RETRIEVED TRANSCRIPT SEGMENTS

The following segments were retrieved via semantic search from the original \
full-length meeting transcript. They are the passages most relevant to the \
artifact categories you must extract. Treat them as PRIMARY evidence — they \
contain verbatim details that may have been compressed during summarization.

{rag_context}

## CONDENSED MEETING SUMMARY

The summary below was produced by a multi-pass hierarchical summarization \
pipeline. It covers the entire meeting but may have compressed fine-grained \
details.

{condensed_summary}

## SYNTHESIS INSTRUCTIONS

Combine both sources to produce the final structured artifacts JSON.

1. Retrieved segments take precedence for specific facts (names, dates, \
numbers, acceptance criteria, assignment details, resolution plans). If a \
detail appears in a retrieved segment but is absent from the summary, include it.
2. The condensed summary provides global context — meeting flow, overall \
decisions, and high-level themes. Use it to fill gaps not covered by \
retrieved segments.
3. Extract only what is supported by at least one source. If neither source \
mentions a fact, do not include it.
4. Deduplicate items that appear in both sources.\
"""

# ======================================================================
# DOMAIN 3: MASTER ARTIFACT EXTRACTION
# ======================================================================

_EXTRACTION_ROLE = """\
You are an elite Agile Business Analyst and Scrum Master with 15 years of \
experience in software delivery. You specialise in extracting precise, \
actionable artifacts from unstructured meeting discussions. Your extractions \
are used directly to populate a team's Agile backlog, so accuracy and \
completeness are paramount.

CRITICAL RULES:
- Extract ONLY what is explicitly stated or directly implied by the \
transcript. Never fabricate, hallucinate, or assume information not present.
- Before extracting each artifact, internally evaluate whether the \
transcript segment genuinely supports it. If the evidence is ambiguous, \
lower the confidence_score accordingly.
- Every field you populate must be traceable to a specific passage in the \
transcript. If a field's value cannot be traced, leave it null or empty."""

_EXTRACTION_INSTRUCTIONS = """\
Analyze the transcript systematically. For each artifact category below, \
first scan the transcript for matching evidence, then extract structured \
data only for items with clear support.

1. MEETING OVERVIEW
   - Infer a concise, descriptive title from the meeting content.
   - Write a 2-3 sentence summary capturing the meeting's purpose, key \
outcomes, and agreed next steps.
   - List every participant name mentioned in the transcript.

2. USER STORIES
   - Identify feature requests, user needs, and requirements.
   - Rewrite informal language into standard user-story format: \
"As a [role], I want [action], so that [benefit]."
   - Include acceptance criteria when discussed or clearly implied.
   - Estimate story points using the Fibonacci sequence (1, 2, 3, 5, 8, 13) \
based on the implied complexity and effort discussed.
   - Set priority based on urgency signals in the conversation.

3. TASKS
   - Extract specific work items, assignments, and to-dos.
   - Record the assignee when a name is explicitly stated.
   - Set priority based on deadlines, dependencies, and explicit urgency.
   - Record due dates when mentioned.
   - Provide surrounding context from the discussion in the "context" field.

4. DECISIONS (Chain-of-Thought Analysis)
   - Identify formal agreements, choices, and conclusions reached.
   - For each candidate decision, reason through these questions internally:
     a) What topic was under discussion?
     b) What alternatives or options were considered?
     c) What was ultimately agreed upon, and who confirmed it?
     d) Why was this option chosen over alternatives?
   - Record the reasoning in the "reasoning" field.
   - Populate "decision_summary" with a one-sentence summary of the outcome.
   - Populate "alternatives_rejected" with options that were discussed but \
not chosen. Use an empty array if no alternatives were mentioned.
   - Include only confirmed decisions. Exclude open questions, hypothetical \
suggestions, discarded proposals, and deferred items.

5. BLOCKERS
   - Identify impediments, dependencies, and issues preventing progress.
   - Record the owner responsible for resolution and any discussed plan.

6. ACTION ITEMS
   - Extract follow-up items that do not fit into Tasks above.
   - Record the assignee and deadline when mentioned.
   - Provide a short title and surrounding context from the discussion.
   - Set priority based on urgency signals.

7. IDEAS
   - Extract suggestions, proposals, brainstorming items, and exploratory \
concepts raised during the meeting that are NOT confirmed decisions or \
assigned tasks.
   - Record who proposed the idea (proposed_by) when a name is mentioned.
   - Describe the potential impact or benefit the idea could deliver.
   - Ideas represent forward-looking thoughts — things the team might \
pursue but has not committed to.

8. EXECUTION TASKS (Explicit + Inferred)
   - Compile actionable work items derived from the meeting discussion.
   - Tag each as "Explicit" (directly stated as a task assignment) or \
"Inferred" (logically required to fulfil a stated decision, resolve a \
blocker, or implement a user story).
   - Assign an owner_role based on work nature: Engineering, Design, \
Product, DevOps, or QA — or use a specific name when mentioned.
   - Identify dependencies between tasks.
   - Inferred tasks must be logically necessary consequences of stated \
goals. Do not invent tasks for features that were not discussed.

9. CONFIDENCE SCORING
   - Assign a confidence_score between 0.0 and 1.0 to every artifact.
   - 0.9-1.0: Explicitly and unambiguously stated in the transcript.
   - 0.7-0.8: Clearly implied with strong supporting evidence.
   - 0.5-0.6: Partially supported with some ambiguity.
   - Below 0.5: Weak inference with minimal evidence."""

_EXTRACTION_FEW_SHOT = """\
=== FEW-SHOT EXAMPLE ===

<example_input>
Sarah: Let's start with the auth migration. We agreed to switch from JWT \
to OAuth2. Tom, can you handle the backend changes by next Friday?
Tom: Sure. The main blocker is the identity provider SDK — it hasn't been \
updated for Python 3.12 yet.
Sarah: Noted. Also, Lisa mentioned the onboarding flow needs work. \
Something like "as a new user, I want a guided setup wizard so I can get \
started without reading docs."
Tom: That sounds like a 5-pointer. Acceptance criteria should include \
completion tracking and the ability to skip steps.
Sarah: Good. Last thing — Redis or Memcached for caching?
Tom: Redis performance is fine. No reason to switch.
Sarah: Agreed, we stay with Redis. Meeting adjourned.
Tom: One more thought — what if we added a plugin system for auth providers? \
Could help us onboard enterprise clients faster.
Sarah: Interesting idea. Let's not commit to that now, but worth exploring.
</example_input>

<example_output>
{
  "meeting_title": "Auth Migration & Onboarding Planning",
  "summary": "The team confirmed the OAuth2 migration with Tom owning backend changes by next Friday, identified a Python 3.12 SDK compatibility blocker, and defined a user onboarding wizard story. Redis was retained as the caching layer.",
  "participants": ["Sarah", "Tom", "Lisa"],
  "user_stories": [
    {
      "title": "Guided Onboarding Setup Wizard",
      "as_a": "new user",
      "i_want": "a guided setup wizard",
      "so_that": "I can get started without reading documentation",
      "acceptance_criteria": [
        "Wizard tracks completion progress",
        "User can skip individual steps"
      ],
      "priority": "medium",
      "story_points": 5,
      "confidence_score": 0.95
    }
  ],
  "tasks": [
    {
      "title": "Implement OAuth2 backend migration",
      "description": "Replace JWT authentication with OAuth2 on the backend. Estimated effort: one week.",
      "assignee": "Tom",
      "due_date": "Next Friday",
      "context": "Sarah confirmed the JWT-to-OAuth2 migration and assigned Tom to handle backend changes.",
      "priority": "high",
      "confidence_score": 0.95
    }
  ],
  "decisions": [
    {
      "title": "Retain Redis as caching layer",
      "description": "The team decided to keep Redis instead of migrating to Memcached.",
      "decision_summary": "Stay with Redis for caching; no migration to Memcached.",
      "made_by": "Sarah",
      "rationale": "Current Redis performance is acceptable; migration cost is not justified.",
      "alternatives_rejected": ["Memcached"],
      "reasoning": "Topic: caching layer selection (Redis vs Memcached). Tom assessed Redis performance as adequate. No compelling reason to migrate was raised. Sarah confirmed the decision to stay with Redis.",
      "confidence_score": 0.95
    }
  ],
  "blockers": [
    {
      "title": "Identity provider SDK incompatible with Python 3.12",
      "description": "The identity provider SDK has not been updated for Python 3.12, which may block the OAuth2 migration.",
      "affected_tasks": ["Implement OAuth2 backend migration"],
      "owner": "Tom",
      "resolution_plan": "",
      "confidence_score": 0.9
    }
  ],
  "action_items": [],
  "ideas": [
    {
      "idea_description": "Add a plugin system for authentication providers to simplify enterprise client onboarding.",
      "proposed_by": "Tom",
      "potential_impact": "Could accelerate enterprise client onboarding by allowing pluggable auth provider integrations.",
      "confidence_score": 0.85
    }
  ],
  "execution_tasks": [
    {
      "title": "Implement OAuth2 backend migration",
      "description": "Replace JWT auth with OAuth2 including token management and identity provider integration.",
      "owner_role": "Engineering",
      "priority": "High",
      "task_source": "Explicit",
      "dependencies": ["Identity provider SDK Python 3.12 compatibility"],
      "confidence_score": 0.95
    },
    {
      "title": "Investigate Python 3.12-compatible identity provider SDK",
      "description": "Determine whether an updated SDK version exists or identify an alternative library compatible with Python 3.12.",
      "owner_role": "Engineering",
      "priority": "High",
      "task_source": "Inferred",
      "dependencies": [],
      "confidence_score": 0.7
    },
    {
      "title": "Design onboarding setup wizard",
      "description": "Create wireframes and UX flow for the guided onboarding wizard with completion tracking and skip functionality.",
      "owner_role": "Design",
      "priority": "Medium",
      "task_source": "Inferred",
      "dependencies": [],
      "confidence_score": 0.75
    }
  ]
}
</example_output>

=== END EXAMPLE ==="""

_EXTRACTION_OUTPUT_SCHEMA = """\
OUTPUT FORMAT:
Return a single valid JSON object matching the schema below. Do not include \
markdown fencing, commentary, or text outside the JSON object.

{
  "meeting_title": "string — inferred descriptive title",
  "summary": "string — 2-3 sentence overview",
  "participants": ["string"],
  "user_stories": [
    {
      "title": "string",
      "as_a": "string — user role",
      "i_want": "string — desired action",
      "so_that": "string — benefit",
      "acceptance_criteria": ["string"],
      "priority": "low | medium | high | critical",
      "story_points": "number or null (1, 2, 3, 5, 8, 13)",
      "confidence_score": "number 0.0-1.0 or null"
    }
  ],
  "tasks": [
    {
      "title": "string",
      "description": "string",
      "assignee": "string or null",
      "due_date": "string or null",
      "context": "string — surrounding discussion context",
      "priority": "low | medium | high | critical",
      "confidence_score": "number 0.0-1.0 or null"
    }
  ],
  "decisions": [
    {
      "title": "string — decision title",
      "description": "string — full details",
      "decision_summary": "string — one-sentence summary of the outcome",
      "made_by": "string or null",
      "rationale": "string — why this option was chosen",
      "alternatives_rejected": ["string — options considered but not chosen"],
      "reasoning": "string — step-by-step Chain-of-Thought analysis",
      "confidence_score": "number 0.0-1.0 or null"
    }
  ],
  "blockers": [
    {
      "title": "string",
      "description": "string",
      "affected_tasks": ["string"],
      "owner": "string or null",
      "resolution_plan": "string",
      "confidence_score": "number 0.0-1.0 or null"
    }
  ],
  "action_items": [
    {
      "title": "string — brief title",
      "description": "string — details",
      "assignee": "string or null",
      "due_date": "string or null",
      "context": "string — surrounding discussion context",
      "priority": "low | medium | high | critical",
      "confidence_score": "number 0.0-1.0 or null"
    }
  ],
  "ideas": [
    {
      "idea_description": "string — detailed description of the idea",
      "proposed_by": "string or null — person who proposed it",
      "potential_impact": "string — expected benefit or impact",
      "confidence_score": "number 0.0-1.0 or null"
    }
  ],
  "execution_tasks": [
    {
      "title": "string",
      "description": "string",
      "owner_role": "string — Engineering | Design | Product | DevOps | QA or specific name",
      "priority": "High | Medium | Low",
      "task_source": "Explicit | Inferred",
      "dependencies": ["string"],
      "confidence_score": "number 0.0-1.0 or null"
    }
  ]
}

Return an empty array [] for any category with no matching content."""


def build_extraction_prompt(transcript: str) -> str:
    """Assemble the master artifact extraction prompt.

    Combines role prompting, structured instructions with embedded
    Chain-of-Thought for decisions, a realistic few-shot example,
    and the output JSON schema into a single prompt optimised for
    deterministic structured output.
    """
    return "\n\n".join([
        _EXTRACTION_ROLE,
        "TRANSCRIPT:\n---",
        transcript,
        "---",
        _EXTRACTION_INSTRUCTIONS,
        _EXTRACTION_FEW_SHOT,
        _EXTRACTION_OUTPUT_SCHEMA,
    ])


# ======================================================================
# DOMAIN 4: FOCUSED TASK DETECTION
# ======================================================================

_TASK_DETECTION_ROLE = """\
You are a specialised Task Extraction Analyst focusing exclusively on \
identifying actionable work items from meeting transcripts. You have \
extensive experience decomposing discussions into concrete, assignable \
engineering tasks."""

_TASK_DETECTION_INSTRUCTIONS = """\
Analyze the transcript and extract every actionable task. Classify each as:

- **Explicit**: Directly stated as a work assignment or action item \
(e.g., "Tom will update the API by Friday").
- **Inferred**: Logically required to fulfil a stated decision, resolve a \
blocker, or implement a discussed feature — but not directly assigned.

For each task, determine:
1. A concise, action-oriented title starting with a verb.
2. A detailed description of the work required.
3. The assignee or owner role (Engineering, Design, Product, DevOps, QA).
4. Priority (High, Medium, Low) based on urgency, dependencies, and \
discussion emphasis.
5. Dependencies on other tasks or external conditions.
6. A confidence score reflecting how explicitly the task was discussed.

INFERENCE RULES:
- When a decision is made (e.g., "switch to PostgreSQL"), infer the \
implementation tasks required (e.g., "Provision PostgreSQL instance", \
"Migrate data from current datastore").
- When a blocker is raised (e.g., "missing UI designs"), infer the \
resolution task (e.g., "Complete UI mockups for feature X").
- Inferred tasks must be direct, necessary consequences. Do not invent \
tasks for undiscussed features."""

_TASK_DETECTION_FEW_SHOT = """\
=== FEW-SHOT EXAMPLE ===

<example_input>
Alice: We decided to migrate the notification service to Kafka.
Bob: I'll write the Kafka producer adapter this sprint.
Alice: We also need to update the consumer side, but nobody's been assigned yet.
Carol: The staging environment doesn't have a Kafka cluster. That's a blocker.
</example_input>

<example_output>
{
  "tasks": [
    {
      "title": "Implement Kafka producer adapter",
      "description": "Write the producer adapter for the notification service to publish events to Kafka topics.",
      "assignee": "Bob",
      "owner_role": "Engineering",
      "priority": "High",
      "task_source": "Explicit",
      "due_date": "This sprint",
      "dependencies": [],
      "confidence_score": 0.95
    },
    {
      "title": "Implement Kafka consumer adapter",
      "description": "Update the consumer side of the notification service to read from Kafka topics.",
      "assignee": null,
      "owner_role": "Engineering",
      "priority": "High",
      "task_source": "Explicit",
      "due_date": null,
      "dependencies": ["Implement Kafka producer adapter"],
      "confidence_score": 0.85
    },
    {
      "title": "Provision Kafka cluster in staging environment",
      "description": "Set up and configure a Kafka cluster in the staging environment to unblock notification service testing.",
      "assignee": null,
      "owner_role": "DevOps",
      "priority": "High",
      "task_source": "Inferred",
      "due_date": null,
      "dependencies": [],
      "confidence_score": 0.8
    }
  ]
}
</example_output>

=== END EXAMPLE ==="""

_TASK_DETECTION_SCHEMA = """\
OUTPUT FORMAT:
Return a single valid JSON object. Do not include markdown fencing or \
text outside the JSON.

{
  "tasks": [
    {
      "title": "string — action-oriented, verb-first",
      "description": "string — detailed work description",
      "assignee": "string or null",
      "owner_role": "string — Engineering | Design | Product | DevOps | QA",
      "priority": "High | Medium | Low",
      "task_source": "Explicit | Inferred",
      "due_date": "string or null",
      "dependencies": ["string"],
      "confidence_score": "number 0.0-1.0"
    }
  ]
}

Return an empty array if no tasks are identified."""


def build_task_detection_prompt(transcript: str) -> str:
    """Assemble the focused task detection prompt.

    Uses a dedicated Task Extraction Analyst persona with detailed
    inference rules and a few-shot example covering explicit and
    inferred task classification.
    """
    return "\n\n".join([
        _TASK_DETECTION_ROLE,
        "TRANSCRIPT:\n---",
        transcript,
        "---",
        _TASK_DETECTION_INSTRUCTIONS,
        _TASK_DETECTION_FEW_SHOT,
        _TASK_DETECTION_SCHEMA,
    ])


# ======================================================================
# DOMAIN 5: FOCUSED DECISION DETECTION (Chain of Thought)
# ======================================================================

_DECISION_DETECTION_ROLE = """\
You are a specialised Decision Analyst with expertise in identifying and \
classifying formal decisions made during meetings. You distinguish between \
confirmed decisions (binding agreements the team will act on) and \
non-decisions (open questions, brainstorming ideas, hypothetical \
suggestions, deferred items, and discarded proposals)."""

_DECISION_DETECTION_INSTRUCTIONS = """\
Analyze the transcript and identify every formal decision.

For each candidate statement, apply Chain-of-Thought reasoning:

Step 1 — IDENTIFY the discussion topic.
Step 2 — LIST the alternatives or options that were considered.
Step 3 — DETERMINE the outcome: was a specific option confirmed, or was \
the discussion left open?
Step 4 — CLASSIFY:
  - "confirmed" — A specific choice was agreed upon with clear commitment \
language ("we'll go with X", "decided", "agreed", "let's do X").
  - "discarded" — An option was explicitly rejected or tabled.
  - "open" — No resolution was reached; the topic remains under discussion.
Step 5 — RECORD only "confirmed" decisions in the output. Include the \
full reasoning chain in the "reasoning" field.

CLASSIFICATION SIGNALS:
- Confirmation language: "agreed", "decided", "we'll go with", "let's do", \
"approved", "confirmed", "settled on".
- Non-decision language: "maybe we should", "what if", "we could", \
"let's think about", "I'm not sure", "to be discussed", "TBD", "let's \
revisit", "parking lot"."""

_DECISION_DETECTION_FEW_SHOT = """\
=== FEW-SHOT EXAMPLE ===

<example_input>
PM: Should we use GraphQL or REST for the new API?
Dev1: GraphQL would be more flexible for the frontend team.
Dev2: REST is simpler and we already have the infrastructure.
PM: Good points. Let's go with REST for the v1 launch and revisit GraphQL for v2.
PM: What about the deployment — Kubernetes or plain ECS?
Dev1: I think Kubernetes, but we'd need to train the team.
PM: Let's put that in the parking lot for now.
</example_input>

<example_output>
{
  "decisions": [
    {
      "title": "Use REST for v1 API",
      "description": "The team will implement REST for the initial API launch, with GraphQL considered for a future v2.",
      "decision_summary": "REST selected for v1 launch; GraphQL deferred to v2.",
      "made_by": "PM",
      "rationale": "REST is simpler and leverages existing infrastructure. GraphQL is deferred to v2 to manage scope.",
      "alternatives_rejected": ["GraphQL"],
      "reasoning": "Step 1: Topic — API protocol for the new service (GraphQL vs REST). Step 2: Alternatives — GraphQL (more flexible for frontend) vs REST (simpler, existing infrastructure). Step 3: Outcome — PM confirmed REST for v1 with 'Let's go with REST'. Step 4: Classification — confirmed. Clear commitment language used.",
      "confidence_score": 0.95,
      "status": "confirmed"
    }
  ],
  "non_decisions": [
    {
      "topic": "Deployment platform (Kubernetes vs ECS)",
      "status": "open",
      "reasoning": "Step 1: Topic — deployment platform choice. Step 2: Alternatives — Kubernetes (requires team training) vs ECS. Step 3: Outcome — PM deferred with 'parking lot'. Step 4: Classification — open. No commitment was made."
    }
  ]
}
</example_output>

=== END EXAMPLE ==="""

_DECISION_DETECTION_SCHEMA = """\
OUTPUT FORMAT:
Return a single valid JSON object. Do not include markdown fencing or \
text outside the JSON.

{
  "decisions": [
    {
      "title": "string — confirmed decision title",
      "description": "string — full details of what was decided",
      "decision_summary": "string — one-sentence summary of the outcome",
      "made_by": "string or null — person who confirmed the decision",
      "rationale": "string — why this option was chosen",
      "alternatives_rejected": ["string — options considered but not chosen"],
      "reasoning": "string — step-by-step Chain-of-Thought analysis",
      "confidence_score": "number 0.0-1.0",
      "status": "confirmed"
    }
  ],
  "non_decisions": [
    {
      "topic": "string — topic that was discussed but not resolved",
      "status": "open | discarded",
      "reasoning": "string — Chain-of-Thought explaining why this is not a decision"
    }
  ]
}

Return empty arrays if no decisions or non-decisions are identified."""


def build_decision_detection_prompt(transcript: str) -> str:
    """Assemble the Chain-of-Thought decision detection prompt.

    Uses a dedicated Decision Analyst persona with explicit
    step-by-step reasoning instructions and a few-shot example
    demonstrating both confirmed and open classifications.
    """
    return "\n\n".join([
        _DECISION_DETECTION_ROLE,
        "TRANSCRIPT:\n---",
        transcript,
        "---",
        _DECISION_DETECTION_INSTRUCTIONS,
        _DECISION_DETECTION_FEW_SHOT,
        _DECISION_DETECTION_SCHEMA,
    ])


# ======================================================================
# DOMAIN 6: MEETING-LEVEL SUMMARIZATION
# ======================================================================

_MEETING_SUMMARIZATION_ROLE = """\
You are a senior technical meeting analyst. Write a comprehensive yet \
concise summary of the following meeting transcript. Your summary is used \
by stakeholders who were not present to quickly understand what was \
discussed and what actions were committed."""

_MEETING_SUMMARIZATION_INSTRUCTIONS = """\
Produce a structured summary covering:

1. **Purpose** — Why the meeting was held (1 sentence).
2. **Key Discussion Points** — The main topics discussed, in order.
3. **Decisions Made** — Confirmed agreements and their rationale.
4. **Action Items** — Who is doing what, by when.
5. **Open Items** — Unresolved topics or items deferred to future meetings.

Guidelines:
- Preserve all names, dates, numbers, and technical terms verbatim.
- Keep the summary under 500 words.
- Use bullet points for clarity.

Return the summary as plain text with markdown bullet points for structure."""


def build_summarization_prompt(transcript: str) -> str:
    """Assemble the meeting-level summarization prompt.

    Uses a senior analyst persona with structured output sections
    covering purpose, discussion points, decisions, action items,
    and open items.
    """
    return "\n\n".join([
        _MEETING_SUMMARIZATION_ROLE,
        "TRANSCRIPT:\n---",
        transcript,
        "---",
        _MEETING_SUMMARIZATION_INSTRUCTIONS,
    ])
