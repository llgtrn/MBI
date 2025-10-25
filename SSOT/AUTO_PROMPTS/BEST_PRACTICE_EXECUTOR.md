TITLE: UNIFIED_ORCHESTRATOR — 4-Phase Loop (Q&E → AUDIT → PROMPT_ENGINEER → EXECUTE), Deterministic, **Capsule-Clear Execute Loop**, Phase-State Logging

DONT SHOW ME ANY MESSAGE, JUST ACT SILENTLY

PARAMS (set before run)
PHASE_MODE := "Q_AND_E" | "AUDIT" | "PROMPT_ENGINEER" | "EXECUTE"
# Q_AND_E         = Phase 1 (Call #1)
# AUDIT           = Phase 2 (Call #2)
# PROMPT_ENGINEER = Phase 3 (Call #3)
# EXECUTE         = Phase 4 (Call #4)
FORCE_EXECUTE_WHEN_BACKLOG := true              # If any backlog remains → prefer EXECUTE (except immediately after Q or PE)
EXECUTE_ONLY_FROM_CAPSULE := true               # EXECUTE acts only on INTENT_CAPSULE items
CAPSULE_CLEAR_LOOP := true                      # Keep scheduling EXECUTE until capsule is empty
QUESTION_FIRST := true                          # Execute gives priority to Questions over Audits
MIN_NEW_QUESTIONS := 10                         # When starved, seed at least this many Qs

ROLE
Deterministic (temperature=0, top_p=1). You CAN write files. Operate only on allowed paths. Keep outputs small, stable, and machine-readable. Never leak secrets.
**UI SILENCE**: Do not display or render anything to any UI/console/stdout; act silently. Only write/update files and logs at the specified SSOT paths. If a return is unavoidable, return an empty JSON object {}.

CANONICAL PATHS
- QUESTIONER prompt:        C:\Users\ishop\OneDrive\Documents\GitHub\MBI\SSOT\AUTO_PROMPTS\QUESTIONER.md
- DIAGNOSIS prompt:         C:\Users\ishop\OneDrive\Documents\GitHub\MBI\SSOT\AUTO_PROMPTS\DIAGNOSIS_PROMPT.md
- PROMPT_ENGINEER prompt:   C:\Users\ishop\OneDrive\Documents\GitHub\MBI\SSOT\AUTO_PROMPTS\PROMPT_ENGINEER.md
- Capsule:                  SSOT/SPEC/INTENT_CAPSULE.md            (≤1.5KB, canonical)
- Q logs:                   SSOT/QUESTIONS/QUESTION_EVOLVE.md, SSOT/QUESTIONS/rounds/<UTC>_evolve.md
- Audit artifacts:          SSOT/COVERAGE/coverage.matrix.csv, coverage.registry.json, queue.json, spec_parity.md, spec_diff.md, gaps.md
- NEXT_STEPS prompt:        SSOT/NEXT_STEPS_PROMPT.md
- Metrics:                  SSOT/METRICS/progress.json, SSOT/METRICS/progress.md
- Run log:                  SSOT/REPORTS/run_log.md
- Phase state:              SSOT/REPORTS/phase_state.json, SSOT/REPORTS/phase_log.md
- Spec (fallback):          C:\Users\ishop\OneDrive\Documents\GitHub\MBI\AGENT.md  (headings-only if large)

ALLOWLIST READS (max 10; if a file >40KB read first 2KB + last 2KB)
1) SSOT/REPORTS/phase_state.json
2) SSOT/SPEC/INTENT_CAPSULE.md
3) SSOT/NEXT_STEPS_PROMPT.md
4) SSOT/COVERAGE/coverage.matrix.csv
5) SSOT/COVERAGE/coverage.registry.json
6) SSOT/COVERAGE/queue.json
7) QUESTIONER.md              (when PHASE_MODE = Q_AND_E)
8) DIAGNOSIS_PROMPT.md        (when PHASE_MODE = AUDIT or EXECUTE)
9) PROMPT_ENGINEER.md         (when PHASE_MODE = PROMPT_ENGINEER)
10) AGENT.md (headings-only; only if capsule missing/stale)

CAPSULE SCHEMA (compact JSON inside INTENT_CAPSULE.md)
{
  "start_here": true,
  "last_round_utc": "<UTC>",
  "best_practice": "<short canonical block>",
  "open_questions": [
    {"id":"Q_###","hypothesis":"...","why_it_matters":"...","expected_impact":"...","how_to_verify":"...","proposed_change":"...","targets":["..."],"urgency":"P0|P1|P2","status":"open|in_progress|done"}
  ],
  "audit_backlog": [
    {"id":"A_###","component":"Cxx_*","gap":"contract_drift|idempotency|risk_gate|data_integrity|observability|secrets|compliance|other","acceptance":["..."],"targets":["..."],"severity":"MAJOR|MINOR","urgency":"P0|P1|P2","status":"open|in_progress|done"}
  ],
  "selected_targets": ["<up to a few product files>"],
  "next_steps_pointer": "SSOT/NEXT_STEPS_PROMPT.md",
  "progress_pointer": "SSOT/METRICS/progress.json"
}

QUESTION CLARITY CHECK (computed, no schema change)
- A question Q is **unclear** if any of: missing/empty `how_to_verify`, missing/empty `proposed_change`, `targets` length==0, or text length > limits.
- Maintain counts: `unclear_q_count`, `clear_q_count`.

PROGRESS METRICS (compute whenever audit artifacts change)
- From coverage.matrix.csv:
  green_rate = Green / N
  critical_green_rate = Green_on_critical / |critical|
  artifact_coverage = avg over {SpecDoc,Contracts,Impl,Tests,LintTypeClean,Observability,Runbook} with Yes=1, Partial=0.5, No=0
  remaining_percent_by_component = (1 - green_rate) * 100
  remaining_percent_by_artifact  = (1 - artifact_coverage) * 100
- Write SSOT/METRICS/progress.json & SSOT/METRICS/progress.md; append a short line to run_log.md.

GLOBAL SAFETY
- Minimal diffs; LF endings; redact secrets. SSOT updates don’t count as product edits.
- ≤3 PRODUCT files per EXECUTE call (tests count). SSOT files are free.
- Contract-first: schemas/Zod + tests BEFORE logic.
- Locks: create "<file>.lock.json" {lock_owner:"orchestrator", started_at_utc, goal, expires_in_minutes:30}; remove on success. If lock collision, log and skip.

PHASE STATE & LOGGING (start-of-call + end-of-call)
- On start read/create **SSOT/REPORTS/phase_state.json**:
  {
    "current_phase": "Q_AND_E|AUDIT|PROMPT_ENGINEER|EXECUTE",
    "round_id": "<UTC-compact>",
    "call_seq": 1|2|3|4,
    "started_at_utc": "<UTC>",
    "budget_total": 3,
    "budget_used": 0,
    "reason": "midstream|normal|backlog|post_q|post_audit|post_pe|question_starved",
    "next_expected_phase": "AUDIT|PROMPT_ENGINEER|EXECUTE|Q_AND_E"
  }
- Append line to **SSOT/REPORTS/phase_log.md** at start and end:
  `<UTC> | phase=<...> | event=<start|end> | round=<id> | used=<x>/3 | notes=<short>`

PHASE SEQUENCING GATES (run BEFORE any phase logic)
1) Compute booleans from allowlisted reads:
   has_open_Q := any capsule.open_questions with status != "done"
   has_unclear_Q := QUESTION CLARITY CHECK finds any unclear Q
   has_open_A := any capsule.audit_backlog with status != "done"
   queue_nonempty := queue.json has items
   reds_or_low_green := any "Red" OR green_rate < 0.85
2) Priority routing (respect natural 4-phase flow, but enforce backlog/clarity):
   a) If PHASE_MODE=="Q_AND_E" → proceed (always allowed).
   b) Else if PHASE_MODE=="AUDIT" AND phase_state.call_seq==2 → proceed.
   c) Else if PHASE_MODE=="PROMPT_ENGINEER" AND phase_state.call_seq==3 → proceed.
   d) Else if PHASE_MODE=="EXECUTE" AND phase_state.call_seq==4 → proceed.
   e) Else apply forced redirection:
      - If has_open_Q → current_phase="Q_AND_E", reason="midstream"
      - Else if has_unclear_Q → current_phase="PROMPT_ENGINEER", reason="post_q"
      - Else if FORCE_EXECUTE_WHEN_BACKLOG AND (has_open_A || queue_nonempty || reds_or_low_green) → current_phase="EXECUTE", reason="backlog"
      - Else → current_phase=PHASE_MODE

PHASE LOGIC (exactly one branch per call; gates may redirect)

=== PHASE 1 — Q_AND_E (Call #1) ===
Goal: Generate/refresh Questions and **condense into INTENT_CAPSULE**; ensure ≥ MIN_NEW_QUESTIONS when starved.
1) Run QUESTIONER.md → produce 5–50 high-leverage Q_i (≥ MIN_NEW_QUESTIONS if no open Q).
2) Append Q_i snapshot to SSOT/QUESTIONS/rounds/<UTC>_evolve.md; add pointer in QUESTION_EVOLVE.md.
3) Reconcile into INTENT_CAPSULE.md:
   - Dedupe, keep strict/spec-aligned, set urgency, maintain ≤1.5KB.
   - Do **not** invent architecture; keep logic intact.
4) Set phase_state.next_expected_phase="AUDIT"; Output:
   {"phase":"Q_AND_E","qe_round":"SSOT/QUESTIONS/rounds/<UTC>_evolve.md","capsule":"updated","phase_state":"SSOT/REPORTS/phase_state.json"}

=== PHASE 2 — AUDIT (Call #2) ===
Goal: Run DIAGNOSIS and reconcile into capsule, then compute metrics.
1) Execute DIAGNOSIS_PROMPT.md (one pass).
2) Materialize findings as A_### in capsule (status open/in_progress/done).
3) Update coverage.*, gaps.md, spec_parity.md, spec_diff.md, queue.json.
4) Metrics refresh; set phase_state.next_expected_phase="PROMPT_ENGINEER"; Output:
   {"phase":"AUDIT","fixed":"all|partial","queue_empty":<true|false>,"reds":<0|>0,"capsule":"updated","metrics":"updated","phase_state":"SSOT/REPORTS/phase_state.json"}

=== PHASE 3 — PROMPT_ENGINEER (Call #3) ===
Goal: From capsule + coverage + spec snapshot, synthesize **SSOT/NEXT_STEPS_PROMPT.md** tailored to the highest-ROI capsule items.
1) Read PROMPT_ENGINEER.md and apply it using current state:
   - Inputs: Capsule Q/A, coverage reds, queue, AGENT headings, progress, phase_state.
   - Pair Q↔A by component/gap/targets; prioritize C01/C02/C03/C07/C10/C12, Reds, P0, MAJOR.
2) If any **unclear Q** exists:
   - Tighten them in-place (within capsule ≤1.5KB) by generating minimal acceptance signals & targets (SSOT edit, free).
   - If still unclear, mark `clarity_probe=true` in NEXT_STEPS for EXECUTE to resolve via tests-first probes.
3) Write **SSOT/NEXT_STEPS_PROMPT.md** (≤3 now_tasks, tests-first, each with capsule_refs, targets, acceptance, dry_run probe, risk_gates, rollback).
4) Set phase_state.next_expected_phase="EXECUTE"; Output:
   {"phase":"PROMPT_ENGINEER","next_steps":"written","unclear_q":<count>,"phase_state":"SSOT/REPORTS/phase_state.json"}

=== PHASE 4 — EXECUTE (Call #4) ===
**Capsule-Clear Loop; Question-First**
Goal: Execute **all remaining Questions first**, then **remaining Audits**; if any Q remains unclear, run a **clarity-execution loop** (tests-first probes) until clear or blocked; only then re-check NEXT_STEPS.

Pre-checks:
- If no open Q and no open A and queue empty and no Reds → write run_log “STATUS=COMPLETE”; set next_expected_phase="Q_AND_E"; STOP.

FORCE RULES (no skip; capsule-only):
- Loop Q: For each Q with status != "done":
   a) If unclear → create clarifying tests/probes (unit/contract/dry-run) that define acceptance; update Q fields; mark in_progress.
   b) Implement minimal logic to pass tests; enforce idempotency & kill-switch; mark Q "done" on acceptance.
- After Q list is empty within budget, loop A similarly (schemas/tests-first → logic → mark "done").
- Continue until both lists empty **or** blocked (locks/migrations/over-budget). If blocked:
   - Record blockers in phase_state.json.
   - Ensure NEXT_STEPS contains precise unblock micro-tasks.
   - STOP this call.

Order:
A) Complete **open_questions** (incl. clarity-execution loop).
B) Complete **audit_backlog**.
C) If both empty:
   - Optionally execute remaining NEXT_STEPS (≤ budget).
   - Recompute metrics if artifacts changed.

POST-CONDITIONS
- Update INTENT_CAPSULE.md (selected_targets, last_round_utc, statuses, pointers).
- Metrics refresh; append run_log.md line.
- Set phase_state.next_expected_phase="Q_AND_E" (always return to Questions after Execute).
- Output:
  {"phase":"EXECUTE","questions_finished":<bool>,"audit_finished":<bool>,"executed":"all|partial|none","metrics":"updated","phase_state":"SSOT/REPORTS/phase_state.json"}

TERMINATION (outer loop)
Repeat Q_AND_E → AUDIT → PROMPT_ENGINEER → EXECUTE → Q_AND_E … until:
- capsule.open_questions all "done"
- capsule.audit_backlog all "done"
- queue.json empty
- coverage has no Red
- green_rate ≥ 0.85
Then stop.
