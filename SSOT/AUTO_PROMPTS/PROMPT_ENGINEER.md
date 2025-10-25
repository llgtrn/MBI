TITLE: NEXT_STEPS_SYNTHESIZER — Capsule→NextSteps (Merge Q + A + System), Deterministic, Tests-First, ≤3 Files

ROLE
You synthesize a precise **NEXT_STEPS_PROMPT.md** from the current system state by merging:
1) `INTENT_CAPSULE.md` → open questions (Q_###) and audit backlog (A_###)
2) Coverage + queue artifacts (current Reds, priorities, blockers)
3) System spec snapshot (AGENT.md headings-only) and phase/metrics state
Your output is a compact, executable **prompt** for an Executor to perform the next micro-slice(s). Deterministic, safety-constrained, no new dependencies.

READ (EXACT PATHS; headings-only rule for large files)
- SSOT/SPEC/INTENT_CAPSULE.md                  (≤1.5KB; canonical, REQUIRED)
- SSOT/COVERAGE/coverage.matrix.csv             (status: Green/Yellow/Red; artifact flags)
- SSOT/COVERAGE/coverage.registry.json          (paths, components, ownership)
- SSOT/COVERAGE/queue.json                      (pending items; may mirror A/Q)
- SSOT/METRICS/progress.json                    (latest green_rate/critical_green_rate)
- SSOT/REPORTS/phase_state.json                 (phase/call_seq/round_id/budget)
- C:\Users\ishop\OneDrive\Documents\GitHub\MBI\AGENT.md   (headings-only)
- OPTIONAL: SSOT/REPORTS/run_log.md (tail 10 lines for context)

SELECTION POLICY (Capsule-First & System-Aware)
- **Candidates:** only capsule items with `status in {"open","in_progress"}` from:
  - `open_questions[]` (Q_###) and `audit_backlog[]` (A_###)
- **Pairing:** prefer **Q↔A pairs** sharing component/gap/targets to collapse “why + fix” into one step.
- **Critical path boost:** C01, C02, C03, C07, C10, C12 (+2 weight)
- **Coverage boost:** Red (+3), Yellow (+1)
- **Severity/Urgency:** A.MAJOR (+3), P0 (+2), P1 (+1)
- **Risk/Cost penalty:** coupling_score>2 (−2), file_cost>3 (discard)
- **Queue age:** older items +1 per round aged (cap +3)
- **Deterministic tie-breakers:** (urgency desc, severity desc, coverage status Red>Yellow>Green, critical path yes>no, queue_age desc, id asc)

SCORING FORMULA (document score S; choose top K)
S = 3*isRed + 2*isP0 + 3*isAMajor + 2*isCriticalPath + 1*isYellow + min(3,queue_age) − 2*(coupling>2)
Select **K = min(3, budget_total − budget_used)** “NOW” micro-steps; overflow becomes `deferred`.

OUTPUT TARGET (you must PRINT the full contents of this file)
Path to write: **SSOT/NEXT_STEPS_PROMPT.md**

FILE CONTENT SPEC (what you print)
A Markdown prompt with these exact sections and a single JSON plan block.

--- BEGIN FILE (print exactly this structure, with your synthesized content filled) ---

# NEXT_STEPS_PROMPT — Capsule-Bound Micro-Execution (Tests-First, ≤3 Files, Deterministic)

## Context Snapshot (read-only)
- round_id: <from phase_state.json>
- green_rate: <from progress.json>
- critical_green_rate: <from progress.json>
- capsule_counts: { open_questions: <n>, audit_backlog: <n_open>, in_progress: <n> }
- coverage_reds: <count_of_red_rows>
- queue_len: <len(queue.json)>
- constraints: { max_product_files_per_round: 3, contract_first: true, no_new_deps: true }

## Canonical Sources (do not modify)
- Capsule: SSOT/SPEC/INTENT_CAPSULE.md
- Coverage: SSOT/COVERAGE/coverage.matrix.csv, coverage.registry.json
- Queue:   SSOT/COVERAGE/queue.json
- Spec:    AGENT.md (headings-only snapshot)

## Execution Rules (must follow)
- **Capsule-only:** Every task MUST reference at least one `capsule_ref` (Q_### and/or A_###).
- **Tests-first → contracts → impl → docs**, ≤3 **product** files total in this run (SSOT files free).
- Idempotency & risk gates take precedence; prefer additive diffs; keep LF endings.
- If a target file is locked or migration needed → mark task `blocked` with precise `unblock` steps and STOP before exceeding budget.
- After successful acceptance, update capsule statuses to `"done"` for referenced items.

## Plan (JSON; deterministic order; edit this block only)
```json
{
  "now_tasks": [
    {
      "id": "T001",
      "title": "<short actionable micro-step>",
      "component": "Cxx_*",
      "capsule_refs": ["Q_###", "A_###"],                 // at least one; prefer Q↔A pair
      "targets": ["relative/path.ext", "another/path"],    // ≤3
      "acceptance": [
        "unit: <test_name_or_file> passes",
        "contract: <schema or zod> state == expected",
        "metric/log: <name> emits value/pattern"
      ],
      "dry_run_probe": "how to simulate without side effects",
      "risk_gates": ["idempotency_key", "kill_switch:<flag>"],
      "owner": "Risk|Execution|DataOps|Frontend|Infra",
      "urgency": "P0|P1|P2",
      "est_files": 1,
      "rollback": "revert condition & file(s) to undo",
      "notes": "short rationale linked to coverage/queue evidence"
    }
  ],
  "blocked_tasks": [
    {
      "id": "TB01",
      "capsule_refs": ["A_###"],
      "blocked_by": ["lock:<file>", "await:migration", "external:credential"],
      "unblock": ["micro-migration path/file", "owner_to_ping", "eta_hint"]
    }
  ],
  "deferred_tasks": [
    {
      "id": "TD01",
      "capsule_refs": ["Q_###"],
      "reason": "over budget",
      "next_round_hint": "pick first after capsule_clear"
    }
  ],
  "budget": { "max_product_files": 3, "reserved_for_tests": 1 }
}
