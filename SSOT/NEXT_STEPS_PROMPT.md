# NEXT_STEPS_PROMPT — Capsule-Bound Micro-Execution (Tests-First, ≤3 Files, Deterministic)

## Context Snapshot (read-only)
- round_id: 20251019T204530Z
- green_rate: 0.0
- critical_green_rate: 0.0
- capsule_counts: { open_questions: 8, audit_backlog: 2, in_progress: 0 }
- coverage_reds: 15
- queue_len: 5
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
      "title": "Implement MTA ConversionPath schema with privacy-safe aggregation",
      "component": "C03_MTAAgent",
      "capsule_refs": ["Q_051", "Q_052", "A_006"],
      "targets": [
        "src/agents/mta/schemas.py",
        "tests/test_mta_contracts.py"
      ],
      "acceptance": [
        "unit: test_conversion_path_no_individual_tracking passes - validates user_count field exists, user_ids absent",
        "unit: test_markov_transition_probabilities_sum_to_one passes - validates sum(transitions[state].values()) == 1.0",
        "contract: ConversionPath Pydantic schema enforces min_user_count >= 10 for privacy",
        "contract: TransitionMatrix validator normalizes probabilities if sum != 1.0"
      ],
      "dry_run_probe": "Create mock ConversionPath with user_count=5, verify validation rejects; create TransitionMatrix with sum=0.9, verify normalization",
      "risk_gates": [
        "idempotency_key: path_id hash unique constraint",
        "kill_switch: MTA_PRIVACY_VALIDATION_ENABLED=true env flag"
      ],
      "owner": "Analytics",
      "urgency": "P0",
      "est_files": 2,
      "rollback": "If tests fail, revert schemas.py and keep test file to document expected behavior",
      "notes": "Paired Q_051+Q_052 with A_006; C03 critical path + Red status + P0 = score 10; highest priority"
    },
    {
      "id": "T002",
      "title": "Implement FeatureStore parity validation with latency SLA tracking",
      "component": "C10_FeatureStore",
      "capsule_refs": ["Q_053", "Q_054", "A_008"],
      "targets": [
        "src/feature_store/parity_validator.py",
        "src/feature_store/metrics.py",
        "tests/test_feature_parity.py"
      ],
      "acceptance": [
        "unit: test_parity_validator_detects_drift passes - offline vs online feature values compared",
        "unit: test_latency_histogram_tracks_p99 passes - verifies p99 latency metric emitted",
        "contract: ParityValidator runs on cron schedule (daily at 02:00 UTC)",
        "metric/log: feature_store_parity_drift_pct gauge emits value",
        "metric/log: feature_store_latency_ms histogram with p99 label exists"
      ],
      "dry_run_probe": "Mock online/offline feature store responses with intentional drift; verify alert triggers when drift > threshold",
      "risk_gates": [
        "idempotency_key: parity_check_date unique constraint in validation results table",
        "kill_switch: FEATURE_STORE_PARITY_ENABLED=true env flag"
      ],
      "owner": "DataOps",
      "urgency": "P1",
      "est_files": 3,
      "rollback": "If scheduler fails, disable cron job; if metrics break Prometheus, comment out histogram registration",
      "notes": "Paired Q_053+Q_054 with A_008; C10 high priority + Red + P1 = score 7; EXCEEDS 3-file budget but critical for observability - executor must choose subset or split"
    }
  ],
  "blocked_tasks": [],
  "deferred_tasks": [
    {
      "id": "TD01",
      "capsule_refs": ["Q_061"],
      "reason": "over budget - prioritize MTA and FeatureStore first",
      "next_round_hint": "Analytics Agent idempotency after T001/T002 complete"
    },
    {
      "id": "TD02",
      "capsule_refs": ["Q_062"],
      "reason": "over budget - HMAC validation is P0 but lower scoring than MTA",
      "next_round_hint": "E-commerce HMAC validation after analytics idempotency"
    },
    {
      "id": "TD03",
      "capsule_refs": ["Q_064"],
      "reason": "over budget - GDPR cleanup job",
      "next_round_hint": "GDPR TTL enforcement job after core contracts stable"
    },
    {
      "id": "TD04",
      "capsule_refs": ["Q_071"],
      "reason": "over budget - MMM validation refinement",
      "next_round_hint": "MMM temporal split validation after MTA complete"
    }
  ],
  "budget": {
    "max_product_files": 3,
    "reserved_for_tests": 1,
    "note": "T002 exceeds budget (3 files); executor must either: (a) implement only parity_validator.py + test, defer metrics.py, OR (b) split T002 into two rounds"
  }
}
```

## Scoring Rationale

**Top 3 by score (deterministic tie-breaks):**

1. **T001 (score=10):** Q_051+Q_052+A_006 → C03_MTAAgent
   - Red status (+3), P0 (+2), A.MAJOR (+3), Critical path (+2) = 10
   - Files: 2 (schemas + tests), fits budget
   
2. **T002 (score=7):** Q_053+Q_054+A_008 → C10_FeatureStore
   - Red status (+3), P1 (+1), A.MAJOR (+3), High priority path (+0) = 7
   - Files: 3 (parity + metrics + tests), AT budget limit
   - **Budget conflict:** Exceeds 3-file cap; executor must split or prioritize subset

3. **Deferred (score=6 each):** Q_061 (Analytics idempotency), Q_062 (Shopify HMAC), Q_064 (GDPR cleanup), Q_071 (MMM validation)
   - All P0 (+2) but not paired with audits, lower component priority
   - Queue contains P1/P2 items; these capsule items take precedence

## Executor Decision Tree

**Option A (Conservative):** Execute T001 only (2 files), defer T002 to next round
- Pros: Clean budget compliance, focused MTA delivery
- Cons: FeatureStore observability delayed

**Option B (Aggressive):** Execute T001 + subset of T002 (parity_validator.py + test only)
- Pros: Both critical paths advanced, metrics.py deferred
- Cons: Partial A_008 resolution; must track split work

**Recommendation:** Choose **Option A** for determinism; mark T002 as in_progress with partial completion note in capsule.

