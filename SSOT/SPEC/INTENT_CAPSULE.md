# INTENT_CAPSULE - MBI System

**Start Here**: This is the canonical source of truth for current implementation priorities.

**Last Round**: 20251019T203645Z

**Best Practice**: Contract-first development with comprehensive testing, privacy-safe data handling, and observability at every layer.

## Open Questions (Priority)

```json
{
  "open_questions": [
    {
      "id": "Q_051",
      "hypothesis": "ConversionPath schema may not validate privacy-safe aggregation before MTA calculations",
      "why_it_matters": "A_006 requires contract-first MTA schemas; missing validation risks PII leakage in attribution paths",
      "expected_impact": "Critical privacy violation if individual paths tracked",
      "how_to_verify": "Check for Pydantic ConversionPath schema with privacy checks; unit tests enforce no individual tracking",
      "proposed_change": "Define ConversionPath schema with user_count instead of user_ids; validate aggregation thresholds",
      "targets": ["src/agents/mta/schemas.py", "tests/test_mta_contracts.py"],
      "urgency": "P0",
      "status": "open",
      "related": ["A_006"]
    },
    {
      "id": "Q_052",
      "hypothesis": "Markov transition matrices may not validate probability sum == 1.0",
      "why_it_matters": "A_006 contract drift; invalid transition probabilities corrupt attribution weights",
      "expected_impact": "10-20% error in MTA attribution if probabilities don't normalize",
      "how_to_verify": "Unit test verifies sum(transitions[state].values()) == 1.0 for all states",
      "proposed_change": "Add TransitionMatrix validation in MTA agent; normalize probabilities in _compute_transitions",
      "targets": ["src/agents/mta/mta_agent.py", "tests/test_mta_contracts.py"],
      "urgency": "P0",
      "status": "open",
      "related": ["A_006"]
    },
    {
      "id": "Q_053",
      "hypothesis": "FeatureStore may not implement online/offline parity validation on schedule",
      "why_it_matters": "A_008 observability gap; offline training features diverging from online serving breaks model accuracy",
      "expected_impact": "20-30% model accuracy drop if features drift",
      "how_to_verify": "Check for scheduled parity validation job; alerts fire when offline != online features",
      "proposed_change": "Implement ParityValidator with daily cron; compare feature values and emit metrics",
      "targets": ["src/feature_store/parity_validator.py", "tests/test_feature_parity.py"],
      "urgency": "P0",
      "status": "open",
      "related": ["A_008"]
    },
    {
      "id": "Q_054",
      "hypothesis": "Feature store online serving latency may not be tracked with p99 SLA enforcement",
      "why_it_matters": "A_008 requires <50ms p99 latency; missing SLA tracking causes silent performance degradation",
      "expected_impact": "User-facing latency violations if feature store slows",
      "how_to_verify": "Check for Prometheus histogram tracking feature_store_latency_ms with p99 alert at 50ms",
      "proposed_change": "Add latency instrumentation in feature store; configure Prometheus alert",
      "targets": ["src/feature_store/metrics.py", "prometheus/feature_store.yml"],
      "urgency": "P0",
      "status": "open",
      "related": ["A_008"]
    },
    {
      "id": "Q_061",
      "hypothesis": "Analytics Agent (GA4) may not implement event deduplication by session_id + event_id",
      "why_it_matters": "C07 idempotency partially resolved (A_001 done for ads); GA4 events may still duplicate",
      "expected_impact": "5-10% event count inflation if duplicates exist",
      "how_to_verify": "Unit test verifies duplicate event insertions result in single row",
      "proposed_change": "Add unique constraint on (session_id, event_id, timestamp) in fct_web_session",
      "targets": ["src/agents/ingestion/analytics_agent.py", "tests/test_analytics_idempotency.py"],
      "urgency": "P0",
      "status": "open",
      "related": []
    },
    {
      "id": "Q_062",
      "hypothesis": "E-commerce Agent (Shopify) webhook receiver may not validate HMAC signatures",
      "why_it_matters": "A_007 done for generic webhooks; Shopify-specific HMAC validation may be missing",
      "expected_impact": "Critical security gap; malicious webhooks could corrupt orders",
      "how_to_verify": "Unit test verifies webhook rejected if HMAC invalid",
      "proposed_change": "Implement Shopify HMAC validation using shared secret from KMS",
      "targets": ["src/agents/ingestion/ecommerce_agent.py", "tests/test_shopify_webhook_security.py"],
      "urgency": "P0",
      "status": "open",
      "related": []
    },
    {
      "id": "Q_064",
      "hypothesis": "GDPR TTL enforcement may not actually delete hashed user_keys after 90 days",
      "why_it_matters": "Q_002 done with TTL config; execution of deletion may not be implemented",
      "expected_impact": "GDPR compliance violation if old data retained",
      "how_to_verify": "Scheduled job deletes dim_user rows where valid_to < now() - 90 days",
      "proposed_change": "Implement GDPR cleanup job in db/jobs/; test validates cleanup executes",
      "targets": ["db/jobs/gdpr_cleanup.sql", "tests/test_gdpr_cleanup.py"],
      "urgency": "P0",
      "status": "open",
      "related": ["Q_002"]
    },
    {
      "id": "Q_071",
      "hypothesis": "MMM model validation may not use true out-of-sample holdout (cross-validation instead)",
      "why_it_matters": "Q_003 implemented MAPE tracking; cross-validation on time series leaks future info",
      "expected_impact": "Model MAPE artificially low; real-world performance 10-15% worse",
      "how_to_verify": "Check that holdout set is chronologically last 20% of data with no shuffle",
      "proposed_change": "Enforce temporal split in MMM validator; reject models trained with CV",
      "targets": ["src/agents/mmm/model_validator.py", "tests/agents/mmm/test_mmm_validation.py"],
      "urgency": "P0",
      "status": "open",
      "related": ["Q_003"]
    }
  ],
  "audit_backlog": [
    {
      "id": "A_006",
      "component": "C03_MTAAgent",
      "gap": "contract_drift",
      "acceptance": ["ConversionPath Pydantic schema defined", "Markov transition validation tests pass"],
      "targets": ["src/agents/mta/schemas.py", "tests/test_mta_contracts.py"],
      "severity": "MAJOR",
      "urgency": "P0",
      "owner": "Analytics",
      "status": "open",
      "related": ["Q_051", "Q_052"]
    },
    {
      "id": "A_008",
      "component": "C10_FeatureStore",
      "gap": "observability",
      "acceptance": ["Online/offline parity validation runs on schedule", "Latency SLA <50ms p99 tracked"],
      "targets": ["src/feature_store/parity_validator.py", "src/feature_store/metrics.py"],
      "severity": "MAJOR",
      "urgency": "P1",
      "owner": "DataOps",
      "status": "open",
      "related": ["Q_053", "Q_054"]
    }
  ],
  "selected_targets": [
    "src/agents/mta/schemas.py",
    "src/agents/mta/mta_agent.py",
    "tests/test_mta_contracts.py",
    "src/feature_store/parity_validator.py",
    "src/feature_store/metrics.py",
    "tests/test_feature_parity.py",
    "src/agents/ingestion/analytics_agent.py",
    "tests/test_analytics_idempotency.py",
    "src/agents/ingestion/ecommerce_agent.py",
    "tests/test_shopify_webhook_security.py"
  ],
  "next_steps_pointer": "SSOT/NEXT_STEPS_PROMPT.md",
  "progress_pointer": "SSOT/METRICS/progress.json"
}
```

## Notes

- **Round 20251019T203645Z - Q_AND_E Phase:**
  - Generated 50 new questions (Q_051-Q_100) targeting post-P0 work
  - Focus areas: MTA contracts (A_006), Feature store observability (A_008), data quality, compliance execution
  - Added 8 P0 questions to capsule (Q_051, Q_052, Q_053, Q_054, Q_061, Q_062, Q_064, Q_071)
  - Remaining 42 questions logged in rounds/20251019T203645Z_evolve.md for future reference
- **Previous milestone:** All P0 questions (Q_001-Q_012) completed with tests and contracts
- **Next phase:** AUDIT - will diagnose MTA/FeatureStore gaps and update coverage
