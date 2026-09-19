#pragma once

namespace gevva {
// Persistent JSONL worker for shared-state, independent categorical questions.
int run_decision_worker();
int test_decision_kv_fork();
int test_decision_attention();
}
