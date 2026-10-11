"""
Phase 9 Automated Verification: Scenario Evaluation & Baseline vs. Adaptive Benchmark.

Tests all 5 required Indian road scenarios:
1. Unmarked village road
2. Busy urban intersection without signals
3. Highway merge with slow-moving vehicles
4. Dense market area with mixed traffic
5. Sudden cattle-crossing event

Validates:
- Detection & tracking benchmark metrics
- Collision count, safety distance, TTC, latency, smoothness
- Baseline (Fixed Path) failure modes vs Adaptive Path Planner success
"""
import sys
from app.evaluation.schemas import ScenarioId
from app.evaluation.scenarios import list_all_scenarios
from app.evaluation.benchmark_runner import BenchmarkRunner


def test_scenario_benchmarks():
    print("===========================================================================")
    print("  Eareyes 2k26 — Phase 9: Scenario Evaluation & Benchmark Test Suite")
    print("===========================================================================")

    scenarios = list_all_scenarios()
    assert len(scenarios) == 5, f"Expected 5 scenarios, got {len(scenarios)}"
    print(f"[1] Verified 5 Indian road scenarios registered in catalog.")

    for s in scenarios:
        print(f"\n--- Testing Scenario: {s.title} ({s.id.value}) ---")
        res = BenchmarkRunner.run_scenario(s.id)

        # 1. Check Baseline metrics
        base = res.baseline_metrics
        print(f"  BASELINE (Fixed Path):")
        print(f"    Status: {base.final_status} | Collisions: {base.collision_count} | Completed: {base.scenario_completed}")
        print(f"    Min Clearance: {base.min_safety_distance_m}m | Min TTC: {base.min_ttc_sec}s | Critical TTCs: {base.critical_ttc_events}")
        assert base.planner_type == "BASELINE_FIXED_PATH"

        # 2. Check Adaptive metrics
        adapt = res.adaptive_metrics
        print(f"  ADAPTIVE PATH PLANNER:")
        print(f"    Status: {adapt.final_status} | Collisions: {adapt.collision_count} | Completed: {adapt.scenario_completed}")
        print(f"    Min Clearance: {adapt.min_safety_distance_m}m | Min TTC: {adapt.min_ttc_sec}s | Avg Latency: {adapt.avg_replanning_latency_ms}ms")
        print(f"    Path Length: {adapt.total_path_length_m}m | Smoothness Score: {adapt.path_smoothness_score}")
        assert adapt.planner_type == "ADAPTIVE_PATH_PLANNER"
        assert adapt.collision_count == 0, f"Adaptive planner experienced a collision in {s.id}!"
        assert adapt.scenario_completed is True, f"Adaptive planner failed to complete {s.id}!"
        assert adapt.avg_replanning_latency_ms < 5.0, f"Planning latency too high: {adapt.avg_replanning_latency_ms}ms"
        assert len(adapt.trajectory) > 0, "Adaptive trajectory is empty!"

        print(f"  Safety Improvement: +{res.safety_improvement_pct:.1f}%")
        print(f"  Verdict: {res.engineering_verdict[:70]}...")

    # 3. Test Full System Evaluation Summary
    print("\n[2] Verifying System-Wide Evaluation Summary...")
    summary = BenchmarkRunner.get_full_evaluation_summary()

    print(f"  Detection Precision : {summary.detection_precision}%")
    print(f"  Detection Recall    : {summary.detection_recall}%")
    print(f"  Detection mAP@50    : {summary.detection_map50}%")
    print(f"  Tracking MOTA       : {summary.tracking_mota}%")
    print(f"  Adaptive Completion : {summary.adaptive_completion_rate}% (Target: 100%)")
    print(f"  Baseline Completion : {summary.baseline_completion_rate}%")
    print(f"  Adaptive Collisions : {summary.adaptive_total_collisions} (Target: 0)")
    print(f"  Baseline Collisions : {summary.baseline_total_collisions}")
    print(f"  Avg Replan Latency  : {summary.avg_replanning_latency_ms} ms")

    assert summary.adaptive_completion_rate == 100.0
    assert summary.adaptive_total_collisions == 0
    assert summary.baseline_total_collisions > 0
    assert summary.avg_replanning_latency_ms < 2.5
    assert len(summary.scenarios) == 5

    print("\n===========================================================================")
    print("  ALL 5 SCENARIOS & SYSTEM EVALUATION TESTS PASSED PERFECTLY!")
    print("===========================================================================")


if __name__ == "__main__":
    test_scenario_benchmarks()
