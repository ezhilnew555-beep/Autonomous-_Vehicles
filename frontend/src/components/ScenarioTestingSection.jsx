/**
 * ScenarioTestingSection — Phase 9B: Indian Road Scenario Testing & Performance Evaluation.
 *
 * Implements:
 * 1. 5 Required Indian Road Scenarios (Village Road, Urban Unsignalled Intersection, Highway Merge, Market Area, Cattle Crossing).
 * 2. 10-Stage Pipeline Evaluation (Video -> Detection -> Tracking -> Prediction -> Risk -> Decision -> Env -> Planning -> Replanning -> Metrics).
 * 3. Side-by-Side 5-Scenario Comparison Table (with explicit NOT TESTED for untested scenarios).
 * 4. Summary KPI Cards (Completion Rate, Collision-Free Rate, Avg Latency, Avg Smoothness, Total Replans, Total Collisions).
 * 5. Event History Timeline for Selected Scenario.
 * 6. One-Click JSON & CSV Report Export buttons.
 */
import { useState, useEffect } from 'react'
import {
  getScenarioCatalog,
  getScenarioDetails,
  runScenarioTest,
  runAllScenariosTest,
  getScenariosComparison,
  getScenariosSummary,
  getScenariosExportUrl,
} from '../api/client.js'

export default function ScenarioTestingSection() {
  const [catalog, setCatalog] = useState([])
  const [comparisonTable, setComparisonTable] = useState([])
  const [summary, setSummary] = useState(null)
  const [selectedScenarioId, setSelectedScenarioId] = useState('VILLAGE_ROAD')
  const [selectedDetails, setSelectedDetails] = useState(null)
  const [runningId, setRunningId] = useState(null)
  const [isRunningAll, setIsRunningAll] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load catalog, comparison, and summary on mount
  useEffect(() => {
    fetchScenarioData()
  }, [])

  // Load details when selectedScenarioId changes
  useEffect(() => {
    if (!selectedScenarioId) return
    loadScenarioDetails(selectedScenarioId)
  }, [selectedScenarioId])

  const fetchScenarioData = async () => {
    try {
      setLoading(true)
      setError(null)
      const [catRes, compRes, sumRes] = await Promise.all([
        getScenarioCatalog().catch(() => ({ data: [] })),
        getScenariosComparison().catch(() => ({ data: [] })),
        getScenariosSummary().catch(() => ({ data: null })),
      ])

      setCatalog(catRes.data || [])
      setComparisonTable(compRes.data || [])
      setSummary(sumRes.data || null)

      if (catRes.data && catRes.data.length > 0 && !selectedScenarioId) {
        setSelectedScenarioId(catRes.data[0].scenario_id)
      }
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Failed to load scenario testing data.')
    } finally {
      setLoading(false)
    }
  }

  const loadScenarioDetails = async (id) => {
    try {
      const res = await getScenarioDetails(id)
      setSelectedDetails(res.data)
    } catch (err) {
      console.warn('Could not load scenario details for', id, err)
    }
  }

  const handleRunSingle = async (scenarioId) => {
    try {
      setRunningId(scenarioId)
      setError(null)
      await runScenarioTest(scenarioId)
      await fetchScenarioData()
      await loadScenarioDetails(scenarioId)
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || `Failed to execute scenario ${scenarioId}.`)
    } finally {
      setRunningId(null)
    }
  }

  const handleRunAll = async () => {
    try {
      setIsRunningAll(true)
      setError(null)
      await runAllScenariosTest()
      await fetchScenarioData()
      if (selectedScenarioId) {
        await loadScenarioDetails(selectedScenarioId)
      }
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Failed to execute all scenarios.')
    } finally {
      setIsRunningAll(false)
    }
  }

  const getRiskBadgeColor = (risk) => {
    switch (risk) {
      case 'CRITICAL': return '#ef4444'
      case 'HIGH': return '#f97316'
      case 'MEDIUM': return '#eab308'
      case 'LOW': return '#3b82f6'
      default: return '#10b981'
    }
  }

  const getActionBadgeColor = (action) => {
    switch (action) {
      case 'EMERGENCY_STOP':
      case 'STOP': return '#ef4444'
      case 'AVOID':
      case 'REPLAN': return '#8b5cf6'
      case 'SLOW_DOWN': return '#f59e0b'
      default: return '#10b981'
    }
  }

  return (
    <div className="scenario-testing-section">
      {/* ── SECTION HEADER ────────────────────────────────────────────────── */}
      <div className="section-header-banner">
        <div>
          <div className="section-super-title">SCENARIO BENCHMARK &amp; SYSTEM EVALUATION</div>
          <h2 className="section-main-title">🇮🇳 Indian Road Scenario Testing &amp; Performance Evaluation</h2>
          <p className="section-desc">
            Rigorous evaluation of the 10-stage autonomous pipeline across 5 high-entropy Indian road situations:
            unmarked corridors, chaotic unsignalled junctions, highway merges, dense market crowds, and cattle crossings.
          </p>
        </div>

        <div className="section-header-actions">
          <button
            className="btn btn-primary"
            onClick={handleRunAll}
            disabled={isRunningAll || runningId !== null}
          >
            {isRunningAll ? '⏳ Evaluating All Scenarios...' : '⚡ Run All 5 Scenarios (Batch)'}
          </button>
          <a
            href={getScenariosExportUrl('json')}
            download="scenario_evaluation_report.json"
            className="btn btn-secondary"
          >
            📥 Export JSON
          </a>
          <a
            href={getScenariosExportUrl('csv')}
            download="scenario_evaluation_report.csv"
            className="btn btn-secondary"
          >
            📥 Export CSV
          </a>
        </div>
      </div>

      {error && (
        <div className="alert-banner error-banner" style={{ marginBottom: 16 }}>
          <span>⚠️ {error}</span>
          <button className="btn-text-close" onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* ── SUMMARY KPI CARDS ─────────────────────────────────────────────── */}
      <div className="scenario-kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">SCENARIO COMPLETION RATE</div>
          <div className="kpi-value text-accent">
            {summary ? summary.scenario_completion_rate_text : '0 / 5'}
          </div>
          <div className="kpi-subtext">
            {summary ? `${summary.scenario_completion_rate_pct}% of scenarios executed without fatal crash` : 'Awaiting evaluation'}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">COLLISION-FREE RATE</div>
          <div className="kpi-value text-emerald">
            {summary && summary.collision_free_scenarios > 0 ? `${summary.collision_free_rate_pct}%` : summary?.completed_scenarios > 0 ? '0%' : 'N/A'}
          </div>
          <div className="kpi-subtext">
            {summary ? `${summary.collision_free_scenarios} of ${summary.completed_scenarios} scenarios safely navigated` : 'Zero collisions recorded'}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">AVG REPLANNING LATENCY</div>
          <div className="kpi-value font-mono">
            {summary && summary.average_replanning_latency_ms > 0 ? `${summary.average_replanning_latency_ms} ms` : 'N/A'}
          </div>
          <div className="kpi-subtext">Real-time candidate spline optimization latency</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">AVG PATH SMOOTHNESS</div>
          <div className="kpi-value font-mono text-cyan">
            {summary && summary.average_path_smoothness > 0 ? `${summary.average_path_smoothness} / 1.0` : 'N/A'}
          </div>
          <div className="kpi-subtext">Curvature &amp; angular direction penalty score</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">TOTAL REPLANS</div>
          <div className="kpi-value font-mono text-purple">
            {summary ? summary.total_replans_executed : 0}
          </div>
          <div className="kpi-subtext">Dynamic trajectory re-selections</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">TOTAL COLLISIONS</div>
          <div className="kpi-value font-mono" style={{ color: summary?.total_collision_events === 0 ? '#10b981' : '#ef4444' }}>
            {summary ? summary.total_collision_events : 0}
          </div>
          <div className="kpi-subtext">Physical collisions across all evaluated frames</div>
        </div>
      </div>

      {/* ── 5 SCENARIO SELECTOR CARDS ─────────────────────────────────────── */}
      <div className="scenario-selector-grid">
        {catalog.map((sc) => {
          const isSelected = selectedScenarioId === sc.scenario_id
          const isTesting = runningId === sc.scenario_id
          const isTested = sc.status === 'TESTED'
          return (
            <div
              key={sc.scenario_id}
              className={`scenario-card ${isSelected ? 'selected' : ''} ${isTested ? 'tested' : 'untested'}`}
              onClick={() => setSelectedScenarioId(sc.scenario_id)}
            >
              <div className="scenario-card-top">
                <span className={`status-pill ${isTested ? 'pill-tested' : 'pill-not-tested'}`}>
                  {isTested ? '● TESTED' : '○ NOT TESTED'}
                </span>
                <span className="speed-tag">{sc.nominal_speed_kmh} km/h</span>
              </div>

              <div className="scenario-card-name">{sc.scenario_name}</div>
              <div className="scenario-card-desc">{sc.description}</div>

              <div className="scenario-card-tags">
                {sc.expected_objects?.slice(0, 3).map((obj, i) => (
                  <span key={i} className="micro-tag">{obj}</span>
                ))}
              </div>

              <div className="scenario-card-actions">
                <button
                  className="btn btn-sm btn-outline"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleRunSingle(sc.scenario_id)
                  }}
                  disabled={isTesting || isRunningAll}
                >
                  {isTesting ? '⏳ Testing...' : isTested ? '🔄 Re-test' : '⚡ Run Test'}
                </button>
              </div>
            </div>
          )
        })}
      </div>

      {/* ── 5-SCENARIO COMPARISON TABLE ───────────────────────────────────── */}
      <div className="card full-width comparison-card" style={{ marginTop: 24 }}>
        <div className="card-header-flex">
          <div>
            <div className="card-title">📊 5-Scenario Comparison Table</div>
            <div className="card-subtitle">
              Comprehensive evaluation results across all required Indian road situations. Untested scenarios explicitly display NOT TESTED.
            </div>
          </div>
          <div className="table-actions">
            <button className="btn btn-sm btn-secondary" onClick={fetchScenarioData} disabled={loading}>
              🔄 Refresh
            </button>
          </div>
        </div>

        <div className="table-responsive-wrapper">
          <table className="scenario-comparison-table">
            <thead>
              <tr>
                <th>Scenario Name</th>
                <th>Status</th>
                <th>Completed</th>
                <th>Collision-Free</th>
                <th>Replans</th>
                <th>Avg Latency</th>
                <th>Path Smoothness</th>
                <th>Processing FPS</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {comparisonTable.map((row) => {
                const isTested = row.status === 'TESTED'
                return (
                  <tr key={row.scenario_id} className={selectedScenarioId === row.scenario_id ? 'row-selected' : ''}>
                    <td className="scenario-name-cell">
                      <div className="font-semibold">{row.scenario}</div>
                      <div className="font-mono text-muted text-xs">{row.scenario_id}</div>
                    </td>

                    <td>
                      <span className={`badge-micro ${isTested ? 'badge-green' : 'badge-gray'}`}>
                        {row.status}
                      </span>
                    </td>

                    <td>
                      {isTested ? (
                        <span className={`badge-micro ${row.completed ? 'badge-green' : 'badge-red'}`}>
                          {row.completed ? 'YES' : 'NO'}
                        </span>
                      ) : (
                        <span className="text-muted font-mono">NOT TESTED</span>
                      )}
                    </td>

                    <td>
                      {isTested ? (
                        <span className={`badge-micro ${row.collision_free ? 'badge-green' : 'badge-red'}`}>
                          {row.collision_free ? 'YES' : 'NO'}
                        </span>
                      ) : (
                        <span className="text-muted font-mono">NOT TESTED</span>
                      )}
                    </td>

                    <td className="font-mono">
                      {isTested ? row.replans : '—'}
                    </td>

                    <td className="font-mono">
                      {isTested ? `${row.avg_replanning_latency} ms` : '—'}
                    </td>

                    <td>
                      {isTested && row.path_smoothness != null ? (
                        <div className="smoothness-bar-cell">
                          <span className="font-mono">{row.path_smoothness}</span>
                          <div className="progress-bar-track">
                            <div
                              className="progress-bar-fill"
                              style={{
                                width: `${Math.min(100, row.path_smoothness * 100)}%`,
                                backgroundColor: row.path_smoothness >= 0.8 ? '#10b981' : row.path_smoothness >= 0.6 ? '#f59e0b' : '#ef4444',
                              }}
                            />
                          </div>
                        </div>
                      ) : (
                        <span className="text-muted font-mono">—</span>
                      )}
                    </td>

                    <td className="font-mono">
                      {isTested && row.processing_fps != null ? `${row.processing_fps} fps` : '—'}
                    </td>

                    <td>
                      <button
                        className="btn btn-xs btn-outline"
                        onClick={() => handleRunSingle(row.scenario_id)}
                        disabled={runningId === row.scenario_id || isRunningAll}
                      >
                        {runningId === row.scenario_id ? '⏳' : isTested ? 'Re-run' : 'Run'}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── SELECTED SCENARIO DETAILS & EVENT TIMELINE ─────────────────────── */}
      {selectedDetails && (
        <div className="card full-width scenario-details-card" style={{ marginTop: 24 }}>
          <div className="card-header-flex">
            <div>
              <div className="card-title">
                🔍 {selectedDetails.scenario?.scenario_name || selectedScenarioId} — Execution Analysis
              </div>
              <div className="card-subtitle">
                {selectedDetails.scenario?.description}
              </div>
            </div>

            <div className="header-status-group">
              <span className={`status-pill ${selectedDetails.metrics ? 'pill-tested' : 'pill-not-tested'}`}>
                {selectedDetails.metrics ? 'TESTED & VERIFIED' : 'NOT TESTED'}
              </span>
            </div>
          </div>

          <div className="scenario-details-body">
            {/* Challenges & Environment Specs */}
            <div className="specs-grid">
              <div className="spec-box">
                <div className="spec-title">⚠️ Key Road Challenges</div>
                <ul className="spec-list">
                  {selectedDetails.scenario?.challenges?.map((ch, idx) => (
                    <li key={idx}>{ch}</li>
                  ))}
                </ul>
              </div>

              <div className="spec-box">
                <div className="spec-title">🌐 Road &amp; Environment Context</div>
                <div className="spec-key-value">
                  <div><span>Road Type:</span> <b>{selectedDetails.scenario?.expected_environment?.road_type || 'N/A'}</b></div>
                  <div><span>Unmarked Road:</span> <b>{selectedDetails.scenario?.expected_environment?.is_unmarked ? 'YES' : 'NO'}</b></div>
                  <div><span>Intersection:</span> <b>{selectedDetails.scenario?.expected_environment?.has_intersection ? 'YES' : 'NO'}</b></div>
                  <div><span>Speed Limit:</span> <b>{selectedDetails.scenario?.expected_environment?.speed_limit_kmh || 40} km/h</b></div>
                  <div><span>Expected Objects:</span> <b>{selectedDetails.scenario?.expected_objects?.join(', ')}</b></div>
                </div>
              </div>

              {selectedDetails.metrics && (
                <div className="spec-box">
                  <div className="spec-title">📈 Measured Metrics</div>
                  <div className="spec-key-value">
                    <div><span>Pipeline Completed:</span> <b style={{ color: '#10b981' }}>YES</b></div>
                    <div><span>Collision Free:</span> <b style={{ color: selectedDetails.metrics.collision_free ? '#10b981' : '#ef4444' }}>{selectedDetails.metrics.collision_free ? 'YES' : 'NO'}</b></div>
                    <div><span>Path Smoothness:</span> <b>{selectedDetails.metrics.path_smoothness_score} / 1.0</b></div>
                    <div><span>Avg Replanning Latency:</span> <b>{selectedDetails.metrics.average_replanning_latency_ms} ms</b></div>
                    <div><span>Min / Max Latency:</span> <b>{selectedDetails.metrics.minimum_replanning_latency_ms} / {selectedDetails.metrics.maximum_replanning_latency_ms} ms</b></div>
                    <div><span>Replanning Count:</span> <b>{selectedDetails.metrics.replans}</b></div>
                    <div><span>Processing Speed:</span> <b>{selectedDetails.metrics.processing_fps} FPS</b></div>
                  </div>
                </div>
              )}
            </div>

            {/* Deduplicated Event Log for this scenario */}
            {selectedDetails.metrics?.events && selectedDetails.metrics.events.length > 0 && (
              <div className="events-timeline-container" style={{ marginTop: 20 }}>
                <div className="events-sub-header">
                  <span>📜 State-Change Deduplicated Events ({selectedDetails.metrics.events.length} transitions)</span>
                </div>

                <div className="table-responsive-wrapper">
                  <table className="events-timeline-table">
                    <thead>
                      <tr>
                        <th>Time (s)</th>
                        <th>Frame</th>
                        <th>Threat Obstacle</th>
                        <th>Risk Level</th>
                        <th>TTC (s)</th>
                        <th>Decision Action</th>
                        <th>Path Strategy</th>
                        <th>System Rationale</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedDetails.metrics.events.map((ev, idx) => (
                        <tr key={idx} className={ev.risk_level === 'CRITICAL' ? 'row-critical' : ev.risk_level === 'HIGH' ? 'row-high' : ''}>
                          <td className="font-mono">{ev.timestamp_sec != null ? `${ev.timestamp_sec.toFixed(2)}s` : '—'}</td>
                          <td className="font-mono">#{ev.frame_idx}</td>
                          <td className="threat-cell font-semibold">{ev.threat_obstacle}</td>
                          <td>
                            <span className="badge-micro" style={{ background: getRiskBadgeColor(ev.risk_level) }}>
                              {ev.risk_level}
                            </span>
                          </td>
                          <td className="font-mono">{ev.ttc != null ? `${ev.ttc}s` : 'None'}</td>
                          <td>
                            <span className="badge-micro" style={{ background: getActionBadgeColor(ev.decision) }}>
                              {ev.decision}
                            </span>
                          </td>
                          <td className="path-cell">{ev.path}</td>
                          <td className="reason-cell">{ev.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
