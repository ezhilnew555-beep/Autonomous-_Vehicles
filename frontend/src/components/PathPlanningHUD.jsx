/**
 * PathPlanningHUD — Visual summary of adaptive collision-free path planning.
 *
 * Props:
 *   stats: DetectionStats from API
 */
export default function PathPlanningHUD({ stats }) {
  if (!stats) return null

  const finalStatus = stats.final_path_status || 'CONTINUE_ORIGINAL'
  const replanCount = stats.replanning_count || 0
  const avgTime = stats.avg_planning_time_ms || 0.0

  const getStatusBadge = (status) => {
    switch (status) {
      case 'REPLANNED_AVOID_LEFT':
        return <span className="path-badge path-badge-replan">⬅️ REPLANNED (AVOID LEFT)</span>
      case 'REPLANNED_AVOID_RIGHT':
        return <span className="path-badge path-badge-replan">➡️ REPLANNED (AVOID RIGHT)</span>
      case 'NO_SAFE_PATH':
        return <span className="path-badge path-badge-blocked">❌ NO SAFE PATH (OBSTRUCTED)</span>
      default:
        return <span className="path-badge path-badge-nominal">🛡️ NOMINAL PATH (CLEAR)</span>
    }
  }

  return (
    <div className="card full-width path-hud-card">
      <div className="card-title">
        🛣️ Adaptive Path Planning &amp; Collision Avoidance HUD
        {getStatusBadge(finalStatus)}
      </div>

      <div className="path-hud-grid">
        {/* Trajectory State Overview */}
        <div className="path-hud-box">
          <div className="path-box-title">Trajectory State</div>
          <div className="path-box-value">
            {finalStatus.replace(/_/g, ' ')}
          </div>
          <div className="path-box-desc">
            {replanCount > 0
              ? `${replanCount} replanning event(s) executed to avert dynamic road conflicts.`
              : 'Nominal travel corridor unobstructed. Cruising trajectory maintained.'}
          </div>
        </div>

        {/* Planning Computation Latency */}
        <div className="path-hud-box">
          <div className="path-box-title">Real-Time Plan Latency</div>
          <div className="path-box-value" style={{ color: avgTime < 10 ? 'var(--green)' : 'var(--yellow)' }}>
            {avgTime.toFixed(1)} ms
          </div>
          <div className="path-box-desc">
            Quintic spline generation &amp; multi-objective scoring (target &lt; 20ms).
          </div>
        </div>

        {/* Path Legend */}
        <div className="path-hud-box">
          <div className="path-box-title">Video Trajectory Legend</div>
          <div className="path-legend-list">
            <div className="legend-row">
              <span className="legend-line line-original" />
              <span><strong>ORIGINAL PATH:</strong> Nominal lane centerline</span>
            </div>
            <div className="legend-row">
              <span className="legend-line line-safe" />
              <span><strong>NEW SAFE PATH:</strong> Selected collision-free evasive curve</span>
            </div>
          </div>
        </div>
      </div>

      <div className="note-box" style={{ marginTop: 14 }}>
        ⚠️ <strong>Prototyping &amp; Simulation Notice:</strong> This platform is a video-analysis and simulation prototype
        for Indian road conditions. It demonstrates adaptive path planning and does not control a physical vehicle.
      </div>
    </div>
  )
}
