/**
 * DecisionHUD — Autonomous driving operational command and vehicle status display.
 *
 * Props:
 *   stats: DetectionStats from API
 */
export default function DecisionHUD({ stats }) {
  if (!stats || !stats.decision_history || stats.decision_history.length === 0) {
    return null
  }

  const dominantAction = stats.dominant_action || 'CONTINUE'
  const actionCounts = stats.action_counts || {}
  const totalDecisions = stats.decision_history.length

  // Find the most severe decision in the history
  const emergencyDecisions = stats.decision_history.filter((d) => d.action === 'EMERGENCY_STOP')
  const stopDecisions = stats.decision_history.filter((d) => d.action === 'STOP')
  const avoidDecisions = stats.decision_history.filter((d) => d.action === 'AVOID' || d.action === 'REPLAN')
  const slowDecisions = stats.decision_history.filter((d) => d.action === 'SLOW_DOWN')

  const highlightDecision = emergencyDecisions[0] ||
                            stopDecisions[0] ||
                            avoidDecisions[0] ||
                            slowDecisions[0] ||
                            stats.decision_history[stats.decision_history.length - 1]

  const getActionColor = (act) => {
    switch (act) {
      case 'EMERGENCY_STOP': return 'var(--red)'
      case 'STOP':           return 'var(--orange)'
      case 'AVOID':          return '#06b6d4'
      case 'REPLAN':         return '#c084fc'
      case 'SLOW_DOWN':      return 'var(--yellow)'
      default:               return 'var(--green)'
    }
  }

  const actColor = getActionColor(highlightDecision.action)

  return (
    <div className="card full-width decision-hud-card">
      <div className="card-title">
        🤖 Autonomous Decision Engine Status
        <span style={{
          background: actColor,
          color: '#fff',
          borderRadius: 4,
          padding: '2px 9px',
          fontSize: '0.75rem',
          fontWeight: 700,
          marginLeft: 8,
          letterSpacing: 0.5,
        }}>
          {highlightDecision.action}
          {highlightDecision.avoidance_direction !== 'NONE' && ` (${highlightDecision.avoidance_direction})`}
        </span>
      </div>

      <div className="decision-grid">
        {/* Active Command Overview */}
        <div className="decision-main-box" style={{ borderColor: actColor }}>
          <div className="decision-action-title" style={{ color: actColor }}>
            {highlightDecision.action}
            {highlightDecision.avoidance_direction === 'LEFT' && ' ⬅️ AVOID LEFT'}
            {highlightDecision.avoidance_direction === 'RIGHT' && ' ➡️ AVOID RIGHT'}
          </div>
          <div className="decision-sub-reason">
            {highlightDecision.reason}
          </div>
          {highlightDecision.target_object_id != null && (
            <div className="decision-target-tag">
              Target Obstacle: <strong>#{highlightDecision.target_object_id} ({highlightDecision.target_class})</strong>
              {highlightDecision.ttc_sec != null && ` | TTC: ${highlightDecision.ttc_sec.toFixed(1)}s`}
              {highlightDecision.risk_level && ` | Risk: ${highlightDecision.risk_level}`}
            </div>
          )}
        </div>

        {/* Speed / Throttle Advisory */}
        <div className="decision-speed-box">
          <div className="speed-title">Throttle Advisory</div>
          <div className="speed-gauge" style={{ color: actColor }}>
            {Math.round((highlightDecision.target_speed_factor ?? 1.0) * 100)}%
          </div>
          <div className="speed-label">
            {highlightDecision.target_speed_factor === 0
              ? 'BRAKES FULL ENGAGED'
              : highlightDecision.target_speed_factor < 0.7
              ? 'DECELERATION ACTIVE'
              : 'CRUISING VELOCITY'}
          </div>
        </div>

        {/* Action Distribution Breakdown */}
        <div className="decision-distribution-box">
          <div className="speed-title">Action Distribution ({totalDecisions} Frames)</div>
          <div className="action-distribution-bar">
            {Object.entries(actionCounts)
              .filter(([, count]) => count > 0)
              .map(([action, count]) => {
                const pct = (count / totalDecisions) * 100
                return (
                  <div
                    key={action}
                    className="action-bar-slice"
                    style={{
                      width: `${pct}%`,
                      background: getActionColor(action),
                    }}
                    title={`${action}: ${count} frames (${pct.toFixed(0)}%)`}
                  />
                )
              })}
          </div>
          <div className="action-pill-legend">
            {Object.entries(actionCounts)
              .filter(([, count]) => count > 0)
              .map(([action, count]) => (
                <span key={action} className="action-legend-item">
                  <span className="legend-dot" style={{ background: getActionColor(action) }} />
                  {action}: {count}
                </span>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
