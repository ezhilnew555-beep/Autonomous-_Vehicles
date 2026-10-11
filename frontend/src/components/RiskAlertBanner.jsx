/**
 * RiskAlertBanner — Highlights critical and high collision risks detected in the video.
 *
 * Props:
 *   stats: DetectionStats from API
 */
export default function RiskAlertBanner({ stats }) {
  if (!stats || !stats.risk_events || stats.risk_events.length === 0) {
    return null
  }

  const critCount = stats.critical_events_count || 0
  const highCount = stats.high_events_count || 0

  if (critCount === 0 && highCount === 0) {
    return (
      <div className="risk-banner risk-banner-safe">
        <span className="banner-icon">🛡️</span>
        <div>
          <strong>Autonomous Path Clear:</strong> No high or critical collision conflicts detected along the vehicle corridor.
        </div>
      </div>
    )
  }

  // Find the most urgent event
  const urgentEvent = stats.risk_events.find((e) => e.risk_level === 'CRITICAL') ||
                      stats.risk_events.find((e) => e.risk_level === 'HIGH')

  const isCritical = critCount > 0
  const bannerClass = isCritical ? 'risk-banner risk-banner-critical' : 'risk-banner risk-banner-high'

  return (
    <div className={bannerClass}>
      <span className="banner-icon pulse">⚠️</span>
      <div className="banner-content">
        <div className="banner-title">
          <strong>
            {isCritical ? 'CRITICAL COLLISION ALERT' : 'COLLISION HAZARD DETECTED'}
          </strong>
          <span className="banner-counts">
            {critCount > 0 && <span className="risk-tag risk-tag-critical">{critCount} Critical</span>}
            {highCount > 0 && <span className="risk-tag risk-tag-high">{highCount} High</span>}
          </span>
        </div>
        {urgentEvent && (
          <div className="banner-detail">
            Urgent Incident at {urgentEvent.timestamp_sec.toFixed(1)}s: <strong>{urgentEvent.object_class.toUpperCase()} #{urgentEvent.object_id}</strong>
            {urgentEvent.ttc_sec != null ? ` — TTC: ${urgentEvent.ttc_sec.toFixed(1)}s` : ''} | {urgentEvent.reason}
          </div>
        )}
      </div>
    </div>
  )
}
