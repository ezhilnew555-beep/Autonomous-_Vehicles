/**
 * DetectionTable — scrollable table of all tracked objects.
 *
 * Phase 3 columns: Track ID | Class | Avg Confidence | Frames Seen | Avg Speed | Direction
 * Phase 4 columns: + Prediction | Predicted Direction
 * Phase 5 columns: + Max Risk | Min TTC
 *
 * Props:
 *   stats  {DetectionStats|null}
 */
export default function DetectionTable({ stats }) {
  const objects = stats?.tracked_objects ?? []

  if (!stats) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        Tracked objects will appear here after processing.
      </div>
    )
  }

  if (objects.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        No objects were tracked. Try lowering the confidence threshold.
      </div>
    )
  }

  const confPct = (v) => `${Math.round((v ?? 0) * 100)}%`

  const predMethodBadge = (method) => {
    if (!method || method === 'none') return <span style={{ color: 'var(--text-muted)' }}>—</span>
    const isKalman = method === 'kalman'
    return (
      <span style={{
        background: isKalman ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)',
        color: isKalman ? 'var(--green)' : 'var(--yellow)',
        borderRadius: 4,
        padding: '2px 7px',
        fontSize: '0.72rem',
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}>
        {isKalman ? '🔵 Kalman' : '〰 Linear'}
      </span>
    )
  }

  const riskBadge = (risk) => {
    switch (risk) {
      case 'CRITICAL':
        return <span className="risk-badge risk-badge-critical">CRITICAL</span>
      case 'HIGH':
        return <span className="risk-badge risk-badge-high">HIGH</span>
      case 'MEDIUM':
        return <span className="risk-badge risk-badge-medium">MEDIUM</span>
      default:
        return <span className="risk-badge risk-badge-low">LOW</span>
    }
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Track ID</th>
            <th>Class</th>
            <th>Avg Confidence</th>
            <th>Frames Seen</th>
            <th>Avg Speed (px/f)</th>
            <th>Direction</th>
            <th>Prediction</th>
            <th>Pred Direction</th>
            <th>Max Risk</th>
            <th>Min TTC</th>
          </tr>
        </thead>
        <tbody>
          {objects.map((obj) => (
            <tr key={obj.track_id}>
              <td>
                <strong style={{ color: 'var(--accent)' }}>#{obj.track_id}</strong>
              </td>
              <td>{obj.class_name.charAt(0).toUpperCase() + obj.class_name.slice(1)}</td>
              <td>
                <div className="conf-bar">
                  <div className="conf-track">
                    <div
                      className="conf-fill"
                      style={{ width: confPct(obj.avg_confidence) }}
                    />
                  </div>
                  <span style={{ minWidth: 36, textAlign: 'right' }}>
                    {confPct(obj.avg_confidence)}
                  </span>
                </div>
              </td>
              <td>{obj.frame_count?.toLocaleString()}</td>
              <td style={{ fontFamily: 'monospace' }}>
                {obj.avg_pixel_speed != null ? obj.avg_pixel_speed.toFixed(1) : '—'}
              </td>
              <td>
                {obj.last_direction && obj.last_direction !== '—' ? (
                  <span className="dir-badge">{obj.last_direction}</span>
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>—</span>
                )}
              </td>
              <td>{predMethodBadge(obj.prediction_method)}</td>
              <td>
                {obj.predicted_direction && obj.predicted_direction !== '—' ? (
                  <span className="dir-badge" style={{ borderColor: 'var(--green)', color: 'var(--green)' }}>
                    {obj.predicted_direction}
                  </span>
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>—</span>
                )}
              </td>
              <td>{riskBadge(obj.max_risk_level)}</td>
              <td style={{ fontFamily: 'monospace' }}>
                {obj.min_ttc_sec != null ? (
                  <strong style={{
                    color: obj.min_ttc_sec < 1.0 ? 'var(--red)' : obj.min_ttc_sec < 2.5 ? 'var(--orange)' : 'var(--yellow)'
                  }}>
                    {obj.min_ttc_sec.toFixed(1)} s
                  </strong>
                ) : (
                  <span style={{ color: 'var(--text-muted)' }}>—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
