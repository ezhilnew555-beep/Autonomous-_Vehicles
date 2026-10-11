/**
 * RiskEventsTable — filterable log of all collision risk events recorded by the backend.
 *
 * Props:
 *   stats: DetectionStats from API
 */
import { useState } from 'react'

export default function RiskEventsTable({ stats }) {
  const [filter, setFilter] = useState('ALL') // 'ALL' | 'CRITICAL' | 'HIGH' | 'MEDIUM'

  const events = stats?.risk_events || []

  if (!stats) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        Collision risk events will appear here after video analysis.
      </div>
    )
  }

  if (events.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        No hazardous collision events were recorded. The ego vehicle travel corridor remained unobstructed.
      </div>
    )
  }

  const filteredEvents = filter === 'ALL'
    ? events
    : events.filter((e) => e.risk_level === filter)

  const renderRiskBadge = (level) => {
    switch (level) {
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

  const formatTimestamp = (sec) => {
    const mins = Math.floor(sec / 60)
    const remainingSecs = (sec % 60).toFixed(1)
    return `${mins.toString().padStart(2, '0')}:${remainingSecs.padStart(4, '0')}`
  }

  return (
    <div>
      <div className="table-filter-bar">
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
          Filter Risk Level:
        </span>
        <div className="filter-buttons">
          {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'].map((f) => (
            <button
              key={f}
              className={`filter-btn ${filter === f ? 'active' : ''}`}
              onClick={() => setFilter(f)}
            >
              {f}
              {f === 'ALL' && ` (${events.length})`}
              {f === 'CRITICAL' && ` (${stats.critical_events_count || 0})`}
              {f === 'HIGH' && ` (${stats.high_events_count || 0})`}
            </button>
          ))}
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Frame</th>
              <th>Object</th>
              <th>Distance</th>
              <th>TTC</th>
              <th>Risk Level</th>
              <th>Mathematical / Behavioral Rationale</th>
            </tr>
          </thead>
          <tbody>
            {filteredEvents.map((ev, idx) => (
              <tr key={`${ev.frame_idx}-${ev.object_id}-${idx}`}>
                <td style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>
                  {formatTimestamp(ev.timestamp_sec)}
                </td>
                <td style={{ fontFamily: 'monospace' }}>#{ev.frame_idx}</td>
                <td>
                  <strong>#{ev.object_id}</strong> {ev.object_class.charAt(0).toUpperCase() + ev.object_class.slice(1)}
                </td>
                <td style={{ fontFamily: 'monospace' }}>{ev.distance_px.toFixed(0)} px</td>
                <td style={{ fontFamily: 'monospace' }}>
                  {ev.ttc_sec != null ? (
                    <strong style={{ color: ev.ttc_sec < 1.0 ? 'var(--red)' : ev.ttc_sec < 2.5 ? 'var(--orange)' : 'var(--yellow)' }}>
                      {ev.ttc_sec.toFixed(1)} s
                    </strong>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>—</span>
                  )}
                </td>
                <td>{renderRiskBadge(ev.risk_level)}</td>
                <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {ev.reason}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
