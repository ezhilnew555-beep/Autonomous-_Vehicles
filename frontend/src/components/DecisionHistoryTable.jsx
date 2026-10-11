/**
 * DecisionHistoryTable — Filterable timeline of autonomous decision commands and rationales.
 *
 * Props:
 *   stats: DetectionStats from API
 */
import { useState } from 'react'

export default function DecisionHistoryTable({ stats }) {
  const [filter, setFilter] = useState('ALL') // 'ALL' | 'EMERGENCY_STOP' | 'STOP' | 'AVOID' | 'REPLAN' | 'SLOW_DOWN'

  const history = stats?.decision_history || []

  if (!stats) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        Decision history will appear here after video processing.
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        No decision records generated.
      </div>
    )
  }

  // Deduplicate for display: only show when action or target changes, or every 20 frames
  const significantDecisions = []
  let prevAction = null
  let prevTarget = null

  history.forEach((d, idx) => {
    const isChange = d.action !== prevAction || d.target_object_id !== prevTarget
    const isPeriodic = idx % 24 === 0
    if (isChange || isPeriodic || d.action === 'EMERGENCY_STOP') {
      significantDecisions.push(d)
      prevAction = d.action
      prevTarget = d.target_object_id
    }
  })

  const filteredDecisions = filter === 'ALL'
    ? significantDecisions
    : significantDecisions.filter((d) => d.action === filter)

  const renderActionBadge = (action, dir) => {
    switch (action) {
      case 'EMERGENCY_STOP':
        return <span className="action-badge action-badge-emergency">EMERGENCY STOP</span>
      case 'STOP':
        return <span className="action-badge action-badge-stop">STOP</span>
      case 'AVOID':
        return (
          <span className="action-badge action-badge-avoid">
            AVOID {dir === 'LEFT' ? '⬅️' : dir === 'RIGHT' ? '➡️' : ''}
          </span>
        )
      case 'REPLAN':
        return <span className="action-badge action-badge-replan">REPLAN</span>
      case 'SLOW_DOWN':
        return <span className="action-badge action-badge-slow">SLOW DOWN</span>
      default:
        return <span className="action-badge action-badge-continue">CONTINUE</span>
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
          Filter Action:
        </span>
        <div className="filter-buttons">
          {['ALL', 'EMERGENCY_STOP', 'STOP', 'AVOID', 'REPLAN', 'SLOW_DOWN', 'CONTINUE'].map((f) => {
            const count = history.filter((d) => (f === 'ALL' ? true : d.action === f)).length
            if (f !== 'ALL' && count === 0) return null
            return (
              <button
                key={f}
                className={`filter-btn ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f.replace('_', ' ')} ({count})
              </button>
            )
          })}
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Frame</th>
              <th>Action Command</th>
              <th>Target Obstacle</th>
              <th>Throttle</th>
              <th>TTC</th>
              <th>Operational Safety Rationale</th>
            </tr>
          </thead>
          <tbody>
            {filteredDecisions.map((d, idx) => (
              <tr key={`${d.frame_idx}-${idx}`}>
                <td style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>
                  {formatTimestamp(d.timestamp_sec)}
                </td>
                <td style={{ fontFamily: 'monospace' }}>#{d.frame_idx}</td>
                <td>{renderActionBadge(d.action, d.avoidance_direction)}</td>
                <td>
                  {d.target_object_id != null ? (
                    <span>
                      <strong>#{d.target_object_id}</strong> {d.target_class?.toUpperCase()}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>— (None)</span>
                  )}
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  {Math.round(d.target_speed_factor * 100)}%
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  {d.ttc_sec != null ? (
                    <strong style={{ color: d.ttc_sec < 1.0 ? 'var(--red)' : d.ttc_sec < 2.5 ? 'var(--orange)' : 'var(--yellow)' }}>
                      {d.ttc_sec.toFixed(1)} s
                    </strong>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>—</span>
                  )}
                </td>
                <td style={{ fontSize: '0.78rem', color: 'var(--text-main)' }}>
                  {d.reason}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
