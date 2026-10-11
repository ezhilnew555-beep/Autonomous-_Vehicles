/**
 * ReplanningEventsTable — Filterable log of path replanning occurrences.
 *
 * Props:
 *   stats: DetectionStats from API
 */
import { useState } from 'react'

export default function ReplanningEventsTable({ stats }) {
  const [filter, setFilter] = useState('ALL')

  const events = stats?.replanning_events || []

  if (!stats) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        Replanning events will appear here after video analysis.
      </div>
    )
  }

  if (events.length === 0) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        No replanning events required. The vehicle maintained its nominal trajectory throughout the video.
      </div>
    )
  }

  const formatTimestamp = (sec) => {
    const mins = Math.floor(sec / 60)
    const remainingSecs = (sec % 60).toFixed(1)
    return `${mins.toString().padStart(2, '0')}:${remainingSecs.padStart(4, '0')}`
  }

  return (
    <div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Frame</th>
              <th>Trigger Obstacle</th>
              <th>Previous Path</th>
              <th>New Planned Path</th>
              <th>Plan Latency</th>
              <th>Replanning Rationale</th>
            </tr>
          </thead>
          <tbody>
            {events.map((ev, idx) => (
              <tr key={`${ev.frame_idx}-${idx}`}>
                <td style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>
                  {formatTimestamp(ev.timestamp_sec)}
                </td>
                <td style={{ fontFamily: 'monospace' }}>#{ev.frame_idx}</td>
                <td>
                  {ev.obstacle_id != null ? (
                    <strong>#{ev.obstacle_id} ({ev.obstacle_class?.toUpperCase()})</strong>
                  ) : (
                    <span style={{ color: 'var(--text-muted)' }}>Corridor Obstruction</span>
                  )}
                </td>
                <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {ev.previous_status.replace(/_/g, ' ')}
                </td>
                <td>
                  <span className="path-badge path-badge-replan" style={{ fontSize: '0.7rem' }}>
                    {ev.new_status.replace(/_/g, ' ')}
                  </span>
                </td>
                <td style={{ fontFamily: 'monospace' }}>
                  {ev.planning_time_ms.toFixed(1)} ms
                </td>
                <td style={{ fontSize: '0.78rem', color: 'var(--text-main)' }}>
                  {ev.trigger_reason}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
