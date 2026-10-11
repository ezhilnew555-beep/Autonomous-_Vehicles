/**
 * EnvironmentEventsTable — Filterable log of detected road signals and signs.
 *
 * Props:
 *   stats: DetectionStats from API
 */
import { useState } from 'react'

export default function EnvironmentEventsTable({ stats }) {
  const [tab, setTab] = useState('ALL') // ALL | SIGNALS | SIGNS

  const signals = stats?.traffic_signals_detected || []
  const signs = stats?.road_signs_detected || []

  if (!stats) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        Environmental events will appear here after video analysis.
      </div>
    )
  }

  const hasItems = signals.length > 0 || signs.length > 0

  if (!hasItems) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        No explicit traffic lights or road signs were detected during this run. The vehicle navigated according to nominal corridor safety.
      </div>
    )
  }

  const formatTimestamp = (sec) => {
    if (sec == null) return '—'
    const mins = Math.floor(sec / 60)
    const remainingSecs = (sec % 60).toFixed(1)
    return `${mins.toString().padStart(2, '0')}:${remainingSecs.padStart(4, '0')}`
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button
          className={`btn ${tab === 'ALL' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '4px 12px', fontSize: '0.75rem' }}
          onClick={() => setTab('ALL')}
        >
          All Items ({signals.length + signs.length})
        </button>
        <button
          className={`btn ${tab === 'SIGNALS' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '4px 12px', fontSize: '0.75rem' }}
          onClick={() => setTab('SIGNALS')}
        >
          Traffic Signals ({signals.length})
        </button>
        <button
          className={`btn ${tab === 'SIGNS' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '4px 12px', fontSize: '0.75rem' }}
          onClick={() => setTab('SIGNS')}
        >
          Road Signs ({signs.length})
        </button>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Type</th>
              <th>Category</th>
              <th>Recognition / Value</th>
              <th>Confidence</th>
              <th>Action Directive</th>
              <th>Observed Time</th>
            </tr>
          </thead>
          <tbody>
            {/* Traffic signals */}
            {(tab === 'ALL' || tab === 'SIGNALS') &&
              signals.map((sig, idx) => (
                <tr key={`sig-${idx}`}>
                  <td>
                    <span style={{ fontSize: '1.1rem', marginRight: 6 }}>
                      {sig.color === 'RED' ? '🔴' : sig.color === 'YELLOW' ? '🟡' : '🟢'}
                    </span>
                    <strong>TRAFFIC LIGHT</strong>
                  </td>
                  <td>Signal State</td>
                  <td>
                    <span className={`signal-badge signal-badge-${sig.color?.toLowerCase()}`} style={{ fontSize: '0.7rem' }}>
                      {sig.color}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'monospace' }}>
                    {Math.round(sig.confidence * 100)}% ({sig.is_reliable ? 'Verified' : 'Uncertain'})
                  </td>
                  <td style={{ fontSize: '0.78rem', color: 'var(--text-main)' }}>
                    {sig.reason}
                  </td>
                  <td style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                    Active
                  </td>
                </tr>
              ))}

            {/* Road Signs */}
            {(tab === 'ALL' || tab === 'SIGNS') &&
              signs.map((sign, idx) => (
                <tr key={`sign-${idx}`}>
                  <td>
                    <span style={{ fontSize: '1.1rem', marginRight: 6 }}>
                      {sign.sign_type === 'STOP' && '🛑'}
                      {sign.sign_type === 'NO_ENTRY' && '⛔'}
                      {sign.sign_type === 'SPEED_LIMIT' && '🚏'}
                      {sign.sign_type === 'PEDESTRIAN_CROSSING' && '🚸'}
                      {sign.sign_type === 'WARNING' && '⚠️'}
                    </span>
                    <strong>ROAD SIGN</strong>
                  </td>
                  <td>{sign.sign_type.replace(/_/g, ' ')}</td>
                  <td>{sign.value || 'Regulatory'}</td>
                  <td style={{ fontFamily: 'monospace' }}>
                    {Math.round(sign.confidence * 100)}%
                  </td>
                  <td>
                    <span className="path-badge path-badge-replan" style={{ fontSize: '0.68rem' }}>
                      {sign.action_directive?.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>
                    {formatTimestamp(sign.timestamp_sec)} {sign.frame_idx ? `(#${sign.frame_idx})` : ''}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
