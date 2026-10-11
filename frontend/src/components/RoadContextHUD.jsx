/**
 * RoadContextHUD — Phase 8: Environment & Road-Context Awareness
 *
 * Visualizes:
 *  1. Traffic Light State (interactive 3-lamp signal with glowing Red/Yellow/Green bulbs)
 *  2. Active Road Signs (STOP, No Entry, Speed Limit, Pedestrian Crossing, Warning)
 *  3. Modular Map & Road Layout (Geometry, Intersection proximity, Destination directive)
 *  4. Core Safety Axiom Notice: Obstacle safety overrides signals/map
 */
export default function RoadContextHUD({ stats }) {
  if (!stats) return null

  const signal = stats.active_traffic_signal
  const signs = stats.road_signs_detected || []
  const mapProfile = stats.active_map_profile

  const activeColor = signal?.color || 'UNKNOWN'
  const isRed = activeColor === 'RED'
  const isYellow = activeColor === 'YELLOW'
  const isGreen = activeColor === 'GREEN'
  const confPct = signal ? Math.round(signal.confidence * 100) : 0

  const getSignalBadge = () => {
    switch (activeColor) {
      case 'RED':
        return <span className="signal-badge signal-badge-red">🔴 RED SIGNAL — COMPULSORY STOP</span>
      case 'YELLOW':
        return <span className="signal-badge signal-badge-yellow">🟡 YELLOW SIGNAL — PREPARE TO STOP</span>
      case 'GREEN':
        return <span className="signal-badge signal-badge-green">🟢 GREEN SIGNAL — RIGHT OF WAY</span>
      default:
        return <span className="signal-badge signal-badge-dim">⚪ SIGNAL MONITORING / UNMARKED</span>
    }
  }

  const getSignClass = (type) => {
    switch (type) {
      case 'STOP':
      case 'NO_ENTRY':
      case 'NO_LEFT_TURN':
      case 'NO_RIGHT_TURN':
        return 'sign-chip-red'
      case 'WARNING':
      case 'YIELD':
        return 'sign-chip-yellow'
      case 'PEDESTRIAN_CROSSING':
      case 'SPEED_LIMIT':
        return 'sign-chip-blue'
      default:
        return 'sign-chip-default'
    }
  }

  return (
    <div className="card full-width env-hud-card">
      <div className="card-title">
        🌐 Environment &amp; Road-Context Awareness HUD (Phase 8)
        {getSignalBadge()}
      </div>

      <div className="env-hud-grid">

        {/* 1. Traffic Signal Recognition Panel */}
        <div className="env-hud-box signal-box">
          <div className="env-box-title">Traffic Light State</div>
          <div className="signal-content">
            <div className="signal-housing">
              <div className={`signal-lamp lamp-red ${isRed ? 'lit' : ''}`} title="Red Lamp" />
              <div className={`signal-lamp lamp-yellow ${isYellow ? 'lit' : ''}`} title="Yellow Lamp" />
              <div className={`signal-lamp lamp-green ${isGreen ? 'lit' : ''}`} title="Green Lamp" />
            </div>
            <div className="signal-details">
              <div className="signal-color-label" style={{
                color: isRed ? '#ef4444' : isYellow ? '#f59e0b' : isGreen ? '#10b981' : 'var(--text-muted)'
              }}>
                {activeColor} {signal && `(${confPct}% conf)`}
              </div>
              <div className="signal-desc">
                {signal?.reason || 'No active signal fixture detected in corridor.'}
              </div>
              {signal && (
                <div className="signal-meta">
                  Reliability: <strong>{signal.is_reliable ? 'Verified' : 'Uncertain (Monitoring)'}</strong>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 2. Detected Regulatory & Warning Road Signs */}
        <div className="env-hud-box">
          <div className="env-box-title">Active Road Signs ({signs.length})</div>
          {signs.length === 0 ? (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', paddingTop: 10 }}>
              No regulatory signs detected in current corridor view.
            </div>
          ) : (
            <div className="signs-list">
              {signs.map((s, idx) => (
                <div key={`${s.sign_type}-${idx}`} className={`sign-chip ${getSignClass(s.sign_type)}`}>
                  <span className="sign-icon">
                    {s.sign_type === 'STOP' && '🛑'}
                    {s.sign_type === 'NO_ENTRY' && '⛔'}
                    {s.sign_type === 'NO_LEFT_TURN' && '🚫⬅️'}
                    {s.sign_type === 'NO_RIGHT_TURN' && '🚫➡️'}
                    {s.sign_type === 'YIELD' && '🔻'}
                    {s.sign_type === 'SPEED_LIMIT' && '🚏'}
                    {s.sign_type === 'PEDESTRIAN_CROSSING' && '🚸'}
                    {s.sign_type === 'WARNING' && '⚠️'}
                  </span>
                  <div className="sign-info">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <strong>{s.sign_type.replace(/_/g, ' ')}</strong>
                      {s.confidence && (
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                          {Math.round(s.confidence * 100)}%
                        </span>
                      )}
                    </div>
                    {s.value && <div className="sign-val" style={{ fontSize: '0.75rem' }}>Value: {s.value}{typeof s.value === 'number' ? ' km/h' : ''}</div>}
                    <div className="sign-directive" style={{ fontWeight: 600, marginTop: 2 }}>
                      Rule: {(s.rule || s.action_directive)?.replace(/_/g, ' ')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 3. Modular Map & Layout Constraints */}
        <div className="env-hud-box">
          <div className="env-box-title">Modular Map &amp; Road Layout</div>
          <div className="map-info-grid">
            <div>
              <span className="map-prop-lbl">Road Type:</span>
              <span className="map-prop-val" style={{ color: 'var(--accent)', fontWeight: 600 }}>
                {mapProfile?.road_type?.replace(/_/g, ' ') || 'URBAN ROAD'}
              </span>
            </div>
            <div>
              <span className="map-prop-lbl">Geometry:</span>
              <span className="map-prop-val">
                {mapProfile?.geometry_type?.replace(/_/g, ' ') || 'STRAIGHT HIGHWAY'}
              </span>
            </div>
            <div>
              <span className="map-prop-lbl">Destination:</span>
              <span className="map-prop-val">
                {mapProfile?.destination_direction || 'STRAIGHT'}
              </span>
            </div>
            <div>
              <span className="map-prop-lbl">Speed Limit:</span>
              <span className="map-prop-val">
                {mapProfile?.speed_limit_kmh || 50} km/h
              </span>
            </div>
            <div>
              <span className="map-prop-lbl">Intersection:</span>
              <span className="map-prop-val" style={{ color: mapProfile?.has_intersection ? 'var(--yellow)' : 'var(--text-muted)' }}>
                {mapProfile?.has_intersection ? '⚠️ Junction Ahead' : 'None (Cruising)'}
              </span>
            </div>
            <div>
              <span className="map-prop-lbl">Restricted Dirs:</span>
              <span className="map-prop-val">
                {mapProfile?.restricted_directions?.length > 0 ? mapProfile.restricted_directions.join(', ') : 'NONE'}
              </span>
            </div>
            <div>
              <span className="map-prop-lbl">Drivable Region:</span>
              <span className="map-prop-val" style={{ fontSize: '0.75rem' }}>
                X: [35px – 605px] (Width: 570px)
              </span>
            </div>
          </div>
          <div className="map-notes">
            {mapProfile?.road_notes || 'Standard bidirectional corridor without map restrictions.'}
          </div>
        </div>

      </div>

      <div className="note-box" style={{ marginTop: 14 }}>
        🛡️ <strong>Safety Precedence Axiom:</strong> Immediate obstacle collision risk strictly overrides
        green signals and map speed limits. If a pedestrian or vehicle enters the corridor, emergency avoidance
        preempts environmental speed directives.
      </div>
    </div>
  )
}
