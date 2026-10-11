/**
 * StatsPanel — summary statistics cards, class distribution, collision risk, decisions, and adaptive path planning.
 * Supports Phase 2 Detection Summary as well as full autonomous pipeline stats.
 *
 * Props:
 *   stats  {DetectionStats|null}  from the job status API response
 */
export default function StatsPanel({ stats }) {
  if (!stats) {
    return (
      <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', padding: '20px 0' }}>
        Stats will appear here after processing completes.
      </div>
    )
  }

  // ── PHASE 2 PURE DETECTION SUMMARY ──────────────────────────────────────────
  if (stats.pipeline_mode === 'detection') {
    const totalUnique = stats.total_objects_detected || stats.unique_track_ids || Object.values(stats.objects_by_class || {}).reduce((a, b) => a + b, 0)
    const totalFrameDets = stats.total_frame_detections || 0
    const peakConcurrent = stats.peak_concurrent_objects || 0
    const procTime = stats.processing_time_sec != null
      ? stats.processing_time_sec.toFixed(1)
      : (stats.processed_frames / (stats.processing_fps || 1)).toFixed(1)

    const keyCategories = [
      { key: 'car', label: 'Cars', icon: '🚗' },
      { key: 'motorcycle', label: 'Bikes', icon: '🏍️' },
      { key: 'person', label: 'People', icon: '🚶' },
      { key: 'truck', label: 'Trucks', icon: '🚚' },
      { key: 'bus', label: 'Buses', icon: '🚌' },
      { key: 'bicycle', label: 'Bicycles', icon: '🚲' },
      { key: 'animal', label: 'Animals (Cows, etc.)', icon: '🐄', match: ['cow', 'horse', 'sheep', 'dog', 'animal', 'bird', 'cat'] },
    ]

    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="badge" style={{ background: '#0284c7', color: '#fff', fontWeight: 700 }}>
              PHASE 2: AI ROAD OBJECT DETECTION
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              YOLO11 COCO-80 + Temporal Consistency
            </span>
          </div>
          {totalFrameDets > 0 && (
            <span style={{ fontSize: '0.74rem', color: 'var(--accent)', background: 'rgba(56, 189, 248, 0.1)', padding: '2px 8px', borderRadius: 4 }}>
              {totalFrameDets.toLocaleString()} detections evaluated
            </span>
          )}
        </div>

        {/* Top Metric Cards */}
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', marginBottom: 16 }}>
          <div className="stat-box">
            <div className="stat-value" style={{ color: 'var(--accent)' }}>{totalUnique.toLocaleString()}</div>
            <div className="stat-label">Unique Objects Tracked</div>
          </div>
          {peakConcurrent > 0 && (
            <div className="stat-box">
              <div className="stat-value" style={{ color: '#38bdf8' }}>{peakConcurrent.toLocaleString()}</div>
              <div className="stat-label">Peak On-Road at Once</div>
            </div>
          )}
          <div className="stat-box">
            <div className="stat-value">{stats.processed_frames?.toLocaleString()}</div>
            <div className="stat-label">Frames Processed</div>
          </div>
          <div className="stat-box">
            <div className="stat-value">{procTime}s</div>
            <div className="stat-label">Processing Time</div>
          </div>
          <div className="stat-box">
            <div className="stat-value" style={{ color: '#10b981' }}>{stats.processing_fps}</div>
            <div className="stat-label">Average FPS</div>
          </div>
        </div>

        {/* Detection Summary Table */}
        <div style={{ background: 'var(--bg-input)', borderRadius: 8, padding: '12px 16px', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.8 }}>
              Unique Road Objects Summary
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              (Filtered for false alarms & class flickers)
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px 16px' }}>
            {keyCategories.map((cat) => {
              let count = 0
              if (cat.match) {
                count = cat.match.reduce((acc, k) => acc + (stats.objects_by_class?.[k] || 0), 0)
              } else {
                count = stats.objects_by_class?.[cat.key] || 0
              }
              return (
                <div key={cat.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '4px 0', borderBottom: '1px dashed rgba(255,255,255,0.08)' }}>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                    {cat.icon} {cat.label}:
                  </span>
                  <strong style={{ fontSize: '0.9rem', color: count > 0 ? '#10b981' : 'var(--text-muted)' }}>
                    {count.toLocaleString()}
                  </strong>
                </div>
              )
            })}
          </div>
        </div>

        {/* All detected classes pill list */}
        {stats.objects_by_class && Object.keys(stats.objects_by_class).length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>
              Confirmed Unique Classes
            </div>
            <div className="class-pills">
              {Object.entries(stats.objects_by_class)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, count]) => (
                  <div className="class-pill" key={cls}>
                    {cls.charAt(0).toUpperCase() + cls.slice(1)}
                    <span>{count.toLocaleString()} unique</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        <div className="note-box" style={{ marginTop: 14 }}>
          ℹ️ <strong>Temporal Class Smoothing & Tracking:</strong> Object identities and classifications are stabilized across video frames using majority class voting. Fleeting false alarms and camera-car dashboard artifacts are filtered out.
        </div>
      </div>
    )
  }

  // ── PHASE 3 OBJECT TRACKING & MOVEMENT ESTIMATION ────────────────────────────
  if (stats.pipeline_mode === 'tracking') {
    const activeTracksCount = stats.active_tracks_count || (stats.active_tracks ? stats.active_tracks.length : 0)
    const totalTracksCount = stats.total_tracks_count || stats.unique_track_ids || 0
    const trackingDuration = stats.tracking_duration_sec != null
      ? stats.tracking_duration_sec.toFixed(1)
      : stats.duration_seconds ? stats.duration_seconds.toFixed(1) : '—'
    const avgFps = stats.processing_fps || '—'
    const activeTracks = stats.active_tracks || []

    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="badge" style={{ background: '#10b981', color: '#fff', fontWeight: 700 }}>
              PHASE 3: OBJECT TRACKING + MOVEMENT ESTIMATION
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              ByteTrack Persistent IDs + Movement Vectors
            </span>
          </div>
          <span style={{ fontSize: '0.74rem', color: '#10b981', background: 'rgba(16, 185, 129, 0.1)', padding: '2px 8px', borderRadius: 4 }}>
            {totalTracksCount} persistent tracks
          </span>
        </div>

        {/* 4 Required Metric Cards */}
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', marginBottom: 16 }}>
          <div className="stat-box">
            <div className="stat-value" style={{ color: '#10b981' }}>{activeTracksCount}</div>
            <div className="stat-label">Active Tracked Objects</div>
          </div>
          <div className="stat-box">
            <div className="stat-value" style={{ color: 'var(--accent)' }}>{totalTracksCount}</div>
            <div className="stat-label">Total Tracked Objects</div>
          </div>
          <div className="stat-box">
            <div className="stat-value">{trackingDuration}s</div>
            <div className="stat-label">Tracking Duration</div>
          </div>
          <div className="stat-box">
            <div className="stat-value" style={{ color: '#38bdf8' }}>{avgFps}</div>
            <div className="stat-label">Average Processing FPS</div>
          </div>
        </div>

        {/* ACTIVE TRACKS Section */}
        <div style={{ background: 'var(--bg-input)', borderRadius: 8, padding: '14px 16px', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#fff', textTransform: 'uppercase', letterSpacing: 0.8 }}>
              🎯 ACTIVE TRACKS
            </div>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Estimated Pixel Speed (px/f)
            </span>
          </div>

          {activeTracks.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', padding: '12px 0', textAlign: 'center' }}>
              No active tracks in the final frame. Total {totalTracksCount} tracks recorded throughout video.
            </div>
          ) : (
            <div className="table-wrap" style={{ maxHeight: 240, overflowY: 'auto' }}>
              <table style={{ width: '100%', fontSize: '0.82rem', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-muted)', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px' }}>ID</th>
                    <th style={{ padding: '6px 8px' }}>CLASS</th>
                    <th style={{ padding: '6px 8px' }}>CONFIDENCE</th>
                    <th style={{ padding: '6px 8px' }}>DIRECTION</th>
                    <th style={{ padding: '6px 8px' }}>SPEED</th>
                  </tr>
                </thead>
                <tbody>
                  {activeTracks.map((tr) => (
                    <tr key={tr.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '6px 8px' }}>
                        <strong style={{ color: 'var(--accent)' }}>#{tr.id}</strong>
                      </td>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>
                        {tr.class_name.charAt(0).toUpperCase() + tr.class_name.slice(1)}
                      </td>
                      <td style={{ padding: '6px 8px' }}>
                        {Math.round(tr.confidence * 100)}%
                      </td>
                      <td style={{ padding: '6px 8px' }}>
                        <span style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 4,
                          padding: '2px 6px',
                          borderRadius: 4,
                          fontSize: '0.75rem',
                          background: tr.direction === 'STATIONARY' ? 'rgba(255,255,255,0.08)' : 'rgba(16, 185, 129, 0.15)',
                          color: tr.direction === 'STATIONARY' ? 'var(--text-muted)' : '#10b981',
                        }}>
                          <span>{tr.direction_symbol}</span>
                          <span>{tr.direction.charAt(0) + tr.direction.slice(1).toLowerCase()}</span>
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>
                        {tr.speed_px_per_frame != null ? `${tr.speed_px_per_frame.toFixed(1)} px/f` : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Tracked Class Distribution pills */}
        {stats.objects_by_class && Object.keys(stats.objects_by_class).length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>
              Tracked Class Distribution
            </div>
            <div className="class-pills">
              {Object.entries(stats.objects_by_class)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, count]) => (
                  <div className="class-pill" key={cls}>
                    {cls.charAt(0).toUpperCase() + cls.slice(1)}
                    <span>{count.toLocaleString()} unique</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        <div className="note-box" style={{ marginTop: 14 }}>
          ℹ️ <strong>Phase 3 Tracking:</strong> Detected road objects maintain persistent ByteTrack IDs across frames with historical position centers, movement direction (FORWARD, BACKWARD, LEFT, RIGHT, DIAGONAL, STATIONARY), and approximate pixel velocity (px/f). World km/h speed is not claimed without calibration.
        </div>
      </div>
    )
  }

  // ── PHASE 4 SHORT-TERM MOTION PREDICTION ────────────────────────────────────
  if (stats.pipeline_mode === 'prediction') {
    const activePredsCount = stats.active_predictions_count ?? (stats.active_predictions ? stats.active_predictions.filter(p => p.is_available).length : 0)
    const totalTracks = stats.total_tracks_count || stats.unique_track_ids || 0
    const horizonSec = stats.prediction_horizon_sec ?? 2.0
    const intervalSec = stats.prediction_interval_sec ?? 0.2
    const waypointsCount = Math.round(horizonSec / intervalSec)
    const avgConfidence = stats.avg_prediction_confidence != null && stats.avg_prediction_confidence > 0
      ? `${stats.avg_prediction_confidence.toFixed(1)}%`
      : '—'
    const avgFps = stats.processing_fps || '—'
    const activePredictions = stats.active_predictions || []

    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="badge" style={{ background: '#f59e0b', color: '#fff', fontWeight: 700 }}>
              PHASE 4: SHORT-TERM MOTION PREDICTION
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              2D Kalman Filter &bull; Constant-Velocity Model &bull; {horizonSec}s Horizon
            </span>
          </div>
          <span style={{ fontSize: '0.74rem', color: '#f59e0b', background: 'rgba(245, 158, 11, 0.1)', padding: '2px 8px', borderRadius: 4 }}>
            {activePredsCount} active predictions
          </span>
        </div>

        {/* 4 Metric Cards */}
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', marginBottom: 16 }}>
          <div className="stat-box">
            <div className="stat-value" style={{ color: '#f59e0b' }}>{activePredsCount}</div>
            <div className="stat-label">Active Predictions</div>
          </div>
          <div className="stat-box">
            <div className="stat-value" style={{ color: 'var(--accent)' }}>{horizonSec}s</div>
            <div className="stat-label">Horizon ({waypointsCount} Waypoints)</div>
          </div>
          <div className="stat-box">
            <div className="stat-value" style={{ color: '#10b981' }}>{avgConfidence}</div>
            <div className="stat-label">Avg Confidence</div>
          </div>
          <div className="stat-box">
            <div className="stat-value" style={{ color: '#38bdf8' }}>{avgFps}</div>
            <div className="stat-label">Processing FPS</div>
          </div>
        </div>

        {/* ACTIVE PREDICTIONS Section */}
        <div style={{ background: 'var(--bg-input)', borderRadius: 8, padding: '14px 16px', border: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#fff', textTransform: 'uppercase', letterSpacing: 0.8 }}>
              🔮 ACTIVE PREDICTIONS
            </div>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              0.2s Intervals &bull; Class-Weighted Confidence
            </span>
          </div>

          {activePredictions.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', padding: '12px 0', textAlign: 'center' }}>
              No active tracks in the final frame. Total {totalTracks} tracks analyzed.
            </div>
          ) : (
            <div className="table-wrap" style={{ maxHeight: 260, overflowY: 'auto' }}>
              <table style={{ width: '100%', fontSize: '0.82rem', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-muted)', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px' }}>ID</th>
                    <th style={{ padding: '6px 8px' }}>OBJECT</th>
                    <th style={{ padding: '6px 8px' }}>DIRECTION</th>
                    <th style={{ padding: '6px 8px' }}>VELOCITY</th>
                    <th style={{ padding: '6px 8px' }}>HORIZON</th>
                    <th style={{ padding: '6px 8px' }}>CONFIDENCE</th>
                  </tr>
                </thead>
                <tbody>
                  {activePredictions.map((pred) => {
                    const isAvail = pred.is_available
                    const isUnpredictable = ['person', 'cow', 'dog', 'horse', 'sheep', 'animal'].includes(pred.class_name.toLowerCase())

                    return (
                      <tr key={pred.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <td style={{ padding: '6px 8px' }}>
                          <strong style={{ color: 'var(--accent)' }}>#{pred.id}</strong>
                        </td>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span>{pred.class_name.charAt(0).toUpperCase() + pred.class_name.slice(1)}</span>
                            {isUnpredictable && (
                              <span style={{ fontSize: '0.68rem', color: '#c084fc', background: 'rgba(192, 132, 252, 0.15)', padding: '1px 5px', borderRadius: 4 }}>
                                lower certainty
                              </span>
                            )}
                          </span>
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          <span style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 4,
                            padding: '2px 6px',
                            borderRadius: 4,
                            fontSize: '0.75rem',
                            background: pred.direction === 'STATIONARY' ? 'rgba(255,255,255,0.08)' : 'rgba(56, 189, 248, 0.15)',
                            color: pred.direction === 'STATIONARY' ? 'var(--text-muted)' : '#38bdf8',
                          }}>
                            <span>{pred.direction_symbol || '•'}</span>
                            <span>{pred.direction.charAt(0) + pred.direction.slice(1).toLowerCase()}</span>
                          </span>
                        </td>
                        <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>
                          {pred.speed_px_per_frame != null ? `${pred.speed_px_per_frame.toFixed(1)} px/f` : '—'}
                        </td>
                        <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>
                          {pred.horizon_sec != null ? `+${pred.horizon_sec.toFixed(1)}s` : `${horizonSec}s`}
                        </td>
                        <td style={{ padding: '6px 8px' }}>
                          {isAvail ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <span style={{
                                fontWeight: 700,
                                color: pred.confidence_pct >= 80 ? '#10b981' : pred.confidence_pct >= 60 ? '#f59e0b' : '#c084fc',
                              }}>
                                {pred.confidence_pct.toFixed(0)}%
                              </span>
                              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                ({pred.prediction_method === 'kalman' ? 'Kalman' : 'Linear'})
                              </span>
                            </div>
                          ) : (
                            <span style={{
                              fontSize: '0.72rem',
                              color: 'var(--text-muted)',
                              fontStyle: 'italic',
                              background: 'rgba(255,255,255,0.05)',
                              padding: '2px 6px',
                              borderRadius: 4,
                            }}>
                              {pred.unavailability_reason || 'Unavailable'}
                            </span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Tracked Class Distribution pills */}
        {stats.objects_by_class && Object.keys(stats.objects_by_class).length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>
              Tracked Class Distribution
            </div>
            <div className="class-pills">
              {Object.entries(stats.objects_by_class)
                .sort(([, a], [, b]) => b - a)
                .map(([cls, count]) => (
                  <div className="class-pill" key={cls}>
                    {cls.charAt(0).toUpperCase() + cls.slice(1)}
                    <span>{count.toLocaleString()} unique</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        <div className="note-box" style={{ marginTop: 14 }}>
          ℹ️ <strong>Phase 4 Short-Term Motion Prediction:</strong> Constant-velocity 2D Kalman filter estimates future coordinates at 0.2s steps over a 2.0s horizon. Vehicles (cars, buses, trucks) show higher predictability, while pedestrians and animals reflect lower certainty with wider uncertainty cones. Predictions are stability-gated (&ge;5 frames history) to prevent false spikes.
        </div>
      </div>
    )
  }

  // ── FULL AUTONOMOUS PIPELINE STATS ──────────────────────────────────────────
  const critCount = stats.critical_events_count || 0
  const highCount = stats.high_events_count || 0
  const dominantAct = stats.dominant_action || 'CONTINUE'
  const replanCount = stats.replanning_count || 0
  const avgPlanTime = stats.avg_planning_time_ms || 0.0

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

  const statBoxes = [
    { value: stats.processed_frames?.toLocaleString() ?? '—', label: 'Frames' },
    { value: `${stats.processing_fps ?? '—'}`,                 label: 'Proc FPS' },
    { value: `${stats.duration_seconds ?? '—'}s`,              label: 'Duration' },
    { value: stats.unique_track_ids ?? '—',                    label: 'Unique Objects' },
    { value: stats.peak_concurrent_objects ?? '—',             label: 'Peak On-Road' },
    { value: `${stats.video_fps ?? '—'}`,                      label: 'Video FPS' },
    {
      value: dominantAct,
      label: 'Dominant Action',
      color: getActionColor(dominantAct),
    },
    {
      value: critCount,
      label: 'Critical Risks',
      color: critCount > 0 ? 'var(--red)' : 'var(--text-muted)',
    },
    {
      value: highCount,
      label: 'High Risks',
      color: highCount > 0 ? 'var(--orange)' : 'var(--text-muted)',
    },
    {
      value: replanCount,
      label: 'Path Replans',
      color: replanCount > 0 ? '#06b6d4' : 'var(--green)',
    },
    {
      value: `${avgPlanTime.toFixed(1)} ms`,
      label: 'Avg Plan Latency',
      color: 'var(--green)',
    },
    {
      value: stats.active_traffic_signal ? stats.active_traffic_signal.color : 'None',
      label: 'Traffic Signal',
      color: stats.active_traffic_signal?.color === 'RED' ? 'var(--red)' : stats.active_traffic_signal?.color === 'GREEN' ? 'var(--green)' : 'var(--text-muted)',
    },
    {
      value: (stats.road_signs_detected?.length || 0).toString(),
      label: 'Road Signs',
      color: (stats.road_signs_detected?.length || 0) > 0 ? '#06b6d4' : 'var(--text-muted)',
    },
  ]

  return (
    <div>
      <div className="stats-grid">
        {statBoxes.map((s) => (
          <div className="stat-box" key={s.label}>
            <div className="stat-value" style={{ color: s.color || 'var(--accent)', fontSize: s.label === 'Dominant Action' ? '1.1rem' : undefined }}>
              {s.value}
            </div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {stats.objects_by_class && Object.keys(stats.objects_by_class).length > 0 && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
              Confirmed Unique Objects by Class
            </div>
            {stats.total_frame_detections > 0 && (
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                {stats.total_frame_detections.toLocaleString()} frame detections evaluated
              </div>
            )}
          </div>
          <div className="class-pills">
            {Object.entries(stats.objects_by_class)
              .sort(([, a], [, b]) => b - a)
              .map(([cls, count]) => (
                <div className="class-pill" key={cls}>
                  {cls.charAt(0).toUpperCase() + cls.slice(1)}
                  <span>{count.toLocaleString()} unique</span>
                </div>
              ))}
          </div>
        </>
      )}

      <div className="note-box" style={{ marginTop: 14 }}>
        ℹ️ Phase 7 Adaptive Path Planning: Generates quintic polynomial alternative paths around obstacles.
        Visualizes <strong>ORIGINAL PATH</strong> (dashed) vs <strong>NEW SAFE PATH</strong> (solid emerald green).
      </div>
    </div>
  )
}
