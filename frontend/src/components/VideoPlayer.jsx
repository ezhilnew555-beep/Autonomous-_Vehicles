/**
 * VideoPlayer — HTML5 video player with real-time frame/time readout and download fallback.
 *
 * Props:
 *   src     {string|null}  URL to the video
 *   label   {string}       Card title
 *   empty   {string}       Placeholder message when src is null
 *   fps     {number}       Video FPS (defaults to 24)
 *   totalFrames {number}   Total frames count if known
 */
import { useState, useRef } from 'react'

export default function VideoPlayer({
  src,
  label,
  empty = 'No video yet',
  fps = 24,
  totalFrames = null,
}) {
  const [playbackError, setPlaybackError] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const videoRef = useRef(null)

  const effectiveFps = Math.max(1, fps || 24)
  const currentFrame = Math.floor(currentTime * effectiveFps) + 1
  const calcTotalFrames = totalFrames || (duration > 0 ? Math.floor(duration * effectiveFps) : null)

  const formatTime = (secs) => {
    if (!secs || isNaN(secs)) return '00:00.00'
    const mins = Math.floor(secs / 60)
    const s = Math.floor(secs % 60)
    const ms = Math.floor((secs % 1) * 100)
    return `${String(mins).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(2, '0')}`
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration)
    }
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div className="card-title" style={{ margin: 0 }}>🎬 {label}</div>
        {src && !playbackError && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontSize: '0.75rem',
            fontFamily: 'monospace',
            background: 'var(--bg-secondary)',
            padding: '3px 8px',
            borderRadius: 6,
            border: '1px solid var(--border-color)',
          }}>
            <span style={{ color: 'var(--accent)', fontWeight: 700 }}>
              ⏱ {formatTime(currentTime)} / {formatTime(duration)}
            </span>
            <span style={{ color: 'var(--text-muted)' }}>|</span>
            <span style={{ color: '#10b981', fontWeight: 600 }}>
              🎞 Frame {currentFrame}{calcTotalFrames ? ` / ${calcTotalFrames}` : ''}
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>
              ({effectiveFps} fps)
            </span>
          </div>
        )}
      </div>

      <div className="video-container">
        {src ? (
          <>
            {!playbackError ? (
              <video
                ref={videoRef}
                key={src}
                src={src}
                controls
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onError={() => setPlaybackError(true)}
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              />
            ) : (
              <div className="video-placeholder">
                <span className="icon">⚠️</span>
                <span>Browser cannot play this codec directly.</span>
              </div>
            )}
          </>
        ) : (
          <div className="video-placeholder">
            <span className="icon">📹</span>
            <span>{empty}</span>
          </div>
        )}
      </div>

      {src && playbackError && (
        <div className="download-fallback">
          <a href={src} download>⬇ Download video</a> to play in VLC or another player.
        </div>
      )}

      {src && !playbackError && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8 }}>
          <div className="download-fallback">
            <a href={src} download>⬇ Download Video</a>
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            High-precision hardware accelerated rendering
          </div>
        </div>
      )}
    </div>
  )
}
