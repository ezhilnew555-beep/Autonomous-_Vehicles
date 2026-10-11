/**
 * VideoUploader — drag-and-drop / click file picker.
 *
 * Props:
 *   onFile    {function(File):void}   called when user selects a valid video file
 *   disabled  {boolean}               disable interaction during processing
 */
import { useRef, useState } from 'react'

const ALLOWED = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm']
const ALLOWED_EXT = '.mp4, .avi, .mov, .mkv, .webm'

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function VideoUploader({ onFile, disabled = false }) {
  const inputRef  = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [selected, setSelected] = useState(null)
  const [fileError, setFileError] = useState(null)

  const handleFile = (file) => {
    if (!file) return
    // Validate by MIME or extension fallback
    const ext = file.name.split('.').pop().toLowerCase()
    const validExts = ['mp4','avi','mov','mkv','webm']
    if (!ALLOWED.includes(file.type) && !validExts.includes(ext)) {
      setFileError(`Unsupported format. Allowed: ${ALLOWED_EXT}`)
      return
    }
    setFileError(null)
    setSelected(file)
    onFile(file)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (disabled) return
    const file = e.dataTransfer.files?.[0]
    handleFile(file)
  }

  const onClick = () => {
    if (!disabled) inputRef.current?.click()
  }

  return (
    <div>
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
        onClick={onClick}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && onClick()}
        style={{ opacity: disabled ? 0.55 : 1, pointerEvents: disabled ? 'none' : 'auto' }}
      >
        <div className="icon">🎥</div>
        <div>Drag &amp; drop a road video here</div>
        <div className="hint">or click to browse — {ALLOWED_EXT}</div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_EXT}
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {fileError && (
        <div className="error-box" style={{ marginTop: 10 }}>{fileError}</div>
      )}

      {selected && !fileError && (
        <div className="file-info">
          <strong>{selected.name}</strong>
          <br />
          Size: {formatBytes(selected.size)} &nbsp;|&nbsp; Type: {selected.type || 'video'}
        </div>
      )}
    </div>
  )
}
