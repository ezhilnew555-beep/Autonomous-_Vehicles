/** StatusBadge — coloured pill showing the current job phase. */
export default function StatusBadge({ status }) {
  const map = {
    idle:       { cls: 'badge-pending',    dot: '○', label: 'Idle' },
    uploading:  { cls: 'badge-uploading',  dot: '●', label: 'Uploading', pulse: true },
    pending:    { cls: 'badge-pending',    dot: '○', label: 'Pending' },
    processing: { cls: 'badge-processing', dot: '●', label: 'Processing', pulse: true },
    done:       { cls: 'badge-done',       dot: '✓', label: 'Done' },
    error:      { cls: 'badge-error',      dot: '✕', label: 'Error' },
  }

  const cfg = map[status] ?? map.idle

  return (
    <span className={`badge ${cfg.cls}`}>
      <span className={cfg.pulse ? 'pulse' : ''}>{cfg.dot}</span>
      {cfg.label}
    </span>
  )
}
