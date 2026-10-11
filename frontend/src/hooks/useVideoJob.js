/**
 * useVideoJob — manages the full lifecycle of a video processing job.
 *
 * After a videoId is provided, it:
 *   1. Fires POST /process/{videoId} to start background processing.
 *   2. Polls GET /status/{videoId} every POLL_INTERVAL_MS until done or error.
 *   3. Returns live status, progress (0-100), stats, and any error message.
 */
import { useEffect, useRef, useState } from 'react'
import { startProcessing, getStatus } from '../api/client'

const POLL_INTERVAL_MS = 2000

/**
 * @param {string|null} videoId  - The video ID to track. Pass null to idle.
 * @returns {{ status, progress, stats, error }}
 */
export function useVideoJob(videoId, mode = 'detection') {
  const [status,   setStatus]   = useState('pending')
  const [progress, setProgress] = useState(0)
  const [stats,    setStats]    = useState(null)
  const [error,    setError]    = useState(null)

  const intervalRef = useRef(null)
  const startedRef  = useRef(false)

  // Reset when videoId changes
  useEffect(() => {
    setStatus('pending')
    setProgress(0)
    setStats(null)
    setError(null)
    startedRef.current = false
  }, [videoId])

  useEffect(() => {
    if (!videoId) return

    const startAndPoll = async () => {
      // ── 1. Start processing (only once) ───────────────────────────────────
      // ── 1. Start processing (only once) ───────────────────────────────────
      if (!startedRef.current) {
        startedRef.current = true
        try {
          // Check if job is already done/processing
          const { data: initData } = await getStatus(videoId).catch(() => ({ data: null }))
          if (initData) {
            setStatus(initData.status)
            setProgress(initData.progress ?? 0)
            if (initData.stats) setStats(initData.stats)
            if (initData.status === 'done') return
          }
          await startProcessing(videoId, mode)
        } catch (err) {
          // If already processing or done (409), treat as ongoing
          if (err?.response?.status !== 409) {
            const msg = err?.response?.data?.detail || err.message || 'Failed to start processing.'
            setError(msg)
            setStatus('error')
            return
          }
        }
      }

      // ── 2. Poll status ─────────────────────────────────────────────────────
      intervalRef.current = setInterval(async () => {
        try {
          const { data } = await getStatus(videoId)
          setStatus(data.status)
          setProgress(data.progress ?? 0)

          if (data.stats) setStats(data.stats)

          if (data.status === 'done' || data.status === 'error') {
            clearInterval(intervalRef.current)
            if (data.status === 'error') {
              setError(data.message || 'Processing failed.')
            }
          }
        } catch (err) {
          // Network hiccup — don't stop polling yet, just log
          console.warn('Status poll failed:', err.message)
        }
      }, POLL_INTERVAL_MS)
    }

    startAndPoll()

    return () => clearInterval(intervalRef.current)
  }, [videoId])

  return { status, progress, stats, error }
}
