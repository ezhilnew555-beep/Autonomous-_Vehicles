/**
 * Axios API client for the Eareyes 2k26 FastAPI backend.
 *
 * Base URL is resolved in priority order:
 *   1. VITE_API_BASE_URL env var (set for production)
 *   2. Empty string → Vite dev proxy handles /api/* calls
 */
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30_000,
})

// ── Video endpoints ────────────────────────────────────────────────────────────

/**
 * Upload a video file. Returns { video_id, filename, size_bytes, message }.
 * @param {File} file
 * @param {function(number):void} [onProgress] - Upload progress callback (0-100)
 */
export const uploadVideo = (file, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/api/v1/video/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded * 100) / evt.total))
      }
    },
  })
}

/**
 * Start background processing for a video.
 * @param {string} videoId
 * @param {string} [mode] - 'detection' (Phase 2) or 'full' (Phases 3-9)
 */
export const startProcessing = (videoId, mode = 'detection') =>
  api.post(`/api/v1/video/process/${videoId}${mode ? `?mode=${mode}` : ''}`)

/**
 * Poll job status. Returns StatusResponse schema.
 * @param {string} videoId
 */
export const getStatus = (videoId) =>
  api.get(`/api/v1/video/status/${videoId}`)

/**
 * Return the URL of the processed annotated video (for <video src>).
 * @param {string} videoId
 */
export const getResultUrl = (videoId) =>
  `${BASE_URL}/api/v1/video/result/${videoId}`

/**
 * Return the URL of the original uploaded video (for <video src>).
 * @param {string} videoId
 */
export const getOriginalUrl = (videoId) =>
  `${BASE_URL}/api/v1/video/original/${videoId}`

// ── Phase 9: Evaluation & Benchmark endpoints ──────────────────────────────────

/**
 * Retrieve list of the 5 Indian road testing scenarios.
 */
export const getScenarios = () =>
  api.get('/api/v1/evaluation/scenarios')

/**
 * Run comparative benchmark for a scenario (Baseline vs Adaptive).
 * @param {string} scenarioId
 */
export const runScenarioBenchmark = (scenarioId) =>
  api.post(`/api/v1/evaluation/run/${scenarioId}`)

/**
 * Retrieve full system evaluation report.
 */
export const getEvaluationSummary = () =>
  api.get('/api/v1/evaluation/summary')

// ── Phase 9A: Dashboard endpoints ──────────────────────────────────────────────

/**
 * Retrieve current vehicle state & decision overview.
 * @param {string} [videoId]
 */
export const getDashboardState = (videoId) =>
  api.get(`/api/v1/dashboard/state${videoId ? `?video_id=${videoId}` : ''}`)

/**
 * Retrieve real system performance metrics.
 * @param {string} [videoId]
 */
export const getDashboardMetrics = (videoId) =>
  api.get(`/api/v1/dashboard/metrics${videoId ? `?video_id=${videoId}` : ''}`)

/**
 * Retrieve state-change deduplicated event history.
 * @param {string} [videoId]
 * @param {number} [limit]
 */
export const getDashboardEvents = (videoId, limit = 50) =>
  api.get(`/api/v1/dashboard/events?limit=${limit}${videoId ? `&video_id=${videoId}` : ''}`)

/**
 * Retrieve unified environmental context.
 * @param {string} [videoId]
 */
export const getDashboardEnvironment = (videoId) =>
  api.get(`/api/v1/dashboard/environment${videoId ? `?video_id=${videoId}` : ''}`)

/**
 * Retrieve path planning status & 2D trajectory coordinates.
 * @param {string} [videoId]
 */
export const getDashboardPath = (videoId) =>
  api.get(`/api/v1/dashboard/path${videoId ? `?video_id=${videoId}` : ''}`)

// ── Phase 9B: Indian Road Scenario Testing endpoints ─────────────────────────

/**
 * Retrieve scenario testing catalog (5 Indian road situations).
 */
export const getScenarioCatalog = () =>
  api.get('/api/v1/scenarios')

/**
 * Retrieve detailed specification of a specific scenario.
 * @param {string} scenarioId
 */
export const getScenarioDetails = (scenarioId) =>
  api.get(`/api/v1/scenarios/${scenarioId}`)

/**
 * Run evaluation execution for a scenario.
 * @param {string} scenarioId
 * @param {string} [videoPath]
 */
export const runScenarioTest = (scenarioId, videoPath) =>
  api.post(`/api/v1/scenarios/${scenarioId}/run${videoPath ? `?video_path=${encodeURIComponent(videoPath)}` : ''}`)

/**
 * Run batch evaluation across all 5 scenarios.
 */
export const runAllScenariosTest = () =>
  api.post('/api/v1/scenarios/run-all')

/**
 * Retrieve scenario comparison table.
 */
export const getScenariosComparison = () =>
  api.get('/api/v1/scenarios/comparison')

/**
 * Retrieve summary metrics across all evaluated scenarios.
 */
export const getScenariosSummary = () =>
  api.get('/api/v1/scenarios/summary')

/**
 * Get export URL for scenario evaluation results.
 * @param {'json'|'csv'} format
 */
export const getScenariosExportUrl = (format = 'json') =>
  `${BASE_URL}/api/v1/scenarios/export?format=${format}`

// ── Unified Road Analysis endpoints ──────────────────────────────────────────

/**
 * Start the unified autonomous road analysis for an uploaded video.
 * @param {string} videoId
 */
export const startRoadAnalysis = (videoId) =>
  api.post(`/api/v1/analysis/start/${videoId}`)

/**
 * Poll analysis progress and current checklist step.
 * @param {string} analysisId
 */
export const getAnalysisStatus = (analysisId) =>
  api.get(`/api/v1/analysis/status/${analysisId}`)

/**
 * Retrieve the full unified road analysis result.
 * @param {string} analysisId
 */
export const getAnalysisResult = (analysisId) =>
  api.get(`/api/v1/analysis/${analysisId}`)

/**
 * Retrieve analysis result by video ID.
 * @param {string} videoId
 */
export const getAnalysisByVideo = (videoId) =>
  api.get(`/api/v1/analysis/video/${videoId}`)

/**
 * Retrieve past analysis runs.
 */
export const getAnalysisHistory = () =>
  api.get('/api/v1/analysis/history/all')

export default api

