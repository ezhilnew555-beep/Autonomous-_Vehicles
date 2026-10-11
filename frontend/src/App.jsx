/**
 * Eareyes 2k26 — Autonomous Road Safety Analyzer
 *
 * Streamlined 2-Page Portal & Comprehensive Analysis Dashboard:
 * - Page 1: Video Ingestion & Analysis Portal (upload, preview, real-time 8-stage progress tracker)
 * - Page 2: Comprehensive Road Analysis Dashboard (status hero, vehicle action, persistent object counts,
 *           synchronized video + 2D path corridor, safety timeline, and natural-language summary)
 */
import { useState } from 'react'
import UnifiedAnalysisDashboard from './components/UnifiedAnalysisDashboard.jsx'

export default function App() {
  // Navigation mode: exactly 2 pages as requested
  const [activeTab, setActiveTab] = useState('portal') // 'portal' | 'dashboard'

  return (
    <div className="app-wrapper">

      {/* ── Executive Autonomous AI Header ───────────────────────────────── */}
      <header className="app-header">
        <div className="header-top-bar">
          <div className="brand-badge-group">
            <span className="brand-logo-icon">🚗</span>
            <div className="brand-title-wrap">
              <span className="brand-title">EAREYES 2K26</span>
              <span className="brand-subtitle-tag">AUTONOMOUS ROAD SAFETY ANALYZER</span>
            </div>
            <div className="system-live-pill">
              <span className="live-pulse-dot" />
              <span>LIVE AI ENGINE</span>
            </div>
          </div>
          <div className="header-tech-stack">
            <span className="tech-tag">YOLO11 Perception</span>
            <span className="tech-tag">ByteTrack IDs</span>
            <span className="tech-tag">Kalman 2.0s</span>
            <span className="tech-tag">Quintic Path</span>
          </div>
        </div>
        <p className="app-header-desc">
          Autonomous Multi-Agent Trajectory Prediction, Dynamic Replanning &amp; Collision Avoidance for Mixed Indian Traffic
        </p>
      </header>

      {/* ── Exactly 2 Pages: Executive Navigation Bar ───────────────────── */}
      <nav className="cockpit-nav-bar">
        <button
          className={`cockpit-nav-btn ${activeTab === 'portal' ? 'active' : ''}`}
          onClick={() => setActiveTab('portal')}
        >
          <span>📤</span>
          <span>1. Video Ingestion Portal</span>
        </button>
        <button
          className={`cockpit-nav-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <span>📊</span>
          <span>2. Road Analysis Dashboard</span>
        </button>
      </nav>

      {/* ── 2-Page Portal & Comprehensive Analysis View ─────────────────── */}
      <UnifiedAnalysisDashboard
        activePage={activeTab}
        onNavigateToDashboard={() => setActiveTab('dashboard')}
        onNavigateToPortal={() => setActiveTab('portal')}
      />

    </div>
  )
}
