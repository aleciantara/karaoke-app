import { useState, useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import axios, { type AxiosError } from 'axios'
import { KaraokeDisplay } from './components/KaraokeDisplay'
import type { LyricWord } from './hooks/useKaraokeSync'
import './App.css'

const API_BASE = 'http://localhost:8000'

interface SyncResponse {
  video_id: string
  title: string
  words: LyricWord[]
}

interface ApiError {
  detail?: string
}

const LOADING_STEPS = [
  { key: 'download',    label: 'Downloading audio from YouTube...' },
  { key: 'model',       label: 'Loading transcription model...' },
  { key: 'transcribe',  label: 'Transcribing audio...' },
  { key: 'align',       label: 'Aligning lyrics to audio...' },
  { key: 'finalise',    label: 'Generating timestamps...' },
]

function matchStep(message: string): number {
  const m = message.toLowerCase()
  if (m.includes('download') || m.includes('converting') || m.includes('cached'))  return 0
  if (m.includes('loading transcription') || m.includes('model ready') || m.includes('starting')) return 1
  if (m.includes('transcrib') || m.includes('transcription complete'))  return 2
  if (m.includes('align') || m.includes('force') || m.includes('phoneme') || m.includes('vocal') || m.includes('anchor')) return 3
  if (m.includes('generat') || m.includes('timestamp') || m.includes('words')) return 4
  return 0
}

function useElapsed(active: boolean) {
  const [seconds, setSeconds] = useState(0)
  const start = useRef(Date.now())
  useEffect(() => {
    if (!active) { setSeconds(0); return }
    start.current = Date.now()
    const id = window.setInterval(() => {
      setSeconds(Math.floor((Date.now() - start.current) / 1000))
    }, 1000)
    return () => clearInterval(id)
  }, [active])
  return seconds
}

function LoadingProgress({ videoId }: { videoId: string }) {
  const [liveMsg, setLiveMsg] = useState('')
  const [step, setStep] = useState(0)
  const elapsed = useElapsed(true)

  // Poll the backend progress endpoint every 2 s
  useEffect(() => {
    if (!videoId) return
    const id = window.setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/progress/${videoId}`)
        if (!res.ok) return
        const data: { message: string; updated_at: number } = await res.json()
        if (data.message) {
          setLiveMsg(data.message)
          setStep(matchStep(data.message))
        }
      } catch {
        // network hiccup — ignore silently
      }
    }, 800)
    return () => clearInterval(id)
  }, [videoId])

  const mins = Math.floor(elapsed / 60)
  const secs = elapsed % 60
  const elapsedStr = mins > 0
    ? `${mins}m ${secs.toString().padStart(2, '0')}s`
    : `${secs}s`

  return (
    <div className="loading-wrap">
      <div className="loading-spinner" />

      <div className="loading-elapsed">{elapsedStr}</div>

      {liveMsg && (
        <div className="loading-live-msg">{liveMsg}</div>
      )}

      <ul className="loading-steps">
        {LOADING_STEPS.map((s, i) => (
          <li
            key={s.key}
            className={`loading-step ${
              i < step ? 'done' : i === step ? 'active' : 'pending'
            }`}
          >
            <span className="step-icon">
              {i < step ? '✓' : i === step ? '◉' : '○'}
            </span>
            {s.label}
          </li>
        ))}
      </ul>
    </div>
  )
}

function App() {
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [lyrics, setLyrics] = useState('')
  const [result, setResult] = useState<SyncResponse | null>(null)

  const syncMutation = useMutation({
    mutationFn: () => {
      const controller = new AbortController()
      const promise = axios
        .post<SyncResponse>(
          `${API_BASE}/api/sync`,
          { youtube_url: youtubeUrl, lyrics },
          { signal: controller.signal, timeout: 0 },
        )
        .then(r => r.data)
      // Attach cancel so React Query can abort on unmount
      ;(promise as any).cancel = () => controller.abort()
      return promise
    },
    retry: 0,
    onSuccess: data => setResult(data),
  })

  const errorMsg =
    (syncMutation.error as AxiosError<ApiError>)?.response?.data?.detail ??
    (syncMutation.error as Error)?.message ??
    null

  // ── Playback view ──────────────────────────────────────────────────────
  if (result) {
    return (
      <div className="app app--player">
        <header className="player-header">
          <span className="player-title">🎤 {result.title}</span>
          <button
            className="btn-back"
            onClick={() => {
              setResult(null)
              syncMutation.reset()
            }}
          >
            ← New Song
          </button>
        </header>
        <main className="player-body">
          <KaraokeDisplay videoId={result.video_id} lyrics={result.words} />
        </main>
      </div>
    )
  }

  // ── Input form ─────────────────────────────────────────────────────────
  return (
    <div className="app app--form">
      <div className="form-card">
        <h1 className="form-title">🎤 Karaoke Maker</h1>
        <p className="form-sub">
          Paste a YouTube link and song lyrics to create your karaoke video
          with automatically synced, word-by-word highlighted lyrics.
        </p>

        {syncMutation.isPending ? (
          <LoadingProgress videoId={youtubeUrl.match(/(?:v=|youtu\.be\/)([A-Za-z0-9_-]{11})/)?.[1] ?? ''} />
        ) : (
          <form
            className="form"
            onSubmit={e => {
              e.preventDefault()
              syncMutation.mutate()
            }}
          >
            <div className="form-group">
              <label className="form-label">YouTube URL</label>
              <input
                className="form-input"
                type="url"
                placeholder="https://www.youtube.com/watch?v=…"
                value={youtubeUrl}
                onChange={e => setYoutubeUrl(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Lyrics</label>
              <textarea
                className="form-textarea"
                placeholder={
                  'Paste the full song lyrics here…\n\nOne line per lyric line works best.'
                }
                value={lyrics}
                onChange={e => setLyrics(e.target.value)}
                rows={12}
                required
              />
            </div>

            {errorMsg && <p className="form-error">{errorMsg}</p>}

            <button
              className="form-submit"
              type="submit"
              disabled={!youtubeUrl.trim() || !lyrics.trim()}
            >
              Generate Karaoke
            </button>
          </form>
        )}
      </div>
    </div>
  )
}

export default App
