

// src/hooks/useKaraokeSync.ts
import { useEffect, useRef, useState, useMemo } from 'react'
import type { YouTubePlayer } from 'react-youtube'

export interface LyricWord {
  word: string
  start: number   // milliseconds
  end: number     // milliseconds
  line?: number   // source line index from original lyrics
}

export type WordState = 'upcoming' | 'active' | 'sung'

export interface DisplayWord {
  word: string
  state: WordState
  /** 0-1 progress through the word (for partial-fill effects, if desired) */
  progress: number
}

const LINE_GAP_MS  = 600
const POLL_MS      = 50
const LEAD_IN_MS   = 500   // show line this many ms before first word
const TAIL_MS      = 800   // keep line after last word ends

function groupIntoLines(words: LyricWord[]): LyricWord[][] {
  if (words.length === 0) return []

  // Use backend line indices if present
  if (words[0].line != null) {
    const map = new Map<number, LyricWord[]>()
    for (const w of words) {
      const l = w.line!
      if (!map.has(l)) map.set(l, [])
      map.get(l)!.push(w)
    }
    return Array.from(map.entries())
      .sort(([a], [b]) => a - b)
      .map(([, ws]) => ws)
  }

  // Fallback: split on timing gaps
  const lines: LyricWord[][] = []
  let line: LyricWord[] = []
  words.forEach((w, i) => {
    line.push(w)
    const gap = (words[i + 1]?.start ?? Infinity) - w.end
    if (gap > LINE_GAP_MS) {
      lines.push(line)
      line = []
    }
  })
  if (line.length > 0) lines.push(line)
  return lines
}

export function useKaraokeSync(
  playerRef: React.MutableRefObject<YouTubePlayer | null>,
  lyrics: LyricWord[],
  playerReady: boolean = false,
) {
  const [currentMs, setCurrentMs] = useState(0)
  const timerRef   = useRef<number>()
  const rafRef     = useRef<number>()
  const lastPollMs = useRef(0)
  const lastPollAt = useRef(0)

  useEffect(() => {
    if (!playerReady) return

    // Poll the player every POLL_MS
    timerRef.current = window.setInterval(async () => {
      const t = await playerRef.current?.getCurrentTime()
      if (t != null) {
        lastPollMs.current = Math.round(t * 1000)
        lastPollAt.current = performance.now()
      }
    }, POLL_MS)

    // Use rAF to interpolate between polls for smooth rendering
    const tick = () => {
      const elapsed = performance.now() - lastPollAt.current
      // Interpolate but cap at POLL_MS ahead to avoid over-shooting
      const interpolated = lastPollMs.current + Math.min(elapsed, POLL_MS)
      setCurrentMs(Math.round(interpolated))
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)

    return () => {
      clearInterval(timerRef.current)
      cancelAnimationFrame(rafRef.current!)
    }
  }, [playerRef, playerReady])

  const lines = useMemo(() => groupIntoLines(lyrics), [lyrics])

  // Find the active line index
  // A line is active if currentMs is within [firstWord.start - LEAD_IN, lastWord.end + TAIL]
  const lineIdx = useMemo(() => {
    return lines.findIndex(line =>
      line[0].start - LEAD_IN_MS <= currentMs &&
      line[line.length - 1].end + TAIL_MS > currentMs
    )
  }, [lines, currentMs])

  // Previous line — last line that has fully ended
  const prevLineIdx = useMemo(() => {
    if (lineIdx > 0) return lineIdx - 1
    // In a gap or before song starts — find last ended line
    for (let i = lines.length - 1; i >= 0; i--) {
      if (lines[i][lines[i].length - 1].end + TAIL_MS <= currentMs) return i
    }
    return -1
  }, [lines, lineIdx, currentMs])

  const currentLine = lineIdx >= 0 ? lines[lineIdx] : []
  const prevLine    = prevLineIdx >= 0
    ? lines[prevLineIdx].map(w => w.word).join(' ')
    : ''

  const currentWords: DisplayWord[] = useMemo(() => {
    return currentLine.map(w => {
      let state: WordState
      let progress: number
      if (currentMs >= w.end) {
        state = 'sung'; progress = 1
      } else if (currentMs >= w.start) {
        state = 'active'
        const dur = w.end - w.start
        progress = dur > 0 ? Math.min(1, (currentMs - w.start) / dur) : 1
      } else {
        state = 'upcoming'; progress = 0
      }
      return { word: w.word, state, progress }
    })
  }, [currentLine, currentMs])

  return { currentWords, prevLine, currentMs }
}

