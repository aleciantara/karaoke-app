// src/components/KaraokeDisplay.tsx
import { useRef, useState, useEffect } from 'react'
import YouTube, { type YouTubeEvent, type YouTubePlayer } from 'react-youtube'
import { useKaraokeSync } from '../hooks/useKaraokeSync'
import { KaraokeWord } from './KaraokeWord'
import type { LyricWord } from '../hooks/useKaraokeSync'
import '../styles/karaoke.css'

interface Props {
  videoId: string
  lyrics: LyricWord[]
}

export function KaraokeDisplay({ videoId, lyrics }: Props) {
  const playerRef = useRef<YouTubePlayer | null>(null)
  const [playerReady, setPlayerReady] = useState(false)
  const { currentWords, prevLine } = useKaraokeSync(playerRef, lyrics, playerReady)
  const [fadePrev, setFadePrev] = useState('')

  // Smooth fade for previous line text changes
  useEffect(() => {
    if (prevLine !== fadePrev) {
      setFadePrev(prevLine)
    }
  }, [prevLine])

  return (
    <div className="karaoke-root">

      {/* YouTube player */}
      <YouTube
        videoId={videoId}
        onReady={(e: YouTubeEvent) => {
          playerRef.current = e.target
          setPlayerReady(true)
        }}
        className="karaoke-player"
        opts={{
          width: '100%',
          height: '100%',
          playerVars: { controls: 1, rel: 0 }
        }}
      />

      {/* Lyric overlay — sits over bottom of video */}
      <div className="karaoke-overlay">
        {/* Previous line — faded */}
        <p className="karaoke-prev-line" key={fadePrev}>
          {fadePrev}
        </p>

        {/* Current line — word by word */}
        <p className="karaoke-curr-line">
          {currentWords.map((w, i) => (
            <KaraokeWord key={i} word={w} />
          ))}
        </p>
      </div>

    </div>
  )
}