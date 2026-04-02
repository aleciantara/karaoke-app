// src/components/KaraokeWord.tsx
import { memo, useRef, useEffect, useState } from 'react'
import type { DisplayWord } from '../hooks/useKaraokeSync'
import '../styles/karaoke.css'

interface Props {
  word: DisplayWord
}

// memo() prevents re-rendering words that haven't changed state
export const KaraokeWord = memo(({ word }: Props) => {
  const prevState = useRef(word.state)
  const [animKey, setAnimKey] = useState(0)

  useEffect(() => {
    // Bump the key to retrigger CSS animation when word becomes active
    if (word.state === 'active' && prevState.current !== 'active') {
      setAnimKey(k => k + 1)
    }
    prevState.current = word.state
  }, [word.state])

  return (
    <span key={animKey} className={`karaoke-word karaoke-word--${word.state}`}>
      {word.word}{' '}
    </span>
  )
})