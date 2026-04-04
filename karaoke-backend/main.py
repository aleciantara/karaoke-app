import re
import os
import time
import asyncio
from difflib import SequenceMatcher
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
from database import engine, Song, Session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization", "X-Requested-With"],
    max_age=600,
)

# â”€â”€ Module-level caches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_whisper_model = None
_align_models: dict = {}
_video_locks: dict[str, asyncio.Lock] = {}
_progress: dict[str, dict] = {}
_SYNC_ALGO = "v5"   # bump to invalidate cached timestamps when algorithm changes


def _set_progress(video_id: str, message: str) -> None:
    print(f"[progress] {message}")
    _progress[video_id] = {"message": message, "updated_at": time.time()}


def _get_video_lock(video_id: str) -> asyncio.Lock:
    if video_id not in _video_locks:
        _video_locks[video_id] = asyncio.Lock()
    return _video_locks[video_id]


def _get_whisper_model(device: str, compute_type: str):
    global _whisper_model
    if _whisper_model is None:
        import whisperx
        print(f"[whisperx] Loading transcription model (device={device})â€¦")
        _whisper_model = whisperx.load_model(
            "base", device, compute_type=compute_type,
            download_root=os.path.join(os.path.dirname(__file__), "models"),
        )
        print("[whisperx] Model ready.")
    return _whisper_model


def _get_align_model(language_code: str, device: str):
    if language_code not in _align_models:
        import whisperx
        print(f"[whisperx] Loading phoneme alignment model for '{language_code}'â€¦")
        model, metadata = whisperx.load_align_model(
            language_code=language_code, device=device
        )
        _align_models[language_code] = (model, metadata)
        print("[whisperx] Phoneme alignment model ready.")
    return _align_models[language_code]


class SyncRequest(BaseModel):
    youtube_url: str
    lyrics: str


def extract_video_id(url: str) -> str:
    match = re.search(r'(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})', url)
    if not match:
        raise ValueError("Could not find a valid YouTube video ID in the URL.")
    return match.group(1)


def clean_youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# â”€â”€ Forced alignment helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _match_lines_to_segments(
    user_lines: list[str],
    trans_segs: list[dict],
    audio_duration: float,
) -> dict[int, tuple[float, float]]:
    """
    Map each user lyrics line â†’ (start_sec, end_sec) using the transcription
    segments as a temporal scaffold. Uses Jaccard word overlap + greedy
    left-to-right ordering so the windows stay monotonic in time.
    """
    n_segs = len(trans_segs)
    if n_segs == 0:
        step = audio_duration / max(1, len(user_lines))
        return {li: (li * step, min(audio_duration, (li + 1) * step))
                for li in range(len(user_lines))}

    windows: dict[int, tuple[float, float]] = {}
    seg_ptr = 0
    same_count = 0

    for li, user_line in enumerate(user_lines):
        look_end = min(seg_ptr + 12, n_segs)
        best_score, best_si = -1.0, seg_ptr
        for si in range(seg_ptr, look_end):
            score = _jaccard(user_line, trans_segs[si].get("text", ""))
            if score > best_score:
                best_score, best_si = score, si

        windows[li] = (
            float(trans_segs[best_si]["start"]),
            float(trans_segs[best_si]["end"]),
        )

        if best_si > seg_ptr:
            seg_ptr = best_si
            same_count = 0
        else:
            same_count += 1
            # Nudge forward if too many lines pile onto the same segment
            if same_count >= 4 and seg_ptr < n_segs - 1:
                seg_ptr += 1
                same_count = 0

    # Enforce monotonic start times
    for li in range(1, len(user_lines)):
        prev_start, prev_end = windows[li - 1]
        this_start, this_end = windows[li]
        if this_start < prev_start:
            windows[li] = (prev_start, max(prev_end, this_end))

    return windows


def _interpolate_segment(
    seg_words: list[dict],
    fallback_start: float,
    fallback_end: float,
    line_idx: int,
) -> list[dict]:
    """Return word dicts for one segment, interpolating any without timestamps."""
    result = []
    n = len(seg_words)
    if n == 0:
        return result

    # Build anchor list: [(word_pos, start_sec, end_sec)]
    anchors = [(i, w["start"], w["end"])
               for i, w in enumerate(seg_words) if "start" in w and "end" in w]

    if not anchors:
        # Nothing aligned â€” distribute evenly in fallback window
        step = (fallback_end - fallback_start) / n
        for i, w in enumerate(seg_words):
            txt = w.get("word", "").strip()
            if txt:
                result.append({
                    "word": txt,
                    "start": int((fallback_start + i * step) * 1000),
                    "end":   int((fallback_start + (i + 1) * step) * 1000),
                    "line":  line_idx,
                })
        return result

    for i, w in enumerate(seg_words):
        txt = w.get("word", "").strip()
        if not txt:
            continue
        if "start" in w and "end" in w:
            result.append({
                "word":  txt,
                "start": int(w["start"] * 1000),
                "end":   int(w["end"] * 1000),
                "line":  line_idx,
            })
        else:
            # Interpolate between closest anchors
            prev_a = next((a for a in reversed(anchors) if a[0] < i), None)
            next_a = next((a for a in anchors if a[0] > i), None)
            if prev_a and next_a:
                gap = next_a[1] - prev_a[2]
                span = max(1, next_a[0] - prev_a[0])
                step = gap / span
                pos = i - prev_a[0]
                t0 = prev_a[2] + pos * step
                t1 = t0 + step
            elif prev_a:
                t0 = prev_a[2] + 0.12 * (i - prev_a[0])
                t1 = t0 + 0.12
            else:  # next_a only
                t0 = max(0.0, next_a[1] - 0.12 * (next_a[0] - i))
                t1 = t0 + 0.12
            result.append({
                "word":  txt,
                "start": int(max(0, t0) * 1000),
                "end":   int(max(0, t1) * 1000),
                "line":  line_idx,
            })

    return result


def forced_align_lyrics(
    lyrics: str,
    audio,              # numpy array at 16 kHz
    trans_segs: list[dict],
    align_model,
    align_metadata,
    device: str,
) -> list[dict]:
    """
    AI-powered lyric sync using whisperx's CTC phoneme forced aligner.

    Instead of matching text to text (which breaks when whisperx transcribes
    differently), we feed the user's own lyrics directly to the phoneme aligner
    within the time windows provided by the rough transcription. The aligner
    works at the audio-signal level â€” it doesn't care how whisperx spelt words.
    """
    import whisperx

    user_lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    if not user_lines:
        return []

    audio_duration = float(len(audio)) / 16000.0

    # Step 1 â”€ find time windows for each lyrics line from the transcription
    line_windows = _match_lines_to_segments(user_lines, trans_segs, audio_duration)

    # Step 2 â”€ build synthetic segments: user lyrics text + transcription windows
    synthetic: list[dict] = []
    for li, line_text in enumerate(user_lines):
        w_start, w_end = line_windows[li]
        # Small padding so aligner has a little context around the window
        seg_start = max(0.0, w_start - 0.25)
        seg_end   = min(audio_duration, w_end + 0.25)
        if seg_end - seg_start < 1.0:
            seg_end = min(audio_duration, seg_start + 3.0)
        synthetic.append({"start": seg_start, "end": seg_end, "text": line_text})

    # Step 3 â”€ phoneme forced alignment of user lyrics to audio
    print(f"[whisperx] Force-aligning {len(user_lines)} lyric lines to audioâ€¦")
    try:
        aligned = whisperx.align(
            synthetic, align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
    except Exception as exc:
        print(f"[whisperx] Forced alignment error: {exc} â€” using window interpolation fallback")
        # Fallback: distribute words evenly within their transcription windows
        result = []
        for li, line_text in enumerate(user_lines):
            words_in_line = re.findall(r"[\w''\-]+", line_text)
            if not words_in_line:
                continue
            ws, we = line_windows[li]
            step = max(0.1, (we - ws) / len(words_in_line))
            for wi, word in enumerate(words_in_line):
                result.append({
                    "word":  word,
                    "start": int((ws + wi * step) * 1000),
                    "end":   int((ws + (wi + 1) * step) * 1000),
                    "line":  li,
                })
        return result

    # Step 4 â”€ extract per-word timestamps, interpolating any gaps
    result = []
    for li, seg in enumerate(aligned.get("segments", [])):
        ws, we = line_windows.get(li, (synthetic[li]["start"], synthetic[li]["end"]))
        result.extend(_interpolate_segment(seg.get("words", []), ws, we, li))

    print(f"[whisperx] Done. {len(result)} words aligned.")
    return result


# â”€â”€ Endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# ── Lyric-to-transcription word mapping ───────────────────────────────────────

def _normalize(w: str) -> str:
    """Lowercase and strip non-word characters for fuzzy comparison."""
    return re.sub(r"[^\w]", "", w.lower())


def _distribute_lyrics_by_segments(
    lyric_pairs: list[tuple[str, int]],   # [(word, line_idx), …]
    trans_segs: list[dict],               # raw whisperx segments with start/end
    audio_duration: float,
) -> list[dict]:
    """
    Fallback when whisperx gives very few aligned words.
    Distributes lyrics evenly across the audio, using transcription segment
    boundaries as timing guides where available. Each lyrics line gets a
    proportional time slice.
    """
    if not lyric_pairs:
        return []

    # Group lyrics back into lines
    lines: list[list[tuple[str, int]]] = []
    cur_li = -1
    for word, li in lyric_pairs:
        if li != cur_li:
            lines.append([])
            cur_li = li
        lines[-1].append((word, li))

    n_lines = len(lines)

    # If we have transcription segments, use them as timing scaffolding
    if trans_segs and len(trans_segs) >= 2:
        # Merge nearby segments into vocal "blocks"
        blocks: list[tuple[float, float]] = []
        for seg in trans_segs:
            s, e = seg.get("start", 0), seg.get("end", 0)
            if blocks and s - blocks[-1][1] < 2.0:
                blocks[-1] = (blocks[-1][0], e)
            else:
                blocks.append((s, e))

        # Assign lyrics lines to blocks proportionally
        lines_per_block = max(1, n_lines / max(1, len(blocks)))
        windows: list[tuple[float, float]] = []
        block_idx = 0
        for i in range(n_lines):
            bi = min(int(i / lines_per_block), len(blocks) - 1)
            b_start, b_end = blocks[bi]
            local_lines = max(1, int(lines_per_block))
            local_idx   = i - int(bi * lines_per_block)
            step        = (b_end - b_start) / local_lines
            w_start     = b_start + local_idx * step
            w_end       = w_start + step
            windows.append((w_start, min(audio_duration, w_end)))
    else:
        # No useful segments — distribute evenly across entire audio
        step = audio_duration / n_lines
        windows = [(i * step, (i + 1) * step) for i in range(n_lines)]

    # Distribute words within each line's time window
    result: list[dict] = []
    for line_words, (ws, we) in zip(lines, windows):
        n_w = len(line_words)
        dur = we - ws
        word_dur = dur / max(1, n_w)
        # Each word gets a slightly shorter end to create a gap feel
        for wi, (word, li) in enumerate(line_words):
            s_ms = round((ws + wi * word_dur) * 1000)
            e_ms = round((ws + (wi + 1) * word_dur) * 1000)
            # Small gap between words (20ms) for natural feel
            if wi < n_w - 1:
                e_ms = max(s_ms + 50, e_ms - 20)
            result.append({"word": word, "start": s_ms, "end": e_ms, "line": li})

    # Enforce monotonic
    for i in range(1, len(result)):
        if result[i]["start"] < result[i - 1]["start"]:
            result[i]["start"] = result[i - 1]["end"]
            result[i]["end"]   = max(result[i]["end"], result[i]["start"] + 50)

    print(f"[lyrics] Distributed {len(result)} words across {n_lines} lines "
          f"(segment-based fallback).")
    return result


# ── Vocal-aware force alignment ───────────────────────────────────────────────

def _find_vocal_regions(
    trans_segs: list[dict],
    audio_duration: float,
    merge_gap: float = 3.0,
    min_dur: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Merge transcription segments into vocal regions.
    Adjacent segments within `merge_gap` seconds are merged.
    Regions shorter than `min_dur` are dropped.
    """
    if not trans_segs:
        return [(0.0, audio_duration)]

    sorted_segs = sorted(trans_segs, key=lambda s: s.get("start", 0))
    regions: list[tuple[float, float]] = []

    for seg in sorted_segs:
        s = float(seg.get("start", 0))
        e = float(seg.get("end", s + 0.5))
        if regions and s - regions[-1][1] < merge_gap:
            regions[-1] = (regions[-1][0], max(regions[-1][1], e))
        else:
            regions.append((s, e))

    regions = [(s, e) for s, e in regions if e - s >= min_dur]
    if not regions:
        return [(0.0, audio_duration)]
    return regions


def _assign_lines_to_vocal_windows(
    n_lines: int,
    regions: list[tuple[float, float]],
    audio_duration: float,
) -> list[tuple[float, float]]:
    """
    Assign lyrics lines to vocal regions proportionally.
    Longer regions get more lines.  Returns [(start_s, end_s), ...] per line.
    """
    if n_lines == 0:
        return []
    total_dur = sum(e - s for s, e in regions)
    if total_dur <= 0:
        step = audio_duration / n_lines
        return [(i * step, (i + 1) * step) for i in range(n_lines)]

    # How many lines per region (fractional), then round.
    raw_counts = [(e - s) / total_dur * n_lines for s, e in regions]
    int_counts = [max(0, round(c)) for c in raw_counts]

    # Fix rounding so total == n_lines
    while sum(int_counts) > n_lines:
        idx = int_counts.index(max(int_counts))
        int_counts[idx] -= 1
    while sum(int_counts) < n_lines:
        # add to region with most remaining capacity
        best, best_ratio = 0, -1.0
        for i, (s, e) in enumerate(regions):
            ratio = (e - s) / max(1, int_counts[i])
            if ratio > best_ratio:
                best_ratio, best = ratio, i
        int_counts[best] += 1

    windows: list[tuple[float, float]] = []
    for (r_start, r_end), n_r in zip(regions, int_counts):
        if n_r <= 0:
            continue
        step = (r_end - r_start) / n_r
        for i in range(n_r):
            ws = r_start + i * step
            windows.append((ws, min(audio_duration, ws + step)))
    return windows


def _force_align_user_lyrics(
    lyrics: str,
    audio,                  # numpy array 16 kHz
    trans_segs: list[dict],
    align_model,
    align_metadata,
    device: str,
    audio_duration: float,
) -> list[dict]:
    """
    Improved 4-stage pipeline:
    1. Get WORD-LEVEL timestamps from whisperx transcription (not just segments)
    2. Build tight per-line windows from those word timestamps
    3. Run CTC phoneme alignment within each tight window
    4. Interpolate any gaps, enforce monotonic
    """
    import whisperx

    user_lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
    if not user_lines:
        return []

    # Stage 1: Get word-level timestamps from the transcription
    print(f"[align] Getting word-level timestamps from transcription...")
    try:
        aligned_trans = whisperx.align(
            trans_segs, align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
        trans_words = _get_trans_word_list(aligned_trans.get("segments", []))
        print(f"[align] Got {len(trans_words)} transcription word timestamps")
    except Exception as e:
        print(f"[align] Transcription alignment failed: {e}, using segment boundaries")
        trans_words = []

    # Stage 2: Build tight line windows using trans word timestamps
    vocal_regions = _find_vocal_regions(trans_segs, audio_duration, merge_gap=2.0)
    line_windows = _build_tight_line_windows(
        user_lines, trans_words, vocal_regions, audio_duration
    )

    # Stage 3: CTC phoneme alignment per line in its tight window
    synthetic = []
    for li, line_text in enumerate(user_lines):
        ws, we = line_windows[li]
        # Tight pad — only 0.2s so we don't bleed into adjacent lines
        prev_end = line_windows[li - 1][1] if li > 0 else 0.0
        next_start = line_windows[li + 1][0] if li + 1 < len(line_windows) else audio_duration
        seg_start = max(prev_end + 0.05, ws - 0.2)
        seg_end   = min(next_start - 0.05, we + 0.2)
        if seg_end - seg_start < 0.8:
            mid = (ws + we) / 2
            seg_start = max(prev_end + 0.05, mid - 0.4)
            seg_end   = min(next_start - 0.05, mid + 0.4)
        synthetic.append({"start": seg_start, "end": seg_end, "text": line_text})

    print(f"[align] Force-aligning {len(user_lines)} lyrics lines to audio…")
    try:
        aligned = whisperx.align(
            synthetic, align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
        aligned_segs = aligned.get("segments", [])
    except Exception as exc:
        import traceback
        print(f"[align] CTC alignment error: {exc} — using window interpolation")
        traceback.print_exc()
        aligned_segs = []

    # Stage 4: Extract, interpolate, enforce monotonic
    result: list[dict] = []
    for li, line_text in enumerate(user_lines):
        ws, we = line_windows[li]
        seg_result = []
        if li < len(aligned_segs):
            seg_result = _interpolate_segment(
                aligned_segs[li].get("words", []), ws, we, li
            )
        if seg_result:
            result.extend(seg_result)
        else:
            # Fallback: distribute within the tight window
            words_in_line = re.findall(r"[\w''\u2019\-]+", line_text)
            if not words_in_line:
                continue
            step = max(0.15, (we - ws) / len(words_in_line))
            for wi, word in enumerate(words_in_line):
                result.append({
                    "word":  word,
                    "start": round((ws + wi * step) * 1000),
                    "end":   round(min((ws + (wi + 1) * step), we) * 1000),
                    "line":  li,
                })

    # Enforce monotonic + minimum word duration
    MIN_MS = 120
    for i in range(1, len(result)):
        if result[i]["start"] <= result[i - 1]["start"]:
            result[i]["start"] = result[i - 1]["end"] + 20
        result[i]["end"] = max(result[i]["end"], result[i]["start"] + MIN_MS)

    n_aligned = sum(1 for r in result if r["end"] - r["start"] > MIN_MS)
    print(f"[align] Force-aligned {len(result)} words ({n_aligned} with audio timestamps).")
    return result


def _get_trans_word_list(trans_segs: list[dict]) -> list[dict]:
    """Flatten all word-level timestamps from transcription segments."""
    words = []
    for seg in trans_segs:
        for w in seg.get("words", []):
            if "start" in w and "end" in w and w.get("word", "").strip():
                words.append({
                    "word":  w["word"].strip(),
                    "start": float(w["start"]),
                    "end":   float(w["end"]),
                })
    return words


def _build_tight_line_windows(
    user_lines: list[str],
    trans_words: list[dict],
    vocal_regions: list[tuple[float, float]],
    audio_duration: float,
) -> list[tuple[float, float]]:
    """
    Build tight time windows per lyrics line using transcription WORD timestamps.

    Strategy:
    - Match each lyrics line to the most phonetically similar cluster of
      transcription words using sliding window + Jaccard similarity
    - Windows are tight (just around the matched words) not whole segments
    - Falls back to vocal-region distribution if no word timestamps
    """
    if not trans_words:
        # No word timestamps — fall back to vocal region distribution
        return _assign_lines_to_vocal_windows(
            len(user_lines), vocal_regions, audio_duration
        )

    n_lines = len(user_lines)
    n_trans = len(trans_words)
    windows: list[tuple[float, float]] = []

    # Estimate words per line from the lyrics
    lyric_word_counts = [
        len(re.findall(r"[\w''\u2019\-]+", line)) for line in user_lines
    ]
    total_lyric_words = max(1, sum(lyric_word_counts))

    # Walk through transcription words proportionally
    trans_ptr = 0
    for li, line_text in enumerate(user_lines):
        # How many trans words to assign to this line proportionally
        n_lw = lyric_word_counts[li]
        expected_n_trans = max(1, round(n_lw / total_lyric_words * n_trans))

        # Lookahead window — check proportional slice + some slack
        look_start = trans_ptr
        look_end   = min(n_trans, trans_ptr + expected_n_trans + 6)

        # Find best match using Jaccard over sliding windows
        best_score, best_start, best_end = -1.0, trans_ptr, min(n_trans, trans_ptr + expected_n_trans)
        for start_i in range(look_start, min(look_end, n_trans)):
            for end_i in range(start_i + 1, min(start_i + expected_n_trans + 4, n_trans + 1)):
                chunk = " ".join(w["word"] for w in trans_words[start_i:end_i])
                score = _jaccard(line_text, chunk)
                # Penalise windows much larger than expected
                size_penalty = max(0, (end_i - start_i) - expected_n_trans) * 0.05
                score -= size_penalty
                if score > best_score:
                    best_score = score
                    best_start, best_end = start_i, end_i

        matched = trans_words[best_start:best_end]
        if matched:
            ws = matched[0]["start"]
            we = matched[-1]["end"]
            # Ensure minimum window of 0.5s
            if we - ws < 0.5:
                we = ws + 0.5
            windows.append((ws, we))
            trans_ptr = best_end
        else:
            # No match found — use a proportional fallback
            if windows:
                ws = windows[-1][1] + 0.1
            else:
                ws = vocal_regions[0][0] if vocal_regions else 0.0
            we = ws + max(0.5, n_lw * 0.25)
            windows.append((ws, we))

    # Enforce monotonic windows (never go backward)
    for i in range(1, len(windows)):
        prev_end = windows[i - 1][1]
        ws, we = windows[i]
        if ws < prev_end:
            ws = prev_end + 0.05
            we = max(we, ws + 0.5)
            windows[i] = (ws, we)

    return windows


def _map_lyrics_to_trans_words(
    lyrics: str,
    trans_words: list[dict],   # [{word, start (s), end (s)}, …]
    audio_duration: float,
    trans_segs: list[dict] | None = None,  # raw transcription segments for fallback
) -> list[dict]:
    """
    Map user lyrics words to transcribed word timestamps via SequenceMatcher.

    1. Parse lyrics into (word, line_idx) pairs.
    2. LCS-match normalised user words against normalised transcribed words.
    3. Copy transcription timestamps to matched user words.
    4. Interpolate timestamps for unmatched user words from nearest anchors.
    5. Enforce monotonically non-decreasing start times.
    Returns [{word, start (ms), end (ms), line}, …].
    """
    lyric_pairs: list[tuple[str, int]] = []
    for li, raw_line in enumerate(lyrics.splitlines()):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        for tok in re.findall(r"[\w''\u2019\-]+", raw_line):
            lyric_pairs.append((tok, li))

    if not lyric_pairs:
        return []

    # ── Fallback: if whisperx captured very few words (< 10% of lyrics),
    #    the SequenceMatcher mapping will be garbage. Use segment-based
    #    distribution instead ──────────────────────────────────────────────
    min_useful = max(5, len(lyric_pairs) * 0.10)
    if len(trans_words) < min_useful:
        print(f"[lyrics] Low transcription coverage ({len(trans_words)} words vs "
              f"{len(lyric_pairs)} lyrics). Using segment-based distribution.")
        return _distribute_lyrics_by_segments(
            lyric_pairs, trans_segs or [], audio_duration
        )

    user_norm  = [_normalize(w) for w, _ in lyric_pairs]
    trans_norm = [_normalize(tw["word"]) for tw in trans_words]

    sm = SequenceMatcher(None, user_norm, trans_norm, autojunk=False)
    assignments: dict[int, tuple[int, int]] = {}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                tw = trans_words[j1 + k]
                assignments[i1 + k] = (
                    round(tw["start"] * 1000),
                    round(tw["end"]   * 1000),
                )
        elif tag == "replace":
            t0  = trans_words[j1]["start"]
            t1  = trans_words[j2 - 1]["end"]
            n_u = i2 - i1
            stp = (t1 - t0) / max(1, n_u)
            for k in range(n_u):
                assignments[i1 + k] = (
                    round((t0 + k * stp) * 1000),
                    round((t0 + (k + 1) * stp) * 1000),
                )

    anchors = sorted((ui, s, e) for ui, (s, e) in assignments.items())
    result: list[dict] = []
    for ui, (word, li) in enumerate(lyric_pairs):
        if ui in assignments:
            s_ms, e_ms = assignments[ui]
        else:
            prev_a = next(
                ((a_ui, a_s, a_e) for a_ui, a_s, a_e in reversed(anchors) if a_ui < ui),
                None,
            )
            next_a = next(
                ((a_ui, a_s, a_e) for a_ui, a_s, a_e in anchors if a_ui > ui),
                None,
            )
            if prev_a and next_a:
                gap  = next_a[1] - prev_a[2]
                span = max(1, next_a[0] - prev_a[0])
                pos  = ui - prev_a[0]
                stp_ = gap / span
                s_ms = round(prev_a[2] + pos * stp_)
                e_ms = round(s_ms + stp_)
            elif prev_a:
                s_ms = prev_a[2] + 200 * (ui - prev_a[0])
                e_ms = s_ms + 200
            elif next_a:
                # Place unmatched words in the lead-up to the first anchor
                # Use at most 3 s before it, capped at 250 ms per word
                n_before = next_a[0] - ui
                max_lead = min(3000, next_a[1])
                space = min(250, max_lead / max(1, n_before + 1))
                s_ms = max(0, round(next_a[1] - n_before * space))
                e_ms = round(s_ms + space)
            else:
                s_ms, e_ms = 0, 200

        s_ms = max(0, s_ms)
        e_ms = max(s_ms + 50, e_ms)
        result.append({"word": word, "start": s_ms, "end": e_ms, "line": li})

    for i in range(1, len(result)):
        if result[i]["start"] < result[i - 1]["start"]:
            result[i]["start"] = result[i - 1]["end"]
            result[i]["end"]   = max(result[i]["end"], result[i]["start"] + 50)

    print(f"[lyrics] Mapped {len(result)} user words onto transcription timestamps.")
    return result


@app.get("/api/progress/{video_id}")
async def get_progress(video_id: str):
    p = _progress.get(video_id)
    if not p:
        return {"message": "", "updated_at": 0.0}
    return p


@app.post("/api/sync")
async def sync_song(req: SyncRequest):
    try:
        video_id = extract_video_id(req.youtube_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    async with _get_video_lock(video_id):

        # â”€â”€ Cache: return immediately if same video + same lyrics â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _lyrics_key = f"{_SYNC_ALGO}:{req.lyrics}"
        with Session(engine) as session:
            cached = session.get(Song, video_id)
            if cached and cached.words and cached.lyrics_raw == _lyrics_key:
                return {"video_id": video_id, "title": cached.title, "words": cached.words}

        _set_progress(video_id, "Starting...")

        # â”€â”€ Download audio â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        os.makedirs("tmp", exist_ok=True)
        audio_path = os.path.join(os.path.dirname(__file__), "tmp", f"{video_id}.wav")
        single_url = clean_youtube_url(video_id)
        title, thumbnail = video_id, ""

        if not os.path.exists(audio_path):
            def _dl_progress_hook(d: dict) -> None:
                if d["status"] == "downloading":
                    speed = d.get("speed") or 0
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    done  = d.get("downloaded_bytes") or 0
                    spd_str = f" @ {speed / 1_048_576:.1f} MB/s" if speed > 0 else ""
                    if total > 0:
                        pct = int(done / total * 100)
                        _set_progress(video_id, f"Downloading audio: {pct}%{spd_str}")
                    else:
                        _set_progress(video_id, f"Downloading audio{spd_str}...")
                elif d["status"] == "finished":
                    _set_progress(video_id, "Download complete, converting to WAV...")

            _set_progress(video_id, "Downloading audio from YouTube...")
            print(f"[yt-dlp] Downloading audio for {video_id}...")
            ydl_opts = {
                "format":  "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
                "outtmpl": os.path.join(os.path.dirname(__file__), "tmp", f"{video_id}.%(ext)s"),
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
                "noplaylist":  True,
                "no_overwrites": True,
                "quiet": False,
                "progress_hooks": [_dl_progress_hook],
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(single_url, download=True)
                    title     = info.get("title", video_id)
                    thumbnail = info.get("thumbnail", "")
                    print(f"[yt-dlp] Done: {title}")
            except Exception as exc:
                _progress.pop(video_id, None)
                raise HTTPException(status_code=500, detail=f"Failed to download audio: {exc}")
        else:
            _set_progress(video_id, "Audio already cached, loading model...")
            print(f"[yt-dlp] WAV already on disk, skipping download.")
            try:
                with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
                    info = ydl.extract_info(single_url, download=False)
                    title     = info.get("title", video_id)
                    thumbnail = info.get("thumbnail", "")
            except Exception:
                pass

        # â”€â”€ Transcribe â†’ forced-align user lyrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        words: list[dict] = []
        try:
            import torch
            import whisperx

            device       = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            batch_size   = 8 if device == "cuda" else 1

            audio = whisperx.load_audio(audio_path)

            # Step A: transcribe — gives us language + rough segment timestamps
            _set_progress(video_id, "Loading transcription model...")
            model = _get_whisper_model(device, compute_type)
            _set_progress(video_id, "Transcribing audio — this may take a few minutes...")
            print(f"[whisperx] Transcribing (batch_size={batch_size})â€¦")
            trans_result = model.transcribe(audio, batch_size=batch_size)
            trans_segs   = trans_result["segments"]
            language     = trans_result["language"]
            print(f"[whisperx] Transcription complete. Detected language: {language}")

            audio_dur = float(len(audio)) / 16000.0

            # Step B: load alignment model
            _set_progress(video_id, f"Aligning lyrics to audio (language: {language})...")
            align_model, align_metadata = _get_align_model(language, device)

            if req.lyrics.strip():
                # Step C: force-align user lyrics against the audio signal.
                _set_progress(video_id, "Building word-level timing anchors...")
                print(f"[pipeline] Starting force alignment of user lyrics...")
                words = _force_align_user_lyrics(
                    req.lyrics, audio, trans_segs,
                    align_model, align_metadata, device, audio_dur,
                )
                print(f"[pipeline] Force alignment returned {len(words)} words.")
                _set_progress(video_id, f"Generating timestamps ({len(words)} words)...")
            else:
                # No user lyrics — align the transcription itself
                print('[whisperx] Aligning transcription for word-level timestamps...')
                aligned = whisperx.align(
                    trans_segs, align_model, align_metadata, audio, device,
                    return_char_alignments=False,
                )
                words = [
                    {
                        'word':  w['word'].strip(),
                        'start': round(w['start'] * 1000),
                        'end':   round(w['end']   * 1000),
                        'line':  None,
                    }
                    for seg in aligned.get('segments', [])
                    for w in seg.get('words', [])
                    if 'start' in w and 'end' in w and w['word'].strip()
                ]

        except Exception as exc:
            _progress.pop(video_id, None)
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

        # â”€â”€ Cache result â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        with Session(engine) as session:
            session.merge(Song(
                id=video_id, title=title, youtube_url=req.youtube_url,
                thumbnail=thumbnail, words=words, lyrics_raw=_lyrics_key,
            ))
            session.commit()

        _progress.pop(video_id, None)
        return {"video_id": video_id, "title": title, "words": words}