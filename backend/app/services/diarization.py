"""Speaker diarization service for Meetolog.

Runs a global speaker-diarization pipeline over the full audio recording
to produce a timeline of speaker turns, then aligns that timeline with
the timestamped segments produced by Whisper chunked transcription.

Design constraints
------------------
* The pyannote model is loaded, used, and **explicitly deleted** before
  the Whisper model runs, so only one large model is resident at a time.
* All blocking inference is offloaded to the thread-pool via
  ``asyncio.to_thread`` so the event loop stays responsive.
* ``gc.collect()`` + optional CUDA cache clearing run after every heavy
  operation to keep peak RSS within the 2 GB Fargate Spot budget.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiarizedSegment:
    """A single speaker turn from the diarization timeline."""
    start: float
    end: float
    speaker: str


class SpeakerDiarizer:
    """Thin wrapper around the pyannote speaker-diarization pipeline.

    Parameters
    ----------
    hf_token:
        HuggingFace access token with read access to the gated
        ``pyannote/speaker-diarization-3.1`` model.
    """

    def __init__(self, hf_token: str) -> None:
        self._hf_token = hf_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def diarize(self, audio_path: Path) -> list[DiarizedSegment]:
        """Run diarization on *audio_path* and return the speaker timeline.

        The pipeline is loaded inside the worker thread, used once, and
        then deleted with an explicit ``gc.collect()`` to reclaim memory
        before the Whisper model is loaded.
        """
        return await asyncio.to_thread(self._run_pipeline, audio_path)

    @staticmethod
    def assign_speakers(
        whisper_segments: list,
        diarization: list[DiarizedSegment],
    ) -> str:
        """Merge Whisper segments with the diarization timeline.

        Each Whisper segment is attributed to the speaker whose
        diarization turn overlaps the segment's midpoint.  Consecutive
        segments by the same speaker are merged into a single paragraph.

        Parameters
        ----------
        whisper_segments:
            ``TranscriptSegment`` instances produced by
            ``WhisperTranscriber.transcribe_with_segments``.
        diarization:
            Global speaker timeline from :meth:`diarize`.

        Returns
        -------
        str
            Labelled transcript, e.g.::

                Speaker 1: Hello everyone.
                Speaker 2: Thanks for joining.
        """
        if not diarization or not whisper_segments:
            return " ".join(seg.text for seg in whisper_segments)

        # Build normalised speaker labels (SPEAKER_00 → Speaker 1)
        raw_speakers = sorted({s.speaker for s in diarization})
        speaker_map = {raw: f"Speaker {i + 1}" for i, raw in enumerate(raw_speakers)}

        lines: list[str] = []
        current_speaker: str | None = None
        current_parts: list[str] = []

        for ws in whisper_segments:
            midpoint = (ws.start + ws.end) / 2.0
            raw = _find_speaker_at(midpoint, diarization)
            label = speaker_map.get(raw, "Unknown Speaker")

            if label != current_speaker:
                if current_speaker is not None and current_parts:
                    lines.append(f"{current_speaker}: {' '.join(current_parts)}")
                current_speaker = label
                current_parts = [ws.text.strip()]
            else:
                current_parts.append(ws.text.strip())

        # Flush the last accumulated turn
        if current_speaker is not None and current_parts:
            lines.append(f"{current_speaker}: {' '.join(current_parts)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_pipeline(self, audio_path: Path) -> list[DiarizedSegment]:
        """Blocking helper — executed inside ``asyncio.to_thread``."""
        import torch
        from pyannote.audio import Pipeline

        logger.info("Loading pyannote speaker-diarization-3.1 pipeline")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self._hf_token,
        )
        pipeline.to(torch.device("cpu"))

        logger.info("Running diarization on %s", audio_path.name)
        diarization_result = pipeline(str(audio_path))

        segments: list[DiarizedSegment] = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
            ))

        num_speakers = len({s.speaker for s in segments})
        logger.info(
            "Diarization complete: %d turns, %d unique speakers",
            len(segments), num_speakers,
        )

        # Aggressive memory cleanup — free the pipeline before Whisper loads
        del diarization_result
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segments


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _find_speaker_at(
    time_point: float,
    segments: list[DiarizedSegment],
) -> str:
    """Return the speaker active at *time_point* in the timeline.

    If *time_point* falls inside a diarization turn, that speaker is
    returned.  Otherwise the speaker of the nearest turn boundary is
    returned (useful for short Whisper segments that straddle a gap).
    """
    best_speaker: str | None = None
    best_distance = float("inf")

    for seg in segments:
        if seg.start <= time_point <= seg.end:
            return seg.speaker
        distance = min(abs(seg.start - time_point), abs(seg.end - time_point))
        if distance < best_distance:
            best_distance = distance
            best_speaker = seg.speaker

    return best_speaker or "UNKNOWN"
