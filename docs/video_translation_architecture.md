# Video translation v0 architecture

This v0 keeps the system simple and composable. The existing
`VoiceTranslationPipeline` remains responsible for speech-to-speech work. A new
video layer orchestrates media extraction, segmented translation, audio timeline
assembly, and Wav2Lip.

## Pipeline

```text
input video
-> extract mono WAV for ASR
-> ASR with timestamps
-> refine ASR boundaries with source-audio energy
-> merge short adjacent speech segments into dubbing chunks
-> translate each chunk
-> adapt/clean each translated chunk for spoken dubbing
-> synthesize each translated chunk with OmniVoice
-> trim/fade/RMS/peak condition each generated segment
-> place generated segments back on the original timeline
-> selected lip sync backend(original video, translated timeline audio)
-> translated video
```

## Design choices

- When `--ref-audio` is not provided, the extracted `source_audio.wav` is cut
  into per-segment reference clips for OmniVoice. The source text for that chunk
  is passed as `ref_text` unless the user supplied an explicit reference text.
  `--reference-audio-mode source` keeps the older full-source reference behavior.
  The original video is kept intact for the lip sync backend.
- Segmentation is based on ASR timestamps. If the ASR backend does not return
  chunks, the pipeline falls back to one segment covering the source audio.
- Segment boundaries are refined with `librosa.effects.split` over the extracted
  source WAV. The adjustment is intentionally bounded so ASR semantics still
  drive the chunking.
- Each translated segment is synthesized separately. The requested OmniVoice
  duration is the source segment duration when `--tts-duration-mode segment` is
  used. In `natural` mode, segments that come back much shorter than their video
  slot are retried once with explicit duration so the final timeline does not
  leave long silent tails while the mouth is still moving.
- The final translated WAV preserves the original video timeline by placing each
  segment at its source start time. It can either pad/trim or use moderate
  time-stretching. Overlapping segments are trimmed instead of mixed to avoid
  doubled speech.
- Generated segment WAVs are conditioned before timeline assembly: optional
  silence trim, short fades, RMS normalization, and peak limiting.
- Lip sync is pluggable. Wav2Lip, LatentSync, and MuseTalk are external
  backends. This project builds and validates commands, but does not vendor
  third-party repositories or checkpoints.
- Output resolution is validated against input resolution by default.

## Module contracts

- `segments.py`: normalizes ASR results into `SpeechSegment` objects.
- `audio_segments.py`: pads/trims segment audio and renders a full timeline WAV.
- `audio_quality.py`: audits generated audio with librosa-based signal,
  spectral, and segment-level prosody checks.
- `media.py`: wraps `ffmpeg` and `ffprobe`.
- `segment_planner.py`: merges ASR segments into dubbing chunks.
- `segment_refinement.py`: refines ASR segment edges from source-audio energy.
- `text_refinement.py`: cleans/adapts translated text before TTS.
- `lipsync.py`: wraps Wav2Lip, LatentSync, and MuseTalk inference.
- `lipsync_box.py`: estimates a lower-face Wav2Lip box automatically.
- `model_security.py`: audits local model artifacts and flags pickle-risk
  checkpoint formats.
- `video_pipeline.py`: orchestrates the full video translation flow.
- `video_cli.py`: exposes the v0 as `omnivoice-translate-video`.

## Known v0 limits

- Only one active face is expected. Multi-speaker face selection is out of scope.
- Segment fitting is simple pad/trim or bounded time-stretching, not full
  prosody transfer.
- Lip sync quality depends on face visibility, pose, resolution, backend, and
  checkpoint.
- Real execution requires local `ffmpeg`, `ffprobe`, a Wav2Lip checkout, and a
  backend-specific checkout/checkpoint.
