# Lip sync backends

The video pipeline can use multiple lip sync backends behind the same contract:

```text
original video + translated timeline audio -> translated video
```

## Wav2Lip

Backend name: `wav2lip`

Good baseline and fast fallback. It is older and can look like a pasted mouth
patch on compressed or low-resolution video.

Required local files:

- Wav2Lip repository with `inference.py`
- Wav2Lip checkpoint, usually `.pth`

Example:

```powershell
.\.venv\Scripts\python.exe -m omnivoice_translator.video_cli `
  --lip-sync-backend wav2lip `
  --input downloads\input.mp4 `
  --output downloads\translated_wav2lip.mp4 `
  --source-lang por_Latn `
  --target-lang eng_Latn `
  --target-language English `
  --wav2lip-repo downloads\Wav2Lip `
  --wav2lip-checkpoint E:\wav2lip\wav2lip_gan.pth `
  --wav2lip-auto-box
```

## LatentSync

Backend name: `latentsync`

Preferred quality candidate. LatentSync uses audio-conditioned latent diffusion
and recent releases support 512x512 inference, which should reduce blur versus
older Wav2Lip-style output.

Required local files:

- LatentSync repository
- LatentSync checkpoint
- LatentSync UNet config YAML

Example:

```powershell
.\.venv\Scripts\python.exe -m omnivoice_translator.video_cli `
  --lip-sync-backend latentsync `
  --input downloads\input.mp4 `
  --output downloads\translated_latentsync.mp4 `
  --source-lang por_Latn `
  --target-lang eng_Latn `
  --target-language English `
  --latentsync-repo E:\models\LatentSync `
  --latentsync-checkpoint E:\models\LatentSync\checkpoints\latentsync_unet.pt `
  --latentsync-unet-config E:\models\LatentSync\configs\unet\stage2_512.yaml
```

## MuseTalk

Backend name: `musetalk`

Practical alternative focused on video dubbing. It usually needs a local model
directory and an inference config. This project generates the per-run inference
config automatically.

Example:

```powershell
.\.venv\Scripts\python.exe -m omnivoice_translator.video_cli `
  --lip-sync-backend musetalk `
  --input downloads\input.mp4 `
  --output downloads\translated_musetalk.mp4 `
  --source-lang por_Latn `
  --target-lang eng_Latn `
  --target-language English `
  --musetalk-repo E:\models\MuseTalk
```

## Quality Defaults

For more natural speech, use merged speech chunks and natural TTS duration:

```powershell
--segment-min-duration 2.4 `
--segment-max-duration 7.5 `
--segment-max-gap 0.45 `
--tts-duration-mode natural `
--timeline-fit stretch `
--max-stretch-ratio 1.25
```

`--timeline-fit stretch` avoids hard cutting generated speech when the duration
is close to the source slot. If the required stretch is too aggressive, the
pipeline falls back to pad/trim behavior.

In `--tts-duration-mode natural`, the CLI retries segments whose generated audio
is much shorter than the source slot. Tune this with `--tts-min-duration-ratio`
or disable it with `--no-tts-retry-short-segments`.

When no explicit `--ref-audio` is supplied, the video CLI now uses per-segment
source-audio clips as OmniVoice references. Use `--reference-audio-mode source`
only when you specifically want the older behavior of sending the whole extracted
source audio to every TTS call.
