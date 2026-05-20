# Model artifact security

Model files can be unsafe. Treat `.pth`, `.pt`, `.bin`, `.ckpt`, `.pkl`, and
`.pickle` artifacts as executable-code risk unless they come from a source you
trust.

Why: many PyTorch loading paths historically use pickle-compatible formats.
Pickle can execute code during deserialization. Prefer `.safetensors` when a
project supports it.

## Project policy

- The project does not automatically download lip sync weights.
- Backends require explicit local paths for repositories and checkpoints.
- Audit model files before use and keep hashes in your experiment notes.
- Prefer official repositories, official release pages, or Hugging Face repos
  from the model authors.
- Do not load random checkpoints from mirrors, comments, forums, or unknown
  file shares.

## Audit command

```powershell
.\.venv\Scripts\python.exe -m omnivoice_translator.model_security `
  E:\wav2lip\wav2lip_gan.pth
```

or, after reinstalling entry points:

```powershell
omnivoice-audit-models E:\wav2lip\wav2lip_gan.pth
```

The audit reports:

- file size
- SHA-256 hash
- extension
- risk label
- short explanation

This does not prove a file is safe. It makes provenance and risk explicit before
the file is loaded by a backend.
