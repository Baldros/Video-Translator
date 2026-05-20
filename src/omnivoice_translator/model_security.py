from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PICKLE_RISK_EXTENSIONS = {".pth", ".pt", ".bin", ".ckpt", ".pkl", ".pickle"}
SAFER_WEIGHT_EXTENSIONS = {".safetensors"}


@dataclass(frozen=True)
class ModelArtifactAudit:
    path: str
    size_bytes: int
    sha256: str
    extension: str
    risk: str
    message: str


def audit_model_artifact(path: str | Path) -> ModelArtifactAudit:
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact}")
    if not artifact.is_file():
        raise ValueError(f"Model artifact must be a file: {artifact}")

    extension = artifact.suffix.lower()
    if extension in SAFER_WEIGHT_EXTENSIONS:
        risk = "lower"
        message = "safetensors avoids Python pickle execution during weight load."
    elif extension in PICKLE_RISK_EXTENSIONS:
        risk = "high"
        message = (
            "This extension is commonly loaded through pickle-capable PyTorch "
            "paths. Treat it as executable code unless the source is trusted."
        )
    else:
        risk = "unknown"
        message = "Unknown model artifact format. Verify source and loader behavior."

    return ModelArtifactAudit(
        path=str(artifact),
        size_bytes=artifact.stat().st_size,
        sha256=_sha256_file(artifact),
        extension=extension,
        risk=risk,
        message=message,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit local model artifacts before loading them."
    )
    parser.add_argument("paths", nargs="+", help="Model artifact paths.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audits = [asdict(audit_model_artifact(path)) for path in args.paths]
    print(json.dumps(audits, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
