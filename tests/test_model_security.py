import hashlib

import pytest

from omnivoice_translator.model_security import audit_model_artifact


def test_audit_model_artifact_flags_pickle_risk(tmp_path):
    artifact = tmp_path / "model.pth"
    artifact.write_bytes(b"weights")

    audit = audit_model_artifact(artifact)

    assert audit.risk == "high"
    assert audit.sha256 == hashlib.sha256(b"weights").hexdigest()
    assert "pickle" in audit.message


def test_audit_model_artifact_marks_safetensors_lower_risk(tmp_path):
    artifact = tmp_path / "model.safetensors"
    artifact.write_bytes(b"weights")

    audit = audit_model_artifact(artifact)

    assert audit.risk == "lower"
    assert "avoids Python pickle" in audit.message


def test_audit_model_artifact_requires_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_model_artifact(tmp_path / "missing.pth")
