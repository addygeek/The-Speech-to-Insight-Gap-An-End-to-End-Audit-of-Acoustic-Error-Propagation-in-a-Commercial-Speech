from pathlib import Path

from src.manifest import Manifest, Stage, Status


def test_manifest_resume(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    mf = Manifest(path)
    mf.record("u1", Stage.TTS, Status.SUCCESS)
    mf.save()
    mf2 = Manifest(path)
    assert mf2.should_skip(True, "u1", Stage.TTS)
    assert not mf2.should_skip(True, "u1", Stage.LISTEN, "clean")
