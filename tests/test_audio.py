from pathlib import Path

import numpy as np

from src.perturb_audio import mix_at_snr, write_dry_run_tone


def test_mix_at_snr_changes_signal():
    sig = np.ones(1000)
    noise = np.random.default_rng(0).standard_normal(1000)
    mixed = mix_at_snr(sig, noise, snr_db=10)
    assert not np.allclose(mixed, sig)


def test_dry_run_wav_written(tmp_path: Path):
    p = tmp_path / "tone.wav"
    write_dry_run_tone(p)
    assert p.stat().st_size > 0
