"""
Tests for meteor_setup.py's decision logic, using mocks so these run
deterministically in CI regardless of whether the test runner actually
has network access to nltk's data server (it usually won't).
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

from metrics.common import meteor_setup


def test_main_returns_zero_when_already_available():
    with patch.object(meteor_setup, "_check_available", return_value=True):
        with patch("sys.argv", ["aixpert-setup-meteor"]):
            assert meteor_setup.main() == 0


def test_main_detects_permission_warning_and_fails_with_guidance(capsys):
    with patch.object(meteor_setup, "_check_available", return_value=False):
        with patch.object(
            meteor_setup, "_attempt_download",
            return_value=(True, ["will not authorize the non-private download directory '/home/x'"]),
        ):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--nltk-data-dir" in captured.err
    assert "NLTK_DATA" in captured.err


def test_main_succeeds_on_clean_download(capsys):
    with patch.object(meteor_setup, "_check_available", side_effect=[False, True]):
        with patch.object(meteor_setup, "_attempt_download", return_value=(True, [])):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "installed and verified" in captured.out


def test_main_handles_download_exception_with_clear_message(capsys):
    with patch.object(meteor_setup, "_check_available", return_value=False):
        with patch.object(meteor_setup, "_attempt_download", side_effect=ConnectionError("no route to host")):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "network restriction" in captured.err
    assert "administrator" in captured.err


def test_main_handles_silent_failure_after_reported_success(capsys):
    """The real failure mode observed on a network-restricted sandbox:
    nltk.download() reports success (True) but the corpus is still not
    actually present afterward -- no exception raised, so this needs its
    own branch rather than relying on exception handling alone."""
    with patch.object(meteor_setup, "_check_available", side_effect=[False, False]):
        with patch.object(meteor_setup, "_attempt_download", return_value=(True, [])):
            with patch("sys.argv", ["aixpert-setup-meteor"]):
                exit_code = meteor_setup.main()
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "network restriction" in captured.err


def test_zip_packaged_corpora_are_available(tmp_path, monkeypatch):
    import zipfile
    import nltk
    from metrics.common.meteor import _wordnet_available

    corpora = tmp_path / "corpora"
    corpora.mkdir()
    for name in ("wordnet", "omw-1.4"):
        with zipfile.ZipFile(corpora / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}/README", "test corpus")
    monkeypatch.setattr(nltk.data, "path", [str(tmp_path)])
    assert meteor_setup._check_available()
    assert _wordnet_available()


def test_custom_download_directory_is_verified(tmp_path, monkeypatch):
    import nltk

    monkeypatch.setattr(nltk.data, "path", [])
    monkeypatch.setattr("sys.argv", ["aixpert-setup-meteor", "--nltk-data-dir", str(tmp_path)])

    def download(download_dir):
        for name in ("wordnet", "omw-1.4"):
            (tmp_path / "corpora" / name).mkdir(parents=True)
        return True, []

    monkeypatch.setattr(meteor_setup, "_attempt_download", download)
    assert meteor_setup.main() == 0
