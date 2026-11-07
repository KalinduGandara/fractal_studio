import json
import subprocess

import pytest

from fractal_studio.cli import main
from fractal_studio.scenes import JuliaScene
from fractal_studio.studio import RenderOptions, render_video


def test_missing_music_fails_fast(tmp_path):
    opts = RenderOptions(width=32, height=32, duration=0.2, out_dir=tmp_path,
                         music=tmp_path / "nope.wav", quiet=True)  # fmt: skip
    with pytest.raises(FileNotFoundError):
        render_video(JuliaScene(duration=0.2), opts)
    assert list(tmp_path.iterdir()) == []


def test_cli_missing_music_is_usage_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["julia", "--music", str(tmp_path / "nope.wav"), "--out", str(tmp_path)])
    assert e.value.code == 2


def test_export_pack_end_to_end(tmp_path):
    opts = RenderOptions(width=64, height=96, fps=10, duration=1.0, out_dir=tmp_path,
                         stem="t", quiet=True)  # fmt: skip
    render_video(JuliaScene(duration=1.0, max_iter=100), opts)
    for name in ("t.mp4", "t_thumb.png", "t_post.json", "t.json"):
        assert (tmp_path / name).is_file()
    post = json.loads((tmp_path / "t_post.json").read_text())
    assert len(post["title"]) <= 100
    assert all(not h.startswith("#") for h in post["hashtags"])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
         "-of", "csv=p=0", str(tmp_path / "t.mp4")],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert "h264" in probe and "aac,48000,2" in probe
