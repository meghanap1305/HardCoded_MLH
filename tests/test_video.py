"""Unit tests for video frame extraction helper (TASK-03)."""

from pathlib import Path

from hypelens.config import FIXTURES_DIR
from hypelens.data.video import VideoExtractionResult, extract_keyframes


def test_extract_keyframes_success(tmp_path):
    """Verify successful keyframe extraction from sample video clip."""
    clip_path = FIXTURES_DIR / "sample_clip.mp4"
    assert clip_path.exists(), "Sample clip fixture must exist."

    out_dir = tmp_path / "extracted_frames"
    result = extract_keyframes(clip_path, max_frames=3, output_dir=out_dir)

    assert isinstance(result, VideoExtractionResult)
    assert result.success is True
    assert result.total_frames > 0
    assert len(result.frames) <= 3
    assert len(result.frames) > 0

    first_frame = result.frames[0]
    assert first_frame.width > 0
    assert first_frame.height > 0
    assert first_frame.frame_rgb.ndim == 3
    assert first_frame.saved_path is not None
    assert Path(first_frame.saved_path).exists()


def test_extract_keyframes_missing_file():
    """Verify graceful handling when video path does not exist."""
    result = extract_keyframes("non_existent_video_path.mp4")
    assert result.success is False
    assert "does not exist" in result.error.lower()


def test_extract_keyframes_corrupt_file(tmp_path):
    """Verify graceful failure on non-video or corrupt file."""
    bad_file = tmp_path / "corrupt.mp4"
    bad_file.write_text("This is not a video file.")

    result = extract_keyframes(bad_file)
    assert result.success is False
    assert result.error is not None
