import os, glob, subprocess, sys

def test_headless_run_generates_frames():
    os.makedirs("artifacts", exist_ok=True)
    subprocess.check_call([sys.executable, "tools/run_headless.py"])  # raises if fails
    frames = sorted(glob.glob("artifacts/frame_*.png"))
    assert len(frames) >= 8, "Expected at least 8 frames to be produced"
