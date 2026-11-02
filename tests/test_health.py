from pathlib import Path


def test_yolo_model_is_present() -> None:
    assert Path("models/yolo/best.pt").is_file()
