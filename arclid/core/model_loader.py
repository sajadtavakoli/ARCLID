from ultralytics import YOLO
from pathlib import Path
import urllib.request

HIFI_5K_URL = "https://drive.google.com/file/d/17jDHlwMyQHtjE269QVe5yaGd9tkY58cX/view?usp=sharing"

HIFI_50k_URL = "https://drive.google.com/file/d/1nKgiPE3rD7ag6YULb1OO-MN-7cF4Qlpj/view?usp=sharing"


def load_model(model_path_hifi_5k: str | None = None, model_path_hifi_50k: str | None = None):
    if model_path_hifi_5k is None:
        model_path_hifi_5k = Path(__file__).resolve().parent.parent / "weights" / "model_hifi_5k.pt"
    model_path_hifi_5k = Path(model_path_hifi_5k)

    if model_path_hifi_50k is None:
        model_path_hifi_50k = Path(__file__).resolve().parent.parent / "weights" / "model_hifi_50k.pt"
    model_path_hifi_50k = Path(model_path_hifi_50k)

    if not model_path_hifi_5k.exists():
        model_path_hifi_5k.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ARCLID] Downloading hifi model 1 ...")
        urllib.request.urlretrieve(HIFI_5K_URL, model_path_hifi_5k)
    model_hifi_5k = YOLO(str(model_path_hifi_5k))

    if not model_path_hifi_50k.exists():
        model_path_hifi_50k.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ARCLID] Downloading hifi model 2 ...")
        urllib.request.urlretrieve(HIFI_50k_URL, model_path_hifi_50k)
    model_hifi_50k = YOLO(str(model_path_hifi_50k))

    return model_hifi_5k, model_hifi_50k
