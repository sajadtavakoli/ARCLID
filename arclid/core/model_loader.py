from ultralytics import YOLO
from pathlib import Path
import urllib.request


HIFI_5K_URL = "https://drive.usercontent.google.com/download?id=17jDHlwMyQHtjE269QVe5yaGd9tkY58cX&export=download&authuser=0&confirm=t&uuid=7733230b-b097-40ba-93cf-3cdd54dee221&at=AGN2oQ2mPF99EDpCtjV3YHwl4UDR%3A1773260064723"

HIFI_50K_URL = "https://drive.usercontent.google.com/download?id=1nKgiPE3rD7ag6YULb1OO-MN-7cF4Qlpj&export=download&authuser=0&confirm=t&uuid=b237b464-fc81-4324-9bd4-3952aae83515&at=AGN2oQ3SO-K09rtFb12QFTMn5XXU%3A1773260279916"

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
        urllib.request.urlretrieve(HIFI_50K_URL, model_path_hifi_50k)
    model_hifi_50k = YOLO(str(model_path_hifi_50k))

    return model_hifi_5k, model_hifi_50k
