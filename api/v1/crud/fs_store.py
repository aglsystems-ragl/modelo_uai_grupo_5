
from pathlib import Path
import json
from joblib import load, dump

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT/"artifacts"; ART.mkdir(parents=True, exist_ok=True)

def save_model(pipe): dump(pipe, ART/"model.joblib")
def load_model(): return load(ART/"model.joblib")
def save_meta(d: dict): (ART/"metrics.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
def load_meta(): return json.loads((ART/"metrics.json").read_text(encoding="utf-8"))
def save_features(cols): (ART/"feature_order.json").write_text(json.dumps({"features": cols}, ensure_ascii=False, indent=2), encoding="utf-8")
def load_features(): return json.loads((ART/"feature_order.json").read_text(encoding="utf-8"))["features"]
