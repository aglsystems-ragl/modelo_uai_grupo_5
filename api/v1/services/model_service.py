# api/v1/services/model_service.py
import os, sys, subprocess, json
from pathlib import Path

# Raíz del proyecto: .../backend
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
ART = ROOT / "artifacts"

def _abs_path(p: str) -> str:
    """Devuelve ruta absoluta: si p es relativa, se resuelve desde ROOT."""
    pp = Path(p)
    return str(pp if pp.is_absolute() else (ROOT / pp))

def train_via_script(data_path: str, target: str, threshold: float, test_size: float, sheet):
    """
    Lanza el entrenamiento con el script src/train.py usando el Python del venv.
    Devuelve el returncode del proceso (0 = OK).
    """
    script = SRC / "train.py"
    if not script.exists():
        print(f"[train_via_script] No existe el script: {script}")
        return -2

    cmd = [
        sys.executable,            # Python del venv activo
        str(script),               # .../backend/src/train.py
        "--data", _abs_path(data_path),
        "--target", str(target),
        "--threshold", str(threshold),
        "--test_size", str(test_size),
    ]
    if sheet is not None:
        cmd += ["--sheet", str(sheet)]

    print("[train_via_script] Ejecutando:", " ".join(cmd))
    try:
        # cwd=ROOT garantiza rutas relativas coherentes
        res = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
        return res.returncode
    except Exception as e:
        print("[train_via_script][ERROR]", e)
        return -1

def read_metrics():
    """Lee artifacts/metrics.json y lo devuelve como dict."""
    meta = ART / "metrics.json"
    with open(meta, "r", encoding="utf-8") as f:
        return json.load(f)
