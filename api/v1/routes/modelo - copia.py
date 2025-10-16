# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from ..schemas.io import TrainRequest, PredictRequest
from ..services.model_service import train_via_script, read_metrics
from ..crud import fs_store
import pandas as pd

app = FastAPI(title="Proyecto Cloud UAI - API - Rodrigo Andrés Gómez López, Solangel Santana, Hector Alfaro y Francisco Fernandez", version="3.0.1")

# Redirección automática a Swagger UI
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

HELP = {
    "variables": {
        "Edad": "numérico (>=18)",
        "Nivel_Educacional": "categórico [SupInc, SupCom, Posg, Bas, Med]",
        "Anios_Trabajando": "numérico (0–60 y NO puede superar la Edad)",
        "Ingresos": "numérico (>=0)",
        "Deuda_Comercial": "numérico (>=0)",
        "Deuda_Credito": "numérico (>=0)",
        "Otras_Deudas": "numérico (>=0)",
        "Ratio_Ingresos_Deudas": "numérico (0–5) (se calcula automáticamente)",
    },
    "reglas": [
        "Edad debe ser >= 18.",
        "Anios_Trabajando no puede ser mayor que la Edad.",
        "El ratio se calcula como Ingresos / (Deuda_Comercial + Deuda_Credito + Otras_Deudas)."
    ]
}

@app.get("/api/v1/help")
def help_info():
    """
    Visualiza ayuda frente a reglas al introducir datos para la predicción.
    - Edad: numérico >=18.
    - Nivel_Educacional: categórico (SupInc, SupCom, Posg, Bas, Med).
    - Anios_Trabajando: numérico (0–60).
    - Ingresos: numérico (>=0).
    - etc.
    """
    return HELP

@app.post("/api/v1/train")
def train(req: TrainRequest):
    rc = train_via_script(
        req.data_path, req.target, req.threshold, req.test_size, req.sheet
    )
    if rc != 0:
        raise HTTPException(status_code=500, detail="Error entrenando. Revise la consola del servidor.")
    return read_metrics()

@app.get("/api/v1/metrics")
def metrics():
    try:
        return read_metrics()
    except Exception:
        raise HTTPException(status_code=400, detail="Debe entrenar primero.")

# ------------------ Utilidades de normalización/validación --------------------

def _norm_key(k: str) -> str:
    return (
        k.replace(" ", "_")
         .replace("Á", "A").replace("É", "E").replace("Í", "I").replace("Ó", "O").replace("Ú", "U")
         .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
         .replace("ñ", "n").replace("Ñ", "N")
    )

def _normalize_payload_items(items: list[dict]) -> list[dict]:
    norm_items = []
    for d in items:
        nd = {_norm_key(k): v for k, v in d.items()}
        # Alias de años
        if "AÃ±os_Trabajando" in nd and "Anios_Trabajando" not in nd:
            nd["Anios_Trabajando"] = nd.pop("AÃ±os_Trabajando")
        if "Años_Trabajando" in nd and "Anios_Trabajando" not in nd:
            nd["Anios_Trabajando"] = nd.pop("Años_Trabajando")
        if "Anos_Trabajando" in nd and "Anios_Trabajando" not in nd:
            nd["Anios_Trabajando"] = nd.pop("Anos_Trabajando")
        # Remover campos no usados
        nd.pop("Id_Cliente", None)
        norm_items.append(nd)
    return norm_items

def _validate_df_strict(df: pd.DataFrame):
    errors = []
    for i, row in df.iterrows():
        try:
            edad = float(row["Edad"])
            anios = float(row["Anios_Trabajando"])
            if edad < 18:
                raise ValueError("Edad debe ser >= 18 (menor de edad no accede a crédito).")
            if anios > edad:
                raise ValueError("Anios_Trabajando no puede ser mayor que la Edad.")
        except KeyError as ke:
            raise HTTPException(status_code=422, detail=f"Falta columna requerida: {ke}")
        except ValueError as ve:
            errors.append({"index": int(i), "error": str(ve)})
    if errors:
        raise HTTPException(status_code=422, detail={"validation_errors": errors})

# -------------------------------- Predicción ----------------------------------

@app.post("/api/v1/predict")
def predict(req: PredictRequest):
    """
    Predice probabilidades y decisiones para una lista de casos.
    - Normaliza el payload.
    - Valida reglas de negocio (Edad>=18, Anios_Trabajando<=Edad).
    - Calcula Ratio_Ingresos_Deudas siempre.
    - Reindexa al orden de feature_order.json.
    """
    model = fs_store.load_model()
    feats = fs_store.load_features()  # 8 features

    items = [i.model_dump() for i in req.items]
    norm_items = _normalize_payload_items(items)
    df = pd.DataFrame(norm_items)

    # Validación estricta
    _validate_df_strict(df)

    # Calcular ratio (ignorando si vino)
    eps = 1e-9
    if "Ratio_Ingresos_Deudas" in df.columns:
        df = df.drop(columns=["Ratio_Ingresos_Deudas"])
    total = (
        df["Deuda_Comercial"].astype(float)
        + df["Deuda_Credito"].astype(float)
        + df["Otras_Deudas"].astype(float)
    )
    df["Ratio_Ingresos_Deudas"] = (df["Ingresos"].astype(float) / total.clip(lower=eps)).clip(0, 5)

    # Reindexar
    df = df.reindex(columns=feats)

    thr = req.threshold
    if thr is None:
        thr = float(fs_store.load_meta().get("threshold", 0.5))

    proba = model.predict_proba(df)[:, 1]
    yhat = (proba >= thr).astype(int).tolist()
    
       
    # (Opcional) devolver el ratio para trazabilidad
    return {
        "proba": proba.tolist(),
        "decision": yhat,
        "threshold": thr,
        "ratio_ingresos_deudas": df["Ratio_Ingresos_Deudas"].round(4).tolist()
    }
