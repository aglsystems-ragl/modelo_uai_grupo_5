# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from ..schemas.io import TrainRequest, PredictRequest
from ..services.model_service import train_via_script, read_metrics
from ..crud import fs_store

import pandas as pd
from pathlib import Path
import logging

app = FastAPI(
    title="Proyecto Cloud UAI - API - Rodrigo Andrés Gómez López, Solangel Quintana, Hector Alfaro y Francisco Fernandez",
    version="3.0.2"
)

# ---------- Paths y logger (para imprimir en la terminal de VS Code) ----------
ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger("uvicorn")  # imprime en consola donde corre uvicorn

# Redirección automática a Swagger UI
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

HELP = {
    "variables": {
        "Edad": "numérico (>=18)",
        "Nivel_Educacional": "categórico [SupInc, SupCom, Posg, Bas, Med]",
        "Anios_Trabajando": "numérico (0–60; NO puede superar la Edad, ni ser igual, ni superar (Edad-18))",
        "Ingresos": "numérico (>=0)",
        "Deuda_Comercial": "numérico (>=0)",
        "Deuda_Credito": "numérico (>=0)",
        "Otras_Deudas": "numérico (>=0)",
        "Ratio_Ingresos_Deudas": "numérico (0–5) (se calcula automáticamente)",
    },
    "reglas": [
        "Edad debe ser >= 18.",
        "Anios_Trabajando no puede ser mayor que la Edad.",
        "Anios_Trabajando no puede ser igual a la Edad.",
        "Anios_Trabajando no puede superar (Edad - 18), años posibles desde la mayoría de edad.",
        "El ratio se calcula como Ingresos / (Deuda_Comercial + Deuda_Credito + Otras_Deudas)."
    ]
}

@app.get("/api/v1/help")
def help_info():
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

            # Regla 1: mayoría de edad
            if edad < 18:
                raise ValueError("Edad debe ser >= 18 (menor de edad no accede a crédito).")

            # Regla 2: años trabajados vs edad
            if anios > edad:
                raise ValueError("Anios_Trabajando no puede ser mayor que la Edad.")
            if anios == edad:
                raise ValueError("Anios_Trabajando no puede ser igual a la Edad.")

            # Regla 3: máximo posible desde la mayoría de edad
            max_posible = max(0.0, edad - 18.0)
            if anios > max_posible:
                raise ValueError(
                    f"Anios_Trabajando ({anios}) supera los años potenciales desde la mayoría de edad "
                    f"({max_posible:.1f} = Edad({edad}) - 18)."
                )

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
    - Valida reglas de negocio.
    - Calcula Ratio_Ingresos_Deudas siempre.
    - Reindexa al orden de feature_order.json.
    - Imprime resultados 'bonitos' en la terminal (sin guardar CSV).
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

    # Reindexar al orden del entrenamiento
    df = df.reindex(columns=feats)

    thr = req.threshold
    if thr is None:
        thr = float(fs_store.load_meta().get("threshold", 0.5))

    proba = model.predict_proba(df)[:, 1]
    yhat = (proba >= thr).astype(int)

    # ----------- Imprimir en consola (estilo consola interactiva) -----------
    logger.info("_________________________________________________________________________________________________________")
    logger.info("____________________________________ PREDICCIÓN DEL MODELO ______________________________________________")
    for i, p in enumerate(proba):
        decision = "RIESGO (1)" if int(yhat[i]) == 1 else "NO RIESGO (0)"
        logger.info("")
        logger.info("Caso %d: prob=%.4f -> decisión (umbral=%.2f): %s", i, float(p), float(thr), decision)
        logger.info("Ratio_Ingresos_Deudas: %.4f", float(df.iloc[i]["Ratio_Ingresos_Deudas"]))
        logger.info("Sugerencia: %s",
                    "revise endeudamiento" if int(yhat[i]) == 1 else "mantenga hábitos financieros sanos")
    logger.info("_________________________________________________________________________________________________________")
    # -----------------------------------------------------------------------

    # Respuesta API
    return {
        "proba": [float(p) for p in proba],
        "decision": yhat.astype(int).tolist(),
        "threshold": float(thr),
        "ratio_ingresos_deudas": df["Ratio_Ingresos_Deudas"].round(4).tolist()
    }
