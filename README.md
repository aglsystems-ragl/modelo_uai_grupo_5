# Despliegue de Modelo ML con FastAPI — Trabajo Grupal N°2 (Grupo 06)

## Integrantes
- Solange Quintanilla (API, despliegue)
- Rodrigo Andrés Gómez López (Modelado, EDA)
- Héctor Alfaro (Pipeline, testing)
- Francisco Fernandez (Documentación, GitHub)

---

## Resumen del proyecto
Se entrenó un pipeline (preprocesamiento + regresión logística) para predecir el riesgo de **Default** usando el dataset `Tabla_data_set.xlsx`. El pipeline se serializa en `artifacts/model.joblib`.

---

## Estructura del repositorio
```
backend/
├─ api/
├─ src/
├─ artifacts/
├─ requirements.txt
└─ README.md
```

---

## Requisitos
- Python 3.12 (recomendado)

Crear y activar entorno virtual:
```
python -m venv modelo_cloud
# PowerShell
.\\modelo_cloud\\Scripts\\Activate.ps1
# CMD
.\\modelo_cloud\\Scripts\\activate.bat
```
Instalar dependencias:
```
pip install -r requirements.txt
```

---

## Ejecutar la API localmente
1. Verifica que `artifacts/model.joblib` y `artifacts/feature_order.json` estén presentes.
2. Desde la raíz del proyecto ejecuta:
```
uvicorn api.v1.routes.modelo:app --reload --port 8000
```
3. Abre Swagger UI:
```
http://127.0.0.1:8000/docs
```

---

## Endpoints principales
- **POST** `/api/v1/train` — Entrena el modelo usando un archivo local (payload: `data_path`, `target`, `threshold`, `test_size`, `sheet`). Devuelve métricas y rutas a artifacts.
- **POST** `/api/v1/predict` — Predict para una lista de items (ver esquema en `/docs`). Devuelve probabilidades, decisiones y el ratio calculado.
- **GET** `/api/v1/health` — Estado simple de la API.

---

## Detalles del modelo y métricas

Se incluyen las métricas guardadas en `artifacts/metrics.json`:

```json
{
  "threshold": 0.5,
  "roc_auc": 0.8455980861244019,
  "average_precision": 0.9076426648246901,
  "classification_report": {
    "0": {
      "precision": 0.7411764705882353,
      "recall": 0.5727272727272728,
      "f1-score": 0.6461538461538462,
      "support": 110.0
    },
    "1": {
      "precision": 0.7813953488372093,
      "recall": 0.8842105263157894,
      "f1-score": 0.8296296296296296,
      "support": 190.0
    },
    "accuracy": 0.77,
    "macro avg": {
      "precision": 0.7612859097127223,
      "recall": 0.7284688995215312,
      "f1-score": 0.7378917378917379,
      "support": 300.0
    },
    "weighted avg": {
      "precision": 0.7666484268125855,
      "recall": 0.77,
      "f1-score": 0.7623551756885091,
      "support": 300.0
    }
  },
  "test_size": 0.2,
  "paths": {
    "model": "artifacts/model.joblib",
    "confusion_matrix": "artifacts/cm.png",
    "roc": "artifacts/roc.png",
    "pr": "artifacts/pr.png",
    "importances": "artifacts/importances.png",
    "train_clean": "outputs/train_clean.csv",
    "valid_clean": "outputs/valid_clean.csv",
    "preds_validacion": "outputs/predicciones_validacion.csv"
  }
}
```

**Orden de features esperado por la API** (desde `artifacts/feature_order.json`):

```
Edad, Nivel_Educacional, Anios_Trabajando, Ingresos, Deuda_Comercial, Deuda_Credito, Otras_Deudas, Ratio_Ingresos_Deudas
```

> Nota: La API reindexa las columnas de entrada según `feature_order.json`.

---

## Reproducir entrenamiento (opcional)
Para regenerar artifacts desde el dataset:
```
python src/train.py --data .\\data\\Tabla_data_set.xlsx --sheet 0 --target Default --threshold 0.4 --interactive
```

---

## Evidencia (entrega)
Incluyan en el repo:
- `assets/screenshot_local.png` — captura del terminal con `uvicorn` y del navegador en `/docs`.
- `assets/demo_video.mp4` — video corto (1–2 minutos).

---

## Despliegue en la nube (opcional)
Sugerencia rápida para Render.com:
1. Conectar repo GitHub a Render.
2. Build command: `pip install -r requirements_clean.txt`
3. Start command: `uvicorn api.v1.routes.modelo:app --host 0.0.0.0 --port $PORT`

---

## Notas finales
- El pipeline se serializa completo en `artifacts/model.joblib`.

