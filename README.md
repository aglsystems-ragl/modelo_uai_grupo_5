# MODELO CLOUD — ENTRENAMIENTO Y PREDICCIÓN   (Consola, Swagger & RENDER)

# SOLAGEL QUINTANILLA - RODRIGO ANDRES GOMEZ LOPEZ - FRANCISCO FERNANDEZ - HECTOR ALFARO
---
---

RESUMEN: 
Desarrollo del proyecto de análisis y modelado de riesgo crediticio aplicado a la predicción de incumplimiento de clientes (default). El trabajo se enmarca en la metodología CRISP-DM, abordando de forma ordenada las fases de comprensión de datos, preparación, modelado, evaluación y deployment.

---
---
El preente proyecto de **CLOUD COMPUTING PARA CIENCIA DE DATOS** para entrenar y desplegar un modelo de clasificación binaria (riesgo de **Default**) con **FastAPI**. Permite:
- Entrenar desde **consola** y desde **Swagger**.
- Hacer **predicciones** desde consola (interactivo, JSON/CSV) y desde Swagger.
- Ver **gráficas** (ROC, PR, matriz de confusión, coeficientes) y abrir automáticamente la carpeta `artifacts/`.
- Aplicar **reglas de negocio** y calcular automáticamente `Ratio_Ingresos_Deudas`.



En síntesis, el informe presenta no solo un pipeline robusto y metodológicamente justificado para modelar riesgo crediticio, sino también la capacidad estratégica de adaptar la solución a distintos objetivos: robustez técnica para un modelo explicable y optimización competitiva para maximizar desempeño según la métrica de evaluación.

> Repositorio pensado para Windows/PowerShell (probado en la práctica). También funciona en macOS/Linux (ajustando activación del venv).

---
---

##  Objetivos clave
- **Consola**: entrenamiento y predicción con mensajes paso a paso y validaciones.
- **Web (Swagger)**: entrenamiento y predicción con payloads JSON; impresión del resultado también en la **terminal** (logs Uvicorn).
- **Trazabilidad**: artefactos, métricas, orden de features y gráficas se guardan en `artifacts/` y `outputs/`.

---
## ENLACE DEL VIDEO DE ENTRENAMIENTO Y DESPLIEGUE DEL MODELO PREDICTIVO
https://drive.google.com/file/d/1Y_5R9vyhAIFSL68KwWoQ8vA80Hjt651y/view?usp=sharing
---

## Estructura de directorios (resumen)
```
backend/
├─ api/
│  └─ v1/
│     ├─ routes/
│     │  └─ modelo.py              # Endpoints FastAPI (help/train/metrics/predict, logs en terminal)
│     ├─ schemas/
│     │  └─ io.py                  # Pydantic: TrainRequest / PredictRequest
│     └─ services/
│        └─ model_service.py       # Lanza src/train.py con el Python del venv
│
├─ src/
│  ├─ train.py                     # Entrenamiento (mensajes 1/9…9/9, guarda y muestra imágenes)
│  ├─ predict.py                   # Predicción (interactivo/JSON/CSV; valida reglas y calcula ratio)
│  └─ main.py                      # Wrapper: usa sys.executable para invocar train/predict
│
├─ artifacts/                      # Se crea al entrenar
│  ├─ model.joblib                 # Pipeline serializado (pre + modelo)
│  ├─ feature_order.json           # Orden de columnas esperado por el modelo
│  ├─ metrics.json                 # Métricas, umbral, rutas a imágenes/outputs
│  ├─ cm.png, roc.png, pr.png      # Matriz de confusión, curva ROC, Precision-Recall
│  └─ importances.png              # Importancias/coeficientes (si aplica)
│
├─ outputs/                        # Se crea al entrenar
│  ├─ train_clean.csv              # DF de entrenamiento con target
│  ├─ valid_clean.csv              # DF de validación con target
│  └─ predicciones_validacion.csv  # Auditoría de validación (y_true, proba, yhat)
│
├─ data/
│  ├─ Tabla_data_set.xlsx          # Dataset base (hoja 0 por defecto)
│  └─ plantilla_prediccion.csv     # (Opcional) Ejemplo para predicción por CSV
│
├─ tools/
│  ├─ run_train_api.ps1            # Crea/activa venv, instala, entrena y abre Swagger
│  └─ predict_json.ps1             # Helper PowerShell para pasar JSON a predict.py
│
├─ requirements.txt                # Dependencias (Python 3.12 recomendado)
├─ render.yaml                     # permite a Render conocer cómo construir y ejecutar la aplicación sin necesidad de configurar manualmente los comandos desde la interfaz web
└─ README.md                       # Guia de entrenamiento y prediccion en consola, y guia de uso de la API


```
---
## ENTRENAMIENTO DEL MODELO
---
Archivo del modelo serializado: archivo donde se guarda el modelo entrenado para poder cargarlo luego y predecir sin volver a entrenar.
# - artifacts/model.joblib
# Resumen de resultados: 
# artifacts/metrics.json 

contiene umbral, AUC, AP, classification report y rutas a imágenes; y las imágenes en artifacts/cm.png, artifacts/roc.png, artifacts/pr.png, artifacts/importances.png

---
---
## GUIA DE USO DE LA API

## Archivo principal de la API:  api/v1/routes/modelo.py

> Las líneas que empiezan con `#` son comentarios. Copia/pega los **comandos** tal como están.

### 0) Descarga y descompresión del proyecto
```
# Descarga el paquete listo: modelo.zip
# Descomprímelo, por ejemplo en: D:\MODELO_CLOUD\MODELO
# Dentro viene la carpeta data/ (con plantilla CSV/Excel) y la estructura completa.
```

### 1) Limpiar intentos anteriores (opcional pero recomendado)
```powershell
cd D:\MODELO_CLOUD\MODELO\backend

# si tenemos un venv previo activo, lo desactivamos (ignora si no aplica)
deactivate 2>$null

# borramos venv anterior si existe
Remove-Item -Recurse -Force .\modelo_cloud -ErrorAction SilentlyContinue

# limpia cachés de Python
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Remove-Item -Recurse -Force .\.pytest_cache, .\.mypy_cache, .\.ruff_cache -ErrorAction SilentlyContinue
```

### 2) Crear entorno con **Python 3.12 x64** (evita fallos de pandas)
```powershell
# (opcional) permite scripts de venv durante esta sesión
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# crear entorno con 3.12 x64
py -3.12-64 -m venv modelo_cloud

# activar
.\modelo_cloud\Scripts\Activate.ps1

# mostrar versión (debe decir 3.12.x)
python -V
```

### 3) Instalar dependencias
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r .\requirements.txt
```

### 4) Verificar que el Excel esté en `data/`
```powershell
Get-Item .\data\Tabla_data_set.xlsx
```

### 5) Entrenar el modelo **desde consola** (con mensajes paso a paso e imágenes)
```powershell
python .\src\train.py --data .\data\Tabla_data_set.xlsx --sheet 0 --target Default --threshold 0.4 --interactive
```
- Mensajes: “1/9 Cargando dataset…”, “2/9 Limpiando datos…”, “Controlando outliers…”, “Entrenando…”, “Generando imágenes…”, etc.
- Archivos generados:
  - `outputs/train_clean.csv` (DF entrenamiento con target)
  - `outputs/valid_clean.csv` (DF validación con target)
  - `outputs/predicciones_validacion.csv` (auditoría: y_true, proba, yhat)
  - `artifacts/model.joblib`, `feature_order.json`, `metrics.json`
  - `artifacts/cm.png`, `roc.png`, `pr.png`, `importances.png`  
- El script **abre automáticamente** las imágenes y la carpeta `artifacts/`.

### 6) Predecir **desde consola de VS Code**
```powershell
# Interactivo (te pide cada variable)
python .\src\predict.py --interactive

# Alternativa usando el wrapper (siempre que main.py use sys.executable)
python .\src\main.py predict --interactive
```
- Pedirá: `Edad`, `Nivel_Educacional (SupInc, SupCom, Posg, Bas, Med)`, `Anios_Trabajando`, `Ingresos`, `Deuda_Comercial`, `Deuda_Credito`, `Otras_Deudas`.
- El **ratio** se calcula automáticamente y se muestra en pantalla.
- Imprime probabilidad, decisión (umbral entrenado) y una sugerencia.

### 7) Probar **Swagger** (API FastAPI)
```powershell
uvicorn api.v1.routes.modelo:app --reload
# Abre automáticamente /docs. Si no, navega a:
# http://127.0.0.1:8000/docs
```

**Payload para train** (`POST /api/v1/train`):
```json
{
  "data_path": ".\\data\\Tabla_data_set.xlsx",
  "target": "Default",
  "threshold": 0.4,
  "test_size": 0.2,
  "sheet": 0
}
```
**Payload para predict** (`POST /api/v1/predict`): *(el ratio lo calcula el backend)*
```json
{
  "items": [
    {
      "Edad": 40,
      "Nivel_Educacional": "SupCom",
      "Anios_Trabajando": 8,
      "Ingresos": 2500,
      "Deuda_Comercial": 300,
      "Deuda_Credito": 400,
      "Otras_Deudas": 50
    }
  ]
}
```

> La API también imprime el resultado en la **terminal** de Uvicorn con un bloque de texto similar al de la consola interactiva (sin crear CSV).


---

## Reglas de negocio (validadas en consola y API)
- `Edad >= 18` (menor de edad no accede a crédito).
- `Anios_Trabajando <= Edad` y **no puede ser igual** a `Edad`.
- **Límite real de experiencia**: `Anios_Trabajando <= (Edad - 18)` (años posibles desde la mayoría de edad).
- `Ratio_Ingresos_Deudas = Ingresos / (Deuda_Comercial + Deuda_Credito + Otras_Deudas)` (limitado a [0, 5]).

Si los datos no cumplen, la API devuelve **422** y la consola muestra un mensaje claro.

---

##  Requisitos rápidos
- **Python 3.12 x64**
- PowerShell (Windows) o bash (macOS/Linux).
- Instalar con `pip install -r requirements.txt` dentro del **venv**.

---

##  Solución de problemas (FAQ)
- **FileNotFoundError** al entrenar → Ruta/hoja incorrecta en `data_path` o `sheet`.

- **SyntaxWarning: invalid escape sequence** → usa strings crudas `r"..."` o `/` en rutas.

- **422 Unprocessable Entity (Swagger)**  
→ El payload no cumple el esquema (p.ej., faltaba `Edad` o valores fuera de regla). Corrige según la sección *Reglas de negocio*.

- **ValueError: Input X contains NaN (scikit-learn)**  
→ No envíes `NaN`. La consola y la API imputan/validan; revisa tu CSV/JSON.

- **JSONDecodeError en PowerShell** → usa here-string o `--json (Get-Content ... -Raw)`.

- **ModuleNotFoundError** → activa venv y reinstala `requirements.txt`.

---
---

## 8. DESPLIEGUE EN LA NUBE CON RENDER

El despliegue en la nube corresponde a la última etapa del proyecto, en la que el modelo predictivo y su API fueron publicados en Internet para poder ser utilizados desde cualquier dispositivo sin instalación local.

Se utilizó la plataforma **Render**, un servicio *PaaS (Platform as a Service)* que automatiza el proceso de instalación, construcción y ejecución de aplicaciones web.

### Funcionamiento de Render

Render sigue tres fases principales:

1. **Construcción (Build):** descarga el código desde GitHub y ejecuta  
   `pip install -r requirements.txt` para instalar dependencias.
2. **Ejecución (Deploy):** inicia el servidor FastAPI con:  
   `uvicorn api.v1.routes.modelo:app --host 0.0.0.0 --port $PORT`
3. **Publicación:** asigna una URL pública para acceder al servicio:  
   [https://modelo-uai-grupo-5.onrender.com/docs](https://modelo-uai-grupo-5.onrender.com)




### Archivo `render.yaml` (Configuración automática)

El archivo `render.yaml` permite a Render conocer cómo construir y ejecutar la aplicación sin necesidad de configurar manualmente los comandos desde la interfaz web.  
Su contenido (ya incluido en el proyecto) especifica:


```yaml
services:
  - type: web
    name: modelo-cloud
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn api.v1.routes.modelo:app --host 0.0.0.0 --port $PORT"
    plan: free


```
---
---

##  Checklist de cumplimiento
- [x] Estructura clara de directorios y explicación de cada archivo.
- [x] Entrenamiento desde **consola** con progreso, gráficas y apertura de `artifacts/`.
- [x] Entrenamiento desde **Swagger** (POST `/api/v1/train`).  
- [x] Predicción desde **consola** (interactivo, JSON, CSV).
- [x] Predicción desde **Swagger** (POST `/api/v1/predict`) e impresión en terminal.
- [x] **Preprocesamiento** desde Excel, split **train/valid**, y guardado de DFs.
- [x] **Reglas de negocio** y cálculo automático del **ratio**.
- [x] Swagger accesible con redirección `"/" → "/docs"`.
- [x] Despliegue en la nube con RENDER.
- [x] Paso a paso documentado y troubleshooting.
- [x] Documentacion del proyecto.

---
### CREDITOS
Grupo 06 — Trabajo Grupal N°2  
- Solange Quintanilla (API, despliegue)  
- Rodrigo Andrés Gómez López (Modelado, EDA, API, despliegue en la nube)  
- Héctor Alfaro (Pipeline, testing)  
- Francisco Fernandez (Documentación, GitHub)




