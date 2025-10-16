# api/v1/schemas/io.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# -------- Entrenamiento --------
class TrainRequest(BaseModel):
    data_path: str = Field(..., description="Ruta al archivo de datos (CSV o XLSX)")
    target: str = Field(..., description="Nombre de la columna objetivo")
    threshold: float = Field(0.5, description="Umbral de decisión")
    test_size: float = Field(0.2, description="Proporción para validación")
    sheet: Optional[int] = Field(None, description="Índice de hoja si es Excel")

# -------- Predicción --------
# IMPORTANTE: ya NO requerimos Ratio_Ingresos_Deudas (se calcula en backend)
NivelEduc = Literal["SupInc", "SupCom", "Posg", "Bas", "Med"]

class PredictItem(BaseModel):
    Edad: float = Field(..., ge=0)
    Nivel_Educacional: NivelEduc
    Anios_Trabajando: float = Field(..., ge=0)
    Ingresos: float = Field(..., ge=0)
    Deuda_Comercial: float = Field(..., ge=0)
    Deuda_Credito: float = Field(..., ge=0)
    Otras_Deudas: float = Field(..., ge=0)
    # Campo opcional (si llega lo ignoramos); queda como compatibilidad
    Ratio_Ingresos_Deudas: Optional[float] = Field(
        None, description="No enviar; el backend lo calcula automáticamente"
    )

class PredictRequest(BaseModel):
    items: List[PredictItem]
    threshold: Optional[float] = Field(
        None, description="Si no se envía, se usa el umbral guardado en entrenamiento"
    )
