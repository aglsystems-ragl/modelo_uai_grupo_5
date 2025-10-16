
# Informe de procesos (v3)
Pipeline: Ingesta (Excel/CSV) → Limpieza + outliers → Split estratificado → Guardado DFs (train/valid) →
Preprocesamiento (OneHot + StandardScaler) → Entrenamiento (LogReg) → Métricas y gráficas →
Auditoría (predicciones_validacion.csv) → Serialización (model.joblib/metrics.json) → Predicción (CLI/API).
