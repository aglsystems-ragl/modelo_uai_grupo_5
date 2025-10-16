# -*- coding: utf-8 -*-
import argparse, json, os, sys, warnings, subprocess
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "outputs"
ART.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

HELP_TEXT = """
Variables esperadas y rangos sugeridos (el ratio se calcula automáticamente):

- Edad: (18–99)
- Nivel_Educacional: [SupInc, SupCom, Posg, Bas, Med]
- Anios_Trabajando: (0–60 y NO puede superar la Edad)
- Ingresos: (>=0)
- Deuda_Comercial: (>=0)
- Deuda_Credito: (>=0)
- Otras_Deudas: (>=0)
- Ratio_Ingresos_Deudas: Ingresos / (Deuda_Comercial + Deuda_Credito + Otras_Deudas), clip(0,5)
"""

# ----------------------------- Helpers ---------------------------------------

def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def _is_windows() -> bool:
    return sys.platform.startswith("win")

def _open_file(p: Path):
    """Abre un archivo con el visor por defecto (Windows/macOS/Linux)."""
    try:
        if _is_windows():
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as e:
        log(f"No pude abrir {p}: {e}")

def _open_folder(p: Path):
    """Abre una carpeta en el explorador (Windows/macOS/Linux)."""
    try:
        if _is_windows():
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception as e:
        log(f"No pude abrir carpeta {p}: {e}")

def load_dataframe(path: str, target: str, sheet=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        log(f"Leyendo Excel: {path} (sheet={sheet if sheet is not None else 0})")
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    else:
        log(f"Leyendo CSV: {path}")
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";")
    if target not in df.columns:
        if "Default" in df.columns:
            log("No se encontró 'target'. Usando 'Default' como objetivo.")
            target = "Default"
        else:
            raise KeyError(
                f"No se encontró columna objetivo '{target}'. Columnas: {list(df.columns)}"
            )
    return df, target

def norm(s: str) -> str:
    """Normaliza nombres de columnas: sin espacios, sin tildes, ñ->n."""
    rep = (
        str(s).strip()
        .replace(" ", "_")
        .replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U")
        .replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
        .replace("ñ","n").replace("Ñ","N")
    )
    return rep

def outlier_clip(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    low = df[cols].quantile(0.01)
    high = df[cols].quantile(0.99)
    for c in cols:
        df[c] = df[c].clip(lower=low[c], upper=high[c])
    return df

def plot_confusion_matrix(cm, labels, path: Path):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Real",
        xlabel="Predicho",
        title="Matriz de confusión",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plot_roc(y_true, y_score, path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC AUC={roc_auc:0.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(xlabel="FPR", ylabel="TPR", title="Curva ROC")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return roc_auc

def plot_pr(y_true, y_score, path: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP={ap:0.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precisión-Recall")
    ax.legend(loc="lower left")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ap

# ------------------------------ Main -----------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train.csv"))
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--target", default="target")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    if args.interactive:
        input("1/9) Cargando dataset [ENTER]")
    log(f"Cargando dataset desde: {args.data}")

    sheet_arg = int(args.sheet) if (args.sheet is not None and str(args.sheet).isdigit()) else args.sheet
    df, args.target = load_dataframe(args.data, args.target, sheet=sheet_arg)

    # Normalización de nombres
    df.columns = [norm(c) for c in df.columns]
    args.target = norm(args.target)

    # Aliases para años
    aliases = {
        "AÃ±os_Trabajando": "Anios_Trabajando",
        "Años_Trabajando": "Anios_Trabajando",
        "Anos_Trabajando": "Anios_Trabajando",
    }
    for a, b in aliases.items():
        if a in df.columns and "Anios_Trabajando" not in df.columns:
            df = df.rename(columns={a: b})

    # Columnas base requeridas (sin el ratio, se calculará)
    BASE_COLS = [
        "Edad",
        "Nivel_Educacional",
        "Anios_Trabajando",
        "Ingresos",
        "Deuda_Comercial",
        "Deuda_Credito",
        "Otras_Deudas",
    ]
    faltantes_base = [c for c in BASE_COLS if c not in df.columns]
    if faltantes_base:
        raise ValueError(f"Faltan columnas requeridas en el dataset normalizado: {faltantes_base}")

    # Validaciones de negocio
    log("Aplicando validaciones de negocio (Edad>=18 y Anios_Trabajando<=Edad)")
    before = len(df)
    df = df[df["Edad"].astype(float) >= 18]
    df = df[df["Anios_Trabajando"].astype(float) <= df["Edad"].astype(float)]
    dropped = before - len(df)
    if dropped > 0:
        log(f"Se descartaron {dropped} filas por reglas de negocio.")

    # Calcular Ratio_Ingresos_Deudas
    eps = 1e-9
    deu_total = (
        df["Deuda_Comercial"].astype(float)
        + df["Deuda_Credito"].astype(float)
        + df["Otras_Deudas"].astype(float)
    )
    df["Ratio_Ingresos_Deudas"] = (df["Ingresos"].astype(float) / (deu_total.clip(lower=eps))).clip(0, 5)

    FEATS = BASE_COLS + ["Ratio_Ingresos_Deudas"]
    keep = set(FEATS + [args.target])
    df = df[[c for c in df.columns if c in keep]].copy()

    if args.interactive:
        input("2/9) Limpiando datos (duplicados, outliers) [ENTER]")
    log("Eliminando duplicados")
    df = df.drop_duplicates().reset_index(drop=True)

    y = df[args.target].astype(int).values
    X = df[FEATS].copy()

    # Outliers numéricos
    num_cols_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if num_cols_all:
        log("Controlando outliers (clip 1%-99%)")
        X = outlier_clip(X, num_cols_all)

    if args.interactive:
        input("3/9) Split train/test [ENTER]")
    log(f"Split estratificado test_size={args.test_size}")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Guardar DFs con target
    pd.concat([Xtr, pd.Series(ytr, name=args.target)], axis=1).to_csv(OUT / "train_clean.csv", index=False)
    pd.concat([Xte, pd.Series(yte, name=args.target)], axis=1).to_csv(OUT / "valid_clean.csv", index=False)
    
    # pd.concat([Xte, pd.Series[yte,], name=args.target], axis=1).to_csv(OUT / "valid_clean.csv", index=False)  # type: ignore
    # ^^^ Si tu editor marca esto, puedes dejar la versión original:
    # pd.concat([Xte, pd.Series(yte, name=args.target)], axis=1).to_csv(OUT / "valid_clean.csv", index=False)
    log("Guardados outputs/train_clean.csv y outputs/valid_clean.csv")

    if args.interactive:
        input("4/9) Preparando pipeline (Imputer + OneHot + Escalado + LR) [ENTER]")

    # Pipeline con imputación
    cat_cols = ["Nivel_Educacional"]
    num_cols = [c for c in FEATS if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )

    pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=800))])

    if args.interactive:
        input("5/9) Entrenando modelo [ENTER]")
    log("Entrenando...")
    pipe.fit(Xtr, ytr)

    if args.interactive:
        input("6/9) Evaluando y generando imágenes [ENTER]")
    yscore = pipe.predict_proba(Xte)[:, 1]
    yhat = (yscore >= args.threshold).astype(int)

    cm = confusion_matrix(yte, yhat)
    cm_path  = ART / "cm.png"
    roc_path = ART / "roc.png"
    pr_path  = ART / "pr.png"

    plot_confusion_matrix(cm, ["No", "Sí"], cm_path)
    roc_auc = plot_roc(yte, yscore, roc_path)
    ap_score = plot_pr(yte, yscore, pr_path)

    # Abrir imágenes en visor por defecto (más robusto que plt.show())
    log("Abriendo imágenes generadas…")
    for p in [cm_path, roc_path, pr_path]:
        _open_file(p)

    # Importancias (coeficientes LR)
    try:
        coef = pipe.named_steps["model"].coef_[0]
        feat_names = num_cols[:]
        ohe = pipe.named_steps["pre"].named_transformers_.get("cat")
        if ohe is not None and cat_cols:
            feat_names += list(ohe.get_feature_names_out(cat_cols))
        order = np.argsort(np.abs(coef))[::-1][:20]
        fig, ax = plt.subplots(figsize=(6, max(4, len(order) * 0.3)))
        ax.barh(np.array(feat_names)[order], coef[order])
        ax.set(title="Coeficientes (LogReg)")
        ax.invert_yaxis()
        fig.tight_layout()
        imp_path = ART / "importances.png"
        fig.savefig(imp_path, dpi=160)
        plt.close(fig)
        _open_file(imp_path)
    except Exception:
        pass

    # Auditoría de validación
    pd.DataFrame({"y_true": yte, "proba": yscore, "yhat": yhat}).to_csv(
        OUT / "predicciones_validacion.csv", index=False
    )

    if args.interactive:
        input("7/9) Guardando artefactos [ENTER]")
    dump(pipe, ART / "model.joblib")
    json.dump({"features": FEATS}, open(ART / "feature_order.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(
        {
            "threshold": args.threshold,
            "roc_auc": float(roc_auc),
            "average_precision": float(ap_score),
            "classification_report": classification_report(yte, yhat, output_dict=True),
            "test_size": args.test_size,
            "paths": {
                "model": "artifacts/model.joblib",
                "confusion_matrix": "artifacts/cm.png",
                "roc": "artifacts/roc.png",
                "pr": "artifacts/pr.png",
                "importances": "artifacts/importances.png",
                "train_clean": "outputs/train_clean.csv",
                "valid_clean": "outputs/valid_clean.csv",
                "preds_validacion": "outputs/predicciones_validacion.csv",
            },
        },
        open(ART / "metrics.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2,
    )

    # Abrir explorador en artifacts/
    log("Abriendo carpeta de artefactos…")
    _open_folder(ART)

    if args.interactive:
        input("8/9) Síntesis [ENTER]")
    print("\nSÍNTESIS:")
    print(f" - test_size: {args.test_size}")
    print(f" - umbral usado: {args.threshold}")
    print(" - Archivos: artifacts/*.png, model.joblib, feature_order.json, metrics.json; outputs/train_clean.csv, valid_clean.csv, predicciones_validacion.csv")

    if args.interactive:
        input("9/9) Finalizando [ENTER]")
    log(" Modelo entrenado correctamente.")
    print("\nPrediga con:")
    print(r" - Interactivo:  python .\src\predict.py --interactive")
    print(r" - CSV:          python .\src\predict.py --csv .\data\plantilla_prediccion.csv")
    print("\nPara API/Swagger:")
    print(r" - uvicorn api.v1.routes.modelo:app --reload  (abre http://127.0.0.1:8000/docs)")
    print("\nAyuda de variables:\n" + HELP_TEXT)
