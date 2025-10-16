import argparse, json, sys
import pandas as pd
from pathlib import Path
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

HELP_TEXT = """
El campo Ratio_Ingresos_Deudas se calcula automáticamente como:
  Ratio = Ingresos / (Deuda_Comercial + Deuda_Credito + Otras_Deudas), clip(0,5)

Reglas:
- Edad >= 18 (si no, no procede).
- Anios_Trabajando <= Edad (si no, el usuario debe corregir).

Variables a ingresar:
- Edad
- Nivel_Educacional [SupInc, SupCom, Posg, Bas, Med]
- Anios_Trabajando
- Ingresos
- Deuda_Comercial
- Deuda_Credito
- Otras_Deudas
"""
print("\nAyuda:\n" + HELP_TEXT)
def load_artifacts():
    model = load(ART / "model.joblib")
    import json as j
    feats = j.load(open(ART / "feature_order.json", "r", encoding="utf-8"))["features"]
    thr = j.load(open(ART / "metrics.json", "r", encoding="utf-8")).get("threshold", 0.5)
    return model, feats, thr

def add_ratio(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    total = (
        df["Deuda_Comercial"].astype(float)
        + df["Deuda_Credito"].astype(float)
        + df["Otras_Deudas"].astype(float)
    )
    df["Ratio_Ingresos_Deudas"] = (df["Ingresos"].astype(float) / total.clip(lower=eps)).clip(0, 5)
    return df

def validate_row(row: pd.Series):
    edad = float(row["Edad"])
    anios = float(row["Anios_Trabajando"])
    if edad < 18:
        raise ValueError("La Edad debe ser >= 18. Un menor de edad no puede acceder a crédito.")
    if anios > edad:
        raise ValueError("La cantidad de años Trabajando, no puede ser mayor que la Edad de la persona. Corrija el dato.")
    if anios == edad:
        raise ValueError("La cantidad de años Trabajando, no puede ser igual que la Edad de la persona. Corrija el dato.")
    
    # Regla 3 (nueva): máximo posible desde la mayoría de edad
    max_posible = max(0.0, edad - 18.0)
    if anios > max_posible:
        raise ValueError(
                    f" La cantidad de años Trabajando, ({anios}) supera los años potenciales de labor desde la mayoría de edad "
                    f"({max_posible:.1f} = Edad({edad}) - 18)."
                )
    
# fin de Regla 3 (nueva): máximo posible desde la mayoría de edad

def validate_df(df: pd.DataFrame):
    for i, row in df.iterrows():
        validate_row(row)

def interactive_input(features):
    print("== Modo interactivo. Ingrese valores ==")
    ask_order = [f for f in features if f != "Ratio_Ingresos_Deudas"]
    vals = {}
    for f in ask_order:
        hint = " (SupInc, SupCom, Posg, Bas, Med)" if f == "Nivel_Educacional" else ""
        while True:
            v = input(f"{f}{hint}: ").strip()
            if f == "Nivel_Educacional":
                if v in ["SupInc", "SupCom", "Posg", "Bas", "Med"]:
                    vals[f] = v; break
                else:
                    print("Opciones: SupInc, SupCom, Posg, Bas, Med")
            else:
                try:
                    vals[f] = float(v); break
                except:
                    print("Ingrese número válido.")
    df = pd.DataFrame([vals])

    # Validaciones antes del ratio (para avisar al usuario y repita si quiere)
    try:
        validate_df(df)
    except ValueError as e:
        print(f"\n[ERROR] {e}\nPor favor, vuelva a ejecutar e ingrese valores válidos.")
        sys.exit(1)

    # Calcular ratio y mostrarlo al usuario
    df = add_ratio(df)
    ratio = float(df.loc[0, "Ratio_Ingresos_Deudas"])
    print(f"\n>>> Ratio_Ingresos_Deudas calculado: {ratio:0.4f}\n")

    return df.reindex(columns=features)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--json")
    g.add_argument("--csv")
    g.add_argument("--interactive", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    model, feats, thr_def = load_artifacts()
    thr = args.threshold if args.threshold is not None else thr_def

    if args.json:
        d = json.loads(args.json)
        df = pd.DataFrame([d])
        validate_df(df)
        if "Ratio_Ingresos_Deudas" in df.columns:
            df = df.drop(columns=["Ratio_Ingresos_Deudas"])
        df = add_ratio(df)
        X = df.reindex(columns=feats)

    elif args.csv:
        df = pd.read_csv(args.csv)
        validate_df(df)
        if "Ratio_Ingresos_Deudas" in df.columns:
            df = df.drop(columns=["Ratio_Ingresos_Deudas"])
        df = add_ratio(df)
        X = df.reindex(columns=feats)

    else:
        X = interactive_input(feats)

    proba = model.predict_proba(X)[:, 1]
    yhat = (proba >= thr).astype(int)
    for i, p in enumerate(proba):
        print("_________________________________________________________________________________________________________")
        print("____________________________________ PREDICCIÓN DEL MODELO ______________________________________________")
        print("")
        print(f"Caso {i}: prob={p:0.4f} -> decisión (umbral={thr:0.2f}): {'RIESGO (1)' if yhat[i]==1 else 'NO RIESGO (0)'}")
        print("Sugerencia:", "revise endeudamiento" if yhat[i]==1 else "mantenga hábitos financieros sanos")
        print("")
        print("_________________________________________________________________________________________________________")
        print("")
    #print("\nAyuda:\n" + HELP_TEXT)
