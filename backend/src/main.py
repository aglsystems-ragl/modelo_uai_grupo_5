# src/main.py
import argparse, subprocess, sys, shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

HELP_MSG = r"""
Comandos:
  python .\src\main.py train --data .\data\Tabla_data_set.xlsx --sheet 0 --target Default --threshold 0.4 --interactive
  python .\src\main.py predict --json "{...}" | --csv .\data\plantilla_prediccion.csv | --interactive
  python .\src\main.py train_and_predict --data .\data\Tabla_data_set.xlsx --sheet 0 --target Default
"""

def run(script: str, *args: str) -> int:
    cmd = [sys.executable, str(SRC / script), *args]  # <--- usa el Python del venv
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd, cwd=str(ROOT))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data", default=str(ROOT/"data"/"train.csv"))
    tr.add_argument("--sheet", default=None)
    tr.add_argument("--target", default="target")
    tr.add_argument("--threshold", type=float, default=0.5)
    tr.add_argument("--test_size", type=float, default=0.2)
    tr.add_argument("--interactive", action="store_true")

    pr = sub.add_parser("predict")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--json")
    g.add_argument("--csv")
    g.add_argument("--interactive", action="store_true")
    # si añadiste jsonfile/stdin en predict.py, puedes descomentar:
    # g.add_argument("--jsonfile")
    # g.add_argument("--stdin", action="store_true")
    pr.add_argument("--threshold", type=float, default=None)

    tap = sub.add_parser("train_and_predict")
    tap.add_argument("--data", default=str(ROOT/"data"/"train.csv"))
    tap.add_argument("--sheet", default=None)
    tap.add_argument("--target", default="target")

    hlp = sub.add_parser("help")

    args = ap.parse_args()

    if args.cmd=="help":
        print(HELP_MSG); sys.exit(0)

    if args.cmd=="train":
        extra = []
        if args.sheet is not None: extra += ["--sheet", str(args.sheet)]
        if args.interactive: extra += ["--interactive"]
        sys.exit(run("train.py",
                     "--data", args.data,
                     "--target", args.target,
                     "--threshold", str(args.threshold),
                     "--test_size", str(args.test_size),
                     *extra))

    if args.cmd=="predict":
        extra = []
        if args.json: extra += ["--json", args.json]
        if args.csv: extra += ["--csv", args.csv]
        if args.interactive: extra += ["--interactive"]
        if args.threshold is not None: extra += ["--threshold", str(args.threshold)]
        # si añadiste jsonfile/stdin en predict.py:
        # if args.jsonfile: extra += ["--jsonfile", args.jsonfile]
        # if args.stdin: extra += ["--stdin"]
        sys.exit(run("predict.py", *extra))

    if args.cmd=="train_and_predict":
        rc = run("train.py",
                 "--data", args.data,
                 "--target", args.target,
                 *(["--sheet", str(args.sheet)] if args.sheet is not None else []))
        if rc!=0: sys.exit(rc)
        sys.exit(run("predict.py", "--interactive"))
