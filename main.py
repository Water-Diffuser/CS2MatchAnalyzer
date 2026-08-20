"""Frozen-binary entry point.

Deliberately a separate file from analyzer/cli.py rather than pointing
PyInstaller at that module directly. PyInstaller executes its entry script as
`__main__`, which strips it of a parent package and makes every relative
import inside it fail at startup — the binary builds cleanly and then dies on
first run. Importing the package by absolute name from outside it keeps
`analyzer` a real package in the bundle.
"""
from analyzer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
