import json

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, "r", encoding="utf-8") as f:
        code = f.read()

    # Split code by markdown headers/sections to make clean notebook cells
    sections = code.split("# ── ")
    cells = []
    
    # Add a markdown title cell at the start
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Traffic Demand Prediction Solution\n",
            "This notebook runs the complete stacked ensemble model (LightGBM, XGBoost, and CatBoost) with Ridge stacking to predict traffic demand."
        ]
    })

    # Add code cells
    for i, sect in enumerate(sections):
        if not sect.strip():
            continue
        
        lines = sect.splitlines()
        title = lines[0].strip(" ─")
        body_lines = [line + "\n" for line in lines[1:]]
        
        # Add section title as markdown
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title}\n"]
        })
        
        # Add code cell
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": body_lines
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=4)
    print(f"✓ Converted {py_path} to {ipynb_path}")

if __name__ == "__main__":
    convert_py_to_ipynb("run.py", "solution.ipynb")
