# Environment Setup

The project uses **[uv](https://docs.astral.sh/uv/)** as the standard way to manage Python and
packages, but everything also works with plain `python` + `pip`. Pick **one** option.

| File | Used by | What it is |
|---|---|---|
| `pyproject.toml` | uv | The project definition: Python version range and the packages we need. |
| `uv.lock` | uv | The exact version of *every* package (created by `uv sync`, commit it). |
| `.python-version` | uv, pyenv | The Python version to use: 3.13. |
| `requirements.txt` | pip | The same pinned packages, for people without uv. |

> **Why Python 3.13:** some packages (e.g. `catboost==1.2.10`) only publish ready-made installers
> ("wheels") up to Python 3.14. On a newer Python (3.15+), the install tries to compile them from
> source, which needs a C++ toolchain and usually fails with a long build error. With 3.13 every
> package installs without compiling. `pyproject.toml` refuses Python 3.15+ for that reason.

## Option A - with uv (recommended)

Install uv once ([instructions](https://docs.astral.sh/uv/getting-started/installation/)):
`curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS / Linux, or
`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows.

Then, from the project root (same commands on macOS, Linux and Windows):

```bash
uv sync
```

That's the whole setup: uv downloads Python 3.13 if you don't have it, creates the environment in
a `.venv` folder, and installs the exact versions from `uv.lock` (it creates `uv.lock` the first
time). Run it again whenever someone changes `pyproject.toml` or `uv.lock`.

Run the project with `uv run` (no need to activate anything):

```bash
uv run python main.py train regression
```

Or activate `.venv` once and use the plain `python ...` commands from the Usage section:
`source .venv/bin/activate` (macOS / Linux), `.venv\Scripts\Activate.ps1` (PowerShell) or
`.venv\Scripts\activate.bat` (cmd.exe).

To add a package: `uv add <package>` (or `uv add --dev <package>` for notebook-only tools), then
refresh the file for pip users: `uv export --format requirements-txt --no-hashes -o requirements.txt`.

## Option B - without uv (Python 3.13 must already be installed)

Check your version with `python3 --version` (macOS / Linux) or `py --list` (Windows). If you don't
have 3.13, install it from [python.org](https://www.python.org/downloads/) (or
`brew install python@3.13` on macOS).

**macOS / Linux**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows -- PowerShell**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows -- cmd.exe**
```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Once the environment is active you'll see `(.venv)` at the start of your prompt. To leave it,
run `deactivate`. If you had an old `venv` folder from before, you can delete it.

## Check it worked

```bash
python --version
```

It should print `Python 3.13.x` (with uv: `uv run python --version`).

## Jupyter kernel (for the notebooks)

```bash
python -m ipykernel install --user --name ETAIProject --display-name "Python (ETAI Group Project)"
```
(with uv: `uv run python -m ipykernel install ...`). In your notebook application, select the
kernel `Python (ETAI Group Project)`.

## Coming back to the project

With uv there is nothing to do: just use `uv run ...`. With pip, activate the environment again
from the project root: `source .venv/bin/activate` (macOS / Linux), `.venv\Scripts\Activate.ps1`
(PowerShell) or `.venv\Scripts\activate.bat` (cmd.exe).

# Usage

The commands below assume the environment is active. With uv you can instead prefix them
with `uv run`, e.g. `uv run python main.py train classification`.

## Train Command

Train the classifier first: predicting spend needs both saved models (the regressor is only
applied to customers the classifier predicts as returning).

To train a classification model and tune the pipeline, run:

```cmd
python main.py train classification
```

To train the regression model (spend in the next 90 days, for customers who return), run:

```cmd
python main.py train regression
```

The algorithm comes from `regression -> models -> selected_model` in `config.yaml`. To compare algorithms without editing the config, pass `--model`:

```cmd
python main.py train regression --model random_forest
python main.py train regression --model hist_gradient_boosting
python main.py train regression --model xgboost
```

Every run writes a report to `results/` (`run_<id>.txt` for classification, `run_<id>_regression.txt` for regression) and adds a row to `results/runs.csv` (the scoreboard). Models are saved to `models/classification.joblib` and `models/regression.joblib` (paths set under `output:` in `config.yaml`). Add `--no-save` to skip saving the model.

## Predict Command

### Regression predictions
Writes `ID, repeat_purchase_90d, spend_next_90d` for the rows the classifier predicts as returning (the regressor is only applied to them):

```cmd
python main.py predict models/classification.joblib models/regression.joblib --output_prediction --output_regression --out results/regression_predictions.csv
```

Add `--output_probability` to any predict command to also write the probability of a repeat purchase.

### Kaggle Submission

To create a Kaggle submission, do:

```cmd
python main.py predict models/classification.joblib --output_prediction --out results/kaggle_submission.csv
```

or equivalently:

```cmd
python main.py predict models/classification.joblib --output_prediction --input data/test.csv --out results/kaggle_submission.csv
```

There is also a shortened version:

```cmd
python main.py p models/classification.joblib -oc -i data/test.csv -o results/kaggle_submission.csv
```
