## Environment Setup

You only need to do steps 1 and 2 per machine.

## Step 1 - Create and activate the virtual environment

**macOS / Linux**
```bash
python3 -m venv venv           
source venv/bin/activate
pip install -r requirements.txt
```

**Windows -- PowerShell**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows -- cmd.exe**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

## Step 2 - Register the Jupyter kernel

To make the Jupyter kernel use the activated environment, do:

```bash
python -m ipykernel install --user --name ETAIProject --display-name "Python (ETAI Group Project)"
```

In your notebook application, select the kernel `Python (ETAI Group Project)`.

## Step 3 - Reopen the project

Whenever you return to the project, activate the environment from the project root:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows -- PowerShell**
```powershell
venv\Scripts\Activate.ps1
```

**Windows -- cmd.exe**
```cmd
venv\Scripts\activate.bat
```