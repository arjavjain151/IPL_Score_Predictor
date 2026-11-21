@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train_model.py
if errorlevel 1 (
  echo Model training failed. Read the error above.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
