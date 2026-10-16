@echo off
rem Start the RetinaAI demo: double-click this file, then open http://localhost:8501 if no browser opens.
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app.py --server.port 8501
pause
