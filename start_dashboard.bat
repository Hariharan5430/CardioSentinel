@echo off
title CardioSentinel - Central ICU Telemetry Console
cls
echo ================================================================
echo  CardioSentinel(TM) - Continuous ICU Telemetry & Arrhythmia AI
echo ================================================================
echo  Starting CardioSentinel Local ICU Monitoring Server...
echo  Console URL: http://127.0.0.1:5000
echo ================================================================
echo.
start "" "http://127.0.0.1:5000"
python run.py
pause
