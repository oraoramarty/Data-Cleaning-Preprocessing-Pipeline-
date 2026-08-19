@echo off
chcp 65001 > nul
echo ==========================================
echo       Starting Data Preparation Pipeline
echo ==========================================

echo.
echo [1/4] Fetching Dataset and running EDA...
python -m src.eda
if %errorlevel% neq 0 goto error

echo.
echo [2/4] Preprocessing (Clean, Balance, Standardize)...
python src/preprocessing.py
if %errorlevel% neq 0 goto error

echo.
echo [3/4] Image Processing (Resize, Denoise, Augment)...
python src/image_processing.py
if %errorlevel% neq 0 goto error

echo.
echo [4/4] Splitting Data into Train/Val/Test...
python src/data_split.py
if %errorlevel% neq 0 goto error

echo.
echo ==========================================
echo    [SUCCESS] Pipeline executed perfectly!
echo ==========================================
pause
exit

:error
echo.
echo ==========================================
echo    [ERROR] Something went wrong! Script stopped.
echo ==========================================
pause
