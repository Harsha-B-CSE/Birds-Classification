@echo off
title Feathered Insights - Swin Transformer Training (RTX 3060)
cd /d "D:\Birds-classification\Birds-classification"
set PYTHONIOENCODING=utf-8
echo ======================================================================
echo Launching 224-Class Swin Transformer Training on RTX 3060 (12GB)
echo ======================================================================
C:\Python313\python.exe -m myapp.ml.train_swin --epochs 30 --warmup-epochs 5 --batch-size 32 --device cuda --export-onnx
echo.
echo Training run finished.
pause
