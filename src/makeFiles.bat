@echo off
for /l %%i in (1,1,100) do (
    copy nul question%%i.py
)