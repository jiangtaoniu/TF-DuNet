@echo off
setlocal enabledelayedexpansion
for /R %%f in (*.py) do (
    python -m py_compile "%%f"
    if !errorlevel! neq 0 (
        echo Error compiling: %%f
    )
)
