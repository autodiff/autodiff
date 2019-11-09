echo --- current directory: %cd% ---

REM Activate the conda environment
set PATH=%CONDA%;%CONDA%\Scripts;%CONDA%\Library\bin;%PATH% || goto :error
call activate autodiff || goto :error

echo === Configuring autodiff...
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install || goto :error
echo === Configuring autodiff...finished!

echo === Building and installing autodiff...
cmake --build build --config %CONFIGURATION% --target install || goto :error
echo === Building and installing autodiff...finished!

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
