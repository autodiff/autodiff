set PATH=%CONDA%;%CONDA%\Scripts;%CONDA%\Library\bin;%PATH% || goto :error
call activate autodiff || goto :error

echo === Configuring autodiff...
cmake -S . -B build -GNinja || goto :error
echo === Configuring autodiff...finished!

echo === Building and installing autodiff...
ninja install || goto :error
echo === Building and installing autodiff...finished!

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
