set PATH=%CONDA%;%CONDA%\Scripts;%CONDA%\Library\bin;%PATH% || goto :error
call activate autodiff || goto :error

echo === Configuring autodiff...
cmake -S . -B build || goto :error
echo === Configuring autodiff...finished!

echo === Building and installing autodiff...
cmake --build build --config ${{ env.configuration }} --target install || goto :error
echo === Building and installing autodiff...finished!

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
