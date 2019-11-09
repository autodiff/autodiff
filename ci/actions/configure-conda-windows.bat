echo === Creating conda environment... || goto :error
set PATH=%CONDA%;%CONDA%\Scripts;%CONDA%\Library\bin;%PATH% || goto :error
echo PATH: %PATH% || goto :error
conda config --set always_yes yes --set changeps1 no || goto :error
conda config --add channels conda-forge || goto :error
conda install conda-devenv || goto :error
conda update -q conda || goto :error
conda info -a || goto :error
conda devenv || goto :error
echo === Creating conda environment...finished! || goto :error

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
