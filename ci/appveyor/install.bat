conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda install conda-devenv
conda update -q conda
conda info -a
conda devenv
echo "Activating the conda environment..."
call activate autodiff
echo "Activating the conda environment...finished!"

REM Call this after activating the conda env to ensure MSVC 2017 is selected.
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
