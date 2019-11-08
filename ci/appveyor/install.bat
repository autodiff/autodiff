conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda install conda-devenv
conda update -q conda
conda info -a
conda devenv
echo "Activating the conda environment..."
call activate autodiff
echo "Activating the conda environment...finished!"
