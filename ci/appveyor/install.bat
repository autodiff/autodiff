conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda install conda-devenv
conda update -q conda
conda info -a
conda devenv
source activate autodiff
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=install
make -j
make install
