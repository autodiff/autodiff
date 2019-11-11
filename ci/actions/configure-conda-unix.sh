echo "Installing conda..."
if [ $ENV_OS = "ubuntu-latest" ]; then OS=Linux-x86_64; else OS=MacOSX-x86_64; fi
wget -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-$OS.sh
rm -rf $HOME/miniconda
bash miniconda.sh -b -p $HOME/miniconda
echo "Installing conda...finished!"

echo "Creating conda environment..."
bash $HOME/miniconda/etc/profile.d/conda.sh
export PATH=$HOME/miniconda/bin/:$PATH
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda install conda-devenv
conda update -q conda
conda info -a
conda devenv
echo "Creating conda environment...finished!"
