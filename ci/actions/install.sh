#!/usr/bin/env bash

conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda install conda-devenv
conda update -q conda
conda info -a
conda devenv
source activate autodiff
mkdir .build
cd .build || exit
cmake .. -GNinja
ninja
