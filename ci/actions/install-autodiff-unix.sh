export PATH=$HOME/miniconda/bin/:$PATH
source activate autodiff

echo "=== Configuring autodiff..."
cmake -S . -B build -GNinja
echo "=== Configuring autodiff...finished!"

echo "=== Building and installing autodiff..."
cmake --build build --target install
echo "=== Building and installing autodiff...finished!"
