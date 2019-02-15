mkdir build
cd build
echo "Configuring..."
cmake .. -GNinja
echo "Building..."
ninja
