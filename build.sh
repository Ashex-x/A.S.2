rm -rf build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=${HOME}/Dev/lib/libtorch \
      -DCMAKE_C_COMPILER=/usr/bin/gcc \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DCUDAHOSTCXX=/usr/bin/g++ \
      ..
make -j
