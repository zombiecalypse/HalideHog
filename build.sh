clang++ hog.cpp -I ../lib/include/ -o hog  `libpng-config --cflags --ldflags` -lpthread -ldl -L ../lib/bin/ -lHalide -lz -g -O3
