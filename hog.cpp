#include <Halide.h>
#include "image_io.h"
#include <time.h>
#include <iostream>
#include <math.h>

using namespace Halide;
using namespace std;


Var x("x"), y("y"), c("c"), bin("bin");
Var cx("cx"), cy("cy");

Expr pi = (float) M_PI;

Func Gradient(Func input) {
  Func grad("grad");
  grad(x, y, c) = Tuple(
      input(x+2, y+1, c) - input(x, y+1, c),
      input(x+1, y+2, c) - input(x+1, y, c));
  return grad;
}

Func Abs(Func input) {
  Func abs("abs");
  abs(x, y, c) = hypot(input(x, y, c)[0], input(x, y, c)[1]);
  return abs;
}

Func Atan2(Func input) {
  Func fatan("fatan");
  fatan(x, y, c) = atan2(input(x, y, c)[1], input(x, y, c)[0]);
  return fatan;
}

Expr BinLookup(Expr angle, Param<uint8_t> nBins) {
  return cast<int8_t>(clamp(floor((angle + pi) * nBins / (2 * pi)), 0, nBins-1));
}

Func Bin(Func angles, Param<uint8_t> nBins) {
  Func bins("bins");
  bins(x, y, c) = BinLookup(angles(x, y, c), nBins);
  return bins;
}

Func Cell(Func bins, Param<uint32_t> cellX, Param<uint32_t> cellY) {
  RDom cell(0, cellX, 0, cellY);
  Func hist("hist");
  hist(x, y, c, bin) = 0;
  hist(x, y, c, bins(cell.x + x*cellX, cell.y + y*cellY, c)) += 1;

  return hist;
}

Func Proj(Func f, Param<uint8_t> nBins) {
  Func proj;
  proj(x, y, c) = f(x, y, c, 0) << 24;
  return proj;
}

int main(int argc, char** argv) {
  using namespace Halide;
  if (argc < 3 || argc > 5) {
    cerr << "USAGE:" << endl
         << "\t" << argv[0] << " INPUT OUTPUT" << endl;
  }
  Image<float> img = load<float>(argv[1]);
  Func input;
  input(x, y, c) = img(x, y, c);
  Param<uint8_t> nBins;
  Param<uint32_t> cellX, cellY;

  int32_t cellSize = 16;
  nBins.set(8);
  cellX.set(cellSize);
  cellY.set(cellSize);
  Func binned = Proj(Cell(Bin(Atan2(Gradient(input)), nBins), cellX, cellY), nBins);
  clock_t c1 = clock();
  binned.compute_root();
  binned.parallel(c);
  binned.vectorize(y, 16);
  binned.compile_jit();
  clock_t c2 = clock();
  cout << "compile: " << (c2 - c1) << endl;
  Image<int32_t> output;
  for (int i = 0; i < 20; i++) {
    clock_t c3 = clock();
    output = binned.realize((img.width() - 2)/cellSize, (img.height() - 2)/cellSize, img.channels());
    clock_t c4 = clock();
    cout << "run      " << (c4 - c3) << endl;
  }
  save(output, argv[2]);
}
