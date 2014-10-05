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

Func GradientX(Func input) {
  Func gradx("gradx");
  gradx(x, y, c) = clamp(cast<int32_t>(input(x+2, y+1, c)) - cast<int32_t>(input(x, y+1, c)), -256, 256);
  return gradx;
}

Func GradientY(Func input) {
  Func grady("grady");
  grady(x, y, c) = clamp(cast<int32_t>(input(x+1, y+2, c)) - cast<int32_t>(input(x+1, y, c)),-256, 256);
  return grady;
}

Func Abs(Func input) {
  Func abs("abs");
  abs(x, y, c) = hypot(input(x, y, c)[0], input(x, y, c)[1]);
  return abs;
}

Expr BinLookup(Tuple p, Param<uint8_t> nBins) {
  return cast<int8_t>(clamp(floor((atan2(p[1], p[0]) + pi) * nBins / (2 * pi)), 0, nBins-1));
}

Func Bin(Func gradx, Func grady, Func lut) {
  Func bins("bins");
  bins(x, y, c) = lut(gradx(x, y, c)+256, grady(x,y,c)+256);
  return bins;
}

Func Atan2Lut(Param<uint8_t> nBins, Param<uint16_t> central) {
  Func lut("lut");
  Expr xx = x - 256;
  Expr yy = y - 256;
  lut(x, y) = select(xx*xx + yy*yy > central, BinLookup(Tuple(xx, yy), nBins)+1, 0);
  return lut;
}

Func Cell(Func bins, Param<uint32_t> cellX, Param<uint32_t> cellY) {
  RDom cell(0, cellX, 0, cellY);
  Func hist("hist");
  hist(x, y, c, bin) = 0.f;
  hist(x, y, c, bins(cell.x + x*cellX, cell.y + y*cellY, c)) += 1.f;

  return hist;
}

Func Proj(Func f, Param<uint8_t> nBins) {
  Func proj;
  RDom t(0, nBins+1);
  proj(x, y, c) = sum(f(x, y, c, t))/(100*nBins+100);
  return proj;
}

int main(int argc, char** argv) {
  using namespace Halide;
  if (argc < 3 || argc > 5) {
    cerr << "USAGE:" << endl
         << "\t" << argv[0] << " INPUT OUTPUT" << endl;
  }
  Image<int8_t> img = load<int8_t>(argv[1]);
  Func input;
  input(x, y, c) = img(x, y, c);
  Param<uint8_t> nBins;
  Param<uint16_t> central;
  Param<uint32_t> cellX, cellY;

  int32_t cellSize = 16;
  nBins.set(8);
  central.set(30*30);
  cellX.set(cellSize);
  cellY.set(cellSize);
  Func lut = Atan2Lut(nBins, central);
  lut.bound(x, 0, 512);
  lut.bound(y, 0, 512);
  lut.compute_root();

  Func gradx = GradientX(input);
  Func grady = GradientY(input);
  Func binned = Bin(gradx, grady, lut);
  Func cell = Cell(binned, cellX, cellY);
  Func proj = Proj(cell, nBins);
  clock_t c1 = clock();

  binned.compute_root();
  binned.unroll(c);
  binned.vectorize(y, cellSize);
  proj.bound(c, 0, 3).unroll(c);
  proj.bound(x, 0, (img.width() - 2)/cellSize);
  proj.bound(y, 0, (img.height() - 2)/cellSize);
  proj.compile_jit();
  proj.compute_root();
  clock_t c2 = clock();
  cout << "compile: " << (c2 - c1) << endl;
  Image<float> output;
  for (int i = 0; i < 200; i++) {
    clock_t c3 = clock();
    output = proj.realize((img.width() - 2)/cellSize, (img.height() - 2)/cellSize, img.channels());
    clock_t c4 = clock();
    cout << "run      " << (c4 - c3) << endl;
  }
  save(output, argv[2]);
}
