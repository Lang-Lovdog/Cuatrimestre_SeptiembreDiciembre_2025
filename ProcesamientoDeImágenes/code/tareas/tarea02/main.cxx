#include "../../src/ProcesamientoDigitalImagenes.hxx"
#include <opencv2/highgui.hpp>

int main(void) {
  lovdog::DIGIMPROC Foto01("../../res/testImg/C01.bmp");
  lovdog::DIGIMPROC Foto02("../../res/testImg/C02.bmp");
  lovdog::DIGIMPROC Foto03("../../res/testImg/Family.png");

  //Foto01.Resize(0.5,2);
  Foto02.Resize(1,3);
  //Foto03.Resize(0.5,0.5);
  Foto01.doxeIt();
  Foto02.doxeIt();
  Foto03.doxeIt();
  Foto01.showIt();
  Foto02.showIt();
  Foto03.showIt();
  cv::waitKey(0);
  return 0;
}
