#include "../ProcesamientoDigitalImagenes.hxx"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[]) {
  lovdog::DIGIMPROC Foto01("../../res/testImg/C01.bmp", 3);
  lovdog::DIGIMPROC Foto02("../../res/testImg/C02.bmp", 3);
//  lovdog::DIGIMPROC Foto03("../../res/testImg/Family.png", 3);

  Foto01.showIt();
  Foto02.showIt();
  cv::waitKey(0);
  Foto01.Negative();
  Foto02.Negative();
  Foto01.showIt();
  Foto02.showIt();
  cv::waitKey(0);
  return 0;
}
