#include "../ProcesamientoDigitalImagenes.hxx"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char *argv[]) {
  lovdog::DIGIMPROC Foto01("../../res/testImg/C01.bmp");
  lovdog::DIGIMPROC Foto02("../../res/testImg/C02.bmp");
  lovdog::DIGIMPROC Foto03;

  Foto01.ComputeDynamicRange();
  Foto01.doxeIt();
  Foto01.showIt();
  Foto02.ComputeDynamicRange();
  Foto02.doxeIt();
  Foto02.showIt();
  Foto03.newImage(480, 640, CV_32FC1);
  Foto03.Sum(Foto01);
  //Foto03.Sum(Foto02);
  Foto03.doxeIt();
  Foto03.showIt();
  cv::waitKey();
  return 0;
}
