#include "../../src/ProcesamientoDigitalImagenes.hxx"
#include "opencv2/highgui.hpp"


int main(void) {
  lovdog::DIGIMPROC img[6]{
    lovdog::DIGIMPROC("../../res/testImg/C01.bmp"),
    lovdog::DIGIMPROC("../../res/testImg/C01.bmp"),
    lovdog::DIGIMPROC("../../res/testImg/C01.bmp"),
    lovdog::DIGIMPROC("../../res/testImg/C01.bmp"),
 };
  uchar th[8] = {  0,  64, 128, 192,
                 192, 128,  64,   0};
  uchar *tr = nullptr;

  img[0].renameIt("Multithres");
  img[1].renameIt("Multithres Inv Order");
  img[2].renameIt("Auto Multithres");

  img[0].MultilevelThresholding(th, 4);
  img[1].MultilevelThresholding(th+4, 4);
  img[0].MakeHistogram();
  img[1].MakeHistogram();
  img[0].showIt();
  img[1].showIt();

  img[2].MakeHistogram();
  img[2].set_thres_s(100);
  img[2].AutoMultilevelThresholding();
  img[2].MakeHistogram();
  img[2].showIt();

  img[0].PlotHistogram();
  img[1].PlotHistogram();
  img[2].PlotHistogram();

  img[0].ShowHistogram();
  img[1].ShowHistogram();
  img[2].ShowHistogram();

  cv::waitKey(0);
  return 0;
}

