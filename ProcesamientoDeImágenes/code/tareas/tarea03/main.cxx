#include "../../src/ProcesamientoDigitalImagenes.hxx"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"


int main(void) {
  lovdog::DIGIMPROC img[6]{
    lovdog::DIGIMPROC("./res/HistoTest01.bmp",cv::IMREAD_GRAYSCALE),
    lovdog::DIGIMPROC("./res/HistoTest02.bmp",cv::IMREAD_GRAYSCALE),
    lovdog::DIGIMPROC("./res/HistoTest03.bmp",cv::IMREAD_GRAYSCALE),
    lovdog::DIGIMPROC("./res/HistoTestReal01.bmp",cv::IMREAD_GRAYSCALE),
    lovdog::DIGIMPROC("./res/HistoTestReal02.bmp",cv::IMREAD_GRAYSCALE),
    lovdog::DIGIMPROC("./res/HistoTestReal03.bmp",cv::IMREAD_GRAYSCALE),
 };

  img[0].renameIt("./Reporte.TEX/img.png/HistoTest01");
  img[1].renameIt("./Reporte.TEX/img.png/HistoTest02");
  img[2].renameIt("./Reporte.TEX/img.png/HistoTest03");
  img[3].renameIt("./Reporte.TEX/img.png/HistoTestReal01");
  img[4].renameIt("./Reporte.TEX/img.png/HistoTestReal02");
  img[5].renameIt("./Reporte.TEX/img.png/HistoTestReal03");

  img[0].saveIt();
  img[1].saveIt();
  img[2].saveIt();
  img[3].saveIt();
  img[4].saveIt();
  img[5].saveIt();

  img[0].MakeHistogram();
  img[1].MakeHistogram();
  img[2].MakeHistogram();
  img[3].MakeHistogram();
  img[4].MakeHistogram();
  img[5].MakeHistogram();

  img[0].PlotHistogram();
  img[1].PlotHistogram();
  img[2].PlotHistogram();
  img[3].PlotHistogram();
  img[4].PlotHistogram();
  img[5].PlotHistogram();

  img[0].SaveHistogram();
  img[1].SaveHistogram();
  img[2].SaveHistogram();
  img[3].SaveHistogram();
  img[4].SaveHistogram();
  img[5].SaveHistogram();

  //img[0].ShowHistogram();
  //img[1].ShowHistogram();
  //img[2].ShowHistogram();
  //img[3].ShowHistogram();
  //img[4].ShowHistogram();
  //img[5].ShowHistogram();

  //cv::waitKey(0);

  img[0].set_thres_s(80);
  img[1].set_thres_s(80);
  img[2].set_thres_s(80);
  img[3].set_thres_s(80);
  img[4].set_thres_s(80);
  img[5].set_thres_s(80);

  img[0].AutoMultilevelThresholding();
  img[1].AutoMultilevelThresholding();
  img[2].AutoMultilevelThresholding();
  img[3].AutoMultilevelThresholding();
  img[4].AutoMultilevelThresholding();
  img[5].AutoMultilevelThresholding();

  img[0].renameIt("./Reporte.TEX/img.png/HistoTest01_Thres");
  img[1].renameIt("./Reporte.TEX/img.png/HistoTest02_Thres");
  img[2].renameIt("./Reporte.TEX/img.png/HistoTest03_Thres");
  img[3].renameIt("./Reporte.TEX/img.png/HistoTestReal01_Thres");
  img[4].renameIt("./Reporte.TEX/img.png/HistoTestReal02_Thres");
  img[5].renameIt("./Reporte.TEX/img.png/HistoTestReal03_Thres");

  img[0].MakeHistogram();
  img[1].MakeHistogram();
  img[2].MakeHistogram();
  img[3].MakeHistogram();
  img[4].MakeHistogram();
  img[5].MakeHistogram();

  img[0].PlotHistogram();
  img[1].PlotHistogram();
  img[2].PlotHistogram();
  img[3].PlotHistogram();
  img[4].PlotHistogram();
  img[5].PlotHistogram();

  img[0].SaveHistogram();
  img[1].SaveHistogram();
  img[2].SaveHistogram();
  img[3].SaveHistogram();
  img[4].SaveHistogram();
  img[5].SaveHistogram();

  img[0].saveIt();
  img[1].saveIt();
  img[2].saveIt();
  img[3].saveIt();
  img[4].saveIt();
  img[5].saveIt();

  //img[0].showIt();
  //img[1].showIt();
  //img[2].showIt();
  //img[3].showIt();
  //img[4].showIt();
  //img[5].showIt();

  //img[0].ShowHistogram();
  //img[1].ShowHistogram();
  //img[2].ShowHistogram();
  //img[3].ShowHistogram();
  //img[4].ShowHistogram();
  //img[5].ShowHistogram();

  //cv::waitKey(0);
  return 0;
}
