#include "../ProcesamientoDigitalImagenes.hxx"
#include "opencv2/highgui.hpp"

uchar sine255(uchar x);
uchar sigmoid255(uchar x);

int main(int argc, char *argv[]) {
  lovdog::DIGIMPROC img01("../../res/testImg/C01.bmp");
  lovdog::DIGIMPROC img02("../../res/testImg/C01.bmp");
  lovdog::DIGIMPROC img03("../../res/testImg/C01.bmp");
  lovdog::DIGIMPROC img04("../../res/testImg/C01.bmp");
  uchar th, lim=255;
  img01.renameIt("img01");
  img02.renameIt("img02");
  img03.renameIt("Sigmoid funct");
  img04.renameIt("Sine funct");
  th=0; while(th<lim){
    img04.Thresholding(th,0, false, true, false, sine255);
    img03.Thresholding(th,0, false, true, false, sigmoid255);
    img02.Thresholding(th);
    img01.Binarization(th);
    img04.MakeHistogram();
    img03.MakeHistogram();
    img02.MakeHistogram();
    img01.MakeHistogram();
    img04.showIt();
    img03.showIt();
    img02.showIt();
    img01.showIt();
    img01.PlotHistogram(2*256, 256);
    img02.PlotHistogram(2*256, 256);
    img03.PlotHistogram(2*256, 256);
    img04.PlotHistogram(2*256, 256);
    img01.ShowHistogram();
    img02.ShowHistogram();
    img03.ShowHistogram();
    img04.ShowHistogram();
    cv::waitKey(20);
    img01.restoreIt();
    img02.restoreIt();
    img03.restoreIt();
    img04.restoreIt();
    ++th;
  }
  return 0;
}

uchar sine255(uchar x){
  float pi2=2*3.14159;
  float inFactor = pi2/255;
  return (uchar)255*(1+sin(x*inFactor))/2;
};

uchar sigmoid255(uchar x){
    // 1. Normalize the input from [0, 255] to a linear [0.0, 1.0] scale.
    // Casting to double is crucial for floating-point division.
    double normalized_x = (double)x / 255.0; //

    // 2. Adjust the normalized value to a range suitable for the standard sigmoid's domain, 
    // typically a symmetric range like [-6, 6] or similar, so the S-curve is visible.
    // A linear mapping to [-6, 6] is a common choice for capturing the full slope of the curve.
    double scaled_x = (normalized_x * 12.0) - 6.0;

    // 3. Apply the standard sigmoid function.
    return 255.0 / (1.0 + exp(-scaled_x)); 
};

