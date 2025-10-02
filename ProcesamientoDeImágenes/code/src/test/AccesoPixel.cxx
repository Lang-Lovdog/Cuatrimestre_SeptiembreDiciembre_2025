#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "../ProcesamientoDigitalImagenes.hxx"

void mitadNegro(cv::Mat& image);
void cuadritoAzul(cv::Mat& image);
void PromediaPixel(cv::Mat& image);

int main(int argc, char *argv[]) {
  lovdog::DIGIMPROC Imagen01;
  lovdog::DIGIMPROC Imagen02("../../res/testImg/Family.png");
  lovdog::DIGIMPROC Imagen03("../../res/testImg/Family.png");

  Imagen01.newImage(260, 260);
  Imagen01.modifyIt(mitadNegro);
  Imagen02.modifyIt(cuadritoAzul);
  Imagen03.modifyIt(PromediaPixel);
  Imagen02.doxeIt();
  Imagen03.doxeIt();
  Imagen01.doxeIt();
  Imagen01.showIt();
  Imagen02.showIt();
  Imagen03.showIt();
  return 0;
}

void mitadNegro(cv::Mat& image){
  int i, j;
  i=j=0;
  while(i<image.rows){ j=0;
    while(j<image.cols){
      if(j<image.cols/2) image.at<uchar>(i, j) = 255;
      else image.at<uchar>(i, j) = 0;
      ++j;
    } ++i;
  }
}

void cuadritoAzul(cv::Mat& image){
  int i, j;
  i=j=0;
  while(i<image.rows/4){ j=0;
    while(j<image.cols/5){
      image.at<cv::Vec3b>(i, j)[0] = 255;
      image.at<cv::Vec3b>(i, j)[1] = 0;
      image.at<cv::Vec3b>(i, j)[2] = 0;
      ++j;
    } ++i;
  }
}

void PromediaPixel(cv::Mat& image){
  if(image.channels() < 3) std::cerr << "La imagen no tiene 3 canales" ;
  int i, j;
  uchar Pixel;
  i=j=0;
  while(i<image.rows){ j=0;
    while(j<image.cols){
      Pixel=(
      image.at<cv::Vec3b>(i, j)[0] +
      image.at<cv::Vec3b>(i, j)[1] +
      image.at<cv::Vec3b>(i, j)[2] 
      )/3;
      image.at<cv::Vec3b>(i, j)[0] = Pixel;
      image.at<cv::Vec3b>(i, j)[1] = Pixel;
      image.at<cv::Vec3b>(i, j)[2] = Pixel;
      ++j;
    } ++i;
  }
}
