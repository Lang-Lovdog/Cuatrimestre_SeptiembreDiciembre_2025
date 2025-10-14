#include "ProcesamientoDigitalImagenes.hxx"
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

namespace lovdog {

  DIGIMPROC::DIGIMPROC(){
    this->image = cv::Mat();
    this->Nombre="New Image";
    this->MinPx = NAN;
    this->MaxPx = NAN;
  }

  DIGIMPROC::DIGIMPROC(const char* image, char verbosity) {
    this->image = cv::imread(image, cv::IMREAD_UNCHANGED);
    if(this->image.empty()){
      std::cout << "Error al abrir la imagen" << std::endl;
      return;
    }
    if(verbosity > 0) std::cout
      << "Número de filas:    "  << this->image.rows       << std::endl
      << "Número de columnas: "  << this->image.cols       << std::endl
      << "Total de pixeles:   "  << this->image.total()    << std::endl
      << "Número de canales:  "  << this->image.channels() << std::endl
    ;
    this->Nombre = image;
    this->MinPx = NAN;
    this->MaxPx = NAN;
    if(verbosity > 2) this->showIt();

  }

  void DIGIMPROC::doxeIt(){
    std::cout
      << "Nombre:             "  << this->Nombre           << std::endl
      << "Número de filas:    "  << this->image.rows       << std::endl
      << "Número de columnas: "  << this->image.cols       << std::endl
      << "Total de pixeles:   "  << this->image.total()    << std::endl
      << "Número de canales:  "  << this->image.channels() << std::endl
      << "Intensidad máxima:  "  << this->MaxPx            << std::endl
      << "Intensidad mínima:  "  << this->MinPx            << std::endl
    ;
  }

  void DIGIMPROC::showIt(const char* WindowName, bool renameObject){
    if(WindowName){
      if(renameObject) this->Nombre = WindowName;
      cv::namedWindow(WindowName,cv::WINDOW_NORMAL);
      cv::imshow(WindowName, this->image);
      cv::waitKey(0);
      return;
    }
    cv::namedWindow(this->Nombre,cv::WINDOW_NORMAL);
    cv::imshow(this->Nombre, this->image);
  }

  void DIGIMPROC::saveIt(const char* name, const char* extension){
    cv::imwrite(std::string(name)+std::string(".")+std::string(extension), this->image);
  }

  void DIGIMPROC::newImage(int height, int width, const int type){
    this->image.create(height,width, type);
    int i,j;
    switch(this->image.depth()){
      case CV_8U:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols) this->image.at<uchar>(i,j++)=0; ++i; }
        break;
      case CV_8S:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols) this->image.at<schar>(i,j++)=0; ++i; }
        break;
      case CV_16U:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols) this->image.at<ushort>(i,j++)=0; ++i; }
        break;
      case CV_16S:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols) this->image.at<short>(i,j++)=0; ++i; }
        break;
      case CV_32S:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols) this->image.at<int>(i,j++)=0; ++i; }
        break;
      case CV_32F:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols) this->image.at<float>(i,j++)=0; ++i; }
        break;
      case CV_64F:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols) this->image.at<double>(i,j++)=0; ++i; }
        break;
    }
    std::cout << "Imagen creada";
  }

  void DIGIMPROC::modifyIt(std::function<void(cv::Mat& image)>modifier){
    modifier(this->image);
  }

  void DIGIMPROC::modifyIt(std::function<void(cv::Mat& image, const dummystruct& xtra)>modifier, const dummystruct& extra){
    modifier(this->image,extra);
  }

  void DIGIMPROC::changeDomain(const int Type, bool custom){
    this->image.convertTo(this->image, Type);
  }

  void DIGIMPROC::Negative(void){
    int i,j;
    i=0; while(i<this->image.rows){
      j=0; while(j<this->image.cols){
        this->image.at<uchar>(i,j)=255-this->image.at<uchar>(i,j);
        ++j;
      } ++i;
    }
  }

  void DIGIMPROC::Binarization(void){}

  void DIGIMPROC::Tresholding(void){}

  void DIGIMPROC::MultilevelThresholding(void){}

  void DIGIMPROC::ComputeDynamicRange(void){
    int i,j;
    uchar min, max, px;
    min=max=this->image.at<uchar>(0,0);
    if(this->image.channels()==1)switch(this->image.depth()){
      case CV_8U:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols){
            px=this->image.at<uchar>(i,j); if(px<min) min=px; if(px>max) max=px;
        ++j; } ++i; }
        break;
      case CV_8S:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols){
            px=this->image.at<schar>(i,j); if(px<min) min=px; if(px>max) max=px;
        ++j; } ++i; }
        break;
      case CV_16U:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols){
            px=this->image.at<ushort>(i,j); if(px<min) min=px; if(px>max) max=px;
        ++j; } ++i; }
        break;
      case CV_16S:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols){
            px=this->image.at<short>(i,j); if(px<min) min=px; if(px>max) max=px;
        ++j; } ++i; }
        break;
      case CV_32S:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols){
            px=this->image.at<int>(i,j); if(px<min) min=px; if(px>max) max=px;
        ++j; } ++i; }
        break;
      case CV_32F:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols){
            px=this->image.at<float>(i,j); if(px<min) min=px; if(px>max) max=px;
        ++j; } ++i; }
        break;
      case CV_64F:
        i=0; while(i<this->image.rows){ j=0; while(j<this->image.cols){
            px=this->image.at<double>(i,j); if(px<min) min=px; if(px>max) max=px;
        ++j; } ++i; }
        break;
    }
    this->MaxPx = max;
    this->MinPx = min;
  }

  void DIGIMPROC::DynamicRangeNormalization(void){
  }

  void DIGIMPROC::Sum(cv::Mat& operand){
    int i, j;
    i=0; while(i<this->image.rows){
      j=0; while(j<this->image.cols){
        this->image.at<uchar>(i,j) += operand.at<uchar>(i,j);
        ++j;
      } ++i;
    }
  }

  void DIGIMPROC::Sum(lovdog::DIGIMPROC& operand){
    int i, j;
    i=0; while(i<this->image.rows){
      j=0; while(j<this->image.cols){
        this->image.at<uchar>(i,j) += operand.image.at<uchar>(i,j);
        ++j;
      } ++i;
    }
  }

}
