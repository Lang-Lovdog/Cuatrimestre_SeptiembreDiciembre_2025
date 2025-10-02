#include "ProcesamientoDigitalImagenes.hxx"
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

namespace lovdog {

  DIGIMPROC::DIGIMPROC(){
    this->image = cv::Mat();
    this->Nombre="New Image";
  }

  DIGIMPROC::DIGIMPROC(const char* image, char verbosity) {
    this->image = cv::imread(image, cv::IMREAD_COLOR);
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
    if(verbosity > 2) this->showIt();

  }

  void DIGIMPROC::doxeIt(){
    std::cout
      << "Número de filas:    "  << this->image.rows       << std::endl
      << "Número de columnas: "  << this->image.cols       << std::endl
      << "Total de pixeles:   "  << this->image.total()    << std::endl
      << "Número de canales:  "  << this->image.channels() << std::endl
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
    cv::waitKey(0);
  }

  void DIGIMPROC::saveIt(const char* name, const char* extension){
    cv::imwrite(std::string(name)+std::string(".")+std::string(extension), this->image);
  }

  void DIGIMPROC::newImage(int height, int width, const int type){
    this->image.create(height,width, type);
    std::cout << "Imagen creada";
  }

  void DIGIMPROC::modifyIt(std::function<void(cv::Mat& image)>modifier){
    modifier(this->image);
  }

}
