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

  void DIGIMPROC::copyTo(DIGIMPROC& destination){
    this->image.copyTo(destination.image);
    destination.Nombre = this->Nombre;
    destination.MinPx = this->MinPx;
    destination.MaxPx = this->MaxPx;
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

  void DIGIMPROC::Resize(float factor_r){
    int Paso;
    float suma;
    unsigned char Pixel;
    cv::Mat Ent, Sal;
    Ent=this->image;
    if(factor_r>1.0){
      Paso=factor_r;
      std::cout<<"Factor de escala usado="<<Paso<<std::endl;
      Sal.create(Ent.rows*Paso,Ent.cols*Paso,CV_8UC1);
      for(int u=0; u<Ent.rows; u++)
        for(int v=0; v<Ent.cols; v++){
          Pixel=Ent.at<unsigned char>(u,v);
          for(int i=u*Paso; i<(u*Paso)+Paso; i++)
            for(int j=v*Paso; j<(v*Paso)+Paso; j++)
                Sal.at<unsigned char>(i,j)=Pixel;
            }
           Sal.convertTo(Sal,CV_8UC1);
           return;
        }
      Paso=1/factor_r;
    std::cout<<"Factor de escala usado="<<1.0/Paso<<std::endl;
    if((Ent.rows%Paso)||(Ent.cols%Paso)){
      std::cout<<"No se puede escalar la imagen debido a sus dimensiones."<<std::endl;
      Sal=Ent;
      return;
    }
    Sal.create(Ent.rows/Paso,Ent.cols/Paso,CV_32FC1);
    for(int u=0; u<Ent.rows; u+=Paso)
      for(int v=0; v<Ent.cols; v+=Paso){
        suma=0;
        for(int i=u; i<(u+Paso); i++)
          for(int j=v; j<(v+Paso); j++)
            suma+=Ent.at<unsigned char>(i,j);
        Sal.at<float>(u/Paso,v/Paso)=suma/(Paso*Paso);
      }
    Sal.convertTo(this->image,CV_8UC1);
  }

  void DIGIMPROC::Resize(float factor_x, float factor_y){
    int Paso_x, Paso_y;
    float suma;
    unsigned char Pixel;
    cv::Mat Ent, Sal;
    Ent=this->image;
    if(factor_x>1.0 && factor_y>1.0){
      Paso_x=factor_x;
      Paso_y=factor_y;
      std::cout<<"Factor de escala usado="<<Paso_x<<"x"<<Paso_y<<std::endl;
      Sal.create(Ent.rows*Paso_y,Ent.cols*Paso_x,CV_8UC1);
      for(int u=0; u<Ent.rows; u++)
        for(int v=0; v<Ent.cols; v++){
          Pixel=Ent.at<unsigned char>(u,v);
          for(int i=u*Paso_y; i<(u*Paso_y)+Paso_y; i++)
            for(int j=v*Paso_x; j<(v*Paso_x)+Paso_x; j++)
                Sal.at<unsigned char>(i,j)=Pixel;
          }
           Sal.convertTo(Sal,CV_8UC1);
    }else if(factor_x<1.0 && factor_y>1.0){
      Paso_x=1/factor_x;
      Paso_y=factor_y;
      std::cout<<"Factor de escala usado="<<1.0/Paso_x<<"x"<<Paso_y<<std::endl;
      if(Ent.cols%Paso_x){
        std::cout<<"No se puede escalar la imagen debido a sus dimensiones."<<std::endl;
        return;
      }else{
      Sal.create(Ent.rows*Paso_y,Ent.cols/Paso_x,CV_32FC1);
      for(int u=0; u<Ent.rows; u++)
        for(int v=0; v<Ent.cols; v+=Paso_x){
          suma=0;
          for(int i=u; i<(u+Paso_y); i++)
            for(int j=v; j<(v+Paso_x); j++)
              suma+=Ent.at<unsigned char>(i,j);
          Sal.at<float>(u/Paso_y,v)=suma/(Paso_y*Paso_x);
        }
      }
    }else if(factor_x>1 && factor_y<1.0){
      Paso_y=1/factor_y;
      Paso_x=factor_x;
      std::cout<<"Factor de escala usado="<<Paso_x<<"x"<<1.0/Paso_y<<std::endl;
      if(Ent.rows%Paso_y){
        std::cout<<"No se puede escalar la imagen debido a sus dimensiones."<<std::endl;
        return;
      }else{
        Sal.create(Ent.rows/Paso_y,Ent.cols*Paso_x,CV_32FC1);
        for(int u=0; u<Ent.rows; u+=Paso_y)
          for(int v=0; v<Ent.cols; v++){
            suma=0;
            for(int i=u; i<(u+Paso_y); i++)
              for(int j=v; j<(v+Paso_x); j++)
                suma+=Ent.at<unsigned char>(i,j);
            Sal.at<float>(u,v/Paso_x)=suma/(Paso_y*Paso_x);
          }
      }
    }else if(factor_x<1.0 && factor_y<1.0){
      Paso_x=1/factor_x;
      Paso_y=1/factor_y;
      std::cout<<"Factor de escala usado="<<1.0/Paso_x<<"x"<<1.0/Paso_y<<std::endl;
      if((Ent.rows%Paso_y)||(Ent.cols%Paso_x)){
        std::cout<<"No se puede escalar la imagen debido a sus dimensiones."<<std::endl;
        return;
      }else{
        Sal.create(Ent.rows/Paso_y,Ent.cols/Paso_x,CV_32FC1);
        for(int u=0; u<Ent.rows; u+=Paso_y)
          for(int v=0; v<Ent.cols; v+=Paso_x){ {
            suma=0;
             for(int i=u; i<(u+Paso_y); i++)
               for(int j=v; j<(v+Paso_x); j++)
                 suma+=Ent.at<unsigned char>(i,j);
            Sal.at<float>(u/Paso_y,v/Paso_x)=suma/(Paso_y*Paso_x);
          }
        AjusteDeRango(Sal);    
        Sal.convertTo(Sal,CV_8UC1);
     }
    }
  }
  Sal.convertTo(this->image,CV_8UC1);
}

  void DIGIMPROC::AjusteDeRango(cv::Mat &Ent){
    float Min,Max,Pixel,rango;
    Min=Ent.at<float>(0,0);
    Max=Min;
    for(int i=0; i<Ent.rows; i++)
      for(int j=0; j<Ent.cols; j++){
        Pixel=Ent.at<float>(i,j);
        if(Pixel<Min) Min=Pixel;
        if(Pixel>Max) Max=Pixel;
      }
    rango=Max-Min;
    for(int i=0; i<Ent.rows; i++)
      for(int j=0; j<Ent.cols; j++)
        Ent.at<float>(i,j)=255*((Ent.at<float>(i,j)-Min)/rango);   
  }

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

  void DIGIMPROC::DynamicRangeNormalization(const char type){
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
