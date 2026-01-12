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
    this->Histogram.H = nullptr;
    this->fromOriginal = false;
    this->thres_s = 25;
    this->verbosity = 0;
  }

  DIGIMPROC::DIGIMPROC(const char* image, int type,int verbosity) {
    this->image = cv::imread(image, type);
    this->image.copyTo(this->original);
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
    this->Histogram.H = nullptr;
    if(verbosity > 2) this->showIt();
    this->fromOriginal = false;
    this->thres_s = 25;
    this->verbosity = verbosity;
  }

  DIGIMPROC::~DIGIMPROC(){
    if(this->Histogram.H) free(this->Histogram.H);
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

  void DIGIMPROC::renameIt(const char* name){
    this->Nombre = name;
  }
  
  void DIGIMPROC::restoreIt(void){
    this->original.copyTo(this->image);
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
    if(this->image.empty()) return;
    std::string nombre = (name?name:this->Nombre)+"."+extension;
    cv::imwrite(nombre, this->image);
  }

  void DIGIMPROC::SaveHistogram(const char* name, const char* extension){
    if(this->plot.empty()){ std::cerr << "Plot is empty" << std::endl; return; }
    // Check if name is a path and extract basename from it
    std::string path, basename, fullname;
    size_t pos;
    path  = name? name:this->Nombre;
    pos = path.find_last_of("/");
    // If it is a path. Separate basename and parentdir path
    if(pos!=std::string::npos){
      basename = path.substr(pos+1);
      path = path.substr(0,pos);
      fullname = path+"/"+"Histogram_of_"+basename+"."+extension;
    }else{
      basename = path;
      fullname = "Histogram_of_"+basename+"."+extension;
    }
    // Biuld the new full name
    cv::imwrite(fullname, this->plot);
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

  void DIGIMPROC::changeDomain(const int Type){
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

  void DIGIMPROC::Binarization(uchar threshold){
    int i, j;
    i=0; while(i<this->image.rows){
      j=0; while(j<this->image.cols){
        this->image.at<uchar>(i,j)=
          this->image.at<uchar>(i,j)>threshold?255:0;
        ++j;
      }
      ++i;
    }
  }

  void DIGIMPROC::Thresholding(uchar threshold, uchar rpxval, bool solid, bool phantom, bool inverse, std::function<uchar(uchar value)> func){
    int i, j;
    uchar px, *npx, *rpx;
    if(solid || func) { npx=(uchar*)malloc(sizeof(uchar)); *npx=threshold; }
    else npx=&px;
    if(phantom) rpx=&px;
    else { rpx=(uchar*)malloc(sizeof(uchar)); *rpx=rpxval; }
    if(inverse && func){
      i=0; while(i<this->image.rows){
        j=0; while(j<this->image.cols){
          *npx = func(px = this->image.at<uchar>(i,j));
          this->image.at<uchar>(i,j) = px<threshold? *npx:*rpx;
          ++j;
        }
        ++i;
      }
    }else if(func){
      i=0; while(i<this->image.rows){
        j=0; while(j<this->image.cols){
          *npx = func(px = this->image.at<uchar>(i,j));
          this->image.at<uchar>(i,j) = px>threshold? *npx:*rpx;
          ++j;
        }
        ++i;
      }
    }else if(inverse){
      i=0; while(i<this->image.rows){
        j=0; while(j<this->image.cols){
          px = this->image.at<uchar>(i,j);
          this->image.at<uchar>(i,j) = px<threshold? *npx:*rpx;
          ++j;
        }
        ++i;
      }
    }else{
      i=0; while(i<this->image.rows){
        j=0; while(j<this->image.cols){
          px = this->image.at<uchar>(i,j);
          this->image.at<uchar>(i,j) = px>threshold? *npx:*rpx;
          ++j;
        }
        ++i;
      }
    }
    if(solid) free(npx);
  }

  void DIGIMPROC::MultilevelThresholding(uchar *threshold, int n, bool inverse){
    int l, i, j;
    char px;
    uchar levels[256];
    i=0; while(i<256) levels[i++]=0;

    j=i=0; while(i<n){
      while(j<threshold[i]) levels[j++]=threshold[i];
      ++i;
    }

    if(inverse){
      i=0; while(i<this->image.rows){
        j=0; while(j<this->image.cols){
          px = this->image.at<uchar>(i,j);
          l=0; while(l<n){
            if(px>threshold[l]+1){ ++l; continue; }
            this->image.at<uchar>(i,j) = threshold[l];
            ++l;
          }
          ++j;
        }
        ++i;
      }
    }else{
      i=0; while(i<this->image.rows){
        j=0; while(j<this->image.cols){
          this->image.at<uchar>(i,j) = levels[this->image.at<uchar>(i,j)];
          ++j;
        }
        ++i;
      }
    }
  }

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

  void DIGIMPROC::MakeHistogram(void){
    if(this->Histogram.H) free(this->Histogram.H);
    this->Histogram.H = (uint*)calloc(256,sizeof(uint));
    this->Histogram.bins = 256;
    int i, j;
    this->Histogram.max = 0; this->Histogram.min = this->image.total();
    this->Histogram.bins=256;
    i=0; while(i < this->image.rows){
      j=0; while(j < this->image.cols){
        ++this->Histogram.H[this->image.at<uchar>(i,j)];
        ++j;
      } ++i;
    }
    i=0; while(i<(int)this->Histogram.bins){
      if(this->Histogram.H[i]>this->Histogram.max) this->Histogram.max = this->Histogram.H[i];
      if(this->Histogram.H[i]<this->Histogram.min) this->Histogram.min = this->Histogram.H[i];
      ++i;
    }
  }

  void DIGIMPROC::ShowHistogram(void){
    std::string nombre = "Histogram_of_"+this->Nombre;
    cv::namedWindow(nombre,cv::WINDOW_NORMAL);
    cv::imshow(nombre, plot);
  }

  void DIGIMPROC::PrntHistogram(void){
    uint i;
    i=0; while(i<256){
      printf("%d: %d\n", i, this->Histogram.H[i]);
      ++i;
    }
  }

  void DIGIMPROC::PlotHistogram(int _w, int _h){
    if(!this->Histogram.H) return;
    if(_w<256 || _h<256) return;
    if(!(_w%256)) _w=_w+1;

    plot = cv::Mat::zeros(_h,_w,CV_8UC1);
    uint l, h, w, a, b;
    uint binHeight;
    uint binWidth;
    uint faux;
    float _faux;

    faux = plot.cols/256;
    binWidth = faux>1 ? faux - 1 : faux;

    if(faux > 1){
      w=1; // Esta variable corresponde al ancho actual de la barra (variable de control)
           // Habrá que constatar que la forma en que está definida, recorrerá todos los pixeles de plot,
           // de modo que deberá haber una variable de control que nos ayude a recorrer esos pixeles correctamente.
           // En esta parte de la condición, se deja un padding de 1
      _faux = ((float)plot.rows-1) / this->Histogram.max; // Redefinición de faux, esta vez para alturas relativas

      l=0; while(l<this->Histogram.bins){
        if(w>(uint)plot.cols+1){
          std::cerr << "Clipping at " << w << " for bin " << l << std::endl;
          break;
        }
        // N'esta zona voy a hacer la posición del pixel correcto
        // Calculamos la altura de la barra
        // y la posición de la barra según el bin
        binHeight = _faux*(int)this->Histogram.H[l] + 1;
        a=0; // Variable de control complementaria a w, va de 0 a binWidth
        while(a<binWidth){
          h=0; // Esta es la altura actual del bin dibujado (variable de control), va de 0 a binHeight
          b=this->plot.rows-1; // Posición real del pixel
          while(h<binHeight){
            if(h>(uint)plot.rows-1){
              std::cerr << "Clipping at " << h << " for bin " << l << std::endl;
              break;
            }
            plot.at<uchar>(b,w+a) = 255;
            ++h;
            --b;
          }
          ++a;
        }
        ++l;
        w+=binWidth+1;
      } 
    }else{
      return;
    }
  }

  void DIGIMPROC::AutoMultilevelThresholding(uint s){
    this->GetMaxima();
    if(s) this->maxima_s=s;
    uchar *maxindices = (uchar*)calloc(this->maxima_s+1,sizeof(uchar));
    uint i, j;
    j=0; i=0; while(i<256) {
      if(this->maxima[i]) maxindices[j++] = i;
      ++i;
    }
    maxindices[j] = 255;
    if(this->verbosity>2){
      i=0; while(i<this->maxima_s) printf("%d\n", maxindices[i++]);
    }
    this->MultilevelThresholding(maxindices, this->maxima_s);
    free(maxindices);
  }

  void DIGIMPROC::set_thres_s(uint ts){
    this->thres_s = ts;
  }

  void DIGIMPROC::GetMaxima(void){
    if(!this->Histogram.H) return;
    if(!this->Histogram.bins) return;

    uint i, s, idx;
    bool f;

    s=this->thres_s;

    this->maxima_s=0;
    i=0; while(i<256) maxima[i++]=false;
    i=0; while(i<this->Histogram.bins){
      if(!(i+s<this->Histogram.bins)){ s=this->Histogram.bins-i; }
      if((f=maxidx(this->Histogram.H+i, s, idx))) maxima[i+idx]=true;
      if(f) {i+=s; ++this->maxima_s; }else ++i;
    }
    if(this->verbosity>2) this->showMaxima();
  }

  uint DIGIMPROC::maxidx(uint* array, uint s){
    uint i;
    uchar max;
    max=0;
    i=0; while(i<s){
      if(array[i]>max) max=array[i];
      ++i;
    }
    return max;
  }

  bool DIGIMPROC::maxidx(uint* array, uint s, uint &maxid){
    if(!s) return false;
    if(!array) return false;
    uint i;
    float delta;
    float mean;
    bool found;

    delta = s/2.0;
    mean=i=0; while(i<s) mean+=array[i++];
    mean/=s;

    maxid=0; found=false;
    i=0; while(i<s){
      if(array[i]>array[maxid]) maxid=i;
      ++i;
    }
    if(mean<array[maxid] && maxid<delta) found=true;
    return found;
  }

  void DIGIMPROC::showMaxima(void){
    uint i;
    i=0; while(i<256){
      if(this->maxima[i]) printf("%d ", i);
      ++i;
    }
    printf("\n");
  }

}
