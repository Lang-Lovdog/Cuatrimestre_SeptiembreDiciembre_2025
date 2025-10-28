/*Programa No.07
  Escalamiento
  El proceso de escalamiento implica modificar
  las dimensiones de una imagen, en función
  de un factor de escala.

  El escalamiento implica necesariamente la
  perdida de información en la imgen ya sea
  por:
  a) disminución de la cantadidad de pixeles
     (perdida de pixeles)
  b) Por falta de pixeles debido al aumento
     de las dimensiones de la imagen.
     (falta de pixeles en la imagen)

 Para solventar el segundo problema existen
 distintos métodos denominados métodos de
 interpolación que buscan determinar los
 valores aproximados de intensidad de los
 cuales no se cuenta con información.

 Otro aspecto importante, es la RELACIÓN DE
 ASPECTO, la cual se puede modificar si
 se aplica un factor de escalamiento distinto
 en cada dimesión de la imágen.     
  
*/
#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
using namespace std;
using namespace cv;
void AjusteDeRango(Mat &Ent);
void AjusteDeRangoLog(Mat &Ent);
void Escalar(Mat &Ent,Mat &Sal, float factor);
int main()
{Mat Img1,Escalada;
 //String FileName="C01.bmp";
 String FileName="Escalada.bmp";
 int factor=2;
 Img1=imread(FileName,IMREAD_GRAYSCALE);
 //Img2=imread("C02.bmp",IMREAD_UNCHANGED);
 if(!Img1.empty())
   {cout<<FileName<<endl;   
    cout<<"Numero de Filas: "<<Img1.rows<<endl;
    cout<<"Numero de Columnas: "<<Img1.cols<<endl;
    cout<<"Total de pixeles: "<<Img1.total()<<endl;
    cout<<"Numero de Canales: "<<Img1.channels()<<endl;
    Escalar(Img1,Escalada,8);
    imshow("Imagen de Entrada",Img1);
    imshow("Imagen Escalada",Escalada);
    //imwrite("Escalada.bmp",Escalada);
    cout<<"(esc)Numero de Filas: "<<Escalada.rows<<endl;
    cout<<"(esc)Numero de Columnas: "<<Escalada.cols<<endl;
    cout<<"(esc)Total de pixeles: "<<Escalada.total()<<endl;
    cout<<"(esc)Numero de Canales: "<<Escalada.channels()<<endl;
    waitKey();
   }
else
 cout<<"No se puede abrir la imagen...";
return 0;
}
void Escalar(Mat &Ent,Mat &Sal, float factor)
{int Paso;
 float suma;
 unsigned char Pixel;
 if(factor>1.0)
   {Paso=factor;
    cout<<"Factor de escala usado="<<Paso<<endl;
    Sal.create(Ent.rows*Paso,Ent.cols*Paso,CV_8UC1);
    for(int u=0; u<Ent.rows; u++)
    for(int v=0; v<Ent.cols; v++)
       {Pixel=Ent.at<unsigned char>(u,v);
        for(int i=u*Paso; i<(u*Paso)+Paso; i++)
        for(int j=v*Paso; j<(v*Paso)+Paso; j++)
           Sal.at<unsigned char>(i,j)=Pixel;
        }
    Sal.convertTo(Sal,CV_8UC1);
   }
  else
  {Paso=1/factor;
   cout<<"Factor de escala usado="<<1.0/Paso<<endl;
   if((Ent.rows%Paso)||(Ent.cols%Paso))
      {cout<<"No se puede escalar la imagen debido a sus dimensiones."<<endl;
       Sal=Ent;
      }
   else
     { Sal.create(Ent.rows/Paso,Ent.cols/Paso,CV_32FC1);
       for(int u=0; u<Ent.rows; u+=Paso)
       for(int v=0; v<Ent.cols; v+=Paso)
          {suma=0;
           for(int i=u; i<(u+Paso); i++)
           for(int j=v; j<(v+Paso); j++)
              suma+=Ent.at<unsigned char>(i,j);
           Sal.at<float>(u/Paso,v/Paso)=suma/(Paso*Paso);
           }
       AjusteDeRango(Sal);    
       Sal.convertTo(Sal,CV_8UC1);
       }
  }
}
void AjusteDeRango(Mat &Ent)
{ float Min,Max,Pixel,rango;
  Min=Ent.at<float>(0,0);
  Max=Min;
  for(int i=0; i<Ent.rows; i++)
  for(int j=0; j<Ent.cols; j++)
       { Pixel=Ent.at<float>(i,j);
        if(Pixel<Min)
           Min=Pixel;
         if(Pixel>Max)
           Max=Pixel;
       }
  rango=Max-Min;
  for(int i=0; i<Ent.rows; i++)
  for(int j=0; j<Ent.cols; j++)
     Ent.at<float>(i,j)=255*((Ent.at<float>(i,j)-Min)/rango);   
}

void AjusteDeRangoLog(Mat &Ent)
{float C,Min,Max,Pixel,rango;
 Min=Ent.at<float>(0,0);
 Max=Min;
 for(int i=0; i<Ent.rows; i++)
 for(int j=0; j<Ent.cols; j++)
    {Pixel=Ent.at<float>(i,j);
     if(Pixel<Min) Min=Pixel;
     if(Pixel>Max) Max=Pixel;
     }
   rango=Max-Min;  
   C=255.0/log(1+rango);
   for(int i=0; i<Ent.rows; i++)
   for(int j=0; j<Ent.cols; j++)
      Ent.at<float>(i,j)= C*log(1+(Ent.at<float>(i,j)-Min));
}
