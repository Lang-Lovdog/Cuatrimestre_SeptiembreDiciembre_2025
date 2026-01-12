#include <iostream>

uint MeanThres(uint* Hist, uchar* ThresMean, uint n, uint* Mean){
  if(ThresMean) return 0;
  if(!n) return 0;
  uint i, l, mean;
  i=mean=0; while(i<n) mean += Hist[i++];
  mean /= n;
  ThresMean = (uchar*)calloc(256,sizeof(uchar));
  l=i=0; while(i<n) { if(Hist[i]>mean) l+=(ThresMean[i]=1); ++i; }
  *Mean = mean;
  return l;
}

uint MeanThres(uint* Hist, uchar* ThresMean, uchar* ThresHist, uint l, uint* Mean){
  if(ThresMean) return 0;
  if(!l) return 0;
  uint i, m, mean;
  i=mean=0; while(i<l){
    if(ThresHist[i]){ mean += Hist[i++];}
  }
  mean /= l;
  ThresMean = (uchar*)calloc(256,sizeof(uchar));
  m=i=0; while(i<l)
    if(ThresHist[i]){
      if(Hist[i]>mean) m+=(ThresMean[i]=1);
      ++i;
    }
  *Mean = mean;
  return m;
}

void getMaxima(uint *Hist, uchar *max, uint nbins){
  if(!nbins){
    std::cerr << "No hay histograma" << std::endl;
    return;
  }
  if(max) return;
  std::vector<uchar> maxima;
  
  uint l, m, i;
  uint mean[2]; // Las dos medias a utilizar
  uchar* index[2]={nullptr,nullptr}; // Los índices que pasan el umbral
  uchar sensibility = 10;
  uchar maxaux;
  bool ascendent = true;

  l=MeanThres(Hist, index[0], nbins, mean);
  m=MeanThres(Hist, index[1], index[0], l, mean+1);

  if(l && m) {
    maxaux = 0;
    i=0; while(i<nbins){
      if(!index[1][i]) {++i; continue;}
      if(Hist[i]>Hist[maxaux]) {
        maxaux = index[1][i];
        ascendent = true;
      }else if(Hist[i]<Hist[maxaux] && ascendent){
        maxima.push_back(maxaux);
        ascendent = !ascendent;
      }
      ++i;
    }
  }
  if(index[0]) free(index[0]);
  if(index[1]) free(index[1]);
  printMaxima(maxima);
}

void printMaxima(std::vector<uchar> maxima){
  int l;
  l=0; while((size_t)l<maxima.size()){
    std::cout << maxima[l] << " ";
    ++l;
  }
}
