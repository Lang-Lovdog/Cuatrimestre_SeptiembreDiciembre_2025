#ifndef LOVDOG_PDI_2025
#define LOVDOG_PDI_2025 "Version alpha 0.0.0"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace lovdog {

  typedef struct dummystruct dummystruct;

  typedef struct lovdog_histogram {
    uint *H;
    uint bins;
    uint min;
    uint max;
  }lovdog_histogram;

  class DIGIMPROC{
    public:
      DIGIMPROC();
      DIGIMPROC(const char* image, int type=cv::IMREAD_UNCHANGED, int verbosity=0);
      ~DIGIMPROC();
      void showIt(const char* WindowName=nullptr, bool renameObject=false);
      void saveIt(const char* name=nullptr, const char* extension="png");
      void renameIt(const char* name);
      void restoreIt(void);
      void doxeIt(void);
      void copyTo(DIGIMPROC& destination);
      void newImage(int width, int height, const int type = CV_8UC1);
      void modifyIt(std::function<void(cv::Mat& image)>modifier);
      void modifyIt(std::function<void(cv::Mat& image, const dummystruct& xtra)>modifier, const dummystruct& extra);
      void changeDomain(const int Type=CV_8UC1);
      void ComputeDynamicRange(void);
      void Scale(void);
      void Negative(void);
      /* Thresholding section */
      void Binarization(uchar threshold=128);
      void Thresholding(uchar threshold=128, uchar rpxval=0, bool solid=true, bool phantom=false, bool inverse=false, std::function<uchar(uchar value)> f=nullptr);
      void MultilevelThresholding(uchar *threshold, int n, bool inverse=false);
      void AutoMultilevelThresholding(uint s=0);
      void set_thres_s(uint ts);
      /* Thresholding section */
      /* Histogram section */
      void MakeHistogram(void);
      void ShowHistogram(void);
      void PlotHistogram(int w=256*3, int h=256*2);
      void SaveHistogram(const char* name=nullptr, const char* extension="png");
      void PrntHistogram(void);
      void GetMaxima(void);
      /* Histogram section */
      void DynamicRangeNormalization(const char type = 0);
      void Resize(float factor_r=1);
      void Resize(float factor_x=1, float factor_y=1);
      void Sum(cv::Mat& operand);
      void Sum(lovdog::DIGIMPROC& operand);
      static void AjusteDeRango(cv::Mat &Ent);

    private:
      cv::Mat original;
      cv::Mat image;
      cv::Mat plot;
      std::string Nombre;
      int verbosity;
      float MinPx, MaxPx;
      char ImageTypeGeneral(void);
      lovdog_histogram Histogram;
      static double DynRngLog();
      static uint maxidx(uint* array, uint s);
      static bool maxidx(uint* array, uint s, uint &maxid);
      void showMaxima(void);
      bool fromOriginal;
      bool maxima[256];
      uint maxima_s;
      uint thres_s;
  };
}

#endif /* ifndef LOVDOG_PDI_2025 */
