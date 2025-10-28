#ifndef LOVDOG_PDI_2025
#define LOVDOG_PDI_2025 "Version alpha 0.0.0"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace lovdog {

  typedef struct dummystruct dummystruct;

  class DIGIMPROC{
    public:
      DIGIMPROC();
      DIGIMPROC(const char* image, char verbosity=0);
      void showIt(const char* WindowName=nullptr, bool renameObject=false);
      void saveIt(const char* name=nullptr, const char* extension="png");
      void doxeIt(void);
      void copyTo(DIGIMPROC& destination);
      void newImage(int width, int height, const int type = CV_8UC1);
      void modifyIt(std::function<void(cv::Mat& image)>modifier);
      void modifyIt(std::function<void(cv::Mat& image, const dummystruct& xtra)>modifier, const dummystruct& extra);
      void changeDomain(const int Type=CV_8UC1, bool custom=false);
      void ComputeDynamicRange(void);
      void Scale(void);
      void Negative(void);
      void Binarization(void);
      void Tresholding(void);
      void MultilevelThresholding(void);
      void DynamicRangeNormalization(const char type = 0);
      void Resize(float factor_r=1);
      void Resize(float factor_x=1, float factor_y=1);
      void Sum(cv::Mat& operand);
      void Sum(lovdog::DIGIMPROC& operand);
      static void AjusteDeRango(cv::Mat &Ent);

    private:
      cv::Mat image;
      std::string Nombre;
      float MinPx, MaxPx;
      char ImageTypeGeneral(void);
      static double DynRngLog();
  };
}

#endif /* ifndef LOVDOG_PDI_2025 */
