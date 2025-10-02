#ifndef LOVDOG_PDI_2025
#define LOVDOG_PDI_2025 "Version alpha 0.0.0"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace lovdog {

  class DIGIMPROC{
    public:
      DIGIMPROC();
      DIGIMPROC(const char* image, char verbosity=0);
      void showIt(const char* WindowName=nullptr, bool renameObject=false);
      void saveIt(const char* name=nullptr, const char* extension="png");
      void doxeIt();
      void newImage(int width, int height, const int type = CV_8U);
      void modifyIt(std::function<void(cv::Mat& image)>modifier);

    private:
      cv::Mat image;
      std::string Nombre;
  };

}

#endif /* ifndef LOVDOG_PDI_2025 */
