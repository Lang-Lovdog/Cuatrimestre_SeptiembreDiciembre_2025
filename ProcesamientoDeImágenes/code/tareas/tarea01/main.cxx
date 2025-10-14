#include "../../src/ProcesamientoDigitalImagenes.hxx"
#include "opencv2/highgui.hpp"
#include "main.hxx"

struct lovdog::dummystruct {
  size_t chess_width;
  size_t chess_height;
  size_t chess_rows;
  size_t chess_cols;
  size_t spiral_img_width;
  size_t spiral_img_height;
  size_t spiral_thickness;
};

void chessboard(cv::Mat& image, const lovdog::dummystruct& dummy){
  size_t x,y, i, j, a;
  image.create(
      dummy.chess_height * dummy.chess_rows,
      dummy.chess_width  * dummy.chess_cols,
      CV_8UC1
  );
  i=0; while(i<(size_t)image.rows){j=0; while(j<(size_t)image.cols) image.at<uchar>(i,j++)=255; ++i; }
  i=0; while(i<dummy.chess_rows){     // Avance fluído y libre
    j=(i&1)*dummy.chess_width; // Depende del estado de i
    while((int)j<image.cols){
      a=i*dummy.chess_height;
      x=0; while(x < dummy.chess_width ){
        y=0; while(y < dummy.chess_height){
          image.at<uchar>(a+(y++),j+x) = 0;
          //cv::namedWindow("chessboard", cv::WINDOW_NORMAL);
          //cv::imshow("chessboard", image);
          //cv::waitKey(1);
        }
        ++x;
      } j+=2*dummy.chess_width;
    } ++i;
  }
}

void spiral(cv::Mat& image, const lovdog::dummystruct& dummy){
    image.create(dummy.spiral_img_height, dummy.spiral_img_width, CV_8UC1);
    image.setTo(255);
    
    // Calcular márgenes
    size_t msp = (dummy.spiral_img_width % dummy.spiral_thickness) / 2;
    size_t miz = (dummy.spiral_img_height % dummy.spiral_thickness) / 2;
    
    int x = miz, y = msp;
    int dx = dummy.spiral_thickness, dy = 0;
    int min_x = miz, max_x = dummy.spiral_img_height - miz - 1;
    int min_y = msp, max_y = dummy.spiral_img_width - msp - 1;
    
    while(min_x <= max_x && min_y <= max_y) {
        // Dibujar el grosor completo en la posición actual
        for(int i = 0; i < dummy.spiral_thickness; i++) {
            for(int j = 0; j < dummy.spiral_thickness; j++) {
                int px = x + i;
                int py = y + j;
                if(px >= 0 && px < image.rows && py >= 0 && py < image.cols) {
                    image.at<uchar>(px, py) = 0;
                }
            }
        }
        
        // Mostrar progreso
        cv::imshow("spiral", image);
        cv::waitKey(1);
        
        // Mover
        x += dx;
        y += dy;
        
        // Cambiar dirección al llegar a los límites
        if(y + dummy.spiral_thickness > max_y && dx > 0) { 
            dx = 0; 
            dy = dummy.spiral_thickness; 
            y = max_y - dummy.spiral_thickness + 1;
            min_x += dummy.spiral_thickness;
        }
        else if(x + dummy.spiral_thickness > max_x && dy > 0) { 
            dx = -dummy.spiral_thickness; 
            dy = 0; 
            x = max_x - dummy.spiral_thickness + 1;
            max_y -= dummy.spiral_thickness;
        }
        else if(y < min_y && dx < 0) { 
            dx = 0; 
            dy = -dummy.spiral_thickness; 
            y = min_y;
            max_x -= dummy.spiral_thickness;
        }
        else if(x < min_x && dy < 0) { 
            dx = dummy.spiral_thickness; 
            dy = 0; 
            x = min_x;
            min_y += dummy.spiral_thickness;
        }
    }
}

int main(void) {
  lovdog::dummystruct dummy = {
    .chess_width = 40,
    .chess_height = 20,
    .chess_rows = 8,
    .chess_cols = 8,
    .spiral_img_width = 40,
    .spiral_img_height = 50,
    .spiral_thickness = 5
  };
  lovdog::DIGIMPROC pdi, pdj;
  pdi.modifyIt(chessboard, dummy);
  pdi.showIt("chessboard");
  pdi.saveIt("chessboard.png");
  cv::waitKey();
  return 0;
}
