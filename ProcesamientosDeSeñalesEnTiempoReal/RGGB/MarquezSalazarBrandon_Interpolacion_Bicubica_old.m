%% INTERPOLACIÓN BICÚBICA COMPLETA PARA PATRÓN BAYER RGGB
% Aplicación directa de las fórmulas del documento con manejo completo
% I(u+3) = (-I(u) + 9*I(u+2) + 9*I(u+4) - I(u+6)) / 16
% Se procesa directamente la imagen Bayer en un solo canal

clear; close all; clc;

file_img = 'Frutas06r_rggb.jpg';
d = '/';
path_img = './RGGB/';

if ~exist(path_img, 'dir')
    path_img = uigetdir(pwd, 'Selecciona el directorio.');
    path_img = [path_img d];
end

imgRGGB = imread([path_img file_img]);
[Rows, Cols] = size(imgRGGB);

imgRGB = zeros(Rows, Cols, 3, 'uint8');

RowP = 2:2:Rows;
RowI = 1:2:Rows;
ColP = 2:2:Cols;
ColI = 1:2:Cols;

imgRGB(RowI, ColI, 1) = imgRGGB(RowI, ColI);
imgRGB(RowI, ColP, 2) = imgRGGB(RowI, ColP);
imgRGB(RowP, ColI, 2) = imgRGGB(RowP, ColI);
imgRGB(RowP, ColP, 3) = imgRGGB(RowP, ColP);

%% Interpolación horizontal - vectorizada por filas
for i = RowI
    j_R = 4:2:Cols-6;
    if ~isempty(j_R)
        imgRGB(i, j_R, 1) = uint8((-double(imgRGGB(i, j_R-3)) + ...
                                    9*double(imgRGGB(i, j_R-1)) + ...
                                    9*double(imgRGGB(i, j_R+1)) - ...
                                      double(imgRGGB(i, j_R+3))) / 16);
    end
    
    j_G1 = 5:2:Cols-6;
    if ~isempty(j_G1)
        imgRGB(i, j_G1, 2) = uint8((-double(imgRGGB(i, j_G1-3)) + ...
                                     9*double(imgRGGB(i, j_G1-1)) + ...
                                     9*double(imgRGGB(i, j_G1+1)) - ...
                                       double(imgRGGB(i, j_G1+3))) / 16);
    end
end

for i = RowP
    j_G2 = 4:2:Cols-6;
    if ~isempty(j_G2)
        imgRGB(i, j_G2, 2) = uint8((-double(imgRGGB(i, j_G2-3)) + ...
                                     9*double(imgRGGB(i, j_G2-1)) + ...
                                     9*double(imgRGGB(i, j_G2+1)) - ...
                                       double(imgRGGB(i, j_G2+3))) / 16);
    end
    
    j_B = 5:2:Cols-6;
    if ~isempty(j_B)
        imgRGB(i, j_B, 3) = uint8((-double(imgRGGB(i, j_B-3)) + ...
                                    9*double(imgRGGB(i, j_B-1)) + ...
                                    9*double(imgRGGB(i, j_B+1)) - ...
                                      double(imgRGGB(i, j_B+3))) / 16);
    end
end

%% Interpolación vertical - vectorizada por columnas
for c = 1:3
    for j = 1:Cols
        i_vec = 4:Rows-6;
        if ~isempty(i_vec)
            zero_mask = imgRGB(i_vec, j, c) == 0;
            if any(zero_mask)
                i_active = i_vec(zero_mask);
                imgRGB(i_active, j, c) = uint8((-double(imgRGB(i_active-3, j, c)) + ...
                                                 9*double(imgRGB(i_active-1, j, c)) + ...
                                                 9*double(imgRGB(i_active+1, j, c)) - ...
                                                   double(imgRGB(i_active+3, j, c))) / 16);
            end
        end
    end
end

figure('Position', [100 100 800 400]);
subplot(1,2,1); imshow(imgRGGB); title('Imagen Bayer Original');
subplot(1,2,2); imshow(imgRGB);  title('Interpolación Bicúbica Completa');