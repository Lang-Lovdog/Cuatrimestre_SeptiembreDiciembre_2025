clc;
close all;
clear;

file_img = 'Frutas06r_rggb.jpg';
if ispc
    d = '\';
else
    d = '/';
end


path_img = './RGGB/';

if ~exist(path_img, 'dir')
    path_img = uigetdir(pwd, 'Selecciona el directorio.');
    path_img = [path_img d];
end

imgRGGB = imread([path_img file_img]);
[Rows, Cols] = size(imgRGGB);

%% Proximos vecinos
imgRGB1 = zeros(Rows, Cols, 3, 'uint8');

%figure(1);
% Figure title = Original
%imshow(imgRGGB);

Rojo = 1;
Verd = 2;
Azul = 3;

MatrixB = zeros(Rows, Cols,  'uint8');
MatrixG = zeros(Rows, Cols,  'uint8');
MatrixR = zeros(Rows, Cols,  'uint8');

% Indexados
RowP = 2:2:Rows; 
RowI = 1:2:Rows;
ColP = 2:2:Cols;
ColI = 1:2:Cols;

[size(imgRGGB) ; size(imgRGGB(RowI, ColI))]

% Separación de los colores de la imagen
MatrixR(RowI, ColI) = imgRGGB(RowI, ColI);
MatrixR(RowI, ColP) = imgRGGB(RowI, ColI);
MatrixR(RowP, ColI) = imgRGGB(RowI, ColI);
MatrixR(RowP, ColP) = imgRGGB(RowI, ColI);

MatrixG(RowI, ColI) = imgRGGB(RowI, ColP);
MatrixG(RowI, ColP) = imgRGGB(RowI, ColP);
MatrixG(RowP, ColI) = imgRGGB(RowP, ColI);
MatrixG(RowP, ColP) = imgRGGB(RowP, ColI);

MatrixB(RowI, ColI) = imgRGGB(RowP, ColP);
MatrixB(RowI, ColP) = imgRGGB(RowP, ColP);
MatrixB(RowP, ColI) = imgRGGB(RowP, ColP);
MatrixB(RowP, ColP) = imgRGGB(RowP, ColP);

imgRGB1 = cat(3, MatrixB, MatrixG, MatrixR);

% Figure title = Separación de colores
%figure(2);
%imshow(imgRGB1);

%% Lineal
imgRGB2 = zeros(Rows/2, Cols/2, 3, 'uint8');

imgRGB2(:,:,Rojo) = imgRGGB(RowI,ColI);
imgRGB2(:,:,Verd) = (double(imgRGGB(RowI,ColP))+double(imgRGGB(RowP,ColI)))/2;
imgRGB2(:,:,Azul) = imgRGGB(RowP,ColP);
%imshow(imgRGB2(:, :, Rojo));

%figure(3);
% Figure title = Interpolación lineal
%imshow(imgRGB2);

%% Lineal sin reduccion
imgRGB3 = zeros(Rows, Cols, 3, 'uint8');

imgRGGB_d = double(imgRGGB);
indu1 = 1:2:Cols;
indu2 = 2:2:Cols;
indv1 = 1:2:Rows;
indv2 = 2:2:Rows;

% Originales de RGGB a RGB
imgRGB3(RowI,ColI,Rojo) = imgRGGB_d(RowI,ColI);
imgRGB3(RowI,ColP,Verd) = imgRGGB_d(RowI,ColP);
imgRGB3(RowP,ColI,Verd) = imgRGGB_d(RowP,ColI);
imgRGB3(RowP,ColP,Azul) = imgRGGB_d(RowP,ColP);
%figure(4);
%imshow(imgRGB3);

imgRGB3(indv2(1:end-1), indu1     , Rojo) = (imgRGGB_d(indv1(1:end-1),indu1)+imgRGGB_d(indv1(1:end-1)+2,indu1))/2;
imgRGB3(:             , 2:2:Cols-2, Rojo) = (double(imgRGB3(:,1:2:Cols-2,1))+double(imgRGB3(:,3:2:Cols,1)))/2;

imgRGB3(3:2:Rows-1    , 1:2:Cols  , Verd) = (imgRGGB_d(2:2:Rows-2,1:2:Cols)+imgRGGB_d(4:2:Rows, 1:2:Cols))/2;
imgRGB3(2:2:Rows-1    , 2:2:Cols  , Verd) = (imgRGGB_d(1:2:Rows-2,2:2:Cols)+imgRGGB_d(3:2:Rows, 2:2:Cols))/2;

imgRGB3(3:2:Rows-1    , 2:2:Cols  , Azul) = (imgRGGB_d(2:2:Rows-2,2:2:Cols)+imgRGGB_d(4:2:Rows,2:2:Cols))/2;
imgRGB3(:             , 3:2:Cols-1, Azul) = (double(imgRGB3(:, 2:2:Cols-2, 3))+double(imgRGB3(:, 4:2:Cols, 3)))/2;

%figure(5);
%imshow(imgRGB3);
