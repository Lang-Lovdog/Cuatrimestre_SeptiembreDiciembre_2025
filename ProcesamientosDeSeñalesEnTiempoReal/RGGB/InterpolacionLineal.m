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

% Leer imagen en formato de mosaico RGGB
imgRGGB = imread([path_img file_img]);
[lv, lu] = size(imgRGGB);
imgRGGB_d = double(imgRGGB);

%% Proximos vecinos
imgRGB1 = zeros(lv, lu, 3, 'uint8');

figure(1);
% Figure title = Original
imshow(imgRGGB);

% Rojo
imgRGB1(1:2:lv, 1:2:lu, 1) = imgRGGB(1:2:lv, 1:2:lu);
imgRGB1(2:2:lv, 1:2:lu, 1) = imgRGGB(1:2:lv, 1:2:lu);
imgRGB1(1:2:lv, 2:2:lu, 1) = imgRGGB(1:2:lv, 1:2:lu);
imgRGB1(2:2:lv, 2:2:lu, 1) = imgRGGB(1:2:lv, 1:2:lu);

% Verde
imgRGB1(1:2:lv, 1:2:lu, 2) = imgRGGB(1:2:lv, 2:2:lu);
imgRGB1(1:2:lv, 2:2:lu, 2) = imgRGGB(1:2:lv, 2:2:lu);
imgRGB1(2:2:lv, 1:2:lu, 2) = imgRGGB(2:2:lv, 1:2:lu);
imgRGB1(2:2:lv, 2:2:lu, 2) = imgRGGB(2:2:lv, 1:2:lu);

% Azul
imgRGB1(2:2:lv, 2:2:lu, 3) = imgRGGB(2:2:lv, 2:2:lu);
imgRGB1(1:2:lv, 1:2:lu, 3) = imgRGGB(2:2:lv, 2:2:lu);
imgRGB1(2:2:lv, 1:2:lu, 3) = imgRGGB(2:2:lv, 2:2:lu);
imgRGB1(1:2:lv, 2:2:lu, 3) = imgRGGB(2:2:lv, 2:2:lu);

% Figure title = Separación de colores
figure(2);
imshow(imgRGB1);
figure(3);
imshow(imgRGB1(:,:,1));
figure(4);
imshow(imgRGB1(:,:,2));
figure(5);
imshow(imgRGB1(:,:,3));

%% Lineal
imgRGB2 = zeros(lv/2, lu/2, 3, 'uint8');

imgRGB2(:,:,1) = imgRGGB(1:2:lv, 1:2:lu);
imgRGB2(:,:,2) = imgRGGB_d(1:2:lv, 2:2:lu)+imgRGGB_d(2:2:lv, 1:2:lu)/2;
imgRGB2(:,:,3) = imgRGGB(2:2:lv, 2:2:lu);

figure(3);
% Figure title = Interpolación lineal
imshow(imgRGB2);

%% Lineal sin reduccion
imgRGB3 = zeros(lv, lu, 3, 'uint8');

indu1 = 1:2:lu;
indu2 = 2:2:lu;
indv1 = 1:2:lv;
indv2 = 2:2:lv;

% Originales de RGGB a RGB
imgRGB3(indv1, indu1, 1) = imgRGGB_d(indv1, indu1);
imgRGB3(indv1, indu2, 2) = imgRGGB_d(indv1, indu2);
imgRGB3(indv2, indu1, 2) = imgRGGB_d(indv2, indu1);
imgRGB3(indv2, indu2, 3) = imgRGGB_d(indv2, indu2);

imgRGB3(indv2(1:end-1), indu1, 1) = (imgRGGB_d(indv1(1:end-1),indu1)+imgRGGB_d(indv1(1:end-1)+2,indu1))/2;
imgRGB3(:,2:2:lu-2,1) = (double(imgRGB3(:,1:2:lu-2,1))+double(imgRGB3(:,3:2:lu,1)))/2;

imgRGB3(3:2:lv-1,1:2:lu,2) = (imgRGGB_d(2:2:lv-2,1:2:lu)+imgRGGB_d(4:2:lv, 1:2:lu))/2;
imgRGB3(2:2:lv-1,2:2:lu,2) = (imgRGGB_d(1:2:lv-2,2:2:lu)+imgRGGB_d(3:2:lv, 2:2:lu))/2;

imgRGB3(3:2:lv-1, 2:2:lu, 3) = (imgRGGB_d(2:2:lv-2,2:2:lu)+imgRGGB_d(4:2:lv,2:2:lu))/2;
imgRGB3(:,3:2:lu-1, 3) = (double(imgRGB3(:, 2:2:lu-2, 3))+double(imgRGB3(:, 4:2:lu, 3)))/2;

figure(6);
imshow(imgRGB3);
