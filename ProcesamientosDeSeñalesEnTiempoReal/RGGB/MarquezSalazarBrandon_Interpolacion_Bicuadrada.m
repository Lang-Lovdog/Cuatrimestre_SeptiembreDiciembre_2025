%% INTERPOLACIÓN BICUADRADA PARA PATRÓN BAYER RGGB - VERSIÓN VECTORIZADA
% Aplicación directa de las fórmulas del documento con indexación vectorizada:
% I(u+1) = (3*I(u) + 6*I(u+2) - I(u+4)) / 8
% Se procesa directamente la imagen Bayer en un solo canal

clear; close all; clc;

%% Cargar la imagen Bayer RGGB
file_img = 'Frutas06r_rggb.jpg';
d = '/';
path_img = './RGGB/';

if ~exist(path_img, 'dir')
    path_img = uigetdir(pwd, 'Selecciona el directorio.');
    path_img = [path_img d];
end

imgRGGB = imread([path_img file_img]);
[Rows, Cols] = size(imgRGGB);

%% Crear la imagen RGB destino
imgRGB = zeros(Rows, Cols, 3, 'uint8');

%% Definir índices de filas y columnas
RowP = 2:2:Rows;
RowI = 1:2:Rows;
ColP = 2:2:Cols;
ColI = 1:2:Cols;

%% Asignar píxeles originales
imgRGB(RowI, ColI, 1) = imgRGGB(RowI, ColI);  % R en (impar, impar)
imgRGB(RowI, ColP, 2) = imgRGGB(RowI, ColP);  % G1 en (impar, par)
imgRGB(RowP, ColI, 2) = imgRGGB(RowP, ColI);  % G2 en (par, impar)
imgRGB(RowP, ColP, 3) = imgRGGB(RowP, ColP);  % B en (par, par)

%% INTERPOLACIÓN HORIZONTAL CON FÓRMULA BICUADRADA - VECTORIZADA
% Fórmula: I(u+1) = (3*I(u) + 6*I(u+2) - I(u+4)) / 8
% Se aplica a cada canal independientemente usando indexación vectorizada

%% Definir rangos de columnas para interpolación horizontal
% Para interpolación en posición u+1, necesitamos columnas u, u+2, u+4
% Las columnas u+1 donde se interpola son las pares para R y G2, impares para G1 y B
cols_u1_RG2 = 2:2:Cols-4;  % Columnas pares para R y G2 (posición u+1)
cols_u1_G1B = 3:2:Cols-4;  % Columnas impares para G1 y B (posición u+1)

cols_u_RG2   = cols_u1_RG2 - 1;  % Columnas u (impares)
cols_u2_RG2  = cols_u1_RG2 + 1;  % Columnas u+2 (impares)
cols_u4_RG2  = cols_u1_RG2 + 3;  % Columnas u+4 (impares)

cols_u_G1B   = cols_u1_G1B - 1;  % Columnas u (pares)
cols_u2_G1B  = cols_u1_G1B + 1;  % Columnas u+2 (pares)
cols_u4_G1B  = cols_u1_G1B + 3;  % Columnas u+4 (pares)

%% Canal Rojo: interpolar en columnas pares (u+1) usando columnas impares
% R original está en filas impares, columnas impares
for i = RowI
    imgRGB(i, cols_u1_RG2, 1) = uint8((3*double(imgRGGB(i, cols_u_RG2)) + ...
                                        6*double(imgRGGB(i, cols_u2_RG2)) - ...
                                        double(imgRGGB(i, cols_u4_RG2))) / 8);
end

%% Canal Verde (G1): interpolar en columnas impares usando columnas pares
% G1 original está en filas impares, columnas pares
for i = RowI
    imgRGB(i, cols_u1_G1B, 2) = uint8((3*double(imgRGGB(i, cols_u_G1B)) + ...
                                        6*double(imgRGGB(i, cols_u2_G1B)) - ...
                                        double(imgRGGB(i, cols_u4_G1B))) / 8);
end

%% Canal Verde (G2): interpolar en columnas pares usando columnas impares
% G2 original está en filas pares, columnas impares
for i = RowP
    imgRGB(i, cols_u1_RG2, 2) = uint8((3*double(imgRGGB(i, cols_u_RG2)) + ...
                                        6*double(imgRGGB(i, cols_u2_RG2)) - ...
                                        double(imgRGGB(i, cols_u4_RG2))) / 8);
end

%% Canal Azul: interpolar en columnas impares usando columnas pares
% B original está en filas pares, columnas pares
for i = RowP
    imgRGB(i, cols_u1_G1B, 3) = uint8((3*double(imgRGGB(i, cols_u_G1B)) + ...
                                        6*double(imgRGGB(i, cols_u2_G1B)) - ...
                                        double(imgRGGB(i, cols_u4_G1B))) / 8);
end

%% INTERPOLACIÓN VERTICAL CON FÓRMULA BICUADRADA - VECTORIZADA
% Fórmula: I(v+1) = (3*I(v) + 6*I(v+2) - I(v+4)) / 8
% Se aplica a posiciones que aún están en cero después de la interpolación horizontal

%% Definir rangos de filas para interpolación vertical
% Para interpolación en posición v+1, necesitamos filas v, v+2, v+4
% Las filas v+1 donde se interpola son las pares
rows_v1 = 2:2:Rows-4;  % Filas pares (posición v+1)

rows_v   = rows_v1 - 1;  % Filas v (impares)
rows_v2  = rows_v1 + 1;  % Filas v+2 (impares)
rows_v4  = rows_v1 + 3;  % Filas v+4 (impares)

%% Completar interpolación vertical para cada canal
for c = 1:3
    % Para cada columna
    for j = 1:Cols
        % Solo procesar posiciones que están en cero
        zero_mask = imgRGB(rows_v1, j, c) == 0;
        if any(zero_mask)
            rows_active = rows_v1(zero_mask);
            v_active    = rows_v(zero_mask);
            v2_active   = rows_v2(zero_mask);
            v4_active   = rows_v4(zero_mask);
            
            imgRGB(rows_active, j, c) = uint8((3*double(imgRGB(v_active, j, c)) + ...
                                               6*double(imgRGB(v2_active, j, c)) - ...
                                               double(imgRGB(v4_active, j, c))) / 8);
        end
    end
end

%% Visualizar resultados
figure('Position', [100 100 800 400]);
subplot(1,2,1); imshow(imgRGGB); title('Imagen Bayer Original');
subplot(1,2,2); imshow(imgRGB);  title('Interpolación Bicuadrada Vectorizada');