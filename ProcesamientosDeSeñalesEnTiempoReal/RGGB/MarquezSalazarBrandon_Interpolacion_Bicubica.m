%% INTERPOLACIÓN BICÚBICA PARA PATRÓN BAYER RGGB - VERSIÓN VECTORIZADA
% Aplicación directa de las fórmulas del documento con indexación vectorizada:
% I(u+3) = (-I(u) + 9*I(u+2) + 9*I(u+4) - I(u+6)) / 16
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

%% INTERPOLACIÓN HORIZONTAL CON FÓRMULA BICÚBICA - VECTORIZADA
% Fórmula: I(u+3) = (-I(u) + 9*I(u+2) + 9*I(u+4) - I(u+6)) / 16
% Interpola en posición u+3 usando puntos u, u+2, u+4, u+6

%% Definir rangos de columnas para interpolación horizontal
% Para interpolación en posición u+3, necesitamos columnas u, u+2, u+4, u+6
% Las columnas u+3 donde se interpola dependen del canal
cols_u3_RG2 = 4:2:Cols-6;  % Columnas para R y G2 (posición u+3)
cols_u3_G1B = 5:2:Cols-6;  % Columnas para G1 y B (posición u+3)

cols_u_RG2   = cols_u3_RG2 - 3;  % Columnas u
cols_u2_RG2  = cols_u3_RG2 - 1;  % Columnas u+2
cols_u4_RG2  = cols_u3_RG2 + 1;  % Columnas u+4
cols_u6_RG2  = cols_u3_RG2 + 3;  % Columnas u+6

cols_u_G1B   = cols_u3_G1B - 3;  % Columnas u
cols_u2_G1B  = cols_u3_G1B - 1;  % Columnas u+2
cols_u4_G1B  = cols_u3_G1B + 1;  % Columnas u+4
cols_u6_G1B  = cols_u3_G1B + 3;  % Columnas u+6

%% Canal Rojo: interpolar en posición u+3 usando columnas impares
% R original está en filas impares, columnas impares
for i = RowI
    imgRGB(i, cols_u3_RG2, 1) = uint8((-double(imgRGGB(i, cols_u_RG2)) + ...
                                        9*double(imgRGGB(i, cols_u2_RG2)) + ...
                                        9*double(imgRGGB(i, cols_u4_RG2)) - ...
                                        double(imgRGGB(i, cols_u6_RG2))) / 16);
end

%% Canal Verde (G1): interpolar en posición u+3 usando columnas pares
% G1 original está en filas impares, columnas pares
for i = RowI
    imgRGB(i, cols_u3_G1B, 2) = uint8((-double(imgRGGB(i, cols_u_G1B)) + ...
                                        9*double(imgRGGB(i, cols_u2_G1B)) + ...
                                        9*double(imgRGGB(i, cols_u4_G1B)) - ...
                                        double(imgRGGB(i, cols_u6_G1B))) / 16);
end

%% Canal Verde (G2): interpolar en posición u+3 usando columnas impares
% G2 original está en filas pares, columnas impares
for i = RowP
    imgRGB(i, cols_u3_RG2, 2) = uint8((-double(imgRGGB(i, cols_u_RG2)) + ...
                                        9*double(imgRGGB(i, cols_u2_RG2)) + ...
                                        9*double(imgRGGB(i, cols_u4_RG2)) - ...
                                        double(imgRGGB(i, cols_u6_RG2))) / 16);
end

%% Canal Azul: interpolar en posición u+3 usando columnas pares
% B original está en filas pares, columnas pares
for i = RowP
    imgRGB(i, cols_u3_G1B, 3) = uint8((-double(imgRGGB(i, cols_u_G1B)) + ...
                                        9*double(imgRGGB(i, cols_u2_G1B)) + ...
                                        9*double(imgRGGB(i, cols_u4_G1B)) - ...
                                        double(imgRGGB(i, cols_u6_G1B))) / 16);
end

%% INTERPOLACIÓN VERTICAL CON FÓRMULA BICÚBICA - VECTORIZADA
% Fórmula: I(v+3) = (-I(v) + 9*I(v+2) + 9*I(v+4) - I(v+6)) / 16
% Se aplica a posiciones que aún están en cero después de la interpolación horizontal

%% Definir rangos de filas para interpolación vertical
% Para interpolación en posición v+3, necesitamos filas v, v+2, v+4, v+6
% Las filas v+3 donde se interpola son las que cumplen el patrón
rows_v3 = 4:2:Rows-6;  % Filas (posición v+3)

rows_v   = rows_v3 - 3;  % Filas v
rows_v2  = rows_v3 - 1;  % Filas v+2
rows_v4  = rows_v3 + 1;  % Filas v+4
rows_v6  = rows_v3 + 3;  % Filas v+6

%% Completar interpolación vertical para cada canal
for c = 1:3
    % Para cada columna
    for j = 1:Cols
        % Solo procesar posiciones que están en cero
        zero_mask = imgRGB(rows_v3, j, c) == 0;
        if any(zero_mask)
            rows_active = rows_v3(zero_mask);
            v_active    = rows_v(zero_mask);
            v2_active   = rows_v2(zero_mask);
            v4_active   = rows_v4(zero_mask);
            v6_active   = rows_v6(zero_mask);
            
            imgRGB(rows_active, j, c) = uint8((-double(imgRGB(v_active, j, c)) + ...
                                               9*double(imgRGB(v2_active, j, c)) + ...
                                               9*double(imgRGB(v4_active, j, c)) - ...
                                               double(imgRGB(v6_active, j, c))) / 16);
        end
    end
end

%% Visualizar resultados
figure('Position', [100 100 800 400]);
subplot(1,2,1); imshow(imgRGGB); title('Imagen Bayer Original');
subplot(1,2,2); imshow(imgRGB);  title('Interpolación Bicúbica Vectorizada');