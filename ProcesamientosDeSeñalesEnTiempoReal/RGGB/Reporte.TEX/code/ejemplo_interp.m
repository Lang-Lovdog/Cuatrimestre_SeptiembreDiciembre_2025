%% EJEMPLO DE USO DE LAS FUNCIONES DE INTERPOLACIÓN
% Este archivo demuestra cómo usar las funciones de interpolación

clear; close all; clc;

%% 1. CREAR IMAGEN SINTÉTICA BAYER (para pruebas)
disp('1. Creando imagen Bayer sintética...');
Rows = 256;
Cols = 256;

% Crear patrón Bayer RGGB
imgBayer = zeros(Rows, Cols, 'uint8');

% Definir índices
RowI = 1:2:Rows;
RowP = 2:2:Rows;
ColI = 1:2:Cols;
ColP = 2:2:Cols;

% Asignar valores sintéticos
imgBayer(RowI, ColI) = 200;  % R
imgBayer(RowI, ColP) = 150;  % G₁
imgBayer(RowP, ColI) = 100;  % G₂
imgBayer(RowP, ColP) = 50;   % B

% Agregar algún patrón
[X, Y] = meshgrid(1:Cols, 1:Rows);
imgBayer = imgBayer + uint8(50 * sin(X/20) .* cos(Y/20));

%% 2. PROBAR INTERPOLACIONES
disp('2. Probando interpolaciones...');

imgRGGB_d = double(imgBayer);

% Interpolación bicuadrada
tic;
img_bicua = interp_bicua(imgRGGB_d, RowI, RowP, ColI, ColP);
time_bicua = toc;

% Interpolación bicúbica
tic;
img_bicub = interp_bicub(imgRGGB_d, RowI, RowP, ColI, ColP);
time_bicub = toc;

%% 3. VISUALIZAR RESULTADOS
disp('3. Visualizando resultados...');

figure('Position', [100 100 1200 300]);

% Imagen Bayer original
subplot(1,3,1);
imshow(imgBayer);
title('Imagen Bayer Original');
xlabel(sprintf('%dx%d', Rows, Cols));

% Interpolación bicuadrada
subplot(1,3,2);
imshow(img_bicua);
title('Interpolación Bicuadrada');
xlabel(sprintf('Tiempo: %.3f s', time_bicua));

% Interpolación bicúbica
subplot(1,3,3);
imshow(img_bicub);
title('Interpolación Bicúbica');
xlabel(sprintf('Tiempo: %.3f s', time_bicub));

%% 4. COMPARAR DIFERENCIAS
disp('4. Calculando diferencias...');

figure('Position', [100 500 800 300]);

% Diferencia entre métodos
diff_img = imabsdiff(img_bicua, img_bicub);

subplot(1,2,1);
imshow(diff_img, []);
title('Diferencia Bicuadrada-Bicúbica');
colorbar;

% Histograma de diferencias
subplot(1,2,2);
histogram(diff_img(:), 50);
title('Histograma de Diferencias');
xlabel('Valor de diferencia');
ylabel('Frecuencia');
grid on;

%% 5. GUARDAR RESULTADOS
disp('5. Guardando resultados de ejemplo...');

% Crear carpeta si no existe
if ~exist('ejemplo_resultados', 'dir')
    mkdir('ejemplo_resultados');
end

% Guardar imágenes
imwrite(imgBayer, 'ejemplo_resultados/bayer_sintetico.jpg');
imwrite(img_bicua, 'ejemplo_resultados/interp_bicuadrada.jpg');
imwrite(img_bicub, 'ejemplo_resultados/interp_bicubica.jpg');
imwrite(diff_img, 'ejemplo_resultados/diferencias.jpg');

disp('=== EJEMPLO COMPLETADO ===');
disp('Resultados guardados en carpeta "ejemplo_resultados/"');