%% INTERPOLACIÓN POLINOMIAL PARA PATRÓN BAYER
% Autor: Dr. Mario Alberto Ibarra Manzano
% Universidad de Guanajuato
% Fecha: 2024

clc; clear; close all;

%% CONFIGURACIÓN INICIAL
disp('=== INTERPOLACIÓN POLINOMIAL PARA PATRÓN BAYER ===');
disp('1. Cargando imagen...');

% Seleccionar imagen
[file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Imágenes'}, ...
    'Seleccione imagen Bayer RGGB');
if isequal(file, 0)
    error('No se seleccionó ninguna imagen.');
end

% Cargar imagen
imgRGGB = imread(fullfile(path, file));
[Rows, Cols] = size(imgRGGB);
fprintf('   Tamaño: %dx%d píxeles\n', Rows, Cols);

% Verificar que las dimensiones sean pares
if mod(Rows,2) ~= 0 || mod(Cols,2) ~= 0
    error('La imagen debe tener dimensiones pares para patrón Bayer.');
end

%% PREPROCESAMIENTO
disp('2. Preprocesando imagen...');
imgRGGB_d = double(imgRGGB);

% Definir índices de paridad
RowI = 1:2:Rows;  % Filas impares
RowP = 2:2:Rows;  % Filas pares
ColI = 1:2:Cols;  % Columnas impares
ColP = 2:2:Cols;  % Columnas pares

%% INTERPOLACIÓN LINEAL (REFERENCIA)
disp('3. Ejecutando interpolación lineal...');
tic;
imgRGB_lin = zeros(Rows, Cols, 3, 'uint8');
imgRGB_lin(RowI, ColI, 1) = imgRGGB_d(RowI, ColI);
imgRGB_lin(RowI, ColP, 2) = imgRGGB_d(RowI, ColP);
imgRGB_lin(RowP, ColI, 2) = imgRGGB_d(RowP, ColI);
imgRGB_lin(RowP, ColP, 3) = imgRGGB_d(RowP, ColP);

% Interpolación lineal (ya implementada en el código original)
imgRGB_lin = interp_lineal(imgRGGB_d, RowI, RowP, ColI, ColP);
t_lin = toc;
fprintf('   Tiempo: %.3f segundos\n', t_lin);

%% INTERPOLACIÓN BICUADRADA
disp('4. Ejecutando interpolación bicuadrada...');
tic;
imgRGB_bicua = interp_bicua(imgRGGB_d, RowI, RowP, ColI, ColP);
t_bicua = toc;
fprintf('   Tiempo: %.3f segundos\n', t_bicua);

%% INTERPOLACIÓN BICÚBICA
disp('5. Ejecutando interpolación bicúbica...');
tic;
imgRGB_bicub = interp_bicub(imgRGGB_d, RowI, RowP, ColI, ColP);
t_bicub = toc;
fprintf('   Tiempo: %.3f segundos\n', t_bicub);

%% ANÁLISIS DE CALIDAD
disp('6. Calculando métricas de calidad...');

% Calcular MSE
mse_lin = mean((double(imgRGB_lin(:)) - double(imgRGB_lin(:))).^2);
mse_bicua = mean((double(imgRGB_lin(:)) - double(imgRGB_bicua(:))).^2);
mse_bicub = mean((double(imgRGB_lin(:)) - double(imgRGB_bicub(:))).^2);

% Calcular PSNR
psnr_lin = 10*log10(255^2/mse_lin);
psnr_bicua = 10*log10(255^2/mse_bicua);
psnr_bicub = 10*log10(255^2/mse_bicub);

fprintf('\n=== RESULTADOS ===\n');
fprintf('Método         MSE       PSNR (dB)   Tiempo (s)\n');
fprintf('---------------------------------------------\n');
fprintf('Lineal      %8.2f   %8.2f   %8.3f\n', mse_lin, psnr_lin, t_lin);
fprintf('Bicuadrada  %8.2f   %8.2f   %8.3f\n', mse_bicua, psnr_bicua, t_bicua);
fprintf('Bicúbica    %8.2f   %8.2f   %8.3f\n', mse_bicub, psnr_bicub, t_bicub);

%% VISUALIZACIÓN
disp('7. Generando visualizaciones...');

figure('Name', 'Comparación de Métodos', 'Position', [100 100 1200 400]);

% Imagen original Bayer
subplot(2,4,1);
imshow(imgRGGB);
title('Original Bayer');
xlabel(sprintf('%dx%d', Rows, Cols));

% Métodos de interpolación
subplot(2,4,2);
imshow(imgRGB_lin);
title('Interpolación Lineal');
xlabel(sprintf('PSNR: %.2f dB', psnr_lin));

subplot(2,4,3);
imshow(imgRGB_bicua);
title('Interpolación Bicuadrada');
xlabel(sprintf('PSNR: %.2f dB', psnr_bicua));

subplot(2,4,4);
imshow(imgRGB_bicub);
title('Interpolación Bicúbica');
xlabel(sprintf('PSNR: %.2f dB', psnr_bicub));

% Diferencias
subplot(2,4,6);
imshow(imabsdiff(imgRGB_lin, imgRGB_bicua));
title('Diferencia Lineal-Bicuadrada');
xlabel(sprintf('MSE: %.2f', mse_bicua));

subplot(2,4,7);
imshow(imabsdiff(imgRGB_lin, imgRGB_bicub));
title('Diferencia Lineal-Bicúbica');
xlabel(sprintf('MSE: %.2f', mse_bicub));

% Gráfico de tiempos
subplot(2,4,8);
bar([t_lin, t_bicua, t_bicub]);
set(gca, 'XTickLabel', {'Lineal', 'Bicuad.', 'Bicúb.'});
ylabel('Tiempo (s)');
title('Comparación de Tiempos');
grid on;

%% GUARDAR RESULTADOS
disp('8. Guardando resultados...');

% Crear carpeta de resultados
if ~exist('resultados', 'dir')
    mkdir('resultados');
end

% Guardar imágenes
imwrite(imgRGB_lin, 'resultados/lineal.jpg', 'Quality', 95);
imwrite(imgRGB_bicua, 'resultados/bicuadrada.jpg', 'Quality', 95);
imwrite(imgRGB_bicub, 'resultados/bicubica.jpg', 'Quality', 95);

% Guardar métricas en archivo
fid = fopen('resultados/metricas.txt', 'w');
fprintf(fid, '=== MÉTRICAS DE CALIDAD ===\n\n');
fprintf(fid, 'Fecha: %s\n', datestr(now));
fprintf(fid, 'Imagen: %s\n', file);
fprintf(fid, 'Dimensiones: %dx%d\n\n', Rows, Cols);
fprintf(fid, 'Método         MSE       PSNR (dB)   Tiempo (s)\n');
fprintf(fid, '---------------------------------------------\n');
fprintf(fid, 'Lineal      %8.2f   %8.2f   %8.3f\n', mse_lin, psnr_lin, t_lin);
fprintf(fid, 'Bicuadrada  %8.2f   %8.2f   %8.3f\n', mse_bicua, psnr_bicua, t_bicua);
fprintf(fid, 'Bicúbica    %8.2f   %8.2f   %8.3f\n', mse_bicub, psnr_bicub, t_bicub);
fclose(fid);

disp('=== PROCESO COMPLETADO ===');
disp('Resultados guardados en carpeta "resultados/"');