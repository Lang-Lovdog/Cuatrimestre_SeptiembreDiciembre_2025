function imgRGB = interp_bicua(imgRGGB_d, RowI, RowP, ColI, ColP)
%% INTERPOLACIÓN BICUADRADA PARA PATRÓN BAYER RGGB
% Entrada:
%   imgRGGB_d - Imagen Bayer en formato double
%   RowI, RowP - Índices de filas impares y pares
%   ColI, ColP - Índices de columnas impares y pares
% Salida:
%   imgRGB - Imagen RGB interpolada (uint8)

[Rows, Cols] = size(imgRGGB_d);
imgRGB = zeros(Rows, Cols, 3, 'uint8');

% Coeficientes bicuadrados (3 puntos)
c1 = 3/8;  % Peso para I(u)
c2 = 3/4;  % Peso para I(u+2)
c3 = -1/8; % Peso para I(u+4)

% 1. ASIGNAR PÍXELES ORIGINALES
imgRGB(RowI, ColI, 1) = imgRGGB_d(RowI, ColI);  % R
imgRGB(RowI, ColP, 2) = imgRGGB_d(RowI, ColP);  % G₁
imgRGB(RowP, ColI, 2) = imgRGGB_d(RowP, ColI);  % G₂
imgRGB(RowP, ColP, 3) = imgRGGB_d(RowP, ColP);  % B

% 2. INTERPOLACIÓN VERTICAL (ROJO)
% R en filas pares, columnas impares usando vecindad vertical
k_rv = 2:2:Rows-2;  % Filas pares con vecinos suficientes
imgRGB(k_rv, ColI, 1) = uint8(...
    c1 * imgRGGB_d(k_rv-1, ColI) + ...
    c2 * imgRGGB_d(k_rv+1, ColI) + ...
    c3 * imgRGGB_d(k_rv+3, ColI));

% 3. INTERPOLACIÓN HORIZONTAL (ROJO)
% R en todas las filas, columnas pares usando valores ya calculados
l_rh = 2:2:Cols-2;  % Columnas pares con vecinos
imgRGB(:, l_rh, 1) = uint8(...
    c1 * double(imgRGB(:, l_rh-1, 1)) + ...
    c2 * double(imgRGB(:, l_rh+1, 1)) + ...
    c3 * double(imgRGB(:, l_rh+3, 1)));

% 4. INTERPOLACIÓN VERTICAL (VERDE)
% Subred G₁: Verde en filas impares ≥ 3, columnas pares
k_g1 = 3:2:Rows-3;
imgRGB(k_g1, ColP, 2) = uint8(...
    c1 * imgRGGB_d(k_g1-1, ColP) + ...
    c2 * imgRGGB_d(k_g1+1, ColP) + ...
    c3 * imgRGGB_d(k_g1+3, ColP));

% Subred G₂: Verde en filas pares, columnas impares
k_g2 = 2:2:Rows-2;
imgRGB(k_g2, ColI, 2) = uint8(...
    c1 * imgRGGB_d(k_g2-1, ColI) + ...
    c2 * imgRGGB_d(k_g2+1, ColI) + ...
    c3 * imgRGGB_d(k_g2+3, ColI));

% 5. INTERPOLACIÓN VERTICAL (AZUL)
% B en filas impares ≥ 3, columnas pares
k_bv = 3:2:Rows-3;
imgRGB(k_bv, ColP, 3) = uint8(...
    c1 * imgRGGB_d(k_bv-1, ColP) + ...
    c2 * imgRGGB_d(k_bv+1, ColP) + ...
    c3 * imgRGGB_d(k_bv+3, ColP));

% 6. INTERPOLACIÓN HORIZONTAL (AZUL)
% B en todas las filas, columnas impares ≥ 3
l_bh = 3:2:Cols-3;
imgRGB(:, l_bh, 3) = uint8(...
    c1 * double(imgRGB(:, l_bh-1, 3)) + ...
    c2 * double(imgRGB(:, l_bh+1, 3)) + ...
    c3 * double(imgRGB(:, l_bh+3, 3)));

end