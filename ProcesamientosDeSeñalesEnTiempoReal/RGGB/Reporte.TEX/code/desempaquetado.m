function RGB_IMG = desempaquetado(img)
   
    [Rows, Cols] = size(imgRGGB);
   
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

    % Rojo, es la primera posición de cada submatriz de 2x2, por tanto
    % todos los índices impar,impar
    MatrixR(RowI, ColI) = imgRGGB(RowI, ColI);

    % El verde es el segundo y tercer elemento de cada submatriz de 2x2,
    % por tanto todos los índices par,impar e impar,par
    % El verde es promediado debido a su posicion en la matriz
    MatrixG(RowI, ColI) = (imgRGGB(RowI, ColP)+imgRGGB(RowP, ColI))/2;

    % El azul es el cuarto elemento de cada submatriz de 2x2, por tanto
    % todos los índices par,par
    MatrixB(RowI, ColI) = imgRGGB(RowP, ColP);

    % Al final, la matriz queda con las mismas dimensiones que la de la
    % imagen original y con los mismos pixeles que en la imagen original
    % pero ahora con los colores por canal.
    % Aquí cabe resaltar que cada sumatrix de 2x2, tendrá únicamente el
    % primer pixel activo. El resto serán ceros.
    RGB_IMG = cat(3, MatrixR, MatrixG, MatrixB);

end
