// =============================================================================
// SIMULACIÓN Y ANÁLISIS DE UN MOTOR DE CD EN ESPACIO DE ESTADOS
// Modelo: 3 estados [corriente i, posición θ, velocidad ω]
// Entrada: Voltaje Vin (escalón unitario)
// Salida: Velocidad angular ω
// =============================================================================

// Limpiar ambiente
clc;
clear;
close(winsid());

// -----------------------------------------------------------------------------
// 1. PARÁMETROS DEL MOTOR (valores experimentales de ejemplo)
// -----------------------------------------------------------------------------
R  = 2.1;      // Resistencia (Ω)
L  = 0.005;    // Inductancia (H)
Kt = 0.12;     // Constante de par (N·m/A)
Ke = 0.12;     // Constante fuerza contraelectromotriz (V/(rad/s))
J  = 0.0001;   // Momento de inercia (kg·m²)
B  = 1e-5;     // Coeficiente de fricción viscosa (N·m·s/rad)

// -----------------------------------------------------------------------------
// 2. MATRICES DEL ESPACIO DE ESTADOS
// -----------------------------------------------------------------------------
A = [-R/L,   0, -Ke/L;
        0,   0,     1;
     Kt/J,   0, -B/J];

B = [1/L; 0; 0];
C = [0, 0, 1];
D = 0;

// -----------------------------------------------------------------------------
// 3. TIEMPO DE SIMULACIÓN
// -----------------------------------------------------------------------------
t0 = 0;           // Tiempo inicial (s)
tf = 0.5;         // Tiempo final (s) – suficiente para observar dinámica
dt = 0.001;       // Paso de tiempo (s)
t  = t0:dt:tf;    // Vector de tiempo
nt = length(t);   // Número de puntos

// -----------------------------------------------------------------------------
// 4. SISTEMA LINEAL Y FUNCIÓN DE TRANSFERENCIA
// -----------------------------------------------------------------------------
sisMotor = syslin('c', A, B, C, D);
G = ss2tf(sisMotor);
disp("================================================");
disp("FUNCIÓN DE TRANSFERENCIA G(s):");
disp(G);
disp("================================================");

// -----------------------------------------------------------------------------
// 5. SIMULACIÓN (respuesta al escalón unitario)
// -----------------------------------------------------------------------------
u = ones(1, nt);          // Entrada escalón unitario
[y, x] = csim(u, t, sisMotor);   // y: salida, x: estados (3×nt)

// -----------------------------------------------------------------------------
// 6. ANÁLISIS DE ESTABILIDAD: AUTOVALORES
// -----------------------------------------------------------------------------
lambda = spec(A);
disp("AUTOVALORES de la matriz A:");
disp("λ₁ = " + string(lambda(1)));
disp("λ₂ = " + string(lambda(2)));
disp("λ₃ = " + string(lambda(3)));
disp("================================================");

// -----------------------------------------------------------------------------
// 7. CONTROLABILIDAD
// -----------------------------------------------------------------------------
Co = cont_mat(A, B);
rCo = rank(Co);
disp("MATRIZ DE CONTROLABILIDAD (Co):");
disp(Co);
disp("Rango de Co: " + string(rCo));
if rCo == size(A,1) then
    disp("Sistema controlable (rango completo).");
else
    disp("Sistema NO controlable.");
end
disp("================================================");

// -----------------------------------------------------------------------------
// 8. OBSERVABILIDAD
// -----------------------------------------------------------------------------
Ob = obsv_mat(A, C);
rOb = rank(Ob);
disp("MATRIZ DE OBSERVABILIDAD (Ob):");
disp(Ob);
disp("Rango de Ob: " + string(rOb));
if rOb == size(A,1) then
    disp("Sistema observable (rango completo).");
else
    disp("Sistema NO observable (posición θ no observable).");
end
disp("================================================");

// -----------------------------------------------------------------------------
// 9. DIAGRAMA DE BODE (opcional: comentar si no se necesita)
// -----------------------------------------------------------------------------
figure(1);
bode(sisMotor);
title("Diagrama de Bode del motor de CD");

// -----------------------------------------------------------------------------
// 10. GRÁFICAS DE ESTADOS Y SALIDA
// -----------------------------------------------------------------------------
figure(2);
subplot(3,1,1);
plot(t, x(1,:), 'b');
ylabel("Corriente i (A)");
title("Estados del motor DC – Respuesta al escalón");
subplot(3,1,2);
plot(t, x(2,:), 'r');
ylabel("Posición θ (rad)");
subplot(3,1,3);
plot(t, x(3,:), 'g');
xlabel("Tiempo (s)");
ylabel("Velocidad ω (rad/s)");

figure(3);
plot(t, y, 'k');
xlabel("Tiempo (s)");
ylabel("Velocidad ω (rad/s)");
title("Salida del motor DC (ω)");

// -----------------------------------------------------------------------------
// 11. EXPORTACIÓN DE DATOS (para el informe)
// -----------------------------------------------------------------------------
// Los resultados ya se muestran en consola con disp().
// Para exportar a archivos de texto (opcional):
// write("autovalores.txt", string(lambda));
// write("controlabilidad.txt", string(Co));
// write("observabilidad.txt", string(Ob));

disp("Simulación completada. Revise las figuras.");
