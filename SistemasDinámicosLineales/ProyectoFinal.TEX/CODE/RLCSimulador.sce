// Simulador de una red RLC

// Modelo en espacio de estado:
// u = Vin
// y = Vout
// x = [iL Vc]^{T}
//
// dx/dt = [0, -1/L; 1/C, -1/RC]x(t) + [1/L; 0]u(t)
//  y(t) = [0, 1]x(t) + 0u(t)

// Limpiar el ambiente
clc;
clear;
close(winsid());


// Parámetros
R = 1; // Ohms
L = 1; // Henrios
C = 1; // Faradios

// Tiempo de simulación
t0 = 0; // Segundos
tf = 20; // Segundos
t = t0:0.1:tf;
lt = length(t);


// Definir el modelo en espacio de estados
A = [0, -1/L; 1/C, -1/(R*C)];
B = [1/L; 0];
C = [0, 1];
D = 0;

sisLTIRLC = syslin('c',A,B,C,D);


// Funciones
// Nombre: odeRLC
// Objetivo: Definir el sistema de ODE del modelo en espacio de estados de la red RLC
// Entradas: Matrices A y B, y la entrada de control u
// Salida: La definición de la ODE
function dx = odeRLC(t, x, A, B, u)
    dx = (A*x) + (B*u);
endfunction

// Nombre: evalLMI
// Objetivo: Definir un problema de optimización semidefinida para resolver una LMI
// Entradas: MAtrices que definen la LMI
// Salida: Solución de la LMI
function [LME, LMI, obj] = SDOptProb(xList)
    P = xList(1);
    
    LME = P-P';             // P simétrica
    LMI = list(P-eye(2,2),-((P*A)+(A'*P))-eye(2,2));
    obj = [];
endfunction


// Función de transferencia
Gs = ss2tf(sisLTIRLC);
disp(Gs)

// Simulador

// Condición inicial
x0 = [0; 0];

// Simulación usando csim
//u = ones(1,lt); // Escalón unitario
//[y, x] = csim(u, t, sisLTIRLC, x0);

// Resolver la ODE para obtener x
u = 1;  // Escalón unitario
x = ode(x0, t0, t, list(odeRLC,A,B,u));
// Resolver las ecuaciones algebraicas para obtener y
y = (C*x) + (D*u);

// Diagrama de Bode
figure;
bode(sisLTIRLC);


// Análisis de estabilidad
// Calcular los valores propios de la matriz dinámica
eigA = spec(A);
disp(eigA);

// Graficar los valores propios
figure;
plzr(Gs);


// Resolver LMI para comprobar la estabilidad asintótica.
P0 = zeros(2,2);
solProbOpt = lmisolver(list(P0),SDOptProb);
P = solProbOpt(1);
disp(P);
eigP = spec(P);
disp(eigP);


//Controlabilidad
Co = cont_mat(A,B);
rangoCo = rank(Co);
disp(Co);
disp(rangoCo);

// Observabilidad
O = obsv_mat(A,C);
rangoO = rank(O);
disp(O);
disp(rangoO);


// Graficar los resultados
figure;
plot(t,x(1,:),'-b',t,x(2,:),'-k');
xlabel('Time (s)');
ylabel('State vector');
legend('iL (Amp)','Vc (Volts)');

figure;
plot(t,y,'-r');
xlabel('Time (s)');
ylabel('Vout (Volts)');
