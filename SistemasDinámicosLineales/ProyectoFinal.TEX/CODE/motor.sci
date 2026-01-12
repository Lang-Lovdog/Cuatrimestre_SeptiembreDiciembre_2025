// Simulador de un motor electrico

// Modelo en espacio de estado:
// u = Vin
// y = Wm
// x = [iL Vc]^{T}
//
// dx/dt = [0, -1/L; 1/C, -1/RC]x(t) + [1/L; 0]u(t)
//  y(t) = [0, 1]x(t) + 0u(t)

// Limpiar el ambiente
clc;
clear;
close(winsid());


// Parámetros
R  = 1; // Ohms
L  = 1; // Henrios
ke = 1; // constante de fuerza electromotriz
kt = 1; // constante de torque 
B  = 1; // coeficiente de friccion viscosa
J  = 1; //
s  = poly(0,'s');

// Tiempo de simulación
t0 = 0; // Segundos
tf = 20; // Segundos
t  = t0:0.1:tf;
lt = length(t);


// Definir el modelo en espacio de estados
A = [- R/L,     0, -ke/L  ;
         0,     1,     1  ;
      kt/J,     0, - B/J 
    ];
B = [  1/L ;
         0 ;
         0
    ];
C = [0, 0, 1];
D = 0;

sI = [ s, 0, 0 ; 0, s, 0; 0, 0, s ];

MotorDescriptor = syslin('c',A,B,C,D);

G = ss2tf(MotorDescriptor);
disp(G)

/*
// Funciones
// Nombre: odeRLC 
// Objetivo: Definir el sistema de ODE del modelo en espacio de estados del motor
// Entradas: Matrices A y B, y la entrada de control u
// Salida: La definición de la ODE
function dx = odeRLC(t, i, w, teta, A, B, C, u)
    di   =(A*u) + (B*t)+(C*u);
    dw   =(A*u)-(B*t)+(C*u);
    dteta=(A*u)+(B*w)(C*t);
endfunction

// Nombre: evalLMI
// Objetivo: Definir un problema de optimización semidefinida para resolver una LMI
// Entradas: MAtrices que definen la LMI
// Salida: Solución de la LMI

function [LME, LMI, obj] = SDOptProb(xList)
    P = xList(1);
    
    LME = P-P';             // P simétrica
    LMI = list(P-eye(3,3),-((P*A)+(A'*P))-eye(3,3));
    obj = [];
endfunction


// Función de transferencia
Gs = ss2tf(sisLTIRLC);
disp(Gs)

// Simulador

// Condición inicial
x0 = [0; 0; 0];

// Simulación usando csim
//u = ones(1,lt); // Escalón unitario
//[y, x] = csim(u, t, sisLTIRLC, x0);

// Resolver la ODE para obtener x
u = 1;  // Escalón unitario
x = ode(x0, t0, t, list(odeRLC,A,B,C,u));
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
*/
