//SIMULACION DE UN MOTOR ELECTRICO DE CD
//Israel Alejandro Ramirez Vazquez

//Limpiar el Ambiente
clc;
clear;
close(winsid());

//Funcion ODE
function dx = DCMode(t,x,A,B,u)
    dx = A*x + B*u;
endfunction

//Parametros del Modelo
Ra = 1;//ohms
La = 1;//henries
Kb = 1;//volts
Ki = 1;//N*m/A
Jm = 1;//kg/m^2
Bm = 1;//0.1;//N*m*s

s = %s;//variable compleja

//PID
Kp = 0.5;//proporcion
Ti = 1;//integrador
Td = 2;//derivativo

//Frecuencia natural (wn)
wn2 = (Kb*Ki+Ra*Bm)/(La*Bm);//
wn = sqrt(wn2);
disp('Frecuencia natural wn: ',wn)

//Factor de amortiguamiento (xi)
xiwn = (Ra*Jm+Bm*La)/(2*La*Jm); 
xi = xiwn/wn;
disp('Factor de amortiguamiento xi: ',xi)

//Matrices Lineales del SSM
A = [-Ra/La,-Kb/La; Ki/Jm,-Bm/Jm];
B = [1/La;0];
C = [0,1];
D = 0;

//Vector de Tiempo
t0 = 0; //(s)
tf = 12; //(s)
Dt = 0.01; //(s)
t = t0:Dt:tf; //vector
lt = length(t); //longitud del vector

//Entrada
u = 1;//escalon
//u(1) = 1;//impulso
/*
A = 12;//amplitud
f = 60;//frecuencia (Hz)
w = f*2*%pi;//f. angular
H = syslin('c', A*w, s^2 + w^2);
u = A*sin(w*t);
disp(H)
*/

//Entrada variante de referencia
ref = ones(1, lt);
for k=1:lt
    if t(k)>0 & t(k)<=4 then
        ref(k) = 3*ref(k);
    elseif t(k)>4 & t(k)<=8 then
        ref(k) = 9*ref(k);
    else
        ref(k) = 6*ref(k);
end

//Error
err = zeros(1,lt);

//Entrada de control
//u = zeros(1, lt);


/*Solucion de la ODE
x0 = [0;1];//condicion inicial
//x0 = zeros(1, lt);

for k=1:lt-1//por instantes de tiempo
    xOut = ode("stiff",x0,t(k),t(k+1),list(DCMode,A,B,u(k)));
    x(:,k+1) = xOut(:,$);
    x0 = x(:,k+1);
    
    err = ref(k+1) - xOut(k+1);//calculo del error
    
    u(k+1) = (Kp*err(k+1)) + ((Kp/Ti)*inttrap(t(1:k+1),err(1:k+1))) + (Kp*Td*((err(k+1)-err(k))/Dt));
end

yN = C*x;*/


//Soluciones Analiticas
yA = zeros(1,lt);
if xi<1 then //caso 1
    yA = 1-((exp(-xi*wn*t)/sqrt(1-xi^2)).*sin((wn*sqrt(xi^2)*t)+acos(xi)));  
elseif xi==1 then //caso 2
    yA = 1-(exp(-wn*t).*(1+wn*t));
else //caso 3
    s1 = (xi*wn) + (wn*sqrt((xi^2)-1));
    s2 = (xi*wn) - (wn*sqrt((xi^2)-1));
    yA = 1-((wn/(2*sqrt((xi^2)-1)))*((exp(-s1*t)/s1)-(exp(-s2*t)/s2)));
end

//Solucion usando csim
ssDCM = syslin('c',A,B,C,D);//SSM
G = ss2tf(ssDCM);//FT
disp('Función de Transferencia G(s): ',G)
yS = csim('step',t,G);

//FT Lazo Cerrado
x = Ki*Kp*Td;
y = Ki*Kp;
a = La*Bm;
b = x + Ra*Jm + Bm*La;
c = y + Kb*Ki + Ra*Bm;
d = y/Ti;
CL = syslin('c', x*s^2 + y*s + d, a*s^3 + b*s^2 + c*s + d);
disp("CL =", CL)

//bode(CL)//diagrama de Bode
//roots(CL)//polos
//evans(CL)//g. polos
//plzr(CL)//g. polos y ceros


//Graficas
figure(1);//entrada
title('Entrada u(t)');
plot(t,u,'-m');
xlabel("Tiempo (s)");
ylabel("Voltaje (V)");

/*figure(2);//salidas
title("Salidas y(t)");
plot(t,yN,'-b');//, t, yA,'-r',t,yS,'--k');
xlabel('Tiempo (s)');
ylabel('Velocidad angular (rad/s)');
legend("Solución numérica","Solución analítica","Solución con csim");*/

figure(3);//entrada vs. salida
title('Respuesta a una entrada variante');
plot(t,u,'-b');//,t,yN,'-r');
xlabel("Tiempo (s)");
ylabel("Voltaje (V) / Velocidad (rad/s)");
legend("Entrada","Salida");

end
