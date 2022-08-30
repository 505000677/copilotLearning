  

function S = cspline_eval(t,y,z,x_vec)
% function S = cspline_eval(t,y,z,x_vec)
% compute the value of the natural cubic spline at the points x_vec when
% t,y,z are given
%
% Example:   t = [0.9,1.3,1.9,2.1];
%            y = [1.3,1.5,1.85,2.1]
%            z = cspline(t,y)
%            x = [0.9:0.1:2.1]
%            v = cspline_eval(t,y,z,x)

m = length(x_vec);
S = zeros(size(x_vec));  
n = length(t);
for j=1:m
  x = x_vec(j);
  for i=n-1:-1:1
    if (x-t(i)) >= 0
      break
    end
  end
  h = t(i+1)-t(i);
  S(j) = z(i+1)/(6*h)*(x-t(i))^3-z(i)/(6*h)*(x-t(i+1))^3 ...
       +(y(i+1)/h-z(i+1)*h/6)*(x-t(i)) - (y(i)/h-z(i)*h/6)*(x-t(i+1));
end


function A=divdiff(x,y)
% input: x,y: the data set to be interpolated
% output: A: table for Newton’s divided differences.
    [~,n]=size(x);
    A=zeros(n,n);
    A(:,1)=y';
% for k from 2 to n
%   for i from k to n
%       Compute the value A(i,k) according to the divided difference formula
%   end
% end
    for k=2:n
        for i=k:n %x(i-k+1) is for x_0
            A(i,k)=( (A(i,k-1)-A(i-1,k-1))/( x(i)-x(i-k+1) ) );
        end
    end
end

function x=GaussianX(n,d,a,b)
  % function x=GaussianX(n,d,a,b)
  % input: n: system size, must be odd
  %        (d,a,b): vectors of length n
  % output: x=solution
  x=sparse(zeros(1,9));
  i = (n + 1) / 2;
  x(i) = b(i) / d(i);

function df=dfasd(x0)
    df=2*x0;
end

function g=nlshooting(t,x)
%x(1)=y,x(2)=y'
g=zeros(size(x));
g(1)=x(2);
g(2)=-(x(2))^2-x(1)+cos(t)*cos(t);
end

% calculate the right-hand side
for i=1:nt-2
dfd(i) = c(i) - 2*c(i+1) + c(i+2);
end
dfd = dfd/delx2;

function fr=frn(x)
    fr=3./(x+1);
end

% put result in the ydot
ydot = dfd';

% transfer the y to the c
for i=1:nt-2
c(i+1) = y(i);
end

clear;
% right hand equation
f=@(x) cos(x);
%Test for N=20
N=20;
a=0;b=pi/2;
h=(b-a)/N;x=a:h:b;
%Matrix A:
A(1,1)=(-2/(h)^2)-2;
A(1,2)=(1/(h)^2)-(1/(2*h));
for i=2:N-2
    A(i,i-1)=(1/(h)^2)+(1/(2*h));
    A(i,i)=(-2/(h)^2)-2;
    A(i,i+1)=(1/(h)^2)-(1/(2*h));
end
A(N-1,N-2)=(1/(h^2))+(1/(2*h));
A(N-1,N-1)=(-2/(h)^2)-2;
%Matrix B:
%y0=alpha,yN=beta
y0=-0.3;yN=-0.1;
b(1)=f(x(1))-y0*((1/(h^2))+(1/(2*h)));
for i=2:N-2
    b(i)=f(x(i+1));
end
b(N-1)=f(x(N-1))-yN*((1/(h^2))-(1/(2*h)));
%solve for matrix xsolution
xs=A\b';
y(1)=y0;
for i=2:N
    y(i)=xs(i-1);
end
y(N+1)=yN;
figure(1),plot(x,y,'r');
print -dpng hw10_4_N20_my.png;
%exact solution
yex=-(1/10)*(sin(x)+3*cos(x));
error=abs(y-yex);
figure(2),plot(x,error,'b');
print -dpng hw10_4_N20_error.png;

n = length(t);
z = zeros(n,1);
h = zeros(n-1,1);
b = zeros(n-1,1);
u = zeros(n,1);
v = zeros(n,1);

function x=mynewton(f,df,x0,tol,nmax)
% input variables:
% f,df are the function f and its derivative f′, 
% x0 is the initial guess, 
% tol is the error tolerance, 
% and nmax is the maximum number of iterations. 
% Theoutput variable: 
% x is the result of the Newton iterations. 
    xx=x0;
    n=0;
    dx=f(xx)/df(xx);
    while( (dx>tol)||(f(xx)>tol))&&(n<nmax)
        n=n+1;
        xx=xx-dx;
        dx=f(xx)/df(xx);
        fprintf('I have n=%d and xx=%g.\n',n,xx);
    end
    x=xx-dx;
    fprintf('I have x=%f.\n',x);
end

% fd_rhs.m
function ydot=fd_rhs(time,y)
global nt delx2

  for j = 1:i-1
    sol = [d(j) a(n-j+1); a(j) d(n-j+1)] \ [b(j); b(n-j+1)];
    %forward elimination:
    x(j) = sol(1);
    %Backward elimination:
    x(n-j+1) = sol(2);
  end
end

function f = funItg(x)
    f=exp(-x);
end

for i=n-1:-1:2
  z(i) = (v(i)-h(i)*z(i+1))/u(i);
end


function R = romberg(fr,a,b,n)
%where f is the name of the function where f(x) is implemented, 
% and a and b definesthe integrating interval, 
% and n is the size of your Romberg table. 
% The functionshould return the whole Romberg table. 
% The best approximation of the interval would be the value in R(n,n).
R=zeros(n,n);
h=b-a;
R(1,1)=((feval(fr,a)+feval(fr,b))*h/2);
for i=1:n-1
    R(i+1,1)=R(i,1)/2;
    h=h/2;
    for k=1:2^(i-1)
        R(i+1,1)=R(i+1,1)+h*feval(fr,(a+(2*k-1)*h));
    end
end
for j=2:n
    for i=j:n
        R(i,j)=R(i,j-1)+(1/(4^(j-1)-1))*(R(i,j-1)-R(i-1,j-1));
    end
end
end

function x=mysecant(f,x0,x1,tol,nmax)
% Here f is the function f, 
% x0,x1 are the initial guesses, 
% tol is the error tolerance, 
% nmax is the maximum number of iterations, 
% x is the output of your function.
    xx=x0;
    xxx=x1;
    n=0;
    dx=(f(xxx)-f(xx))/(xxx-xx);
    while( (dx>tol)||(f(xx)>tol)||f(xxx)>tol )&&(n<nmax)
        n=n+1;
        xx=xxx;
        xxx=xxx-((f(xxx))/dx);
        dx=(f(xxx)-f(xx))/(xxx-xx);
        fprintf('I have n=%d and xx=%g and xxx=%g.\n',n,xx,xxx);
    end
    x=xxx;
    fprintf('I have x=%f.\n',x);
end

clear;
%Test for N=20
N=20;
a=0;b=pi/2;
h=(b-a)/N;x=a:h:b;
% right hand equation
f=@(x) cos(x)*h^2;
%Matrix A:
A(1,1)=(-2-2*h^2);
A(1,2)=(1-h/2);
for i=2:N-2
    A(i,i-1)=(1+h/2);
    A(i,i)=(-2-2*h^2);
    A(i,i+1)=(1-h/2);
end
A(N-1,N-2)=(1+h/2);
A(N-1,N-1)=(-2-2*h^2);
%Matrix B:
%y0=alpha,yN=beta
y0=-0.3;yN=-0.1;
b(1)=f(x(1))-y0*((1+h/2));
for i=2:N-2
    b(i)=f(x(i+1));
end
b(N-1)=f(x(N-1))-yN*(1-h/2);
%solve for matrix xsolution
xs=A\b';
y(1)=y0;
for i=2:N
    y(i)=xs(i-1);
end
y(N+1)=yN;
figure(1),plot(x,y,'r');
print -dpng hw10_4_M_N20_my.png;
%exact solution
yex=-(1/10)*(sin(x)+3*cos(x));
error=abs(y-yex);
figure(2),plot(x,error,'b');
print -dpng hw10_4_M_N20_error.png;

u(2) = 2*(h(1)+h(2));
v(2) = 6*(b(2)-b(1));
for i=3:n-1  % solve the tri-diag system
  u(i) = 2*(h(i)+h(i-1))-h(i-1)^2/u(i-1);
  v(i) = 6*(b(i)-b(i-1))-h(i-1)*v(i-1)/u(i-1);
end

function [y1,y2] = test24(x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
y1=5*x^3-27*x^2+45*x-21;
y2=x^4-5*x^3+8*x^2-5*x+3;
end

%y0 is for first and second shoot guess
y0=[0  ;0;  0;  1];tspan=[0 1];
[t,xa]=ode45(@moreshooting,tspan,y0);
N=length(xa(:,1,:,1));
Lamda=(0-xa(N,3))/(xa(N,1)-xa(N,3));
y1=[0;1-Lamda;0;0];
[t,xs]=ode45(@moreshooting,tspan,y1);
figure(1),plot(t,xs(:,1),'r');
print -dpng hw10_2_my.png;
%for error
y=t.^3-t;
error=abs(xs(:,1)-y);
figure(2),plot(t,error);
print -dpng hw10_2_error.png;
figure(3),plot(t,y);

function [x,nit]=sor(A,b,x0,w,d,tol,nmax)
% SOR : solve linear system with SOR iteration
% Usage: [x,nit]=sor(A,b,x0,omega,d,tol,nmax)
% Inputs:
% A : an n x n-matrix,
% b : the rhs vector, with length n
% x0 : the start vector for the iteration
% tol: error tolerance
% w: relaxation parameter, (1 < w < 2),
% d : band width of A.
% Outputs::
% x : the solution vector
% nit: number of iterations
    x=x0;
    n=length(x);
    nit=0;
    while nit< nmax && norm (A*x-b)>tol
        for i=1:n
            aii=A(i,i);
            eq1=sum(A(i,1:i-1)' .*x(1:i-1));
            eq2=sum(A(i,i+1:n)' .*x(i+1:n));
            equ=eq1+eq2;
            equu=(b(i)-equ)/aii;
            x(i)=x(i)*(1-w)+w*equu;
        end
        nit=nit+1;
    end
end

function v=polyvalue(a,x,t)
% input: a= Newton’s divided differences
% x= the points for the data set to interpolate,
% same as in divdiff.
% t= the points where the polynomial should be evaluated
% output: v= value of polynomial at the points in t
    ai=diag(a);%get a_i
    tl = length(t);
    xl = length(x);
    for i = 1:tl
        Pnx = 1;
        y = ai(1);
        for j = 2:xl
            %Nested form of Newton's polynomial
            Pnx = (t(i)-x(j-1))*Pnx;
            y = y+ai(j)* Pnx;
        end
        v(i) = y;
    end
end

function ls=lspline(t,y,x)
% lspline computes the linear spline
% Inputs:
% t: vector, contains the knots
% y: vector, contains the interpolating values at knots
% x: vector, contains points where the lspline function
% should be evaluated and plotted
% Output:
% ls: vector, contains the values of lspline at points x
m = length(x);
for j=1:m
  xi = x(j);
  n = length(t);
  for i=n-1:-1:1
    if (xi-t(i)) >= 0
      break
    end
  end
  ls(j)=y(i)+(y(i+1)-y(i))/(t(i+1)-t(i))*(xi-t(i))
end
end

clear;
z1=0;z2=1;
tol=abs(z2-z1);
nmax=0;
tspan=[0 pi];
y0=[0;z1];
y1=[0;z2];
while tol>10^(-9) && nmax <6
    [t,xa]=ode45(@nlshooting,tspan,y0);
    [t,xb]=ode45(@nlshooting,tspan,y1);
    N1=length(xa(:,1));
    N2=length(xb(:,1));
    phi=z2+(0-xb(N2,1))*(z2-z1)/(xb(N2,1)-xa(N1,1));
    z1=z2;
    z2=phi;
    tol=abs(z2-z1);
    nmax=nmax+1;
end
y2=[0;phi];
[t,xs]=ode45(@nlshooting,tspan,y2);
figure(1), plot(t,xs(:,1),'r');
hold on;
plot(t,sin(t),'b');
legend('mysol','exactsol');
print -dpng hw10_3_my.png;
hold off;
figure(2),plot(t,abs(xs(:,1)-sin(t)));
print -dpng hw10_3_error.png;

% function test1_7()
%     a=input('coefficient a for the polynomial ax^2+bx+c=0.:');
%     b=input('coefficient b for the polynomial ax^2+bx+c=0.:');
%     c=input('coefficient c for the polynomial ax^2+bx+c=0.:');
%     [r1,r2]=quadroots(a,b,c);
%     fprintf('result for quadroots r1: %.10f and r2: %.20f\n\n',r1,r2);
%     [r1,r2]=smartquadroots(a,b,c);
%     fprintf('result for smartquadroots r1: %.10f and r2: %.20f\n\n',r1,r2);
% end
function [r1,r2]=quadroots(a,b,c)
% input: a, b, c: coefficients for the polynomial ax^2+bx+c=0.
% output: r1, r2: The two roots for the polynomial.
    r1 = (-b + sqrt(b^2 - 4 * a * c))/(2*a);
    r2 = (-b - sqrt(b^2 - 4 * a * c))/(2*a);
end
% function [r1,r2]=smartquadroots(a,b,c)
% % input: a, b, c: coefficients for the polynomial ax^2+bx+c=0.
% % output: r1, r2: The two roots for the polynomial.
%     r1 = (-b + sqrt(b^2 - 4 * a * c))/(2*a);
%     r2 = c/(a*r1);
% end

function v = trapezoid(f, a, b, n)
    % where funItg.m is the name of the file of the function f(x), 
    % and a,b is the interval, 
    % and n is the number of sub-intervals
    h = (b - a) / n;
    x = a + h:h:b - h;
    v = ((feval(f, a) + feval(f, b)) / 2 + sum(feval(f, x))) * h;
end

end

function g=moreshooting(t,x)
%x(1)=(\bar)u,x(2)=(\bar)u'
%x(3)=(\tilde)u,x(4)=(\tilde)u'
g=zeros(size(x));
g(1)=x(2);
g(2)=6*t^3-6*x(1);
g(3)=x(4);
g(4)=6*t^3-6*x(3);
end

function [x,nit]=jacobi(A,b,x0,tol,nmax)
% Inputs:
% A : an n x n-matrix,i
% b : the rhs vector, with length n
% x0 : the start vector for the iteration
% tol: error tolerance
% Outputs::
% x : the solution vector
% nit: number of iterations
  x = x0;
  n = length(x);
  nit = 0;
  while nit < nmax && norm(A * x - b) > tol
    equ1 = zeros(n, 1);
    for i = 1:n
      aii = A(i, i);
      equu = sum(A(i, 1:n)' .* x) - aii * x(i);
      equ1(i) = (b(i) - equu) / aii;
    end
    x = equ1;
    nit = nit + 1;
  end

function [t,x] = rk4(f,t0,x0,tend,N)
% f : Differential equation xp = f(t,x)
% x0 : initial condition
% t0,tend : initial and final time
% N : number of time steps
h = (tend-t0)/N;
t = [t0:h:tend];
s = length(x0); % x0 can be a vector
x = zeros(s,N+1);
x(:,1) = x0;
for n = 1:N
k1 = feval(f,t(n),x(:,n));
k2 = feval(f,t(n)+0.5*h,x(:,n)+0.5*h*k1);
k3 = feval(f,t(n)+0.5*h,x(:,n)+0.5*h*k2);
k4 = feval(f,t(n)+h,x(:,n)+h*k3);
x(:,n+1) = x(:,n) + h/6*(k1+2*(k2+k3)+k4);
end

%yo is for first shoot guess,y1 is for the second one
y0=[-0.3;0];tspan=[0 pi/2];
[t,xa]=ode45(@shooting,tspan,y0);
y1=[-0.3;1];
[t,xb]=ode45(@shooting,tspan,y1);
N1=length(xa(:,1));
N2=length(xb(:,1));
%line 9
Lamda=(-0.1-xb(N2,1))/(xa(N1,1)-xb(N2,1));
y2=[-0.3;1-Lamda];
[t,xs]=ode45(@shooting,tspan,y2);

function fff=fffasd(x0)
    fff=x0^2-sqrt(2);
end

h = t(2:n)-t(1:n-1);
b = (y(2:n)-y(1:n-1))./h;

function z = cspline(t,y)
% function z = cspline(t,y)
% compute z-coefficients for natural cubic spline
% where t is a vector with knots, and y is the interpolating values
% z is the output vector

%for error
figure(1),plot(t,xs(:,1),'r');
print -dpng hw10_1_my.png;
y=-(1/10)*(sin(t)+3*cos(t));
error=abs(xs(:,1)-y);
figure(2),plot(t,error);
print -dpng hw10_1_error.png;
figure(3),plot(t,y);


function ff = funItg47(x)
    if x == 0
        ff = 1;
    else
        ff = sin(x) ./ x;
    end
end

% function test()
%     a=input('Enter a: vector:');
%     b=input('Enter b: vector(same length as a):');
%     v=myValue(a,b);
%     fprintf('output: v: the computed value %.4f',v);
% end
function v=myValue(a,b)
% input: a: vector
% b: vector (same length as a)
% output: v: the computed value
    x=0;
    for i=1:length(a)
        for j=1:i
            x=x+b(i)*b(i)*a(j);
        end
    end
    v=x;
end

00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000function r=bisection(f,a,b,tol,nmax)
% function r=bisection(f,a,b,tol,nmax)
% inputs: f: function handle or string
% a,b: the interval where there is a root
% tol: error tolerance
% nmax: max number of iterations
% output: r: a root
    if f(a)*f(b)>0 
        disp('f(a)*f(b) not less than zero.')
        r='r not found';
    else
        r = (a + b)/2;
        err = abs(f(r));
        iter=0;
        while err > tol && iter<=nmax &&(b-a)>tol
             if f(a)*f(r)<0 
                 b = r;
             else
                a = r;          
             end
            r = (a + b)/2; 
            err = abs(f(r));
            iter=iter+1;
        end
    end
end


function [r1,r2]=smartquadroots(a,b,c)
% input: a, b, c: coefficients for the polynomial ax^2+bx+c=0.
% output: r1, r2: The two roots for the polynomial.
    r1 = (-b + sqrt(b^2 - 4 * a * c))/(2*a);
    r2 = c/(a*r1);
end

%phi 1 phi 2 depend on z1 and z2
%z1 and z2 is intial guess
%tspan 是time
%[t,xa]=ode45(@funtion,tspan,z1);
%phi1=xa[N1,1]

function g=shooting(t,x)
    %X(1)=y,x(2)=y'
    g=zeros(size(x));%intial the and create zeros of size of x
    g(1)=x(2);
    g(2)=x(2)+2*x(1)+cos(t);
    %[t,xa]=ode45(@shooting,tspan,y0);
%     Error using odearguments
% SHOOTING must return a column vector.
% 
% Error in ode45 (line 107)
%   odearguments(odeIsFuncHandle,odeTreatAsMFile, solver_name, ode, tspan, y0, options, varargin);
%  
% y0=[-0.3;-0.1];tspan=[0;pi/2];
%
% y0=[-0.3;0];tspan=[0 pi/2];
% y1=[]
% [t,xa]=ode45(@shooting,tspan,y0);
% [t,xa]=ode45(@shooting,tspan,y0)
end

% add in the boundary conditions
c(1) = 1;
c(nt) = 0;

for k=1:n-1
  for i=k+1:n
    xmult = A(i,k)/A(k,k);
    A(i,k) = xmult;
    for j=k+1:n
      A(i,j) = A(i,j)-xmult*A(k,j);
    end
    b(i) = b(i)-xmult*b(k);
  end
end
x(n) = b(n)/A(n,n);
for i=n-1:-1:1
  sum = b(i);
  for j=i+1:n
    sum = sum-A(i,j)*x(j);
  end
  x(i) = sum/A(i,i);
end

function x = naiv_gauss(A,b)
n = length(b);
x = zeros(n,1);