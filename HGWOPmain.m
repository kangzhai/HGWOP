function HGWOPmain()
%Hybrid particle swarm and grey wolf optimizer,HGWOP
%_________________________________________________________________________%
% Hybrid particle swarm and grey wolf optimizer source codes demo 1.0     %
%  Developed in MATLAB R2014b                                             %
%  Author and programmer: Xinming Zhang                                   %
%   Main paper:
%   Zhang, X.M., Lin, Q.Y., & Mao, W.T.et al (2021). Hybrid Particle Swarm
%   and Grey Wolf Optimizer and its application toclustering optimization.
%   Applied Soft Computing, 2020, 
%   DOI: https://doi.org/10.1016/j.asoc.2020.107061
%    e-Mail: xinmingzhang@126.com                                         %
%https://www.sciencedirect.com/science/article/abs/pii/S1568494620309996?via%3Dihub
%_________________________________________________________________________%
clc;
clear;
Num=30;
D=30;
if D==100
    N=20;MaxDT=8000;
elseif D==50
    N=20;MaxDT=5000;
else
    N=50;MaxDT=1000;D=30;
end
Vt=zeros(1,Num);
f ='Sphere';a=-100*ones(1,D);b=100*ones(1,D);

CBest=zeros(1,MaxDT);
time=0;
for i=1:Num
    tic;
    [u0,s,~] =HGWOP(f,a,b,D,MaxDT,N);
    time=((i-1)*time+toc)/i;
    CBest(1,:)=CBest(1,:)+s;
    Vt(i)=u0;
end
MeanValue=mean(Vt);
StdValue=std(Vt);
GoodValue=min(Vt);
BadValue=max(Vt);
plot(Vt)
title([' HGWOP£ºMean=',num2str(MeanValue),'£¬Std=',num2str(StdValue)]);
xlabel(['Vma=',num2str(BadValue),'£¬Vmi=',num2str(GoodValue),'£¬Time=',num2str(time)]);


function y = Sphere(x)
%Sphere function
y = sum(x.^2);

