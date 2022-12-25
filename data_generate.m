clear;

clc;
tic
%% 输入数据：总流量、路段初始阻抗、路段通行能力、路网领接矩阵

Q=3000;

%    1   2   3   4   5   6   7   8   9   10  11  12  13
Mxf=[1   0   1   0   0   0   0   0   0   0   0   0   0; %1
     0   1   1   0   0   0   0   0   0   0   0   0   0; %2
     0   0   0   1   0   1   0   0   0   0   0   0   0; %3
     0   0   0   0   1   1   0   0   0   0   0   0   0; %4
     0   0   0   0   0   1   1   1   0   0   0   0   0; %5
     0   0   0   0   0   1   1   0   1   0   0   0   0; %6
     0   0   0   0   0   0   1   0   0   1   0   0   0; %7
     0   0   0   0   0   0   0   0   0   0   1   0   1; %8
     0   0   0   0   0   0   0   0   0   0   0   1   1  %9
     ];


t1_1 = 9+rand(1,1);
t1_2 = 5+rand(1,1);
t1_3 = 4+rand(1,1);
t1_4 = 3+rand(1,1);
t1_5 = 2+rand(1,1);
t1_6 = 4+rand(1,1);
t1_7 = 6+rand(1,1);
t1_8 = 5+rand(1,1);
t1_9 = 8+rand(1,1);

c1_1 = 450+rand(1,1)*10;
c1_2 = 700+rand(1,1)*10;
c1_3 = 300+rand(1,1)*10;
c1_4 = 800+rand(1,1)*10;
c1_5 = 900+rand(1,1)*10;
c1_6 = 200+rand(1,1)*10;
c1_7 = 700+rand(1,1)*10;
c1_8 = 400+rand(1,1)*10;
c1_9 = 800+rand(1,1)*10;

t2_1 = 0.5+0.1*rand(1,1);
t2_2 = 0;
t2_3 = 1+0.1*rand(1,1);
t2_4 = 0;
t2_5 = 0;
t2_6 = 1+0.1*rand(1,1);
t2_7 = 0;
t2_8 = 0.4+0.1*rand(1,1);
t2_9 = 0;

c2_1 = 20+rand(1,1)*5;
c2_2 = 0;
c2_3 = 20+rand(1,1)*5;
c2_4 = 0;
c2_5 = 0;
c2_6 = 40+rand(1,1)*5;
c2_7 = 0;
c2_8 = 10+rand(1,1)*5;
c2_9 = 0;

l2_1 = 0.7+rand(1,1)*0.5;
l2_2 = 0;
l2_3 = 0.7+rand(1,1)*0.5;
l2_4 = 0;
l2_5 = 0;
l2_6 = 0.7+rand(1,1)*0.5;
l2_7 = 0;
l2_8 = 0.7+rand(1,1)*0.5;
l2_9 = 0;
%行驶初始时间
t1= [t1_1   t1_1   t1_2   t1_3   t1_3   t1_4   t1_5   t1_6   t1_6   t1_7   t1_8   t1_8   t1_9]
%初始排队时间
t2= [0      t2_1   0      0      t2_3   0      0      0      t2_6   0      0      t2_8   0]
%道路容量
c1= [c1_1   c1_1   c1_2   c1_3   c1_3   c1_4   c1_5   c1_6   c1_6   c1_7   c1_8   c1_8   c1_9]
%充电站容量
c2= [10000  c2_1  10000   10000  c2_3   10000  10000  10000  c2_6  10000   10000  c2_8   10000]
%电价
ld= [0      l2_1   0      0      l2_3   0      0      0      l2_6   0      0      l2_8   0]

t1_train = [t1_1  t1_2  t1_3  t1_4  t1_5  t1_6  t1_7  t1_8  t1_9];
c1_train = [c1_1  c1_2  c1_3  c1_4  c1_5  c1_6  c1_7  c1_8  c1_9];
t2_train = [t2_1  t2_2  t2_3  t2_4  t2_5  t2_6  t2_7  t2_8  t2_9];
c2_train = [c2_1  c2_2  c2_3  c2_4  c2_5  c2_6  c2_7  c2_8  c2_9];
l2_train = [l2_1  l2_2  l2_3  l2_4  l2_5  l2_6  l2_7  l2_8  l2_9];

%% 计算迭代过程

numf=size(Mxf,1);

numx=size(Mxf,2);

x1=zeros(1,numx);

y1=zeros(1,numx);

y0=zeros(1,numf);

t=zeros(1,numx);

count=0;

e=inf;

t=(t1).*(1+0.15*(x1./c1).^4)+ (t2).*(1+0.05*x1./(c2-x1)) + (ld).*(x1);

Ckrs=(Mxf*t')';

[Min,index]=min(Ckrs);

x1=Mxf(index,:).*Q;

syms zeta real;

zeta=0.5;


while e>0.001

count=count+1;

t=(t1).*(1+0.15*(x1./c1).^4)+ (t2).*(1+0.05*x1./(c2-x1))+(ld).*(x1);

Ckrs=(Mxf*t')';

y0=exp(-zeta*Ckrs);

h=sum(y0,2);

y2=exp(-zeta*Ckrs)/h;

y1=Q*y2*Mxf;

s=y1-x1;

x2=x1+1/count*s;

e=sqrt(sum((x2-x1).^2))/sum(x1);

x1=x2;

end

epoch = [t1_train t2_train l2_train c1_train c2_train x1];

disp(['迭代次数：',num2str(count)]);

disp(['配流结果：',num2str(x1)]);

disp(['路径阻抗：',num2str(Ckrs)]);

epochs = epoch();
step = 1;
while step < 10000
step = step+1;
t1_1 = 9+rand(1,1);
t1_2 = 5+rand(1,1);
t1_3 = 4+rand(1,1);
t1_4 = 3+rand(1,1);
t1_5 = 2+rand(1,1);
t1_6 = 4+rand(1,1);
t1_7 = 6+rand(1,1);
t1_8 = 5+rand(1,1);
t1_9 = 8+rand(1,1);

c1_1 = 450+rand(1,1)*10;
c1_2 = 700+rand(1,1)*10;
c1_3 = 300+rand(1,1)*10;
c1_4 = 800+rand(1,1)*10;
c1_5 = 900+rand(1,1)*10;
c1_6 = 200+rand(1,1)*10;
c1_7 = 700+rand(1,1)*10;
c1_8 = 400+rand(1,1)*10;
c1_9 = 800+rand(1,1)*10;

t2_1 = 0.5+0.1*rand(1,1);
t2_2 = 0;
t2_3 = 1+0.1*rand(1,1);
t2_4 = 0;
t2_5 = 0;
t2_6 = 1+0.1*rand(1,1);
t2_7 = 0;
t2_8 = 0.4+0.1*rand(1,1);
t2_9 = 0;

c2_1 = 20+rand(1,1)*5;
c2_2 = 0;
c2_3 = 20+rand(1,1)*5;
c2_4 = 0;
c2_5 = 0;
c2_6 = 40+rand(1,1)*5;
c2_7 = 0;
c2_8 = 10+rand(1,1)*5;
c2_9 = 0;

l2_1 = 0.7+rand(1,1)*0.5;
l2_2 = 0;
l2_3 = 0.7+rand(1,1)*0.5;
l2_4 = 0;
l2_5 = 0;
l2_6 = 0.7+rand(1,1)*0.5;
l2_7 = 0;
l2_8 = 0.7+rand(1,1)*0.5;
l2_9 = 0;
%行驶初始时间
t1= [t1_1   t1_1   t1_2   t1_3   t1_3   t1_4   t1_5   t1_6   t1_6   t1_7   t1_8   t1_8   t1_9];
%初始排队时间
t2= [0      t2_1   0      0      t2_3   0      0      0      t2_6   0      0      t2_8   0];
%道路容量
c1= [c1_1   c1_1   c1_2   c1_3   c1_3   c1_4   c1_5   c1_6   c1_6   c1_7   c1_8   c1_8   c1_9];
%充电站容量
c2= [10000  c2_1  10000   10000  c2_3   10000  10000  10000  c2_6  10000   10000  c2_8   10000];
%电价
ld= [0      l2_1   0      0      l2_3   0      0      0      l2_6   0      0      l2_8   0];

t1_train = [t1_1  t1_2  t1_3  t1_4  t1_5  t1_6  t1_7  t1_8  t1_9];
c1_train = [c1_1  c1_2  c1_3  c1_4  c1_5  c1_6  c1_7  c1_8  c1_9];
t2_train = [t2_1  t2_2  t2_3  t2_4  t2_5  t2_6  t2_7  t2_8  t2_9];
c2_train = [c2_1  c2_2  c2_3  c2_4  c2_5  c2_6  c2_7  c2_8  c2_9];
l2_train = [l2_1  l2_2  l2_3  l2_4  l2_5  l2_6  l2_7  l2_8  l2_9];

%% 计算迭代过程

numf=size(Mxf,1);

numx=size(Mxf,2);

x1=zeros(1,numx);

y1=zeros(1,numx);

y0=zeros(1,numf);

t=zeros(1,numx);

count=0;

e=inf;

t=(t1).*(1+0.15*(x1./c1).^4)+ (t2).*(1+0.05*x1./(c2-x1)) + (ld).*(x1);

Ckrs=(Mxf*t')';

[Min,index]=min(Ckrs);

x1=Mxf(index,:).*Q;

syms zeta real;

zeta=0.5;


while e>0.001

count=count+1;

t=(t1).*(1+0.15*(x1./c1).^4)+ (t2).*(1+0.05*x1./(c2-x1))+(ld).*(x1);

Ckrs=(Mxf*t')';

y0=exp(-zeta*Ckrs);

h=sum(y0,2);

y2=exp(-zeta*Ckrs)/h;

y1=Q*y2*Mxf;

s=y1-x1;

x2=x1+1/count*s;

e=sqrt(sum((x2-x1).^2))/sum(x1);

x1=x2;

end

epoch = [t1_train t2_train l2_train c1_train c2_train x1];
epochs = [epochs;epoch];
end

csvwrite('1.csv',epochs);
