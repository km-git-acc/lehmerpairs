{
filename="C:\\path\\zetazero_neighbors_87761.csv";
vals = readstr(filename);
numvals = #vals;

k=87761;
kindex=0;
R1 = 0.005776248278;
R2 = 2.468831965543*10^-5;
if(k>20000,Rterm=R2,Rterm=R1);

zindex=List();
x=List();

for(i=2,#vals,
zerodata = strsplit(vals[i],",");
zeroindex = eval(zerodata[1]);
zeroval = eval(zerodata[2]);
listput(zindex,zeroindex);
listput(x,2*zeroval);
if(zeroindex==k,kindex=i-1);
);

x_k = x[kindex];
x_kp1 = x[kindex+1];

M=0;
for(j=kindex-5000,kindex+5001,
    if(j!=kindex && j!=kindex+1, M = M + 1/(x_k - x[j])^2 + 1/(x_kp1 - x[j])^2);
);

I_bound = 2*(2*k+1)/(x_k - x[kindex-5001])^2;

x_kp5002 = x[kindex+5002];
R_bound = 2*(x_kp5002^2/(x_kp5002 - x_k)^2 + x_kp5002^2/(x_kp5002 - x_kp1)^2 + 2)*Rterm;

g_bound = M + I_bound + R_bound;
delta = x_kp1 - x_k;
lehmertest = (delta^2)*g_bound;
approx_t_c = -(1/8)*delta^2;

\\print(List([M,I_bound,R_bound,g_bound,k,x_k,x_kp1]));

if(lehmertest<4/5, 
    printf("(x_%d, x_%d) = (%.10f, %.10f) is a Lehmer pair colliding approximately at t_c = %.10f", k, k+1, x_k, x_kp1, approx_t_c),
    printf("(x_%d, x_%d) is not a Lehmer pair", k, k+1)
);

}
