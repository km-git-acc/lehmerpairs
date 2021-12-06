{
default(parisize, 1200000000);

filename="C:\\path\\zetazero_list_first_150000.csv";
vals = readstr(filename);
numvals = #vals;

x=List();

for(i=2,numvals,
zerodata = strsplit(vals[i],",");
zeroval = eval(zerodata[2]);
listput(x,2*zeroval);
);

R1 = 0.005776248278;
R2 = 2.468831965543*10^-5;

klist = List([6707,6708,6709,6710,6711, 18857,18858,18859,18860,18861, 44553,44544,44555,44556,44557, 73995,73996,73997,73998,73999, 82550,82551,82552,82553,82554, 87759,87760,87761,87762,87763,  95246,95247,95248,95249,95250]);

for(counter=1,#klist,
k=klist[counter];
if(k>20000,Rterm=R2,Rterm=R1);

x_k = x[k];
x_kp1 = x[k+1];

M=0;
for(j=k-5000,k+5001,
    if(j!=k && j!=k+1, M = M + 1/(x_k - x[j])^2 + 1/(x_kp1 - x[j])^2);
);

I_bound = 2*(2*k+1)/(x_k - x[k-5001])^2;

x_kp5002 = x[k+5002];
R_bound = 2*(x_kp5002^2/(x_kp5002 - x_k)^2 + x_kp5002^2/(x_kp5002 - x_kp1)^2 + 2)*Rterm;

g_bound = M + I_bound + R_bound;
delta = x_kp1 - x_k;
lehmertest = (delta^2)*g_bound;
approx_t_c = -(1/8)*delta^2;

\\print(List([M,I_bound,R_bound,g_bound,k,x_k,x_kp1]));

if(lehmertest<4/5, 
    printf("(x_%d, x_%d) = (%.10f, %.10f) is a Lehmer pair colliding approximately at t_c = %.10f \n", k, k+1, x_k, x_kp1, approx_t_c),
    printf("(x_%d, x_%d) is not a Lehmer pair \n", k, k+1)
);
);
}
