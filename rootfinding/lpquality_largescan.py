import csv
import math

a=[]
hq=[];
i=0;
kmax = 100000000-5
with open("C:/path/zeros100mlncopy.txt") as f:
    for line in f:
        f = float(line.strip());
        a.append(f)
        i=i+1;
        if i%100000==0: print(i)
        if i>kmax: break;
        if(i>3):
              delta = a[2] - a[1];
              qual = min(a[3] - a[2], a[1] - a[0])/delta
              midp = (a[1]+a[2])/2
              avggap = 2*math.pi/math.log(midp/(2*math.pi))
              normgap = delta/avggap
              if(qual>20):
                      hqelem = [i-2,a[1],a[2],delta,normgap,qual]
                      hq.append(hqelem)
                      print(hqelem)
              m=a.pop(0)
			  

hq = sorted(hq, key=lambda x: x[5])
hq.reverse()

hq.insert(0,["k","z_k","z_{k+1}","delta","normalized gap","quality"])

with open("C:/path/zeros100mln_qualsorted.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(hq)

