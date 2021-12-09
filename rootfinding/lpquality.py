import csv

a=[0]
i=0;
with open("C:/path/zeros10mlncopy.txt") as f:
    for line in f:
        a.append(float(line.strip()))
        i=i+1
        print(i)

q=[[0,0,0,0],[1,0,0,0],[2,0,0,0]];
for i in range(3,len(a)-5):
    delta = a[i+1] - a[i];
    qual = min(a[i+2] - a[i+1], a[i] - a[i-1])/delta
    q.append([i,a[i],delta,qual])
    if i%1000==0 : print(i)

hq = sorted(q, key=lambda x: x[3])
hq.reverse()

vhq=hq[0:1000]

with open("C:/path/zeros10mln_qualsorted.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(vhq)

