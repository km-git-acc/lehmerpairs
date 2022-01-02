J(t,s) = s+abs(t/4)*log(s/2/Pi);

crudeinvJ(t,s) = s - abs(t/4)*log(s/2/Pi);

invJ(t,s) = {
prec=1/10^15;
err=0;
zs=crudeinvJ(t,s);
until(abs(err)<prec,
                    zs = zs-err;
                    js = J(t,zs);
                    err = js-s;
     );
return(zs);
}

F(t,s) = s + abs(t/4)*log(abs(s/2/Pi));

crudeinvF(t,s) = s - abs(t/4)*log(abs(s/2/Pi));

invF(t,s) = {
prec=1/10^15;
err=0;
zs=crudeinvF(t,s);
until(abs(err)<prec,
                    zs = zs-err;
                    fs = F(t,zs);
                    err = fs-s;
     );
return(zs);
}


gamfac(s)=(1/2)*s*(s-1)*(Pi^(-s/2))*gamma(s/2);
gam(t,s)=gamfac(s)*exp(abs(1/t)*(s-J(t,s))^2);
xi(s)=gamfac(s)*zeta(s);
Ht(t,s)=1/8*intnum(v=-8,8,xi((1+I*s)/2+t^(1/2)*v)*1/sqrt(Pi)*exp(-v^2));
dHt(t,s)=I/(8*sqrt(t*Pi))*intnum(v=-8,8,xi((1+I*s)/2+sqrt(t)*v)*exp(-v^2)*v);
ddHt(t,s)=-1/(8*sqrt(Pi)*t)*intnum(v=-8,8,xi((1+I*s)/2+sqrt(t)*v)*exp(-v^2)*(v^2-1/2));
xit(t,s)=Ht(t,(2*s-1)/I);
zet(t,s)=sum(n=1,1000,exp(-abs(t/4)*log(n)*log(n))/n^s);
xitalt(t,s)=gam(t,invJ(t,s))*zet(t,invJ(t,s));
Z(t,s)=sum(n=1,1000,exp(-abs(t/4)*log(n)*log(n))/n^invF(t,s));
xiZ(t,s)=gam(t,invF(t,s))*Z(t,s);
