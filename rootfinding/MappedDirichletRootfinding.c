/*
    Copyright (C) 2018 Association des collaborateurs de D.H.J Polymath
 
    This is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.  See <http://www.gnu.org/licenses/>.
*/
 
#include "acb_mat.h"
#include "acb_calc.h"
#include "acb_dirichlet.h"
#include "flint/profiler.h"
#include "pthread.h"

arb_t t;
arb_mat_t Logmem;
arb_mat_t Zerorangetprev;

acb_t z;
acb_mat_t Rsums;

slong numthreads;

struct ThreadData {
slong start, stop, id, prec;
}; 

//map s from Zeta-t to Xi_t world
void
Jt(acb_t res, acb_t s, slong prec)
{   
	arb_t a, pi, ta;
	arb_init(a);
	arb_init(pi);
	arb_init(ta);

	acb_t ac;
	acb_init(ac);

	arb_const_pi(pi, prec);
	
	arb_abs(ta, t);
	arb_mul_2exp_si(a, ta, -2);
	
	acb_mul_2exp_si(ac, s, -1);
	acb_div_arb(ac, ac, pi, prec);
	acb_log(ac, ac, prec);
	
	acb_mul_arb(ac, ac, a, prec);
	
	acb_add(res, s, ac, prec);

	arb_clear(a);
	arb_clear(pi);
	arb_clear(ta);

	acb_clear(ac);
}

//map s from Xi-t to Zeta_t world
void
Jtr(acb_t res, acb_t s, slong prec)
{   
	arb_t a, T, g, h, pi, ta, tdiv4;
	arb_init(a);
	arb_init(T);
	arb_init(g);
	arb_init(h);
	arb_init(pi);
	arb_init(ta);
	arb_init(tdiv4);

	arb_const_pi(pi, prec);
	
	arb_abs(ta, t);
	arb_mul_2exp_si(tdiv4, ta, -2);
	
	acb_get_real(g, s);
	acb_get_imag(h, s);
	
	//T ~ h - (|t|/4*Pi/2)
	arb_mul(T, tdiv4, pi, prec);
	arb_mul_2exp_si(T, T, -1);
	arb_sub(T, h, T, prec);
	
	//a ~ g - |t|/4*log(T/2)
	arb_mul_2exp_si(a, pi, 1);
	arb_div(a, T, a, prec);
	arb_log(a, a, prec);
	arb_mul(a, a, tdiv4, prec);
	arb_sub(a, g, a, prec);
 
	acb_set_arb_arb(res, a, T);
	
	arb_clear(a);
	arb_clear(T);
	arb_clear(g);
	arb_clear(h);
	arb_clear(pi);
	arb_clear(ta);
	arb_clear(tdiv4);
}


//Evaluate the Riemann sum (thread).
void* Ht_Riemann_sum_thread(void *voidData)
{
    //recover the data passed to this specific thread
    struct ThreadData* data=voidData;
   
    slong start=data->start;
    slong stop=data->stop;
    slong id=data->id;
    slong prec=data->prec;

	slong n;

	arb_t a, b, ta;
    arb_init(a);
    arb_init(b);
    arb_init(ta);
	
	acb_t res, resd, resdd, rsum, rsumd, rsumdd;
    acb_init(res);
    acb_init(resd);
    acb_init(resdd);
    acb_init(rsum);
    acb_init(rsumd);
    acb_init(rsumdd);
	
	arb_abs(ta, t);
	acb_zero(rsum); acb_zero(rsumd); acb_zero(rsumdd); 
	//printf("\n inthread id: %ld, start: %ld ,stop: %ld \n", id, start, stop);
	
    for (n = start; n < stop; n++)
    {
		//n^-z
		acb_mul_arb(res, z, arb_mat_entry(Logmem, n, 0), prec);
		acb_neg(res, res);
		acb_exp(res, res, prec);
		
		//n^-z * exp(-|t|/4*(log(n))^2)
		acb_mul_arb(res, res, arb_mat_entry(Logmem, n, 2), prec);

        //Xi_t(z)
		acb_add(rsum, rsum, res, prec);

		//first derivative Xi'_t(z)
		acb_mul_arb(resd, res, arb_mat_entry(Logmem, n, 0), prec);
		acb_neg(resd, resd);
		acb_add(rsumd, rsumd, resd, prec);
		
		//second derivative Xi''(z)
		acb_mul_arb(resdd, res, arb_mat_entry(Logmem, n, 1), prec);
		acb_add(rsumdd, rsumdd, resdd, prec);
    }
	
    acb_set(acb_mat_entry(Rsums, id, 0), rsum);
    acb_set(acb_mat_entry(Rsums, id, 1), rsumd);
    acb_set(acb_mat_entry(Rsums, id, 2), rsumdd);
	
    arb_clear(a);
    arb_clear(b);
    arb_clear(ta);

    acb_clear(res);
    acb_clear(resd);
    acb_clear(resdd);
    acb_clear(rsum);
    acb_clear(rsumd);
    acb_clear(rsumdd);

    flint_cleanup();

    return(NULL);
}

//Evaluate the Riemann sum. The loop is split up in line with the number of threads chosen.
void
Ht_Riemann_sum_new(acb_struct rs[3], slong N, slong numthreads, slong prec)
{	
    slong i, threadtasks; i = 0;

    //prep the threads
    pthread_t thread[numthreads];

    struct ThreadData data[numthreads];

    //prep all the thread data (divide up ranges in the overall loop with start and stop of range)
    threadtasks = N/numthreads;
		
    for (i = 0; i < numthreads; i++)
    {
    data[i].start= i*threadtasks;
    data[i].stop= (i+1)*threadtasks;
    data[i].id= i;
    data[i].prec=prec;
	//printf("\n id: %ld, start: %ld ,stop: %ld ", i, data[i].start, data[i].stop);
    }

	if (data[0].start == 0)
		data[0].start = 1;
    data[numthreads-1].stop = N + 1;

	
    //start the threads with an indexed (array) of a data-structure (with pointers to it from the threads).
    for (i = 0; i < numthreads; i++)
    {
        pthread_create(&thread[i], 0, Ht_Riemann_sum_thread, &data[i]);
    }

    //wait for all threads to complete
    for (i = 0; i < numthreads; i++)
    {
        pthread_join(thread[i], NULL);
    }

    acb_zero(rs); acb_zero(rs + 1); acb_zero(rs + 2);
    for (i = 0; i < numthreads; i++)
    {
        acb_add(rs + 0, rs + 0, acb_mat_entry(Rsums, i, 0), prec);
        acb_add(rs + 1, rs + 1, acb_mat_entry(Rsums, i, 1), prec);
        acb_add(rs + 2, rs + 2, acb_mat_entry(Rsums, i, 2), prec);
    }
}

//precompute and memoise log(n) and log(n)^2.
void
Memoise_logn_logn2(slong res, slong N, slong prec)
{   
	arb_t a, n;
	arb_init(a);
	arb_init(n);
	
	slong j;
	
	for (j = 1; j < N + 1; j++)
    {
		arb_set_si(n, j);
		arb_log(a, n, prec);
		arb_set(arb_mat_entry(Logmem, j, 0), a);
		arb_mul(a, a, a, prec);
		arb_set(arb_mat_entry(Logmem, j, 1), a);
	}
	
	arb_clear(a);
	arb_clear(n);
}

//precompute and memoise exp(-|t|/4*log(n)^2).
void
Memoise_exp_t4logn2(slong res, slong N, slong prec)
{   
	arb_t a, ta;
	arb_init(a);
	arb_init(ta);
	
	slong n;
	
	arb_abs(ta, t);
	
	for (n = 1; n < N + 1; n++)
    {
		arb_mul_2exp_si(a, ta, -2);
		arb_neg(a, a);
		arb_mul(a, a, arb_mat_entry(Logmem, n, 1), prec);
		arb_exp(a, a, prec);
		
		arb_set(arb_mat_entry(Logmem, n, 2), a);
	}
	
	arb_clear(a);
	arb_clear(ta);
}

//print all data for the current t
void
Print_all_data_t(slong res, slong kl, slong kh, slong prec)
{    
	slong k;
	
	arb_t x, y;
	arb_init(x);
	arb_init(y);
	
	acb_t ac;
	acb_init(ac);
	
	for (k = kl; k < kh + 1; k++)
    {
		arb_printn(t, 5, ARB_STR_NO_RADIUS);
		printf(", %ld, ", k);
		arb_printn(arb_mat_entry(Zerorangetprev, k - kl, 0), 20, ARB_STR_NO_RADIUS);
		printf(", ");
		arb_printn(arb_mat_entry(Zerorangetprev, k - kl, 1), 20, ARB_STR_NO_RADIUS);
		printf(", ");
		acb_set_arb_arb(ac, arb_mat_entry(Zerorangetprev, k - kl, 0), arb_mat_entry(Zerorangetprev, k - kl, 1));
		
		//convert Zeta_t to Xi_t (Jt)
		Jt(ac, ac, prec);
		acb_get_real(x, ac);
		arb_printn(x, 20, ARB_STR_NO_RADIUS);
		printf(", ");
		acb_get_imag(y, ac);
		arb_printn(y, 20, ARB_STR_NO_RADIUS);
		printf(", ");
		
		//convert Xi_t to Ht(z)
		arb_mul_2exp_si(y, y, 1);
		arb_printn(y, 20, ARB_STR_NO_RADIUS);
		printf(", ");
		arb_mul_2exp_si(x, x, 1);
		arb_neg(x, x);
		arb_add_si(x, x, 1, prec);
		arb_neg(x, x);
		arb_printn(x, 20, ARB_STR_NO_RADIUS);
		
		printf("\n");
	}

	arb_clear(x);
	arb_clear(y);
	
	acb_clear(ac);
}

//initial fill of the global matrix with zeros of H_0(x) 
void
Initial_fill_with_zeros_at_t0(slong out, slong kl, slong kh, slong N, slong prec)
{    
	slong k;
	
	fmpz_t n;
	fmpz_init(n);
	
	arb_t x, y;
	arb_init(x);
	arb_init(y);
	
	acb_t root;
	acb_init(root);
	
	arb_set_str(t, "-0.00000000000000000001", prec);
	
	for (k = kl; k < kh + 1; k++)
    {
		fmpz_set_si(n, k);
		acb_dirichlet_zeta_zero(root, n, prec);
		acb_get_real(x, root);
		acb_get_imag(y, root);
		
		//store zero Zeta-t(s)
		arb_set(arb_mat_entry(Zerorangetprev, k - kl, 0), x);
		arb_set(arb_mat_entry(Zerorangetprev, k - kl, 1), y);
	}

	arb_zero(t);
	Print_all_data_t(out, kl, kh, prec);

	fmpz_clear(n);
	
	arb_clear(x);
	arb_clear(y);
	
	acb_clear(root);
}

// f(z) = Ht(z) / Ht'(z)
void
f_Ht_frac1(acb_t logder, slong N, slong prec)
{
    acb_struct s[3];
    acb_init(s);
    acb_init(s + 1);
    acb_init(s + 2);

    Ht_Riemann_sum_new(s, N, numthreads, prec);
    acb_div(logder, s , s + 1, prec);

    acb_clear(s);
    acb_clear(s + 1);
    acb_clear(s + 2);
}

// Find exact (acc=accuracy) complex root using Newton-Raphson method
void
Newton_Raphson(acb_t x1, acb_t xx, slong N, arb_t acc, slong prec)
{
    arb_t a, b;
    arb_init(a);
    arb_init(b);

    acb_t h, x0;
    acb_init(h);
    acb_init(x0);
    acb_init(x1);

    slong itr, maxmitr;
    maxmitr = 200;

    acb_set(x0, xx);

    for (itr=1; itr<=maxmitr; itr++)
    {
        acb_set(z, x0);
        f_Ht_frac1(h, N, prec);
        acb_sub(x1, x0, h, prec);
        acb_abs(a, h, prec);
        if (arb_lt(a, acc))
            break;
        acb_set (x0, x1);
        acb_get_real(a, x0);
        acb_get_imag(b, x0);
        arb_get_mid_arb(a, a);
        arb_get_mid_arb(b, b);
        acb_set_arb_arb(x0, a, b);
    }

    if (itr > maxmitr)
       acb_zero(x1);

    arb_clear(a);
    arb_clear(b);

    acb_clear(h);
    acb_clear(x0);
}

void
FindRoots(acb_t root, arb_t xs, arb_t ys, slong N, arb_t acc, slong prec)
{
	arb_t a, x, res, rootre, rootim;
    arb_init(a);
    arb_init(x);
	arb_init(rootim);
	arb_init(rootre);
	arb_init(res);

    acb_t x0;
    acb_init(x0);

    arb_zero(res);

	acb_set_arb_arb(x0, xs, ys);
	Newton_Raphson(root, x0, N, acc, prec);

	if (!acb_is_zero(root))
	{
		acb_get_real(rootre, root);
		acb_get_imag(rootim, root);
		arb_abs(rootim, rootim);
		
		if (arb_lt(rootim, acc))
		{
			arb_set(res, x);
			goto end;
		}
	}

end:

    arb_clear(a);
    arb_clear(x);
    arb_clear(rootre);
    arb_clear(rootim);
    arb_clear(res);

    acb_clear(x0);
}

int main(int argc, char *argv[])
{
    arb_t a, acc, x, y, xs, ys, ta, te, ts, ti, outcome;
    arb_init(a);
    arb_init(acc);
    arb_init(x);
    arb_init(y);
    arb_init(xs);
    arb_init(ys);
    arb_init(ta);
    arb_init(ts);
    arb_init(te);
    arb_init(ti);
    arb_init(outcome);

	//init the global variables
    arb_init(t);
    acb_init(z);

	acb_t zder, resc, root;
	acb_init(zder);
	acb_init(root);
	acb_init(resc);

    const char *kl_str, *kh_str, *te_str, *ti_str, *numthreads_str;
	
    slong i, kl, kh, N, prec, res; res = 0; 

    int result = EXIT_SUCCESS;

    if (argc != 6)
    {
        result = EXIT_FAILURE;
        goto finish;
    }

    kl_str = argv[1];
    kh_str = argv[2];
    te_str = argv[3];
    ti_str = argv[4];
    numthreads_str = argv[5];

    //Working precision 
    prec = 128;

    kl = atol(kl_str);
    kh = atol(kh_str);
    arb_set_str(te, te_str, prec);
    arb_set_str(ti, ti_str, prec);
    numthreads = atol(numthreads_str);
	
	if(kl < 1 || kh < kl || numthreads < 1 || numthreads > 64 || !(arb_is_negative(te) || !arb_is_negative(ti)))
    {
		printf("Invalid option chosen. \n");
        result = EXIT_FAILURE; 
        goto finish;
    }
	
	N = 3000;
	
	arb_mat_init(Logmem, N + 1, 3);
	arb_mat_init(Zerorangetprev, kh - kl + 1, 2);
	acb_mat_init(Rsums, numthreads, 3);
	
	//memoize the log(n) and log(n)^2
	Memoise_logn_logn2(res, N, prec);

	// set target accuracy for all Newton-Raphson rootfinding
	arb_set_str(acc, "0.00000000001", prec); 

	arb_zero(t);	
	//fill the startingvalues with the Ht-zeros at t = 0
	Initial_fill_with_zeros_at_t0(res, kl, kh, N, prec);

    acb_struct s[3];
    acb_init(s);
    acb_init(s + 1);
    acb_init(s + 2);

//    arb_set_str(t, "-1", prec);
//    arb_set_str(x, "-0.4", prec);
//    arb_set_str(y, "300", prec);
//	
//	//precompute and memoise exp(-|t|/4*log(n)^2).
//	Memoise_exp_t4logn2(res, N, prec);
//		
//	acb_set_arb_arb(z, x, y);
//	Ht_Riemann_sum_new(s, N, numthreads, prec);
//	
//	acb_printd(s, 20);
//	printf(" =s zeta_t\n");
//	
//	FindRoots(root, x, y, N, acc, prec);
//	acb_get_real(xs, root);
//	acb_get_imag(ys, root);
//	
//	acb_printd(root, 20);
//	printf(" =root zeta_t\n");
//	
//	Jt(zder, root, prec);
//	
//	acb_printd(zder, 20);
//	printf(" =root zeta_t -> xi_t \n");
//	
//	Jtr(resc, zder, prec);
//	
//	acb_printd(resc, 20);
//	printf(" =root xi_t -> zeta_t \n");
//	
//	acb_printd(s + 1, 20);
//	printf("\n");
//	acb_printd(s + 2, 20);
//	printf("\n");
//
//goto finish;

	//loop through the t range
	arb_abs(te, te);
	arb_abs(ta, t);

	while(arb_lt(ta, te))
	{
		arb_add(t, t, ti, prec);
		
		//precompute and memoise exp(-|t|/4*log(n)^2).
		Memoise_exp_t4logn2(res, N, prec);
		
		arb_abs(ta, t);

		for (i = kl; i < kh + 1; i++)
		{
			//set xs and ys to the coodinates of the zero found and the previous t
			arb_set(xs, arb_mat_entry(Zerorangetprev, i - kl, 0));
			arb_set(ys, arb_mat_entry(Zerorangetprev, i - kl, 1));
						
			FindRoots(root, xs, ys, N, acc, prec);
			acb_get_real(x, root);
			acb_get_imag(y, root);
				
			//make y negative when the y_k = y_k-1, i.e. zero k+1 of the colliding pair becomes the conjugate 
			//if (i - kl -1 >= 0)
				//arb_sub(a, arb_mat_entry(Zerorangetprev, i - kl - 1, 1), y, prec);
		
			//store zero Ht(x+yi)
			arb_set(arb_mat_entry(Zerorangetprev, i - kl, 0), x);
			arb_set(arb_mat_entry(Zerorangetprev, i - kl, 1), y);
		}
		
		Print_all_data_t(res, kl, kh, prec);
	}
	
finish:
 
    if (result == EXIT_FAILURE)
    {
        flint_printf("Required inputs:\n");
        flint_printf("%s kstart, kend, tend, tinc, numthreads \n\n", argv[0]);
        flint_printf(
    "This script traces the k_th zero of Xi_t(s) over negative time t = 0 .. tend in steps of tinc.\n"
	"Multiple k could be traced simultaneoulsy bij specifying a range kstart ..kend.\n"
	"This scripts find the zeros of a finite t-dependent Dirichlet series (N=3000) that are 'mapped' to Xi_t and Ht"
	"Output is: t, k, Zeta_t(x), Zeta_t(y), Xi_t(x), Xi_t(y), Ht(x), Ht(y)"
    "Accuracy of the zeros has been set at 10 digits. \n"
    "The number of threads chosen helps to speed up evaluation of the Dirichlet series.\n\n");
    }

    arb_clear(a);
    arb_clear(acc);
    arb_clear(x);
    arb_clear(y);
    arb_clear(xs);
    arb_clear(ys);
    arb_clear(t);
    arb_clear(ta);
    arb_clear(ts);
    arb_clear(te);
    arb_clear(ti);
    arb_clear(outcome);

	acb_clear(zder);
	acb_clear(root);
    acb_clear(z);
    acb_clear(s);
    acb_clear(s + 1);
    acb_clear(s + 2);

    arb_mat_clear(Logmem);
    arb_mat_clear(Zerorangetprev);
	
    acb_mat_clear(Rsums);
 
    flint_cleanup();

    return result;
}
