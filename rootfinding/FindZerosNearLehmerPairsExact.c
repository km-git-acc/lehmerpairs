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

arb_t t, step;
arb_mat_t Threadwork;

acb_t z;
acb_mat_t Rsums;

slong numthreads;

struct ThreadData {
    slong id, prec;
}; 

// Prepare the 'integrand' ofr the Riemann summation
void
xi_integrand(acb_t result, const arb_t u, slong prec)
{
    arb_t a;
    arb_init(a);

    acb_t ac, bc, xi;
    acb_init(ac);
    acb_init(bc);
    acb_init(xi);

    //(1+z*I)/2
    acb_mul_onei(ac, z);
    acb_add_si(ac, ac, 1, prec);
    acb_mul_2exp_si(ac, ac, -1);

    //sqrt(t)*u (t can be negative, hence acb)
    acb_set_arb(bc, t);
    acb_sqrt(bc, bc, prec);	
    acb_mul_arb(bc, bc, u, prec);

    acb_add(ac, ac, bc, prec);

    //xi((1+z*I)/2 + sqrt(t)*u)
    acb_dirichlet_xi(xi, ac, prec);

    arb_mul(a, u, u, prec);
    arb_neg(a, a);
    arb_exp(a, a, prec);

    acb_mul_arb(result, xi, a, prec);

    arb_clear(a);

    acb_clear(ac);
    acb_clear(bc);
    acb_clear(xi);
}

//Evaluate the Riemann sum (thread).
void* Ht_Riemann_sum_thread(void *voidData)
{
    //recover the data passed to this specific thread
    struct ThreadData* data=voidData;
    slong id=data->id;
    slong prec=data->prec;

    arb_t a, i, start, stop, step;
    arb_init(a);
    arb_init(i);
    arb_init(start);
    arb_init(stop);
    arb_init(step);

    acb_t res, resd, rsum, rsumd;
    acb_init(res);
    acb_init(resd);
    acb_init(rsum);
    acb_init(rsumd);

    arb_set(start, arb_mat_entry(Threadwork, id, 0));
    arb_set(stop, arb_mat_entry(Threadwork, id, 1));
    arb_set(step, arb_mat_entry(Threadwork, id, 2));

    acb_zero(rsum); acb_zero(rsumd); arb_set(i, start);

    while(arb_lt(i, stop))
    {
        arb_get_mid_arb(i, i);
        xi_integrand(res, i, prec);
        acb_add(rsum, rsum, res, prec);
        acb_mul_arb(resd, res, i, prec);
        acb_add(rsumd, rsumd, resd, prec);
        arb_add(i, i, step, prec);
    }

    acb_set(acb_mat_entry(Rsums, id, 0), rsum);
    acb_set(acb_mat_entry(Rsums, id, 1), rsumd);

    arb_clear(a);
    arb_clear(i);
    arb_clear(start);
    arb_clear(stop);
    arb_clear(step);

    acb_clear(res);
    acb_clear(resd);
    acb_clear(rsum);
    acb_clear(rsumd);

    flint_cleanup();

    return(NULL);
}

//Evaluate the Riemann sum. The loop is split up in line with the number of threads chosen.
void
Ht_Riemann_sum(acb_struct rs[2], arb_t step, slong numthreads, slong prec)
{
    arb_t a, b;
    arb_init(a);
    arb_init(b);

    acb_t ac;
    acb_init(ac);
	
    slong i;
    
    //prep the threads
    pthread_t thread[numthreads];

    struct ThreadData data[numthreads];

    //prep all the thread data
    for (i = 0; i < numthreads; i++)
    {
        data[i].id = i;
        data[i].prec = prec;
    }

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

    acb_zero(rs); acb_zero(rs + 1);
    for (i = 0; i < numthreads; i++)
    {
        acb_add(rs, rs, acb_mat_entry(Rsums, i, 0), prec);
        acb_add(rs + 1, rs + 1, acb_mat_entry(Rsums, i, 1), prec);
    }

    //rs*1/(8*sqrt(pi)*step)
    acb_mul_2exp_si(rs, rs, -3);
    arb_const_pi(a, prec);
    arb_sqrt(a, a, prec);
    acb_div_arb(rs, rs, a, prec);
    acb_mul_arb(rs, rs, step, prec);

    //rsd*I/(8*sqrt(pi*t)*step)
    acb_mul_2exp_si(rs + 1, rs + 1, -3);
    acb_mul_onei(rs + 1, rs + 1);
    acb_set_arb(ac, t);
    acb_sqrt(ac, ac, prec);
    acb_mul_arb(ac, ac, a, prec);
    acb_div(rs + 1, rs + 1, ac, prec);
    acb_mul_arb(rs + 1, rs + 1, step, prec);

    arb_clear(a);
    arb_clear(b);

    acb_clear(ac);
}

//Allocate the work across threads (stored in global matrix Threadwork)
void
Allocate_worktothreads(slong nix, arb_t intlim, arb_t step, slong numthreads, slong prec)
{
    arb_t a, workload;
    arb_init(a);
    arb_init(workload);

    slong i;

    arb_mul_2exp_si(workload, intlim, 1);
    arb_div(workload, workload, step, prec);
    arb_add_si(workload, workload, 1, prec);
    arb_div_si(workload, workload, numthreads, prec);
    arb_floor(workload, workload, prec);

    for(i = 0; i < numthreads; i++)
    {
        arb_mul_si(a, workload, i, prec);
        arb_mul(a, a, step, prec);
        arb_sub(a, a, intlim, prec);
        arb_set(arb_mat_entry(Threadwork, i, 0), a);

        arb_mul_si(a, workload, i + 1, prec);
        arb_mul(a, a, step, prec);
        arb_sub(a, a, intlim, prec);
        arb_set(arb_mat_entry(Threadwork, i, 1), a);

        arb_set(arb_mat_entry(Threadwork, i, 2), step);
    }
    arb_set(arb_mat_entry(Threadwork, numthreads - 1, 1), intlim);

    arb_clear(a);
    arb_clear(workload);
}

// f(z) = Ht'(z) / Ht(z)
void
f_Ht_frac1(acb_t logder, slong prec)
{
    acb_struct s[2];
    acb_init(s);
    acb_init(s + 1);

    Ht_Riemann_sum(s, step, numthreads, prec);
    acb_div(logder, s + 1, s, prec);

    acb_clear(s);
    acb_clear(s + 1);
}

// f(z) = Ht(z) / Ht'(z)
void
f_Ht_frac2(acb_t logder, slong prec)
{
    acb_struct s[2];
    acb_init(s);
    acb_init(s + 1);

    Ht_Riemann_sum(s, step, numthreads, prec);
    acb_div(logder,s ,s + 1 , prec);

    acb_clear(s);
    acb_clear(s + 1);
}

// Find exact (acc=accuracy) complex root using Newton-Raphson method
void
Newton_Raphson(acb_t x1, acb_t xx, arb_t acc, slong prec)
{
    arb_t a, b;
    arb_init(a);
    arb_init(b);

    acb_t h, x0;
    acb_init(h);
    acb_init(x0);
    acb_init(x1);

    slong itr, maxmitr;
    maxmitr = 40;

    acb_set(x0, xx);

    for (itr=1; itr<=maxmitr; itr++)
    {
        acb_set(z, x0);
        f_Ht_frac2(h, prec);
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
FindComplexroots(arb_t res, arb_t xmin, arb_t xmax, arb_t y, arb_t acc, slong prec)
{
    arb_t a, x, rootre, rootim;
    arb_init(a);
    arb_init(x);
	arb_init(rootim);
	arb_init(rootre);

    acb_t root, x0;
    acb_init(x0);
    acb_init(root);

    arb_zero(res);

	arb_add(a, xmin, xmax, prec);
	arb_mul_2exp_si(x, a, -1);

	acb_set_arb_arb(x0, x, y);
	Newton_Raphson(root, x0, acc, prec);

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
		
		arb_printn(rootre, 30, ARB_STR_NO_RADIUS);
		printf(", ");
		arb_printn(rootim, 20, ARB_STR_NO_RADIUS);
		printf(", ");
		arb_printn(t, 20, ARB_STR_NO_RADIUS);
		printf("\n");
		arb_printn(rootre, 30, ARB_STR_NO_RADIUS);
		printf(", -");
		arb_printn(rootim, 20, ARB_STR_NO_RADIUS);
		printf(", ");
		arb_printn(t, 20, ARB_STR_NO_RADIUS);
		printf("\n");
	}

end:

    arb_clear(a);
    arb_clear(x);
    arb_clear(rootre);
    arb_clear(rootim);

    acb_clear(x0);
    acb_clear(root);
}

void
FindRealroots(arb_t res, arb_t xmin, arb_t xmax, arb_t acc, slong prec)
{
    arb_t a, x, rootre, rootim;
    arb_init(a);
    arb_init(x);
	arb_init(rootim);
	arb_init(rootre);

    acb_t root, x0;
    acb_init(x0);
    acb_init(root);

    arb_zero(res);

	arb_zero(a);
	acb_set_arb_arb(x0, xmin, a);
	Newton_Raphson(root, x0, acc, prec);

	acb_get_real(rootre, root);
	arb_zero(rootim);
		
	arb_printn(rootre, 30, ARB_STR_NO_RADIUS);
	printf(", ");
	arb_printn(rootim, 20, ARB_STR_NO_RADIUS);
	printf(", ");
	arb_printn(t, 20, ARB_STR_NO_RADIUS);
	printf("\n");

	arb_zero(a);
	acb_set_arb_arb(x0, xmax, a);
	Newton_Raphson(root, x0, acc, prec);

	acb_get_real(rootre, root);
	arb_zero(rootim);
		
	arb_printn(rootre, 30, ARB_STR_NO_RADIUS);
	printf(", ");
	arb_printn(rootim, 20, ARB_STR_NO_RADIUS);
	printf(", ");
	arb_printn(t, 20, ARB_STR_NO_RADIUS);
	printf("\n");
	
    arb_clear(a);
    arb_clear(x);
    arb_clear(rootre);
    arb_clear(rootim);

    acb_clear(x0);
    acb_clear(root);
}

int main(int argc, char *argv[])
{
    arb_t a, acc, xs, xe, ys, intlim, te, ts, ti, outcome;
    arb_init(a);
    arb_init(acc);
    arb_init(xs);
    arb_init(xe);
    arb_init(ys);
    arb_init(intlim);
    arb_init(ts);
    arb_init(te);
    arb_init(ti);
    arb_init(outcome);

	//init the global variables
    arb_init(step);
    arb_init(t);

    acb_init(z);

    const char *xs_str, *xe_str, *ts_str, *te_str, *ti_str, *numthreads_str;
	
    slong prec, res; res = 0;

    int result = EXIT_SUCCESS;

    if (argc != 7)
    {
        result = EXIT_FAILURE;
        goto finish;
    }

    xs_str = argv[1];
    xe_str = argv[2];
    ts_str = argv[3];
    te_str = argv[4];
    ti_str = argv[5];
    numthreads_str = argv[6];

    //Working precision 
    prec = 20 * 3.32192809488736 + 210;

    arb_set_str(xs, xs_str, prec);
    arb_set_str(xe, xe_str, prec);
    arb_set_str(ts, ts_str, prec);
    arb_set_str(te, te_str, prec);
    arb_set_str(ti, ti_str, prec);

    numthreads = atol(numthreads_str);

    if(numthreads < 1 || numthreads > 64)
    {
        result = EXIT_FAILURE; 
        goto finish;
    }

    if(arb_lt(xe, xs))
    {
        result = EXIT_FAILURE; 
        goto finish;
    }

    arb_mat_init(Threadwork, numthreads, 3);
    acb_mat_init(Rsums, numthreads, 2);

	//Riemann summation is fixed at 160 steps, which proved sufficient for evaluation Lehmer pairs < 10^8
	arb_set_d(intlim, 8); arb_set_d(step, 0.1);

    Allocate_worktothreads(res, intlim, step, numthreads, prec);
	
	// set target accuracy for all Newton-Raphson rootfinding
	arb_set_str(acc, "0.000000000000000001", prec); 
	
	// set the imaginary starting value for complex Newton-Raphson rootfinding
	arb_set_str(ys, "0.1", prec); 

    arb_set(t, ts);
	arb_add(te, te, acc, prec);

    // loop through the specified t range by steps tinc until the first real zero is found
    while(!arb_is_positive(outcome))
    {
		FindComplexroots(outcome, xs, xe, ys, acc, prec);
		
		arb_add(t, t, ti, prec);
    }
	
	// remove 't-overshoot'
	arb_sub(t, t, ti, prec);

    // loop through the specified t range by steps tinc until the te has been reached
	while(arb_le(t, te))
	{
		//t=0 is difficult to compute, so load a small number (acc)
		arb_abs(a, t);
		if(arb_le(a, acc))
			arb_set(t, acc);

		FindRealroots(outcome, xs, xe, acc, prec);

		arb_add(t, t, ti, prec);
	}

finish:
 
    if (result == EXIT_FAILURE)
    {
        flint_printf("Required inputs:\n");
        flint_printf("%s xs, xe, ts, te, ti numthreads \n\n", argv[0]);
        flint_printf(
    "This script computes all complex and real zeros of H_t(x+yi) near a Lehmer-pair over a range of t.\n"
    "In the x-range, xs(tart) and xe(nd) are best chosen with 0.1 below and above the Lehmer-pair.\n"
    "The t-range runs from ts(tart) to te(nd) in steps of ti(ncr) that determines the #zeros printed (x, y, t).\n"
    "The number of threads chosen helps to speed up the Riemann-sum evaluation of the Ht-integral.\n");
    }

    arb_clear(a);
    arb_clear(acc);
    arb_clear(xs);
    arb_clear(xe);
    arb_clear(ys);
    arb_clear(intlim);
    arb_clear(step);
    arb_clear(t);
    arb_clear(ts);
    arb_clear(te);
    arb_clear(ti);
    arb_clear(outcome);

    acb_clear(z);

    arb_mat_clear(Threadwork);
    acb_mat_clear(Rsums);
 
    flint_cleanup();

    return result;
}