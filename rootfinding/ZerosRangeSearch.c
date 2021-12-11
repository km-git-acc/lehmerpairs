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
arb_mat_t Threadwork, Zerorangetprev;

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

    arb_t a, i, half, start, stop, step;
    arb_init(a);
    arb_init(i);
    arb_init(half);
    arb_init(start);
    arb_init(stop);
    arb_init(step);

    acb_t res, resd, resdd, rsum, rsumd, rsumdd;
    acb_init(res);
    acb_init(resd);
    acb_init(resdd);
    acb_init(rsum);
    acb_init(rsumd);
    acb_init(rsumdd);

    arb_set(start, arb_mat_entry(Threadwork, id, 0));
    arb_set(stop, arb_mat_entry(Threadwork, id, 1));
    arb_set(step, arb_mat_entry(Threadwork, id, 2));
	
	arb_one(half);
	arb_mul_2exp_si(half, half, -1);

    acb_zero(rsum); acb_zero(rsumd); arb_set(i, start);

    while(arb_lt(i, stop))
    {
        arb_get_mid_arb(i, i);
        xi_integrand(res, i, prec);
        // Ht(z)
		acb_add(rsum, rsum, res, prec);

		//first derivative Ht'(z)
        acb_mul_arb(resd, res, i, prec);
		acb_add(rsumd, rsumd, resd, prec);
		
		//second derivative Ht''(z)
		arb_pow_ui(a, i, 2, prec);
		arb_sub(a, a, half, prec);
		acb_mul_arb(resdd, res, a, prec);
		acb_add(rsumdd, rsumdd, resdd, prec);
		
		arb_add(i, i, step, prec);
    }

    acb_set(acb_mat_entry(Rsums, id, 0), rsum);
    acb_set(acb_mat_entry(Rsums, id, 1), rsumd);
    acb_set(acb_mat_entry(Rsums, id, 2), rsumdd);

    arb_clear(a);
    arb_clear(i);
    arb_clear(half);
    arb_clear(start);
    arb_clear(stop);
    arb_clear(step);

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
Ht_Riemann_sum(acb_struct rs[3], arb_t step, slong numthreads, slong prec)
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

    acb_zero(rs); acb_zero(rs + 1); acb_zero(rs + 2);
    for (i = 0; i < numthreads; i++)
    {
        acb_add(rs + 0, rs + 0, acb_mat_entry(Rsums, i, 0), prec);
        acb_add(rs + 1, rs + 1, acb_mat_entry(Rsums, i, 1), prec);
        acb_add(rs + 2, rs + 2, acb_mat_entry(Rsums, i, 2), prec);
    }

    //rs*1/(8*sqrt(pi)*step) for Ht(z)
    acb_mul_2exp_si(rs, rs, -3);
    arb_const_pi(a, prec);
    arb_sqrt(a, a, prec);
    acb_div_arb(rs, rs, a, prec);
    acb_mul_arb(rs, rs, step, prec);

    //rsd*I/(8*sqrt(pi*t)*step) for Ht'(z)
    acb_mul_2exp_si(rs + 1, rs + 1, -3);
    acb_mul_onei(rs + 1, rs + 1);
    acb_set_arb(ac, t);
    acb_sqrt(ac, ac, prec);
    acb_mul_arb(ac, ac, a, prec);
    acb_div(rs + 1, rs + 1, ac, prec);
    acb_mul_arb(rs + 1, rs + 1, step, prec);
	
	//rsd*-1/(8*sqrt(pi)*t*step) for Ht''(z)
    acb_mul_2exp_si(rs + 2, rs + 2, -3);
    acb_set_arb(ac, t);
    acb_mul_arb(ac, ac, a, prec);
    acb_div(rs + 2, rs + 2, ac, prec);
	acb_neg(rs + 2, rs + 2);
    acb_mul_arb(rs + 2, rs + 2, step, prec);

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

//establish the partial derivative of dx/dt H_t(x,y)
void
Establish_partial_x_derivative(acb_t xder, arb_t x, arb_t y, slong prec)
{
	arb_t h, xs, ts;
	arb_init(h);
	arb_init(xs);
	arb_init(ts);
	
	arb_set_str(h, "0.000000000000000000000000000001", prec);
	
	acb_t fun, funh, deltat, deltax;
	acb_init(fun);
	acb_init(funh);
	acb_init(deltat);
	acb_init(deltax);
	
	acb_struct s[3];
    acb_init(s);
    acb_init(s + 1);
    acb_init(s + 2);
	
	arb_set(ts, t);
	acb_zero(xder);
    acb_zero(s);
    acb_zero(s + 1);
    acb_zero(s + 2);
	
	//f_t(x+iy)
	acb_set_arb_arb(z, x, y);
	Ht_Riemann_sum(s, step, numthreads, prec);
	acb_set(fun, s);

	//f_t(x+i(x+h))
	arb_add(xs, x, h, prec);
	acb_set_arb_arb(z, xs, y);
	Ht_Riemann_sum(s, step, numthreads, prec);
	acb_set(funh, s);

	//(f(x+i(x+h))-f(x+yi))/h
	acb_sub(deltax, funh, fun, prec);

	//f_t(x+i(t+h))
	arb_add(t, t, h, prec);
	acb_set_arb_arb(z, x, y);
	Ht_Riemann_sum(s, step, numthreads, prec);
	acb_set(funh, s);

	//(f(x+i(y+h))-f(x+yi))/h
	acb_sub(deltat, funh, fun, prec);
	
	acb_div(xder, deltat, deltax, prec);

	arb_set(t, ts);
	
	arb_clear(h);
	arb_clear(xs);
	arb_clear(ts);

	acb_clear(fun);
	acb_clear(funh);
	acb_clear(deltat);
	acb_clear(deltax);
	
    acb_clear(s);
    acb_clear(s + 1);
    acb_clear(s + 2);
}

//establish Ht(z_k), Ht'(z_k), Ht''(z_k)
void
Establish_z_accent_k(acb_t zder, arb_t x, arb_t y, slong prec)
{	
	acb_struct s[3];
    acb_init(s);
    acb_init(s + 1);
    acb_init(s + 2);
	
    acb_zero(s);
    acb_zero(s + 1);
    acb_zero(s + 2);
	
	acb_zero(zder);
	
	//obtain Ht(z_k), Ht'(z_k), Ht''(z_k)
	acb_set_arb_arb(z, x, y);
	Ht_Riemann_sum(s, step, numthreads, prec);

	//compute z'_k(t) = Ht''(z_k)/Ht'(z_k) 
	acb_div(zder, s + 2, s + 1, prec);
	
    acb_clear(s);
    acb_clear(s + 1);
    acb_clear(s + 2);
}

//compute the sum of the zeros left and right of ks with y_k > 0.
void
Compute_dysum_zerosleft_right(slong res, slong kl, slong kh, slong ks, slong prec)
{    
	slong j;
	
	arb_t xk, yk, yk2, summand, zerosum;
	arb_init(xk);
	arb_init(yk);
	arb_init(yk2);
	arb_init(summand);
	arb_init(zerosum);
	
	//take midpoint between two subsequent zeros as the start
	arb_add(xk, arb_mat_entry(Zerorangetprev, ks - kl, 0), arb_mat_entry(Zerorangetprev, ks - kl + 1, 0), prec);
	arb_mul_2exp_si(xk, xk, -1);
	
	arb_set(yk, arb_mat_entry(Zerorangetprev, ks - kl, 1));
	arb_pow_ui(yk2, yk, 2, prec);

	arb_zero(summand);
	arb_zero(zerosum);
	for (j = kl; j < kh + 1; j++)
    {
		if (!(j == ks) && !(j == ks + 1))
		{
			arb_sub(summand, xk, arb_mat_entry(Zerorangetprev, j - kl, 0), prec);
			arb_pow_ui(summand, summand, 2, prec);
			arb_add(summand, summand, yk2, prec);
			arb_div(summand, yk, summand, prec);
			arb_add(zerosum, zerosum, summand, prec);
		}

	}
	
    arb_mul_2exp_si(zerosum, zerosum, 1);
	
	arb_inv(yk, yk, prec);
	arb_neg(yk, yk);
	arb_set(arb_mat_entry(Zerorangetprev, ks - kl, 5), yk);
	arb_set(arb_mat_entry(Zerorangetprev, ks - kl, 6), zerosum);
	
	arb_clear(xk);
	arb_clear(yk);
	arb_clear(yk2);
	arb_clear(summand);
	arb_clear(zerosum);
}

//compute the sum of the zeros left and right of ks with y_k = 0.
void
Compute_dxsum_zerosleft_right(slong res, slong kl, slong kh, slong ks, slong prec)
{    
	slong j;
	
	arb_t a, xk, summand, summand1, zerosum;
	arb_init(a);
	arb_init(xk);
	arb_init(summand);
	arb_init(summand1);
	arb_init(zerosum);
	
	arb_set(xk, arb_mat_entry(Zerorangetprev, ks - kl, 0));

	arb_zero(summand);
	arb_zero(zerosum);
	for (j = kl; j < kh + 1; j++)
    {
		if (!(j == ks))
		{
			//1/ (xk-xj)
			arb_sub(summand, xk, arb_mat_entry(Zerorangetprev, j - kl, 0), prec);
			arb_inv(summand, summand, prec);
			
			//include the negative x as 1/(xk+xj) if needed (small effect)
			//arb_add(summand1, xk, arb_mat_entry(Zerorangetprev, j - kl, 0), prec);
			//arb_inv(summand1, summand1, prec);
			//arb_add(summand, summand, summand1, prec);
			
			arb_add(zerosum, zerosum, summand, prec);
		}
	}
	
    arb_mul_2exp_si(zerosum, zerosum, 1);
	
	//add the -Pi/4 factor that comes from higher pressure on the left of xs (real zeros travel north west).
	arb_const_pi(a, prec);
	arb_mul_2exp_si(a, a, -2);
	arb_neg(a, a);
	
	arb_add(zerosum, zerosum, a, prec);

	arb_set(arb_mat_entry(Zerorangetprev, ks - kl, 4), zerosum);
	
	arb_clear(a);
	arb_clear(xk);
	arb_clear(summand);
	arb_clear(summand1);
	arb_clear(zerosum);
}

//print all data for the current t
void
Print_all_data_t(slong res, slong kl, slong kh, slong ks, slong sw, slong prec)
{    
	slong j;
	arb_t a;
	arb_init(a);
	
	for (j = kl; j < kh + 1; j++)
    {
		if( (sw == 0) || ((sw ==1) && (j == ks)))
		{
			arb_printn(t, 5, ARB_STR_NO_RADIUS);
			printf(", %ld, ", j);
			arb_printn(arb_mat_entry(Zerorangetprev, j - kl, 0), 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_printn(arb_mat_entry(Zerorangetprev, j - kl, 1), 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_printn(arb_mat_entry(Zerorangetprev, j - kl, 2), 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_printn(arb_mat_entry(Zerorangetprev, j - kl, 3), 20, ARB_STR_NO_RADIUS);
		
			if (sw == 1)
			{
				printf(", ");
				arb_printn(arb_mat_entry(Zerorangetprev, j - kl, 4), 20, ARB_STR_NO_RADIUS);
				printf(", ");
				arb_printn(arb_mat_entry(Zerorangetprev, j - kl, 5), 20, ARB_STR_NO_RADIUS);
				printf(", ");
				arb_printn(arb_mat_entry(Zerorangetprev, j - kl, 6), 20, ARB_STR_NO_RADIUS);
				printf(", ");
				arb_sub(a, arb_mat_entry(Zerorangetprev, j - kl, 5), arb_mat_entry(Zerorangetprev, j - kl, 6), prec);
				arb_printn(a, 20, ARB_STR_NO_RADIUS);
				printf("\n");
			}
			if(sw == 0)
				printf("\n");
		}
		
		arb_zero(arb_mat_entry(Zerorangetprev, j - kl, 4));
	}

	arb_clear(a);
}

//initial fill of the global matrix with zeros of H_0(x) 
void
Initial_fill_with_zeros_at_t0(slong out, slong kl, slong kh, slong ks, slong sw, slong prec)
{    
	slong i, res; res = 0;
	
	fmpz_t n;
	fmpz_init(n);
	
	arb_t x, y;
	arb_init(x);
	arb_init(y);
	
	acb_t zder, root;
	acb_init(zder);
	acb_init(root);
	
	acb_zero(zder);
	
	for (i = kl; i < kh + 1; i++)
    {
		fmpz_set_si(n, i);
		acb_dirichlet_zeta_zero(root, n, prec);
		acb_get_imag(x, root);

		//multiply by 2 to move into Ht domain
		arb_mul_2exp_si(x, x, 1);
		arb_zero(y);
		
		//store zero Ht(x)
		arb_set(arb_mat_entry(Zerorangetprev, i - kl, 0), x);
		arb_set(arb_mat_entry(Zerorangetprev, i - kl, 1), y);

		//establish and store z'_k(t)
		Establish_z_accent_k(zder, x, y, prec);
		acb_get_real(x, zder);
		arb_zero(y);
		arb_set(arb_mat_entry(Zerorangetprev, i - kl, 2), x);
		arb_set(arb_mat_entry(Zerorangetprev, i - kl, 3), y);
	}

	if (sw == 1)
		Compute_dxsum_zerosleft_right(res, kl, kh, ks, prec);

	arb_zero(t);
	Print_all_data_t(res, kl, kh, ks, sw, prec);

	fmpz_clear(n);
	
	arb_clear(x);
	arb_clear(y);
	
	acb_clear(zder);
	acb_clear(root);
}

// f(z) = Ht(z) / Ht'(z)
void
f_Ht_frac1(acb_t logder, slong prec)
{
    acb_struct s[3];
    acb_init(s);
    acb_init(s + 1);
    acb_init(s + 2);

    Ht_Riemann_sum(s, step, numthreads, prec);
    acb_div(logder,s ,s + 1 , prec);

    acb_clear(s);
    acb_clear(s + 1);
    acb_clear(s + 2);
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
        f_Ht_frac1(h, prec);
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
FindRoots(acb_t root, arb_t xs, arb_t ys, arb_t acc, slong prec)
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
	}

end:

    arb_clear(a);
    arb_clear(x);
    arb_clear(rootre);
    arb_clear(rootim);
    arb_clear(res);

    acb_clear(x0);
}

void
Establish_settings_Riemann_sum(arb_t intlim, arb_t acc, slong kh, arb_t te, slong prec)
{
	slong res; res = 0;
	
	fmpz_t n;
	fmpz_init(n);
	
	arb_t a, b, curr, prev, diff, prevdiff, xe, ye;
	arb_init(a);
	arb_init(b);
	arb_init(curr);
	arb_init(prev);
	arb_init(diff);
	arb_init(prevdiff);
	arb_init(xe);
	arb_init(ye);
	
	acb_t root;
	acb_init(root);
	
	acb_struct s[3];
    acb_init(s);
    acb_init(s + 1);
    acb_init(s + 2);
	
	printf("Establishing required accuracy for the Ht Riemann summation...\n");
	
	//establsih the maximum x value of the range (Ht has 2 times the x of a zeta zero) 
	fmpz_set_si(n, kh);
	acb_dirichlet_zeta_zero(root, n, prec);
	acb_get_imag(xe, root);
	arb_mul_2exp_si(xe, xe, 2);
	
	//dervive the max y from te, xe via the curve formula -te/2*log(x/(8*pi)) - 1 
	arb_zero(ye);
	if (arb_is_negative(te))
	{
		arb_mul_2exp_si(a, te, -1);
		arb_neg(a, a);
	
		arb_const_pi(b, prec);
		arb_mul_2exp_si(b, b, 3);
		arb_div(b, xe, b, prec);
		arb_log(b, b, prec);
		arb_mul(a, a, b, prec);
		arb_sub_si(ye, a, 1, prec);
	}
	
	arb_set_str(intlim, "6", prec); arb_set_str(step, "0.1", prec);
	Allocate_worktothreads(res, intlim, step, 1, prec);
	
	arb_set(t, te);
	arb_set_si(diff, 1);
	arb_zero(prev);
	
	//establish intlim
	while(arb_gt(diff, acc))
	{
		//obtain Ht(z)
		Allocate_worktothreads(res, intlim, step, 1, prec);
		acb_set_arb_arb(z, xe, ye);
		Ht_Riemann_sum(s, step, 1, prec);
		acb_div(s, s, s + 1, prec); 
		acb_abs(curr, s, prec);
		arb_sub(diff, curr, prev, prec);
		arb_abs(diff, diff);
	arb_printd(curr, 20);
	printf("diff\n");
		arb_add_si(intlim, intlim, 1, prec);
		arb_set(prev, curr);
	}

	arb_add_si(intlim, intlim, 1, prec);
	printf("Done!\n");

	arb_printd(xe, 20);
	printf(", ");
	arb_printd(ye, 20);
	printf(", ");
	arb_printd(te, 20);
	printf("\n");
	
	arb_printd(intlim, 20);
	printf(", ");
	arb_printd(step, 20);
	printf("\n");

	fmpz_clear(n);
	
	arb_clear(a);
	arb_clear(b);
	arb_clear(curr);
	arb_clear(prev);
	arb_clear(diff);
	arb_clear(prevdiff);
	arb_clear(xe);
	arb_clear(ye);
	
	acb_clear(root);
	
	acb_clear(s + 0);
    acb_clear(s + 1);
    acb_clear(s + 2);
}

int main(int argc, char *argv[])
{
    arb_t a, acc, x, y, xs, ys, intlim, ta, te, ts, ti, outcome;
    arb_init(a);
    arb_init(acc);
    arb_init(x);
    arb_init(y);
    arb_init(xs);
    arb_init(ys);
    arb_init(intlim);
    arb_init(ta);
    arb_init(ts);
    arb_init(te);
    arb_init(ti);
    arb_init(outcome);

	//init the global variables
    arb_init(step);
    arb_init(t);
    acb_init(z);

	acb_t zder, root;
	acb_init(zder);
	acb_init(root);

    const char *ks_str, *kr_str, *te_str, *ti_str, *sw_str, *numthreads_str;
	
    slong i, kl, kh, kr, ks, prec, res, sw; res = 0;

    int result = EXIT_SUCCESS;

    if (argc != 7)
    {
        result = EXIT_FAILURE;
        goto finish;
    }

    ks_str = argv[1];
    kr_str = argv[2];
    te_str = argv[3];
    ti_str = argv[4];
    sw_str = argv[5];
    numthreads_str = argv[6];

    //Working precision 
    prec = 20 * 3.32192809488736 + 210;

    ks = atol(ks_str);
    kr = atol(kr_str);
    arb_set_str(te, te_str, prec);
    arb_set_str(ti, ti_str, prec);
    sw = atol(sw_str);
    numthreads = atol(numthreads_str);
	
	//establish range kl .. kh with ks as midpoint
	kl = ks - kr;
	kh = ks + kr;

    if(kr == 0 && sw == 1)
    {
		printf("\n This option requires a positive range of zeros. \n\n");
        result = EXIT_FAILURE; 
        goto finish;
    }
	
	if(ks < 1 || (sw != 0 && sw != 1) || kr < 0 || numthreads < 1 || numthreads > 64 || (arb_is_negative(te) && arb_is_positive(ti)) 
		|| (arb_is_negative(ti) && arb_is_positive(te)) || arb_is_zero(te) || arb_is_zero(ti))
    {
		printf("Invalid option chosen. \n");
        result = EXIT_FAILURE; 
        goto finish;
    }
	
	arb_mat_init(Threadwork, numthreads, 3);
	arb_mat_init(Zerorangetprev, kh - kl + 1, 7);
	
	acb_mat_init(Rsums, numthreads, 3);

	// set target accuracy for all Newton-Raphson rootfinding
	arb_set_str(acc, "0.0000000001", prec); 

	//Estabish step and intlim parameters to assure target accuracy is met.
	//Establish_settings_Riemann_sum(intlim, acc, kh, te, prec);
	
	arb_set_str(intlim, "10", prec); arb_set_str(step, "0.1", prec);

	Allocate_worktothreads(res, intlim, step, numthreads, prec);

	//initialise t very close to zero to use the Ht-derivative formulae (with a div t)
    arb_set_str(t, "0.0000000000000000000000001", prec);

	//fill the startingvalues with the Ht-zeros at t = 0
	Initial_fill_with_zeros_at_t0(res, kl, kh, ks, sw, prec);

	//loop through the t range
	arb_abs(te, te);
	arb_abs(ta, t);

	while(arb_lt(ta, te))
	{
		arb_add(t, t, ti, prec);
		arb_abs(ta, t);

		for (i = kl; i < kh + 1; i++)
		{
			//set xs and ys to the coodinates of the zero found and the previous t
			arb_set(xs, arb_mat_entry(Zerorangetprev, i - kl, 0));
			arb_set(ys, arb_mat_entry(Zerorangetprev, i - kl, 1));
			
			//if the prev y-value = 0, then lift it up a bit to ensure complex zeros are found after collision
			arb_abs(a, ys);
			if (arb_le(a, acc))
					arb_set_str(ys, "0.05", prec); 
						
			FindRoots(root, xs, ys, acc, prec);
			acb_get_real(x, root);
			acb_get_imag(y, root);

			arb_abs(a, y);
			if (arb_lt(a, acc))
				arb_zero(y);
				
			//make y negative when the y_k = y_k-1, i.e. zero k+1 of the colliding pair becomes the conjugate 
			if (i - kl -1 >= 0)
				arb_sub(a, arb_mat_entry(Zerorangetprev, i - kl - 1, 1), y, prec);
			
			arb_abs(a, a);
			if (arb_le(a, acc))
				arb_neg(y, y);
		
			//store zero Ht(x+yi)
			arb_set(arb_mat_entry(Zerorangetprev, i - kl, 0), x);
			arb_set(arb_mat_entry(Zerorangetprev, i - kl, 1), y);
		
			//establish and store z'_k(t)
			Establish_z_accent_k(zder, x, y, prec);
			acb_get_real(xs, zder);
			acb_get_imag(ys, zder);
			arb_abs(a, ys);
			if (arb_lt(a, acc))
				arb_zero(ys);
			arb_set(arb_mat_entry(Zerorangetprev, i - kl, 2), xs);
			arb_set(arb_mat_entry(Zerorangetprev, i - kl, 3), ys);
		}

		if (sw == 1)
		{
			//if sw == 1 then if Im(z_ks) > 0 compute y'_k(t) through the zerosums. Otherwise compute x'_k(t)
			if (arb_is_positive(arb_mat_entry(Zerorangetprev, ks - kl, 1)))
				Compute_dysum_zerosleft_right(res, kl, kh, ks, prec);
			else
				Compute_dxsum_zerosleft_right(res, kl, kh, ks, prec);		
		}
		
		Print_all_data_t(res, kl, kh, ks, sw, prec);
	}
	
finish:
 
    if (result == EXIT_FAILURE)
    {
        flint_printf("Required inputs:\n");
        flint_printf("%s kstart, kr, tend, tinc, switch, numthreads \n\n", argv[0]);
        flint_printf(
    "This script traces the k_th real or complex zero of H_t(x+yi) over time t = 0 .. tend in steps of tinc.\n"
    "With tend and tinc > 0 it explores the postive t-region and with tend and tinc < 0 the negative t-region.\n"
	"Multiple k could be traced simultaneoulsy bij specifying kr, so that: k_range = kstart - kr ... kstart + kr.\n"
    "\n"
    "switch = 0, output for all k in range : t, k, x_k, y_k, velocities x'_k, y'_k.\n"
    "switch = 1, output only for kstart    : t, k, x_k, y_k, velocities x'_k, y'_k, estimated x'_k, -1/y_k, zerosums, estimated y'_k.\n"
    "\n"
    "Accuracy of the zeros has been set at 10 digits and with the current parameters the range t = -10 .. 10 could be covered.\n"
    "The number of threads chosen helps to speed up the Riemann-sum evaluation of the Ht-integral.\n\n");
    }

    arb_clear(a);
    arb_clear(acc);
    arb_clear(x);
    arb_clear(y);
    arb_clear(xs);
    arb_clear(ys);
    arb_clear(intlim);
    arb_clear(step);
    arb_clear(t);
    arb_clear(ta);
    arb_clear(ts);
    arb_clear(te);
    arb_clear(ti);
    arb_clear(outcome);

	acb_clear(zder);
	acb_clear(root);
    acb_clear(z);

    arb_mat_clear(Threadwork);
    arb_mat_clear(Zerorangetprev);
	
    acb_mat_clear(Rsums);
 
    flint_cleanup();

    return result;
}
