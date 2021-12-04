/*
    Copyright (C) 2019 D.H.J. Polymath
    This file is part of Arb.
    Arb is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.  See <http://www.gnu.org/licenses/>.
*/

#include "acb_dirichlet.h"
#include "arb_mat.h"
#include "flint/profiler.h"
#include "pthread.h"

void Xi(acb_t res, arb_t d, slong prec)
{
	arb_t half;
	arb_init(half);
	
	acb_t ac;
	acb_init(ac);
	
	arb_one(half);
	arb_mul_2exp_si(half, half, -1);
	
	acb_set_arb_arb(ac, half, d);
    acb_dirichlet_xi(res, ac, prec);
	
	arb_clear(half);
	
	acb_clear(ac);
}

void nthderivXi(acb_t dersum, arb_t h, arb_t t, slong n, slong prec)
{
    slong k;
	
	arb_t a, b, c, d, min1;
	arb_init(a);
	arb_init(b);
	arb_init(c);
	arb_init(d);
	arb_init(min1);
	
	acb_t res;
	acb_init(res);
	
	arb_set_si(min1, -1);
	
	arb_pow_ui(a, h, n, prec);
	arb_inv(a, a, prec);
	
	acb_zero(dersum);

    for (k = 0; k < n + 1; k++)
    {
		arb_pow_ui(b, min1, k + n, prec);
		arb_bin_uiui(c, n, k, prec);
		arb_mul(c, c, b, prec);
		
		arb_mul_si(d, h, k, prec);
		arb_add(d, d, t, prec);
		Xi(res, d, prec);
		
		acb_mul_arb(res, res, c, prec);
		acb_add(dersum, dersum, res, prec);
	}
	
	acb_mul_arb(dersum, dersum, a, prec);
	
	arb_clear(a);
	arb_clear(b);
	arb_clear(c);
	arb_clear(d);
	arb_clear(min1);
	
	acb_clear(res);
}

void PXi(acb_t res, arb_t h, arb_t t, slong prec)
{	
	acb_t ac, bc;
	acb_init(ac);
	acb_init(bc);
	
	nthderivXi(ac, h, t, 2, prec);
	nthderivXi(bc, h, t, 1, prec);
	
	acb_div(res, ac, bc, prec);
	
	acb_clear(ac);
	acb_clear(bc);
}

void PXid(acb_t dersum, arb_t h, arb_t t, slong prec)
{
    slong k;
	
	arb_t a, b, c, d, min1;
	arb_init(a);
	arb_init(b);
	arb_init(c);
	arb_init(d);
	arb_init(min1);
	
	acb_t res;
	acb_init(res);
	
	arb_set_si(min1, -1);
	
	arb_inv(a, h, prec);
	
	acb_zero(dersum);

    for (k = 0; k < 2; k++)
    {
		arb_pow_ui(b, min1, k + 1, prec);
		arb_bin_uiui(c, 1, k, prec);
		arb_mul(c, c, b, prec);
		
		arb_mul_si(d, h, k, prec);
		arb_add(d, d, t, prec);
		PXi(res, h, d, prec);
		
		acb_mul_arb(res, res, c, prec);
		acb_add(dersum, dersum, res, prec);
	}
	
	acb_mul_arb(dersum, dersum, a, prec);
	
	arb_clear(a);
	arb_clear(b);
	arb_clear(c);
	arb_clear(d);
	arb_clear(min1);
	
	acb_clear(res);
}

void ComputeTarget(arb_t out, arb_t yplus, arb_t ymin, arb_t h, slong prec)
{
	
	arb_t a, b, delt2;
	arb_init(a);
	arb_init(b);
	arb_init(delt2);

	arb_sub(delt2, yplus, ymin, prec);
	arb_pow_ui(delt2, delt2, 2, prec);
	
	acb_t res;
	acb_init(res);
	
	PXid(res, h, yplus, prec);
	acb_get_real(a, res);
	PXid(res, h, ymin, prec);
	acb_get_real(b, res);
	
	arb_add(a, a, b, prec);
	arb_mul(a, a, delt2, prec);
	arb_neg(out, a);
	
	arb_clear(a);
	arb_clear(b);
	arb_clear(delt2);
	
	acb_clear(res);
}

int main(int argc, char *argv[])
{

	slong i, prec;
	
	const char *t_str;

	prec = 1024;
	
	arb_t h, t, outcome, yplus, ymin, delta, t42d5;
	arb_init(h);
	arb_init(t);
	arb_init(outcome);
	arb_init(yplus);
	arb_init(ymin);
	arb_init(delta);
	arb_init(t42d5);
	
	acb_t res;
	acb_init(res);
	
	fmpz_t n;
    fmpz_init(n);	

    t_str = argv[1];
    arb_set_str(t, t_str, prec);
	
	arb_set_str(h, "0.000000000000000000000000000000000000000000000000000000000000000000000000001", prec);

	TIMEIT_ONCE_START
	
	arb_set_si(yplus, 42);
	arb_set_si(ymin, 5);
	arb_div(t42d5, yplus, ymin, prec);

    for (i = 1; i < 600001; i++)
    {
		fmpz_set_si(n, i);
		acb_dirichlet_zeta_zero(res, n, prec);
		acb_get_imag(ymin, res);
		
		fmpz_set_si(n, i + 1);
		acb_dirichlet_zeta_zero(res, n, prec);
		acb_get_imag(yplus, res);

		ComputeTarget(outcome, yplus, ymin, h,prec);
		
		if(arb_le(outcome, t42d5))
		{
			printf("%ld", i);
			printf(", ");
			arb_printn(ymin, 20, ARB_STR_NO_RADIUS);
			printf(", ");
			printf("%ld", i + 1);
			printf(", ");
			arb_printn(yplus, 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_sub(delta, yplus, ymin, prec);
			arb_printn(delta, 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_printn(outcome, 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_mul_2exp_si(ymin, ymin, 1);
			arb_printn(ymin, 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_mul_2exp_si(yplus, yplus, 1);
			arb_printn(yplus, 20, ARB_STR_NO_RADIUS);
			printf(", ");
			arb_sub(delta, yplus, ymin, prec);
			arb_printn(delta, 20, ARB_STR_NO_RADIUS);
			printf("\n");
		}
    }

	TIMEIT_ONCE_STOP
	
	arb_clear(h);
	arb_clear(t);
	arb_clear(outcome);
	arb_clear(yplus);
	arb_clear(ymin);
	arb_clear(delta);
	arb_clear(t42d5);
	
	acb_clear(res);

    fmpz_init(n);	

    flint_cleanup();

    return EXIT_SUCCESS;
}