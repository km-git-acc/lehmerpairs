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
#include "acb_poly.h"
#include "flint/profiler.h"
#include "pthread.h"

struct ThreadData {
    slong id, prec;
}; 

//start of ABCEFF code
void acb_poly_onei(acb_poly_t res)
{
    acb_poly_one(res);
    acb_onei(acb_poly_get_coeff_ptr(res, 0));
}

void
acb_poly_re_inplace(acb_poly_t p)
{
    slong i;
    for (i = 0; i < acb_poly_length(p); i++)
    {
        acb_ptr z = acb_poly_get_coeff_ptr(p, i);
        arb_zero(acb_imagref(z));
    }
}
 
void
acb_poly_re(acb_poly_t p, const acb_poly_t q)
{
    if (p != q)
        acb_poly_set(p, q);
 
    acb_poly_re_inplace(p);
}
 
void
acb_poly_im_inplace(acb_poly_t p)
{
    slong i;
    for (i = 0; i < acb_poly_length(p); i++)
    {
        acb_ptr z = acb_poly_get_coeff_ptr(p, i);
        arb_set(acb_realref(z), acb_imagref(z));
        arb_zero(acb_imagref(z));
    }
}
 
void
acb_poly_im(acb_poly_t p, const acb_poly_t q)
{
    if (p != q)
        acb_poly_set(p, q);
 
    acb_poly_im_inplace(p);
}

//procedure to calculate the conjugate of an acb_poly function 
void
acb_poly_conj_series(acb_poly_t res, const acb_poly_t p, slong n, slong prec)
{
    acb_poly_t a, b;
    acb_t onei;
 
    acb_poly_init(a);
    acb_poly_init(b);
    acb_init(onei);
    acb_onei(onei);
 
    acb_poly_re(a, p);
    
    acb_poly_im(b, p);
    acb_poly_neg(b, b);
    acb_poly_scalar_mul(b, b, onei, prec);
    acb_poly_add_series(res, a, b, n, prec);

    acb_poly_clear(a);
    acb_poly_clear(b);
    acb_clear(onei);
}
 
void _acb_poly_alpha1_series(acb_poly_t z, const acb_poly_t s,
        slong n, slong prec)
{
    acb_poly_t a;
    acb_poly_init(a);
 
    acb_poly_zero(z);
 
    acb_poly_inv_series(a, s, n, prec);
    acb_poly_scalar_mul_2exp_si(a, a, -1);
    acb_poly_add_series(z, z, a, n, prec);
 
    acb_poly_add_si(a, s, -1, prec);
    acb_poly_inv_series(a, a, n, prec);
    acb_poly_add_series(z, z, a, n, prec);
 
    acb_poly_log_series(a, s, n, prec);
    acb_poly_scalar_mul_2exp_si(a, a, -1);
    acb_poly_add_series(z, z, a, n, prec);
 
    {
        arb_t c;
        arb_ptr u;
        arb_init(c);
        arb_const_log_sqrt2pi(c, prec);
        u = acb_realref(acb_poly_get_coeff_ptr(z, 0));
        arb_sub(u, u, c, prec);
        arb_clear(c);
    }
 
    acb_poly_clear(a);
}
 
void acb_poly_alpha1_series(acb_poly_t z, const acb_poly_t s,
        slong n, slong prec)
{
    if (z == s)
    {
        acb_poly_t u;
        acb_poly_init(u);
        _acb_poly_alpha1_series(u, s, n, prec);
        acb_poly_swap(z, u);
        acb_poly_clear(u);
    }
    else
    {
        _acb_poly_alpha1_series(z, s, n, prec);
    }
}
 
void _acb_poly_H01_series(acb_poly_t z, const acb_poly_t s,
        slong n, slong prec)
{
    arb_t c;
    acb_poly_t a, b;
    acb_t u;
    arb_ptr x;
 
    acb_poly_init(a);
    acb_poly_init(b);
 
    acb_init(u);
 
    arb_init(c);
    arb_const_log_sqrt2pi(c, prec);
 
    acb_poly_zero(z);
 
    acb_poly_add_si(a, s, -1, prec);
    acb_poly_log_series(a, a, n, prec);
    acb_poly_add_series(z, z, a, n, prec);
 
    acb_zero(u);
    x = acb_realref(u);
    arb_log_ui(x, 2, prec);
    arb_neg(x, x);
    arb_add_ui(x, x, 1, prec);
    arb_mul_2exp_si(x, x, -1);
    arb_add(x, x, c, prec);
    acb_poly_scalar_mul(a, s, u, prec);
    acb_poly_sub_series(z, z, a, n, prec);
 
    x = acb_realref(acb_poly_get_coeff_ptr(z, 0));
    arb_add(x, x, c, prec);
 
    acb_poly_add_si(b, s, 1, prec);
    acb_poly_scalar_mul_2exp_si(a, s, -1);
    acb_poly_log_series(a, a, n, prec);
    acb_poly_scalar_mul_2exp_si(a, a, -1);
    acb_poly_mullow(a, a, b, n, prec);
    acb_poly_add_series(z, z, a, n, prec);
 
    acb_poly_exp_series(z, z, n, prec);
 
    acb_poly_clear(a);
    acb_poly_clear(b);
    acb_clear(u);
    arb_clear(c);
}
 
void acb_poly_H01_series(acb_poly_t z, const acb_poly_t s,
        slong n, slong prec)
{
    if (z == s)
    {
        acb_poly_t u;
        acb_poly_init(u);
        _acb_poly_H01_series(u, s, n, prec);
        acb_poly_swap(z, u);
        acb_poly_clear(u);
    }
    else
    {
        _acb_poly_H01_series(z, s, n, prec);
    }
}
 
static void
_acb_poly_abbeff_summand_series(acb_poly_t res,
        const acb_t logk, const acb_poly_t a, const acb_poly_t b,
        slong n, slong prec)
{
    acb_poly_scalar_mul(res, a, logk, prec);
    acb_poly_sub_series(res, res, b, n, prec);
    acb_poly_scalar_mul(res, res, logk, prec);
    acb_poly_exp_series(res, res, n, prec);
}

void
abbeff_series(acb_t res, const arb_t rx, const arb_t ry, const arb_t rt, slong sw, slong n, slong prec)
{
    slong N;
    arb_t ar;
    acb_t one, pi, tmp, logk;
    acb_poly_t x, y, t;
    acb_poly_t s, sc, sprime, scprime;
    acb_poly_t alph1, alph2;
    acb_poly_t hs, hsc;
    acb_poly_t ta1, ta2;
    acb_poly_t A0, B0;
    acb_poly_t a, b, c, d, b1, b2;
    acb_poly_t summand;
    acb_poly_t Asum, Bsum;
    acb_poly_t A, B, C;
    acb_poly_t termC1, termC2;
    acb_poly_t alf, p, polyN, C0p, C0pc, U, Uc;
    acb_poly_t sig, T, Tprime;
    acb_poly_t onei, polyone, two, three, pip;

    acb_poly_init(a); 
    acb_poly_init(b); 
    acb_poly_init(c); 
    acb_poly_init(d); 
    acb_poly_init(x); 
    acb_poly_init(y); 
    acb_poly_init(t); 
    acb_poly_init(sig);
    acb_poly_init(T);
    acb_poly_init(Tprime);
    acb_poly_init(s);
    acb_poly_init(sc);
    acb_poly_init(sprime);
    acb_poly_init(scprime);
    acb_poly_init(alf); 
    acb_poly_init(p); 
    acb_poly_init(polyN); 
    acb_poly_init(C0p); 
    acb_poly_init(C0pc); 
    acb_poly_init(U);
    acb_poly_init(Uc);
    acb_poly_init(termC1);
    acb_poly_init(termC2);


    acb_poly_init(onei);
    acb_poly_onei(onei);

    acb_init(tmp);
    acb_init(one);
    acb_set_si(one, 1);
    acb_init(pi);
    acb_const_pi(pi, prec);

    acb_poly_init(polyone);
    acb_poly_set_si(polyone, 1);
    acb_poly_init(two);
    acb_poly_set_si(two, 2);
    acb_poly_init(three);
    acb_poly_set_si(three, 3);
    acb_poly_init(pip);
    acb_poly_set_acb(pip, pi);

    arb_init(ar);

    //convert arb-inputs to acb_poly variables
    acb_poly_one(x);
    arb_set(acb_realref(acb_poly_get_coeff_ptr(x, 0)), rx);
    acb_poly_one(y);
    arb_set(acb_realref(acb_poly_get_coeff_ptr(y, 0)), ry);
    acb_poly_one(t);
    arb_set(acb_realref(acb_poly_get_coeff_ptr(t, 0)), rt);

    //T
    acb_poly_scalar_mul_2exp_si(T, x, -1);

    //Tprime
    acb_poly_mullow(b, pip, t, n, prec);
    acb_poly_scalar_mul_2exp_si(b, b, -3);
    acb_poly_add_series(Tprime, b, T, n, prec);

    //N
    acb_poly_scalar_mul_2exp_si(a, pip, 1);
    acb_poly_div_series(a, Tprime, a, n, prec);
    acb_poly_sqrt_series(alf, a, n, prec);
    acb_poly_evaluate(tmp, alf, one, prec);
    acb_get_real(ar, tmp);
    arb_floor(ar, ar, prec);
    N = arf_get_d(arb_midref(ar), ARF_RND_DOWN);
    acb_poly_set_si(polyN, N);

    //s & sprime
    acb_poly_neg(a, y);
    acb_poly_add_si(a, a, 1, prec);
    acb_poly_scalar_mul_2exp_si(sig, a, -1);
    acb_poly_mullow(b, onei, T, n, prec);
    acb_poly_add_series(s, sig, b, n, prec);
    acb_poly_mullow(b, onei, Tprime, n, prec);
    acb_poly_add_series(sprime, sig, b, n, prec);

    acb_poly_add_si(sc, s, -1, prec);
    acb_poly_neg(sc, sc);

    acb_poly_add_si(scprime, sprime, -1, prec);
    acb_poly_neg(scprime, scprime);
 
    acb_poly_init(alph1);
    acb_poly_alpha1_series(alph1, s, n, prec);
 
    acb_poly_init(alph2);
    acb_poly_alpha1_series(alph2, sc, n, prec);
 
    acb_poly_init(hs);
    acb_poly_H01_series(hs, s, n, prec);
 
    acb_poly_init(hsc);
    acb_poly_H01_series(hsc, sc, n, prec);
 
    acb_poly_init(ta1);
    acb_poly_mullow(ta1, t, alph1, n, prec);
 
    acb_poly_init(ta2);
    acb_poly_mullow(ta2, t, alph2, n, prec);
 
    acb_poly_init(A0);
    acb_poly_mullow(A0, ta1, alph1, n, prec);
    acb_poly_scalar_mul_2exp_si(A0, A0, -2);
    acb_poly_exp_series(A0, A0, n, prec);
    acb_poly_mullow(A0, A0, hs, n, prec);
    acb_poly_scalar_mul_2exp_si(A0, A0, -3);
 
    acb_poly_init(B0);
    acb_poly_mullow(B0, ta2, alph2, n, prec);
    acb_poly_scalar_mul_2exp_si(B0, B0, -2);
    acb_poly_exp_series(B0, B0, n, prec);
    acb_poly_mullow(B0, B0, hsc, n, prec);
    acb_poly_scalar_mul_2exp_si(B0, B0, -3);
 
    acb_poly_scalar_mul_2exp_si(a, t, -2);
 
    acb_poly_init(b1);
    acb_poly_scalar_mul_2exp_si(b1, ta1, -1);
    acb_poly_add_series(b1, b1, s, n, prec);
 
    acb_poly_init(b2);
    acb_poly_scalar_mul_2exp_si(b2, ta2, -1);
    acb_poly_add_series(b2, b2, sc, n, prec);
 
    acb_init(logk);
    acb_poly_init(summand);
    acb_poly_init(Asum);
    acb_poly_init(Bsum);
    {
        slong k;
        for (k = N; k >= 2; k--)
        {
            acb_set_si(logk, k);
            acb_log(logk, logk, prec);
 
            _acb_poly_abbeff_summand_series(summand, logk, a, b1, n, prec);
            acb_poly_add_series(Asum, Asum, summand, n, prec);
 
            _acb_poly_abbeff_summand_series(summand, logk, a, b2, n, prec);
            acb_poly_add_series(Bsum, Bsum, summand, n, prec);
        }
    }
    acb_poly_add_si(Asum, Asum, 1, prec);
    acb_poly_add_si(Bsum, Bsum, 1, prec);
 
    acb_poly_init(A);
    acb_poly_mullow(A, A0, Asum, n, prec);
 
    acb_poly_init(B);
    acb_poly_mullow(B, B0, Bsum, n, prec);

    //calculate C-term if required
    acb_poly_init(C);
    acb_poly_zero(C);
    if (sw == 1)
    {
       //p
       acb_poly_sub_series(b, alf, polyN, n, prec);
       acb_poly_scalar_mul_2exp_si(b, b, 1);
       acb_poly_sub_series(p, polyone, b, n, prec);

       //U
       acb_poly_scalar_mul_2exp_si(a, pip, 1);
       acb_poly_div_series(a, Tprime, a, n, prec);
       acb_poly_log_series(a, a, n, prec);
       acb_poly_scalar_mul_2exp_si(b, Tprime, -1);
       acb_poly_mullow(a, a, b, n, prec);
       acb_poly_sub_series(a, a, b, n, prec);
       acb_poly_scalar_mul_2exp_si(b, pip, -3);
       acb_poly_sub_series(a, a, b, n, prec);
       acb_poly_mullow(a, a, onei, n, prec);
       acb_poly_neg(a, a);
       acb_poly_exp_series(U, a, n, prec);
       acb_poly_conj_series(Uc, U, n, prec);

       //C0p
       acb_poly_mullow(a, p, p, n, prec);
       acb_poly_scalar_mul_2exp_si(a, a, -1);
       acb_poly_scalar_mul_2exp_si(b, three, -3);
       acb_poly_add_series(a, a, b, n, prec);
       acb_poly_mullow(a, a, onei, n, prec);
       acb_poly_mullow(a, a, pip, n, prec);
       acb_poly_exp_series(a, a, n, prec);

       acb_poly_mullow(b, p, pip, n, prec);
       acb_poly_scalar_mul_2exp_si(b, b, -1);
       acb_poly_cos_series(b, b, n, prec);
       acb_poly_sqrt_series(c, two, n, prec);
       acb_poly_mullow(b, b, c, n, prec);
       acb_poly_mullow(b, b, onei, n, prec);
       acb_poly_sub_series(a, a, b, n, prec);

       acb_poly_mullow(b, p, pip, n, prec);
       acb_poly_cos_series(b, b, n, prec);
       acb_poly_scalar_mul_2exp_si(b, b, 1);
       acb_poly_div_series(C0p, a, b, n, prec);
       acb_poly_conj_series(C0pc, C0p, n, prec);

       //termC1
       acb_poly_scalar_mul_2exp_si(d, sprime, -1);
       acb_poly_neg(a, d);
       acb_poly_pow_series(a, pip, a, n, prec);

       acb_poly_gamma_series(b, d, n, prec);
       acb_poly_mullow(a, a, b, n, prec);

       acb_poly_neg(b, sig);
       acb_poly_pow_series(b, alf, b, n, prec);
       acb_poly_mullow(a, a, b, n, prec);

       acb_poly_mullow(b, C0p, U, n, prec);
       acb_poly_mullow(termC1, a, b, n, prec);

       //termC2
       acb_poly_scalar_mul_2exp_si(d, scprime, -1);
       acb_poly_neg(a, d);
       acb_poly_pow_series(a, pip, a, n, prec);

       acb_poly_gamma_series(b, d, n, prec);
       acb_poly_mullow(a, a, b, n, prec);

       acb_poly_add_si(b, sig, -1, prec);
       acb_poly_pow_series(b, alf, b, n, prec);
       acb_poly_mullow(a, a, b, n, prec);

       acb_poly_mullow(b, C0pc, Uc, n, prec);
       acb_poly_mullow(termC2, a, b, n, prec);

       //C
       acb_poly_mullow(a, pip, pip, n, prec);
       acb_poly_mullow(a, a, t, n, prec);
       acb_poly_scalar_mul_2exp_si(a, a, -6);
       acb_poly_exp_series(a, a, n, prec);

       acb_poly_add_si(b, sprime, -1, prec);
       acb_poly_mullow(b, b, sprime, n, prec);
       acb_poly_scalar_mul_2exp_si(b, b, -1);
       acb_poly_mullow(a, a, b, n, prec);

       acb_poly_add_series(b, termC1, termC2, n, prec);
       acb_poly_mullow(a, a, b, n, prec);

       acb_poly_set_si(b, -1);
       acb_poly_pow_series(b, b, polyN, n, prec);
       acb_poly_mullow(a, a, b, n, prec);
       acb_poly_scalar_mul_2exp_si(C, a, -3);
    }

    acb_poly_add_series(a, A, B, n, prec);
    acb_poly_sub_series(a, a, C, n, prec);

    acb_poly_evaluate(res, a, one, prec);
 
    arb_clear(ar);

    acb_clear(pi);
    acb_clear(one);
    acb_clear(tmp);
    acb_clear(logk);

    acb_poly_clear(x);
    acb_poly_clear(y);
    acb_poly_clear(t);
    acb_poly_clear(alph1);
    acb_poly_clear(alph2);
    acb_poly_clear(hs);
    acb_poly_clear(hsc);
    acb_poly_clear(ta1);
    acb_poly_clear(ta2);
    acb_poly_clear(A0);
    acb_poly_clear(B0);
    acb_poly_clear(a);
    acb_poly_clear(b);
    acb_poly_clear(c);
    acb_poly_clear(d);
    acb_poly_clear(b1);
    acb_poly_clear(b2);
    acb_poly_clear(summand);
    acb_poly_clear(Asum);
    acb_poly_clear(Bsum);
    acb_poly_clear(A);
    acb_poly_clear(B);
    acb_poly_clear(C);
    acb_poly_clear(onei);
    acb_poly_clear(two);
    acb_poly_clear(three);
    acb_poly_clear(T);
    acb_poly_clear(Tprime);
    acb_poly_clear(s);
    acb_poly_clear(sc);
    acb_poly_clear(sprime);
    acb_poly_clear(scprime);
	acb_poly_clear(U);
    acb_poly_clear(Uc);
    acb_poly_clear(C0p);
    acb_poly_clear(C0pc);
    acb_poly_clear(pip);
    acb_poly_clear(sig);
    acb_poly_clear(alf);
    acb_poly_clear(p);
    acb_poly_clear(termC1);
    acb_poly_clear(termC2);
    acb_poly_clear(polyN);
    acb_poly_clear(polyone);
}
//end of ABCeff code

// f(z) = Abbeff(z) / Abbeff'(z)
void
f_Ht_frac1(acb_t derfactor, acb_t z, arb_t t, slong prec)
{
	arb_t h, x, y;
	arb_init(x);
	arb_init(y);
	arb_init(h);
	
	arb_set_str(h, "0.00000000000000000000000000000000000000001", prec);
	
	acb_t nor, norh, der;
	acb_init(nor);
	acb_init(norh);
	acb_init(der);
	
	acb_get_real(x, z);
	acb_get_imag(y, z);

	abbeff_series(nor, x, y, t, 1, 1, prec);

	arb_add(x, x, h, prec);
	abbeff_series(norh, x, y, t, 1, 1, prec);
	
	acb_sub(der, norh, nor, prec);
	acb_div_arb(der, der, h, prec);
	
    acb_div(derfactor, nor ,der , prec);
	
	arb_clear(h);
	arb_clear(x);
	arb_clear(y);

    acb_clear(nor);
    acb_clear(norh);
    acb_clear(der);
}

// Find exact (acc=accuracy) complex root using Newton-Raphson method
void
Newton_Raphson(acb_t x1, acb_t xx, arb_t t, arb_t acc, slong prec)
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
        f_Ht_frac1(h, x0, t, prec);
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
FindComplexroots(arb_t res, arb_t xmin, arb_t xmax, arb_t y, arb_t t, arb_t acc, slong prec)
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
	Newton_Raphson(root, x0, t, acc, prec);

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
FindRealroots(arb_t res, arb_t xmin, arb_t xmax, arb_t t, arb_t acc, slong prec)
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
	Newton_Raphson(root, x0, t, acc, prec);

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
	Newton_Raphson(root, x0, t, acc, prec);

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
    arb_t a, acc, xs, xe, ys, t, te, ts, ti, outcome;
    arb_init(a);
    arb_init(acc);
    arb_init(xs);
    arb_init(xe);
    arb_init(ys);
    arb_init(t);
    arb_init(ts);
    arb_init(te);
    arb_init(ti);
    arb_init(outcome);

    const char *xs_str, *xe_str, *ts_str, *te_str, *ti_str;
	
    slong prec, res; res = 0;

    int result = EXIT_SUCCESS;

    if (argc != 6)
    {
        result = EXIT_FAILURE;
        goto finish;
    }

    xs_str = argv[1];
    xe_str = argv[2];
    ts_str = argv[3];
    te_str = argv[4];
    ti_str = argv[5];

    //Working precision 
    prec = 20 * 3.32192809488736 + 210;

    arb_set_str(xs, xs_str, prec);
    arb_set_str(xe, xe_str, prec);
    arb_set_str(ts, ts_str, prec);
    arb_set_str(te, te_str, prec);
    arb_set_str(ti, ti_str, prec);


    if(arb_lt(xe, xs))
    {
        result = EXIT_FAILURE; 
        goto finish;
    }
	
	// set target accuracy for all Newton-Raphson rootfinding
	arb_set_str(acc, "0.000000000000000001", prec); 
	
	// set the imaginary starting value for complex Newton-Raphson rootfinding
	arb_set_str(ys, "0.1", prec); 

    arb_set(t, ts);
	arb_add(te, te, acc, prec);

    // loop through the specified t range by steps tinc until the first real zero is found
    while(!arb_is_positive(outcome))
    {
		FindComplexroots(outcome, xs, xe, ys, t, acc, prec);
		
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

		FindRealroots(outcome, xs, xe, t, acc, prec);

		arb_add(t, t, ti, prec);
	}

finish:
 
    if (result == EXIT_FAILURE)
    {
        flint_printf("Required inputs:\n");
        flint_printf("%s xs, xe, ts, te, ti\n\n", argv[0]);
        flint_printf(
    "This script computes all complex and real zeros of ABCeff(x,y,t) near a Lehmer-pair over a range of t.\n"
    "In the x-range, xs(tart) and xe(nd) are best chosen with 0.1 below and above the Lehmer-pair.\n"
    "The t-range runs from ts(tart) to te(nd) in steps of ti(ncr) that determines the #zeros printed (x, y, t).\n"
    "\n");
    }

    arb_clear(a);
    arb_clear(acc);
    arb_clear(xs);
    arb_clear(xe);
    arb_clear(ys);
    arb_clear(t);
    arb_clear(ts);
    arb_clear(te);
    arb_clear(ti);
    arb_clear(outcome);
 
    flint_cleanup();

    return result;
}
