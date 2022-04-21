// optm_minpack2.i -
//
// Unconstrained and box constrained differentiable non-linear optimization
// problems from MINPACK-2 Project.
//
// History:
// - FORTRAN 77 code.  MINPACK-2 Project. November 1993.
//   Argonne National Laboratory and University of Minnesota.
//   Brett M. Averick and Jorge J. Moré.
// - Conversion to Yorick.  April 2022.  Éric Thiébaut.
//
// References:
// - Brett M. Averick, Richard G. Carter, Jorge J. Moré, and Guo-Liang Xue,
//   "The MINPACK-2 Test Problem Collection", Office of Scientific and
//   Technical Information, https://doi.org/10.2172/79972 (1992).

func optm_minpack2_test(ctx,
                        // optm_vmlmb options:
                        mem=, lnsrch=, epsilon=,
                        f2nd=, fmin=, dxrel=, dxabs=,
                        ftol=, gtol=, xtol=, blmvm=,
                        maxiter=, maxeval=, verb=, cputime=, output=)
{
    local status, fx, gx, xs, xl, xu;
    if (is_void(verb)) verb = 1;
    if (ctx.prob == "ept") {
        xl = ctx("xl");
        xu = ctx("xu");
    } else if (ctx.prob == "ssc") {
        xl = [];
        xu = [];
    }
    xs = ctx("xs");
    n = numberof(xs);
    if (is_void(mem)) mem = min(n, 7);
    if (verb > 0) {
        write, output,
            format="\n# MINPACK-2 Problem %s (%d variables): %s.\n",
            strcase(1, ctx.prob), n, ctx.descr;
        write, output, format="# Algorithm: %s (mem=%d).\n",
            (blmvm ? "BLMVM" : "VMLMB"), mem;
    }
    fg = optm_minpack2_fg(ctx);
    x = optm_vmlmb(fg, xs, fx, gx, status, lower=xl, upper=xu,
                   mem=m, lnsrch=lnsrch, epsilon=epsilon,
                   f2nd=f2nd, fmin=fmin, dxrel=dxrel, dxabs=dxabs,
                   ftol=ftol, gtol=gtol, xtol=xtol, blmvm=blmvm,
                   maxiter=maxiter, maxeval=maxeval, verb=verb,
                   cputime=cputime, output=output);
    return x;
}

func optm_minpack2_long_parameter(&x)
/* DOCUMENT optm_minpack2_long_parameter(x)

     Yields whether `x` is a scalar integer and convert `x` into a `long` if
     this is the case.

   SEE ALSO: optm_minpack2_double_parameter.
 */
{
    if (is_scalar(x) && is_integer(x)) {
        x = long(x);
        return 1n;
    } else {
        return 0n;
    }
}

func optm_minpack2_double_parameter(&x)
/* DOCUMENT optm_minpack2_double_parameter(x)

     Yields whether `x` is a scalar floating-point and convert `x` into a
     `double` if this is the case.

   SEE ALSO: optm_minpack2_long_parameter.
 */
{
    if (is_scalar(x) && (is_real(x) || is_integer(x))) {
        x = double(x);
        return 1n;
    } else {
        return 0n;
    }
}

local _optm_minpack2_fg;
func optm_minpack2_fg(ctx)
/* DOCUMENT fg = optm_minpack2_fg(ctx);

     Yields an object callable as:

         fx = fg(x, gx);

     that yields the value objective function defined by the context `ctx` at
     `x` and stores in caller's vraible `gx` the gradient of the objective
     function at `x`.

   SEE ALSO:
 */
{
    return closure("_optm_minpack2_fg", ctx);
}

func _optm_minpack2_fg(ctx, x, &gx)
{
    return ctx("fg", x, gx);
}

//-----------------------------------------------------------------------------
// ELASTIC-PLASTIC TORSION PROBLEM

func optm_minpack2_ept(nil, nx=, ny=, c=)
/* DOCUMENT ctx = optm_minpack2_ept(nx=, ny=, lambda=);

     Get context for the elastic-plastic torsion problem.  All parameters are
     specified by keywords:

     `nx` is the number of grid points in the first coordinate direction.

     `ny is the number of grid points in the second coordinate direction.

     `c` is the angle of twist per unit length, `c=25` by default.

   SEE ALSO:
 */
{
    if (is_void(nx)) nx = 40;
    if (is_void(ny)) ny = nx;
    if (is_void(c)) c = 25.0;
    if (!optm_minpack2_double_parameter(c)) {
        error, "invalid value for `c`";
    }
    if (!optm_minpack2_long_parameter(nx) || nx < 1) {
        error, "invalid value for `nx`";
    }
    if (!optm_minpack2_long_parameter(ny) || ny < 1) {
        error, "invalid value for `ny`";
    }
    ctx = h_new(prob="ept", descr=" elastic-plastic torsion problem",
                c=c, nx=nx, ny=ny);
    h_evaluator, ctx, "_optm_minpack2_ept";
    return ctx;
}

func _optm_minpack2_ept(ctx, task, x, &gx)
/* DOCUMENT xl = _optm_minpack2_ept(ctx, "xl");
         or xu = _optm_minpack2_ept(ctx, "xu");
         or xs = _optm_minpack2_ept(ctx, "xs");
         or fx = _optm_minpack2_ept(ctx, "f", x);
         or gx = _optm_minpack2_ept(ctx, "g", x);
         or fx = _optm_minpack2_ept(ctx, "fg", x, gx);

     Evaluator for the elastic-plastic torsion problem whose parameters are in
     the context `ctx`.

     Depending on the 2nd argument, this function yields:

     `xl` the lower bounds for the variables.

     `xu` the upper bounds for the variables.

     `xs` the the standard starting point.

     `fx` the value of the objective function at `x`.

     `gx` the gradient of the objective function at `x`.

     `fx` and `gx` the value and the gradient of the objective function at `x`.
     The latter is stored in caller's variable `gx`.

   SEE ALSO: optm_minpack2_ept.
*/
{
    c = ctx.c;
    nx = ctx.nx;
    ny = ctx.ny;
    hx = 1.0/(nx + 1);
    hy = 1.0/(ny + 1);
    area = hx*hy/2.0;
    cdiv3 = c/3.0;

    if (task == "xl" || task == "xu") {
        // Yield a lower bound for an upper bound for the variables.
        by = min(indgen(1:ny), indgen(ny:1:-1))*hy;
        bx = min(indgen(1:nx), indgen(nx:1:-1))*hx;
        b = min(bx, by(-,));
        return (task == "xl") ? -b : b;
    }

    if (task == "xs") {
        // Compute the standard starting point.
        by = min(indgen(1:ny), indgen(ny:1:-1))*hy;
        bx = min(indgen(1:nx), indgen(nx:1:-1))*hx;
        return min(bx, by(-,));
    }

    // Evaluate the function if task = "f", the gradient if task = "g",
    // or both if task = "fg".
    feval = (task == "f" || task == "fg");
    geval = (task == "g" || task == "fg");
    if (feval) {
        fquad = 0.0;
        flin = 0.0;
    }
    if (geval) {
        fgrad = array(double, nx, ny);
    }

    // Computation of the function and the gradient over the lower triangular
    // elements.
    for (j = 0; j <= ny; ++j) {
        for (i = 0; i <= nx; ++i) {
            v = (i > 0 && j > 0) ? x(i,j) : 0.0;
            vr = (i < nx && j > 0) ? x(i+1,j) : 0.0;
            vt = (i > 0 && j < ny) ? x(i,j+1) : 0.0;
            dvdx = (vr - v)/hx;
            dvdy = (vt - v)/hy;
            if (feval) {
                fquad += dvdx*dvdx + dvdy*dvdy;
                flin -= cdiv3*(v + vr + vt);
            }
            if (geval) {
                if (i > 0 && j > 0) {
                    fgrad(i,j) -= dvdx/hx + dvdy/hy + cdiv3;
                }
                if (i < nx && j > 0) {
                    fgrad(i+1,j) += dvdx/hx - cdiv3;
                }
                if (i > 0 && j < ny) {
                    fgrad(i,j+1) += dvdy/hy - cdiv3;
                }
            }
        }
    }

    // Computation of the function and the gradient over the upper triangular
    // elements.
    for (j = 1; j <= ny+1; ++j) {
        for (i = 1; i <= nx+1; ++i) {
            vb = (i <= nx && j > 1) ? x(i,j-1) : 0.0;
            vl = (i > 1 && j <= ny) ? x(i-1,j) : 0.0;
            v = (i <= nx && j <= ny) ? x(i,j) : 0.0;
            dvdx = (v - vl)/hx;
            dvdy = (v - vb)/hy;
            if (feval) {
                fquad += dvdx*dvdx + dvdy*dvdy;
                flin -= cdiv3*(vb + vl + v);
            }
            if (geval) {
                if (i <= nx && j > 1) {
                    fgrad(i,j-1) -= dvdy/hy + cdiv3;
                }
                if (i > 1 && j <= ny) {
                    fgrad(i-1,j) -= dvdx/hx + cdiv3;
                }
                if (i <= nx && j <= ny) {
                    fgrad(i,j) += dvdx/hx + dvdy/hy - cdiv3;
                }
            }
        }
    }

    // Scale the result.
    if (feval) {
        f = area*(fquad/2.0 + flin);
    }
    if (geval) {
        fgrad *= area;
        if (feval) {
            eq_nocopy, gx, fgrad;
            return f;
        }
        return fgrad;
    }
    if (feval) {
        return f;
    }
}

//-----------------------------------------------------------------------------
// STEADY STATE COMBUSTION PROBLEM

func optm_minpack2_ssc(nil, nx=, ny=, lambda=)
/* DOCUMENT ctx = optm_minpack2_ssc(nx=, ny=, lambda=);

     Get context for the steady state combustion problem.  All parameters are
     specified by keywords:

     `nx` is the number of grid points in the first coordinate direction.

     `ny is the number of grid points in the second coordinate direction.

     `lambda` is a nonnegative Frank-Kamenetski parameter in the `[0, 6.81]`,
     default value is `lambda = 5`.

   SEE ALSO:
 */
{
    if (is_void(nx)) nx = 40;
    if (is_void(ny)) ny = nx;
    if (is_void(lambda)) lambda = 5.0;
    if (!optm_minpack2_double_parameter(lambda) ||
        lambda < 0 || lambda > 6.81) {
        error, "invalid value for `lambda`";
    }
    if (!optm_minpack2_long_parameter(nx) || nx < 1) {
        error, "invalid value for `nx`";
    }
    if (!optm_minpack2_long_parameter(ny) || ny < 1) {
        error, "invalid value for `ny`";
    }
    ctx = h_new(prob="ssc", descr="steady state combustion problem",
                lambda=lambda, nx=nx, ny=ny);
    h_evaluator, ctx, "_optm_minpack2_ssc";
    return ctx;
}

func _optm_minpack2_ssc(ctx, task, x, &gx)
/* DOCUMENT xs = _optm_minpack2_ssc(ctx, "xs");
         or fx = _optm_minpack2_ssc(ctx, "f", x);
         or gx = _optm_minpack2_ssc(ctx, "g", x);
         or fx = _optm_minpack2_ssc(ctx, "fg", x, gx);

     Evaluator for the steady state combustion problem whose parameters are
     in the context `ctx`.

     Depending on the 2nd argument, this function yields:

     `xl` the lower bounds for the variables.

     `xu` the upper bounds for the variables.

     `xs` the the standard starting point.

     `fx` the value of the objective function at `x`.

     `gx` the gradient of the objective function at `x`.

     `fx` and `gx` the value and the gradient of the objective function at `x`.
     The latter is stored in caller's variable `gx`.

   SEE ALSO: optm_minpack2_ssc.
*/
{
    // Initialization.
    nx = ctx.nx;
    ny = ctx.ny;
    lambda = ctx.lambda;
    hx = 1.0/(nx + 1);
    hy = 1.0/(ny + 1);
    area = hx*hy/2.0;

    // Compute the standard starting point if task = 'XS'.
    if (task == "xs") {
        q = lambda/(lambda + 1.0);
        tx = min(indgen(1:nx), indgen(nx:1:-1))*hx;
        ty = min(indgen(1:ny), indgen(ny:1:-1))*hy;
        return q*sqrt(min(tx, ty(-,)));
    }

    // Compute the function if task = 'F', the gradient if task = 'G', or both
    // if task = 'FG'.
    feval = (task == "f" || task == "fg");
    geval = (task == "g" || task == "fg");
    if (feval) {
        fquad = 0.0;
        fexp = 0.0;
    }
    if (geval) {
        fgrad = array(double, nx, ny);
    }

    // Computation of the function and the gradient over the lower triangular
    // elements.  The trapezoidal rule is used to estimate the integral of the
    // exponential term.
    for (j = 0; j <= ny; ++j) {
        for (i = 0; i <= nx; ++i) {
            if (i != 0 && j != 0) {
                v = x(i,j);
            } else {
                v = 0.0;
            }
            if (i != nx && j != 0) {
                vr = x(i+1,j);
            } else {
                vr = 0.0;
            }
            if (i != 0 && j != ny) {
                vt = x(i,j+1);
            } else {
                vt = 0.0;
            }
            dvdx = (vr - v)/hx;
            dvdy = (vt - v)/hy;
            expv = exp(v);
            expvr = exp(vr);
            expvt = exp(vt);
            if (feval) {
                fquad += dvdx*dvdx + dvdy*dvdy;
                fexp -= lambda*(expv + expvr + expvt)/3.0;
            }
            if (geval) {
                if (i != 0 && j != 0) {
                    fgrad(i,j) -= dvdx/hx + dvdy/hy + lambda*expv/3.0;
                }
                if (i != nx && j != 0) {
                    fgrad(i+1,j) += dvdx/hx - lambda*expvr/3.0;
                }
                if (i != 0 && j != ny) {
                    fgrad(i,j+1) += dvdy/hy - lambda*expvt/3.0;
                }
            }
        }
    }

    // Computation of the function and the gradient over the upper triangular
    // elements.  The trapezoidal rule is used to estimate the integral of the
    // exponential term.
    for (j = 1; j <= ny + 1; ++j) {
        for (i = 1; i <= nx + 1; ++i) {
            if (i <= nx && j > 1) {
                vb = x(i,j-1);
            } else {
                vb = 0.0;
            }
            if (i > 1 && j <= ny) {
                vl = x(i-1,j);
            } else {
                vl = 0.0;
            }
            if (i <= nx && j <= ny) {
                v = x(i,j);
            } else {
                v = 0.0;
            }
            dvdx = (v-vl)/hx;
            dvdy = (v-vb)/hy;
            expvb = exp(vb);
            expvl = exp(vl);
            expv = exp(v);
            if (feval) {
                fquad = fquad + dvdx*dvdx + dvdy*dvdy;
                fexp = fexp - lambda*(expvb+expvl+expv)/3.0;
            }
            if (geval) {
                if (i <= nx && j > 1) {
                    fgrad(i,j-1) -= dvdy/hy + lambda*expvb/3.0;
                }
                if (i > 1 && j <= ny) {
                    fgrad(i-1,j) -= dvdx/hx + lambda*expvl/3.0;
                }
                if (i <= nx && j <= ny) {
                    fgrad(i,j) += dvdx/hx + dvdy/hy - lambda*expv/3.0;
                }
            }
        }
    }

    // Scale the result.
    if (feval) {
        f = area*(fquad/2.0 + fexp);
    }
    if (geval) {
        fgrad *= area;
        if (feval) {
            eq_nocopy, gx, fgrad;
            return f;
        }
        return fgrad;
    }
    if (feval) {
        return f;
    }
}

//-----------------------------------------------------------------------------

if (batch()) {
    // Run all tests.
    include, dirname(current_include()) + "/optm.i", 1;
    optm_minpack2_test, optm_minpack2_ept(), verb=1, gtol=0, xtol=0, ftol=0;
    optm_minpack2_test, optm_minpack2_ssc(), verb=1, gtol=0, xtol=0, ftol=0;
}
