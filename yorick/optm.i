// optm.i -
//
// Pure Yorick implementations of the linear conjugate gradients (for solving
// uncontrained linear optimization problems) and VMLMB (for solving non-linear
// of bound constrained optimization problems).  VMLMB is a quasi-Newton method
// ("VM" is for "Variable Metric") with low memory requirements ("LM" is for
// "Limited Memory") and which can optionally take into account separable bound
// constraints (the final "B") on the variables.
//
//-----------------------------------------------------------------------------
//
// This file is part of the VMLMB software which is licensed under the "Expat"
// MIT license, <https://github.com/emmt/VMLMB>.
//
// Copyright (C) 2002-2022, Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>

if (is_void(OPTM_DEBUG)) OPTM_DEBUG = 1n;

OPTM_NOT_POSITIVE_DEFINITE = -1;
OPTM_TOO_MANY_EVALUATIONS  =  1;
OPTM_TOO_MANY_ITERATIONS   =  2;
OPTM_FTEST_SATISFIED       =  3;
OPTM_XTEST_SATISFIED       =  4;
OPTM_GTEST_SATISFIED       =  5;

func optm_reason(status)
/* DOCUMENT str = optm_reason(status);

     Get a textual description of the meaning of the termination code `status`.

   SEE ALSO: optm_status, optm_conjgrad.
 */
{
    if (status == OPTM_NOT_POSITIVE_DEFINITE) {
        return "LHS operator is not positive definite";
    } else if (status == OPTM_TOO_MANY_EVALUATIONS) {
        return "too many evaluations";
    } else if (status == OPTM_TOO_MANY_ITERATIONS) {
        return "too many iterations";
    } else if (status ==  OPTM_FTEST_SATISFIED) {
        return "function reduction test satisfied";
    } else if (status == OPTM_GTEST_SATISFIED) {
        return "(projected) gradient test satisfied";
    } else if (status == OPTM_XTEST_SATISFIED) {
        return "variables change test satisfied";
    } else {
        return "unknown status code";
    }
}

//-----------------------------------------------------------------------------
// LINEAR CONJUGATE GRADIENT

func optm_conjgrad(A, b, x0, &status, precond=, maxiter=, restart=, verb=,
                   printer=, output=, ftol=, gtol=, xtol=)
/* DOCUMENT x = optm_conjgrad(A, b, [x0, status]);

     Run the (preconditioned) linear conjugate gradient algorithm to solve
     the system of equations `A⋅x = b` in `x`.

     Argument `A` implements the left-hand-side (LHS) "matrix" of the
     equations.  It is called as `A(x)` to compute the result of `A⋅x`.
     Note that, as `A` and the preconditioner `M` must be symmetric, it
     may be faster to apply their adjoint.

     Argument `b` is the right-hand-side (RHS) "vector" of the equations.

     Optional argument `x0` provides the initial solution.  If `x0` is
     unspecified, `x` is initially an array of zeros.

     Provided `A` be positive definite, the solution `x` of the equations
     `A⋅x  = b` is unique and is also the minimum of the following convex
     quadratic objective function:

         f(x) = (1/2)⋅x'⋅A⋅x - b'⋅x + ϵ

     where `ϵ` is an arbitrary constant.  The gradient of this objective
     function is:

         ∇f(x) = A⋅x - b

     hence solving `A⋅x = b` for `x` yields the minimum of `f(x)`.  The
     variations of `f(x)` between successive iterations, the norm of the
     gradient `∇f(x)` or the norm of the variation of variables `x` may be
     used to decide the convergence of the algorithm (see keywords `ftol`,
     `gtol` and `xtol`).

     Algorithm parameters can be specified by the following keywords:

     - Keyword `precond` is to specify a preconditioner `M`.  It may be a
       function name or handle and is called as `M(x)` to compute the result
       of `M⋅x`.  By default, the un-preconditioned version of the algorithm
       is run.

     - Keyword `maxiter` is to specify the maximum number of iterations to
       perform which is `2⋅numberof(b) + 1` by default.

     - Keyword `restart` is to specify the number of consecutive iterations
       before restarting the conjugate gradient recurrence.  Restarting the
       algorithm is to cope with the accumulation of rounding errors.  By
       default, `restart = min(50,numberof(b)+1)`.  Set `restart` to a value
       less or equal zero or greater than `maxiter` if you do not want that any
       restarts ever occur.

     - Keyword `verb`, if positive, specifies to print information every `verb`
       iterations.  Nothing is printed if `verb ≤ 0`.  By default, `verb = 0`.

     - Optional argument `printer` specifies a function to call to print
       information every `verb` iterations.  This function is called as:

           printer(output, itr, t, x, phi, r, z, rho)

       with `output` the output stream specified by keyword `output`, `itr` the
       iteration number, `t` the elapsed time in seconds, `phi` the reduction
       of the objective function, `r` the residuals, `z` the preconditioned
       residuals, and `rho` the squared Euclidean norm of `z`.  You can use `z
       is r` to verify whether a preconditioner is used or not (`z` is
       different from `r` if this is the case).

     - Keyword `output` specifies the file stream to print information.

     - Keywords `ftol`, `gtol` and `xtol` specify tolerances for deciding the
       convergence of the algorithm.  In what follows, `x_{k}`,
       `f_{k}=f(x_{k})`, and `∇f_{k}=∇f(x_{k})` denotes the variables, the
       objective function and its gradient after `k` iterations ot the
       algorithm (`x_{0} = x0` the initial estimate).

       Convergence in the function occurs at iteration `k ≥ 1` if the following
       condition holds:

           f_{k-1} - f_{k} ≤ max(0, fatol, frtol⋅max_{j≤k}(f_{j-1} - f_{j}))

       where `fatol` and `frtol` are absolute and relative tolerances specified
       by `ftol` which can be `ftol=[fatol,frtol]` or `ftol=frtol` and assume
       that `fatol=0`.  The default is `ftol=1e-8`.

       Convergence in the gradient occurs at iteration `k ≥ 0` if the following
       condition holds:

           ‖∇f_{k}‖_M ≤ max(0, gatol, grtol⋅‖∇f_{0}‖_M)

       where `‖u‖_M = sqrt(u'⋅M⋅u)` is the Mahalanobis norm of `u` with
       precision matrix the preconditioner `M` (which is equal to the usual
       Euclidean norm of `u` if no preconditioner is used or if `M` is the
       identity).  In this condition, `gatol` and `grtol` are absolute and
       relative gradient tolerances specified by `gtol` which can be
       `gtol=[gatol,grtol]` or `gtol=grtol` and assume that `gatol=0`.  The
       default is `gtol=1e-5`.

       Convergence in the variables occurs at iteration `k ≥ 1` if the
       following condition holds:

           ‖x_{k} - x_{k-1}‖ ≤ max(0, xatol, xrtol⋅‖x_{k}‖)


       where `xatol` and `xrtol` are absolute and relative tolerances specified
       by `xtol` which can be `xtol=[fatol,frtol]` or `xtol=xrtol` and assume
       that `xatol=0`.  The default is `xtol=1e-6`.

       In the conjugate gradient algorithm, the objective function is always
       reduced at each iteration, but be aware that the gradient and the change
       of variables norms are not always reduced at each iteration.


     ## Returned Status

     The function returns the solution `x` and set optional output variable
     `status` with one of the following termination codes:

     - `status = OPTM_NOT_POSITIVE_DEFINITE` if the left-hand-side matrix `A`
       is found to be not positive definite;

     - `status = OPTM_TOO_MANY_ITERATIONS` if the maximum number of iterations
       has been reached;

     - `status = OPTM_FTEST_SATISFIED` if convergence occurred because the
       function reduction satisfies the criterion specified by `ftol`;

     - `status = OPTM_GTEST_SATISFIED` if convergence occurred because the
       gradient norm satisfies the criterion specified by `gtol`;

     - `status = OPTM_XTEST_SATISFIED` if convergence occurred because the norm
       of the variation of variables satisfies the criterion specified by
       `xtol`.

     The function `optm_reason` may be called to have a textual description of
     the meaning of the returned status.

  SEE ALSO: optm_reason, optm_inner.
 */
{
    // Get options.
    preconditioned = !is_void(precond);
    if (is_void(ftol)) ftol = 1.0E-8;
    if (is_void(gtol)) gtol = 1.0E-5;
    if (is_void(xtol)) xtol = 1.0E-6;
    if (is_void(verb)) verb = 0;
    if (is_void(printer)) printer = optm_conjgrad_printer;
    if (is_void(maxiter)) maxiter = 2*numberof(b) + 1;
    if (is_void(restart)) restart = min(50, numberof(b) + 1);
    if (is_scalar(ftol)) {
        fatol = 0.0;
        frtol = ftol;
    } else {
        fatol = ftol(1);
        frtol = ftol(2);
    }
    if (is_scalar(gtol)) {
        gatol = 0.0;
        grtol = gtol;
    } else {
        gatol = gtol(1);
        grtol = gtol(2);
    }
    if (is_scalar(xtol)) {
        xatol = 0.0;
        xrtol = xtol;
    } else {
        xatol = xtol(1);
        xrtol = xtol(2);
    }

    // Initial solution.
    if (is_void(x0)) {
        x = array(structof(b), dimsof(b));
        x_is_zero = 1n;
    } else {
        x = x0; // force a copy
        x_is_zero = (optm_norm2(x) == 0.0);
    }

    // Define local variables.
    r = [];       // residuals `r = b - A⋅x`
    z = [];       // preconditioned residuals `z = M⋅r`
    p = [];       // search direction `p = z + β⋅p`
    q = [];       // `q = A⋅p`
    rho = [];     // `rho = ⟨r,z⟩`
    oldrho = [];  // previous value of `rho`
    phi = 0.0;    // function reduction
    phimax = 0.0; // maximum function reduction
    xtest = (xatol > 0.0 || xrtol > 0.0);
    if (verb > 0) {
        elapsed = array(double, 3);
        timer, elapsed;
        t0 = elapsed(3);
    }

    // Conjugate gradient iterations.
    for (k = 0; ; ++k) {
        // Is this the initial or a restarted iteration?
        restarting = (k == 0 || restart > 0 && (k%restart) == 0);

        // Compute residuals and their squared norm.
        if (restarting) {
            // Compute residuals.
            if (x_is_zero) {
                // Spare applying A since x = 0.
                r = b;
                x_is_zero = 0n;
            } else {
                // Compute `r = b - A*x`.
                r = b - A(x); // FIXME: optm_combine, r, +1, b, -1, A(x);
            }
        } else {
            // Update residuals: `r -= alpha*q`.
            optm_update, r, -alpha, q;
        }
        if (preconditioned) {
            // Apply preconditioner `z = M⋅r`.
            z = precond(r);
        } else {
            // No preconditioner `z = I⋅r = r`.
            eq_nocopy, z, r;
        }
        oldrho = rho;
        rho = optm_inner(r, z); // rho = ‖r‖_M^2
        if (k == 0) {
            // Pre-compute the minimal Mahalanobis norm of the gradient for
            // convergence.  The Mahalanobis norm of the gradient is equal
            // to `2⋅sqrt(rho)`.
            gtest = max(0.0, gatol, 2.0*grtol*sqrt(rho));
        }
        if (verb > 0 && (k % verb) == 0) {
            timer, elapsed;
            printer, output, k, elapsed(3) - t0, x, phi, r, z, rho;
        }
        if (2.0*sqrt(rho) <= gtest) {
            // Normal convergence in the gradient norm.
            status = OPTM_GTEST_SATISFIED;
            mesg = "Convergence in the gradient norm.";
            break;
        }
        if (k >= maxiter) {
            status = OPTM_TOO_MANY_ITERATIONS;
            mesg = "Too many iteration(s).";
            break;
        }

        // Compute search direction `p`.
        if (restarting) {
            // Restarting or first iteration.
            eq_nocopy, p, z;
        } else {
            // Apply recurrence.
            beta = rho/oldrho;
            p = z + beta*p; // FIXME: optm_combine, p, +1, z, beta, p;
        }

        // Compute optimal step size `alpha` along search direction `p`.
        q = [];
        q = A(p); // q = A*p
        gamma = optm_inner(p, q);
        if (!(gamma > 0.0)) {
            status = OPTM_NOT_POSITIVE_DEFINITE;
            mesg = "Operator is not positive definite.";
            break;
        }
        alpha = rho/gamma;

        // Update variables and check for convergence.
        optm_update, x, alpha, p; // x += alpha*p
        phi = alpha*rho/2.0; // phi = f(x_{k}) - f(x_{k+1}) ≥ 0
        phimax = max(phi, phimax);
        if (phi <= optm_tolerance(phimax, fatol, frtol)) {
            // Normal convergence in the function reduction.
            status = OPTM_FTEST_SATISFIED;
            mesg = "Convergence in the function reduction.";
            break;
        }
        if (xtest && alpha*optm_norm2(p) <= optm_tolerance(x, xatol, xrtol)) {
            // Normal convergence in the variables.
            status = OPTM_XTEST_SATISFIED;
            mesg = "Convergence in the variables.";
            break;
        }
    }
    if (verb > 0) {
        // Print last iteration if not yet done and termination message.
        if ((k % verb) != 0) {
            timer, elapsed;
            printer, output, k, elapsed(3) - t0, x, phi, r, z, rho;
        }
        if (mesg && printer == optm_conjgrad_printer) {
            write, output, format="# %s\n", mesg;
        }
    }
    return x;
}

func optm_conjgrad_printer(output, itr, t, x, phi, r, z, rho)
/* DOCUMENT optm_conjgrad_printer, output, itr, t, x, phi, r, z, rho;
     Default printer function for `optm_conjgrad`.

   SEE ALSO:optm_conjgrad.
 */
{
    if (preconditioned) {
        if (itr == 0) {
            write, output, format="%s\n%s\n",
                "# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖     ‖∇f(x)‖_M",
                "# ---------------------------------------------------------";
        }
        write, output, format="%7d %11.3f %12.4e %12.4e %12.4e\n",
            itr, t*1e3, phi, optm_norm2(r), sqrt(rho);
    } else {
        if (itr == 0) {
            write, output, format="%s\n%s\n",
                "# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖",
                "# --------------------------------------------";
        }
        write, output, format="%7d %11.3f %12.4e %12.4e\n",
            itr, t*1e3, phi, sqrt(rho);
    }
}

//-----------------------------------------------------------------------------
// LINE-SEARCH

struct OptmLineSearch {
    double finit;
    double ginit;
    double ftol;
    double smin;
    double smax;
    double step;
    long stage;
};

local optm_new_line_search;
local optm_start_line_search;
local optm_iterate_line_search;
/* DOCUMENT lnsrch = optm_new_line_search();
         or lnsrch = optm_new_line_search(tmpl);
         or optm_start_line_search, lnsrch, f0, df0, stp;
         or optm_iterate_line_search, lnsrch, f;

     The function `optm_new_line_search` creates a new line-search instance.
     Optional argument `tmpl` is another line-search instance to use as a
     template to define default values for the line-search parameters.  All
     parameters are specified by keyword.

     Keyword `ftol` can be used to specify the function decrease tolerance.  A
     step `alpha` is considered as successful if the following condition (known
     as Armijo's condition) holds:

         f(x0 + alpha*d) ≤ f(x0) + ftol*df(x0)*alpha

     where `f(x)` is the objective function at `x`, `x0` denotes the variables
     at the start of the line-search, `d` is the search direction, and `df(x0)
     = d'⋅∇f(x0)` is the directional derivative of the objective function at
     `x0`.  The value of `ftol` must be in the range `(0,0.5]`, the default
     value is `ftol = 1E-4`.

     Keywords `smin` and `smax` can be used to specify relative bounds for
     safeguarding the step length.  When a step `alpha` is unsuccessful, a new
     backtracking step is computed based on a parabolic interpolation of the
     objective function along the search direction.  The new step writes:

         new_alpha = gamma*alpha

     with `gamma` safeguarded in the range `[smin,smax]`.  The following
     constraints must hold: `0 < smin ≤ smax < 1`.  Taking `smin = smax = 0.5`
     emulates the usual Armijo's method.  Default values are `smin = 0.2` and
     `smax = 1/(2 - 2*ftol)`.

     The subroutine `optm_start_line_search` shall be called to initialize each
     new line-search with arguments: `lnsrch` the line-search instance, `f0`
     the objective function at `x0` the variables at the start of the
     line-search, `df0` the directional derivative of the objective function at
     `x0` and `stp > 0` a guess for the first step to try.

     Note that when Armijo's condition does not hold, the quadratic
     interpolation yields `gamma < 1/(2 - 2*ftol)`.  Hence, taking an upper
     bound `smax > 1/(2 - 2*ftol)` has no effects while taking a lower bound
     `smin ≥ 1/(2 - 2*ftol)` yields a safeguarded `gamma` always equal to
     `smin`.  Therefore, to benefit from quadratic interpolation, one should
     choose `smin < smax ≤ 1/(2 - 2*ftol)`.

     The subroutine `optm_iterate_line_search` shall be called to pursue the
     line-search arguments: `lnsrch` the line-search instance, `fx = f(x)` the
     objective function at `x = x0 + lnsrch.step*d` the variables at the
     current position on the line-search.

     These two subroutines updates the line-search instance in-place hence
     `lnsrch` must be a simple variable, not an expression.  On return of one
     of these subroutines, `lnsrch.stage` is normally one of:

     1: Line-search in progress.  The next step to try is `lnsrch.step`,
        `optm_iterate_line_search` shall be called with the new function value
        at `x0 + lnsrch.step*d`.

     2: Line-search has converged.  The step length `lnsrch.step` is left
        unchanged and `x0 + lnsrch.step*d` is the new iterate of the
        optimization method.

     Upon creation of the line-search instance, `lnsrch.stage` is set to 0.  A
     negative value for `lnsrch.stage` may be used to indicate an error.

     A typical usage is:

         lnsrch = optm_new_line_search();
         x = ...;   // initial solution
         fx = f(x); // objective function
         while (1) {
             gx = grad_f(x);
             if (optm_norm2(gx) <= gtol) {
                 break; // a solution has been found
             }
             d = next_search_direction(...);
             stp = guess_step_size(...);
             f0 = fx;
             x0 = x;
             df0 = optm_inner(d, gx);
             optm_start_line_search, lnsrch, f0, df0, stp;
             while (lnsrch.stage == 1) {
                 x = x0 + lnsrch.step*d;
                 fx = f(x);
                 optm_iterate_line_search, lnsrch, fx;
             }
         }

   SEE ALSO:
 */
func optm_new_line_search(lnsrch, ftol=, smin=, smax=)
{
    if (is_void(lnsrch)) {
        lnsrch = OptmLineSearch(ftol=1E-4, smin=0.2, smax=0.9);
    }
    if (is_void(ftol)) ftol = lnsrch.ftol;
    if (ftol <= 0 || ftol > 0.5) error, "ftol must be in the range (0,0.5]";
    if (is_void(smin)) smin = lnsrch.smin;
    if (smin <= 0) error, "smin must be strictly greater than 0";
    if (is_void(smax)) smax = max(smin, lnsrch.smax, 1.0/(2.0 + 2.0*ftol));
    if (smax >= 1) error, "smax must be strictly less than 1";
    if (smin > smax) error, "smin must be less or equal smax";
    lnsrch.stage = 0;
    lnsrch.ftol = ftol;
    lnsrch.smin = smin;
    lnsrch.smax = smax;
    lnsrch.step = 0;
    return lnsrch;
}

func optm_start_line_search(&lnsrch, f0, df0, stp)
{
    if (df0 < 0) {
        if (stp <= 0) {
            stage = -1;
            stp = 0.0;
            error, "first step to try must be strictly greater than 0";
        } else {
            stage = 1;
        }
    } else {
        stage = -1;
        stp = 0.0;
        error, "not a descent direction";
    }
    lnsrch.finit = f0
    lnsrch.ginit = df0;
    lnsrch.step  = stp;
    lnsrch.stage = stage;
    return lnsrch;
}

func optm_iterate_line_search(&lnsrch, fx)
{
    finit = lnsrch.finit;
    ginit = lnsrch.ginit;
    step  = lnsrch.step;
    ftol  = lnsrch.ftol;
    if (fx <= finit + ftol*(ginit*step)) {
        // Line-search has converged.
        lnsrch.stage = 2;
    } else {
        // Line-search has not converged.
        smin = lnsrch.smin;
        smax = lnsrch.smax;
        if (smin < smax) {
            // Compute a safeguarded parabolic interpolation step.
            q = -ginit*step;
            r = 2*((fx - finit) + q);
            if (q <= smin*r) {
                gamma = smin;
            } else if (q >= smax*r) {
                gamma = smax;
            } else {
                gamma = q/r;
            }
        } else if (smin == smax) {
            gamma = smin;
        } else {
            error, "invalid fields SMIN and SMAX";
        }
        lnsrch.step = gamma*step;
        lnsrch.stage = 1;
    }
    return lnsrch;
}

func optm_steepest_descent_step(x, d, fx, fmin, delta, lambda)
/* DOCUMENT alpha = optm_steepest_descent_step(x, d, fx, fmin, delta, lambda);

     Determine the length `alpha` of the first trial step along the steepest
     descent direction.  Arguments are:

     - `x` the current variables (or their Euclidean norm).

     - `d` the search direction `d` (or its Euclidean norm) at `x`.  This
       direction shall be the gradient (or the projected gradient for a
       constrained problem) at `x` up to a change of sign.

     - `fx` the value of the objective function at `x`.

     - `fmin` an estimate of the minimal value of the objective function.

     - `delta` a small step size relative to the norm of the variables.

     - `lambda` an estimate of the magnitude of the eigenvalues of the
       Hessian of the objective function.

   SEE ALSO: optm_start_line_search.
 */
{
    // We use the statement `val == val` to ensure that `val` is not a NaN.
    INF = OPTM_INFINITE;
    if (fmin == fmin && fx > fmin) {
        // For a quadratic objective function, the minimum is such that:
        //
        //     fmin = f(x) - (1/2)*alpha*d'*∇f(x)
        //
        // with `alpha` the optimal step.  Hence:
        //
        //     alpha = 2*(f(x) - fmin)/(d'*∇f(x)) = 2*(f(x) - fmin)/‖d‖²
        //
        // is an estimate of the step size along `d` if it is plus or minus the
        // (projected) gradient.
        dnorm = optm_norm2(d);
        alpha = 2*(fx - fmin)/dnorm^2;
        if (alpha > 0 && alpha < INF) {
            return alpha;
        }
    } else {
        dnorm = -1; // Euclidean norm of `d` not yet computed.
    }
    if (delta == delta && delta > 0 && delta < 1) {
        // Use the specified small relative step size.
        if (dnorm < 0) {
            dnorm = optm_norm2(d);
        }
        xnorm = optm_norm2(x);
        alpha = delta*xnorm/dnorm;
        if (alpha > 0 && alpha < INF) {
            return alpha;
        }
    }
    if (lambda == lambda && lambda > 0 && lambda < INF) {
        // Use typical Hessian eigenvalue if suitable.
        alpha = 1.0/lambda;
        if (alpha > 0 && alpha < INF) {
            return alpha;
        }
    }
    // Eventually use 1/‖d‖.
    if (dnorm < 0) {
        dnorm = optm_norm2(d);
    }
    alpha = 1.0/dnorm;
    return alpha;
}

//-----------------------------------------------------------------------------
// L-BFGS APPROXIMATION

struct OptmLBFGS {
    double gamma; // gradient scaling
    long m;       // maximum number of previous step to memorize
    long mp;      // current number of memorized steps
    long mrk;     // index of last update
    pointer S;    // memorized variable changes
    pointer Y;    // memorized gradient changes
    pointer rho;  // memorized values of s'⋅y
};

func optm_new_lbfgs(m)
/* DOCUMENT lbfgs = optm_new_lbfgs(m);

     Create a new L-BFGS instance for storing up to `m` previous steps.

   SEE ALSO: optm_update_lbfgs, optm_apply_lbfgs.
 */
{
    if (m < 0) error, "invalid number of previous step to memorize";
    lbfgs = OptmLBFGS(m = m, mp = 0, mrk = 0);
    if (m > 0) {
        lbfgs.S   = &array(pointer, m);
        lbfgs.Y   = &array(pointer, m);
        lbfgs.rho = &array(double, m);
    }
    return lbfgs;
}

func optm_reset_lbfgs(&lbfgs)
/* DOCUMENT optm_reset_lbfgs, lbfgs;

     Reset the L-BFGS model stored in context `lbfgs`, thus forgetting any
     memorized information.  This also frees most memory allocated by the
     context.

   SEE ALSO: optm_update_lbfgs, optm_apply_lbfgs.
 */
{
    lbfgs.mp = 0;
    lbfgs.mrk = 0;
    lbfgs.gamma = 0.0;
    m = lbfgs.m;
    if (m > 0) {
        lbfgs.S   = &array(pointer, m);
        lbfgs.Y   = &array(pointer, m);
        lbfgs.rho = &array(double, m);
    } else {
        lbfgs.S   = &[];
        lbfgs.Y   = &[];
        lbfgs.rho = &[];
    }
    return lbfgs;
}

func optm_update_lbfgs(lbfgs, s, y)
/* DOCUMENT flg = optm_update_lbfgs(lbfgs, s, y);

     Update information stored by L-BFGS instance `lbfgs`.  Arguments `s` and
     `y` are the change in variables and gradient for the last iterate.  The
     returned value is a boolean indicating whether `s` and `y` were suitable
     to update an L-BFGS approximation that be positive definite.

     Even if `lbfgs.m = 0`, the value of the optimal gradient scaling
     `lbfgs.gamma` is updated if possible (i.e., when true is returned).

   SEE ALSO: optm_new_lbfgs, optm_apply_lbfgs, optm_inner.
 */
{
    sty = optm_inner(s, y);
    accept = (sty > 0);
    if (accept) {
        lbfgs.gamma = sty/optm_inner(y, y);
        m = lbfgs.m;
        if (m >= 1) {
            mrk = (lbfgs.mrk % m) + 1;
            (*lbfgs.S)(mrk) = &s;
            (*lbfgs.Y)(mrk) = &y;
            (*lbfgs.rho)(mrk) = sty;
            lbfgs.mrk = mrk;
            lbfgs.mp = min(lbfgs.mp + 1, m);
        }
    }
    return accept;
}

func optm_apply_lbfgs(lbfgs, d, &scaled, freevars)
/* DOCUMENT d = optm_apply_lbfgs(lbfgs, g, scaled);
         or d = optm_apply_lbfgs(lbfgs, g, scaled, freevars);

     Apply L-BFGS approximation of inverse Hessian to the "vector" `g`.
     Argument `lbfgs` is the structure storing the L-BFGS data.

     Optional argument `freevars` is to restrict the L-BFGS approximation to
     the sub-space spanned by the "free variables" not blocked by the
     constraints.  If specified and not empty, `freevars` shall have the size
     as `d` and shall be equal to zero where variables are blocked and to one
     elsewhere.

     On return, output variable `scaled` indicates whether any curvature
     information was taken into account.  If `scaled` is false, it means that
     the result `d` is identical to `g` except that `d(i)=0` if the `i`-th
     variable is blocked according to `freevars`.

   SEE ALSO: optm_new_lbfgs, optm_reset_lbfgs, optm_update_lbfgs, optm_inner.
 */
{
    // Variables.
    local S, Y, alpha, rho, gamma;

    // Determine the number of variables and of free variables.
    if (is_void(freevars) || min(freevars) != 0) {
        // All variables are free.
        regular = 1n;
    } else {
        // Convert `freevars` in an array of weights of suitable type.
        regular = 0n;
        T = (structof(d) == float ? float : double);
        if (structof(freevars) != T) {
            freevars = T(freevars);
        }
    }

    // Apply the 2-loop L-BFGS recursion algorithm by Matthies & Strang.
    mp = lbfgs.mp;
    if (mp >= 1) {
        eq_nocopy, S, *lbfgs.S;
        eq_nocopy, Y, *lbfgs.Y;
        m = lbfgs.m;
        off = lbfgs.mrk + m;
        alpha = array(double, m);
    }
    if (regular) {
        // Apply the regular L-BFGS recursion.
        if (mp >= 1) {
            eq_nocopy, rho, *lbfgs.rho;
        }
        gamma = lbfgs.gamma;
        for (j = 1; j <= mp; ++j) {
            i = (off - j)%m + 1;
            alpha_i = optm_inner(d, *S(i))/rho(i);
            optm_update, d, -alpha_i, *Y(i);
            alpha(i) = alpha_i;
        }
        if (gamma > 0 && gamma != 1) {
            optm_scale, d, gamma;
        }
        for (j = mp; j >= 1; --j) {
            i = (off - j)%m + 1;
            beta = optm_inner(d, *Y(i))/rho(i);
            optm_update, d, alpha(i) - beta, *S(i);
        }
    } else {
        // L-BFGS recursion on a subset of free variables specified by a
        // selection of indices.
        local s_i, y_i;
        rho = array(double, m);
        gamma = 0.0;
        d *= freevars; // restrict argument to the subset of free variables
        for (j = 1; j <= mp; ++j) {
            i = (off - j)%m + 1;
            eq_nocopy, s_i, *S(i);
            y_i = freevars*(*Y(i));
            rho_i = optm_inner(s_i, y_i);
            if (rho_i > 0) {
                if (gamma <= 0.0) {
                    gamma = rho_i/optm_inner(y_i, y_i);
                }
                alpha_i = optm_inner(d, s_i)/rho_i;
                optm_update, d, -alpha_i, y_i;
                alpha(i) = alpha_i;
                rho(i) = rho_i;
            }
            y_i = []; // free memory
        }
        if (gamma > 0 && gamma != 1) {
            optm_scale, d, gamma;
        }
        for (j = mp; j >= 1; --j) {
            i = (off - j)%m + 1;
            rho_i = rho(i);
            if (rho_i > 0) {
                beta = optm_inner(d, *Y(i))/rho_i;
                optm_update, d, alpha(i) - beta, freevars*(*S(i));
            }
        }
    }
    scaled = (gamma > 0);
    if (OPTM_DEBUG && !regular && anyof(d(where(!freevars)))) {
        error, "non-zero search direction for some blocked variables";
    }
    return d;
}

//-----------------------------------------------------------------------------
// SEPARABLE BOUND CONSTRAINTS

func optm_clamp(x, xmin, xmax)
/* DOCUMENT xp = optm_clamp(x, xmin, xmax);

     Restrict `x` to the range `[xmin,xmax]` element-wise.  Empty bounds, that
     is `xmin = []` or `xmax = []`, are interpreted as unlimited.

     If both bounds are specified, it is the caller's responsibility to ensure
     that the bounds are compatible, in other words that `xmin ≤ xmax` holds.

   SEE ALSO: optm_unblocked_variables and optm_line_search_limits.
 */
{
    if (!is_void(xmin)) {
        x = max(x, xmin);
    }
    if (!is_void(xmax)) {
        x = min(x, xmax);
    }
    return x;
}

func optm_unblocked_variables(x, xmin, xmax, g)
/* DOCUMENT msk = optm_unblocked_variables(x, xmin, xmax, g);

     Build a logical mask `msk` of the same size as `x` indicating which
     entries in `x` are not blocked by the bounds `xmin` and `xmax` when
     minimizing an objective function whose gradient is `g` at `x`.  In other
     words, the mask is false everywhere K.K.T. conditions are satisfied.

     Empty bounds, that is `xmin = []` or `xmax = []`, are interpreted as
     unlimited (as if `xmin = -Inf` and `xmax = +Inf`).

     It is the caller's responsibility to ensure that the bounds are compatible
     and that the variables are feasible, in other words that `xmin ≤ x ≤ xmax`
     holds element-wise.

     The type of of the entries of `msk` is `float` if `x` and `g` are single
     precision floating-point, `double` otherwise.

   SEE ALSO: optm_clamp and optm_line_search_limits.
 */
{
    T = structof(g);
    zero = T(0);
    T = (T == float && structof(x) == float ? float : double);
    if (is_void(xmin)) {
        if (is_void(xmax)) {
            return T(g != zero);
        } else {
            return T((g > zero)|((g < zero)&(x < xmax)));
        }
    } else {
        if (is_void(xmax)) {
            return T(((g > zero)&(x > xmin))|(g < zero));
        } else {
            return T(((g > zero)&(x > xmin))|((g < zero)&(x < xmax)));
        }
    }
}

func optm_line_search_limits(&amin, &amax, x0, xmin, xmax, d, dir)
/* DOCUMENT optm_line_search_limits, amin, amax, x0, xmin, xmax, d;
         or optm_line_search_limits, amin, amax, x0, xmin, xmax, d, dir;

     Determine the limits `amin` and `amax` for the step length `alpha` in a
     line-search where iterates `x` are given by:

         x = proj(x0 ± alpha*d)

     where `proj(x)` denotes the orthogonal projection on the convex set
     defined by separable lower and upper bounds `xmin` and `xmax` (unless
     empty) and where `±` is `-` if `dir` is specified and negative and `+`
     otherwise.

     On return, output variable `amin` is set to the largest nonnegative step
     length such that if `alpha ≤ amin`, then:

         proj(x0 ± alpha*d) = x0 ± alpha*d

     On return, output variable `amax` is set to the least nonnegative step
     length such that if `alpha ≥ amax`, then:

         proj(x0 ± alpha*d) = proj(x0 ± amax*d)

     In other words, no bounds are overcome if `0 ≤ alpha ≤ amin` and the
     projected variables are all the same for any `alpha` such that
     `alpha ≥ amax ≥ 0`.

     Restrictions: `x0` must be feasible and must have the same size as `d`;
     this is not verified for efficiency reasons.

   SEE ALSO: optm_clamp and optm_unblocked_variables.
 */
{
    INF = OPTM_INFINITE; // for nicer code ;-)
    unbounded_below = is_void(xmin);
    unbounded_above = is_void(xmax);
    amin = INF;
    if (unbounded_below && unbounded_above) {
        // Quick return if unconstrained.
        amax = INF;
        return;
    }
    amax = 0.0;
    backward = (!is_void(dir) && dir < 0); // Move in backward direction?
    if (unbounded_below) {
        if (backward ? (max(d) > 0) : (min(d) < 0)) {
            amax = INF;
        }
    } else {
        // Find positive step sizes to reach any lower bounds.
        a = [];
        if (backward) {
            i = where(d > 0);
            if (is_array(i)) {
                a = (x0 - xmin)(i)/d(i);
            }
        } else {
            i = where(d < 0);
            if (is_array(i)) {
                a = (xmin - x0)(i)/d(i);
            }
        }
        i = [];
        if (!is_void(a)) {
            amin = min(amin, min(a));
            amax = max(amax, max(a));
        }
    }
    if (unbounded_above) {
        // No upper bound set.
        if (amax < INF && (backward ? (min(d) < 0) : (max(d) > 0))) {
            amax = INF;
        }
    } else {
        // Find positive step sizes to reach any upper bounds.
        a = [];
        if (backward) {
            i = where(d < 0);
            if (is_array(i)) {
                a = (x0 - xmax)(i)/d(i);
            }
        } else {
            i = where(d > 0);
            if (is_array(i)) {
                a = (xmax - x0)(i)/d(i);
            }
        }
        i = [];
        if (!is_void(a)) {
            amin = min(amin, min(a));
            if (amax < INF) {
                amax = max(amax, max(a));
            }
        }
    }
}

//-----------------------------------------------------------------------------
// OPTIMIZATION METHODS
func optm_vmlmb(fg, x0, &f, &g, &status, lower=, upper=, mem=, fmin=, lnsrch=,
                delta=, epsilon=, lambda=, ftol=, gtol=, xtol=, blmvm=,
                maxiter=, maxeval=, verb=, printer=, output=, cputime=,
                observer=, throwerrors=)
/* DOCUMENT x = optm_vmlmb(fg, x0, [f, g, status,] lower=, upper=, mem=);

     Apply VMLMB algorithm to minimize a multi-variate differentiable objective
     function possibly under separable bound constraints.  VMLMB is a
     quasi-Newton method ("VM" is for "Variable Metric") with low memory
     requirements ("LM" is for "Limited Memory") and which can optionally take
     into account separable bound constraints (the final "B") on the variables.

     To determine efficient search directions, VMLMB approximates the Hessian
     of the objective function by a limited memory version of the model assumed
     in Broyden-Fletcher-Goldfarb-Shanno algorithm (called L-BFGS for short).
     Hence VMLMB is well suited to solving optimization problems with a very
     large number of variables possibly with bound constraints.

     The method has two required arguments: `fg`, the function to call to
     compute the objective function and its gradient, and `x0`, the initial
     variables (VMLMB is an iterative method).  The initial variables may be an
     array of any dimensions.

     The method returns `x` the best solution found during iterations.
     Arguments `f`, `g` and `status` are optional output variables to store the
     value and the gradient of the objective at `x` and an integer code
     indicating the reason of the termination of the algorithm (see
     `optm_reason`).

     The function `fg` shall be implemented as follows:

         func fg(x, &g)
         {
             f = ...; // value of the objective function at `x`
             g = ...; // gradient of the objective function at `x`
             return f;
         }

     Arguments `f`, `g` and `status` are optional output variables to store the
     value and the gradient of the objective at the returned solution and to
     store the integer code indicating the reason of the termination of the
     algorithm (see `optm_reason`).

     All other settings are specified by keywords:

     - Keywords `upper` and `lower` are to specify a lower and/or an upper
       bounds for the variables.  If unspecified or set to an empty array, a
       given bound is considered as unlimited.  Bounds must be conformable with
       the variables.

     - Keyword `mem` specifies the memory used by the algorithm, that is the
       number of previous steps memorized to approximate the Hessian of the
       objective function.  With `mem=0`, the algorithm behaves as a steepest
       descent method.  The default is `mem=5`.

     - Keywords `ftol`, `gtol` and `xtol` specify tolerances for deciding the
       convergence of the algorithm.

       Convergence in the function occurs if one of the following conditions
       hold:

           f ≤ fatol
           |f - fp| ≤ frtol⋅max(|f|, |fp|)

       where `f` and `fp` are the values of the objective function at the
       current and previous iterates.  In these conditions, `fatol` and `frtol`
       are absolute and relative tolerances specified by `ftol` which can be
       `ftol=[fatol,frtol]` or `ftol=frtol` and assume that `fatol=-Inf`.  The
       default is `ftol=1e-8`.

       Convergence in the gradient occurs if the following condition holds:

           ‖g‖ ≤ max(0, gatol, grtol⋅‖g0‖)

       where `‖g‖` is the Euclidean norm of the projected gradient, `g0` is the
       projected gradient at the initial solution.  In this condition, `gatol`
       and `grtol` are absolute and relative gradient tolerances specified by
       `gtol` which can be `gtol=[gatol,grtol]` or `gtol=grtol` and assume that
       `gatol=0`.  The default is `gtol=1e-5`.

       Convergence in the variables occurs if the following condition holds:

           ‖x - xp‖ ≤ max(0, xatol, xrtol*‖x‖)

       where `x` and `xp` are the current and previous variables.  In this
       condition, `xatol` and `xrtol` are absolute and relative tolerances
       specified by `xtol` which can be `xtol=[fatol,frtol]` or `xtol=xrtol`
       and assume that `xatol=0`.  The default is `xtol=1e-6`.

     - Keywords `maxiter` and `maxeval` are to specify a maximum number of
       algorithm iterations or or evaluations of the objective function
       implemented by `fg`.  By default, these are unlimited.

     - Keyword `lnsrch` is to specify line-search settings different than the
       default (see `optm_new_line_search`).

     - Keyword `fmin` is to specify an estimation of the minimum possible value
       of the objective function.  This setting may be used to determine the
       step length along the steepest descent.

     - Keyword `delta` specifies a small size relative to the variables.  This
       setting may be used to determine the step length along the steepest
       descent.

     - Keyword `lambda` specifies an estimate of the magnitude of the
       eigenvalues of the Hessian of the objective function.  This setting may
       be used to determine the step length along the steepest descent.

     - Keyword `epsilon` specifies a threshold for a sufficient descent
       condition.  If `epsilon > 0`, then a search direction `d` computed by
       the L-BFGS approximation is considered as acceptable if:

           ⟨d,g⟩ ≤ -epsilon⋅‖d‖⋅‖g‖

       where `g` denotes the projected gradient of the objective function
       (which is just the gradient in unconstrained case).  Otherwise, the
       condition writes `⟨d,g⟩ < 0`.  The default is `epsilon = 0` so only the
       latter condition is checked.

     - Keyword `blmvm` (false by default) specifies whether to use BLMVM trick
       to account for the bound constraints in the L-BFGS model of the Hessian.
       If `blmvm` is set true, the overhead of the algorithm may be reduced,
       but the L-BFGS model of the Hessian is more likely to be inaccurate
       causing the algorithm to choose the steepest descent direction more
       often.

     - Keyword `verb`, if positive, specifies to print information every `verb`
       iterations.  Nothing is printed if `verb ≤ 0`.  By default, `verb = 0`.

     - Keyword `printer` is to specify a user-defined subroutine to print
       information every `verb` iterations.  This subroutine is called as:

           printer, output, iters, evals, rejects, t, x, f, g, pgnorm,
               alpha, fg;

       with `output` the output stream specified by keyword `output`, `iters`
       the number of algorithm iterations, `evals` the number of calls to `fg`,
       `rejects` the number of rejections of the LBFGS direction, `t` the
       elapsed time in seconds, `x` the current variables, `f` and `g` the
       value and the gradient of the objective function at `x`, `pgnorm` the
       Euclidean norm of the projected gradient of the objective function at
       `x`, `alpha` the last step length, and `fg` the objective function
       itself.

     - Keyword `output` specifies the file stream to print information.

     - Keyword `observer` is to specify a user-defined subroutine to be called
       at every iteration as follows:

           observer, iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg;

       with the same arguments as for the printer (except `output`).

     - If keyword `cputime` is true, the CPU time instead of the WALL time is
       used for the printer and the observer.

     - Keyword `throwerrors` (true by default), specifies whether to call
       `error` in case of errors instead or returning a `status` indicating the
       problem.  Note that early termination due to limits set on the number of
       iterations or of evaluations of the objective function are not
       considered as an error.
 */
{
    // Constants.
    INF = OPTM_INFINITE;
    NAN = OPTM_QUIET_NAN;
    TRUE = 1n;
    FALSE = 0n;

    // Parse settings.
    if (is_void(mem)) mem = 5;
    if (is_void(maxiter)) maxiter = INF;
    if (is_void(maxeval)) maxeval = INF;
    if (is_void(ftol)) ftol = 1.0E-8;
    if (is_void(gtol)) gtol = 1.0E-5;
    if (is_void(xtol)) xtol = 1.0E-6;
    if (is_void(lnsrch)) lnsrch = optm_new_line_search();
    if (is_void(verb)) verb = 0;
    if (is_void(printer)) printer = _optm_vmlmb_printer;
    if (is_void(fmin)) fmin = NAN;
    if (is_void(delta)) delta = NAN;
    if (is_void(epsilon)) epsilon = 0.0;
    if (is_void(lambda)) lambda = NAN;
    if (is_void(blmvm)) blmvm = FALSE;
    if (is_void(throwerrors)) throwerrors = TRUE;

    // Tolerances.  Most of these are forced to be nonnegative to simplify
    // tests.
    if (is_scalar(ftol)) {
        fatol = -INF;
        frtol = max(0.0, ftol);
    } else {
        fatol = max(0.0, ftol(1));
        frtol = max(0.0, ftol(2));
    }
    if (is_scalar(gtol)) {
        gatol = 0.0;
        grtol = max(0.0, gtol);
    } else {
        gatol = max(0.0, gtol(1));
        grtol = max(0.0, gtol(2));
    }
    if (is_scalar(xtol)) {
        xatol = 0.0;
        xrtol = max(0.0, xtol);
    } else {
        xatol = max(0.0, xtol(1));
        xrtol = max(0.0, xtol(2));
    }

    // Bound constraints.  For faster code, unlimited bounds are preferentially
    // represented by empty arrays.
    if (is_array(lower) && allof(lower == -INF)) lower = [];
    if (is_array(upper) && allof(upper == +INF)) upper = [];
    bounded = (!is_void(lower) || !is_void(upper));
    if (!bounded) {
        blmvm = FALSE; // no needs to use BLMVM trick in the unconstrained case
    }

    // Other initialization.
    x = unref(x0);   // initial iterate (avoiding copy)
    g = [];          // gradient
    f0 = +INF;       // function value at start of line-search
    g0 = [];         // gradient at start of line-search
    d = [];          // search direction
    s = [];          // effective step
    pg = [];         // projected gradient
    pg0 = [];        // projected gradient at start of line search
    pgnorm = 0.0;    // Euclidean norm of the (projected) gradient
    alpha = 0.0;     // step length
    amin = -INF;     // first step length threshold
    amax = +INF;     // last step length threshold
    evals = 0;       // number of calls to `fg`
    iters = 0;       // number of iterations
    projs = 0;       // number of projections onto the feasible set
    rejects = 0;     // number of search direction rejections
    status = 0;      // non-zero when algorithm is about to terminate
    best_f = +INF;   // function value at `best_x`
    best_g = [];     // gradient at `best_x`
    best_x = [];     // best solution found so far
    best_pgnorm = -1;// norm of projected gradient at `best_x` (< 0 if unknown)
    best_alpha =  0; // step length at `best_x` (< 0 if unknown)
    best_evals = -1; // number of calls to `fg` at `best_x`
    last_evals = -1; // number of calls to `fg` at last iterate
    last_print = -1; // iteration number for last print
    last_obsrv = -1; // iteration number for last call to observer
    freevars = [];   // subset of free variables (not yet known)
    lbfgs = optm_new_lbfgs(mem);
    if (verb > 0) {
        time_index = (cputime ? 1 : 3);
        elapsed = array(double, 3);
        timer, elapsed;
        t0 = elapsed(time_index);
    }
    call_observer = !is_void(observer);

    // Algorithm stage follows that of the line-search, it is one of:
    // 0 = initially;
    // 1 = line-search in progress;
    // 2 = line-search has converged.
    stage = 0;

    while (TRUE) {
        // Make the variables feasible.
        if (bounded) {
            // In principle, we can avoid projecting the variables whenever
            // `alpha ≤ amin` (because the feasible set is convex) but rounding
            // errors could make this wrong.  It is safer to always project the
            // variables.  This cost O(n) operations which are probably
            // negligible compared to, say, computing the objective function
            // and its gradient.
            x = optm_clamp(unref(x), lower, upper);
            projs += 1;
        }
        // Compute the objective function and its gradient.
        g = [];
        f = fg(x, g);
        evals += 1;
        if (f < best_f || evals == 1) {
            // Save best solution so far.
            best_f = f;
            eq_nocopy, best_g, g;
            eq_nocopy, best_x, x;
            best_pgnorm = -1; // must be recomputed
            best_alpha = alpha;
            best_evals = evals;
        }
        if (stage != 0) {
            // Line-search in progress, check for line-search convergence.
            stage = optm_iterate_line_search(lnsrch, f).stage;
            if (stage == 2) {
                // Line-search has converged, `x` is the next iterate.
                iters += 1;
                last_evals = evals;
            } else if (stage == 1) {
                // Line-search has not converged, peek next trial step.
                alpha = lnsrch.step;
            } else {
                error, "something is wrong!";
            }
        }
        if (stage != 1) {
            // Initial or next iterate after convergence of line-search.
            if (bounded) {
                // Determine the subset of free variables and compute the norm
                // of the projected gradient (needed to check for convergence).
                freevars = optm_unblocked_variables(x, lower, upper, g);
                pg = freevars*g;
                pgnorm = optm_norm2(pg);
                if (!blmvm) {
                    // Projected gradient no longer needed, free some memory.
                    pg = [];
                }
            } else {
                // Just compute the norm of the gradient.
                pgnorm = optm_norm2(g);
            }
            if (evals == best_evals) {
                // Now we know the norm of the (projected) gradient at the best
                // solution so far.
                best_pgnorm = pgnorm;
            }
            // Check for algorithm convergence or termination.
            if (evals == 1) {
                // Compute value for testing the convergence in the gradient.
                gtest = max(gatol, grtol*pgnorm);
            }
            if (pgnorm <= gtest) {
                // Convergence in gradient.
                status = OPTM_GTEST_SATISFIED;
                break;
            }
            if (stage == 2) {
                // Check convergence in relative function reduction.
                if (f <= fatol || abs(f - f0) <= frtol*max(abs(f), abs(f0))) {
                    status = OPTM_FTEST_SATISFIED;
                    break;
                }
                // Compute the effective change of variables.
                s = x - unref(x0);
                snorm = optm_norm2(s);
                // Check convergence in variables.
                if (snorm <= xatol
                    || (xrtol > 0 && snorm <= xrtol*optm_norm2(x))) {
                    status = OPTM_XTEST_SATISFIED;
                    break;
                }
            }
            if (iters >= maxiter) {
                status = OPTM_TOO_MANY_ITERATIONS;
                break;
            }
        }
        if (evals >= maxeval) {
            status = OPTM_TOO_MANY_EVALUATIONS;
            break;
        }
        if (stage != 1) {
            // Call user defined observer.
            timer, elapsed;
            t = elapsed(time_index) - t0;
            if (call_observer) {
                observer, iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg;
                last_obsrv = iters;
            }
            // Possibly print iteration information.
            if (verb > 0 && (iters % verb) == 0) {
                printer, output, iters, evals, rejects, t, x, f, g, pgnorm,
                    alpha, fg;
                last_print = iters;
            }
            if (stage != 0) {
                // At least one step has been performed, L-BFGS approximation
                // can be updated.
                if (blmvm) {
                    optm_update_lbfgs, lbfgs, unref(s), pg - pg0;
                } else {
                    optm_update_lbfgs, lbfgs, unref(s), g - g0;
                }
            }
            // Determine a new search direction `d`.  Parameter `dir` is set to:
            //   0 if `d` is not a search direction,
            //   1 if `d` is unscaled steepest descent,
            //   2 if `d` is scaled sufficient descent.
            dir = 0;
            // Use L-BFGS approximation to compute a search direction and check
            // that it is an acceptable descent direction.
            local scaled;
            if (blmvm) {
                d = optm_apply_lbfgs(lbfgs, -pg, scaled)*freevars;
            } else {
                d = optm_apply_lbfgs(lbfgs, -g, scaled, freevars);
            }
            dg = optm_inner(d, g);
            if (!scaled) {
                // No exploitable curvature information, `d` is the unscaled
                // steepest feasible direction, that is the opposite of the
                // projected gradient.
                dir = 1;
            } else {
                // Some exploitable curvature information were available.
                dir = 2;
                if (dg >= 0) {
                    // L-BFGS approximation does not yield a descent direction.
                    dir = 0; // discard search direction
                    if (!bounded) {
                        if (throwerrors) {
                            error, "L-BFGS approximation is not positive definite";
                        }
                        status = OPTM_NOT_POSITIVE_DEFINITE;
                        break;
                    }
                } else if (epsilon > 0) {
                    // A more restrictive criterion has been specified for
                    // accepting a descent direction.
                    if (dg > -epsilon*optm_norm2(d)*pgnorm) {
                        dir = 0; // discard search direction
                    }
                }
            }
            if (dir == 0) {
                // No exploitable information about the Hessian is available or
                // the direction computed using the L-BFGS approximation failed
                // to be a sufficient descent direction.  Take the steepest
                // feasible descent direction.
                d = -(bounded ? g*freevars : g);
                dg = -pgnorm^2;
                dir = 1; // scaling needed
            }
            if (dir != 2 && iters > 0) {
                ++rejects;
            }
            // Determine the length `alpha` of the initial step along `d`.
            if (dir == 2) {
                // The search direction is already scaled.
                alpha = 1.0;
            } else {
                // Find a suitable step size along the steepest feasible
                // descent direction `d`.  Note that `pgnorm`, the Euclidean
                // norm of the (projected) gradient, is also that of `d` in
                // that case.
                alpha = optm_steepest_descent_step(x, pgnorm, f, fmin,
                                                   delta, lambda);
            }
            if (bounded) {
                // Safeguard the step to avoid searching in a region where
                // all bounds are overreached.
                optm_line_search_limits, amin, amax, x, lower, upper, d, alpha;
                alpha = min(alpha, amax);
            }
            // Initialize line-search.
            stage = optm_start_line_search(lnsrch, f, dg, alpha).stage;
            if (stage != 1) {
                error, "something is wrong!";
            }
            // Save iterate at start of line-search.
            f0 = f;
            eq_nocopy, g0, g;
            eq_nocopy, x0, x;
            if (blmvm) {
                eq_nocopy, pg0, pg;
            }
        }
        // Compute next iterate.
        if (alpha == 1) {
            x = x0 + d;
        } else {
            x = x0 + alpha*d;
        }
    }

    // In case of abnormal termination, some progresses may have been made
    // since the start of the line-search.  In that case, we restore the best
    // solution so far.
    if (best_f < f) {
        f = best_f;
        eq_nocopy, g, best_g;
        eq_nocopy, x, best_x;
        if (verb > 0) {
            // Restore other information for printing.
            alpha = best_alpha;
            if (best_pgnorm >= 0) {
                pgnorm = best_pgnorm;
            } else {
                // Re-compute the norm of the (projected) gradient.
                if (bounded) {
                    freevars = optm_unblocked_variables(x, lower, upper, g);
                    pgnorm = optm_norm2(g*freevars);
                } else {
                    pgnorm = optm_norm2(g);
                }
            }
            if (f < f0) {
                // Some progresses since last iterate, pretend that one more
                // iteration has been performed.
                ++iters;
            }
        }
    }
    timer, elapsed;
    t = elapsed(time_index) - t0;
    if (call_observer && iters > last_obsrv) {
        observer, iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg;
    }
    if (verb > 0) {
        if (iters > last_print) {
            printer, output, iters, evals, rejects, t, x, f, g, pgnorm,
                alpha, fg;
        }
        write, output, format="# Termination: %s\n", optm_reason(status);
    }
    return x;
}

func _optm_vmlmb_printer(output, iters, evals, rejects, t, x, f, g, pgnorm,
                         alpha, fg)
{
    if (iters < 1) {
        write, output, format="%s%s\n%s%s\n",
            "# Iter.   Time (ms)    Eval. Reject.",
            "       Obj. Func.           Grad.       Step",
            "# ----------------------------------",
            "-----------------------------------------------";
    }
    write, output, format="%7d %11.3f %7d %7d %23.15e %11.3e %11.3e\n",
        iters, t*1e3, evals, rejects, f, pgnorm, alpha;
}

//-----------------------------------------------------------------------------
// UTILITIES AND ALGEBRA

local optm_inner;
func _optm_inner(x, y)
/* DOCUMENT optm_inner(x, y);

     Yields the inner product of `x` and `y` that is `sum(x*y)`, but computed as
     efficiently as possible.

   SEE ALSO: optm_norm2.
 */
{
    return sum(x*y);
}

local optm_norm1;
func _optm_norm1(x)
/* DOCUMENT optm_norm1(x);

     Yields the L1-norm of `x`, that is `sum(abs(x))` but computed as
     efficiently as possible.

   SEE ALSO: optm_norm2, optm_norminf.
 */
{
    return (is_scalar(x) ? abs(x) : sum(abs(x)));
}

local optm_norm2;
func _optm_norm2(x)
/* DOCUMENT optm_norm2(x);

     Yields the Euclidean (L2) norm of `x` that is `sqrt(sum(x^2))` but
     computed as efficiently as possible.

   SEE ALSO: optm_inner, optm_norm1, optm_norminf.
 */
{
    return (is_scalar(x) ? abs(x) : sqrt(sum(x*x)));
}

local optm_norminf;
func _optm_norminf(x)
/* DOCUMENT optm_norminf(x);

     Yields the infinite norm of `x`, that is `max(abs(x))` but computed as
     efficiently as possible.

   SEE ALSO: optm_norm1, optm_norm2.
 */
{
    return (is_scalar(x) ? abs(x) : max(-min(x), max(x)));
}

local optm_scale;
func _optm_scale(&x, alpha)
/* DOCUMENT optm_scale(x, alpha);
         or optm_scale, x, alpha;

     Compute `alpha*x` efficiently and taking care of preserving the
     floating-point type of `x`.  Argument `x` is overwritten with the result
     when `optm_scale` is called as a subroutine.
 */
{
    alpha = structof(x) == float ? float(alpha) : double(alpha);
    if (am_subroutine()) {
        x *= alpha;
    } else {
        return alpha*x;
    }
}

local optm_update;
func _optm_update(&y, alpha, x)
/* DOCUMENT optm_update, y, alpha, x;

     Compute `y += alpha*x` efficiently and taking care of preserving the
     floating-point type of `x` and `y`.
 */
{
    T = (structof(x) == float && structof(y) == float) ? float : double;
    y += T(alpha)*x;
}

func optm_tolerance(x, atol, rtol)
/* DOCUMENT tol = optm_tolerance(x, atol, rtol);

     Given absolute and relative tolerances ATOL and RTOL, yields:

         max(0, atol, rtol*abs(x))    // if `x` is a scalar
         max(0, atol, rtol*norm(x))   // if `x` is an array

    where norm(X) is the Euclidean norm of `x` as computed by optm_norm2 (which
    to see).  If RTOL ≤ 0, the computation of norm(X) is avoided.

   SEE ALSO: optm_norm2.
 */
{
    tol = max(0.0, atol);
    if (rtol <= 0.0) {
        return tol;
    }
    if (is_scalar(x)) {
        a = abs(x);
    } else {
        a = optm_norm2(x);
    }
    return max(tol, rtol*a);
}

func optm_same_dims(a, b)
/* DOCUMENT optm_same_dims(a, b);

     Check whether arrays `a` and `b` have the same dimensions.

   SEE ALSO: dimsof.
 */
{
    adims = dimsof(a);
    bdims = dimsof(b);
    return numberof(adims) == numberof(bdims) && allof(adims == bdims);
}

func optm_floating_point(type, what)
/* DOCUMENT optm_floating_point(type, what);

     yields a special floating-point value of given `type` (must be `float` or
     `double`) corresponding to `what`:

       what = 1 or "Inf", for positive infinite;
       what = 2 or "qNaN" or "NaN", for quiet NaN (Not a Number);
       what = 3 or "sNaN", for signaling NaN.

    if `what` is a siting case is irrelevant.

   SEE ALSO: ieee_set.
 */
{
    err = 0n;
    if (!is_scalar(what)) {
        err = 1n;
    } else if (is_string(what)) {
        key = strcase(0, what);
        if (key == "inf") {
            what = 1;
        } else if (key == "qnan" || key == "nan") {
            what = 2;
        } else if (key == "snan") {
            what = 3;
        } else {
            err = 1n;
        }
    } else if (!is_integer(what)) {
        err = 1n;
    }
    if (err) {
        error, "invalid value for WHAT";
    }
    val = array(type, 1);
    ieee_set, val, what;
    return val(1);
}

local OPTM_INFINITE, OPTM_QUIET_NAN;
/* DOCUMENT
     Special floating-point constants:

         OPTM_INFINITE  = positive infinite;
         OPTM_QUIET_NAN = quiet NaN (Not a NUmber);

     These constants are double precision floating-point values, call
     `float(val)` to convert `val` to single precision.

     Note using NaN's (even the quiet ones) is very likely to raise a
     floating-point error in Yorick (this is considered as a feature by Dave
     Munro), except for testing equality, i.e. `x == x` and `x = x` repectively
     yield false and true when `x` is a quiet NaN.


   SEE ALSO: optm_floating_point.
*/
OPTM_INFINITE = optm_floating_point(double, "Inf");
OPTM_QUIET_NAN = optm_floating_point(double, "qNaN");

func optm_override_functions(mode)
/* DOCUMENT optm_override_functions, mode;

     If `mode = "fast"`, attempt to load `vops.i` plug-in and use fast
     vectorized functions to perform basic linear algebra operations.

     If `mode = "slow"`, use slower interpreted functions to perform basic
     linear algebra operations.

     If `mode = "tryfast"`, then the fast version is used if the `vops.i`
     plug-in is available and the slow version is used otherwise.

   SEE ALSO: optm_inner, optm_norm1, optm_norm2, optm_norminf, optm_scale,
             optm_update.
 */
{
    extern optm_inner;
    extern optm_norm1;
    extern optm_norm2;
    extern optm_norminf;
    extern optm_scale;
    extern optm_update;
    if (mode == "fast" || mode == "tryfast") {
        // Try to use optimized operations (this must be done last).
        if (is_func(vops_inner) == 3) {
            // vops_inner is an autoload object, include "vops.i" now.
            include, "vops.i", 3;
        }
        if (is_func(vops_inner) == 2) {
            optm_inner   = vops_inner;
            optm_norm1   = vops_norm1;
            optm_norm2   = vops_norm2;
            optm_norminf = vops_norminf;
            optm_scale   = vops_scale;
            optm_update  = vops_update;
            return;
        }
        mesg = "cannot load \"vops.i\"";
        if (mode == "fast") {
            error, mesg;
        }
        write, format="WARNING: %s\n", mesg;
        mode = "slow";
    }
    if (mode == "slow") {
        optm_inner   = _optm_inner;
        optm_norm1   = _optm_norm1;
        optm_norm2   = _optm_norm2;
        optm_norminf = _optm_norminf;
        optm_scale   = _optm_scale;
        optm_update  = _optm_update;
    } else {
        error, "argument must be \"fast\" or \"slow\"";
    }
}

optm_override_functions, "tryfast";
