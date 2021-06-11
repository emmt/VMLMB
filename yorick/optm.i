// optm.i -
//
// Multi-dimensional optimization for Yorick.
//-----------------------------------------------------------------------------

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
    } else if (status == OPTM_TOO_MANY_ITERATIONS) {
        return "too many iterations";
    } else if (status ==  OPTM_FTEST_SATISFIED) {
        return "function reduction test satisfied";
    } else if (status == OPTM_GTEST_SATISFIED) {
        return "gradient test satisfied";
    } else if (status == OPTM_XTEST_SATISFIED) {
        return "variables change test satisfied";
    } else {
        return "unknown status code";
    }
}

//-----------------------------------------------------------------------------
// LINEAR CONJUGATE GRADIENT

func optm_conjgrad(A, b, x, &status, precond=, maxiter=, restart=, verbose=,
                   output=, ftol=, gtol=, xtol=)
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
       default, `restart = min(50,numel(x)+1)`.  Set `restart` to a value less
       or equal zero or greater than `maxiter` if you do not want that any
       restarts ever occur.

     - Keyword `verbose` is to specify whether to print various information at
       each iteration.

     - Keywords `ftol`, `gtol` and `xtol` specify tolerances for deciding the
       convergence of the algorithm.  In what follows, `x_{k}`,
       `f_{k}=f(x_{k})`, and `∇f_{k}=∇f(x_{k})` denotes the variables, the
       objective function and its gradient after `k` iterations ot the
       algorithm (`x_{0} = x0` the intial estimate).

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

     The function retuns the solution `x` and set optional output variable
     `status` with one of the following termination codes:

     * `status = OPTM_NOT_POSITIVE_DEFINITE` if the left-hand-side matrix `A`
       is found to be not positive definite;

     * `status = OPTM_TOO_MANY_ITERATIONS` if the maximum number of iterations
       have been reached;

     * `status = OPTM_FTEST_SATISFIED` if convergence occurred because the
       function reduction satisfies the criterion specified by `ftol`;

     * `status = OPTM_GTEST_SATISFIED` if convergence occurred because the
       gradient norm satisfies the criterion specified by `gtol`;

     * `status = OPTM_XTEST_SATISFIED` if convergence occurred because the norm
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
    if (is_void(xtol)) xtol = 0.0;
    if (is_void(xrtol)) xrtol = 1.0E-6;
    if (is_void(maxiter)) maxiter = 2*numberof(b) + 1;
    if (is_void(restart)) restart = 50;
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
    if (is_void(x)) {
        x = array(structof(b), dimsof(b));
        x_is_zero = 1n;
    } else {
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
    if (verbose) {
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
                eq_nocopy, r, b;
                x_is_zero = 0n;
            } else {
                // Compute r = b - A*x.
                r = b - A(x);
            }
        } else {
            // Update residuals.
            r = unref(r) - alpha*q;
        }
        if (preconditioned) {
            // Apply preconditioner `z = M⋅r`.
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
        if (verbose) {
            timer, elapsed;
            t = (elapsed(3) - t0)*1E3; // elapsed time in ms
            if (preconditioned) {
                if (k == 0) {
                    write, output, format="%s%s\n%s%s\n",
                        "# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖  ",
                        "   ‖∇f(x)‖_M",
                        "# --------------------------------------------",
                        "-------------";
                }
                write, output, format="%7d %11.3f %12.4e %12.4e %12.4e\n",
                    k, t, phi, optm_norm2(r), sqrt(rho);
            } else {
                if(k == 0) {
                    write, output, format="%s\n%s\n",
                        "# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖",
                        "# --------------------------------------------";
                }
                write, output, format="%7d %11.3f %12.4e %12.4e\n",
                    k, t, phi, sqrt(rho);
            }
        }
        if (2.0*sqrt(rho) <= gtest) {
            // Normal convergence in the gradient norm.
            if (verbose) {
                write, output, format="# %s\n",
                    "Convergence in the gradient norm.";
            }
            status = OPTM_GTEST_SATISFIED;
            return x;
        }
        if (k >= maxiter) {
            if (verbose) {
                write, output, format="# %s\n", "Too many iteration(s).";
            }
            status = OPTM_TOO_MANY_ITERATIONS;
            return x;
        }

        // Compute search direction.
        if (restarting) {
            // Restarting or first iteration.
            eq_nocopy, p, z;
        } else {
            // Apply recurrence.
            beta = rho/oldrho;
            p = z + beta*p;
        }

        // Compute optimal step size.
        q = [];
        q = A(p); // q = A*p
        gamma = optm_inner(p, q);
        if (!(gamma > 0.0)) {
            if (verbose) {
                write, output, format="# %s\n",
                    "Operator is not positive definite.";
            }
            status = OPTM_NOT_POSITIVE_DEFINITE;
            return x;
        }
        alpha = rho/gamma;

        // Update variables and check for convergence.
        x = unref(x) + alpha*p;
        phi = alpha*rho/2.0; // phi = f(x_{k}) - f(x_{k+1}) ≥ 0
        phimax = max(phi, phimax);
        if (phi <= optm_tolerance(phimax, fatol, frtol)) {
            // Normal convergence in the function reduction.
            if (verbose) {
                write, output, format="# %s\n",
                    "Convergence in the function reduction.";
            }
            status = OPTM_FTEST_SATISFIED;
            return x;
        }
        if (xtest && alpha*optm_norm2(p) <= optm_tolerance(x, xatol, xrtol)) {
            // Normal convergence in the variables.
            if (verbose) {
                write, output, format="# %s\n",
                    "Convergence in the variables.";
            }
            status = OPTM_XTEST_SATISFIED;
            return x;
        }
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

         f(x0 + alpha*d) <= f(x0) + ftol*df(x0)*alpha

     where `f(x)` is the objective function at `x`, `x0` denotes the variables
     at the start of the line-search, `df(x0)` is the directional derivative of
     the objective function at `x0`, `alpha` is the step length and `d` is the
     search direction.  The value of `ftol` must be in the range `(0,0.5]`, the
     default value is `ftol = 1E-4`.

     Keywords `smin` and `smax` can be used to specify relative bounds for
     safeguarding the step length.  When a step `alpha` is unsuccessful, a new
     backtracking step is computed based on a parabolic interpolation of the
     objective function along the search direction.  The new step writes:

         new_alpha = gamma*alpha

     with `gamma` safeguarded in the range `[smin,smax]`.  The following
     constraints must hold: `0 < smin ≤ smax < 1`.  Taking `smin = smax = 0.5`
     emulates the usual Armijo's method.  Default values are `smin = 0.2` and
     `smax = 0.9`.

     The subroutine `optm_start_line_search` shall be called to initialize each
     new line-search with arguments: `lnsrch` the line-search instance, `f0`
     the objective function at `x0` the variables at the start of the
     line-search, `df0` the directional derivative of the objective function at
     `x0` and `stp > 0` a guess for the first step to try.

     Note that when Armijo's conditon does not hold, the quadratic
     interpolation yields `gamma < 1/(2 - 2*ftol)`.  Hence, taking an upper
     bound `smax > 1/(2 - 2*ftol)` has no effects while taking a lower bound
     `smin ≥ 1/(2 - 2*ftol)` yields a safeguarded `gamma` always equal to
     `smin`.  Therefore, to benefit from quadratic interpolation, one should
     choose `smin < 1/(2 - 2*ftol)`.

     The subroutine `optm_iterate_line_search` shall be called to pursue the
     line-search arguments: `lnsrch` the line-search instance, `f` the
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

func optm_iterate_line_search(&lnsrch, f)
{
    finit = lnsrch.finit;
    ginit = lnsrch.ginit;
    step  = lnsrch.step;
    ftol  = lnsrch.ftol;
    if (f <= finit + ftol*(ginit*step)) {
        // Line-search has converged.
        lnsrch.stage = 2;
    } else {
        // Line-search has not converged.
        smin = lnsrch.smin;
        smax = lnsrch.smax;
        if (smin < smax) {
            // Compute a safeguarded parabolic interpolation step.
            q = -ginit*step;
            r = 2*((f - finit) + q);
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
            dnorm =  optm_norm2(d);
        }
        xnorm = optm_norm2(x);
        alpha = delta*xnorm/dnorm;
        if (alpha > 0 && alpha < INF) {
            return alpha;
        }
    }
    // Use typical Hessian eigenvalue if suitable.
    if (lambda == lambda && lambda > 0 && lambda < INF) {
        alpha = 1/lambda;
        if (alpha > 0 && alpha < INF) {
            return alpha;
        }
    }
    // Eventually use 1/‖d‖.
    if (dnorm < 0) {
        dnorm =  optm_norm2(d);
    }
    alpha = 1/dnorm;
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
     the sub-space spanned by the "free variables".  If specified and not
     empty, `freevars` shall have the size as `d` and shall be equal to zero
     where variables are blocked and to one elsewhere.

     On return, output variable `scaled` indicates whether any curvature
     information was taken into account.  If `scaled` is false, it means that
     the result `d` is identical to `g` except that `d(i)=0` if the `i`-th
     variable is blocked according to `freevars`.

   SEE ALSO: optm_new_lbfgs, optm_reset_lbfgs, optm_update_lbfgs, optm_inner.
 */
{
    // Variables.
    local S, Y, alpha, rho, gamma, msk;

    // Determine the number of variables and of free variables.
    if (is_void(freevars) || allof(freevars)) {
        // All variables are free.
        regular = 1n;
    } else {
        // Convert `freevars` in an array of weights of suitable type.
        regular = 0n;
        T = (structof(d) == float ? float : double);
        if (structof(freevars) == T) {
            eq_nocopy, msk, freevars;
        } else {
            msk = T(freevars);
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
            d -= alpha_i*(*Y(i));
            alpha(i) = alpha_i;
        }
        if (gamma > 0 && gamma != 1) {
            d = gamma*unref(d);
        }
        for (j = mp; j >= 1; --j) {
            i = (off - j)%m + 1;
            beta = optm_inner(d, *Y(i))/rho(i);
            d += (alpha(i) - beta)*(*S(i));
        }
    } else {
        // L-BFGS recursion on a subset of free variables specified by a
        // selection of indices.
        local s_i, y_i;
        rho = array(double, m);
        gamma = 0.0;
        d *= msk; // restrict argument to the subset of free variables
        for (j = 1; j <= mp; ++j) {
            i = (off - j)%m + 1;
            eq_nocopy, s_i, *S(i);
            y_i = msk*(*Y(i));
            rho_i = optm_inner(s_i, y_i);
            if (rho_i > 0) {
                if (gamma <= 0.0) {
                    gamma = rho_i/optm_inner(y_i, y_i);
                }
                alpha_i = optm_inner(d, s_i)/rho_i;
                d -= alpha_i*y_i;
                alpha(i) = alpha_i;
                rho(i) = rho_i;
            }
            y_i = []; // free memory
        }
        if (gamma > 0 && gamma != 1) {
            d = gamma*unref(d);
        }
        for (j = mp; j >= 1; --j) {
            i = (off - j)%m + 1;
            rho_i = rho(i);
            if (rho_i > 0) {
                beta = optm_inner(d, *Y(i))/rho_i;
                d += (alpha(i) - beta)*msk*(*S(i));
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

     It is the caller's responsibility to ensure that the bounds are
     compatible, in other words that `xmin ≤ xmax` holds.

   SEE ALSO: optm_active_variables and optm_line_search_limits.
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

func optm_active_variables(x, xmin, xmax, g)
/* DOCUMENT msk = optm_active_variables(x, xmin, xmax, g);

     Build a logical mask `msk` of the same size as `x` indicating which
     entries in `x` are not blocked by the bounds `xmin` and `xmax` when
     minimizing an objective function whose gradient is `g` at `x`.

     Empty bounds, that is `xmin = []` or `xmax = []`, are interpreted as
     unlimited (as if `xmin = -Inf` and `xmax = +Inf`).

     It is the caller's responsibility to ensure that the bounds are compatible
     and that the variables are feasible, in other words that `xmin ≤ x ≤ xmax`
     holds.

   SEE ALSO: optm_clamp and optm_line_search_limits.
 */
{
    if (is_void(xmin)) {
        if (is_void(xmax)) {
            return array(1n, dimsof(x));
        } else {
            return (x < xmax)|(g > 0);
        }
    } else if (is_void(xmax)) {
        return (x > xmin)|(g < 0);
    } else {
        return ((x > xmin)|(g < 0))&((x < xmax)|(g > 0));
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

   SEE ALSO: optm_clamp and optm_active_variables.
 */
{
    INF = OPTM_INFINITE; // for nicer code ;-)
    no_lower = is_void(xmin);
    no_upper = is_void(xmax);
    amin = INF;
    if (no_lower && no_upper) {
        // Quick return if unconstrained.
        amax = INF;
        return;
    }
    amax = 0.0;
    backward = (!is_void(dir) && dir < 0); // Move in backward direction?
    if (no_lower) {
        if (backward ? (max(d) > 0) : (min(d) < 0)) {
            amax = INF;
        }
    } else {
        // Find step sizes to reach any lower bounds.
        i = a = [];
        if (backward) {
            i = where(d > 0);
            if (is_array(i)) {
                a = x0 - xmin;
            }
        } else {
            i = where(d < 0);
            if (is_array(i)) {
                a = xmin - x0;
            }
        }
        if (!is_void(a)) {
            a = a(i)/d(i);
            amin = min(amin, min(a));
            amax = max(amax, max(a));
        }
    }
    if (no_upper) {
        // No upper bound set.
        if (amax < INF && (backward ? (min(d) < 0) : (max(d) > 0))) {
            amax = INF;
        }
    } else {
        // Find step sizes to reach any upper bounds.
        a = i = [];
        if (backward) {
            i = where(d < 0);
            if (is_array(i)) {
                a = x0 - xmax;
            }
        } else {
            i = where(d > 0);
            if (is_array(i)) {
                a = xmax - x0;
            }
        }
        if (!is_void(a)) {
            a = a(i)/d(i);
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
                delta=, epsilon=, lambda=, ftol=, gtol=, xtol=,
                blmvm=, maxiter=, maxeval=, verbose=, output=, throwerrors=)
/* DOCUMENT x = optm_vmlmb(fg, x0, [f, g, status,] lower=, upper=, mem=);

     Apply VMLMB algorithm to minimize a multi-variate differentiable objective
     function possibly under separble bound constraints.  VMLMB is a
     quasi-Newton method ("VM" is for "Variable Metric") with low memory
     requirements ("LM" is for "Limited Memory") and which can optionally take
     into account separable bound constraints (the final "B") on the variables.
     To determine efficient search directions, VMLMB approximates the Hessian
     of the objective function by a a limited memory version of the
     Broyden-Fletcher-Goldfarb-Shanno model (L-BFGS for short).  Hence VMLMB is
     well suited to solving optimization problems with a very large number of
     variables possibly with bound constraints.

     The method has two required arguments: `fg` the function to call to
     compute the objective function and its gradient and `x0` the initial
     variables (VMLMB is an iterative method).  The initial variables may be an
     array of any dimensions.

     The method returns `x` the best solution found during iterations.
     Arguments `f`, `g` and `status` are optional output variables to store the
     value and the gradient of the objective at `x` and an integer code
     indicating the reason of the termination of the algorithm (see
     `optm_reason`).

     The function `fg` shall be implemented as follows:

         func fg(x, &gx)
         {
             fx = ...; // value of the objective function at `x`
             gx = ...; // gradient of the objective function at `x`
             return fx;
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
       eigenvalues of the Hessaian of the objective function.  This setting may
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

     - Keyword `verbose` specifies whether to print information at each
       iteration.

     - Keyword `output` specifies the file stream to print information.

     - Keyword `throwerrors` (true by default), specifies whether to call
       `error` in case of errors instead or returning a `status` indicating the
       problem.  Note that early termination due to limits set on the number of
       iterations or of evaluations of the objective function are not
       considereed as an error.
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
    if (is_void(verbose)) verbose = FALSE;
    if (is_void(fmin)) fmin = NAN;
    if (is_void(delta)) delta = NAN;
    if (is_void(epsilon)) epsilon = 0.0;
    if (is_void(lambda)) lambda = NAN;
    if (is_void(blmvm)) blmvm = FALSE;
    if (is_void(throwerrors)) throwerrors = TRUE;
    if (is_scalar(ftol)) {
        fatol = -INF;
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

    // Bound constraints.  For faster code, unlimited bounds are preferentially
    // represented by empty arrays.
    if (is_array(lower) && allof(lower == -INF)) lower = [];
    if (is_array(upper) && allof(upper == +INF)) upper = [];
    bounded = (!is_void(lower) || !is_void(upper));
    if (!bounded) {
        blmvm = FALSE; // no needs to use BLMVM trick in the unconstrained case
    }

    // Other initialization.
    x = unref(x0); // initial iterate (avoiding copy)
    g = [];        // gradient
    g0 = [];       // gradient at start of line-search
    alpha = 0.0;   // step length
    amin = 0.0;    // first step length threshold
    amax = INF;    // last step length threshold
    evals = 0;     // number of calls to fg
    iters = 0;     // number of iterations
    projs = 0;     // number of projections onto the feasible set
    status = 0;    // non-zero when algorithm is about to terminate
    best_f = INF;  // best function value so far
    best_g = [];   // corresponding gradient
    best_x = [];   // corresponding variables
    freevars = []; // subset of free variables not yet known
    lbfgs = optm_new_lbfgs(mem);
    if (verbose) {
        elapsed = array(double, 3);
        timer, elapsed;
        t0 = elapsed(3);
    }
    print_now = FALSE;

    // Algorithm stage is one of:
    // 0 = initial;
    // 1 = first trial step in line-search;
    // 2 = second and subsequent trial steps in line-search;
    // 3 = line-search has converged.
    stage = 0;

    while (TRUE) {
        if (bounded && stage < 2) {
            // Make the variables feasible.
            x = optm_clamp(unref(x), lower, upper);
            projs += 1;
        }
        // Compute objective function and its gradient.
        g = [];
        f = fg(x, g);
        evals += 1;
        if (f < best_f || evals == 1) {
            // Save best solution so far.
            best_f = f;
            eq_nocopy, best_g, g;
            eq_nocopy, best_x, x;
        }
        if (stage == 1) {
            // First trial along search direction.
            d = x - x0; // effective step
            dg0 = optm_inner(d, g0);
            alpha = 1.0;
            optm_start_line_search, lnsrch, f0, dg0, alpha;
            if (lnsrch.stage != 1) {
                error, "something is wrong!";
            }
            stage = 2;
        }
        if (stage == 2) {
            // Check for line-search convergence.
            optm_iterate_line_search, lnsrch, f;
            if (lnsrch.stage == 2) {
                // Line-search has converged, `x` is the next iterate.
                stage = 3;
                iters += 1;
            } else if (lnsrch.stage == 1) {
                alpha = lnsrch.step;
            } else {
                error, "something is wrong!";
            }
        }
        if (stage == 3 || stage == 0) {
            // Initial or next iterate after convergence of line-search.
            if (bounded) {
                // Determine the subset of free variables and compute the norm
                // of the projected gradient (needed to check for convergence).
                freevars = optm_active_variables(x, lower, upper, g);
                if (noneof(freevars)) {
                    // Variables are all blocked.
                    status = OPTM_XTEST_SATISFIED;
                    gnorm = 0.0;
                } else {
                    pg = freevars*g;
                    gnorm = optm_norm2(pg);
                    if (!blmvm) {
                        // Projected gradient no longer needed, free some
                        // memory.
                        pg = [];
                    }
                }
            } else {
                // Just compute the norm of the gradient.
                gnorm = optm_norm2(g);
            }
            // Check for algorithm convergence or termination.
            if (evals == 1) {
                // Compute value for testing the convergence in the gradient.
                gtest = max(0.0, gatol, grtol*gnorm);
            }
            if (status == 0 && gnorm <= gtest) {
                // Convergence in gradient.
                status = OPTM_GTEST_SATISFIED;
            }
            if (stage == 3) {
                if (status == 0) {
                    // Check convergence in relative function reduction.
                    if (f <= fatol ||
                        abs(f - f0) <= max(0.0, frtol*max(abs(f), abs(f0)))) {
                        status = OPTM_FTEST_SATISFIED;
                    }
                }
                if (alpha != 1.0) {
                    d = x - x0; // recompute effective step
                }
                if (status == 0) {
                    // Check convergence in variables.
                    dnorm = optm_norm2(d);
                    if (dnorm <= max(0.0, xatol) ||
                        (xrtol > 0 && dnorm <= xrtol*optm_norm2(x))) {
                        status = OPTM_XTEST_SATISFIED;
                    }
                }
            }
            if (status == 0 && iters >= maxiter) {
                status = OPTM_TOO_MANY_ITERATIONS;
            }
            print_now = verbose;
        }
        if (status == 0 && evals >= maxeval) {
            status = OPTM_TOO_MANY_EVALUATIONS;
        }
        if (verbose && status != 0 && !print_now && best_f < f0) {
            // Verbose mode and abnormal termination but some progresses have
            // been made since the start of the line-search.  Restore best
            // solution so far, pretend that one more iteration has been
            // performed and manage to print information about this iteration.
            f = best_f;
            eq_nocopy, g, best_g;
            eq_nocopy, x, best_x;
            if (bounded) {
                gnorm = optm_norm2(freevars*g);
            }
            iters += 1;
            print_now = verbose;
        }
        if (print_now) {
            timer, elapsed;
            t = (elapsed(3) - t0)*1E3; // elapsed milliseconds
            if (iters < 1) {
                write, output, format="%s%s\n%s%s\n",
                    "# Iter.   Time (ms)   Eval.   Proj. ",
                    "       Obj. Func.           Grad.       Step",
                    "# ----------------------------------",
                    "-----------------------------------------------";
            }
            write, output, format="%7d %11.3f %7d %7d %23.15e %11.3e %11.3e\n",
                iters, t, evals, projs, f, gnorm, alpha;
            print_now = !print_now;
        }
        if (status != 0) {
            // Algorithm stops here.
            break;
        }
        if (stage == 3) {
            // Line-search has converged, L-BFGS approximation can be updated.
            // FIXME: if alpha = 1, then d = x - x0;
            if (blmvm) {
                optm_update_lbfgs, lbfgs, x - x0, pg - pg0;
            } else {
                optm_update_lbfgs, lbfgs, x - x0, g - g0;
            }
        }
        if (stage == 3 || stage == 0) {
            // Save iterate.
            f0 = f;
            eq_nocopy, g0, g;
            eq_nocopy, x0, x;
            if (blmvm) {
                pg0 = pg; // FIXME:
            }
            // Determine a new search direction `d`.  Parameter `dir` is set to:
            //   0 if `d` is not a search direction,
            //   1 if `d` is unscaled steepest descent,
            //   2 if `d` is scaled sufficient descent.
            dir = 0;
            // Use L-BFGS approximation to compute a search direction and
            // check that it is an acceptable descent direction.
            local scaled;
            d = optm_apply_lbfgs(lbfgs, -g, scaled, freevars);
            if (!scaled) {
                // No exploitable curvature information, `d` is the unscaled
                // steepest feasible direction.
                dir = 1;
            } else {
                // Some valid (s,y) pairs were available to apply the
                // L-BFGS approximation.
                dir = 2;
                dg = optm_inner(d, g);
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
                    dnorm = optm_norm2(d);
                    if (dg > -epsilon*dnorm*gnorm) {
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
                dir = 1; // scaling needed
            }
            // Determine the length `alpha` of the initial step along `d`.
            if (dir == 2) {
                alpha = 1.0;
            } else {
                // Find a suitable step size along the steepest feasible
                // descent direction `d`.  Note that `gnorm`, the Euclidean
                // norm of the (projected) gradient, is also that of `d`.
                alpha = optm_steepest_descent_step(x, gnorm, f, fmin, delta, lambda);
            }
            stage = 1; // first trial along search direction
            if (bounded) {
                // Safeguard the step to avoid searching in a region where
                // all bounds are overreached.
                local amin, amax;
                optm_line_search_limits, amin, amax, x0, lower, upper, d, alpha;
                alpha = min(alpha, amax);
            }
        }
        // Compute next iterate.
        if (alpha == 1) {
            x = x0 + d;
        } else {
            x = x0 + alpha*d;
        }
    }

    // Restore best solution so far and return solution (and status).
    if (best_f < f) {
        f = best_f;
        eq_nocopy, g, best_g;
        eq_nocopy, x, best_x;
    }
    if (verbose) {
        write, output, format="# Termination: %s\n", optm_reason(status);
    }
    return x;
}


//-----------------------------------------------------------------------------
// UTILITIES AND ALGEBRA

func optm_inner(x, y)
/* DOCUMENT optm_inner(x, y);

     Yields the scalar product of X and Y that is sum(X*Y) but computed as
     efficiently as possible.

   SEE ALSO: optm_norm2.
 */
{
    return sum(x*y);
}

func optm_norm1(x)
/* DOCUMENT optm_norm2(x);

     Yields the L1 norm of X that is sum(abs(X)) but computed as efficiently as
     possible.

   SEE ALSO: optm_norm2, optm_norminf.
 */
{
    return (is_scalar(x) ? abs(x) : sum(abs(x)));
}

func optm_norm2(x)
/* DOCUMENT optm_norm2(x);

     Yields the Euclidean (L2) norm of X that is sqrt(sum(X^2)) but computed as
     efficiently as possible.

   SEE ALSO: optm_inner, optm_norm1, optm_norminf.
 */
{
    return (is_scalar(x) ? abs(x) : sqrt(sum(x*x)));
}

func optm_norminf(x)
/* DOCUMENT optm_norminf(x);

     Yields the infinite norm of X that is max(abs(X)) but computed as
     efficiently as possible.

   SEE ALSO: optm_norm1, optm_norm2.
 */
{
    return (is_scalar(x) ? abs(x) : max(-min(x), max(x)));
}

func optm_scale(alpha, x)
/* DOCUMENT optm_scale(alpha, x);

     Yields ALPHA*X computed as efficiently as possible and taking care of
     preserving the floting-point type of X.
 */
{
    T = structof(x);
    if (T == double || T == float) {
        alpha = T(alpha);
    }
    return alpha*x;
}

func optm_tolerance(x, atol, rtol)
/* DOCUMENT tol = optm_tolerance(x, atol, rtol);

     Given absolute and relative tolerances ATOL and RTOL, yields:

         max(0, atol, rtol*abs(x))    // if X is a scalar
         max(0, atol, rtol*norm(x))   // if X is an array

    where norm(X) is the Euclidean norm of X as computed by optm_norm2 (which
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
