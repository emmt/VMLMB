# -*- coding: utf-8 -*-

"""
Pure Python implementations of the linear conjugate gradients (for solving
uncontrained linear optimization problems) and VMLMB (for solving non-linear of
bound constrained optimization problems).  VMLMB is a quasi-Newton method ("VM"
is for "Variable Metric") with low memory requirements ("LM" is for "Limited
Memory") and which can optionally take into account separable bound constraints
(the final "B") on the variables.

This file is part of the VMLMB software which is licensed under the "Expat" MIT
license, <https://github.com/emmt/VMLMB>.

Copyright (C) 2002-2022, Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>
"""

# Insure compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

import sys          # for sys.stdout
import numpy as _np # to deal with numerical arrays
import time         # to measure time
import math         # for isnan() function
from math import sqrt, isnan

#------------------------------------------------------------------------------
# PREAMBLE

FLOATS = (_np.double, _np.float, _np.longdouble)

DEBUG = True
NOT_POSITIVE_DEFINITE = -1
TOO_MANY_EVALUATIONS  =  1
TOO_MANY_ITERATIONS   =  2
FTEST_SATISFIED       =  3
XTEST_SATISFIED       =  4
GTEST_SATISFIED       =  5

def reason(status):
    """
    Usage:

        str = optm.reason(status)

    Get a textual description of the meaning of the termination code `status`.
    Known codes are:

    - `optm.NOT_POSITIVE_DEFINITE`: the Hessian matrix (or its inverse) of a
      problem is found to be not positive definite;

    - `optm.TOO_MANY_ITERATIONS`: the maximum number of iterations has been
      reached;

    - `optm.FTEST_SATISFIED`: the algorithm terminated because of a sufficient
      reduction of the objective function;

    - `optm.GTEST_SATISFIED`: the algorithm terminated because the norm of the
      gradient of the objective function is small enough;

    - `optm.XTEST_SATISFIED``: the algorithm terminated because the variation
      of variables is small enough.

    See also: `optm.vmlmb`, `optm.conjgrad`.
    """
    if status == NOT_POSITIVE_DEFINITE:
        return "LHS operator is not positive definite"
    elif status == TOO_MANY_EVALUATIONS:
        return "too many evaluations"
    elif status == TOO_MANY_ITERATIONS:
        return "too many iterations"
    elif status ==  FTEST_SATISFIED:
        return "function reduction test satisfied"
    elif status == GTEST_SATISFIED:
        return "(projected) gradient test satisfied"
    elif status == XTEST_SATISFIED:
        return "variables change test satisfied"
    else:
        return "unknown status code"

def identity(x):
    """This function just returns its argument."""
    return x

#------------------------------------------------------------------------------
# LINEAR CONJUGATE GRADIENT

def conjgrad_printer(output, itr, t, x, phi, r, z, rho):
    """Default printer function for `optm.conjgrad`."""
    if z is r:
        # No preconditioner is used.
        if itr == 0:
            print("# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖", file=output)
            print("# --------------------------------------------",
                  file=output)
        print(f"{itr:7d} {t*1e3:11.3f} {phi:12.4e} {sqrt(rho):12.4e}",
              file=output)
    else:
        # A preconditioner is used.
        if itr == 0:
            print("# Iter.   Time (ms)     Δf(x)       ‖∇f(x)‖     ‖∇f(x)‖_M",
                  file=output)
            print("# ---------------------------------------------------------",
                  file=output)
        print(f"{itr:7d} {t*1e3:11.3f} {phi:12.4e} {norm2(r):12.4e} {sqrt(rho):12.4e}",
              file=output)

def conjgrad(A, b, x=None, *, precond=identity, maxiter=None, restart=None,
             verb=0, printer=conjgrad_printer, output=sys.stdout,
             ftol=1.0e-8, gtol=1.0e-5, xtol=1.0e-6):
    """
    Usage:

        (x, status) = conjgrad(A, b, x0=None)

    Run the (preconditioned) linear conjugate gradient algorithm to solve the
    system of equations `A⋅x = b` in `x`.

    Argument `A` implements the left-hand-side (LHS) "matrix" of the equations.
    It is called as `A(x)` to compute the result of `A⋅x`.  Note that, as `A`
    and the preconditioner `M` must be symmetric, it may be faster to apply
    their adjoint.

    Argument `b` is the right-hand-side (RHS) "vector" of the equations.

    Optional argument `x0` provides the initial solution.  If `x0` is
    unspecified, `x` is initially an array of zeros.

    Provided `A` be positive definite, the solution `x` of the equations `A⋅x =
    b` is unique and is also the minimum of the following convex quadratic
    objective function:

        f(x) = (1/2)⋅x'⋅A⋅x - b'⋅x + ϵ

    where `ϵ` is an arbitrary constant.  The gradient of this objective
    function is:

        ∇f(x) = A⋅x - b

    hence solving `A⋅x = b` for `x` yields the minimum of `f(x)`.  The
    variations of `f(x)` between successive iterations, the norm of the
    gradient `∇f(x)` or the norm of the variation of variables `x` may be used
    to decide the convergence of the algorithm (see keywords `ftol`, `gtol` and
    `xtol`).

    Algorithm parameters can be specified by the following keywords:

    - Keyword `precond` is to specify a preconditioner `M`.  It may be a
      function name or handle and is called as `M(x)` to compute the result of
      `M⋅x`.  By default, the un-preconditioned version of the algorithm is
      run.

    - Keyword `maxiter` is to specify the maximum number of iterations to
      perform which is `2⋅b.size + 1` by default.

    - Keyword `restart` is to specify the number of consecutive iterations
      before restarting the conjugate gradient recurrence.  Restarting the
      algorithm is to cope with the accumulation of rounding errors.  By
      default, `restart = min(50,b.size+1)`.  Set `restart` to a value less or
      equal zero or greater than `maxiter` if you do not want that any restarts
      ever occur.

    - Keyword `verb`, if positive, specifies to print information every `verb`
      iterations.  Nothing is printed if `verb ≤ 0`.  By default, `verb = 0`.

    - Optional argument `printer` specifies a function to call to print
      information every `verb` iterations.  This function is called as:

          printer(output, itr, t, x, phi, r, z, rho)

      with `output` the output stream specified by keyword `output`, `itr` the
      iteration number, `t` the elapsed time in seconds, `phi` the reduction of
      the objective function, `r` the residuals, `z` the preconditioned
      residuals, and `rho` the squared Euclidean norm of `z`.  You can use `z
      is r` to verify whether a preconditioner is used or not (`z` is different
      from `r` if this is the case).

    - Keyword `output` specifies the file stream to print information,
      `sys.stdout` by default.

    - Keywords `ftol`, `gtol` and `xtol` specify tolerances for deciding the
      convergence of the algorithm.  In what follows, `x_{k}`,
      `f_{k}=f(x_{k})`, and `∇f_{k}=∇f(x_{k})` denotes the variables, the
      objective function and its gradient after `k` iterations ot the algorithm
      (`x_{0} = x0` the initial estimate).

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

      Convergence in the variables occurs at iteration `k ≥ 1` if the following
      condition holds:

          ‖x_{k} - x_{k-1}‖ ≤ max(0, xatol, xrtol⋅‖x_{k}‖)


      where `xatol` and `xrtol` are absolute and relative tolerances specified
      by `xtol` which can be `xtol=[fatol,frtol]` or `xtol=xrtol` and assume
      that `xatol=0`.  The default is `xtol=1e-6`.

      In the conjugate gradient algorithm, the objective function is always
      reduced at each iteration, but be aware that the gradient and the change
      of variables norms are not always reduced at each iteration.


    ## Returned Status

    The function returns the solution `x` and `status` which indicates the
    reason of the algorithm termination, one of the following codes:

    - `status = NOT_POSITIVE_DEFINITE` if the left-hand-side matrix `A`
      is found to be not positive definite;

    - `status = TOO_MANY_ITERATIONS` if the maximum number of iterations has
      been reached;

    - `status = FTEST_SATISFIED` if convergence occurred because the function
      reduction satisfies the criterion specified by `ftol`;

    - `status = GTEST_SATISFIED` if convergence occurred because the gradient
      norm satisfies the criterion specified by `gtol`;

    - `status = XTEST_SATISFIED` if convergence occurred because the norm of
      the variation of variables satisfies the criterion specified by `xtol`.

    The function `optm_reason` may be called to have a textual description of
    the meaning of the returned status.

    See also: optm.reason, optm.inner.

    """

    # Get options.
    if maxiter is None:
        maxiter = 2*b.size + 1
    if restart is None:
        restart = min(50, b.size + 1)
    (fatol, frtol) = get_tolerances(ftol)
    (gatol, grtol) = get_tolerances(gtol)
    (xatol, xrtol) = get_tolerances(xtol)

    # Initial solution.
    if x is None:
        x = _np.zeros(b.shape, b.dtype)
        x_is_zero = True
    else:
        x_is_zero = (norm2(x) == 0.0)

    # Define local variables.
    mesg = None   # exit message
    #r = ...      # residuals `r = b - A⋅x`
    #z = ...      # preconditioned residuals `z = M⋅r`
    #p = ...      # search direction `p = z + β⋅p`
    #q = ...      # `q = A⋅p`
    #oldrho = ... # previous value of `rho`
    rho = 0.0     # `rho = ⟨r,z⟩`
    phi = 0.0     # function reduction
    phimax = 0.0  # maximum function reduction
    xtest = (xatol > 0.0 or xrtol > 0.0)
    if verb > 0:
        t0 = elapsed_time(0.0)

    # Conjugate gradient iterations.
    k = 0
    while True:
        # Is this the initial or a restarted iteration?
        restarting = (k == 0 or restart > 0 and (k%restart) == 0)

        # Compute residuals and their squared norm.
        if restarting:
            # Compute residuals.
            if x_is_zero:
                # Spare applying A since x = 0.
                r = b
                x_is_zero = False
            else:
                # Compute `r = b - A*x`.
                r = b - A(x)
        else:
            # Update residuals: `r -= alpha*q`.
            r = update(r, -alpha, q)

        # Apply preconditioner `z = M⋅r`.
        z = precond(r)

        oldrho = rho
        rho = inner(r, z); # rho = ‖r‖_M^2
        if k == 0:
            # Pre-compute the minimal Mahalanobis norm of the gradient for
            # convergence.  The Mahalanobis norm of the gradient is equal
            # to `2⋅sqrt(rho)`.
            gtest = max(0.0, gatol, 2.0*grtol*sqrt(rho))

        if verb > 0 and (k % verb) == 0:
            printer(output, k, elapsed_time(t0), x, phi, r, z, rho)

        if 2.0*sqrt(rho) <= gtest:
            # Normal convergence in the gradient norm.
            status = GTEST_SATISFIED
            mesg = "Convergence in the gradient norm."
            break

        if k >= maxiter:
            status = TOO_MANY_ITERATIONS
            mesg = "Too many iteration(s)."
            break

        # Compute search direction `p`.
        if restarting:
            # Restarting or first iteration.
            p = z
        else:
            # Apply recurrence.
            beta = rho/oldrho
            p = z + promote_multiplier(beta, p)*p

        # Compute optimal step size `alpha` along search direction `p`.
        q = A(p) # q = A*p
        gamma = inner(p, q)
        if not (gamma > 0.0):
            status = NOT_POSITIVE_DEFINITE
            mesg = "Operator is not positive definite."
            break
        alpha = rho/gamma

        # Update variables and check for convergence.
        # FIXME: x += promote_multiplier(alpha, p)*p
        x = update(x, alpha, p) # x += alpha*p
        phi = alpha*rho/2.0     # phi = f(x_{k}) - f(x_{k+1}) ≥ 0
        phimax = max(phi, phimax)
        if phi <= tolerance(phimax, fatol, frtol):
            # Normal convergence in the function reduction.
            status = FTEST_SATISFIED
            mesg = "Convergence in the function reduction."
            break

        if xtest and alpha*norm2(p) <= tolerance(x, xatol, xrtol):
            # Normal convergence in the variables.
            status = XTEST_SATISFIED
            mesg = "Convergence in the variables."
            break
        k += 1

    if verb > 0:
        # Print last iteration, if not yet done, and termination message.
        if (k % verb) != 0:
            printer(output, k, elapsed_time(t0), x, phi, r, z, rho)
        if mesg and printer is conjgrad_printer:
            print(f"# {mesg}", file=output)
    return (x, status)

#------------------------------------------------------------------------------
# LINE SEARCH

class LineSearch:
    """Calling:

        lnsrch = LineSearch(ftol=1e-4, smin=0.2, smax=None)

    creates a new line-search instance.  Optional argument `ftol` can be used to
    specify the function decrease tolerance.  A step `alpha` is considered as
    successful if the following condition (known as Armijo's condition) holds:

        f(x0 + alpha*d) ≤ f(x0) + ftol*df(x0)*alpha

    where `f(x)` is the objective function at `x`, `x0` denotes the variables
    at the start of the line-search, `d` is the search direction, and `df(x0) =
    d'⋅∇f(x0)` is the directional derivative of the objective function at `x0`.
    The value of `ftol` must be in the range `(0,0.5]`, the default value is
    `ftol = 1e-4`.

    Optional arguments `smin` and `smax` can be used to specify relative
    bounds for safeguarding the step length.  When a step `alpha` is
    unsuccessful, a new backtracking step is computed based on a parabolic
    interpolation of the objective function along the search direction.  The
    new step writes:

        new_alpha = gamma*alpha

    with `gamma` safeguarded in the range `[smin,smax]`.  The following
    constraints must hold: `0 < smin ≤ smax < 1`.  Taking `smin = smax = 0.5`
    emulates the usual Armijo's method.  Default values are `smin = 0.2` and
    `smax = 1/(2 - 2*ftol)`.

    Note that when Armijo's condition does not hold, the quadratic
    interpolation yields `gamma < 1/(2 - 2*ftol)`.  Hence, taking an upper
    bound `smax > 1/(2 - 2*ftol)` has no effects while taking a lower bound
    `smin ≥ 1/(2 - 2*ftol)` yields a safeguarded `gamma` always equal to
    `smin`.  Therefore, to benefit from quadratic interpolation, one should
    choose `smin < smax ≤ 1/(2 - 2*ftol)`.

    A typical usage is:

        lnsrch = optm.LineSearch()
        x = ...;   # initial solution
        fx = f(x); # objective function
        while True:
            gx = grad_f(x)
            if optm.norm2(gx) <= gtol:
                break # a solution has been found
            d = next_search_direction(...)
            stp = guess_step_size(...)
            f0 = fx
            x0 = x
            df0 = optm.inner(d, gx)
            lnsrch.start(f0, df0, stp)
            while not lnsrch.converged()
                x = x0 + lnsrch.step()*d
                fx = f(x)
                lnsrch.iterate(fx)

    See also: `optm.vmlmb`, `optm.LineSearch.start`, and
    `optm.LineSearch.iterate`.
    """
    def __init__(self, ftol=1e-4, smin=0.2, smax=None):
        if ftol <= 0 or ftol > 0.5:
            raise ValueError("`ftol` must be in the range (0,0.5]")
        if smin <= 0:
            raise ValueError("`smin` must be strictly greater than 0")
        if smax is None:
            smax = max(smin, 1.0/(2.0 + 2.0*ftol))
        if smax >= 1:
            raise ValueError("`smax` must be strictly less than 1")
        if smin > smax:
            raise ValueError("`smin` must be less or equal `smax`")
        self._stage = 0
        self._ftol = ftol
        self._smin = smin
        self._smax = smax
        self._step = 0.0
        self._finit = None
        self._ginit = None

    def start(self, f0, df0, stp):
        """
        Calling:

            lnsrch.start(fx, f0, df0, stp)

        starts a new line-search for line-search instance `lnsrch` with
        arguments: `f0` the objective function at `x0` the variables at the
        start of the line-search, `df0` the directional derivative of the
        objective function at `x0` and `stp > 0` a guess for the first step to
        try.

        See also: `optm.LineSearch`, `optm.LineSearch.iterate`.
        """
        if df0 >= 0:
            raise AssertionError("not a descent direction")
        if stp <= 0:
            raise ValueError("first step to try must be strictly greater than 0")
        self._finit = f0
        self._ginit = df0
        self._step  = stp
        self._stage = 1

    def iterate(self, fx):
        """
        Call:

            lnsrch.iterate(fx)

        to pursue the line-search started for line-search instance `lnsrch`.
        Argument `fx = f(x)` with `x = x0 + lnsrch.step()*d` is the objective
        function at the variables at the current position on the line-search.

        Then, if `lnsrch.converged()` is true, the step length `lnsrch.step()`
        is left unchanged and `x` is the new iterate of the optimization
        method.  Otherwise, line-search is in progress, the next step to try is
        `lnsrch.step()`, and `lnsrch.iterate` shall be called with the function
        value at the new iterate.

        See also: `optm.LineSearch`, `optm.LineSearch.start`.
        """
        finit = self._finit
        ginit = self._ginit
        step  = self._step
        ftol  = self._ftol
        if fx <= finit + ftol*(ginit*step):
            # Line-search has converged.
            self._stage = 2
        else:
            # Line-search has not converged.
            smin = self._smin
            smax = self._smax
            if smin < smax:
                # Compute a safeguarded parabolic interpolation step.
                q = -ginit*step
                r = 2*((fx - finit) + q)
                if q <= smin*r:
                    gamma = smin
                elif q >= smax*r:
                    gamma = smax
                else:
                    gamma = q/r

            elif smin == smax:
                gamma = smin
            else:
                raise AssertionError("invalid fields `smin` and `smax`")
            self._step = gamma*step
            self._stage = 1

    def converged(self):
        """Yield whether line-search has converged."""
        return self._stage == 2

    def step(self):
        """Yield next line-search step to take."""
        return self._step

def steepest_descent_step(x, d, fx, *, fmin=None, xtiny=None, f2nd=None):
    """Calling:

        alpha = optm.steepest_descent_step(x, d, fx, fmin, xtiny, f2nd)

    yields the length `alpha` of the first trial step along the steepest
    descent direction.  Arguments are:

    - `x` the current variables (or their Euclidean norm).

    - `d` the search direction `d` (or its Euclidean norm) at `x`.  This
      direction shall be the gradient of the objective function (or the
      projected gradient for a constrained problem) at `x` up to a change of
      sign.

    - `fx` the value of the objective function at `x`.

    - `fmin` an estimate of the minimal value of the objective function.

    - `xtiny` a small step size relative to the norm of the variables.

    - `f2nd` an estimate of the magnitude of the eigenvalues of the Hessian
      (2nd derivatives) of the objective function.

    See also: `optm.LineSearch`.

    """
    Inf = _np.inf
    if not fmin is None and -Inf < fmin < fx:
        # For a quadratic objective function, the minimum is such that:
        #
        #     fmin = f(x) - (1/2)*alpha*d'*∇f(x)
        #
        # with `alpha` the optimal step.  Hence:
        #
        #     alpha = 2*(f(x) - fmin)/(d'*∇f(x)) = 2*(f(x) - fmin)/‖d‖²
        #
        # is an estimate of the step size along `d` if it is plus or minus the
        # (projected) gradient.
        dnorm = norm2(d)
        alpha = 2*(fx - fmin)/(dnorm*dnorm)
        if 0 < alpha < Inf:
            return alpha
    else:
        dnorm = None # Euclidean norm of `d` not yet computed.

    if not xtiny is None and 0 < xtiny < 1:
        # Use the specified small relative step size.
        if dnorm is None:
            dnorm = norm2(d)
        xnorm = norm2(x)
        alpha = xtiny*xnorm/dnorm
        if 0 < alpha < Inf:
            return alpha

    if not f2nd is None and 0 < f2nd < Inf:
        # Use typical Hessian eigenvalue if suitable.
        alpha = 1.0/f2nd
        if 0 < alpha < Inf:
            return alpha

    # Eventually use 1/‖d‖.
    if dnorm is None:
        dnorm = norm2(d)
    alpha = 1.0/dnorm
    return alpha

#------------------------------------------------------------------------------
# L-BFGS APPROXIMATION

class LBFGS:
    """
    Implementation of a limited memory version of the
    Broyden-Fletcher-Goldfarb-Shanno approximation of the (inverse) Hessian of
    a differentiable function.
    """

    def __init__(self, m):
        """
        Calling:

            lbfgs = optm.LBFGS(m)

        yields a new L-BFGS instance for storing up to `m` previous steps.

        See also: `optm.vmlmb`, `optm.LBFGS.update`, `optm.LBFGS.apply`,
        `optm.LBFGS.reset`.
        """
        if not type(m) is int or m < 0:
            raise ValueError("invalid number of previous step to memorize")
        self.m = m
        self.reset()

    def reset(self):
        """
        Reset the L-BFGS model stored in the instance, thus forgetting any
        memorized information.  This also frees most memory allocated by the
        instance.

        See also: `optm.vmlmb`, `optm.LBFGS`.
        """
        m = self.m
        self.mp    = 0 # number of meorized steps
        self.itr   = 0 # number of updates
        self.gamma = 0.0
        self.S     = [None for i in range(m)]
        self.Y     = [None for i in range(m)]
        self.rho   = _np.full(m, 0.0, _np.double)
        self.alpha = _np.full(m, 0.0, _np.double)

    def update(self, s, y):
        """
        Calling:

            flg = lbfgs.update(s, y)

        updates information stored by L-BFGS instance `lbfgs`.  Arguments `s`
        and `y` are the change in the variables and in the gradient of the
        objective function for the last iterate.  The returned value is a
        boolean indicating whether `s` and `y` were suitable to update an
        L-BFGS approximation that be positive definite.

        Even if `lbfgs.m = 0`, the value of the optimal gradient scaling
        `lbfgs.gamma` is updated if possible (i.e., when true is returned).

        See also: `optm.vmlmb`, `optm.LBFGS`, `optm.inner`.
        """
        sty = inner(s, y)
        if sty <= 0:
            return False
        self.gamma = sty/inner(y, y)
        m = self.m
        if m >= 1:
            k = self.itr % m
            self.S[k] = s
            self.Y[k] = y
            self.rho[k] = sty
        self.itr += 1
        self.mp = min(self.mp + 1, m)
        return True

    def apply(self, d, freevars=None):
        """
        Calling:

            (d, scaled) = lbfgs.apply(g)

        or

            (d, scaled) = lbfgs.apply(g, freevars)

        apply L-BFGS approximation stored by `lbfgs` instance of the inverse
        Hessian to the "vector" `g`.

        Optional argument `freevars` is to restrict the L-BFGS approximation to
        the sub-space spanned by the "free variables" not blocked by the
        constraints.  If specified and not empty, `freevars` shall have the
        size as `d` and shall be equal to zero where variables are blocked and
        to one elsewhere.

        On return, output variable `scaled` indicates whether any curvature
        information was taken into account.  If `scaled` is false, it means
        that the result `d` is identical to `g` except that `d(i)=0` if the
        `i`-th variable is blocked according to `freevars`.

        See also: `optm.vmlmb`, `optm.LBFGS`, `optm.inner`.
        """

        # Determine the number of variables and of free variables.
        if freevars is None or _np.amin(freevars) != 0:
            # All variables are free.
            regular = True
        else:
            # Convert `freevars` in an array of weights of suitable type.
            regular = False
            T = d.dtype
            if not T in FLOATS:
                T = _np.double
            if freevars.dtype == T:
                wgt = freevars
            else:
                wgt = freevars.astype(T)

        # Apply the 2-loop L-BFGS recursion algorithm by Matthies & Strang.
        m = self.m
        mp = self.mp
        if mp >= 1:
            # Aliases for readability.
            S = self.S
            Y = self.Y
            alpha = self.alpha
            rho = self.rho

        inds = [i%m for i in range(self.itr + m - mp, self.itr + m)] # FIXME:

        if regular:
            # Apply the regular L-BFGS recursion.
            gamma = self.gamma
            for i in reversed(inds):
                alpha_i = inner(d, S[i])/rho[i]
                d = update(d, -alpha_i, Y[i]) # d -= alpha_i*Y[i]
                alpha[i] = alpha_i

            if gamma > 0 and gamma != 1:
                d = scale(d, gamma)

            for i in inds:
                beta = inner(d, Y[i])/rho[i]
                d = update(d, alpha[i] - beta, S[i])

        else:
            # L-BFGS recursion on a subset of free variables specified by a
            # selection of indices.
            gamma = 0.0
            d *= wgt # restrict argument to the subset of free variables
            for i in reversed(inds):
                s_i = S[i]
                y_i = wgt*Y[i]
                rho_i = inner(s_i, y_i)
                if rho_i > 0:
                    if gamma <= 0.0:
                        gamma = rho_i/inner(y_i, y_i)
                    alpha_i = inner(d, s_i)/rho_i
                    d = update(d, -alpha_i, y_i)
                    alpha[i] = alpha_i
                    rho[i] = rho_i
                y_i = [] # free memory

            if gamma > 0 and gamma != 1:
                d = scale(d, gamma)

            for i in inds:
                rho_i = rho[i]
                if rho_i > 0:
                    beta = inner(d, Y[i])/rho_i
                    d = update(d, alpha[i] - beta, wgt*S[i])

        return (d, gamma > 0)

#------------------------------------------------------------------------------
# SEPARABLE BOUND CONSTRAINTS

def clamp(x, xmin, xmax):
    """
    Restrict `x` to the range `[xmin,xmax]` element-wise.  Undefined bounds,
    that is `xmin = None` or `xmax = None`, are interpreted as unlimited.

    If both bounds are specified, it is the caller's responsibility to ensure
    that the bounds are compatible, in other words that `xmin ≤ xmax` holds.

    See also: `optm.unblocked_variables` and `optm.line_search_limits`.
    """
    if xmin is None:
        if xmax is None:
            return x
        else:
            return _np.minimum(x, xmax)
    else:
        if xmax is None:
            return _np.maximum(x, xmin)
        else:
            return _np.clip(x, xmin, xmax)

def unblocked_variables(x, xmin, xmax, g):
    """
    Build a boolean mask of the same shape as `x` and `g` indicating which
    entries in `x` are not blocked by the bounds `xmin` and `xmax` when
    minimizing an objective function whose gradient is `g` at `x`.  In other
    words, the mask is false everywhere Karush-Kuhn-Tucker (KKT) conditions are
    satisfied.

    Undefined bounds, that is `xmin = None` or `xmax = None`, are interpreted
    as unlimited.

    It is the caller's responsibility to ensure that the bounds are compatible
    and that the variables are feasible, in other words that `xmin ≤ x ≤ xmax`
    holds element-wise if both bounds are specified.

    See also: `optm.clamp` and `optm.line_search_limits`.
    """
    zero = _np.zeros(g.shape, g.dtype)
    if xmin is None:
        if xmax is None:
            return g != zero
        else:
            return (g > zero)|((g < zero)&(x < xmax))
    else:
        if xmax is None:
            return ((g > zero)&(x > xmin))|(g < zero)
        else:
            return ((g > zero)&(x > xmin))|((g < zero)&(x < xmax))

def line_search_limits(x0, xmin, xmax, pm, d):
    """
    Usage:

        optm.line_search_limits(x0, xmin, xmax, pm, d) -> (amin, amax)

    Determine the limits `amin` and `amax` for the step length `alpha` in a
    line-search where iterates `x` are given by:

        x = proj(x0 ± alpha*d)

    where `proj(x)` denotes the orthogonal projection on the convex set defined
    by separable lower and upper bounds `xmin` and `xmax` (unless unspecified)
    and where `±` is `-` if `pm` is negative and `+` otherwise.

    On return, output value `amin` is the largest nonnegative step length such
    that for any `alpha ≤ amin`:

        proj(x0 ± alpha*d) = x0 ± alpha*d

    On return, output value `amax` is the least nonnegative step length such
    that for any `alpha ≥ amax`:

        proj(x0 ± alpha*d) = proj(x0 ± amax*d)

    In other words, no bounds are overcome if `0 ≤ alpha ≤ amin` and the
    projected variables are all the same for any `alpha` such that `alpha ≥
    amax ≥ 0`.

    Restrictions: `x0` must be feasible and must have the same size as `d`;
    this is not verified for efficiency reasons.

    See also: `optm.clamp` and `optm.unblocked_variables`.
    """
    Inf = _np.inf
    unbounded_below = xmin is None
    unbounded_above = xmax is None
    if unbounded_below and unbounded_above:
        # Quick return if unconstrained.
        return (Inf, Inf)

    # Compute `amin` and `amax`.
    backward = pm < 0
    z = _np.zeros(d.shape, d.dtype)
    amin = Inf
    amax = 0.0
    if unbounded_below:
        if backward:
            if _np.amax(d) > 0:
                amax = Inf
        else:
            if _np.amin(d) < 0:
                amax = Inf
    else:
        # Find positive step sizes to reach any lower bounds.
        a = None
        if backward:
            i = d > z
            if _np.any(i):
                a = (x0 - xmin)[i]/d[i]
        else:
            i = d < z
            if _np.any(i):
                a = (xmin - x0)[i]/d[i]
        if not a is None:
            amin = min(amin, _np.amin(a))
            amax = max(amax, _np.amax(a))
    if unbounded_above:
        # No upper bound set.
        if amax < Inf:
            if backward:
                if _np.amin(d) < 0:
                    amax = Inf
            else:
                if _np.max(d) > 0:
                    amax = Inf
    else:
        # Find positive step sizes to reach any upper bounds.
        a = None
        if backward:
            i = d < z
            if _np.any(i):
                a = (x0 - xmax)[i]/d[i]
        else:
            i = d > z
            if _np.any(i):
                a = (xmax - x0)[i]/d[i]
        if not a is None:
            amin = min(amin, _np.amin(a))
            if amax < Inf:
                amax = max(amax, _np.amax(a))
    return (amin, amax)

#-----------------------------------------------------------------------------
# VMLMB ALGORITHM

def vmlmb_printer(output, iters, evals, rejects,
                  t, x, fx, gx, pgnorm, alpha, fg):
    """Default printer for `optm.vmlmb`."""
    if iters < 1:
        print("# Iter.   Time (ms)    Eval. Reject.       Obj. Func.           Grad.       Step", file=output)
        print("# ---------------------------------------------------------------------------------", file=output)
    print(f"{iters:7d} {t*1e3:11.3f} {evals:7d} {rejects:7d} {fx:23.15e} {pgnorm:11.3e} {alpha:11.3e}", file=output)

def vmlmb(fg, x0, *, lower=None, upper=None, mem=5, blmvm=False,
          lnsrch=LineSearch(), fmin=None, xtiny=None, f2nd=None,
          epsilon=0.0, ftol=1.0e-8, gtol=1.0e-5, xtol=1.0e-6,
          maxiter=_np.inf, maxeval=_np.inf, verb=0, throwerrors=True,
          printer=vmlmb_printer, observer=None, output=sys.stdout):
    """
    Usage:

        (x, fx, gx, status) = optm.vmlmb(fg, x0, lower=, upper=, mem=, ...)

    applies the VMLMB algorithm to minimize a multi-variate differentiable
    objective function possibly under separable bound constraints.  VMLMB is a
    quasi-Newton method ("VM" is for "Variable Metric") with low memory
    requirements ("LM" is for "Limited Memory") and which can optionally take
    into account separable bound constraints (the final "B") on the variables.

    To determine efficient search directions, VMLMB approximates the Hessian of
    the objective function by a limited memory version of the model assumed in
    Broyden-Fletcher-Goldfarb-Shanno algorithm (called L-BFGS for short).
    Hence VMLMB is well suited to solving optimization problems with a very
    large number of variables possibly with bound constraints.

    The method has two required arguments: `fg`, the function to call to
    compute the objective function and its gradient, and `x0`, the initial
    variables (VMLMB is an iterative method).  The initial variables may be an
    array of any dimensions.

    Argument `fg` is a callable object which takes the variables as argument
    and yields the corresponding value and gradient of the objective function.
    It is typically implemented as follows:

        def fg(x):
            fx = ...; # value of the objective function at `x`
            gx = ...; # gradient of the objective function at `x`
            return fx, gx

    The algorithm returns `x` the best solution found during iterations, the
    corresponding value `fx` and gradient `gx` of the objective function, and a
    code `status` indicating the reason of the termination of the algorithm
    (see `optm.reason`).

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
      specified by `xtol` which can be `xtol=[fatol,frtol]` or `xtol=xrtol` and
      assume that `xatol=0`.  The default is `xtol=1e-6`.

    - Keywords `maxiter` and `maxeval` are to specify a maximum number of
      algorithm iterations or or evaluations of the objective function
      implemented by `fg`.  By default, these are unlimited.

    - Keyword `lnsrch` is to specify line-search settings different than the
      default (see `optm.LineSearch`).

    - Keyword `fmin` is to specify an estimation of the minimum possible value
      of the objective function.  This setting may be used to determine the
      step length along the steepest descent.

    - Keyword `xtiny` specifies a small size relative to the variables.  This
      setting may be used to determine the step length along the steepest
      descent.

    - Keyword `f2nd` specifies an estimate of the magnitude of the
      eigenvalues of the Hessian of the objective function.  This setting may
      be used to determine the step length along the steepest descent.

    - Keyword `epsilon` specifies a threshold for a sufficient descent
      condition.  If `epsilon > 0`, then a search direction `d` computed by the
      L-BFGS approximation is considered as acceptable if:

          ⟨d,g⟩ ≤ -epsilon⋅‖d‖⋅‖g‖

      where `g` denotes the projected gradient of the objective function (which
      is just the gradient in unconstrained case).  Otherwise, the condition
      writes `⟨d,g⟩ < 0`.  The default is `epsilon = 0` so only the latter
      condition is checked.

    - Keyword `blmvm` (false by default) specifies whether to use BLMVM trick
      to account for the bound constraints in the L-BFGS model of the Hessian.
      If `blmvm` is set true, the overhead of the algorithm may be reduced, but
      the L-BFGS model of the Hessian is more likely to be inaccurate causing
      the algorithm to choose the steepest descent direction more often.

    - Keyword `verb`, if positive, specifies to print information every `verb`
      iterations.  Nothing is printed if `verb ≤ 0`.  By default, `verb = 0`.

    - Keyword `printer` is to specify a user-defined subroutine to print
      information every `verb` iterations.  This subroutine is called as:

          printer(output, iters, evals, rejects, t, x, f, g, pgnorm,
                  alpha, fg)

      with `output` the output stream specified by keyword `output`, `iters`
      the number of algorithm iterations, `evals` the number of calls to `fg`,
      `rejects` the number of rejections of the LBFGS direction, `t` the
      elapsed time in seconds, `x` the current variables, `f` and `g` the value
      and the gradient of the objective function at `x`, `pgnorm` the Euclidean
      norm of the projected gradient of the objective function at `x`, `alpha`
      the last step length, and `fg` the objective function itself.

    - Keyword `output` specifies the file stream to print information,
      `sys.stdout` by default.

    - Keyword `observer` is to specify a user-defined subroutine to be called
      at every iteration as follows:

          observer(iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg)

      with the same arguments as for the printer (except `output`).

    - Keyword `throwerrors` (true by default), specifies whether to raise an
      exception in case of errors instead or just returning a `status`
      indicating the problem.  Note that early termination due to limits set on
      the number of iterations or of evaluations of the objective function are
      not considered as an error.

    """
    # Constants.
    Inf = _np.inf

    # Tolerances.  Most of these are forced to be nonnegative to simplify
    # tests.
    (fatol, frtol) = get_tolerances(ftol, -Inf)
    (gatol, grtol) = get_tolerances(gtol, 0.0)
    (xatol, xrtol) = get_tolerances(xtol, 0.0)

    # Bound constraints.  For faster code, unlimited bounds are preferentially
    # represented by `None`.
    if not lower is None and _np.all(lower == -Inf):
        lower = None
    if not upper is None and _np.all(upper == +Inf):
        upper = None
    bounded = not lower is None or not upper is None
    if not bounded:
        # No needs to use BLMVM trick in the unconstrained case.
        blmvm = False

    # Other initialization.
    x = x0           # initial iterate (avoiding copy)
    g = None         # gradient
    f0 = +Inf        # function value at start of line-search
    g0 = None        # gradient at start of line-search
    d = None         # search direction
    s = None         # effective step
    pg = None        # projected gradient
    pg0 = None       # projected gradient at start of line search
    pgnorm = 0.0     # Euclidean norm of the (projected) gradient
    alpha = 0.0      # step length
    amin = -Inf      # first step length threshold
    amax = +Inf      # last step length threshold
    evals = 0        # number of calls to `fg`
    iters = 0        # number of iterations
    projs = 0        # number of projections onto the feasible set
    rejects = 0      # number of search direction rejections
    status = 0       # non-zero when algorithm is about to terminate
    best_f = +Inf    # function value at `best_x`
    best_g = None    # gradient at `best_x`
    best_x = None    # best solution found so far
    best_pgnorm = -1 # norm of projected gradient at `best_x` (< 0 if unknown)
    best_alpha =  0  # step length at `best_x` (< 0 if unknown)
    best_evals = -1  # number of calls to `fg` at `best_x`
    last_evals = -1  # number of calls to `fg` at last iterate
    last_print = -1  # iteration number for last print
    last_obsrv = -1  # iteration number for last call to observer
    freevars = None  # subset of free variables (not yet known)
    lbfgs = LBFGS(mem)
    if verb > 0:
        t0 = elapsed_time(0.0)

    call_observer = not observer is None

    # Algorithm stage follows that of the line-search, it is one of:
    # 0 = initially;
    # 1 = line-search in progress;
    # 2 = line-search has converged.
    stage = 0

    while True:
        # Make the variables feasible.
        if bounded:
            # In principle, we can avoid projecting the variables whenever
            # `alpha ≤ amin` (because the feasible set is convex) but rounding
            # errors could make this wrong.  It is safer to always project the
            # variables.  This cost O(n) operations which are probably
            # negligible compared to, say, computing the objective function and
            # its gradient.
            x = clamp(x, lower, upper)
            projs += 1

        # Compute the objective function and its gradient.
        f, g = fg(x)
        evals += 1
        if f < best_f or evals == 1:
            # Save best solution so far.
            best_f = f
            best_g = g # FIXME: this is not a copy
            best_x = x # FIXME: this is not a copy
            best_pgnorm = -1 # must be recomputed
            best_alpha = alpha
            best_evals = evals

        if stage != 0:
            # Line-search in progress, check for line-search convergence.
            lnsrch.iterate(f)
            if lnsrch.converged():
                # Line-search has converged, `x` is the next iterate.
                iters += 1
                last_evals = evals
                stage = 2
            else:
                # Line-search has not converged, peek next trial step.
                alpha = lnsrch.step()
                stage = 1

        if stage != 1:
            # Initial or next iterate after convergence of line-search.
            if bounded:
                # Determine the subset of free variables and compute the norm
                # of the projected gradient (needed to check for convergence).
                freevars = unblocked_variables(x, lower, upper, g)
                pg = freevars*g # FIXME: freevars.astype(T)
                pgnorm = norm2(pg)
                if not blmvm:
                    # Projected gradient no longer needed, free some memory.
                    pg = None

            else:
                # Just compute the norm of the gradient.
                pgnorm = norm2(g)

            if evals == best_evals:
                # Now we know the norm of the (projected) gradient at the best
                # solution so far.
                best_pgnorm = pgnorm

            # Check for algorithm convergence or termination.
            if evals == 1:
                # Compute value for testing the convergence in the gradient.
                gtest = max(gatol, grtol*pgnorm)

            if pgnorm <= gtest:
                # Convergence in gradient.
                status = GTEST_SATISFIED
                break

            if stage == 2:
                # Check convergence in relative function reduction.
                if f <= fatol or abs(f - f0) <= frtol*max(abs(f), abs(f0)):
                    status = FTEST_SATISFIED
                    break
                # Compute the effective change of variables.
                s = x - x0
                snorm = norm2(s)
                # Check convergence in variables.
                if snorm <= xatol or (xrtol > 0 and snorm <= xrtol*norm2(x)):
                    status = XTEST_SATISFIED
                    break

            if iters >= maxiter:
                status = TOO_MANY_ITERATIONS
                break

        if evals >= maxeval:
            status = TOO_MANY_EVALUATIONS
            break

        if stage != 1:
            # Call user defined observer.
            t = elapsed_time(t0)
            if call_observer:
                observer(iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg)
                last_obsrv = iters

            # Possibly print iteration information.
            if verb > 0 and (iters % verb) == 0:
                printer(output, iters, evals, rejects, t, x, f, g, pgnorm,
                        alpha, fg)
                last_print = iters

            if stage != 0:
                # At least one step has been performed, L-BFGS approximation
                # can be updated.
                if blmvm:
                    lbfgs.update(s, pg - pg0)
                else:
                    lbfgs.update(s, g - g0)

            # Determine a new search direction `d`.  Parameter `flg` is set to:
            #   0 if `d` is not a search direction,
            #   1 if `d` is unscaled steepest descent,
            #   2 if `d` is scaled sufficient descent.
            flg = 0
            # Use L-BFGS approximation to compute a search direction and check
            # that it is an acceptable descent direction.
            if blmvm:
                (d, scaled) = lbfgs.apply(-pg)*freevars
            else:
                (d, scaled) = lbfgs.apply(-g, freevars)
            dg = inner(d, g)
            if not scaled:
                # No exploitable curvature information, `d` is the unscaled
                # steepest feasible direction, that is the opposite of the
                # projected gradient.
                flg = 1
            else:
                # Some exploitable curvature information were available.
                flg = 2
                if dg >= 0:
                    # L-BFGS approximation does not yield a descent direction.
                    flg = 0 # discard search direction
                    if not bounded:
                        if throwerrors:
                            raise AssertionError("L-BFGS approximation is not positive definite")
                        status = NOT_POSITIVE_DEFINITE
                        break
                elif epsilon > 0:
                    # A more restrictive criterion has been specified for
                    # accepting a descent direction.
                    if dg > -epsilon*norm2(d)*pgnorm:
                        flg = 0 # discard search direction

            if flg == 0:
                # No exploitable information about the Hessian is available or
                # the direction computed using the L-BFGS approximation failed
                # to be a sufficient descent direction.  Take the steepest
                # feasible descent direction.
                if bounded:
                    d = -g*freevars
                else:
                    d = -g
                dg = -pgnorm**2
                flg = 1 # scaling needed

            # Determine the length `alpha` of the initial step along `d`.
            if flg == 2:
                # The search direction is already scaled.
                alpha = 1.0
            else:
                # Increment number of rejections if not very first iteration.
                if iters > 0:
                    ++rejects
                # Find a suitable step size along the steepest feasible
                # descent direction `d`.  Note that `pgnorm`, the Euclidean
                # norm of the (projected) gradient, is also that of `d` in
                # that case.
                alpha = steepest_descent_step(x, pgnorm, f, fmin=fmin,
                                              xtiny=xtiny, f2nd=f2nd)

            if bounded:
                # Safeguard the step to avoid searching in a region where
                # all bounds are overreached.
                (amin, amax) = line_search_limits(x, lower, upper, alpha, d)
                alpha = min(alpha, amax)

            # Initialize line-search.
            lnsrch.start(f, dg, alpha)
            stage = 1

            # Save iterate at start of line-search.
            f0 = f
            g0 = g # FIXME: not a copy
            x0 = x # FIXME: not a copy
            if blmvm:
                pg0 = pg # FIXME: not a copy

        # Compute next iterate.
        if alpha == 1:
            x = x0 + d
        else:
            x = x0 + alpha*d

    # In case of abnormal termination, some progresses may have been made
    # since the start of the line-search.  In that case, we restore the best
    # solution so far.
    if best_f < f:
        f = best_f
        g = best_g
        x = best_x
        if verb > 0:
            # Restore other information for printing.
            alpha = best_alpha
            if best_pgnorm >= 0:
                pgnorm = best_pgnorm
            else:
                # Re-compute the norm of the (projected) gradient.
                if bounded:
                    freevars = unblocked_variables(x, lower, upper, g)
                    pgnorm = norm2(g*freevars)
                else:
                    pgnorm = norm2(g)

            if f < f0:
                # Some progresses since last iterate, pretend that one more
                # iteration has been performed.
                ++iters

    t = elapsed_time(t0)
    if call_observer and iters > last_obsrv:
        observer(iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg)
    if verb > 0:
        if iters > last_print:
            printer(output, iters, evals, rejects, t, x, f, g, pgnorm,
                    alpha, fg)
        if printer is vmlmb_printer:
            print(f"# Termination: {reason(status)}", file=output)

    return (x, f, g, status)

#------------------------------------------------------------------------------
# VECTORIZED OPERATIONS

def inner(x, y):
    """
    Compute the inner product of `x` and `y` regardless of their shapes
    (their number of elements must however match).  Complex valued arrays
    are not (yet) supported.

    See also: `optm.norm2`.
    """
    return _np.vdot(x, y)

def norm1(x):
    """Compute the L1-norm of `x`, that is `sum(abs(x))` but computed as
    efficiently as possible.

    See also: `optm.norm2`, `optm.norminf`.
    """
    return _np.linalg.norm(x.ravel(), 1)

# to speed-up, see https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray
def norm2(x):
    """
    Compute the Euclidean (L2) norm of `x` that is `sqrt(sum(abs(x)**2))` but
    computed as efficiently as possible.

    See also: `optm.inner`, `optm.norm1`, `optm.norminf`.
    """
    return _np.linalg.norm(x.ravel())

def norminf(x):
    """
    Compute the infinite norm of `x`, that is `max(abs(x))` but computed as
    efficiently as possible.

    See also: `optm.norm1`, `optm.norm2`.
    """
    return _np.linalg.norm(x.ravel(), _np.inf)

def promote_multiplier(alpha, x):
    """
    Convert multiplier `alpha` (a scalar real) to a suitable floating-point
    type for multiplying array `x`.
    """
    T = x.dtype
    if T == _np.double or T == _np.cdouble:
        return _np.double(alpha)
    elif T == _np.float or T == _np.cfloat:
        return _np.float(alpha)
    elif T == _np.longdouble or T == _np.clongdouble:
        return _np.longdouble(alpha)

def scale(x, alpha):
    """
    Compute `alpha*x` efficiently and taking care of preserving the
    floating-point type of `x`.
    """
    if alpha == 0:
        return _np.zeros(x.shape, x.dtype)
    elif alpha == 1:
        return x
    elif alpha == -1:
        return -x
    else:
        return _np.multiply(promote_multiplier(alpha, x), x)

# to speed-up, see https://stackoverflow.com/questions/45200278/numpy-fusing-multiply-and-add-to-avoid-wasting-memory
def update(y, alpha, x):
    """
    Compute `y + alpha*x` efficiently and taking care of preserving the
    floating-point type of `x` and `y`.
    """
    if alpha == 0:
        return y
    elif alpha == 1:
        return y + x
    elif alpha == -1:
        return y - x
    else:
        return _np.add(y, scale(x, alpha))

def abs2(x):
    """Returns the squared absolute value of its argument."""
    if _np.iscomplexobj(x):
        x_re = x.real
        x_im = x.imag
        return x_re*x_re + x_im*x_im
    else:
        return x*x

#------------------------------------------------------------------------------
# UTILITIES

def elapsed_time(t0=0.0):
    """Get elapsed time since `t0` in seconds."""
    return time.clock_gettime(time.CLOCK_MONOTONIC) - t0

def tolerance(x, atol, rtol):
    """
    Given absolute and relative tolerances `atol` and `rtol`, this function
    yields:

         max(0, atol, rtol*abs(x))    # if `x` is a scalar
         max(0, atol, rtol*norm(x))   # if `x` is an array

    where `norm(x)` is the Euclidean norm of `x` as computed by `optm.norm2`
    (which to see).  If `rtol ≤ 0`, the computation of `norm(x)` is avoided.

    See also: `optm.norm2`, `optm.get_tolerances`.
    """
    tol = max(0.0, atol)
    if rtol <= 0.0:
        return tol
    else:
        return max(tol, rtol*norm2(x))

def get_tolerances(tol, atol=0.0):
    """
    Get a 2-tuple of absolute and relative tolerances given tolerance `tol` and
    default absolute tolerance `atol`.  If `tol` is a scalar, it is assumed
    that `atol` and `tol` are the absolute and relative tolerances.  Otherwise,
    `tol` must specify the absolute and relative tolerances as a 2-tuple (or an
    array of 2 elements).

    See also: `optm.tolerance`.
    """
    if _np.isscalar(tol):
        return (atol, tol)
    else:
        return (tol[0], tol[1])
