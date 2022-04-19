# -*- coding: utf-8 -*-

"""Unconstrained non-linear optimization problems from MINPACK-1 Project.

The Python functions implementing the MINPACK-1 problems are named `prob_P`
with `P` the problem number in the range 1:18.  Any problem function can be
called as:

    prob_P() -> yields the name of the problem.

    prob_P(0, n=None, factor=None) -> yields the starting point for the problem
        as a vector `x` of `n` elements which is a multiple (times `factor`,
        default 1) of the standard starting point.  For the 7-th problem the
        standard starting point is 0, so in this case, if `factor` is not
        unity, then the function returns `x` filled with `factor`.  The values
        of `n` for problems 1,2,3,4,5,10,11,12,16 and 17 are 3,6,3,2,3,2,4,3,2
        and 4, respectively.  For problem 7, `n` may be 2 or greater but is
        usually 6 or 9.  For problems 6,8,9,13,14,15 and 18, `n` may be
        variable, however it must be even for problem 14, a multiple of 4 for
        problem 15, and not greater than 50 for problem 18.

    prob_P(1, x) -> yields the value of the objective function of the problem.
        `x` is the parameter array: a vector of length `n`.

    prob(2, x) -> yields the gradient of the objective function of the problem.

Since the execution time may change, you can compare the outputs after
filtering with:

    sed -e 's/^\( *[0-9]*\) *[^ ]*/\1/'

to get rid of the 2nd column.

History:

- FORTRAN 77 code.  Argonne National Laboratory. MINPACK Project.  March 1980.
  Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. Moré.

- Conversion to Python.  April 2022.  Éric Thiébaut.

References:

- J. J. Moré, B. S. Garbow and K. E. Hillstrom, "Testing unconstrained
  optimization software," in ACM Trans. Math. Software 7 (1981), 17-41.

- J. J. Moré, B. S. Garbow and K. E. Hillstrom, "Fortran subroutines for
  testing unconstrained optimization software," in ACM Trans. Math. Software 7
  (1981), 136-140.

"""

import sys
import numpy as np
import math
import optm
from numpy import log, exp, sqrt, sin, cos, arctan

def test(probs = range(1, 19), /, *, n=None, factor=None, fmin=0,
         mem="max", verb=1, blmvm=False, output=sys.stdout, **kwds):
    """Run one or several tests from the MINPACK-1 Project.

    Usage:

         optm_minpack1_test(probs=range(1,19))

    Argument `probs` is a single problem number (in the range 1:18) or a list of
    problem numbers.  By default, all problems are tested with keywords
    `mem="max"`, `fmin=0`, and `verb=1`.

    Keywords:
      n - Size of the problem.

      factor - Scaling factor for the starting point.

      mem, fmin, lnsrch,
      xtiny, epsilon, f2nd,
      ftol, gtol, xtol,
      blmvm, maxiter, maxeval,
      verb, cputime, output - These keywords are passed to `optm_vmlmb` (which
          to see).  All problems can be tested with `fmin=0`.  By default,
          `mem="max"` to indicate that the number of memorized previous iterates
          should be equal to the size of the problem.

    See also: `optm.vmlmb`.

    """
    # Output stream.
    if type(output) == str:
        output = open(output, mode="a")
    if probs is None:
        probs = range(1, 19)
    elif isinstance(probs, int):
        probs = [probs]
    for j in probs:
        f = eval(f"prob_{j:d}")
        x0 = f(0, n=n, factor=factor)
        if mem == "max":
            m = x0.size
        else:
            m = mem
        if verb != 0:
            name = f()
            if blmvm:
                algo = "BLMVM"
            else:
                algo = "VMLMB"
            print(f"# MINPACK-1 Unconstrained Problem #{j:d} ({x0.size} variables): {f():s}.",
                  file=output)
            print(f"# Algorithm: {algo:s} (mem={m:d}).",file=output)
        (x, fx, gx, status) = optm.vmlmb(lambda x: (f(1, x), f(2, x)), x0,
                                         mem=m, blmvm=blmvm, fmin=fmin,
                                         verb=verb, output=output, **kwds)

#-----------------------------------------------------------------------------

# Helical valley function.
def prob_1(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 3
        elif n != 3:
            raise ValueError("N must be 3 for problem #1")
        x = np.array([-1.0, 0.0, 0.0])
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1 or job == 2:
        # Objective function or gradient.
        tpi = 2*np.pi
        if x[0] > 0.0:
            th = arctan(x[1]/x[0])/tpi
        elif x[0] < 0.0:
            th = arctan(x[1]/x[0])/tpi + 0.5
        elif x[1] >= 0.0:
            th = 0.25
        else:
            th = -0.25
        arg = x[0]**2 + x[1]**2
        r = sqrt(arg)
        t = x[2] - 10.0*th
        if job == 1:
            # Objective function.
            f = 100.0*(t*t + (r - 1.0)**2) + x[2]**2
            return f
        else:
            # Gradient.
            s1 = 10.0*t/(tpi*arg)
            g = np.ndarray(x.shape, x.dtype)
            g[0] = 200.0*(x[0] - x[0]/r + x[1]*s1)
            g[1] = 200.0*(x[1] - x[1]/r - x[0]*s1)
            g[2] = 2.0*(100.0*t + x[2])
            return g
    else:
        # Problem name.
        return "Helical valley function"

# Biggs exp6 function.
def prob_2(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 6
        elif n != 6:
            raise ValueError("N must be 6 for problem #2")
        x = np.ndarray(n, np.double)
        x[0] = 1.0
        x[1] = 2.0
        x[2] = 1.0
        x[3] = 1.0
        x[4] = 1.0
        x[5] = 1.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        f = 0.0
        for i in range(1, 14):
            d1 = i/10.0
            d2 = exp(-d1) - 5.0*exp(-10.0*d1) + 3.0*exp(-4.0*d1)
            s1 = exp(-d1*x[0])
            s2 = exp(-d1*x[1])
            s3 = exp(-d1*x[4])
            t = x[2]*s1 - x[3]*s2 + x[5]*s3 - d2
            f += t*t
        return f
    elif job == 2:
        # Gradient.
        g = np.zeros(x.shape, np.double)
        for i in range(1, 14):
            d1 = i/10.0
            d2 = exp(-d1) - 5.0*exp(-10.0*d1) + 3.0*exp(-4.0*d1)
            s1 = exp(-d1*x[0])
            s2 = exp(-d1*x[1])
            s3 = exp(-d1*x[4])
            t = x[2]*s1 - x[3]*s2 + x[5]*s3 - d2
            th = d1*t
            g[0] = g[0] - s1*th
            g[1] = g[1] + s2*th
            g[2] = g[2] + s1*t
            g[3] = g[3] - s2*t
            g[4] = g[4] - s3*th
            g[5] = g[5] + s3*t
        g[0] = 2.0*x[2]*g[0]
        g[1] = 2.0*x[3]*g[1]
        g[2] = 2.0*g[2]
        g[3] = 2.0*g[3]
        g[4] = 2.0*x[5]*g[4]
        g[5] = 2.0*g[5]
        return g
    else:
        # Problem name.
        return "Biggs exp6 function"

# Gaussian function.
PROB_3_Y = np.array([9.0e-4,   4.4e-3,   1.75e-2,  5.4e-2,   1.295e-1,
                     2.42e-1,  3.521e-1, 3.989e-1, 3.521e-1, 2.42e-1,
                     1.295e-1, 5.4e-2,   1.75e-2,  4.4e-3,   9.0e-4])
def prob_3(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 3
        elif n != 3:
            raise ValueError("N must be 3 for problem #3")
        x = np.ndarray(n, np.double)
        x[0] = 0.4
        x[1] = 1.0
        x[2] = 0.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        f = 0.0
        for i in range(0, 15):
            d1 = i/2.0
            d2 = 3.5 - d1 - x[2]
            arg = -0.5*x[1]*d2*d2
            r = exp(arg)
            t = x[0]*r - PROB_3_Y[i]
            f += t*t
        return f
    elif job == 2:
        # Gradient.
        g = np.zeros(x.shape, np.double)
        for i in range(0, 15):
            d1 = i/2.0
            d2 = 3.5 - d1 - x[2]
            arg = -0.5*x[1]*d2*d2
            r = exp(arg)
            t = x[0]*r - PROB_3_Y[i]
            s1 = r*t
            s2 = d2*s1
            g[0] = g[0] + s1
            g[1] = g[1] - d2*s2
            g[2] = g[2] + s2
        g[0] = 2.0*g[0]
        g[1] = x[0]*g[1]
        g[2] = 2.0*x[0]*x[1]*g[2]
        return g
    else:
        # Problem name.
        return "Gaussian function"

# Powell badly scaled function.
def prob_4(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 2
        elif n != 2:
            raise ValueError("N must be 2 for problem #4")
        x = np.ndarray(n, np.double)
        x[0] = 0.0
        x[1] = 1.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        t1 = 1e4*x[0]*x[1] - 1.0
        s1 = exp(-x[0])
        s2 = exp(-x[1])
        t2 = s1 + s2 - 1.0001
        f = t1*t1 + t2*t2
        return f
    elif job == 2:
        # Gradient.
        t1 = 1e4*x[0]*x[1] - 1.0
        s1 = exp(-x[0])
        s2 = exp(-x[1])
        t2 = s1 + s2 - 1.0001
        g = np.ndarray(x.shape, np.double)
        g[0] = 2.0*(1e4*x[1]*t1 - s1*t2)
        g[1] = 2.0*(1e4*x[0]*t1 - s2*t2)
        return g
    else:
        # Problem name.
        return "Powell badly scaled function"

# Box 3-dimensional function.
def prob_5(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 3
        elif n != 3:
            raise ValueError("N must be 3 for problem #5")
        x = np.ndarray(n, np.double)
        x[0] = 0.0
        x[1] = 10.0
        x[2] = 20.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        f = 0.0
        for i in range(1, 11):
            d1 = np.double(i)
            d2 = d1/10.0
            s1 = exp(-d2*x[0])
            s2 = exp(-d2*x[1])
            s3 = exp(-d2) - exp(-d1)
            t = s1 - s2 - s3*x[2]
            f += t*t
        return f
    elif job == 2:
        # Gradient.
        g = np.zeros(x.size, np.double)
        for i in range(1, 11):
            d1 = np.double(i)
            d2 = d1/10.0
            s1 = exp(-d2*x[0])
            s2 = exp(-d2*x[1])
            s3 = exp(-d2) - exp(-d1)
            t = s1 - s2 - s3*x[2]
            th = d2*t
            g[0] = g[0] - s1*th
            g[1] = g[1] + s2*th
            g[2] = g[2] - s3*t
        return 2*g
    else:
        # Problem name.
        return "Box 3-dimensional function"

# Variably dimensioned function.
def prob_6(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 10
        elif n < 1:
            raise ValueError("N must be >= 1 in problem #6")
        x = np.ndarray(n, np.double)
        h = 1.0/n
        for j in range(n):
            x[j] = 1.0 - (j + 1)*h
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        t1 = 0.0
        t2 = 0.0
        for j in range(x.size):
            t1 += (j + 1)*(x[j] - 1.0)
            t = x[j] - 1.0
            t2 += t*t
        t = t1*t1
        f = t2 + t*(1.0 + t)
        return f
    elif job == 2:
        # Gradient.
        n = x.size
        t1 = 0.0
        for j in range(n):
            t1 += (j + 1)*(x[j] - 1.0)
        t = t1*(1.0 + 2.0*t1*t1)
        g = np.ndarray(n, np.double)
        for j in range(n):
            g[j] = 2.0*(x[j] - 1.0 + (j + 1)*t)
        return g
    else:
        # Problem name.
        return "Variably dimensioned function"

# Watson function.
def prob_7(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        msg = "N may be 2 or greater but is usually 6 or 9 for problem #7"
        if n is None:
            print(msg)
            n = 6
        elif n < 2:
            raise ValueError(msg)
        x = np.zeros(n, np.double)
        if not factor is None and factor != 1:
            x[:] = factor
        return x
    elif job == 1:
        # Objective function.
        n = x.size
        f = 0.0
        for i in range(1, 30):
            d1 = i/29.0
            s1 = 0.0
            d2 = 1.0
            for j in range(1, n):
                s1 += j*d2*x[j]
                d2 *= d1
            s2 = 0.0
            d2 = 1.0
            for j in range(n):
                s2 += d2*x[j]
                d2 *= d1
            t = s1 - s2*s2 - 1.0
            f += t*t
        t = x[0]*x[0]
        t1 = x[1] - t - 1.0
        f += t + t1*t1
        return f
    elif job == 2:
        # Gradient.
        n = x.size
        g = np.zeros(n, np.double)
        for i in range(1, 30):
            d1 = i/29.0
            s1 = 0.0
            d2 = 1.0
            for j in range(1, n):
                s1 += j*d2*x[j]
                d2 *= d1
            s2 = 0.0
            d2 = 1.0
            for j in range(n):
                s2 += d2*x[j]
                d2 *= d1
            t = s1 - s2*s2 - 1.0
            s3 = 2.0*d1*s2
            d2 = 2.0/d1
            for j in range(n):
                g[j] += d2*(j - s3)*t
                d2 *= d1
        t1 = x[1] - x[0]*x[0] - 1.0
        g[0] += x[0]*(2.0 - 4.0*t1)
        g[1] += 2.0*t1
        return g
    else:
        # Problem name.
        return "Watson function"

# Penalty function I.
def prob_8(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 10
        elif n < 1:
            raise ValueError("N must be >= 1 in problem #8")
        x = np.ndarray(n, np.double)
        for j in range(x.size):
            x[j] = j + 1
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        n = x.size
        t1 = -0.25
        t2 = 0.0
        for j in range(x.size):
            t1 += x[j]**2
            t = x[j] - 1.0
            t2 += t*t
        f = 1e-5*t2 + t1*t1
        return f
    elif job == 2:
        # Gradient.
        t1 = -0.25
        for j in range(x.size):
            t1 += x[j]**2
        d1 = 2.0*1e-5
        th = 4.0*t1
        g = np.ndarray(x.shape, np.double)
        for j in range(x.size):
            g[j] = d1*(x[j] - 1.0) + x[j]*th
        return g
    else:
        # Problem name.
        return "Penalty function I"

# Penalty function II.
def prob_9(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 10
        elif n < 1:
            raise ValueError("N must be >= 1 in problem #9")
        x = np.full(n, 0.5, np.double)
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        n = x.size
        t1 = -1.0
        t2 = 0.0
        t3 = 0.0
        d1 = exp(0.1)
        d2 = 1.0
        s2 = 0.0
        for j in range(n):
            t1 += (n - j)*x[j]**2
            s1 = exp(x[j]/10.0)
            if j > 0:
                s3 = s1 + s2 - d2*(d1 + 1.0)
                t2 += s3*s3
                t = (s1 - 1.0/d1)
                t3 += t*t
            s2 = s1
            d2 *= d1
        t = x[0] - 0.2
        f = 1e-5*(t2 + t3) + t1*t1 + t*t
        return f
    elif job == 2:
        # Gradient.
        n = x.size
        g = np.ndarray(x.shape, np.double)
        s2 = 0.0
        t1 = -1.0
        for j in range(n):
            t1 += (n - j)*x[j]**2
        d1 = exp(0.1)
        d2 = 1.0
        th = 4.0*t1
        for j in range(n):
            g[j] = (n - j)*x[j]*th
            s1 = exp(x[j]/10.0)
            if j > 0:
                s3 = s1 + s2 - d2*(d1 + 1.0)
                g[j] += 1e-5*s1*(s3 + s1 - 1.0/d1)/5.0
                g[j-1] += 1e-5*s2*s3/5.0
            s2 = s1
            d2 *= d1
        g[0] += 2.0*(x[0] - 0.2)
        return g
    else:
        # Problem name.
        return "Penalty function II"

# Brown badly scaled function.
def prob_10(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 2
        elif n != 2:
            raise ValueError("N must be 2 for problem #10")
        x = np.full(n, 1.0, np.double)
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        t1 = x[0] - 1e6
        t2 = x[1] - 2e-6
        t3 = x[0]*x[1] - 2.0
        f = t1*t1 + t2*t2 + t3*t3
        return f
    elif job == 2:
        # Gradient.
        t1 = x[0] - 1e6
        t2 = x[1] - 2e-6
        t3 = x[0]*x[1] - 2.0
        g = np.ndarray(x.shape, np.double)
        g[0] = 2.0*(t1 + x[1]*t3)
        g[1] = 2.0*(t2 + x[0]*t3)
        return g
    else:
        # Problem name.
        return "Brown badly scaled function"

# Brown and Dennis function.
def prob_11(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 4
        elif n != 4:
            raise ValueError("N must be 4 for problem #11")
        x = np.ndarray(n, np.double)
        x[0] = 25.0
        x[1] = 5.0
        x[2] = -5.0
        x[3] = -1.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        f = 0.0
        for i in range(1, 21):
            d1 = i/5.0
            d2 = sin(d1)
            t1 = x[0] + d1*x[1] - exp(d1)
            t2 = x[2] + d2*x[3] - cos(d1)
            t = t1*t1 + t2*t2
            f += t*t
        return f
    elif job == 2:
        # Gradient.
        g = np.zeros(x.shape, np.double)
        for i in range(1, 21):
            d1 = i/5.0
            d2 = sin(d1)
            t1 = x[0] + d1*x[1] - exp(d1)
            t2 = x[2] + d2*x[3] - cos(d1)
            t = t1*t1 + t2*t2
            s1 = t1*t
            s2 = t2*t
            g[0] += s1
            g[1] += d1*s1
            g[2] += s2
            g[3] += d2*s2
        return 4*g
    else:
        # Problem name.
        return "Brown and Dennis function"

# Gulf research and development function.
def prob_12(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 3
        elif n != 3:
            raise ValueError("N must be 3 for problem #12")
        x = np.ndarray(n, np.double)
        x[0] = 5.0
        x[1] = 2.5
        x[2] = 0.15
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        f = 0.0
        d1 = 2.0/3.0
        for i in range(1, 100):
            arg = i/100.0
            r = (-50.0*log(arg))**d1 + 25.0 - x[1]
            t1 = (abs(r)**x[2])/x[0]
            t2 = exp(-t1)
            t = t2 - arg
            f += t*t
        return f
    elif job == 2:
        # Gradient.
        g = np.zeros(x.shape, np.double)
        d1 = 2.0/3.0
        for i in range(1, 100):
            arg = i/100.0
            r = (-50.0*log(arg))**d1 + 25.0 - x[1]
            t1 = (abs(r)**x[2])/x[0]
            t2 = exp(-t1)
            t = t2 - arg
            s1 = t1*t2*t
            g[0] += s1
            g[1] += s1/r
            g[2] -= s1*log(abs(r))
        g[0] *= 2.0/x[0]
        g[1] *= 2.0*x[2]
        g[2] *= 2.0
        return g
    else:
        # Problem name.
        return "Gulf research and development function"

# Trigonometric function.
def prob_13(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 10
        elif n < 1:
            raise ValueError("N must be >= 1 in problem #13")
        h = 1.0/n
        x = np.full(n, h, np.double)
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        n = x.size
        s1 = np.sum(cos(x))
        f = 0.0
        for j in range(n):
            t = (n + j + 1) - sin(x[j]) - s1 - (j + 1)*cos(x[j])
            f += t*t
        return f
    elif job == 2:
        # Gradient.
        n = x.size
        g = cos(x)
        s1 = np.sum(g)
        s2 = 0.0
        for j in range(n):
            th = sin(x[j])
            t = (n + j + 1) - th - s1 - (j + 1)*g[j]
            s2 += t
            g[j] = ((j + 1)*th - g[j])*t
        for j in range(n):
            g[j] = 2.0*(g[j] + sin(x[j])*s2)
        return g
    else:
        # Problem name.
        return "Trigonometric function"

# Extended Rosenbrock function.
def prob_14(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 10
        elif n < 1 or n%2 != 0:
            raise ValueError("N must be a multiple of 2 in problem #14")
        x = np.ndarray(n, np.double)
        x[0:n:2] = -1.2
        x[1:n:2] = 1.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        n = x.size
        f = 0.0
        for j in range(0,n,2):
            t1 = 1.0 - x[j]
            t2 = 10.0*(x[j+1] - x[j]*x[j])
            f += t1*t1 + t2*t2
        return f
    elif job == 2:
        # Gradient.
        n = x.size
        g = np.ndarray(x.shape, np.double)
        for j in range(0,n,2):
            t1 = 1.0 - x[j]
            g[j+1] = 200.0*(x[j+1] - x[j]*x[j])
            g[j] = -2.0*(x[j]*g[j+1] + t1)
        return g
    else:
        # Problem name.
        return "Extended Rosenbrock function"

# Extended Powell function.
def prob_15(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 12
        elif n < 1 or n%4 != 0:
            raise ValueError("N must be a multiple of 4 in problem #15")
        x = np.ndarray(n, np.double)
        x[0:n:4] =  3.0
        x[1:n:4] = -1.0
        x[2:n:4] =  0.0
        x[3:n:4] =  1.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        n = x.size
        f = 0.0
        for j in range(0, n, 4):
            t = x[j] + 10.0*x[j+1]
            t1 = x[j+2] - x[j+3]
            s1 = 5.0*t1
            t2 = x[j+1] - 2.0*x[j+2]
            s2 = t2*t2*t2
            t3 = x[j] - x[j+3]
            s3 = 10.0*t3*t3*t3
            f += t*t + s1*t1 + s2*t2 + s3*t3
        return f
    elif job == 2:
        # Gradient.
        n = x.size
        g = np.ndarray(x.shape, np.double)
        for j in range(0, n, 4):
            t = x[j] + 10.0*x[j+1]
            t1 = x[j+2] - x[j+3]
            s1 = 5.0*t1
            t2 = x[j+1] - 2.0*x[j+2]
            s2 = 4.0*t2*t2*t2
            t3 = x[j] - x[j+3]
            s3 = 20.0*t3*t3*t3
            g[j] = 2.0*(t + s3)
            g[j+1] = 20.0*t + s2
            g[j+2] = 2.0*(s1 - s2)
            g[j+3] = -2.0*(s1 + s3)
        return g
    else:
        # Problem name.
        return "Extended Powell function"

# Beale function.
def prob_16(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 2
        elif n != 2:
            raise ValueError("N must be 2 for problem #16")
        x = np.full(n, 1.0, np.double)
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        s1 = 1.0 - x[1]
        t1 = 1.5 - x[0]*s1
        s2 = 1.0 - x[1]*x[1]
        t2 = 2.25 - x[0]*s2
        s3 = 1.0 - x[1]*x[1]*x[1]
        t3 = 2.625 - x[0]*s3
        f = t1*t1 + t2*t2 + t3*t3
        return f
    elif job == 2:
        # Gradient.
        s1 = 1.0 - x[1]
        t1 = 1.5 - x[0]*s1
        s2 = 1.0 - x[1]*x[1]
        t2 = 2.25 - x[0]*s2
        s3 = 1.0 - x[1]*x[1]*x[1]
        t3 = 2.625 - x[0]*s3
        g = np.ndarray(x.shape, np.double)
        g[0] = -2.0*(s1*t1 + s2*t2 + s3*t3)
        g[1] = 2.0*x[0]*(t1 + x[1]*(2.0*t2 + 3.0*x[1]*t3))
        return g
    else:
        # Problem name.
        return "Beale function"

# Wood function
def prob_17(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 4
        elif n != 4:
            raise ValueError("N must be 4 for problem #17")
        x = np.ndarray(n, np.double)
        x[0] = -3.0
        x[1] = -1.0
        x[2] = -3.0
        x[3] = -1.0
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        s1 = x[1] - x[0]*x[0]
        s2 = 1.0 - x[0]
        s3 = x[1] - 1.0
        t1 = x[3] - x[2]*x[2]
        t2 = 1.0 - x[2]
        t3 = x[3] - 1.0
        f = 100.0*s1*s1 + s2*s2 + 90.0*t1*t1 + t2*t2 + 10.0*(s3 + t3)*(s3 + t3) + (s3 - t3)*(s3 - t3)/10.0
        return f
    elif job == 2:
        # Gradient.
        s1 = x[1] - x[0]*x[0]
        s2 = 1.0 - x[0]
        s3 = x[1] - 1.0
        t1 = x[3] - x[2]*x[2]
        t2 = 1.0 - x[2]
        t3 = x[3] - 1.0
        g = np.ndarray(x.shape, np.double)
        g[0] = -2.0*(200.0*x[0]*s1 + s2)
        g[1] = 200.0*s1 + 20.2*s3 + 19.8*t3
        g[2] = -2.0*(180.0*x[2]*t1 + t2)
        g[3] = 180.0*t1 + 20.2*t3 + 19.8*s3
        return g
    else:
        # Problem name.
        return "Wood function"

# Chebyquad function
def prob_18(job=None, x=None, n=None, factor=None):
    if job == 0:
        # Starting point.
        if n is None:
            n = 25
        elif n < 1 or n > 50:
            raise ValueError("N must be <= 50 for problem #18")
        x = np.ndarray(n, np.double)
        h = 1.0/np.double(n+1)
        for j in range(n):
            x[j] = (j + 1)*h
        if factor is None:
            return x
        else:
            return factor*x
    elif job == 1:
        # Objective function.
        n = x.size
        fvec = np.zeros(x.shape, np.double)
        for j in range(n):
            t1 = 1.0
            t2 = 2.0*x[j] - 1.0
            t = 2.0*t2
            for i in range(n):
                fvec[i] += t2
                th = t*t2 - t1
                t1 = t2
                t2 = th
        f = 0.0
        d1 = 1.0/n
        iev = -1
        for i in range(n):
            t = d1*fvec[i]
            if iev > 0:
                t += 1.0/((i + 1)**2 - 1.0)
            f += t*t
            iev = -iev
        return f
    elif job == 2:
        # Gradient.
        n = x.size
        g = np.ndarray(x.shape, np.double)
        fvec = np.zeros(x.shape, np.double)
        for j in range(n):
            t1 = 1.0
            t2 = 2.0*x[j] - 1.0
            t = 2.0*t2
            for i in range(n):
                fvec[i] += t2
                th = t*t2 - t1
                t1 = t2
                t2 = th
        d1 = 1.0/n
        iev = -1
        for i in range(n):
            fvec[i] *= d1
            if iev > 0:
                fvec[i] += 1.0/((i + 1)**2 - 1.0)
            iev = -iev
        for j in range(n):
            g[j] = 0.0
            t1 = 1.0
            t2 = 2.0*x[j] - 1.0
            t = 2.0*t2
            s1 = 0.0
            s2 = 2.0
            for i in range(n):
                g[j] = g[j] + fvec[i]*s2
                th = 4.0*t2 + t*s2 - s1
                s1 = s2
                s2 = th
                th = t*t2 - t1
                t1 = t2
                t2 = th
        return (2.0*d1)*g
    else:
        # Problem name.
        return "Chebyquad function"

if __name__ == '__main__':
    # Run all tests.
    test(verb=1, gtol=0, xtol=0, ftol=0, fmin=0, mem="max")
