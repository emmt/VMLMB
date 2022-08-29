// optm_minpack1.i -
//
// Unconstrained non-linear optimization problems from MINPACK-1 Project.
//
// The Yorick functions implementing the MINPACK-1 problems are named
// `optm_minpack1_prob_P` with `P` the problem number in the range 1:18.  Any
// problem function, say `prob`, can be called as:
//
//     prob() -> yields the name of the problem.
//
//     prob(0, n=, factor=) -> yields the starting point for the problem as a
//         vector `x` of `n` elements which is a multiple (times `factor`,
//         default 1) of the standard starting point.  For the 7-th problem the
//         standard starting point is 0, so in this case, if `factor` is not
//         unity, then the function returns `x` filled with `factor`.  The
//         values of `n` for problems 1, 2, 3, 4, 5, 10, 11, 12, 16, and 17 are
//         3, 6, 3, 2, 3, 2, 4, 3, 2, and 4, respectively.  For problem 7, `n`
//         may be 2 or greater but is usually 6 or 9.  For problems 6, 8, 9,
//         13, 14, 15, and 18, `n` may be variable, however it must be even for
//         problem 14, a multiple of 4 for problem 15, and not greater than 50
//         for problem 18.
//
//     prob(1, x) -> yields the value of the objective function of the problem.
//         `x` is the parameter array: a vector of length `n`.
//
//     prob(2, x) -> yields the gradient of the objective function of the
//         problem.
//
// Since the execution time may change, you can compare the outputs after
// filtering with:
//
//     sed -e 's/^\( *[0-9]*\) *[^ ]*/\1/'
//
// to get rid of the 2nd column.
//
// History:
// - FORTRAN 77 code.  Argonne National Laboratory. MINPACK Project.  March
//   1980.  Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. Moré.
// - Conversion to C and Yorick.  November 2001, March 2022.  Éric Thiébaut.
//
// References:
// - J. J. Moré, B. S. Garbow and K. E. Hillstrom, "Testing unconstrained
//   optimization software," in ACM Trans. Math. Software 7 (1981), 17-41.
// - J. J. Moré, B. S. Garbow and K. E. Hillstrom, "Fortran subroutines for
//   testing unconstrained optimization software," in ACM Trans. Math. Software
//   7 (1981), 136-140.
//
//-----------------------------------------------------------------------------

func optm_minpack1_test(probs, n=, factor=,
                        // optm_vmlmb options:
                        mem=, lnsrch=, epsilon=,
                        f2nd=, fmin=, dxrel=, dxabs=,
                        ftol=, gtol=, xtol=, blmvm=,
                        maxiter=, maxeval=, verb=, cputime=, output=)
/* DOCUMENT optm_minpack1_test;
         or optm_minpack1_test, p;

     Run one or several tests from the MINPACK-1 Project.  Argument `p` is a
     single problem number (in the range 1:18) or a vector of problem numbers.
     By default, all problems are tested with keywords `mem="max"` and
     `verb=1`.

   KEYWORDS:
     n - Size of the problem.

     factor - Scaling factor for the starting point.

     mem, fmin, lnsrch,
     delta, epsilon, lambda,
     ftol, gtol, xtol,
     blmvm, maxiter, maxeval,
     verb, cputime, output - These keywords are passed to `optm_vmlmb` (which
         to see).  All problems can be tested with `fmin=0`.  By default,
         `mem="max"` to indicate that the number of memorized previous iterates
         should be equal to the size of the problem.

   SEE ALSO: optm_vmlmb. */
{
    // Output stream.
    if (! is_void(output)) {
        if (structof(output) == string) {
            output = open(output, "a");
        } else if (typeof(output) != "text_stream") {
            error, "bad value for keyword OUTPUT";
        }
    }

    if (is_void(mem)) mem = "max";
    if (is_void(verb)) verb = 1;
    if (is_void(probs)) probs = indgen(1:18);
    for (i = 1; i <= numberof(probs); ++i) {
        j = probs(i);
        f = symbol_def(swrite(format="optm_minpack1_prob_%d", j));
        fg = optm_minpack1_fg(f);
        x0 = f(0, n=n, factor=factor);
        m = (mem == "max" ? numberof(x0) : mem);
        if (verb) {
            write, output,
                format="# %s #%d (%d variables): %s.\n",
                "MINPACK-1 Unconstrained Problem", j, numberof(x0), f();
            write, output, format="# Algorithm: %s (mem=%d).\n",
                (blmvm ? "BLMVM" : "VMLMB"), m;
        }
        local status, fx, gx;
        x = optm_vmlmb(fg, x0, fx, gx, status,
                       mem=m, lnsrch=lnsrch, epsilon=epsilon,
                       f2nd=f2nd, fmin=fmin, dxrel=dxrel, dxabs=dxabs,
                       ftol=ftol, gtol=gtol, xtol=xtol, blmvm=blmvm,
                       maxiter=maxiter, maxeval=maxeval, verb=verb,
                       cputime=cputime, output=output);
    }
}

func optm_minpack1_fg(f)
/* DOCUMENT fg = optm_minpack1_fg(f);

     Wrap a problem function `f` from MINPACK-1 into an objective function
     callable by `optm_vmlmb`.

   SEE ALSO: optm_vmlmb
 */
{
    return closure(_optm_minpack1_fg, f);
}

func _optm_minpack1_fg(f, x, &gx)
{
    fx = f(1, x);
    gx = f(2, x);
    return fx;
}

//-----------------------------------------------------------------------------

// Helical valley function.
func optm_minpack1_prob_1(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 3;
        else if (n != 3) error, "N must be 3 for problem #1";
        x = array(double, n);
        x(1) = -1.0;
        x(2) = 0.0;
        x(3) = 0.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        tpi = 8.0*atan(1.0);
        if      (x(1) > 0.0) th = atan(x(2)/x(1))/tpi;
        else if (x(1) < 0.0) th = atan(x(2)/x(1))/tpi + 0.5;
        else                 th = (x(2) >= 0.0 ? 0.25 : -0.25);
        arg = x(1)*x(1) + x(2)*x(2);
        r = sqrt(arg);
        t = x(3) - 10.0*th;
        f = 100.0*(t*t + (r - 1.0)*(r - 1.0)) + x(3)*x(3);
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        tpi = 8.0*atan(1.0);
        if      (x(1) > 0.0) th = atan(x(2)/x(1))/tpi;
        else if (x(1) < 0.0) th = atan(x(2)/x(1))/tpi + 0.5;
        else                 th = (x(2) >= 0.0 ? 0.25 : -0.25);
        arg = x(1)*x(1) + x(2)*x(2);
        r = sqrt(arg);
        t = x(3) - 10.0*th;
        s1 = 10.0*t/(tpi*arg);
        g(1) = 200.0*(x(1) - x(1)/r + x(2)*s1);
        g(2) = 200.0*(x(2) - x(2)/r - x(1)*s1);
        g(3) = 2.0*(100.0*t + x(3));
        return g;
    } else {
        // Problem name.
        return "Helical valley function";
    }
}

// Biggs exp6 function.
func optm_minpack1_prob_2(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 6;
        else if (n != 6) error, "N must be 6 for problem #2";
        x = array(double, n);
        x(1) = 1.0;
        x(2) = 2.0;
        x(3) = 1.0;
        x(4) = 1.0;
        x(5) = 1.0;
        x(6) = 1.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        f = 0.0;
        for (i = 1; i <= 13; ++i) {
            d1 = double(i)/10.0;
            d2 = exp(-d1) - 5.0*exp(-10.0*d1) + 3.0*exp(-4.0*d1);
            s1 = exp(-d1*x(1));
            s2 = exp(-d1*x(2));
            s3 = exp(-d1*x(5));
            t = x(3)*s1 - x(4)*s2 + x(6)*s3 - d2;
            f += t*t;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        for (j = 1; j <= 6; ++j) g(j) = 0.0;
        for (i = 1; i <= 13; ++i) {
            d1 = double(i)/10.0;
            d2 = exp(-d1) - 5.0*exp(-10.0*d1) + 3.0*exp(-4.0*d1);
            s1 = exp(-d1*x(1));
            s2 = exp(-d1*x(2));
            s3 = exp(-d1*x(5));
            t = x(3)*s1 - x(4)*s2 + x(6)*s3 - d2;
            th = d1*t;
            g(1) = g(1) - s1*th;
            g(2) = g(2) + s2*th;
            g(3) = g(3) + s1*t;
            g(4) = g(4) - s2*t;
            g(5) = g(5) - s3*th;
            g(6) = g(6) + s3*t;
        }
        g(1) = 2.0*x(3)*g(1);
        g(2) = 2.0*x(4)*g(2);
        g(3) = 2.0*g(3);
        g(4) = 2.0*g(4);
        g(5) = 2.0*x(6)*g(5);
        g(6) = 2.0*g(6);
        return g;
    } else {
        // Problem name.
        return "Biggs exp6 function";
    }
}

// Gaussian function.
_optm_minpack1_prob_3_y = [9.0e-4,   4.4e-3,   1.75e-2,  5.4e-2,   1.295e-1,
                           2.42e-1,  3.521e-1, 3.989e-1, 3.521e-1, 2.42e-1,
                           1.295e-1, 5.4e-2,   1.75e-2,  4.4e-3,   9.0e-4];
func optm_minpack1_prob_3(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 3;
        else if (n != 3) error, "N must be 3 for problem #3";
        x = array(double, n);
        x(1) = 0.4;
        x(2) = 1.0;
        x(3) = 0.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        f = 0.0;
        for (i = 1; i <= 15; ++i) {
            d1 = 0.5*double(i-1);
            d2 = 3.5 - d1 - x(3);
            arg = -0.5*x(2)*d2*d2;
            r = exp(arg);
            t = x(1)*r - _optm_minpack1_prob_3_y(i);
            f += t*t;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        g(1) = 0.0;
        g(2) = 0.0;
        g(3) = 0.0;
        for (i = 1; i <= 15; ++i) {
            d1 = 0.5*double(i-1);
            d2 = 3.5 - d1 - x(3);
            arg = -0.5*x(2)*d2*d2;
            r = exp(arg);
            t = x(1)*r - _optm_minpack1_prob_3_y(i);
            s1 = r*t;
            s2 = d2*s1;
            g(1) = g(1) + s1;
            g(2) = g(2) - d2*s2;
            g(3) = g(3) + s2;
        }
        g(1) = 2.0*g(1);
        g(2) = x(1)*g(2);
        g(3) = 2.0*x(1)*x(2)*g(3);
        return g;
    } else {
        // Problem name.
        return "Gaussian function";
    }
}

// Powell badly scaled function.
func optm_minpack1_prob_4(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 2;
        else if (n != 2) error, "N must be 2 for problem #4";
        x = array(double, n);
        x(1) = 0.0;
        x(2) = 1.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        t1 = 1e4*x(1)*x(2) - 1.0;
        s1 = exp(-x(1));
        s2 = exp(-x(2));
        t2 = s1 + s2 - 1.0001;
        f = t1*t1 + t2*t2;
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        t1 = 1e4*x(1)*x(2) - 1.0;
        s1 = exp(-x(1));
        s2 = exp(-x(2));
        t2 = s1 + s2 - 1.0001;
        g(1) = 2.0*(1e4*x(2)*t1 - s1*t2);
        g(2) = 2.0*(1e4*x(1)*t1 - s2*t2);
        return g;
    } else {
        // Problem name.
        return "Powell badly scaled function";
    }
}

// Box 3-dimensional function.
func optm_minpack1_prob_5(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 3;
        else if (n != 3) error, "N must be 3 for problem #5";
        x = array(double, n);
        x(1) = 0.0;
        x(2) = 10.0;
        x(3) = 20.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        f = 0.0;
        for (i = 1; i <= 10; ++i) {
            d1 = double(i);
            d2 = d1/10.0;
            s1 = exp(-d2*x(1));
            s2 = exp(-d2*x(2));
            s3 = exp(-d2) - exp(-d1);
            t = s1 - s2 - s3*x(3);
            f += t*t;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        g(1) = 0.0;
        g(2) = 0.0;
        g(3) = 0.0;
        for (i = 1; i <= 10; ++i) {
            d1 = double(i);
            d2 = d1/10.0;
            s1 = exp(-d2*x(1));
            s2 = exp(-d2*x(2));
            s3 = exp(-d2) - exp(-d1);
            t = s1 - s2 - s3*x(3);
            th = d2*t;
            g(1) = g(1) - s1*th;
            g(2) = g(2) + s2*th;
            g(3) = g(3) - s3*t;
        }
        g(1) = 2.0*g(1);
        g(2) = 2.0*g(2);
        g(3) = 2.0*g(3);
        return g;
    } else {
        // Problem name.
        return "Box 3-dimensional function";
    }
}

// Variably dimensioned function.
func optm_minpack1_prob_6(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 10;
        else if (n < 1) error, "N must be >= 1 in problem #6";
        x = array(double, n);
        h = 1.0/double(n);
        for (j = 1; j <= n; ++j) x(j) = 1.0 - double(j)*h;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        t1 = 0.0;
        t2 = 0.0;
        for (j = 1; j <= n; ++j) {
            t1 += double(j)*(x(j) - 1.0);
            t = x(j) - 1.0;
            t2 += t*t;
        }
        t = t1*t1;
        f = t2 + t*(1.0 + t);
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        t1 = 0.0;
        for (j = 1; j <= n; ++j) {
            t1 += double(j)*(x(j) - 1.0);
        }
        t = t1*(1.0 + 2.0*t1*t1);
        for (j = 1; j <= n; ++j) {
            g(j) = 2.0*(x(j) - 1.0 + double(j)*t);
        }
        return g;
    } else {
        // Problem name.
        return "Variably dimensioned function";
    }
}

// Watson function.
func optm_minpack1_prob_7(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        msg = "N may be 2 or greater but is usually 6 or 9 for problem #7";
        if (is_void(n)) {
            write, format="# %s\n", msg;
            n = 6;
        } else if (n < 2) error, msg;
        x = array(double, n);
        if (!is_void(factor)) x(*) = factor;
        return x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        f = 0.0;
        for (i = 1; i <= 29; ++i) {
            d1 = double(i)/29.0;
            s1 = 0.0;
            d2 = 1.0;
            for (j = 2; j <= n; ++j) {
                s1 += double(j-1)*d2*x(j);
                d2 = d1*d2;
            }
            s2 = 0.0;
            d2 = 1.0;
            for (j = 1; j <= n; ++j) {
                s2 += d2*x(j);
                d2 = d1*d2;
            }
            t = s1 - s2*s2 - 1.0;
            f += t*t;
        }
        t = x(1)*x(1);
        t1 = x(2) - t - 1.0;
        f += t + t1*t1;
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        for (j = 1; j <= n; ++j) {
            g(j) = 0.0;
        }
        for (i = 1; i <= 29; ++i) {
            d1 = double(i)/29.0;
            s1 = 0.0;
            d2 = 1.0;
            for (j = 2; j <= n; ++j) {
                s1 += double(j-1)*d2*x(j);
                d2 = d1*d2;
            }
            s2 = 0.0;
            d2 = 1.0;
            for (j = 1; j <= n; ++j) {
                s2 += d2*x(j);
                d2 = d1*d2;
            }
            t = s1 - s2*s2 - 1.0;
            s3 = 2.0*d1*s2;
            d2 = 2.0/d1;
            for (j = 1; j <= n; ++j) {
                g(j) = g(j) + d2*(double(j-1) - s3)*t;
                d2 = d1*d2;
            }
        }
        t1 = x(2) - x(1)*x(1) - 1.0;
        g(1) = g(1) + x(1)*(2.0 - 4.0*t1);
        g(2) = g(2) + 2.0*t1;
        return g;
    } else {
        // Problem name.
        return "Watson function";
    }
}

// Penalty function I.
func optm_minpack1_prob_8(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 10;
        else if (n < 1) error, "N must be >= 1 in problem #8";
        x = array(double, n);
        for (j = 1; j <= n; ++j) x(j) = double(j);
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        t1 = -0.25;
        t2 = 0.0;
        for (j = 1; j <= n; ++j) {
            t1 += x(j)*x(j);
            t = x(j) - 1.0;
            t2 += t*t;
        }
        f = 1e-5*t2 + t1*t1;
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        t1 = -0.25;
        for (j = 1; j <= n; ++j) {
            t1 += x(j)*x(j);
        }
        d1 = 2.0*1e-5;
        th = 4.0*t1;
        for (j = 1; j <= n; ++j) {
            g(j) = d1*(x(j) - 1.0) + x(j)*th;
        }
        return g;
    } else {
        // Problem name.
        return "Penalty function I";
    }
}

// Penalty function II.
func optm_minpack1_prob_9(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 10;
        else if (n < 1) error, "N must be >= 1 in problem #9";
        x = array(double, n);
        for (j = 1; j <= n; ++j) x(j) = 0.5;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        t1 = -1.0;
        t2 = 0.0;
        t3 = 0.0;
        d1 = exp(0.1);
        d2 = 1.0;
        s2 = 0.0;
        for (j = 1; j <= n; ++j) {
            t1 += double(n-j+1)*x(j)*x(j);
            s1 = exp(x(j)/10.0);
            if (j > 1) {
                s3 = s1 + s2 - d2*(d1 + 1.0);
                t2 += s3*s3;
                t = (s1 - 1.0/d1);
                t3 += t*t;
            }
            s2 = s1;
            d2 = d1*d2;
        }
        t = (x(1) - 0.2);
        f = 1e-5*(t2 + t3) + t1*t1 + t*t;
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        s2 = 0.0;
        t1 = -1.0;
        for (j = 1; j <= n; ++j) {
            t1 += double(n-j+1)*x(j)*x(j);
        }
        d1 = exp(0.1);
        d2 = 1.0;
        th = 4.0*t1;
        for (j = 1; j <= n; ++j) {
            g(j) = double(n-j+1)*x(j)*th;
            s1 = exp(x(j)/10.0);
            if (j > 1) {
                s3 = s1 + s2 - d2*(d1 + 1.0);
                g(j) = g(j) + 1e-5*s1*(s3 + s1 - 1.0/d1)/5.0;
                g(j-1) = g(j-1) + 1e-5*s2*s3/5.0;
            }
            s2 = s1;
            d2 = d1*d2;
        }
        g(1) = g(1) + 2.0*(x(1) - 0.2);
        return g;
    } else {
        // Problem name.
        return "Penalty function II";
    }
}

// Brown badly scaled function.
func optm_minpack1_prob_10(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 2;
        else if (n != 2) error, "N must be 2 for problem #10";
        x = array(double, n);
        x(1) = 1.0;
        x(2) = 1.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        t1 = x(1) - 1e6;
        t2 = x(2) - 2e-6;
        t3 = x(1)*x(2) - 2.0;
        f = t1*t1 + t2*t2 + t3*t3;
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        t1 = x(1) - 1e6;
        t2 = x(2) - 2e-6;
        t3 = x(1)*x(2) - 2.0;
        g(1) = 2.0*(t1 + x(2)*t3);
        g(2) = 2.0*(t2 + x(1)*t3);
        return g;
    } else {
        // Problem name.
        return "Brown badly scaled function";
    }
}

// Brown and Dennis function.
func optm_minpack1_prob_11(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 4;
        else if (n != 4) error, "N must be 4 for problem #11";
        x = array(double, n);
        x(1) = 25.0;
        x(2) = 5.0;
        x(3) = -5.0;
        x(4) = -1.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        f = 0.0;
        for (i = 1; i <= 20; ++i) {
            d1 = double(i)/5.0;
            d2 = sin(d1);
            t1 = x(1) + d1*x(2) - exp(d1);
            t2 = x(3) + d2*x(4) - cos(d1);
            t = t1*t1 + t2*t2;
            f += t*t;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        g(1) = 0.0;
        g(2) = 0.0;
        g(3) = 0.0;
        g(4) = 0.0;
        for (i = 1; i <= 20; ++i) {
            d1 = double(i)/5.0;
            d2 = sin(d1);
            t1 = x(1) + d1*x(2) - exp(d1);
            t2 = x(3) + d2*x(4) - cos(d1);
            t = t1*t1 + t2*t2;
            s1 = t1*t;
            s2 = t2*t;
            g(1) = g(1) + s1;
            g(2) = g(2) + d1*s1;
            g(3) = g(3) + s2;
            g(4) = g(4) + d2*s2;
        }
        g(1) = 4.0*g(1);
        g(2) = 4.0*g(2);
        g(3) = 4.0*g(3);
        g(4) = 4.0*g(4);
        return g;
    } else {
        // Problem name.
        return "Brown and Dennis function";
    }
}

// Gulf research and development function.
func optm_minpack1_prob_12(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 3;
        else if (n != 3) error, "N must be 3 for problem #12";
        x = array(double, n);
        x(1) = 5.0;
        x(2) = 2.5;
        x(3) = 0.15;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        f = 0.0;
        d1 = 2.0/3.0;
        for (i = 1; i <= 99; ++i) {
            arg = double(i)/100.0;
            r = (-50.0*log(arg))^d1 + 25.0 - x(2);
            t1 = (abs(r)^x(3))/x(1);
            t2 = exp(-t1);
            t = t2 - arg;
            f += t*t;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        g(1) = 0.0;
        g(2) = 0.0;
        g(3) = 0.0;
        d1 = 2.0/3.0;
        for (i = 1; i <= 99; ++i) {
            arg = double(i)/100.0;
            r = (-50.0*log(arg))^d1 + 25.0 - x(2);
            t1 = (abs(r)^x(3))/x(1);
            t2 = exp(-t1);
            t = t2 - arg;
            s1 = t1*t2*t;
            g(1) = g(1) + s1;
            g(2) = g(2) + s1/r;
            g(3) = g(3) - s1*log(abs(r));
        }
        g(1) = 2.0*g(1)/x(1);
        g(2) = 2.0*x(3)*g(2);
        g(3) = 2.0*g(3);
        return g;
    } else {
        // Problem name.
        return "Gulf research and development function";
    }
}

// Trigonometric function.
func optm_minpack1_prob_13(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 10;
        else if (n < 1) error, "N must be >= 1 in problem #13";
        x = array(double, n);
        h = 1.0/double(n);
        for (j = 1; j <= n; ++j) x(j) = h;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        s1 = 0.0;
        for (j = 1; j <= n; ++j) {
            s1 += cos(x(j));
        }
        f = 0.0;
        for (j = 1; j <= n; ++j) {
            t = double(n+j) - sin(x(j)) - s1 - double(j)*cos(x(j));
            f += t*t;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        s1 = 0.0;
        for (j = 1; j <= n; ++j) {
            g(j) = cos(x(j));
            s1 += g(j);
        }
        s2 = 0.0;
        for (j = 1; j <= n; ++j) {
            th = sin(x(j));
            t = double(n+j) - th - s1 - double(j)*g(j);
            s2 += t;
            g(j) = (double(j)*th - g(j))*t;
        }
        for (j = 1; j <= n; ++j) {
            g(j) = 2.0*(g(j) + sin(x(j))*s2);
        }
        return g;
    } else {
        // Problem name.
        return "Trigonometric function";
    }
}

// Extended Rosenbrock function.
func optm_minpack1_prob_14(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 10;
        else if (n < 1 || n%2 != 0)
            error, "N must be a multiple of 2 in problem #14";
        x = array(double, n);
        for (j = 1; j <= n; j += 2) {
            x(j) = -1.2;
            x(j+1) = 1.0;
        }
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        f = 0.0;
        for (j = 1; j <= n; j += 2) {
            t1 = 1.0 - x(j);
            t2 = 10.0*(x(j+1) - x(j)*x(j));
            f += t1*t1 + t2*t2;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        for (j = 1; j <= n; j += 2) {
            t1 = 1.0 - x(j);
            g(j+1) = 200.0*(x(j+1) - x(j)*x(j));
            g(j) = -2.0*(x(j)*g(j+1) + t1);
        }
        return g;
    } else {
        // Problem name.
        return "Extended Rosenbrock function";
    }
}

// Extended Powell function.
func optm_minpack1_prob_15(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 12;
        else if (n < 1 || n%4 != 0)
            error, "N must be a multiple of 4 in problem #15";
        x = array(double, n);
        for (j = 1; j <= n; j += 4) {
            x(j) = 3.0;
            x(j+1) = -1.0;
            x(j+2) = 0.0;
            x(j+3) = 1.0;
        }
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        f = 0.0;
        for (j = 1; j <= n; j += 4) {
            t = x(j) + 10.0*x(j+1);
            t1 = x(j+2) - x(j+3);
            s1 = 5.0*t1;
            t2 = x(j+1) - 2.0*x(j+2);
            s2 = t2*t2*t2;
            t3 = x(j) - x(j+3);
            s3 = 10.0*t3*t3*t3;
            f += t*t + s1*t1 + s2*t2 + s3*t3;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        for (j = 1; j <= n; j += 4) {
            t = x(j) + 10.0*x(j+1);
            t1 = x(j+2) - x(j+3);
            s1 = 5.0*t1;
            t2 = x(j+1) - 2.0*x(j+2);
            s2 = 4.0*t2*t2*t2;
            t3 = x(j) - x(j+3);
            s3 = 20.0*t3*t3*t3;
            g(j) = 2.0*(t + s3);
            g(j+1) = 20.0*t + s2;
            g(j+2) = 2.0*(s1 - s2);
            g(j+3) = -2.0*(s1 + s3);
        }
        return g;
    } else {
        // Problem name.
        return "Extended Powell function";
    }
}

// Beale function.
func optm_minpack1_prob_16(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 2;
        else if (n != 2) error, "N must be 2 for problem #16";
        x = array(double, n);
        x(1) = 1.0;
        x(2) = 1.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        s1 = 1.0 - x(2);
        t1 = 1.5 - x(1)*s1;
        s2 = 1.0 - x(2)*x(2);
        t2 = 2.25 - x(1)*s2;
        s3 = 1.0 - x(2)*x(2)*x(2);
        t3 = 2.625 - x(1)*s3;
        f = t1*t1 + t2*t2 + t3*t3;
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        s1 = 1.0 - x(2);
        t1 = 1.5 - x(1)*s1;
        s2 = 1.0 - x(2)*x(2);
        t2 = 2.25 - x(1)*s2;
        s3 = 1.0 - x(2)*x(2)*x(2);
        t3 = 2.625 - x(1)*s3;
        g(1) = -2.0*(s1*t1 + s2*t2 + s3*t3);
        g(2) = 2.0*x(1)*(t1 + x(2)*(2.0*t2 + 3.0*x(2)*t3));
        return g;
    } else {
        // Problem name.
        return "Beale function";
    }
}

// Wood function
func optm_minpack1_prob_17(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 4;
        else if (n != 4) error, "N must be 4 for problem #17";
        x = array(double, n);
        x(1) = -3.0;
        x(2) = -1.0;
        x(3) = -3.0;
        x(4) = -1.0;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        s1 = x(2) - x(1)*x(1);
        s2 = 1.0 - x(1);
        s3 = x(2) - 1.0;
        t1 = x(4) - x(3)*x(3);
        t2 = 1.0 - x(3);
        t3 = x(4) - 1.0;
        f = 100.0*s1*s1 + s2*s2 + 90.0*t1*t1 + t2*t2 + \
            10.0*(s3 + t3)*(s3 + t3) + (s3 - t3)*(s3 - t3)/10.0;
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        s1 = x(2) - x(1)*x(1);
        s2 = 1.0 - x(1);
        s3 = x(2) - 1.0;
        t1 = x(4) - x(3)*x(3);
        t2 = 1.0 - x(3);
        t3 = x(4) - 1.0;
        g(1) = -2.0*(200.0*x(1)*s1 + s2);
        g(2) = 200.0*s1 + 20.2*s3 + 19.8*t3;
        g(3) = -2.0*(180.0*x(3)*t1 + t2);
        g(4) = 180.0*t1 + 20.2*t3 + 19.8*s3;
        return g;
    } else {
        // Problem name.
        return "Wood function";
    }
}

// Chebyquad function
func optm_minpack1_prob_18(job, x, n=, factor=)
{
    if (job == 0) {
        // Starting point.
        if (is_void(n)) n = 25;
        else if (n < 1 || n > 50) error, "N must be <= 50 for problem #18";
        x = array(double, n);
        h = 1.0/double(n+1);
        for (j = 1; j <= n; ++j) x(j) = double(j)*h;
        return is_void(factor) ? x : factor*x;
    } else if (job == 1) {
        // Objective function.
        n = numberof(x);
        fvec = array(0.0, n);
        for (j = 1; j <= n; ++j) {
            t1 = 1.0;
            t2 = 2.0*x(j) - 1.0;
            t = 2.0*t2;
            for (i = 1; i <= n; ++i) {
                fvec(i) += t2;
                th = t*t2 - t1;
                t1 = t2;
                t2 = th;
            }
        }
        f = 0.0;
        d1 = 1.0/double(n);
        iev = -1;
        for (i = 1; i <= n; ++i) {
            t = d1*fvec(i);
            if (iev > 0) t += 1.0/(i*i - 1.0);
            f += t*t;
            iev = -iev;
        }
        return f;
    } else if (job == 2) {
        // Gradient.
        n = numberof(x);
        g = array(double, n);
        fvec = array(0.0, n);
        for (j = 1; j <= n; ++j) {
            t1 = 1.0;
            t2 = 2.0*x(j) - 1.0;
            t = 2.0*t2;
            for (i = 1; i <= n; ++i) {
                fvec(i) += t2;
                th = t*t2 - t1;
                t1 = t2;
                t2 = th;
            }
        }
        d1 = 1.0/double(n);
        iev = -1;
        for (i = 1; i <= n; ++i) {
            fvec(i) *= d1;
            if (iev > 0) fvec(i) += 1.0/(i*i - 1.0);
            iev = -iev;
        }
        for (j = 1; j <= n; ++j) {
            g(j) = 0.0;
            t1 = 1.0;
            t2 = 2.0*x(j) - 1.0;
            t = 2.0*t2;
            s1 = 0.0;
            s2 = 2.0;
            for (i = 1; i <= n; ++i) {
                g(j) = g(j) + fvec(i)*s2;
                th = 4.0*t2 + t*s2 - s1;
                s1 = s2;
                s2 = th;
                th = t*t2 - t1;
                t1 = t2;
                t2 = th;
            }
        }
        d2 = 2.0*d1;
        for (j = 1; j <= n; ++j) g(j) *= d2;
        return g;
    } else {
        // Problem name.
        return "Chebyquad function";
    }
}

if (batch()) {
    // Run all tests.
    include, dirname(current_include()) + "/optm.i", 1;
    optm_minpack1_test, verb=1, gtol=0, xtol=0, ftol=0, fmin=0, mem="max";
}
