// optm_tests.i -
//
// Various tests for multi-dimensional optimization in Yorick.
//-----------------------------------------------------------------------------

local optm_assertions, optm_failures;
func optm_assert(expr, mesg)
{
    if (is_void(optm_assertions)) {
        optm_assertions = 1;
    } else {
        ++optm_assertions;
    }
    if (is_void(optm_failures)) {
        optm_failures = 0;
    }
    if (!expr) {
        write, format = "assertion failed: %s\n", mesg;
        ++optm_failures;
    }
}

func optm_summarize_tests(nil, reset=)
{
    write, format="%d failure(s) / %d test(s)\n", optm_failures, optm_assertions;
    if (reset) {
        optm_failures = 0;
        optm_assertions = 0;
    }
}

func optm_test_line_search(f, df, stp, lnsrch=, quadratic=)
{
    if (is_void(lnsrch)) lnsrch = optm_new_line_search();
    if (is_void(f) && is_void(df) && is_void(stdp)) {
        f = optm_test_f1;
        df = optm_test_f1prime;
        stp = 1.7;
    }
    f0 = f(0);
    df0 = df(0);
    fmin = f1 = f0;
    nevals = 1;
    optm_start_line_search, lnsrch, f0, df0, stp;
    optm_assert, lnsrch.step == stp, "lnsrch.step == stp";
    optm_assert, lnsrch.state == 1, "lnsrch.state == 1";
    optm_assert, lnsrch.finit == f0, "lnsrch.finit == f0";
    optm_assert, lnsrch.ginit == df0, "lnsrch.dfinit == df0";
    while (lnsrch.state == 1) {
        stp = lnsrch.step;
        f1 = f(stp);
        ++nevals;
        fmin = min(fmin, f1);
        optm_iterate_line_search, lnsrch, f(stp);
        converged = (f1 <= f0 + lnsrch.ftol*stp*df0);
        optm_assert, lnsrch.state == (converged ? 2 : 1), "lnsrch.state == (converged ? 2 : 1)";
        if (lnsrch.state == 1) {
            optm_assert, lnsrch.step < stp, "lnsrch.step < stp";
            optm_assert, lnsrch.step > 0, "lnsrch.step > 0";
            optm_assert, lnsrch.step >= lnsrch.smin*stp, "lnsrch.step >= lnsrch.smin*stp";
            optm_assert, lnsrch.step <= lnsrch.smax*stp, "lnsrch.step <= lnsrch.smax*stp";

        }
    }
    optm_assert, lnsrch.state == 2, "lnsrch.state == 2";
    optm_assert, !quadratic || nevals <= 3, " !quadratic || nevals <= 3";
    optm_assert, fmin <= f0 + lnsrch.ftol*lnsrch.step*df0, "fmin <= f0 + lnsrch.ftol*lnsrch.stp*df0";
}

// A quadratic convex function.
func optm_test_f1(a) { return 46.0 - 3.0*a + 0.5*a*a; }
func optm_test_f1_prime(a) { return a - 3.0; }
func optm_test_f1_step(nil) { return 5.0; }
optm_test_line_search, optm_test_f1, optm_test_f1_prime, optm_test_f1_step(), quadratic=1;

// A non-quadratic function.
func optm_test_f2(a) { return -sin((0.5*a + 1.0)*a); }
func optm_test_f2_prime(a) { return -cos((0.5*a + 1.0)*a)*(1.0 + a); }
func optm_test_f2_step(nil) { return 1.7; }
optm_test_line_search, optm_test_f2, optm_test_f2_prime, optm_test_f2_step();

optm_summarize_tests, reset=1;
