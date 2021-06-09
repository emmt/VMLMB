%% test_lnsrch.m -
%%
%% Test line-search.
%%-----------------------------------------------------------------------------

global optm_assertions optm_failures;
addpath('../src');

function optm_assert(expr, mesg)
    global optm_assertions optm_failures;
    if ~isscalar(optm_assertions)
        optm_assertions = 1;
    else
        optm_assertions += 1;
    end
    if ~isscalar(optm_failures)
        optm_failures = 0;
    end
    if ~expr
        printf('assertion failed: %s\n', mesg);
        optm_failures += 1;
    end
end

function optm_summarize_tests(reset)
    global optm_assertions optm_failures;
    printf('%d failure(s) / %d test(s)\n', optm_failures, optm_assertions);
    if nargin >= 1 && reset
        optm_failures = 0;
        optm_assertions = 0;
    end
end

function optm_test_line_search(f, df, stp, lnsrch, quadratic)

    if nargin < 4
        lnsrch = optm_new_line_search();
    end
    if nargin < 5
        quadratic = false;
    end
    if ischar(f)
        f = str2func(f);
    end
    if ischar(df)
        df = str2func(df);
    end
    f0 = f(0);
    df0 = df(0);
    fmin = f1 = f0;
    nevals = 1;
    lnsrch = optm_start_line_search(lnsrch, f0, df0, stp);
    optm_assert(lnsrch.step == stp, 'lnsrch.step == stp');
    optm_assert(lnsrch.stage == 1, 'lnsrch.stage == 1');
    optm_assert(lnsrch.finit == f0, 'lnsrch.finit == f0');
    optm_assert(lnsrch.ginit == df0, 'lnsrch.dfinit == df0');
    while (lnsrch.stage == 1)
        stp = lnsrch.step;
        f1 = f(stp);
        ++nevals;
        fmin = min(fmin, f1);
        lnsrch = optm_iterate_line_search(lnsrch, f(stp));
        converged = (f1 <= f0 + lnsrch.ftol*stp*df0);
        if converged
            stage = 2;
        else
            stage = 1;
        end
        optm_assert(lnsrch.stage == stage, 'lnsrch.stage == stage');
        if (lnsrch.stage == 1)
            optm_assert(lnsrch.step < stp, 'lnsrch.step < stp');
            optm_assert(lnsrch.step > 0, 'lnsrch.step > 0');
            optm_assert(lnsrch.step >= lnsrch.smin*stp, 'lnsrch.step >= lnsrch.smin*stp');
            optm_assert(lnsrch.step <= lnsrch.smax*stp, 'lnsrch.step <= lnsrch.smax*stp');

        end
    end
    optm_assert(lnsrch.stage == 2, 'lnsrch.stage == 2');
    optm_assert(!quadratic || nevals <= 3, ' !quadratic || nevals <= 3');
    optm_assert(fmin <= f0 + lnsrch.ftol*lnsrch.step*df0, 'fmin <= f0 + lnsrch.ftol*lnsrch.stp*df0');
end

%% A quadratic convex function.
function f   = optm_test_f1(a);        f = 46.0 - 3.0*a + 0.5*a*a; end
function df  = optm_test_f1_prime(a); df = a - 3.0; end
function stp = optm_test_f1_step();  stp = 5.0; end
optm_test_line_search(@optm_test_f1, @optm_test_f1_prime, optm_test_f1_step(), ...
                      optm_new_line_search(), true);

%% A non-quadratic function.
function f   = optm_test_f2(a);        f = -sin((0.5*a + 1.0)*a); end
function df  = optm_test_f2_prime(a); df = -cos((0.5*a + 1.0)*a)*(1.0 + a); end
function stp = optm_test_f2_step();  stp = 1.7; end
optm_test_line_search(@optm_test_f2, @optm_test_f2_prime, optm_test_f2_step());

optm_summarize_tests(true);
