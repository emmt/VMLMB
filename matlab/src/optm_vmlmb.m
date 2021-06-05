%% optm_vmlmb - Apply VMLMB algorithm.
%%
%%     [x, f, g, status] = optm_vmlmb(fg, x, ...);
%%
%% runs VMLMB (for Variable Metric Limited Memory with Bounds) algorithm
%% to minimize a given smooth objective function, possibly with separable
%% bounds constraints.
function [x, f, g, status] = optm_vmlmb(fg, x, varargin)
    if nargin < 2
        print_usage;
    end

    %% Parse settings.
    lower = [];
    upper = [];
    mem = 5;
    maxiter = Inf;
    maxeval = Inf;
    ftol = 1e-8;
    gtol = 1e-5;
    xtol = 1e-6;
    lnsrch = [];
    verbose = false;
    fmin = NaN;
    delta = NaN;
    epsilon = 0.0;
    lambda = NaN;
    blmvm = false;
    if mod(length(varargin), 2) != 0
        error("parameters must be specified as pairs of names and values");
    end
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        switch key
            case "lower"
                lower = val;
            case "upper"
                upper = val;
            case "mem"
                mem = val;
            case "maxiter"
                maxiter = val;
            case "maxeval"
                maxeval = val;
            case "ftol"
                ftol = val;
            case "gtol"
                gtol = val;
            case "xtol"
                xtol = val;
            case "lnsrch"
                lnsrch = val;
            case "verbose"
                verbose = val;
            case "fmin"
                fmin = val;
            case "delta"
                delta = val;
            case "epsilon"
                epsilon = val;
            case "lambda"
                lambda = val;
            case "blmvm"
                blmvm = (val != 0);
            otherwise
                error("invalid parameter name '%s'", key);
        end
    end

    %% If the caller does not retrieve the status argument, failures are
    %% reported by throwing an error.
    throwerrors = (nargout < 4);
    status = 0;

    %% Other initialization.
    if isempty(lnsrch)
        lnsrch = optm_new_line_search();
    end
    if ischar(fg)
        fg = str2func(fg);
    end
    lbfgs = optm_new_lbfgs(mem);
    bounded = (~isempty(lower) || ~isempty(upper));
    if ~bounded
        %% No needs to use BLMVM trick in the inconstrained case.
        blmvm = false;
    end
    alpha = 0.0;   % step size
    evals = 0;     % number of calls to fg
    iters = 0;     % number of iterations
    projs = 0;     % number of projections onto the feasible set
    status = 0;    % algorithm status is zero until termination.
    freevars = []; % subset of free variables not yet known
    print_now = false;
    if verbose
        t0 = time;
    end

    %% Algorithm stage is one of:
    %% 0 = initial;
    %% 1 = first trial step in line-search;
    %% 2 = second and subsequent trial steps in line-search;
    %% 3 = line-search has converged.
    stage = 0;

    while true
        if bounded && stage < 2
            %% Make the variables feasible.
            x = optm_clamp(x, lower, upper);
            projs += 1;
        end
        %% Compute objective function and its gradient.
        [f, g] = fg(x);
        evals += 1;
        if evals == 1 || f < best_f
            %% Save best solution so far.
            best_f = f;
            best_g = g;
            best_x = x;
        end
        if stage == 1
            %% First trial along search direction.
            d = x - x0; % effective step
            dg0 = optm_inner(d, g0);
            alpha = 1.0;
            lnsrch = optm_start_line_search(lnsrch, f0, dg0, alpha);
            if lnsrch.stage != 1
                error("something is wrong!");
            end
            stage = 2;
        end
        if stage == 2
            %% Check for line-search convergence.
            lnsrch = optm_iterate_line_search(lnsrch, f);
            if lnsrch.stage == 2
                %% Line-search has converged, `x` is the next iterate.
                stage = 3;
                iters += 1;
            elseif lnsrch.stage == 1
                alpha = lnsrch.step;
            else
                error("something is wrong!");
            end
        end
        if stage == 3 || stage == 0
            %% Initial or next iterate after convergence of line-search.
            if bounded
                %% Determine the subset of free variables and compute the norm
                %% of the projected gradient (needed to check for convergence).
                freevars = optm_freevars(x, lower, upper, g);
                if ~any(freevars(:))
                    %% Variables are all blocked.
                    status = optm_status("XTEST_SATISFIED");
                    gnorm = 0.0;
                else
                    pg = freevars.*g;
                    gnorm = optm_norm2(pg);
                    if ~blmvm
                        %% Projected gradient no longer needed, free some
                        %% memory.
                        pg = [];
                    end
                end
            else
                %% Just compute the norm of the gradient.
                gnorm = optm_norm2(g);
            end
            %% Check for algorithm convergence or termination.
            if evals == 1
                %% Compute value for testing the convergence in the gradient.
                gtest = optm_tolerance(gnorm, gtol);
            end
            if status == 0 && gnorm <= gtest
                %% Convergence in gradient
                status = optm_status("GTEST_SATISFIED");
            end
            if stage == 3
                if status == 0
                    %% Check convergence in relative function reduction.
                    if f == f0 || (ftol > 0 && abs(f - f0) <= ftol*max(abs(f), abs(f0)))
                        status = optm_status("FTEST_SATISFIED");
                    end
                end
                if alpha != 1.0
                    d = x - x0; % recompute effective step
                end
                if status == 0
                    %% Check convergence in variables.
                    dnorm = optm_norm2(d);
                    if dnorm <= 0 || (xtol > 0 && dnorm <= xtol*optm_norm2(x))
                        status = optm_status("XTEST_SATISFIED");
                    end
                end
            end
            if status == 0 && iters >= maxiter
                status = optm_status("TOO_MANY_ITERATIONS");
            end
            print_now = verbose;
        end
        if status == 0 && evals >= maxeval
            status = optm_status("TOO_MANY_EVALUATIONS");
        end
        if verbose && status != 0 && ~print_now && best_f < f0
            %% Verbose mode and abnormal termination but some progress have
            %% been made since the start of the line-search.  Restore best
            %% solution so far, pretend that one more iteration has been
            %% performed and manage to print information about this iteration.
            f = best_f;
            g = best_g;
            x = best_x;
            if bounded
                gnorm = optm_norm2(freevars.*g);
            end
            iters += 1;
            print_now = verbose;
        end
        if print_now
            t = (time - t0)*1E3; % elapsed milliseconds
            if iters < 1
                printf("%s%s\n%s%s\n", ...
                       "# Iter.   Time (ms)   Eval.   Proj. ", ...
                       "       Obj. Func.           Grad.       Step", ...
                       "# ----------------------------------", ...
                       "-----------------------------------------------");
            end
            printf("%7d %11.3f %7d %7d %23.15e %11.3e %11.3e\n", ...
                   iters, t, evals, projs, f, gnorm, alpha);
            print_now = ~print_now;
        end
        if status != 0
            %% Algorithm stops here.
            break
        end
        if stage == 3
            %% Line-search has converged, L-BFGS approximation can be updated.
            %% FIXME: if alpha = 1, then d = x - x0;
            if blmvm
                lbfgs = optm_update_lbfgs(lbfgs, x - x0, pg - pg0);
            else
                lbfgs = optm_update_lbfgs(lbfgs, x - x0, g - g0);
            end
        end
        if stage == 3 || stage == 0
            %% Save iterate and determine a new search direction.
            f0 = f;
            g0 = g;
            x0 = x;
            if blmvm
                pg0 = pg;
            end
            if lbfgs.mp > 0
                %% Use L-BFGS approximation to compute a search direction and
                %% check that it is an acceptable descent direction.
                [alpha, d] = optm_apply_lbfgs(lbfgs, -g, freevars);
                if alpha > 0
                    %% Some valid (s,y) pairs were available to apply the
                    %% L-BFGS approximation.
                    dg = optm_inner(d, g);
                    if dg >= 0
                        %% L-BFGS approximation does not yield a descent
                        %% direction.
                        if ~bounded
                            if throwerrors
                                error("L-BFGS approximation is not positive definite");
                            end
                            status = optm_status("NOT_POSITIVE_DEFINITE");
                            break
                        end
                        alpha = 0;
                    elseif epsilon > 0
                        %% A more restrictive criterion has been specified for
                        %% accepting a descent direction.
                        dnorm = optm_norm2(d);
                        if dg > -epsilon*dnorm*gnorm
                            alpha = 0;
                        end
                    end
                    if alpha <= 0
                        %% The direction computed using the L-BFGS
                        %% approximation failed to be a sufficient descent
                        %% direction.  Take the steepest feasible descent
                        %% direction.
                        d = -g;
                    end
                end
            else
                %% No L-BFGS approximation is available yet, will take the
                %% steepest feasible descent direction.
                d = -g;
                alpha = 0;
            end
            if alpha <= 0
                %% Find a suitable step size along the steepest feasible
                %% descent direction `d`.  Note that gnorm, the Euclidean norm
                %% of the (projected) gradient, is also that of `d`.
                alpha = optm_steepest_descent_step(x, gnorm, f, fmin, ...
                                                   delta, lambda);
            end
            stage = 1; % first trial along search direction
            if bounded
                %% Safeguard the step to avoid searching in a region where
                %% all bounds are overreached.
                [amin, amax] = optm_line_search_limits(x0, lower, upper, ...
                                                       d, alpha);
                alpha = min(alpha, amax);
            end
        end
        %% Compute next iterate.
        if alpha == 1
            x = x0 + d;
        else
            x = x0 + alpha*d;
        end
    end
    if best_f < f
        %% Restore best solution so far.
        f = best_f;
        g = best_g;
        x = best_x;
    end
end
