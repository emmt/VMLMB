%%     [x, f, g, status] = optm_vmlmb(fg, x0, 'key', val, ...);
%%
%% Apply VMLMB algorithm to minimize a multi-variate differentiable objective
%% function possibly under separable bound constraints.  VMLMB is a
%% quasi-Newton method ("VM" is for "Variable Metric") with low memory
%% requirements ("LM" is for "Limited Memory") and which can optionally take
%% into account separable bound constraints (the final "B") on the variables.
%% To determine efficient search directions, VMLMB approximates the Hessian of
%% the objective function by a limited memory version of the model assumed in
%% Broyden-Fletcher-Goldfarb-Shanno algorithm (called L-BFGS for short).  Hence
%% VMLMB is well suited to solving optimization problems with a very large
%% number of variables possibly with bound constraints.
%%
%% The method has two required arguments: `fg` the function to call to compute
%% the objective function and its gradient and `x0` the initial variables
%% (VMLMB is an iterative method).  The initial variables may be an array of
%% any dimensions.
%%
%% The method returns `x` the best solution found during iterations, `f` and
%% `g` the value and the gradient of the objective at `x` and `status` an
%% integer code indicating the reason of the termination of the algorithm (see
%% `optm_reason`).
%%
%% The function `fg` shall be implemented as follows:
%%
%%     function [f, g] = fg(x)
%%         f = ...; % value of the objective function at `x`
%%         g = ...: % gradient of the objective function at `x`
%%     end
%%
%% All other settings are specified by keyword names followed by their value
%% (e.g.. `'key',val`), possible keywords are listed below.
%%
%% - Keywords `upper` and `lower` are to specify a lower and/or an upper bounds
%%   for the variables.  If unspecified or set to an empty array, a given bound
%%   is considered as unlimited.  Bounds must be conformable with the
%%   variables.
%%
%% - Keyword `mem` specifies the memory used by the algorithm, that is the
%%   number of previous steps memorized to approximate the Hessian of the
%%   objective function.  With `mem=0`, the algorithm behaves as a steepest
%%   descent method.  The default is `mem=5`.
%%
%% - Keywords `ftol`, `gtol` and `xtol` specify tolerances for deciding the
%%   convergence of the algorithm.
%%
%%   Convergence in the function occurs if one of the following conditions
%%   hold:
%%
%%       f ≤ fatol
%%       |f - fp| ≤ frtol⋅max(|f|, |fp|)
%%
%%   where `f` and `fp` are the values of the objective function at the current
%%   and previous iterates.  In these conditions, `fatol` and `frtol` are
%%   absolute and relative tolerances specified by `ftol` which can be
%%   `ftol=[fatol,frtol]` or `ftol=frtol` and assume that `fatol=-Inf`.  The
%%   default is `ftol=1e-8`.
%%
%%   Convergence in the gradient occurs if the following condition holds:
%%
%%       ‖g‖ ≤ max(0, gatol, grtol⋅‖g0‖)
%%
%%   where `‖g‖` is the Euclidean norm of the projected gradient, `g0` is the
%%   projected gradient at the initial solution.  In this condition, `gatol`
%%   and `grtol` are absolute and relative gradient tolerances specified by
%%   `gtol` which can be `gtol=[gatol,grtol]` or `gtol=grtol` and assume that
%%   `gatol=0`.  The default is `gtol=1e-5`.
%%
%%   Convergence in the variables occurs if the following condition holds:
%%
%%       ‖x - xp‖ ≤ max(0, xatol, xrtol*‖x‖)
%%
%%   where `x` and `xp` are the current and previous variables.  In this
%%   condition, `xatol` and `xrtol` are absolute and relative tolerances
%%   specified by `xtol` which can be `xtol=[fatol,frtol]` or `xtol=xrtol` and
%%   assume that `xatol=0`.  The default is `xtol=1e-6`.
%%
%% - Keywords `maxiter` and `maxeval` are to specify a maximum number of
%%   algorithm iterations or or evaluations of the objective function
%%   implemented by `fg`.  By default, these are unlimited.
%%
%% - Keyword `lnsrch` is to specify line-search settings different than the
%%   default (see `optm_new_line_search`).
%%
%% - Keyword `fmin` is to specify an estimation of the minimum possible value
%%   of the objective function.  This setting may be used to determine the step
%%   length along the steepest descent.
%%
%% - Keyword `delta` specifies a small size relative to the variables.  This
%%   setting may be used to determine the step length along the steepest
%%   descent.
%%
%% - Keyword `lambda` specifies an estimate of the magnitude of the eigenvalues
%%   of the Hessian of the objective function.  This setting may be used to
%%   determine the step length along the steepest descent.
%%
%% - Keyword `epsilon` specifies a threshold for a sufficient descent
%%   condition.  If `epsilon > 0`, then a search direction `d` computed by the
%%   L-BFGS approximation is considered as acceptable if:
%%
%%       ⟨d,g⟩ ≤ -epsilon⋅‖d‖⋅‖g‖
%%
%%   where `g` denotes the projected gradient of the objective function (which
%%   is just the gradient in unconstrained case).  Otherwise, the condition
%%   writes `⟨d,g⟩ < 0`.  The default is `epsilon = 0` so only the latter
%%   condition is checked.
%%
%% - Keyword `blmvm` (false by default) specifies whether to use BLMVM trick to
%%   account for the bound constraints in the L-BFGS model of the Hessian.  If
%%   `blmvm` is set true, the overhead of the algorithm may be reduced, but the
%%   L-BFGS model of the Hessian is more likely to be inaccurate causing the
%%   algorithm to choose the steepest descent direction more often.
%%
%%  - Keyword `verb`, if positive, specifies to print information every `verb`
%%    iterations.  Nothing is printed if `verb ≤ 0`.  By default, `verb = 0`.

function [x, f, g, status] = optm_vmlmb(fg, x, varargin)
    if nargin < 2
        print_usage;
    end

    %% Constants.  Calling inf, nan, true or false takes too much time (2.1µs
    %% instead of 0.2µs if stored in a variable), so use local variables to pay
    %% the price only once.
    INF = inf();
    NAN = nan();
    TRUE = true();
    FALSE = false();

    %% Parse settings.
    lower = [];
    upper = [];
    mem = 5;
    maxiter = INF;
    maxeval = INF;
    ftol = 1e-8;
    gtol = 1e-5;
    xtol = 1e-6;
    lnsrch = [];
    verb = 0;
    fmin = NAN;
    delta = NAN;
    epsilon = 0.0;
    lambda = NAN;
    blmvm = FALSE;
    if mod(length(varargin), 2) ~= 0
        error('parameters must be specified as pairs of names and values');
    end
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        switch key
            case 'lower'
                lower = val;
            case 'upper'
                upper = val;
            case 'mem'
                mem = val;
            case 'maxiter'
                maxiter = val;
            case 'maxeval'
                maxeval = val;
            case 'ftol'
                ftol = val;
            case 'gtol'
                gtol = val;
            case 'xtol'
                xtol = val;
            case 'lnsrch'
                lnsrch = val;
            case 'verb'
                verb = val;
            case 'fmin'
                fmin = val;
            case 'delta'
                delta = val;
            case 'epsilon'
                epsilon = val;
            case 'lambda'
                lambda = val;
            case 'blmvm'
                blmvm = (val ~= 0);
            otherwise
                error('invalid parameter name `%s`', key);
        end
    end

    %% Tolerances.  Most of these are forced to be nonnegative to simplify
    %% tests.
    if isscalar(ftol)
        fatol = -INF;
        frtol = max(0.0, ftol);
    else
        fatol = max(0.0, ftol(1));
        frtol = max(0.0, ftol(2));
    end
    if isscalar(gtol)
        gatol = 0.0;
        grtol = max(0.0, gtol);
    else
        gatol = max(0.0, gtol(1));
        grtol = max(0.0, gtol(2));
    end
    if isscalar(xtol)
        xatol = 0.0;
        xrtol = max(0.0, xtol);
    else
        xatol = max(0.0, xtol(1));
        xrtol = max(0.0, xtol(2));
    end

    %% Bound constraints.  For faster code, unlimited bounds are preferentially
    %% represented by empty arrays.
    if ~isempty(lower) && all(lower(:) == -INF)
        lower = [];
    end
    if ~isempty(upper) && all(upper(:) == +INF)
        upper = [];
    end
    bounded = (~isempty(lower) || ~isempty(upper));
    if ~bounded
        blmvm = FALSE; % no needs to use BLMVM trick in the unconstrained case
    end

    %% If the caller does not retrieve the status argument, failures are
    %% reported by throwing an error.
    throwerrors = (nargout < 4);

    %% Other initialization.
    g = [];          % gradient
    f0 = +INF;       % function value at start of line-search
    g0 = [];         % gradient at start of line-search
    d = [];          % search direction
    s = [];          % effective step
    pg = [];         % projected gradient
    pg0 = [];        % projected gradient at start of line search
    pgnorm = 0.0;    % Euclidean norm of the (projected) gradient
    alpha = 0.0;     % step length
    amin = -INF;     % first step length threshold
    amax = +INF;     % last step length threshold
    evals = 0;       % number of calls to `fg`
    iters = 0;       % number of iterations
    projs = 0;       % number of projections onto the feasible set
    rejects = 0;     % number of search direction rejections
    status = 0;      % non-zero when algorithm is about to terminate
    best_f = +INF;   % function value at `best_x`
    best_g = [];     % gradient at `best_x`
    best_x = [];     % best solution found so far
    best_pgnorm = -1;% norm of projected gradient at `best_x` (< 0 if unknown)
    best_alpha =  0; % step length at `best_x` (< 0 if unknown)
    best_evals = -1; % number of calls to `fg` at `best_x`
    last_evals = -1; % number of calls to `fg` at last iterate
    last_print = -1; % iteration number for last print
    freevars = [];   % subset of free variables (not yet known)
    if isempty(lnsrch)
        lnsrch = optm_new_line_search();
    end
    if ischar(fg)
        fg = str2func(fg);
    end
    lbfgs = optm_new_lbfgs(mem);
    if verb > 0
        time = @() 86400E3*now(); % yields number of milliseconds
        t0 = time();
    end

    %% Algorithm stage follows that of the line-search, it is one of:
    %% 0 = initially;
    %% 1 = line-search in progress;
    %% 2 = line-search has converged.
    stage = 0;

    while TRUE
        %% Make the variables feasible.
        if bounded
            %% In principle, we can avoid projecting the variables whenever
            %% `alpha ≤ amin` (because the feasible set is convex) but rounding
            %% errors could make this wrong.  It is safer to always project the
            %% variables.  This cost O(n) operations which are probably
            %% negligible compared to, say, computing the objective function
            %% and its gradient.
            x = optm_clamp(x, lower, upper);
            projs = projs + 1;
        end
        %% Compute objective function and its gradient.
        [f, g] = fg(x);
        evals = evals + 1;
        if f < best_f || evals == 1
            %% Save best solution so far.
            best_f = f;
            best_g = g;
            best_x = x;
            best_pgnorm = -1; % must be recomputed
            best_alpha = alpha;
            best_evals = evals;
        end
        if stage ~= 0
            %% Line-search in progress, check for line-search convergence.
            lnsrch = optm_iterate_line_search(lnsrch, f);
            stage = lnsrch.stage;
            if stage == 2
                %% Line-search has converged, `x` is the next iterate.
                iters = iters + 1;
                last_evals = evals;
            elseif stage == 1
                %% Line-search has not converged, peek next trial step.
                alpha = lnsrch.step;
            else
                error('something is wrong!');
            end
        end
        if stage ~= 1
            %% Initial or next iterate after convergence of line-search.
            if bounded
                %% Determine the subset of free variables and compute the norm
                %% of the projected gradient (needed to check for convergence).
                freevars = optm_unblocked_variables(x, lower, upper, g);
                pg = freevars .* g;
                pgnorm = optm_norm2(pg);
                if ~blmvm
                    %% Projected gradient no longer needed, free some memory.
                    pg = [];
                end
            else
                %% Just compute the norm of the gradient.
                pgnorm = optm_norm2(g);
            end
            if evals == best_evals
                %% Now we know the norm of the (projected) gradient at the best
                %% solution so far.
                best_pgnorm = pgnorm;
            end
            %% Check for algorithm convergence or termination.
            if evals == 1
                %% Compute value for testing the convergence in the gradient.
                gtest = max(gatol, grtol*pgnorm);
            end
            if pgnorm <= gtest
                %% Convergence in gradient.
                status = optm_status('GTEST_SATISFIED');
                break
            end
            if stage == 2
                %% Check convergence in relative function reduction.
                if f <= fatol || abs(f - f0) <= frtol*max(abs(f), abs(f0))
                    status = optm_status('FTEST_SATISFIED');
                    break
                end
                %% Compute the effective change of variables.
                s = x - x0;
                snorm = optm_norm2(s);
                %% Check convergence in variables.
                if snorm <= xatol || (xrtol > 0 && snorm <= xrtol*optm_norm2(x))
                    status = optm_status('XTEST_SATISFIED');
                    break
                end
            end
            if iters >= maxiter
                status = optm_status('TOO_MANY_ITERATIONS');
                break
            end
        end
        if evals >= maxeval
            status = optm_status('TOO_MANY_EVALUATIONS');
            break
        end
        if stage ~= 1
            %% Possibly print iteration information.
            if verb > 0 && mod(iters, verb) == 0
                print_iteration(iters, time() - t0, evals, rejects, ...
                                f, pgnorm, alpha);
                last_print = iters;
            end
            if stage ~= 0
                %% At least one step has been performed, L-BFGS approximation
                %% can be updated.
                if blmvm
                    lbfgs = optm_update_lbfgs(lbfgs, s, pg - pg0);
                else
                    lbfgs = optm_update_lbfgs(lbfgs, s, g - g0);
                end
            end
            %% Determine a new search direction `d`.  Parameter `dir` is set to:
            %%   0 if `d` is not a search direction,
            %%   1 if `d` is unscaled steepest descent,
            %%   2 if `d` is scaled sufficient descent.
            dir = 0;
            %% Use L-BFGS approximation to compute a search direction and check
            %% that it is an acceptable descent direction.
            if blmvm
                [d, scaled] = optm_apply_lbfgs(lbfgs, -pg);
                d = d .* freevars;
            else
                [d, scaled] = optm_apply_lbfgs(lbfgs, -g, freevars);
            end
            dg = optm_inner(d, g);
            if ~scaled
                %% No exploitable curvature information, `d` is the unscaled
                %% steepest feasible direction, that is the opposite of the
                %% projected gradient.
                dir = 1;
            else
                %% Some exploitable curvature information were available.
                dir = 2;
                if dg >= 0
                    %% L-BFGS approximation does not yield a descent direction.
                    dir = 0; % discard search direction
                    if ~bounded
                        if throwerrors
                            error('L-BFGS approximation is not positive definite');
                        end
                        status = optm_status('NOT_POSITIVE_DEFINITE');
                        break
                    end
                elseif epsilon > 0
                    %% A more restrictive criterion has been specified for
                    %% accepting a descent direction.
                    if dg > -epsilon*optm_norm2(d)*pgnorm
                        dir = 0; % discard search direction
                    end
                end
            end
            if dir == 0
                %% No exploitable information about the Hessian is available or
                %% the direction computed using the L-BFGS approximation failed
                %% to be a sufficient descent direction.  Take the steepest
                %% feasible descent direction.
                if bounded
                    d = -g .* freevars;
                else
                    d = -g;
                end
                dg = -pgnorm^2;
                dir = 1; % scaling needed
            end
            if dir ~= 2 && iters > 0
                rejects = rejects + 1;
            end
            %% Determine the length `alpha` of the initial step along `d`.
            if dir == 2
                %% The search direction is already scaled.
                alpha = 1.0;
            else
                %% Find a suitable step size along the steepest feasible
                %% descent direction `d`.  Note that `pgnorm`, the Euclidean
                %% norm of the (projected) gradient, is also that of `d` in
                %% that case.
                alpha = optm_steepest_descent_step(x, pgnorm, f, fmin, ...
                                                   delta, lambda);
            end
            if bounded
                %% Safeguard the step to avoid searching in a region where
                %% all bounds are overreached.
                [amin, amax] = optm_line_search_limits(x, lower, upper, ...
                                                       d, alpha);
                alpha = min(alpha, amax);
            end
            %% Initialize line-search.
            lnsrch = optm_start_line_search(lnsrch, f, dg, alpha);
            stage = lnsrch.stage;
            if stage ~= 1
                error('something is wrong!');
            end
            %% Save iterate at start of line-search.
            f0 = f;
            g0 = g;
            x0 = x;
            if blmvm
                pg0 = pg;
            end
        end
        %% Compute next iterate.
        if alpha == 1
            x = x0 + d;
        else
            x = x0 + alpha*d;
        end
    end

    %% In case of abnormal termination, some progresses may have been made
    %% since the start of the line-search.  In that case, we restore the best
    %% solution so far.
    if best_f < f
        f = best_f;
        g = best_g;
        x = best_x;
        if verb > 0
            %% Restore other information for printing.
            alpha = best_alpha;
            if best_pgnorm >= 0
                pgnorm = best_pgnorm;
            else
                %% Re-compute the norm of the (projected) gradient.
                if bounded
                    freevars = optm_unblocked_variables(x, lower, upper, g);
                    pgnorm = optm_norm2(g .* freevars);
                else
                    pgnorm = optm_norm2(g);
                end
            end
            if f < f0
                %% Some progresses since last iterate, pretend that one more
                %% iteration has been performed.
                ++iters;
            end
        end
    end
    if verb > 0
        if iters > last_print
            print_iteration(iters, time() - t0, evals, rejects, ...
                            f, pgnorm, alpha);
            last_print = iters;
        end
        fprintf('# Termination: %s\n', optm_reason(status));
    end
end

function print_iteration(iters, t, evals, rejects, f, pgnorm, alpha)
    if iters < 1
        fprintf('%s%s\n%s%s\n', ...
                '# Iter.   Time (ms)    Eval. Reject.', ...
                '       Obj. Func.           Grad.       Step', ...
                '# ----------------------------------', ...
                '-----------------------------------------------');
    end
    fprintf('%7d %11.3f %7d %7d %23.15e %11.3e %11.3e\n', ...
            iters, t, evals, rejects, f, pgnorm, alpha);
end
