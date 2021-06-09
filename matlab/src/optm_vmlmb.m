%% optm_vmlmb - Apply VMLMB algorithm.
%%
%%     [x, f, g, status] = optm_vmlmb(fg, x0, ...);
%%
%% Apply VMLMB algorithm to minimize a multi-variate differentiable objective
%% function possibly under separble bound constraints.  VMLMB is a quasi-Newton
%% method ("VM" is for "Variable Metric") with low memory requirements ("LM" is
%% for "Limited Memory") and which can optionally take into account separable
%% bound constraints (the final "B") on the variables.  To determine efficient
%% search directions, VMLMB approximates the Hessian of the objective function
%% by a model, called L-BFGS for short, which is a limited memory version of
%% the one assumed in Broyden-Fletcher-Goldfarb-Shanno algorithm.  Hence VMLMB
%% is well suited to solving optimization problems with a very large number of
%% variables possibly with bound constraints.
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
%%     function [fx, gx] = fg(x)
%%         fx = ...; // value of the objective function at `x`
%%         gx = ...: // gradient of the objective function at `x`
%%     end
%%
%% All other settings are specified by keywords:
%%
%% - Keywords `upper` and `lower` are to specify a lower and/or an upper bounds
%%   for the variables.  If unspecified or set to an empty array, a given bound
%%   is considered as unlimited.  Bounds must be conformable with the
%%   variables.
%%
%% - Keyword `mem` specifies the memory used by the algorithm, that is the
%%   number of previous steps memorized to approximate the Hessian of the
%%   objective function.  With `mem=0`, the algorithm
%%   behaves as a steepest descent method.  The default is `mem=5`.
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
%%   of the Hessaian of the objective function.  This setting may be used to
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
%% - Keyword `verbose` specifies whether to print information at each
%%   iteration.
%%
%% - Keyword `throwerrors` (true by default), specifies whether to call `error`
%%   in case of errors instead or returning a `status` indicating the problem.
%%   Note that early termination due to limits set on the number of iterations
%%   or of evaluations of the objective function are not considereed as an
%%   error.

function [x, f, g, status] = optm_vmlmb(fg, x, varargin)
    if nargin < 2
        print_usage;
    end

    %% Constants.  Calling inf, nan, true or false takes too much time (2.1µs
    %% instead of 0.2µs if stored in a variable), so use local variables
    %% (shadowing the functions) to pay the price once.
    INF = inf;
    NAN = nan;
    TRUE = true;
    FALSE = false;

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
    verbose = FALSE;
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
            case 'verbose'
                verbose = val;
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
    if isscalar(ftol)
        fatol = -INF;
        frtol = ftol;
    else
        fatol = ftol(1);
        frtol = ftol(2);
    end
    if isscalar(gtol)
        gatol = 0.0;
        grtol = gtol;
    else
        gatol = gtol(1);
        grtol = gtol(2);
    end
    if isscalar(xtol)
        xatol = 0.0;
        xrtol = xtol;
    else
        xatol = xtol(1);
        xrtol = xtol(2);
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
    alpha = 0.0;   % step length
    amin = 0.0;    % first step length threshold
    amax = INF;    % last step length threshold
    evals = 0;     % number of calls to fg
    iters = 0;     % number of iterations
    projs = 0;     % number of projections onto the feasible set
    status = 0;    % algorithm status is zero until termination.
    best_f = INF;  % best function value so far
    best_g = [];   % corresponding gradient
    best_x = [];   % corresponding variables
    freevars = []; % subset of free variables not yet known
    if isempty(lnsrch)
        lnsrch = optm_new_line_search();
    end
    if ischar(fg)
        fg = str2func(fg);
    end
    lbfgs = optm_new_lbfgs(mem);
    if verbose
        time = @() 86400E3*now(); % yields number of milliseconds
        t0 = time();
    end
    print_now = FALSE;

    %% Algorithm stage is one of:
    %% 0 = initial;
    %% 1 = first trial step in line-search;
    %% 2 = second and subsequent trial steps in line-search;
    %% 3 = line-search has converged.
    stage = 0;

    while TRUE
        if bounded && stage < 2
            %% Make the variables feasible.
            x = optm_clamp(x, lower, upper);
            projs = projs + 1;
        end
        %% Compute objective function and its gradient.
        [f, g] = fg(x);
        evals = evals + 1;
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
            if lnsrch.stage ~= 1
                error('something is wrong!');
            end
            stage = 2;
        end
        if stage == 2
            %% Check for line-search convergence.
            lnsrch = optm_iterate_line_search(lnsrch, f);
            if lnsrch.stage == 2
                %% Line-search has converged, `x` is the next iterate.
                stage = 3;
                iters = iters + 1;
            elseif lnsrch.stage == 1
                alpha = lnsrch.step;
            else
                error('something is wrong!');
            end
        end
        if stage == 3 || stage == 0
            %% Initial or next iterate after convergence of line-search.
            if bounded
                %% Determine the subset of free variables and compute the norm
                %% of the projected gradient (needed to check for convergence).
                freevars = optm_active_variables(x, lower, upper, g);
                if ~any(freevars(:))
                    %% Variables are all blocked.
                    status = optm_status('XTEST_SATISFIED');
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
                gtest = max(max(0.0, gatol), grtol*gnorm);
            end
            if status == 0 && gnorm <= gtest
                %% Convergence in gradient
                status = optm_status('GTEST_SATISFIED');
            end
            if stage == 3
                if status == 0
                    %% Check convergence in relative function reduction.
                    if f <= fatol || abs(f - f0) <= max(0.0, frtol*max(abs(f), abs(f0)))
                        status = optm_status('FTEST_SATISFIED');
                    end
                end
                if alpha ~= 1.0
                    d = x - x0; % recompute effective step
                end
                if status == 0
                    %% Check convergence in variables.
                    dnorm = optm_norm2(d);
                    if dnorm <= max(0.0, xatol) || (xrtol > 0 && dnorm <= xrtol*optm_norm2(x))
                        status = optm_status('XTEST_SATISFIED');
                    end
                end
            end
            if status == 0 && iters >= maxiter
                status = optm_status('TOO_MANY_ITERATIONS');
            end
            print_now = verbose;
        end
        if status == 0 && evals >= maxeval
            status = optm_status('TOO_MANY_EVALUATIONS');
        end
        if verbose && status ~= 0 && ~print_now && best_f < f0
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
            iters = iters + 1;
            print_now = verbose;
        end
        if print_now
            t = (time() - t0); % elapsed milliseconds
            if iters < 1
                fprintf('%s%s\n%s%s\n', ...
                        '# Iter.   Time (ms)   Eval.   Proj. ', ...
                        '       Obj. Func.           Grad.       Step', ...
                        '# ----------------------------------', ...
                        '-----------------------------------------------');
            end
            fprintf('%7d %11.3f %7d %7d %23.15e %11.3e %11.3e\n', ...
                   iters, t, evals, projs, f, gnorm, alpha);
            print_now = ~print_now;
        end
        if status ~= 0
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
                                error('L-BFGS approximation is not positive definite');
                            end
                            status = optm_status('NOT_POSITIVE_DEFINITE');
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
