%%     lnsrch = optm_new_line_search('key', val, ...);
%%
%% Create a new line-search context.  All parameters are specified by their
%% name followed by their value.  Possible parameters are:
%%
%% - 'ftol' specifies the function decrease tolerance.  A step `alpha` is
%%   considered as successful if the following condition (known as Armijo's
%%   condition) holds:
%%
%%       f(x0 + alpha*d) ≤ f(x0) + ftol*df(x0)*alpha
%%
%%   where `f(x)` is the objective function at `x`, `x0` denotes the variables
%%   at the start of the line-search, `df(x0)` is the directional derivative of
%%   the objective function at `x0`, `alpha` is the step length and `d` is the
%%   search direction.  The value of `ftol` must be in the range `(0,0.5]`, the
%%   default value is `ftol = 0.01`.
%%
%% - 'smin' and 'smax' can be used to specify relative bounds for safeguarding
%%   the step length.  When a step `alpha` is unsuccessful, a new backtracking
%%   step is computed based on a quadratic interpolation of the objective
%%   function along the search direction.  The next step writes:
%%
%%       next_alpha = gamma*alpha
%%
%%   with `gamma` safeguarded in the range `[smin,smax]`.  The following
%%   constraints must hold: `0 < smin ≤ smax < 1`.  Taking `smin = smax = 0.5`
%%   emulates the usual Armijo's method.  Default values are `smin = 0.2` and
%%   `smax = 0.9`.
%%
%% Note that when Armijo's conditon does not hold, the quadratic interpolation
%% yields `gamma < 1/(2 - 2*ftol)`.  Hence, taking an upper bound
%% `smax > 1/(2 - 2*ftol)` has no effects while taking a lower bound
%% `smin ≥ 1/(2 - 2*ftol)` yields a safeguarded `gamma` always equal to
%% `smin`.  Therefore, to benefit from quadratic interpolation, one should
%% choose `smin < 1/(2 - 2*ftol)`.
%%
%% See documentation of `optm_iterate_line_search` for a complete usage example.
function lnsrch = optm_new_line_search(varargin)
    if mod(length(varargin), 2) ~= 0
        error('parameters must be specified as pairs of names and values')
    end
    lnsrch.ftol = 0.01;
    lnsrch.smin = 0.2;
    lnsrch.smax = 0.9;
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        if ~isscalar(val) || ~isreal(val) || ~isfinite(val)
            error('parameter value must be a finite real scalar')
        end
        switch key
            case 'ftol'
                lnsrch.ftol = val;
            case 'smin'
                lnsrch.smin = val;
            case 'smax'
                lnsrch.smax = val;
            otherwise
                error('invalid parameter name');
        end
    end
    if lnsrch.ftol <= 0 || lnsrch.ftol > 0.5
        error('`ftol` must be in the range (0,0.5]');
    end
    if lnsrch.smin <= 0 || lnsrch.smin >= 1
        error('`smin` must be in the range (0,1)');
    end
    if lnsrch.smax <= 0 || lnsrch.smax >= 1
        error('`smax` must be in the range (0,1)');
    end
    if lnsrch.smin > lnsrch.smax
        error('`smin` must be less or equal `smax`');
    end
    lnsrch.stage = 0;
    lnsrch.step  = 0;
    lnsrch.finit = nan();
    lnsrch.ginit = nan();
end
