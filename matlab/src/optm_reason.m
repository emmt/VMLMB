%% optm_reason - Get textual description of status code.
%%
%% Call:
%%
%%     mesg = optm_reason(code);
%%
%% to retrieve a textual description of the status code returned by some
%% optimization method.
%%
%% See also `optm_status`, `optm_conjgrad`, and `optm_vmlmb`.
function mesg = optm_reason(code)
    switch code
        case 1 % TOO_MANY_EVALUATIONS
            mesg = 'too many evaluations';
        case 2 % TOO_MANY_ITERATIONS
            mesg = 'too many iterations';
        case 3 % FTEST_SATISFIED
            mesg = 'function reduction test satisfied';
        case 4 % XTEST_SATISFIED
            mesg = 'variables change test satisfied';
        case 5 % GTEST_SATISFIED
            mesg = '(projected) gradient test satisfied';
        case -1 % NOT_POSITIVE_DEFINITE
            mesg = 'matrix is not positive definite';
        otherwise
            error('unknown status code %d', code)
    end
end
