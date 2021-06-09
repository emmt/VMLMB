%%     code = optm_status(str);
%%
%% Retrieve the status code associated with textual identifier `str`.
%%
%% See also `optm_reason` to convert a status code into a textual message.
function code = optm_status(str)
    switch str
        case 'TOO_MANY_EVALUATIONS'
            code =  1;
        case 'TOO_MANY_ITERATIONS'
            code =  2;
        case 'FTEST_SATISFIED'
            code =  3;
        case 'XTEST_SATISFIED'
            code =  4;
        case 'GTEST_SATISFIED'
            code =  5;
        case 'NOT_POSITIVE_DEFINITE'
            code = -1;
        otherwise
            error('unknown status string `%s`', str)
    end
end
