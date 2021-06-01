function str = optm_conjgrad_reason(status)
    switch status
        case -1 %% NOT_POSITIVE_DEFINITE
            str = "LHS operator is not positive definite";
        case 0 %% TOO_MANY_ITERATIONS   =
            str = "too many iterations";
        case 1 %% F_TEST_SATISFIED      =
            str = "function reduction test satisfied";
        case 2 %% G_TEST_SATISFIED      =
            str = "gradient test satisfied";
        case 3 %% X_TEST_SATISFIED      =
            str = "variables change test satisfied";
        otherwise
            "unknown conjugate gradient result"
    end
end
