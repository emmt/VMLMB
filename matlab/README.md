# VMLMB for Matlab / GNU Octave

This directory contains the code of a [Matlab](https://www.mathworks.com) /
[GNU Octave](https://www.gnu.org/software/octave) implementation of `VMLMB` and
of the linear conjugate gradient method (for unconstrained linear problems).

## VMLMB: a quasi-Newton method with bound constraints

`VMLMB` is an algorithm to minimize a multi-variate differentiable objective
function possibly under separable bound constraints.  `VMLMB` is a quasi-Newton
method ("VM" is for "Variable Metric") with low memory requirements ("LM" is
for "Limited Memory") and which can optionally take into account separable
bound constraints (the final "B") on the variables.  To determine efficient
search directions, `VMLMB` approximates the Hessian of the objective function
by a limited memory version of the Broyden-Fletcher-Goldfarb-Shanno model
(L-BFGS for short).  Hence `VMLMB` is well suited to solving optimization
problems with a very large number of variables possibly with bound constraints.

To run the `VMLMB` algorithm, call [`optm_vmlmb`](./src/optm_vmlmb.m) as
follows:

```Matlab
[x, fx, gx, status] = optm_vmlmb(@fg, x0);
```

with `fg` the function which computes the objective function and its gradient
and `x0` the initial variables.  The initial variables may be an array of any
dimensions.  The method returns `x`, the best solution found during iterations,
`fx` and `gx`, the value and the gradient of the objective function `x`, and
`status`, an integer indicating the reason of the algorithm termination (call
[`optm_reason(status)`](./src/optm_reason.m) for a textual description).  The
function `fg` shall be implemented as follows:

```Matlab
function [fx, gx] = fg(x)
    fx = ...; % value of the objective function at x
    gx = ...; % gradient of the objective function at x
end
```

Options of the algorithm are specified as name-value pairs.  For instance, `'lower'` and
`'upper'` may be used to impose bounds on the variables:

```Matlab
[x, fx, gx, status] = optm_vmlmb(@fg, x0, 'lower', xmin, 'upper', xmax, ...);
```

See the leading comments in [`src/optm_vmlmb.m`](./src/optm_vmlmb.m) for a
description of all arguments and options of the `VMLMB` method.


## Preconditioned linear conjugate gradient method

To solve unconstrained linear optimization problems (those with a quadratic
objective function), you may use the implementation of the preconditioned
linear conjugate gradient method provided by this software.

A general expression for a quadratic objective function is:

```
f(x) = (1/2)*x'*A*x - b'*x + ϵ
```

where `A = ∇²f(x)` is the Hessian of the objective function (which does not
depend on the variables `x`), `b = -∇f(0)` is the opposite of the gradient of
the objective function at the origin, and `ϵ` is a constant.

Minimizing `f(x)` in `x` amounts to solve the following system of linear
equations:

```
A*x = b
```

This system of equations can be iteratively solved by means of the
(preconditioned) linear conjugate gradient method as follows:

```Matlab
[x, status] = optm_conjgrad(A, b, x0, ...)
```

which yields an approximate solution `x` and an integer, `status` indicating
the reason of the algorithm termination (call
[`optm_reason(status)`](./src/optm_reason.m) for a textual description).
Argument `x0` specifies the initial solution.  If `x0` is an empty array, i.e.
`[]`, `x` is initially an array of zeros. Argument `A` implements the
*left-hand-side (LHS) matrix* of the equations. It may be a function name or
handle and is called as `A(x)` to compute the result of `A*x`.  Argument `b` is
the *right-hand-side (RHS) vector* of the equations.

Note that, as `A` and the preconditioner `M` must be symmetric, it may be
faster to apply their adjoint.

Algorithm optional parameters (to specify a preconditioner, the stopping
criteria, etc.) can be specified as name-value pair of arguments. See the
leading comments in [`src/optm_conjgrad.m`](./src/optm_conjgrad.m) for a
description of all arguments and keywords of the linear conjugate gradient
method.


## Installation

To use this software, it is sufficient to add directory
[`./matlab/src`](./matlab/src) to your path (see `addpath` function).  All
*public* functions are prefixed by `optm_` to avoid name collisions.


## Speed-up

Depending on your version of Matlab / GNU Octave, some low level functions may
have to be modified to accelerate the code.  This can be simply done by
uncommenting the fastest expression (and commenting the others) in the
following source files:

- [`src/optm_inner.m`](./src/optm_inner.m) used to compute the inner product of
  variables;

- [`src/optm_norm1.m`](./src/optm_norm1.m) used to compute the L1 norm of
  variables;

- [`src/optm_norm2.m`](./src/optm_norm2.m) used to compute the Euclidean norm of
  variables;

- [`src/optm_norminf.m`](./src/optm_norminf.m) used to compute the infinite norm
  of variables.


## References

- Hestenes, M. R. & Stiefel, E. "*Methods of Conjugate Gradients for Solving
  Linear Systems*," in Journal of Research of the National Bureau of Standards,
  **49**, pp. 409-436 (1952).

- É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*," in
  Astronomical Data Analysis II, SPIE Proc. **4847**, 174-183 (2002).


## License

`VMLMB` is licensed under the MIT "Expat" License:

> Copyright (c) 2002-2022: Éric Thiébaut

> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
> CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
> TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
> SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
