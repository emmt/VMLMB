# VMLMB for Yorick

`VMLMB` is an algorithm to minimize a multi-variate differentiable objective
function possibly under separable bound constraints.  `VMLMB` is a quasi-Newton
method ("VM" is for "Variable Metric") with low memory requirements ("LM" is
for "Limited Memory") and which can optionally take into account separable
bound constraints (the final "B") on the variables.  To determine efficient
search directions, `VMLMB` approximates the Hessian of the objective function
by a a limited memory version of the Broyden-Fletcher-Goldfarb-Shanno model
(L-BFGS for short).  Hence `VMLMB` is well suited to solving optimization
problems with a very large number of variables possibly with bound constraints.

This directory contains the code for a pure Yorick implementation of `VMLMB`.
To run the algorithm, call:

```.c
x = optm_vmlmb(fg, x0, lower=..., upper=...);
```

with `fg` the Yorick function which computes the objective function and its
gradient and `x0` the initial variables.  The initial variables may be an array
of any dimensions.  The method returns `x` the best solution found during
iterations.  Keywords `lower` and `upper` may be used to specify bounds on the
variables.  The function `fg` shall be implemented as follows (note that
argument `gx` is an ouput variable to store the gradient):

```.c
func fg(x, &gx)
{
    fx = ...; // value of the objective function at `x`
    gx = ...: // gradient of the objective function at `x`
    return fx;
}
```

See the documentation (`help, optm_vmlmb`) for a description of optional
arguments and keywords


## Installation

Installation is straightforward: copy file [`optm.i`](./optm.i) in some
suitable directory like `~/Yorick` where it can be found by Yorick's `include`
or `require` built-in functions.

For an all-user installation, copy [`optm.i`](./optm.i) in directory
`${Y_SITE}/i` or `${Y_SITE}/i0` and [`optm_start.i`](./optm_start.i) in
`${Y_SITE}/i-start` directory where `${Y_SITE}` is Yorick's "*site directory*"
(where Yorick's platform independent files are stored).  This may be automated
by calling the [`configure`](./configure) script and then `make install`:

```.sh
./configure
make install
```

The [`configure`](./configure) script can be called with flags `-h` or `--help`
to print a short help message.  The most important option is `--yorick=${EXE}`
to specify that `${EXE}` is the path of Yorick's executable.


## References

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*," in
  Astronomical Data Analysis II, SPIE Proc. **4847**, 174-183 (2002).


## License

`VMLMB` is licensed under the MIT "Expat" License:

> Copyright (c) 2002-2021: Éric Thiébaut

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
