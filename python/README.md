# VMLMB for Python

Module `optm` provides pure Python implementations of the `VMLMB` algorithm
(for solving non-linear bound constrained optimization problems) and of the
linear conjugate gradient method (for solving unconstrained linear optimization
problems).

`VMLMB` is a quasi-Newton method (`VM` is for *Variable Metric*) with low
memory requirements (`LM` is for *Limited Memory*) and which can optionally
take into account separable bound constraints (the final "`B`) on the
variables. To determine efficient search directions, `VMLMB` approximates the
Hessian of the objective function by a limited memory version of the
Broyden-Fletcher-Goldfarb-Shanno model (L-BFGS for short). Hence `VMLMB` is
well suited to solving optimization problems with a very large number of
variables possibly with bound constraints.


## Installing

To use the algorithms of `OptimPack`, you just have to put file
[`optm.py`](./optm.py) in your Python path.  All methods and types are defined
in module `optm`.

After installation, see help for `optm.vmlmb` and `optm.conjgrad` for the
respective documentation of the `VMLMB` algorithm and of the linear conjugate
gradient method.


## Testing

To test the code, move to this directory and run Python interpreter on file
[`minpack1.py`](./minpack1.py):
```python
python3 minpack1.py
```
