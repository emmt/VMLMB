# The VMLMB algorithm for large scale optimization with bound constraints

This repository provides implementations of the `VMLMB` algorithm and
preconditioned linear conjugate gradient method in various high-level
programming languages.

`VMLMB` (for *Variable Metric Limited Memory with Bounds*) is a quasi-Newton
optimization method with small memory requirements and which may take into
account separable bound constraints.  This algorithm is of particular interest
for minimizing a smooth cost function of a potentially very large number of
variables (millions or billions) possibly under bound constraints.  `VMLMB` has
been successfully used to solve many different kinds of problems notably image
restoration in *inverse problems* framework.

The objective of this repository is to provide algorithms that:
* run out of the box (no additional libraries needed);
* are efficient (although maybe not as fast as if implemented in a low level
  compiled language) and usable for serious applications;
* are well documented;
* have readable code;
* can be easily modified.


## Contents

The repository is organized as follows:

- Directory [`matlab`](./matlab) contains a pure
  [Matlab](https://www.mathworks.com)/[GNU
  Octave](https://www.gnu.org/software/octave) version of `VMLMB` and of a
  preconditioned linear conjugate gradient method.  See file
  [`matlab/README.md`](./matlab/README.md) for installation and usage
  instructions.

- Directory [`python`](./python) contains a pure [`NumPy`](https://numpy.org/)
  version of `VMLMB` and of a preconditioned linear conjugate gradient method.
  See file [`python/README.md`](./python/README.md) for installation and usage
  instructions.

- Directory [`yorick`](./yorick) contains a pure
  [Yorick](https://github.com/LLNL/yorick) version of `VMLMB` and of a
  preconditioned linear conjugate gradient method.  See file
  [`yorick/README.md`](./yorick/README.md) for installation and usage
  instructions.


## References

- Hestenes, M. R. & Stiefel, E. "*Methods of Conjugate Gradients for Solving
  Linear Systems*," in Journal of Research of the National Bureau of Standards,
  **49**, pp. 409-436 (1952).

- É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*," in
  Astronomical Data Analysis II, SPIE Proc. **4847**, 174-183 (2002).
