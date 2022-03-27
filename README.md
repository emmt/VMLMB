# The VMLMB algorithm for large scale optimization with bound constraints

This repository provides implementations of the `VMLMB` algorithm in various
high-level programming languages.

`VMLMB` (for *Variable Metric Limited Memory with Bounds*) is a quasi-Newton
optimization method with small memory requirements and which may take into
account separable bound constraints.  This algorithm is of particular interest
for minimizing a smooth cost function of a potentially very large number of
variables (millions or billions) possibly under bound constraints.  `VMLMB` has
been successfully used to solve many different kinds of problems notably image
restoration in *inverse problems* framework.

The objective of this repository is to provide algorithms that:
* run out of the box (no additional libraries needed);
* are efficient (although maybe not as fast as if implemented in low level
  compiled languages) and usable for serious applications;
* are well documented;
* have readable code;
* can be easily modified.


## Contents

The repository is organized as follows:

- Directory [`matlab`](./matlab) contains a pure
  [Matlab](https://www.mathworks.com)/[GNU
  Octave](https://www.gnu.org/software/octave) version of `VMLMB`.  To use the
  provided optimization methods, it is sufficient to add directory
  [`./matlab/src`](./matlab/src) to your path (see `addpath` function).  All
  *public* functions are prefixed by `optm_` to avoid name collisions.  See code
  in [`test_deconv.m`](./matlab/test/test_deconv.m) for a detailed example of
  usage of `optm_conjgrad` (linear conjugate gradient) and `optm_vmlmb`
  (variable metric, limited memory, with bounds) optimizers.

- Directory [`yorick`](./yorick) contains a pure
  [Yorick](https://github.com/LLNL/yorick) version of `VMLMB`. See file
  [yorick/README.md](./yorick/README.md) for installation and usage
  instructions.


## References

- Hestenes, M. R. & Stiefel, E. "*Methods of Conjugate Gradients for Solving
  Linear Systems*," in Journal of Research of the National Bureau of Standards,
  **49**, pp. 409-436 (1952).

- É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*," in
  Astronomical Data Analysis II, SPIE Proc. **4847**, 174-183 (2002).
