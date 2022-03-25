# The VMLMB algorithm for large scale optimization with bound constraints

This repository provides implementations of the `VMLMB` algorithm
in various high-level programming languages.

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

- Directory [`data`](./data) contains data for tests and examples.

- Directory [`matlab`](./matlab) contains a pure
  [MATLAB](https://www.mathworks.com)/[GNU
  Octave](https://www.gnu.org/software/octave) version of `VMLMB`.  To use the
  provided optimization methods, it is sufficient to add directory
  [`./matlab/src`](./matlab/src) to your path (see `addpath` function).  All
  *public* functions are prefixed by `optm_` to avoid name collisions.  See code
  in [`test_deconv.m`](./matlab/test/test_deconv.m) for a detailed example of
  usage of `optm_conjgrad` (linear conjugate gradient) and `optm_vmlmb`
  (variable metric, limited memory, with bounds) optimizers.

- Directory [`yorick`](./yorick) contains a pure Yorick version of `VMLMB`.
  All optimization methods are in [`optm.i`](./yorick/optm.i) which may be
  installed in any of the directories searched by Yorick (see `get_path` and
  `set_path` functions).  to avoid name collisions, all *public*
  functions/variables are prefixed by `optm_` and all *private*
  functions/variables are prefixed by `_optm_`.  The [`yorick`](./yorick)
  directory also contains various examples and demonstrations.


## References

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*," in
  Astronomical Data Analysis II, SPIE Proc. **4847**, 174-183 (2002).
