# The VMLMB algorithm for large scale optimization with bound constraints

This repository provides implementations of the `VMLMB` algorithm
in various high-level programming languages.

`VMLMB` (for *Variable Metric Limited Memory with Bounds*) is a quasi-Newton
optimization method with small memory requirements and which may take into
account for separable bound constraints.  This algorithm is of particular
interest for minimizing a smooth cost function of a potentially very large
number of variables (millions or billions) and under bound constraints.
`VMLMB` has been successfully used to solve many different kinds of problems
notably image restoration in *inverse problems* framework.

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

- Directory [`matlab`](./matlab) contains a pure Matlab/Octave version of `VMLMB`.

- Directory [`yorick`](./yorick) contains a pure Yorick version of `VMLMB`.


## References

* É. Thiébaut, "*Optimization issues in blind deconvolution algorithms*," in
  Astronomical Data Analysis II, SPIE Proc. **4847**, 174-183 (2002).
