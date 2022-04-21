- Change keywords (or trailing arguments) of `optm_steepest_descent_step`:
  ``delta` becomes `dxrel` and `lambda` becomes `f2nd`.  This is to add more
  flexibility (new setting `dxabs`) and to avoid `lambda` which is a reserved
  word in Python.

- Only compute `amax` in line-search limits.
