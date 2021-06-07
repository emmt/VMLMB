// optm_example_utils.i -
//
// Utilities for testing VMLMB in Yorick.
//-----------------------------------------------------------------------------

func zeropad(x, zdims)
/* DOCUMENT Zero-pad array X to given shape.

    The contents X is approximately centered in the result.

   SEE ALSO:
 */
{
  if (! is_integer(zdims) || zdims(1) != numberof(zdims) - 1 ||
      (numberof(zdims) > 1 && zdims(min:2:) <= 0)) {
    error, "bad dimension list";
  }
  xdims = dimsof(x);
  rank = numberof(xdims) - 1;
  if (rank < 1 || numberof(zdims) != numberof(xdims)) {
    error, "bad number of dimensions";
  }
  if (min(zdims - xdims) < 0) {
    error, "output dimensions must be larger or equal input dimensions";
  }
  offset = (zdims(2:) - xdims(2:))/2;
  i = 1 + offset;
  j = xdims(2:) + offset;
  z = array(structof(x), zdims);
  if (rank == 1) {
    z(i(1):j(1)) = x;
  } else if (rank == 2) {
    z(i(1):j(1),i(2):j(2)) = x;
  } else if (rank == 3) {
    z(i(1):j(1),i(2):j(2),i(3):j(3)) = x;
  } else if (rank == 3) {
    z(i(1):j(1),i(2):j(2),i(3):j(3)) = x;
  } else if (rank == 4) {
    z(i(1):j(1),i(2):j(2),i(3):j(3),i(4):j(4)) = x;
  } else if (rank == 5) {
    z(i(1):j(1),i(2):j(2),i(3):j(3),i(4):j(4),i(5):j(5)) = x;
  } else if (rank == 6) {
    z(i(1):j(1),i(2):j(2),i(3):j(3),i(4):j(4),i(5):j(5),i(6):j(6)) = x;
  } else if (rank == 7) {
    z(i(1):j(1),i(2):j(2),i(3):j(3),i(4):j(4),i(5):j(5),i(6):j(6),i(7):j(7)) = x;
  } else if (rank == 8) {
    z(i(1):j(1),i(2):j(2),i(3):j(3),i(4):j(4),i(5):j(5),i(6):j(6),i(7):j(7),i(8):j(8)) = x;
  } else if (rank == 9) {
    z(i(1):j(1),i(2):j(2),i(3):j(3),i(4):j(4),i(5):j(5),i(6):j(6),i(7):j(7),i(8):j(8),i(9):j(9)) = x;
  } else if (rank == 10) {
    z(i(1):j(1),i(2):j(2),i(3):j(3),i(4):j(4),i(5):j(5),i(6):j(6),i(7):j(7),i(8):j(8),i(9):j(9),i(10):j(10)) = x;
  } else {
    error, "too many dimensions";
  }
  return z;
}

func DtD(x)
/* DOCUMENT DtD(x)
     returns the result of D'.D.x where D is a (multi-dimensional)
     finite difference operator and D' is its transpose.

   SEE ALSO:
 */
{
  /* Create an array to store the result.  Manage to use the same type as the
     `dif` operator. */
  if (is_real(x)) {
    type = structof(x);
  } else if (is_integer(x)) {
    type = long;
  } else {
    error, "bad data type";
  }
  dims = dimsof(x);
  r = array(type, dims);
  rank = numberof(dims) - 1; // number of dimensions (up to 10 in Yorick)
  p = 1:-1;
  q = 2:0;
  if (rank >= 1) {
    if (dims(2) >= 2) {
      dx = x(dif,..);
      r(q,..) += dx;
      r(p,..) -= dx;
    }
    if (rank >= 2) {
      if (dims(3) >= 2) {
        dx = x(,dif,..);
        r(,q,..) += dx;
        r(,p,..) -= dx;
      }
      if (rank >= 3) {
        if (dims(4) >= 2) {
          dx = x(,,dif,..);
          r(,,q,..) += dx;
          r(,,p,..) -= dx;
        }
        if (rank >= 4) {
          if (dims(5) >= 2) {
            dx = x(,,,dif,..);
            r(,,,q,..) += dx;
            r(,,,p,..) -= dx;
          }
          if (rank >= 5) {
            if (dims(6) >= 2) {
              dx = x(,,,,dif,..);
              r(,,,,q,..) += dx;
              r(,,,,p,..) -= dx;
            }
            if (rank >= 6) {
              if (dims(7) >= 2) {
                dx = x(,,,,,dif,..);
                r(,,,,,q,..) += dx;
                r(,,,,,p,..) -= dx;
              }
              if (rank >= 7) {
                if (dims(8) >= 2) {
                  dx = x(,,,,,,dif,..);
                  r(,,,,,,q,..) += dx;
                  r(,,,,,,p,..) -= dx;
                }
                if (rank >= 8) {
                  if (dims(9) >= 2) {
                    dx = x(,,,,,,,dif,..);
                    r(,,,,,,,q,..) += dx;
                    r(,,,,,,,p,..) -= dx;
                  }
                  if (rank >= 9) {
                    if (dims(10) >= 2) {
                      dx = x(,,,,,,,,dif,..);
                      r(,,,,,,,,q,..) += dx;
                      r(,,,,,,,,p,..) -= dx;
                    }
                    if (rank >= 10) {
                      if (dims(11) >= 2) {
                        dx = x(,,,,,,,,,dif,..);
                        r(,,,,,,,,,q,..) += dx;
                        r(,,,,,,,,,p,..) -= dx;
                      }
                      if (rank > 10) {
                        error, "too many dimensions";
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return r;
}

func fftshift(a)
{
  dims = dimsof(a);
  rank = numberof(dims) - 1;
  if (rank < 1) return a+0.0;
  return roll(a, dims(2:0)/2);
}

func ifftshift(a)
{
  dims = dimsof(a);
  rank = numberof(dims) - 1;
  if (rank < 1) return a+0.0;
  return roll(a, -(dims(2:0)/2));
}

func ifft(x, setup=)
{
    return (1.0/numberof(x))*double(fft(x, -1, setup=setup));
}

// A simple function to plot an image.
func plimg(img, fig=, colors=, cmin=, cmax=, title=, xlabel=, ylabel=)
{
    dims = dimsof(img);
    if (numberof(dims) == 3) {
        // Gray-scaled image.
        width = dims(2);
        height = dims(3);
    } else if (numberof(dims) == 4 && numberof(dims) == 3 && structof(img) == char) {
        // RGB image.
        width = dims(3);
        height = dims(4);
    } else {
        error, "invalid image format";
    }
    if (!is_void(fig)) {
        window, fig;
    }
    fma;
    if (!is_void(colors)) {
        cmap, colors;
    }
    pli, img, 0.5, 0.5, width + 0.5, height + 0.5, cmin=cmin, cmax=cmax;
    if (!is_void(title)) pltitle, title;
    if (!is_void(xlabel) || !is_void(ylabel)) {
        xytitle, xlabel, ylabel;
    }
}
