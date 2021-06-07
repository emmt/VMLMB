// optm_example_deconv.i -
//
// Bound constrained regularized deconvolution of image with VMLMB in Yorick.
//-----------------------------------------------------------------------------

SRCDIR = dirname(current_include()) + "/";
DATADIR = SRCDIR + "../data/saturn/";

// Require optimization code and, optionally, FFTW plugin.
include, SRCDIR+"optm.i";
include, SRCDIR+"optm_example_utils.i";
include, "xfft.i", 3;

// Read the data.
dat = transpose(fits_read(DATADIR+"saturn.fits"));
psf = transpose(fits_read(DATADIR+"saturn_psf.fits"));
plimg, dat, cmin=0, fig=1, title="Raw data";
plimg, psf, cmin=0, fig=2, title="PSF";

// Regularization level.
mu = 0.01;

// Assume 70% of bad pixels.
fraction_of_bad_pixels = 0*0.7;

// Array of pixelwise weights and corresponding operator.
wgt = array(1.0, dimsof(dat));
bad_pixels = where(random(dimsof(wgt)) < fraction_of_bad_pixels);
if (is_array(bad_pixels)) wgt(bad_pixels) = 0;

// Compute the Modulation Transfer Function (MTF) as the FFT of the PSF after
// proper zero-padding and centering and define functions to apply the
// convolution by the PSF operator and its adjoint.
func H__1(x) { return ifft(mtf*fft(x)); }
func Ht_1(x) { return ifft(conj(mtf)*fft(x)); }
func H__2(x) { return FFT(mtf*FFT(x, 0), 2); }
func Ht_2(x) { return FFT(conj(mtf)*FFT(x, 0), 2); }
if (1 && is_func(xfft_new)) {
    // Use FFTW.
    FFT = xfft_new(dims=dimsof(dat), real=1n, planning=XFFT_ESTIMATE);
    mtf = FFT(fftshift(zeropad(psf, dimsof(dat))));
    H  = H__2;
    Ht = Ht_2;
} else {
    // Use Yorick's fft.
    mtf = fft(fftshift(zeropad(psf, dimsof(dat))));
    H  = H__1;
    Ht = Ht_1;
}

// Weighting operator.
func W(x) { return wgt*x; }

// Function to apply the LHS "matrix" of the normal equations.
func lhs(x) { return Ht(W(H(x))) + mu*DtD(x); }

// RHS "vector" of the normal equations.  *MUST* be recomputed whenever
// the weights change.
rhs = Ht(W(dat));

// Show good pixels.
plimg, W(dat), cmin=0, fig=3, title="Good pixels in data";

// Initial solution.
x0 = array(double, dimsof(dat));

// Iterative deconvolution by the linear conjugate gradients (function name
// "lhs" or handle @lhs both work).
local status;
x1 = optm_conjgrad(lhs, rhs, x0, status, maxiter=50, verbose=1);
write, format="# Algorithm stops because %s.\n\n", optm_reason(status);
plimg, x1, cmin=0, fig=4,
    title="Result of deconvolution by linear conjugate gradient";

// Function to compute the objective function and its gradient.
func fg(x, &g)
{
    r = H(x) - dat;
    Wr = W(r);
    muDtDx = mu*DtD(x);
    g = Ht(Wr) + muDtDx;
    return 0.5*(optm_inner(r, Wr) + optm_inner(x, muDtDx));
}

// Iterative deconvolution by a quasi-Newton method (without bounds).
local f, g, status;
x2 = optm_vmlmb(fg, x0, f, g, status,
                fmin=0, maxiter=50, verbose=1);
write, format = "# %s\n\n", optm_reason(status);
plimg, x2, cmin=0, fig=5,
    title="Result of deconvolution by\nvariable metric method";

// Iterative deconvolution by a quasi-Newton method (with lower bound).
x3 = optm_vmlmb(fg, x0, f, g, status, lower=0,
                fmin=0, maxiter=50, verbose=1);
write, format = "# %s\n\n", optm_reason(status);
plimg, x3, cmin=0, fig=6,
    title="Result of deconvolution by\nvariable metric method and positivity constraints";

// Iterative deconvolution by a quasi-Newton method (with lower and upper bounds).
x4 = optm_vmlmb(fg, x0, f, g, status, lower=0, upper=1e3,
                fmin=0, maxiter=50, verbose=1);
write, format = "# %s\n\n", optm_reason(status);
plimg, x4, cmin=0, fig=7,
    title="Result of deconvolution by\nvariable metric method and lower and upper bounds";
