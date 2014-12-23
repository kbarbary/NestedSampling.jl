using Base.Test
using TimeIt
import NestedSampling: logaddexp, nball_vol_factor, sample

# Test logaddexp()
x = log(1.e-100)
y = log(2.e-100)
@test_approx_eq exp(logaddexp(x, y)) 3.e-100

# Test volume prefactor for an N-ball
@test_approx_eq nball_vol_factor(4) pi^2 / 2
@test_approx_eq nball_vol_factor(5) 8pi^2/15
@test_approx_eq nball_vol_factor(6) pi^3 / 6
@test_approx_eq nball_vol_factor(7) 16pi^3/105


# -----------------------------------------------------------------------------
# Simple likelihood function

    # gaussians centered at (1, 1) and (-1, -1) with a width of 0.1.
const mu1 = [1., 1.]
const mu2 = [-1., -1.]
const sigma = 0.1
const invvar = eye(2) / sigma^2

function logl(x)
    dx1 = x .- mu1
    dx2 = x .- mu2
    return logaddexp((dx1' * invvar * dx1)[1] / 2.0,
                     (dx2' * invvar * dx2)[1] / 2.0)
end



# Use a flat prior, over [-5, 5] in both dimensions
prior(x) = 10.0 .* x .- 5.0

srand(0)
res = sample(logl, prior, 2; npoints=100)
@printf "evidence = %6.3f +/- %6.3f\n" res["logz"] res["logzerr"]

#(Approximate) analytic evidence for two identical Gaussian blobs,
# over a uniform prior [-5:5][-5:5] with density 1/100 in this domain:
analytic_logz = log(2.0 * 2.0*pi*sigma*sigma / 100.)
@printf "analytic = %6.3f\n" analytic_logz 

    # calculate evidence on fine grid.
    #dx = 0.1
    #xv = np.arange(-5.0 + dx/2., 5., dx)
    #yv = np.arange(-5.0 + dx/2., 5., dx)
    #grid_logz = -1.e300
    #for x in xv:
    #    for y in yv:
    #        grid_logz = np.logaddexp(grid_logz, logl(np.array([x, y])))
    #grid_logz += np.log(dx * dx / 100.)  # adjust for point density
    #print "grid_logz =", grid_logz

    #assert abs(res.logz - analytic_logz) < 2.0 * res.logzerr
    #assert abs(res.logz - grid_logz) < 2.0 * res.logzerr

