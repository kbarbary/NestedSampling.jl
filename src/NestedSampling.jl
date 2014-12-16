module NestedSampling

# surprised this isn't in Julia base 
function logaddexp(x::FloatingPoint, y::FloatingPoint)
    if x == y
        return x + log(2.)
    else
        tmp = x - y
        if tmp > 0.
            return x + log1p(exp(-tmp))
        elseif tmp <= 0.
            return y + log1p(exp(tmp))
        else
            return tmp  # NaNs
        end
    end
end


# Represents an N-dimensional ellipsoid
immutable Ellipsoid
    ctr::Vector{Float64}  # center coordinates
    cov::Array{Float64, 2}
    icov::Array{Float64, 2}  # inverse of cov
    vol::Float64
end

# proportionality constant depending on dimension
# for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
# for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
function nball_vol_factor(ndim::Int)
    if ndim % 2 == 0
        c = 1.
        for i=2:2:ndim
            c *= 2pi / i
        end
        return c
    else
        c = 2.
        for i = 3:2:ndim
            c *= 2pi / i
        end
        return c
    end
end

function ellipsoid_volume(scaled_cov::Matrix{Float64})
    ndim = size(scaled_cov, 1)
    return nball_vol_factor(ndim) * sqrt(det(scaled_cov))
end

# find the bounding ellipsoid of points x where 
function bounding_ellipsoid(x::Matrix{Float64}, enlarge=1.0)

    ndim, npoints = size(x)

    ctr = mean(x, 2)
    delta = x .- ctr
    cov = Base.unscaled_covzm(delta, 2)
    icov = inv(cov)

    # Calculate expansion factor necessary to bound each point.
    # This finds the maximum of (delta' * icov * delta) for each point.
    fmax = -Inf
    for k in 1:npoints
        f = 0.0
        for j=1:ndim
            for i=1:ndim
                f += icov[i, j] * delta[i, k] * delta[j, k]
            end
        end
        fmax = max(fmax, f)
    end

    fmax *= enlarge
    scale!(cov, fmax)
    scale!(icov, 1./fmax)
    vol = ellipsoid_volume(cov)

    return Ellipsoid(ctr, cov, icov, vol)
end

# nested sampling algorithm to evaluate Bayesian evidence.
function nest_sample(loglikelihood::Function, prior::Function, ndim::Int;
                     npoints::Int=100)

    enlarge = 1.5  # enlarge vol
    maxiter = 10000

    enlarge_linear = enlarge^(1./ndim)

    # Choose initial points and calculate likelihoods
    u = rand(ndim, npoints)  # position of active points in unit cube
    v = zeros(ndim, npoints)  # position of active points in prior space
    logl = zeros(npoints)  # log(likelihood) at each point
    for i=1:npoints
        tmp = prior(sub(u, :, i))
        v[:, i] = tmp
        logl[i] = loglikelihood(tmp)
    end

    # Initialize values for nested sampling loop.
    samples_v = Float64[]    # stored objects for posterior results
    samples_logl = Float64[]
    samples_logprior = Float64[]
    samples_logwt = Float64[]
    loglstar = 0.  # ln(Likelihood constraint)
    h = 0.         # Information, initially 0.
    logz = -1.e300  # log(evidence Z, initially 0)

    # ln(width in prior mass), outermost width is 1 - e^(-1/n)
    logwidth = log(1. - exp(-1./npoints))  # log(prior volume), outermost
                                           # element is 1 - e^(-1/n)
    ncalls = npoints  # number of calls we already made

    # Nested sampling loop.
    ndecl = 0
    logwt_old = -Inf
    for it=1:maxiter
        lowlogl, lowi = findmin(logl)  # find lowest logl in active points
        logwt = logwidth + lowlogl

        # update evidence and information
        logz_new = logaddexp(logz, logwt)
        h = (exp(logwt - logz_new) * lowlogl +
             exp(logz - logz_new) * (h + logz) - logz_new)
        logz = logz_new

        # Add worst object to samples.
        append!(samples_v, sub(v, :, lowi))
        push!(samples_logwt, logwt)
        push!(samples_logprior, logwidth)
        push!(samples_logl, lowlogl)

        # The new likelihood constraint is that of the worst object.
        loglstar = lowlogl

        expected_vol = exp(-iter/npoints)

        # calculate the ellipsoid in prior space that contains all the
        # samples (including the worst one).
        ellip = bounding_ellipsoid(u, enlarge_linear)

        # choose a point from within the ellipse until it has likelihood
        # better than loglstar
        while true:
            utmp = sample_ellipsoid(ell)
            ok = true
            if any(utmp .< 0.) && any(utmp .> 1.)
                continue
            end
            vtmp = prior(utmp)
            logltmp = loglikelihood(vtmp)
            ncalls += 1

            # Accept if and only if within likelihood constraint.
            if logl > loglstar:
                u[:, lowi] = utmp
                v[:, lowi] = vtmp
                logl[lowi] = logltmp
                break
            end
        end

        # Shrink interval
        logwidth -= 1./nobj

        # stop when the logwt has been declining for more than nobj* 2
        # or niter/4 consecutive iterations.
        ndecl = (logwt < logwt_old) ? ndecl+1 : 0
        (ndecl > 2*npoints) && (ndecl > it/6) && break
        logwt_old = logwt
    end

    # Add remaining objects.
    # After N samples have been taken out, the remaining width is e^(-N/nobj)
    # The remaining width for each object is e^(-N/nobj) / nobj
    # The log of this for each object is:
    # log(e^(-N/nobj) / nobj) = -N/nobj - log(nobj)

end # module
