module NestedSampling

# Represents an N-dimensional ellipsoid
immutable Ellipsoid
    ctr::Vector{Float64}  # center coordinates
    cov::Array{Float64, 2}
    icov::Array{Float64, 2}  # inverse of cov
    vol::Float64
end

# surprised this isn't in Julia base 
function logaddexp(x::FloatingPoint, y::FloatingPoint)
    if x == y
        return x + log(2.)
    else
        tmp = x - y
        if tmp > zero(tmp)
            return x + log1p(exp(-tmp))
        elseif tmp <= 0.
            return y + log1p(exp(tmp))
        else
            return tmp  # NaNs
        end
    end
end

# Draw a random point from within a unit N-ball
function randnball(ndim)
    z = randn(ndim)
    r2 = 0.
    for i=1:ndim
        r2 += z[i]*z[i]
    end
    factor = rand()^(1./ndim) / sqrt(r2)
    for i=1:ndim
        z[i] *= factor
    end
    return z
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
    # This finds the maximum of (delta_i' * icov * delta_i)
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

function sample_ellipsoid(ell::Ellipsoid, nsamples=1)
    ndim = length(ellipsoid.ctr)

    # Get scaled eigenvectors (in columns): vs[:,i] is the i-th eigenvector.
    f = eigfact(ellipsoid.cov)
    v, w = f[:vectors], f[:values]
    for j=1:ndim
        tmp = sqrt(w[j])
        for i=1:ndim
            v[i, j] *= tmp
        end
    end

    return dot(v, randnball(ndim)) .+ ellipsoid.ctr
end


# nested sampling algorithm to evaluate Bayesian evidence.
function sample(loglikelihood::Function, prior::Function, ndim::Int;
                npoints::Int=100, enlarge::Float=1.5, maxiter::Int=10000)

    # enlarge is volume enlargement factor
    enlarge_linear = enlarge^(1./ndim)

    # Choose initial points and calculate likelihoods
    points_u = rand(ndim, npoints)  # position of active pts in unit cube
    points_v = zeros(ndim, npoints)  # position of active pts in prior space
    points_logl = zeros(npoints)  # log(likelihood) at each point
    for i=1:npoints
        v = prior(sub(points_u, :, i))
        points_v[:, i] = v
        points_logl[i] = loglikelihood(v)
    end
    ncall = npoints  # number of likelihood calls we just made

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

    # Nested sampling loop.
    ndecl = 0
    logwt_old = -Inf
    niter = 0
    while niter < maxiter
        niter += 1

        # find lowest logl in active points
        logl, minidx = findmin(points_logl)
        logwt = logwidth + logl

        # update evidence and information
        logz_new = logaddexp(logz, logwt)
        h = (exp(logwt-logz_new) * logl +
             exp(logz-logz_new) * (h+logz) - logz_new)
        logz = logz_new

        # Add worst object to samples.
        append!(samples_v, sub(v, :, minidx))
        push!(samples_logwt, logwt)
        push!(samples_logprior, logwidth)
        push!(samples_logl, logl)

        # The new likelihood constraint is that of the worst object.
        loglstar = logl

        expected_vol = exp(-niter/npoints)

        # calculate the ellipsoid in prior space that contains all the
        # samples (including the worst one).
        ellip = bounding_ellipsoid(points_u, enlarge_linear)

        # choose a point from within the ellipse until it has likelihood
        # better than loglstar
        while true
            u = sample_ellipsoid(ell)
            ok = true
            if any(u .< 0.) && any(u .> 1.)
                continue
            end
            v = prior(u)
            logltmp = loglikelihood(v)
            ncall += 1

            # Accept if and only if within likelihood constraint.
            if logl > loglstar:
                points_u[:, minidx] = u
                points_v[:, minidx] = v
                points_logl[minidx] = logl
                break
            end
        end

        # Shrink interval
        logwidth -= 1./nobj

        # stop when the logwt has been declining for more than nobj* 2
        # or niter/4 consecutive iterations.
        ndecl = (logwt < logwt_old) ? ndecl+1 : 0
        (ndecl > 2*npoints) && (ndecl > niter/6) && break
        logwt_old = logwt
    end

    # Add remaining objects.
    # After N samples have been taken out, the remaining width is e^(-N/nobj)
    # The remaining width for each object is e^(-N/nobj) / nobj
    # The log of this for each object is:
    # log(e^(-N/nobj) / nobj) = -N/nobj - log(nobj)
    nsamples = div(length(samples_v), ndim)
    logwidth = -nsamples/npoints - log(npoints)
    for i in 1:npoints
        logwt = logwidth + points_logl[i]
        logz_new = logaddexp(logz, logwt)
        h = (exp(logwt - logz_new) * points_logl[i] +
             exp(logz - logz_new) * (h + logz) - logz_new)
        logz = logz_new

        append!(samples_v, sub(points_v, :, i))
        push!(samples_logwt, logwt)
        push!(samples_logl, points_logl[i])
        push!(samples_logprior, logwidth)
    end

    nsamples += npoints

    return ["niter" => niter,
            "ncall" => ncall,
            "logz" => logz,
            "logzerr" => sqrt(h/npoints),
            "loglmax" => max(points_logl),
            "h" => h,
            "samples" => reshape(samples_v, (nsamples, ndim)),
            "weights" => exp(samples_logwt .- logz),
            "logprior" => samples_logprior,
            "logl" => samples_logl]
end

end # module
