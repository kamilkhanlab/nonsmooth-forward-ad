#=
module NonsmoothFwdAD
=====================
A quick implementation of:

- the vector forward mode of automatic differentiation (AD) for generalized
  derivative evaluation, developed in the article:
  KA Khan and PI Barton (2015), https://doi.org/10.1080/10556788.2015.1025400

- the "compass difference" rule for generalized derivatives of scalar-valued bivariate
  functions, developed in the article:
  KA Khan and Y Yuan (2020), https://doi.org/10.46298/jnsao-2020-6061

This implementation applies generalized differentiation rules via operator overloading,
and is modeled on the standard "smooth" vector forward AD mode implementation described
in Chapter 6 of "Evaluating Derivatives (2nd ed.)" by Griewank and Walther (2008).
The "AFloat" struct here is analogous to Griewank and Walther's "adouble" class.

The following operators have been overloaded:
x+y, -x, x-y, x*y, inv(x), x/y, x^p, exp, log, sin, cos, abs, max, min, hypot
(where "p" is a fixed integer)

Additional operators may be overloaded in the same way; a new unary operation would be
overloaded just like "log" or "abs", and a new binary operation would be overloaded
just like "x*y" or "hypot".

Written by Kamil Khan on February 5, 2022
Edited by Maha Chaudhry in April 2022
=#
module NonsmoothFwdAD

using LinearAlgebra

export AFloat

export eval_ld_derivative,
    eval_dir_derivative,
    eval_gen_derivative,
    eval_gen_gradient,
    eval_compass_difference

# default tolerances; feel free to change.
const DEFAULT_ZTOL = 1e-8   # used to decide if we're at the kink of "abs" or "hypot"

struct AFloat
    val::Float64
    dot::Vector{Float64}
    ztol::Float64  # used in "abs" and "hypot" to decide if quantities are 0
end

# outer constructors for when this.dot and/or this.ztol aren't provided explicitly
AFloat(val::Float64, dot::Vector{Float64}) = AFloat(val, dot, DEFAULT_ZTOL)
AFloat(val::Float64, n::Int, ztol::Float64 = DEFAULT_ZTOL) = AFloat(val, zeros(n), ztol)

# display e.g. with "println"
Base.show(io::IO, x::AFloat) = print(io, "(", x.val, "; ", x.dot, ")")

# define promotion from Float64 to AFloat.
# Base.convert can't be used, because it doesn't know the intended dimension
# of the new "dot" vector.
Base.promote(uA::AFloat, uB::Float64) = (uA, AFloat(uB, length(uA.dot), uA.ztol))
Base.promote(uA::Float64, uB::AFloat) = reverse(promote(uB, uA))
Base.promote(uA::AFloat, uB::AFloat) = (uA, uB)

# for u::AFloat, define "u[1]" to mean "u", and set length(u)=1. 
# Helps handle vector/scalar outputs.
Base.getindex(u::AFloat, i::Int) = (i == 1) ? u : throw(DomainError(:i, "must be 1"))
Base.length(u::AFloat) = 1

## define high-level generalized differentiation operations, given a mathematical
## function f composed from supported elemental operations, and written as though
## its input is a Vector

# compute:
#  y = the function value f(x), as a Vector, and
#  yDot = the LD-derivative f'(x; xDot)
#  yDot is the same type as xDot, which is either a Matrix or a Vector{Vector}
function eval_ld_derivative(
    f::Function,
    x::Vector{Float64},
    xDot::Matrix{Float64};
    kwargs...
)
    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = eval_AFloat_vec_output(f, x, xDot; kwargs...)

    # recover outputs from AFloats
    y = [vLD.val for vLD in yLD]
    yDot = reduce(vcat, [vLD.dot for vLD in yLD]')

    return y, yDot
end

function eval_ld_derivative(
    f::Function,
    x::Vector{Float64},
    xDot::Vector{Vector{Float64}};
    kwargs...
)
    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = eval_AFloat_vec_output(f, x, xDot; kwargs...)

    # recover outputs from AFloats
    y = [vLD.val for vLD in yLD]
    yDot = [vLD.dot for vLD in yLD]

    return y, yDot
end

function eval_AFloat_vec_output(
    f::Function,
    x::Vector{Float64},
    xDot::Matrix{Float64};
    ztol::Float64 = DEFAULT_ZTOL
)
    # express inputs as AFloats
    xLD = [AFloat(v, Vector(vDot), ztol) for (v, vDot) in zip(x, eachrow(xDot))]

    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = f(xLD)
    if !(yLD isa Vector)
        yLD = [yLD]
    end

    return yLD
end

function eval_AFloat_vec_output(
    f::Function,
    x::Vector{Float64},
    xDot::Vector{Vector{Float64}};
    ztol::Float64 = DEFAULT_ZTOL
)
    # express inputs as AFloats
    xLD = [AFloat(v, vDot, ztol) for (v, vDot) in zip(x, xDot)]

    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = f(xLD)
    if !(yLD isa Vector)
        yLD = [yLD]
    end

    return yLD
end

# compute:
#   y = the function value f(x), as a vector, and
#   yDot = the directional derivative f'(x; xDot::Vector), as a vector
function eval_dir_derivative(
    f::Function,
    x::Vector{Float64},
    xDot::Vector{Float64};
    kwargs...
)
    xDotMatrix = reshape(xDot, :, 1)
    (y, yDotMatrix) = eval_ld_derivative(f, x, xDotMatrix; kwargs...)
    yDot = vec(yDotMatrix)
    return y, yDot
end

# compute:
#   y = the function value f(x), and
#   yDeriv = the lexicographic derivative D_L f(x; xDot)
#   (xDot defaults to I if not provided, skipping the final linear solve.)
function eval_gen_derivative(
    f::Function,
    x::Vector{Float64},
    xDot::Matrix{Float64};
    kwargs...
)
    (y, yDot) = eval_ld_derivative(f, x, xDot; kwargs...)
    yDeriv = yDot / xDot
    return y, yDeriv
end

function eval_gen_derivative(
    f::Function,
    x::Vector{Float64};
    kwargs...
)
    (y, yDot) = eval_ld_derivative(f, x, Matrix{Float64}(I(length(x))); kwargs...)
    return y, yDot
end

# for scalar-valued f (ordinarily returning a Float64), compute:
#   y = the function value f(x), and
#   yGrad = the lexicographic gradient grad_L f(x; xDot),
#           which is a vector (and the transpose of the lex. derivative)
#   (xDot defaults to I if not provided, skipping the final linear solve.)
function eval_gen_gradient(
    f::Function,
    x::Vector{Float64},
    xDot::Matrix{Float64};
    kwargs...
)
    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = eval_AFloat_vec_output(f, x, xDot; kwargs...)

    # compute generalized gradient element
    return yLD[1].val, (yLD[1].dot'/ xDot)'
end

function eval_gen_gradient(
    f::Function,
    x::Vector{Float64};
    kwargs...
)
    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = eval_AFloat_vec_output(f, x, Matrix{Float64}(I(length(x))); kwargs...)

    return yLD[1].val, yLD[1].dot
end

# for a scalar-valued function f, compute:
#   y = the function value f(x), and
#   yCompass = the compass difference of f at x.
# x must be either a scalar or vector, and yCompass is the same type as x.
function eval_compass_difference(
    f::Function,
    x::Vector{Float64};
    kwargs...
)
    y = f(x)

    (length(y) == 1) ||
        throw(DomainError(:f, "must be scalar-valued"))

    fVec(u) = [f(u)[1]] # account for f returning either a Float64 or Vector{Float64}

    yCompass = copy(x)
    coordVec = zeros(length(x))
    for i in eachindex(x)
        coordVec[i] = 1.0
        _, yDotPlus = eval_dir_derivative(fVec, x, coordVec; kwargs...)
        _, yDotMinus = eval_dir_derivative(fVec, x, -coordVec; kwargs...)
        yCompass[i] = 0.5*(yDotPlus[1] - yDotMinus[1])
        coordVec[i] = 0.0
    end
    return y, yCompass
end

function eval_compass_difference(
    f::Function,
    x::Float64;
    kwargs...
)
    (y, yCompassVec) = eval_compass_difference(f, [x]; kwargs...)
    return y, yCompassVec[1]
end

## define generalized differentiation rules for the simplest elemental operations

# macro to define (::AFloat, ::Float64) and (::Float64, ::AFloat) variants
# of a defined bivariate operation(::AFloat, ::AFloat).
macro define_mixed_input_variants(op)
    return quote
        Base.$op(uA::AFloat, uB::Float64) = $op(promote(uA, uB)...)
        Base.$op(uA::Float64, uB::AFloat) = $op(promote(uA, uB)...)
    end
end

# addition
function Base.:+(uA::AFloat, uB::AFloat)
    vVal = uA.val + uB.val
    vDot = uA.dot + uB.dot
    return AFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants +

# negative and subtraction
Base.:-(u::AFloat) = AFloat(-u.val, -u.dot, u.ztol)

Base.:-(uA::AFloat, uB::AFloat) = uA + (-uB)
@define_mixed_input_variants -

# multiplication
function Base.:*(uA::AFloat, uB::AFloat)
    vVal = uA.val * uB.val
    vDot = (uB.val * uA.dot) + (uA.val * uB.dot)
    return AFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants *

# reciprocal and division
function Base.inv(u::AFloat)
    vVal = inv(u.val)
    vDot = -u.dot / ((u.val)^2)
    return AFloat(vVal, vDot, u.ztol)
end

Base.:/(uA::AFloat, uB::AFloat) = uA * inv(uB)
@define_mixed_input_variants /

# integer powers
function Base.:^(u::AFloat, p::Int)
    if p == 0
        return 0.0*u + 1.0
    else
        vVal = (u.val)^p
        vDot = p * (u.val)^(p-1) * u.dot
        return AFloat(vVal, vDot, u.ztol)
    end
end

# exponential, logarithm, sine, and cosine
Base.exp(u::AFloat) = AFloat(exp(u.val), exp(u.val) * u.dot, u.ztol)

Base.log(u::AFloat) = AFloat(log(u.val), u.dot / u.val, u.ztol)

Base.sin(u::AFloat) = AFloat(sin(u.val), cos(u.val) * u.dot, u.ztol)

Base.cos(u::AFloat) = AFloat(cos(u.val), -sin(u.val) * u.dot, u.ztol)

# absolute value.
# When evaluating zDot, if any quantity "q" has abs(q) < u.ztol,
# then "q" is considered to be 0.
function Base.abs(u::AFloat)
    uVec = [u.val; u.dot]
    k = findfirst(!isapprox(0.0, atol=u.ztol), uVec)
    s = isnothing(k) ? 0.0 : sign(uVec[k])

    return AFloat(abs(u.val), s * u.dot, u.ztol)
end

# bivariate max and min.
function Base.max(uA::AFloat, uB::AFloat)
    diffVec = [uA.val - uB.val; uA.dot - uB.dot] 
    k = findfirst(!isapprox(0.0, atol=uA.ztol), diffVec)

    vVal = max(uA.val, uB.val)
    vDot = isnothing(k) ? zeros(2) :
        (diffVec[k] > 0.0 ? uA.dot : uB.dot)
    return AFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants max

function Base.min(uA::AFloat, uB::AFloat)
    diffVec = [uA.val - uB.val; uA.dot - uB.dot] 
    k = findfirst(!isapprox(0.0, atol=uA.ztol), diffVec)

    vVal = min(uA.val, uB.val)
    vDot = isnothing(k) ? zeros(2) :
        (diffVec[k] < 0.0 ? uA.dot : uB.dot)
    return AFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants min

# sqrt(uA^2 + uB^2); this is LinearAlgebra.hypot in Julia.
# uA.ztol is used as in abs(::AFloat)
function Base.hypot(uA::AFloat, uB::AFloat)
    uVecA = [uA.val; uA.dot]
    uVecB = [uB.val; uB.dot]
    vVec = abs.(uVecA) .+ abs.(uVecB)
    k = findfirst(!isapprox(0.0, atol=uA.ztol), vVec)
    s = isnothing(k) ? zeros(2) : normalize([uVecA[k], uVecB[k]])
    
    vVal = hypot(uA.val, uB.val)
    vDot = s[1] * uA.dot + s[2] * uB.dot
    return AFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants hypot

## overload comparisons to permit simple if-statements, but these conditions' values
## should not change under small perturbations in inputs.
Base.:<(uA::AFloat, uB::AFloat) = (uA.val < uB.val)
@define_mixed_input_variants <

Base.:>(uA::AFloat, uB::AFloat) = (uA.val > uB.val)
@define_mixed_input_variants >

Base.:>=(uA::AFloat, uB::AFloat) = (uA.val >= uB.val)
@define_mixed_input_variants >=

Base.:<=(uA::AFloat, uB::AFloat) = (uA.val <= uB.val)
@define_mixed_input_variants <=

end # module
