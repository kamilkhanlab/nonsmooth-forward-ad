#=
module NonsmoothFwdAD
=====================
A quick implementation of the vector forward mode of automatic differentiation (AD) for generalized derivative evaluation, developed in the article:

- KA Khan and PI Barton (2015), https://doi.org/10.1080/10556788.2015.1025400

This implementation applies generalized differentiation rules via operator overloading, and is modeled on the standard "smooth" vector forward AD mode implementation described in Chapter 6 of "Evaluating Derivatives (2nd ed.)" by Griewank and Walther (2008). The "LFloat" struct here is analogous to Griewank and Walther's "adouble" class. 

The following operators have been overloaded:
x+y, -x, x-y, x*y, inv(x), x/y, x^p, exp, log, sin, cos, abs, max, min, hypot
(where "p" is a fixed integer)

Additional operators may be overloaded in the same way; a new unary operation would be overloaded just like "log" or "abs", and a new binary operation would be overloaded just like "x*y" or "hypot".

Written by Kamil Khan on February 5, 2022
=#
module NonsmoothFwdAD

using LinearAlgebra

export LFloat

export eval_ld_derivative,
    eval_dir_derivative,
    eval_gen_derivative,
    eval_gen_gradient,
    eval_compass_difference

# default tolerances; feel free to change.
const DEFAULT_ZTOL = 1e-8   # used to decide if we're at the kink of "abs" or "hypot"

struct LFloat
    val::Float64
    dot::Vector{Float64}
    ztol::Float64  # used in "abs" and "hypot" to decide if quantities are 0
end

# outer constructors for when this.dot and/or this.ztol aren't provided explicitly
LFloat(val::Float64, dot::Vector{Float64}) = LFloat(val, dot, DEFAULT_ZTOL)
LFloat(val::Float64, n::Int, ztol::Float64 = DEFAULT_ZTOL) = LFloat(val, zeros(n), ztol)

# display e.g. with "println"
Base.show(io::IO, x::LFloat) = print(io, "(", x.val, "; ", x.dot, ")") 

# define promotion from Float64 to LFloat.
# Base.convert can't be used, because it doesn't know the intended dimension
# of the new "dot" vector.
Base.promote(uA::LFloat, uB::Float64) = (uA, LFloat(uB, length(uA.dot), uA.ztol))
Base.promote(uA::Float64, uB::LFloat) = reverse(promote(uB, uA))
Base.promote(uA::LFloat, uB::LFloat) = (uA, uB)

## define high-level generalized differentiation operations, given a mathematical function f composed from supported elemental operations, and written as though its input is a Vector

# compute:
#  y = the function value f(x), as a Vector, and
#  yDot = the LD-derivative f'(x; xDot)
#  yDot is the same type as xDot, which is either a Matrix or a Vector{Vector} 
function eval_ld_derivative(f, x::Vector{Float64}, xDot::Matrix{Float64})
    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = eval_LFloat_vec_output(f, x, xDot)

    # recover outputs from LFloats
    y = [vLD.val for vLD in yLD]
    yDot = reduce(vcat, [vLD.dot for vLD in yLD]')
    
    return y, yDot
end

function eval_ld_derivative(f, x::Vector{Float64}, xDot::Vector{Vector{Float64}})
    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = eval_LFloat_vec_output(f, x, xDot)

    # recover outputs from LFloats
    y = [vLD.val for vLD in yLD]
    yDot = [vLD.dot for vLD in yLD]
    
    return y, yDot
end

function eval_LFloat_vec_output(f, x::Vector{Float64}, xDot::Matrix{Float64})
    # express inputs as LFloats            
    xLD = [LFloat(v, Vector(vDot)) for (v, vDot) in zip(x, eachrow(xDot))]

    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = f(xLD)
    if !(yLD isa Vector)
        yLD = [yLD]
    end

    return yLD
end

function eval_LFloat_vec_output(f, x::Vector{Float64}, xDot::Vector{Vector{Float64}})
    # express inputs as LFloats
    xLD = map(LFloat, x, xDot)

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
function eval_dir_derivative(f, x::Vector{Float64}, xDot::Vector{Float64})
    xDotMatrix = reshape(xDot, :, 1)
    (y, yDotMatrix) = eval_ld_derivative(f, x, xDotMatrix)
    yDot = vec(yDotMatrix)
    return y, yDot
end

# compute:
#   y = the function value f(x), and
#   yDeriv = the lexicographic derivative D_L f(x; xDot)
#   (xDot defaults to I if not provided)
function eval_gen_derivative(
    f,
    x::Vector{Float64},
    xDot::Matrix{Float64} = Matrix{Float64}(I(length(x)))
)
    (y, yDot) = eval_ld_derivative(f, x, xDot)
    yDeriv = yDot / xDot
    return y, yDeriv
end

# for scalar-valued f (ordinarily returning a Float64), compute:
#   y = the function value f(x), and
#   yGrad = the lexicographic gradient grad_L f(x; xDot),
#           which is a vector (and the transpose of the lex. derivative)
#   (xDot defaults to I if not provided)
function eval_gen_gradient(
    f,
    x::Vector{Float64},
    xDot::Matrix{Float64} = Matrix{Float64}(I(length(x)))
)
    # use operator overloading to compute f(x) and f'(x; xDot)
    yLD = eval_LFloat_vec_output(f, x, xDot)

    # compute generalized gradient element
    return yLD.val, (yLD.dot' / xDot)'
end

# for a scalar-valued function f, compute:
#   y = the function value f(x), and
#   yCompass = the compass difference of f at x.
# x must be either a scalar or vector, and yCompass is the same type as x.
function eval_compass_difference(f, x::Vector{Float64})
    y = f(x)
    if y isa Float64
        fVec(u) = [f(u)]
    else
        fVec(u) = f(u)
    end
    (length(y) == 1) || throw(DomainError("f; this function is not scalar-valued"))
    yCompass = copy(x)
    coordVec = zeros(length(x))
    for i in eachindex(x)
        coordVec[i] = 1.0
        _, yDotPlus = eval_dir_derivative(fVec, x, coordVec)
        _, yDotMinus = eval_dir_derivative(fVec, x, -coordVec)
        yCompass[i] = 0.5*(yDotPlus[1] - yDotMinus[1])
        coordVec[i] = 0.0
    end
    return y, yCompass
end

function eval_compass_difference(f, x::Float64)
    (y, yCompassVec) = eval_compass_difference(f, [x])
    return y, yCompassVec[1]
end

## define generalized differentiation rules for the simplest elemental operations

# macro to define (::LFloat, ::Float64) and (::Float64, ::LFloat) variants
# of a defined bivariate operation(::LFloat, ::LFloat).
macro define_mixed_input_variants(op)
    return quote
        Base.$op(uA::LFloat, uB::Float64) = $op(promote(uA, uB)...)
        Base.$op(uA::Float64, uB::LFloat) = $op(promote(uA, uB)...)
    end
end

# addition
function Base.:+(uA::LFloat, uB::LFloat)
    vVal = uA.val + uB.val
    vDot = uA.dot + uB.dot
    return LFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants +

# negative and subtraction
Base.:-(u::LFloat) = LFloat(-u.val, -u.dot, u.ztol)

Base.:-(uA::LFloat, uB::LFloat) = uA + (-uB)
@define_mixed_input_variants -

# multiplication
function Base.:*(uA::LFloat, uB::LFloat)
    vVal = uA.val * uB.val
    vDot = (uB.val * uA.dot) + (uA.val * uB.dot)
    return LFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants *

# reciprocal and division
function Base.inv(u::LFloat)
    vVal = inv(u.val)
    vDot = -u.dot / ((u.val)^2)
    return LFloat(vVal, vDot, u.ztol)
end

Base.:/(uA::LFloat, uB::LFloat) = uA * inv(uB)
@define_mixed_input_variants /

# integer powers
function Base.:^(u::LFloat, p::Int)
    if p == 0
        return 0.0*u + 1.0
    else
        vVal = (u.val)^p
        vDot = p * (u.val)^(p-1) * u.dot
        return LFloat(vVal, vDot, u.ztol)
    end
end

# exponential, logarithm, sine, and cosine
Base.exp(u::LFloat) = LFloat(exp(u.val), exp(u.val) * u.dot, u.ztol)

Base.log(u::LFloat) = LFloat(log(u.val), u.dot / u.val, u.ztol)

Base.sin(u::LFloat) = LFloat(sin(u.val), cos(u.val) * u.dot, u.ztol)

Base.cos(u::LFloat) = LFloat(cos(u.val), -sin(u.val) * u.dot, u.ztol)

# absolute value.
# When evaluating zDot, if any quantity "q" has abs(q) < u.ztol,
# then "q" is considered to be 0.
function Base.abs(u::LFloat)
    vVal = abs(u.val)
    if vVal > u.ztol
        s = sign(u.val)
    else
        s = 0.0
        for d in u.dot
            if abs(d) > u.ztol
                s = sign(d)
                break
            end
        end
    end
    vDot = s * u.dot
    return LFloat(vVal, vDot, u.ztol)
end

# bivariate max and min.
# Uses the identity max(uA,uB) = 0.5*(uA+uB+abs(uA-uB)), and similarly for min
Base.max(uA::LFloat, uB::LFloat) = 0.5*(uA + uB + abs(uA - uB))
@define_mixed_input_variants max

Base.min(uA::LFloat, uB::LFloat) = 0.5*(uA + uB - abs(uA - uB))
@define_mixed_input_variants min

# sqrt(uA^2 + uB^2); this is LinearAlgebra.hypot in Julia.
# uA.ztol is used as in abs(::LFloat)
function Base.hypot(uA::LFloat, uB::LFloat)
    vVal = hypot(uA.val, uB.val)
    if vVal > uA.ztol
        sA = uA.val / vVal
        sB = uB.val / vVal
    else
        sA = 0.0
        sB = 0.0
        for (dA, dB) in zip(uA.dot, uB.dot)
            sNorm = hypot(dA, dB)
            if sNorm > uA.ztol
                sA = dA / sNorm
                sB = dB / sNorm
                break
            end
        end
    end
    vDot = sA * uA.dot + sB * uB.dot
    return LFloat(vVal, vDot, uA.ztol)
end
@define_mixed_input_variants hypot

end # module
