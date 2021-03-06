# nonsmooth-forward-ad
The `NonsmoothFwdAD` module in [NonsmoothFwdAD.jl](src/NonsmoothFwdAD.jl) provides an implementation in Julia of two of our recent methods for evaluating generalized derivatives of nonsmooth functions:

- the [vector forward mode of automatic differentiation (AD)][1] for composite functions, and
- the [compass difference rule][2] for bivariate scalar-valued functions.

The methods here apply to continuous functions that are finite compositions of simple "scientific calculator" operations, but may be nonsmooth. Operator overloading is used to automatically apply generalized differentiation rules to each of these simple operations. The computed generalized derivatives preserve many of the beneficial properties of derivatives of smooth functions; [see below](#method-overview) for more details.

This implementation doesn't depend on any packages external to Julia.

## Example

The script [test.jl](test/test.jl) applies `NonsmoothFwdAD` to several examples for illustration; here is one of them. Further implementation details are provided [below](#implementation-overview).

Consider the following nonsmooth function of two variables, to replicate Example 6.2 from Khan and Barton (2013):
```julia
f(x) = max(min(x[1], -x[2]), x[2] - x[1])
```
Using the `NonsmoothFwdAD` module (after `include("NonsmoothFwdAD.jl")` and `using .NonsmoothFwdAD`), we may evaluate a value `y` and a generalized gradient element `yGrad` of `f` at `[0.0, 0.0]` by the following alternative approaches, using the nonsmooth vector forward mode of AD.

- By defining `f` beforehand:
	```julia
	f(x) = max(min(x[1], -x[2]), x[2] - x[1])
	y, yGrad = eval_gen_gradient(f, [0.0, 0.0])
	```
- By defining `f` as an anonymous function:
	```julia
	y, yGrad = eval_gen_gradient([0.0, 0.0]) do x
	    return max(min(x[1], -x[2]), x[2] - x[1])
	end	
	```

Here, `eval_gen_gradient` constructs `yGrad` as a `Vector{Float64}`, and only applies to scalar-valued functions. For vector-valued functions, `eval_gen_derivative` instead produces a generalized derivative element `yDeriv::Matrix{Float64}`.

For scalar-valued functions of one or two variables, the "compass difference" is guaranteed to be an element of Clarke's generalized gradient. We may calculate the compass difference `yCompass::Vector{Float64}` for the above function `f` at `[0.0, 0.0]` as follows:
```julia
_, yCompass = eval_compass_difference([0.0, 0.0]) do x
    return max(min(x[1], -x[2]), x[2] - x[1])
end	
```

## Method overview

The standard vector forward mode of automatic differentiation (AD) evaluates derivative-matrix products efficiently for composite smooth functions, and is described by Griewank and Walther (2008). For a composite smooth function **f** of *n* variables, and with derivative **Df**, the vector forward AD mode takes a domain vector **x** and a matrix **M** as input, and produces the product **Df(x) M** as output. To do this, the method regards **f** as a composition of simple elemental functions (such as the arithmetic operations `+`/`-`/`*`/`/` and trigonometric functions), and handles each elemental function using the standard chain rule for differentiation.

For nonsmooth functions, this becomes more complicated. While generalized derivatives such as Clarke's generalized Jacobian are well-defined for continuous functions that are not differentiable everywhere, they have traditionally been considered difficult to evaluate for composite nonsmooth functions, due to failure of classical chain rules. We expect a "correct" generalized derivative to be the actual derivative/gradient when an ostensibly nonsmooth function is in fact smooth, and to be an actual subgradient when the function is convex. Naive extensions of AD to nonsmooth functions, however, do not have these properties.

Khan and Barton (2012, 2013, 2015) showed that the vector forward AD mode can be generalized to handle composite nonsmooth functions, by defining additional calculus rules for elemental nonsmooth functions such as `abs`, `min`, `max`, and the Euclidean norm. These calculus rules are based on a new construction called the "LD-derivative", which is a variant of an earlier construction by Nesterov (2005). The resulting "derivative" in the output derivative-matrix product is a valid generalized derivative for use in methods for nonsmooth optimization and equation-solving, with essentially the same properties as an element of Clarke's generalized Jacobian. In particular:

- if the function is smooth, then this method will compute the actual derivative,
- if the function is convex, then a subgradient will be computed, 
- if no multivariate Euclidean norms are present, then an element of Clarke's generalized Jacobian will be computed,
- in all cases, an element of Nesterov's lexicographic derivative will be computed.

Khan and Barton's nonsmooth vector forward AD mode is also a generalization of a directional derivative evaluation method by Griewank (1994); Griewank's method is recovered when the chosen matrix **M** has only one column. See Barton et al. (2017) for further discussion of applications and extensions of the nonsmooth vector forward AD mode. 

Khan and Yuan (2020) showed that, for bivariate scalar-valued functions that are locally Lipschitz continuous and directionally differentiable, valid generalized derivatives may be constructed by assembling four directional derivative evaluations into a so-called "compass difference", without the LD-derivative calculus required in the nonsmooth vector forward AD mode. 

## Implementation overview
The provided `NonsmoothFwdAD` module in [NonsmoothFwdAD.jl](src/NonsmoothFwdAD.jl) performs both the nonsmooth vector forward AD mode and compass difference evaluation mentioned above, using operator overloading to carry out nonsmooth calculus rules automatically and efficiently. The implementation is structured just like the archetypal vector forward AD mode described by Griewank and Walther (2008), but with additional handling of nonsmooth functions.

### Usage
The script [test.jl](test/test.jl) illustrates the usage of `NonsmoothFwdAD`, and evaluates generalized derivatives for several nonsmooth functions. This module defines calculus rules for the following scalar-valued elemental operations via operator operloading:
```julia
+, -, *, inv, /, ^, exp, log, sin, cos, abs, min, max, hypot
```
The differentiation methods of this module apply to compositions of these operations. Additional univariate operations may be included by adapting the handling of `log` or `abs`, and additional bivariate operations may be included by adapting the handling of `*` or `hypot`. The overloaded `^` operation only supports integer-valued exponents; rewrite non-integer exponents as e.g. `x^y = exp(y*log(x))`.

### Exported functions
The following functions are exported by `NonsmoothFwdAD`. Except where noted, the provided function `f` must be composed from the above operations, and must be written so that its input and output are both generic `Vector{T}`s, with `T` standing in for `Float64`. In each case, `f` may be provided as an anonymous function using Julia's `do` syntax.

- `(y, yDot) = eval_dir_derivative(f::Function, x::Vector{Float64}, xDot::Vector{Float64})`:

	- evaluates `y = f(x)` and the directional derivative `yDot = f'(x; xDot)` according to Griewank (1994). Both `y` and `yDot` are computed as `Vector{Float64}`s.
	
- `(y, yDeriv) = eval_gen_derivative(f::Function, x::Vector{Float64}, xDot::Matrix{Float64})`:

	- evaluates `y = f(x)` and a generalized derivative `yDeriv = Df(x; xDot)` according to the nonsmooth vector forward AD mode, analogous to the derivative `Df(x)` computed by the smooth vector forward AD mode. `xDot` must have full row rank, and defaults to `I` if not provided. `yDeriv` is a `Matrix{Float64}`.
	
- `(y, yGrad) = eval_gen_gradient(f::Function, x::Vector{Float64}, xDot::Matrix{Float64})`:

	- just like `eval_gen_derivative`, except applies only to scalar-valued functions `f`, and produces the generalized gradient `yGrad = yDeriv'` as output. `yGrad` is a `Vector{Float64}` rather than a `Matrix`.

- `(y, yDot) = eval_ld_derivative(f::Function, x::Vector{Float64}, xDot)`:

	- evaluates `y = f(x)` and the LD-derivative `yDot = f'(x; xDot)`, analogous to the output `Df(x)*xDot` of the smooth vector forward AD mode. `xDot` may be either a `Matrix{Float64}` or a `Vector{Vector{Float64}}`, and the output `yDot` is constructed to be the same type as `xDot`. This is used in methods that require LD-derivatives, such as when computing generalized derivatives for ODE solutions according to Khan and Barton (2014).
	
- `(y, yCompass) = eval_compass_difference(f::Function, x::Vector{Float64})`:	

	- evaluates `y = f(x)` and the compass difference `yCompass` of a scalar-valued function `f` at `x`. If `f` has a domain dimension of 1 or 2, then `yCompass` is guaranteed to be an element of Clarke's generalized gradient.
	
#### Handling nonsmoothness

The nonsmooth calculus rules used here are described by Khan and Barton (2015) and Barton et al. (2017). In particular, they require knowledge of when a nonsmooth elemental function like `abs` is exactly at its "kink" or not, which is difficult using floating point arithmetic. This implementation, by default, considers any domain point within an absolute tolerance of `1e-08` of a kink to be at that kink. In all of the exported functions listed above, this tolerance may be edited via a keyword argument `ztol`. For example, when using `eval_gen_gradient`, we could write:
```julia
y, yGrad = eval_gen_gradient(f, x, ztol=1e-5)
```
	
### Operator overloading

Analogous to the `adouble` class described by Griewank and Walther (2008), this implementation effectively replaces any `Float64` quantity with an `AFloat` object that holds generalized derivative information. This class can be initialized and used directly. An `AFloat` has three fields:

- `val::Float64`: its output value
- `dot::Vector{Float64}`: its output generalized derivative
- `ztol::Float64`: the capture radius for nonsmooth operations with kinks. Any quantity within `ztol` of a kink is considered to be at that kink. Default value: `1e-08`.

## References
- KA Khan and PI Barton, [A vector forward mode of automatic differentiation for generalized derivative evaluation][1], *Optimization Methods and Software*, 30(6):1185-1212, 2015. DOI:10.1080/10556788.2015.1025400
- KA Khan and Y Yuan, [Constructing a subgradient from directional derivatives for functions of two variables][2], *Journal of Nonsmooth Analysis and Optimization*, 1:6551, 2020. DOI:10.46298/jnsao-2020-6061
- PI Barton, KA Khan, P Stechlinski, and HAJ Watson, [Computationally relevant generalized derivatives: theory, evaluation, and applications][3], *Optimization Methods and Software*, 33:1030-1072, 2017. DOI:10.1080/10556788.2017.1374385
- A Griewank, Automatic directional differentiation of nonsmooth composite functions, French-German Conference on Optimization, Dijon, 1994.
- A Griewank and A Walther, Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation (2nd ed.), SIAM, 2008.
- Y Nesterov, Lexicographic differentiation of nonsmooth functions, *Mathematical Programming*, 104:669-700, 2005.


[1]: https://doi.org/10.1080/10556788.2015.1025400
[2]: https://doi.org/10.46298/jnsao-2020-6061
[3]: http://dx.doi.org/10.1080/10556788.2017.1374385
