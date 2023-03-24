# **NonSmoothFwdAD Documentation**

## Installation and Usage

NonSmoothFwdAD is currently not a registered Julia package. To download from the Julia REPL, type `]` to access Pkg REPL mode and then run the following command:

```Julia
add https://github.com/kamilkhanlab/nonsmooth-forward-ad 
```

Then, to use the package:

```Julia
using NonSmoothFwdAD
```

## Example

The usage of `NonSmoothFwdAD` is demonstrated by scripts [test.jl](test/test.jl) and [convexTest.jl](test/convexTest.jl). 

#### GeneralizedDiff

Consider the following nonsmooth function of two variables, to replicate Example 6.2 from Khan and Barton (2013):

```Julia
f(x) = max(min(x[1], -x[2]), x[2] - x[1])
```

Using the `NonsmoothFwdAD` module (after `include("NonsmoothFwdAD.jl")` and using `.NonsmoothFwdAD`, `.GeneralizedDiff`), we may evaluate a value `y` and a generalized gradient element `yGrad` of `f` at `[0.0, 0.0]` by the following alternative approaches, using the nonsmooth vector forward mode of AD.

- By defining f beforehand:

```Julia
    f(x) = max(min(x[1], -x[2]), x[2] - x[1])
    y, yGrad = eval_gen_gradient(f, [0.0, 0.0])

    #output 
    y = 0.0
    yGrad = [0.0, -1.0]
```

- By defining f as an anonymous function:

```Julia
    y, yGrad = eval_gen_gradient([0.0, 0.0]) do x
        return max(min(x[1], -x[2]), x[2] - x[1])
    end	

    #output 
    y = 0.0
    yGrad = [0.0, -1.0]
```

Here, `eval_gen_gradient` constructs `yGrad` as a `Vector{Float64}`, and only applies to scalar-valued functions. For vector-valued functions, `eval_gen_derivative` instead produces a generalized derivative element `yDeriv::Matrix{Float64}`.

For scalar-valued functions of one or two variables, the "compass difference" is guaranteed to be an element of Clarke's generalized gradient. We may calculate the compass difference `yCompass::Vector{Float64}` for the above function `f` at `[0.0, 0.0]` as follows:

```Julia
_, yCompass = eval_compass_difference([0.0, 0.0]) do x
    return max(min(x[1], -x[2]), x[2] - x[1])
end	
```

#### ConvexOptimization

Consider the following optimization problem, replicating Example 6 in F. Facchinei et. al (2014): 

```
min PHI(x) = (x[1] + x[2])*x[4] + 0.5*(x[2] + x[3])^2 
    s.t.    x[1] <= 0
            x[2] >= 1
            x[4] >= 0
            x[1] + x[2] + x[3] >=0 
```

The provided non-smooth reformulation to the Karush-Kuhn-Tucker system is as follows: 

```Julia
uOffset = 4
vOffset = 9
f(x) = [x[4] + x[1+uOffset] - x[2+uOffset] - x[5+uOffset],
    x[4] + x[2] + x[3] - x[3+uOffset] - x[5+uOffset],
    x[2] + x[3] - x[5+uOffset],
    x[1] + x[2] - x[4+uOffset],
    x[1] + x[1+vOffset],
    - x[1] + x[2+vOffset],
    1.0 - x[2] + x[3+vOffset],
    - x[4] + x[4+vOffset],
    - x[1] - x[2] - x[3] + x[5+vOffset],
    min(x[1+uOffset], x[1+vOffset]),
    min(x[2+uOffset], x[2+vOffset]),
    min(x[3+uOffset], x[3+vOffset]),
    min(x[4+uOffset], x[4+vOffset]),
    min(x[5+uOffset], x[5+vOffset])] 
```

Using the `NonsmoothFwdAD` module (after `include("NonsmoothFwdAD.jl")` and using `.NonsmoothFwdAD`, `.GeneralizedDiff`, `.ConvexOptimization`), the LPNewton method can locate the minima of `(x, PHI(x))` by solving `f(x) = 0` given no other binding set constraints. Assume an initial guess of `x0` for `f`.

```Julia
x0 = [1.0, 4.0, -2.0, 1.0,
    3.0, 3.0, 1.0, 4.0, 1.0,
    0.0, 1.0, 3.0, 1.0, 3.0]
x, _, gamma = LPNewton(f, x0)

#output

x = [0.0, 3.32, -3.32, 0.0, 
    2.76, 2.76, 0.0, 3.32, 0.0, 
    0.0, 0.0, 2.32, 0.0, 0.0]
gamma = 4.99
```

Thus, the solution for `PHI(x)` is `(x = [0.0, 3.32, -3.32, 0.0], PHI(x) = 0.0)`. 

Note that the solution set of the optimization problem `PHI(x)` is `X = {(0, t, −t, 0)|t ≥ 1}`. Different initial guesses `x0` could produce different local optima for `PHI(x)` where `f(x) = 0`. This is just one potential solution. 

## Authors

- Maha Chaudhry, Department of Chemical Engineering, McMaster University
- Kamil Khan, Department of Chemical Engineering, McMaster University

## References

- KA Khan and PI Barton, [A vector forward mode of automatic differentiation for generalized derivative evaluation](https://doi.org/10.1080/10556788.2015.1025400), Optimization Methods and Software, 30(6):1185-1212, 2015. DOI:10.1080/10556788.2015.1025400
- KA Khan and Y Yuan, [Constructing a subgradient from directional derivatives for functions of two variables, Journal of Nonsmooth Analysis and Optimization](https://doi.org/10.46298/jnsao-2020-6061), 1:6551, 2020. DOI:10.46298/jnsao-2020-6061
- PI Barton, KA Khan, P Stechlinski, and HAJ Watson, [Computationally relevant generalized derivatives: theory, evaluation, and applications](http://doi.org/10.1080/10556788.2017.1374385), Optimization Methods and Software, 33:1030-1072, 2017. DOI:10.1080/10556788.2017.1374385
- A Griewank, Automatic directional differentiation of nonsmooth composite functions, French-German Conference on Optimization, Dijon, 1994.
- A Griewank and A Walther, Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation (2nd ed.), SIAM, 2008.
- Y Nesterov, Lexicographic differentiation of nonsmooth functions, Mathematical Programming, 104:669-700, 2005.
- F Facchinei, A Fischer, M Herrich, [An LP-Newton method: nonsmooth equations, KKT systems, and nonisolated solutions](https://doi.org/10.1007/s10107-013-0676-6), Mathematical Programming, 146:1-36, 2014, DOI: 10.1007/s10107-013-0676-6