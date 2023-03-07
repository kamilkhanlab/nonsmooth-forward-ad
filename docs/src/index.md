# **NonSmoothFwdAD Documentation**

## Installation and Usage

NonSmoothFwdAD is currently not a registered Julia package. To download from the Julia REPL, type `]` to access Pkg REPL mode and then run the following command:

```Julia
add https://github.com/kamilkhanlab/nonsmooth-forward-ad 
```

Then, to use the package:

```Julia
using ConvexSampling
```

## Example

The usage of `NonSmoothFwdAD` is demonstrated by scripts [test.jl](test/test.jl) and [convexTest.jl](test/convexTest.jl). 

Replicating Example 6 in F. Facchinei et. al (2014) for the given optimization problem.

```
min(x[1] + x[2])*x[4] + 0.5*(x[2] + x[3])^2 
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

For an initial guess of `z0 = [1.0, 4.0, -2.0, 1.0,
    3.0, 3.0, 1.0, 4.0, 1.0,
    0.0, 1.0, 3.0, 1.0, 3.0]` 
and no other binding set constraints, the LPNewton method locates the minima `(z, f(z))` by solving `f(z)=0`. 

```Julia
z0 = [1.0, 4.0, -2.0, 1.0,
    3.0, 3.0, 1.0, 4.0, 1.0,
    0.0, 1.0, 3.0, 1.0, 3.0]
z, _, gamma = LPNewton(f, z0)

#output

z = [0.0, 3.32, -3.32, 0.0, 
    2.76, 2.76, 0.0, 3.32, 0.0, 
    0.0, 0.0, 2.32, 0.0, 0.0]
gamma = 4.99
```

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