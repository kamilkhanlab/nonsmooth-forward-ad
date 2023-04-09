# Implementation Overview

The provided `GeneralizedDiff` module in [GeneralizedDiff.jl](src/GeneralizedDiff.jl) performs both the nonsmooth vector forward AD mode and compass difference evaluation mentioned above, using operator overloading to carry out nonsmooth calculus rules automatically and efficiently. The implementation is structured just like the archetypal vector forward AD mode described by Griewank and Walther (2008), but with additional handling of nonsmooth functions.

## Usage

The script [test.jl](https://github.com/kamilkhanlab/nonsmooth-forward-ad/blob/main/test/test.jl) illustrates the usage of `GeneralizedDiff`, and evaluates generalized derivatives for several nonsmooth functions. This module defines calculus rules for the following scalar-valued elemental operations via operator operloading:
```julia
+, -, *, inv, /, ^, exp, log, sin, cos, abs, min, max, hypot
```
The differentiation methods of this module apply to compositions of these operations. Additional univariate operations may be included by adapting the handling of `log` or `abs`, and additional bivariate operations may be included by adapting the handling of `*` or `hypot`. The overloaded `^` operation only supports integer-valued exponents; rewrite non-integer exponents as e.g. `x^y = exp(y*log(x))`.

## Handling nonsmoothness

The nonsmooth calculus rules used here are described by Khan and Barton (2015) and Barton et al. (2017). In particular, they require knowledge of when a nonsmooth elemental function like `abs` is exactly at its "kink" or not, which is difficult using floating point arithmetic. This implementation, by default, considers any domain point within an absolute tolerance of `1e-08` of a kink to be at that kink. In all of the exported functions listed above, this tolerance may be edited via a keyword argument `ztol`. For example, when using `eval_gen_gradient`, we could write:
```julia
y, yGrad = eval_gen_gradient(f, x, ztol=1e-5)
```

## References

- KA Khan and PI Barton, [A vector forward mode of automatic differentiation for generalized derivative evaluation](https://doi.org/10.1080/10556788.2015.1025400), *Optimization Methods and Software*, 30(6):1185-1212, 2015. DOI:10.1080/10556788.2015.1025400
- PI Barton, KA Khan, P Stechlinski, and HAJ Watson, [Computationally relevant generalized derivatives: theory, evaluation, and applications](http://dx.doi.org/10.1080/10556788.2017.1374385), *Optimization Methods and Software*, 33:1030-1072, 2017. DOI:10.1080/10556788.2017.1374385
- A Griewank and A Walther, Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation (2nd ed.), SIAM, 2008.
