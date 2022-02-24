# nonsmooth-forward-ad
The `NonsmoothFwdAD` module in [NonsmoothFwdAD.jl](src/NonsmoothFwdAD.jl) provides an implementation in Julia of two recent methods for generalized derivative evaluation:

- the [vector forward mode of automatic differentiation (AD)][1] for composite functions, and
- the [compass difference rule][2] for bivariate scalar-valued functions.

These methods apply to continuous functions that are finite compositions of simple "scientific calculator" operations, but may be nonsmooth. Operator overloading is used to automatically apply generalized differentiation rules to each of these simple operations. This implementation doesn't depend on any packages external to Julia.

## Method overview
to be written

## Usage
The script [test.jl](test/test.jl) illustrates the usage of `NonsmoothFwdAD`, and evaluates generalized derivatives for several nonsmooth functions.

## References
- KA Khan and PI Barton, [A vector forward mode of automatic differentiation for generalized derivative evaluation][1], *Optimization Methods and Software*, 30(6):1185-1212, 2015. DOI:10.1080/10556788.2015.1025400
- KA Khan and Y Yuan, [Constructing a subgradient from directional derivatives for functions of two variables][2], *Journal of Nonsmooth Analysis and Optimization*, 1:6551, 2020. DOI:10.46298/jnsao-2020-6061

[1]: https://doi.org/10.1080/10556788.2015.1025400
[2]: https://doi.org/10.46298/jnsao-2020-6061
