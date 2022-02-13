# Nonsmooth vector forward AD
This repository provides an implementation in Julia of the [vector forward mode of automatic differentiation (AD)][1] for generalized derivative evaluation, using operator overloading to carry out generalized differentiation rules. This implementation uses Base.LinearAlgebra, and doesn't depend on any packages external to Julia.

## References
- K.A. Khan and P.I. Barton, [A vector forward mode of automatic differentiation for generalized derivative implementation][1], *Optimization Methods and Software*, 30(6):1185-1212, 2015. DOI:10.1080/10556788.2015.1025400

[1]: https://doi.org/10.1080/10556788.2015.1025400
