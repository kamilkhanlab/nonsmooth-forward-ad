# Method Overview
 
Suppose we have a **composite**, non-smooth function **`f`** of `n` variables with derivative **`Df`**. 

The following [implementation](https://www.tandfonline.com/doi/full/10.1080/10556788.2015.1025400) by Khan and Barton (2015) provides vector-forward AD evaluation methods for a generalized derivative **`Df(x)M`** of `f` on input domain vector **`x`** using a chosen matrix **`M`** through operator overloading.

The overloading defines additional calculus for elemental non-smooth functions, like `abs`, `min`, or `max`, based on “LD-derivative” constructions. This requires knowledge of when said elemental functions are at “kinks” – non-differentiable points. 

The resulting output derivative-matrix product is a valid generalized derivative with the same properties of an element of a Clarke’s generalized Jacobian. 

An additional “compass difference” evaluation procedure was added for bivariate scalar-valued functions. [Khan and Yuan](https://doi.org/10.46298/jnsao-2020-6061) (2020) show generalized derivatives can be constructed using four directional derivative evaluations in a “compass difference” without the usual LD-derivative calculus. 