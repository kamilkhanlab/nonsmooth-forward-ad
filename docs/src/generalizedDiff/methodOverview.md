# Method Overview 

The standard vector forward mode of automatic differentiation (AD) evaluates derivative-matrix products efficiently for composite smooth functions, and is described by Griewank and Walther (2008). For a composite smooth function **f** of *n* variables, and with derivative **Df**, the vector forward AD mode takes a domain vector **x** and a matrix **M** as input, and produces the product **Df(x) M** as output. To do this, the method regards **f** as a composition of simple elemental functions (such as the arithmetic operations `+`/`-`/`*`/`/` and trigonometric functions), and handles each elemental function using the standard chain rule for differentiation.

For nonsmooth functions, this becomes more complicated. While generalized derivatives such as Clarke's generalized Jacobian are well-defined for continuous functions that are not differentiable everywhere, they have traditionally been considered difficult to evaluate for composite nonsmooth functions, due to failure of classical chain rules. We expect a "correct" generalized derivative to be the actual derivative/gradient when an ostensibly nonsmooth function is in fact smooth, and to be an actual subgradient when the function is convex. Naive extensions of AD to nonsmooth functions, however, do not have these properties.

Khan and Barton (2012, 2013, 2015) showed that the vector forward AD mode can be generalized to handle composite nonsmooth functions, by defining additional calculus rules for elemental nonsmooth functions such as `abs`, `min`, `max`, and the Euclidean norm. These calculus rules are based on a new construction called the "LD-derivative", which is a variant of an earlier construction by Nesterov (2005). The resulting "derivative" in the output derivative-matrix product is a valid generalized derivative for use in methods for nonsmooth optimization and equation-solving, with essentially the same properties as an element of Clarke's generalized Jacobian. In particular:

- if the function is smooth, then this method will compute the actual derivative,
- if the function is convex, then a subgradient will be computed, 
- if no multivariate Euclidean norms are present, then an element of Clarke's generalized Jacobian will be computed,
- in all cases, an element of Nesterov's lexicographic derivative will be computed.

Khan and Barton's nonsmooth vector forward AD mode is also a generalization of a directional derivative evaluation method by Griewank (1994); Griewank's method is recovered when the chosen matrix **M** has only one column. See Barton et al. (2017) for further discussion of applications and extensions of the nonsmooth vector forward AD mode. 

Khan and Yuan (2020) showed that, for bivariate scalar-valued functions that are locally Lipschitz continuous and directionally differentiable, valid generalized derivatives may be constructed by assembling four directional derivative evaluations into a so-called "compass difference", without the LD-derivative calculus required in the nonsmooth vector forward AD mode. 

## References

- KA Khan and PI Barton, [A vector forward mode of automatic differentiation for generalized derivative evaluation](https://doi.org/10.1080/10556788.2015.1025400), *Optimization Methods and Software*, 30(6):1185-1212, 2015. DOI:10.1080/10556788.2015.1025400
- KA Khan and Y Yuan, [Constructing a subgradient from directional derivatives for functions of two variables](https://doi.org/10.46298/jnsao-2020-6061), *Journal of Nonsmooth Analysis and Optimization*, 1:6551, 2020. DOI:10.46298/jnsao-2020-6061
- PI Barton, KA Khan, P Stechlinski, and HAJ Watson, [Computationally relevant generalized derivatives: theory, evaluation, and applications](http://dx.doi.org/10.1080/10556788.2017.1374385), *Optimization Methods and Software*, 33:1030-1072, 2017. DOI:10.1080/10556788.2017.1374385
- A Griewank, Automatic directional differentiation of nonsmooth composite functions, French-German Conference on Optimization, Dijon, 1994.
- A Griewank and A Walther, Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation (2nd ed.), SIAM, 2008.
- Y Nesterov, Lexicographic differentiation of nonsmooth functions, *Mathematical Programming*, 104:669-700, 2005.
