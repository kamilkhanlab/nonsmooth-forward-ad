# ConvexOptimization Implementation Overview

There are three methods implemented in `ConvexOptimization`. 

The first is the **semi-smooth Newton's method** - a technique for solving systems of the form `f(x) = 0`. 

For smooth functions of 1 variable, it involves an iterative calculation of where the tangent line of $x_k$ at $f(x_k)$ intersects the x-axis at $x_{k+1}$. 

The equation for the derivative $∂f(x_k)$ is:

$∂f(x_k) = \frac{f(x_k) - 0}{x_k - x_{k+1}}$

Solving for $x_{k+1}$ provides:

$x_{k+1} = x_k + \frac{f(x_k)}{∂f(x_k)}$

And $x_k$ is re-calculated until $f(x_k)$ = 0. 

For semi-smooth functions:

- Of 1 variable, $∂f(x_k)$ is the generalized derivative as calculated by `eval_gen_derivative(f,x)`.
- Of more than 1 variable, $∂f(x_k)$ is the generalized gradient as calculated by `eval_gen_gradient(f,x)`.

The second implemented method was the **LP-Newton method** proposed by [Facchinei, Fischer, and Herrich (2013)](https://doi.org/10.1007/s10107-013-0676-6) for the convergence to a local solution for a constrained system of equations. It removes some of the original assumptions required by the original Newton's method (e.g., convexity). It is particular advantageous for KKT systems derived from optimality conditions for constrained optimization or variational inequalities.

For systems of the form `F(z) = 0` where `z ∈ Ω` where `Ω` is a nonempty closed solution set - either whole space or polyhedral. It involves iteratively solving the following optimization problem for ($z_k, γ_k$) to convergence:

min $_{z_{k+1},γ_{k+1}} (γ)$, 

s.t. $z ∈ Ω$

 $||F(z_k) + G(z_k) \cdot (z_{k+1} - z_{k}) || ≤ γ_{k+1}||F(z_k)||^2$,

 $||z_{k+1} - z_k|| ≤ || F(z_k) ||$,

$ γ_{k+1} ≥ 0$

Each iteration of the problem is solved using `Ipopt.jl` through Julia's `JuMP`. Note that the terms $F(z_k)$ and $G(z_k)$ are calculated using `eval_gen_derivative(f, z)` for each iteration. 

The last method is the **Level method** for convex, nonsmooth minimizations. It minimizes `f(x)` for `x` by iteratively solving (1) an LP for the function subgradient and (2) a QP for x by minimizing the Euclidean norm. 

1. An LP for the function subgradient: 

min $t$,

 s.t. $f(x_i )+〈g(x_i ),x-x_i 〉≤t,i=0,…,k$

 $x ∈ Q$ 

where $x$ is $x^*$ and $t$ is $\^f{}^*(x)$

2. A QP for x by minimizing the Euclidean norm:

min $ ||x-x_k ||^2$,

 s.t. $f(x_i )+〈g(x_i ),x-x_i 〉≤l_k (\alpha),i=0,…,k$

 $x ∈ Q$

where $l_k(\alpha)$ is $(1-\alpha)\^f{}^*(x) + \alpha\^f{}^*(x)$

