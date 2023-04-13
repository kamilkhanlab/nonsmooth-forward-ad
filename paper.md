---
title: 'NonSmoothFwdAD: A Julia package for nonsmooth differentiation'
tags: 
- Julia
- Nonsmooth Optimization
authors:
 - name: Kamil A. Khan
   orcid: 0000-0000-0000-0000
   equal-contrib: true
   affiliation: "1, 2"
 - name: Maha Chaudhry
   equal-contrib: 
   affiliation: 
affilations: 
 - name: X, University, Country
   index: 1
date: April 9 2023
bibliography: paper.bib
---

# Summary

Mathematically, non-smooth functions are those that are not infinitely continuously differentiable. Since classical optimization assumes certain differentiability conditions, it becomes impractical to apply to non-smooth optimization problems. It can often be infeasible to exhaust all potential solutions, especially given that discontinuities may produce multiple feasible regions. Non-smooth analysis techniques were designed to handle concerns surrounding differentiation and allow for applications of more classical optimizations. [Griewank1994ADD]

# Statement of need

`NonSmoothFwdAD` is a Julia package implementing two recent methods for evaluating generalized derivatives of continous, non-smooth functions while retaining the beneficial properties of smooth function derivatives. The first being the vector forward mode of automatic differentiation for functions of finite compositions of simple “scientific calculator” operations. The second is the compass difference rule for bivariate scalar valued functions. 

clearly illustrates the research purpose of the software and places it in the context of related work

# Acknowledgements 

# References
