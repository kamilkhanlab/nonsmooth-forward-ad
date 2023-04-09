include("../src/NonsmoothFwdAD.jl")

using .NonsmoothFwdAD
using .GeneralizedDiff
using .ConvexOptimization

print("\nMethod 1: Semismooth Newton method for solving f(x) = 0:\n")
# Example 1:
f1(x) = [2.0*(x[1]^2) + 5.0*x[1] - 10.0 + 2.0*abs(x[1] - 2.0)]
x1, normFx1, k1 = semiSmoothNewton(f1, [0.0])

print("\nFor f1(x), x = ", x1, " and ||f1(x)|| = ", normFx1)
print("\nIn ", k1, " iterations\n")

# Example 2:
fStarA(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 + x[3] + 3.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 6.0,
    2.0*x[1]^2 + x[1] + x[2]^2 + 10.0*x[3] + 2.0*x[4] - 2.0,
    3.0*x[1]^2 + x[1]*x[2] + 2.0*x[2]^2 + 2.0*x[3] + 9.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 9.0,
    x[1]^2 + 3.0*x[2]^2 + 2.0*x[3] + 3.0*x[4] - 3.0]
gStarA(x) = hypot.(x, fStarA(x)) .- (x .+ fStarA(x))

xStarA, normFxStarA, kStarA = semiSmoothNewton(gStarA, [1.5, -1.0, 3.5, 0.25])

print("\n\nFor fStar(x), x = ", xStarA, " and ||fStar(x)|| = ", normFxStarA)
print("\nIn ", kStarA, " iterations\n\n")

print("Method 2: LP-Newton Method to solve F(z) = 0: \n")
# Example 1:
f2(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 - 2.0,
2.0*x[1]^2 + x[1] + x[2]^2 - 1.45] 

A2 = [1.2  -1.0; 
         -1.0  0.1;
         -1.0  -2.0]
B2 = [0.5; -0.2; -1.0]

s2, _, gamma2 = LPNewton(f2, [0.3, 0.1], lb = [-2.0, -2.0], ub = [0.8, 0.8], A = A2, b = B2, solverTolerance = 1E-6)

print("\n\nFor f2(x), z = ", s2, " and gamma = ", gamma2)

# Example 2:
f3(x) = [x[1]*x[2] - x[3],
            x[1]^2 + x[2] - 1.0 - x[4],
            min(x[1], x[3]),
            min(x[2], x[4])]
s3A, _, gamma3A = LPNewton(f3, [2.0, 3.0, 1.0, 0.0]) 
s3B, _, gamma3B = LPNewton(f3, [2.0, 3.0, 1.0, 0.0], lb = [0.0, 0.0, 0.0, 0.0], ub = [3.0, 3.0, 3.0, 3.0]) 
s3C, _, gamma3C = LPNewton(f3, [1.0, 2.0, 1.0, 3.0], lb = [0.0, 0.0, 0.0, 0.0], ub = [3.0, 3.0, 3.0, 3.0]) 

print("\n\nFor f3(x) with no bounds given x0 = [2.0, 3.0, 1.0, 0.0], z = ", s3A, " and gamma = ", gamma3A)
print("\n\nFor f3(x) with a bounded x set given x0 = [2.0, 3.0, 1.0, 0.0], z = ", s3B, " and gamma = ", gamma3B)
print("\n\nFor f3(x) with a bounded x set given x0 = [1.0, 2.0, 1.0, 0.0], z = ", s3C, " and gamma = ", gamma3C)

# Example 3:
uOffset = 4
vOffset = 9
fStarB(x) = [x[4] + x[1+uOffset] - x[2+uOffset] - x[5+uOffset],
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

z0StarB = [1.0, 4.0, -2.0, 1.0,
    3.0, 3.0, 1.0, 4.0, 1.0,
    0.0, 1.0, 3.0, 1.0, 3.0]

sStarB1, _, gammaStarB1 = LPNewton(fStarB, z0StarB)
sStarB2, _, gammaStarB2 = LPNewton(fStarB, zeros(14))

print("\n\nFor fStarB(x) with no bounds at z0 = ", z0StarB, "z = ", sStarB1, " and gamma = ", gammaStarB1)
print("\n\nFor fStarB(x) with no bounds at z0 = ", zeros(14), "z = ", sStarB2, " and gamma = ", gammaStarB2)

print("\nMethod 3: Level Method for finding minima:\n\n")           
# Example 1: 
f4(x) = (x[1] - 1.0)^2 + (x[2] - 3.0)^2 

x4, _, fHxk4 = levelMethod(f4, [1.0, 0.1], lb = [-1.0, -1.0], ub = [3.0, 4.0], maxIters = 1000, alpha=0.001)
x4, _, fHxk4 = levelMethod(f4, [0.1, 0.1], lb = [-1.0, -1.0], ub = [3.0, 4.0], epsilon = 1e-8)
print("\nFor f4(x), xk = ", x4, " and fHxk = ", fHxk4, "\n\n")    

# Example 2:
f5(x) = max([2.0, -1.0, 2.0]'*[x[1], x[2], x[3]] + 2.0,                    
[4.0, 5.0, -2.0]'*[x[2], x[2], x[2]] - 1.0,
[2.0, 2.0, -3.0]'*[x[3], x[2], x[1]] + 3.5)

x5, _, fHxk5 = levelMethod(f5, [1.0, 2.0, -2.0], lb = [-2.0, -2.5, -4.0], ub = [1.0, 2.0, 3.0])      
print("\nFor f2(x), xk = ", x5, " and fHxk = ", fHxk5, "\n\n") 

# Example 3:
f6(x) = 0.5*x[1]^2 + x[2]^2 - x[1]*x[2] - 2.0*x[1] - 6.0*x[2]  

A6 = [1.0 1.0; -1.0 2.0; 2.0 1.0]
b6 = [2.0; 2.0; 3.0]
lb6 = [0.0, 0.0]
ub6 = [5.0, 5.0]

x6, _, fHxk6 = levelMethod(f6, [1.0, 2.0], lb = lb6, ub = ub6, A = A6, b = b6)
print("\nFor f3(x), xk = ", x6, " and fHxk = ", fHxk6, "\n\n")

