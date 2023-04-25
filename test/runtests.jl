
include("../src/NonsmoothFwdAD.jl")

using .NonsmoothFwdAD
using .GeneralizedDiff
using .ConvexOptimization
using Test
using LinearAlgebra

@info "Testing GeneralizedDiff"

@testset "Example 1: Replicating Example 6.1 from [1]:" begin
    f1(x) = abs(x[1]) - abs(x[1])
    xVector = [0.0]
    xDotMatrix = [1.0 0.0; 0.0 1.0]
    xDotVector = [1.0]
    @test isapprox([eval_ld_derivative(f1, xVector, xDotMatrix)...],
        [[0.0], [0.0 0.0]])
    @test isapprox([eval_dir_derivative(f1, xVector, xDotVector)...],
        [[0.0], [0.0]])
    @test isapprox([eval_gen_derivative(f1, xVector)...],
        [[0.0], [0.0]])
end

@testset "Example 2: Replicating Example 6.2 from [1]:" begin
    f2(x) = max(min(x[1], -x[2]), x[2] - x[1])
    xVector = [0.0, 0.0]
    @test isapprox([eval_gen_gradient(f2, xVector)...],
        [0.0, [0.0, -1.0]])
    @test isapprox([eval_compass_difference(f2, xVector)...],
        [0.0, [-0.5, 0.5]])
end

@testset "Example 3: Replicating Example 6.3 from [1]:" begin
    f3(x) = (1.0 + abs(x[1] - x[2]))*(x[1] - x[2])
    xVector = [0.0, 0.0]
    xDotVector = [1.0, -1.0]

    @test isapprox([eval_dir_derivative(f3, xVector, xDotVector)...],
        [[0.0], [2.0]])
    @test isapprox([eval_gen_derivative(f3, xVector)...],
        [[0.0], [1.0 -1.0]])
    @test isapprox([eval_gen_gradient(f3, xVector)...],
        [0.0, [1.0, -1.0]])
end

@info "Testing ConvexOptimization"

@testset "Example 4: Method 1: Semismooth Newton method for solving f(x) = 0:" begin
    f4(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 + x[3] + 3.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 6.0,
    2.0*x[1]^2 + x[1] + x[2]^2 + 10.0*x[3] + 2.0*x[4] - 2.0,
    3.0*x[1]^2 + x[1]*x[2] + 2.0*x[2]^2 + 2.0*x[3] + 9.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 9.0,
    x[1]^2 + 3.0*x[2]^2 + 2.0*x[3] + 3.0*x[4] - 3.0]
    g4(x) = hypot.(x, fStarA(x)) .- (x .+ fStarA(x))
    g0 = [1.5, -1.0, 3.5, 0.25]

    xStar, normStar, _ = [semiSmoothNewton(gStar, g0)...]
    @test isapprox([xStar, normStar], [[1.0, 0.0, 3.0, 0.0], 0.0], atol = 1e-1)
    @test isapprox(g4(xStar), [0.0, 0.0, 0.0, 0.0], atol = 1e-1)
end

@testset "Example 5: Method 2: LP-Newton Method to solve F(z) = 0:" begin
    f5(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 - 2.0,
    2.0*x[1]^2 + x[1] + x[2]^2 - 1.45] 
    A = [1.2  -1.0; 
            -1.0  0.1;
            -1.0  -2.0]
    B = [0.5; -0.2; -1.0]
    sStar, _, gammaStar = LPNewton(f2, [0.3, 0.1], lb = [-2.0, -2.0], ub = [0.8, 0.8], A = A, b = B, solverTolerance = 1E-6)
    @test isapprox(f2(sStar), [0.0, 0.0], atol = 1e-3)
end

@testset "Example 6: Method 3: Level Method for finding minima:" begin
    f6(x) = max([2.0, -1.0, 2.0]'*[x[1], x[2], x[3]] + 2.0,                    
    [4.0, 5.0, -2.0]'*[x[2], x[2], x[2]] - 1.0,
    [2.0, 2.0, -3.0]'*[x[3], x[2], x[1]] + 3.5)

    xStar, _, fHxkStar = levelMethod(f5, [1.0, 2.0, -2.0], lb = [-2.0, -2.5, -4.0], ub = [1.0, 2.0, 3.0])      
    @test isapprox([xStar, fHxkStar], [[-1.2, -2.5, -4.0], -5.9], atol = 1e-3)
end
