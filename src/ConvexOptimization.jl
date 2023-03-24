module ConvexOptimization 

include("../src/GeneralizedDiff.jl")

using .GeneralizedDiff
using JuMP, Ipopt, LinearAlgebra, NLopt

export semiSmoothNewton, 
    LPNewton, 
    levelMethod

"""
    semiSmoothNewton(f::Function, x0::Vector{Float64}, kwargs…)

Compute vector `x` where `f(x) = 0` using the semi-smooth Newton method where there are multiple inputs to `f` but only one output. 

# Arguments

- `f::Function`: must be continous and of finite compositions of elemental operations. Each operation must be of the form `f(x::N)::Float64` where N is either `Vector{Float64}` or `Float64` ; otherwise implementation cannot map `f`. Supported elemental operations include: `+, -, *, inv, /, ^, exp, log, sin, cos, abs, min, max, hypot`.
- `x0::Vector{Float64}`: initial domain vector guess 

# Keywords

- `epsilon::Float64`: tolerance for solver stopping condition where `epsilon + delta*norm(f(x0))`. Set to `0.05` by default. 
- `delta::Float64`: tolerance for solver stopping condition where `epsilon + delta*norm(f(x0))`. Set to `0.00005` by default. 
- `maxIter::Int64`: maximum number of solver iterations. Set to `1000` by default. 

"""
function semiSmoothNewton(
    f::Function,                # continous convex function `f`  
    x0::Vector{Float64};        # initial guess
    epsilon::Float64 = 0.05,    # solver tolerance type (1)
    delta::Float64 = 0.00005,   # solver tolerance type (2)
    maxIter::Int64 = 1000       # maximum iterations
)

    # Set up initial condition
    x = x0              # initial guess
    n = length(x0)      # vector length
    k = 0               # number of iterations

    # Define static stop condition:
    stoppingCondition = epsilon + delta*norm(f(x0))

    while norm(f(x)) >= stoppingCondition
        # Calculate hessian of 'f' at 'x' to update value of 'x': 
        if n == 1
            y, H = eval_gen_gradient(f, x)
            x = x .- H\[y]  
        else
            y, H = eval_gen_derivative(f, x)
            x = x - H\y
        end #if

        # Set breaking condition on maximum iterations: 
        k = k + 1
        if k > maxIter
            break
        end #if
    end #while

    return x, norm(f(x)), k

end #function

"""
	LPNewton(f::Function, z0::Vector{Float64}, kwargs…)

Compute vector `z` where `f(z) = 0` using the Linear Program Newton method where the number of inputs to `f` is equivalent to number of outputs. 

# Arguments

- `f::Function`: must be continous and of finite compositions of elemental operations. Each operation must be of the form `f(x::N)::Float64` where N is either `Vector{Float64}` or `Float64` ; otherwise implementation cannot map `f`. Supported elemental operations include: `+, -, *, inv, /, ^, exp, log, sin, cos, abs, min, max, hypot`.
- `x0::Vector{Float64}`: initial domain vector guess 

# Keywords

## Constant and tolerance keyword

- `epsilon::Float64`: tolerance for solver stopping condition where `f(z) >= epsilon`. Set to `1e-2` by default. 
- `TOLERANCE::Float64`: tolerance for JuMP solver. Set to `1e-6` by default.
- `maxIter::Int64`: maximum number of solver iterations. Set to `20` by default. 

## Constraint keywords

- `lb::Vector{Float64}`: lower bound on domain vector `x`. Set to `-Inf` vector of size `n` by default.
- `ub::Vector{Float64}`: upper bound on domain vector `x`. Set to `-Inf` vector of size `n` by default.
- `A::Matrix{Float64}`: matrix `A` in inequality constraint set `A*z <= b`. Set to zeros vector of size `n` by default.
- `b::Vector{Float64}`: matrix `b` in inequality constraint set `A*z <= b`. Set to zeros matrix of size `1x1`.
- `Aeq::Matrix{Float64}`: matrix `A` in equality constraint set `A*z == b`. Set to zeros vector of size `n` by default. 
- `beq::Vector{Float64}`: matrix `b` in equality constraint set `A*z == b`. Set to zeros matrix of size `1x1`.

where `n` is the length of vector `z0`.

"""
function LPNewton(
    f::Function,                                 # continous convex function `f` of multiple inputs and single output 
    z0::Vector{Float64};                         # initial guess 
    lb::Vector{Float64} = ones(length(z0))*Inf,  # coordinates for lower bound of box domain on which `f` is defined
    ub::Vector{Float64} = ones(length(z0))*Inf,  # coordinates for upper bound of box domain on which `f` is defined
    epsilon::Float64 = 1e-2,                     # overall method tolerance 
    TOLERANCE::Float64 = 1e-6,                   # solver tolerance
    A::Matrix{Float64} = zeros(0, length(z0)),
    b::Vector{Float64} = zeros(0),
    Aeq::Matrix{Float64} = zeros(0, length(z0)),
    beq::Vector{Float64} = zeros(0),
    maxIters::Int64 = 20
)
    # Initialize system:
    k = 1               # initial iteration value
    n = length(z0)      # length of initial guess vector 
    s = z0              # initial "z" value
    GAMMA = 0           # initial "gamma" value
    fsi = 0             # initial f(s)
    gsi = 0             # initial f'(s)

    # Set up model to minimize gamma for "z": 
    model3 = Model(optimizer_with_attributes(  
        Ipopt.Optimizer,
        "tol" => TOLERANCE,       
        "max_iter" => 1000
        )
    )     
    set_silent(model3)

    @variable(model3, lb[i] <= zk[i=1:n] <= ub[i])   # constraint (1)
    @variable(model3, gamma >= 0)                    # constraint (2)
    @constraint(model3, c1, A*zk .<= b)              # constraint (3)      
    @constraint(model3, c2, Aeq*zk .== beq)          # constraint (4) 
    @objective(model3, Min, gamma)

    while any(abs.(f(s)) .>= epsilon) 
        # Store last iteration's "z" value:
        oldS = s

        # Calculate f(s), f'(s):
        fsi, gsi = eval_gen_derivative(f, s) 

        # Intermediate calculation of norm(f(s)):
        Nsi = norm(fsi, Inf) 

        # Calculate z and gamma from MODEL 3:
        @constraint(model3, [t=1:n], (fsi[t] + gsi[t,:]'*(zk - s)) <= gamma*(Nsi^2))     # constraint (5) x k           
        @constraint(model3, [t=1:n], (-fsi[t] - gsi[t,:]'*(zk - s)) <= gamma*(Nsi^2))    # constraint (6) x k           
        @constraint(model3, [t=1:n], (zk[t] - s[t]) <= gamma*Nsi)                        # constraint (7) x k           
        @constraint(model3, [t=1:n], (s[t] - zk[t]) <= gamma*Nsi)                        # constraint (8) x k           
        JuMP.optimize!(model3)

        s = value.(model3[:zk])   
        GAMMA = value.(model3[:gamma])

        print((k, s, f(s), GAMMA), "\n")

        # Set breaking condition on maximum iterations:
        k = k + 1
        if k == maxIters
            break
        end #if
    end #while

    return s, f(s), GAMMA
end # function 

"""
	levelMethod(f::Function, z0::Vector{Float64}, kwargs…)

Compute vector `x` to minimize `f(x)` using the Level method where there are multiple inputs to `f` but only one output. . 

# Arguments

- `f::Function`: must be continous and of finite compositions of elemental operations. Each operation must be of the form `f(x::N)::Float64` where N is either `Vector{Float64}` or `Float64` ; otherwise implementation cannot map `f`. Supported elemental operations include: `+, -, *, inv, /, ^, exp, log, sin, cos, abs, min, max, hypot`.
- `x0::Vector{Float64}`: initial domain vector guess 

# Keywords

## Constant and tolerance keywords

- `epsilon::Float64`: tolerance for solver stopping condition where `f(z) >= epsilon`. Set to `1e-2` by default. 
- `TOLERANCE::Float64`: tolerance for JuMP solver. Set to `1e-6` by default.
- `alpha::Float64`: level set constant for quadratic programming. Set to `1/(2+sqrt(2)` by default. 
- `maxIter::Int64`: maximum number of solver iterations. Set to `20` by default. 

## Constraint keywords

- `lb::Vector{Float64}`: lower bound on domain vector `x`. Set to `-Inf` vector of size `n` by default.
- `ub::Vector{Float64}`: upper bound on domain vector `x`. Set to `-Inf` vector of size `n` by default.
- `A::Matrix{Float64}`: matrix `A` in inequality constraint set `A*z <= b`. Set to zeros vector of size `n` by default.
- `b::Vector{Float64}`: matrix `b` in inequality constraint set `A*z <= b`. Set to zeros matrix of size `1x1`.
- `Aeq::Matrix{Float64}`: matrix `A` in equality constraint set `A*z == b`. Set to zeros vector of size `n` by default. 
- `beq::Vector{Float64}`: matrix `b` in equality constraint set `A*z == b`. Set to zeros matrix of size `1x1`.

where `n` is the length of vector `z0`.

"""
function levelMethod(
    f::Function,                                    # continous convex function `f` of multiple inputs and single output 
    xi::Vector{Float64};                            # initial guess 
    lb::Vector{Float64} = ones(length(xi))*-Inf,    # coordinates for lower bound of box domain on which `f` is defined
    ub::Vector{Float64} = ones(length(xi))*Inf,     # coordinates for upper bound of box domain on which `f` is defined
    epsilon::Float64 = 1e-4,                        # overall method tolerance 
    TOLERANCE::Float64 = 1e-8,                      # solver tolerance
    alpha::Float64 = 1/(2+sqrt(2)),                 # level coefficient in bounds (0,1)
    A::Matrix{Float64} = zeros(0, length(xi)), 
    b::Vector{Float64} = zeros(0),
    Aeq::Matrix{Float64} = zeros(0, length(xi)),
    beq::Vector{Float64} = zeros(0),
    maxIters = 50
)
    # Initialize system:
    k = 1                       # initial iteration value
    n = length(xi)              # length of initial guess vector 
    fHxk = 0                    # initial f^(xk) value
    stopCondition = 2*epsilon   # initial stopping condition value
    xk = zeros(n)               # initial xk for model 1
    xkOld = zeros(n)             # xkOld for breaking condition 

    # Set up model 1 - linear program to minimize f^(xi):
    model1 = Model(optimizer_with_attributes(  
        Ipopt.Optimizer,
        "tol" => TOLERANCE,       
        "max_iter" => 1000
        # "algorithm" => :LD_MMA
        )
    )     
    set_silent(model1)

    @variable(model1, lb[i] <= xn1[i = 1:n] <= ub[i])   # constraint (1)
    @constraint(model1, A*xn1 .<= b)                    # constraint (2)
    @constraint(model1, Aeq*xn1 .== beq)                # constraint (3)
    @variable(model1, t)  
    @objective(model1, Min, t)

    # Set up model 2 - quadratic program to minimize Euclidean projection:
    model2 = Model(optimizer_with_attributes(  
        NLopt.Optimizer,
        "tol" => TOLERANCE,
        "max_iter" => 1000,
        "algorithm" => :LD_MMA
        )
    )
    set_silent(model2)

    @variable(model2, lb[i] <= xn12[i = 1:n] <= ub[i])  # constraint (1)
    @constraint(model2, con2, A*xn12 .<= b)             # constraint (2)
    @constraint(model2, con3, Aeq*xn12 .== beq)         # constraint (3)
    @objective(model2, Min, sum((xn12 .- xi).^2)) 

    # Loop through to iteratively solve models and add new constraints: 
    while any(stopCondition >= epsilon)    
        # Store last iteration's xk value from model 1:
        xkOld = xk
        
        # Calculate f(xi), f'(xi):
        fxi, gxi = eval_gen_gradient(f, xi)     

        # Calculate fk^(xi) from MODEL 1:
        @constraint(model1, fxi + gxi'*(xn1 - xi) <= t)      
        JuMP.optimize!(model1)
        
        # pull optimal values for iteration:
        xk = value.(model1[:xn1])
        fHxk = value(model1[:t])       # req (1) for stopCondition
        fStarxk = f(xk)                # req (2) for stopCondition

        # Calculate x(i+1) from MODEL 2:
        @constraint(model2, fxi + gxi'*(xn12 - xi) <= (1-alpha)*fHxk + alpha*fStarxk)       #TODO: Potentially HERE
        JuMP.optimize!(model2)
        xi = value.(model2[:xn12])   

        # Update stopping condition:
        stopCondition = abs(fStarxk - fHxk)     #TODO: Potentially here

        print((xk, xi, abs(fStarxk - fHxk), abs(fxi - fHxk)), "\n")    #TODO: Delete
        
        # Set breaking condition on maximum iterations:
        k = k + 1
        if k == maxIters
            break
        end #if 
    end #while

    print(k, " iterations")
    return xi, f(xi), fHxk

end # function 

end # module 