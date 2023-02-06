
# RESTART TERMINAL FOR EACH TEST:

include("../src/NonsmoothFwdAD.jl")

using .NonsmoothFwdAD

using JuMP, Ipopt, LinearAlgebra

println("Initializing...")
_, test1 = eval_gen_gradient(x -> abs(x[1]) - abs(x[1]), [0.0])
_, test2 = eval_gen_derivative(x -> abs(x[1]) - abs(x[1]), [0.0])
println("Initializion complete")

##################################################################################

function method1(
    f::Function,
    x0::Vector{Float64};
    epsilon::Float64 = 0.05,    
    delta::Float64 = 0.00005  
)

# set up initial condition
x = x0              # initial guess
n = length(x0)      # vector length
k = 0               # number of iterations

# define static stop condition
stoppingCondition = epsilon + delta*norm(f(x0))

while norm(f(x)) >= stoppingCondition
    # calculate hessian of 'f' at 'x' to update value of 'x': 
    if n == 1
        y, H = eval_gen_gradient(f, x)
        x = x .- H\[y]  
    else
        y, H = eval_gen_derivative(f, x)
        x = x - H\y
    end #if

    # max iteration condition 
    k = k + 1
    if k > 1000
        break
    end #if
end #while

return x, norm(f(x)), k

end #function

print("\nMethod 1: Semismooth Newton method for solving f(x) = 0:\n")
# Example 1:
f3(x) = [2.0*(x[1]^2) + 5.0*x[1] - 10.0 + 2.0*abs(x[1] - 2.0)]
x3, normFx3, k3 = method1(f3, [0.0])

print("\nFor f3(x), x = ", x3, " and ||f(x)|| = ", normFx3)
print("\nIn ", k3, " iterations\n")

# Example 2:
fStar(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 + x[3] + 3.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 6.0,
    2.0*x[1]^2 + x[1] + x[2]^2 + 10.0*x[3] + 2.0*x[4] - 2.0,
    3.0*x[1]^2 + x[1]*x[2] + 2.0*x[2]^2 + 2.0*x[3] + 9.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 9.0,
    x[1]^2 + 3.0*x[2]^2 + 2.0*x[3] + 3.0*x[4] - 3.0]
gStar(x) = hypot.(x, fStar(x)) .- (x .+ fStar(x))

xStar, normFxStar, kStar = method1(gStar, [1.5, -1.0, 3.5, 0.25])

print("\n\nFor fStar(x), x = ", xStar, " and ||f(x)|| = ", normFxStar)
print("\nIn ", kStar, " iterations\n\n")

##################################################################################

function method3(
    f::Function,                    # continous convex function `f` of multiple inputs and single output 
    xi::Vector{Float64},            # initial guess 
    lb::Vector{Float64},            # coordinates for lower bound of box domain on which `f` is defined
    ub::Vector{Float64};            # coordinates for upper bound of box domain on which `f` is defined
    epsilon::Float64 = 1e-4,        # overall method tolerance 
    TOLERANCE::Float64 = 1e-8,      # solver tolerance
    alpha::Float64 = 1/(2+sqrt(2))  # level coefficient in bounds (0,1)
)

# initialize system:
k = 1                       # initial iteration value
n = length(xi)              # length of initial guess vector 
fHxk = 0                    # variable initialized f^(xk)
stopCondition = 2*epsilon   # initial stopping condition

# set model to solver type:
model1 = Model(optimizer_with_attributes(  
    Ipopt.Optimizer,
    "tol" => TOLERANCE,       
    "max_iter" => 1000
    )
)     
set_silent(model1)

@variable(model1, lb[i] <= xn1[i = 1:n] <= ub[i])  
@variable(model1, t)        # stand in variable for f^(xk) to avoid error
@objective(model1, Min, t)

model2 = Model(optimizer_with_attributes(  
    Ipopt.Optimizer,
    "tol" => TOLERANCE,
    "max_iter" => 1000
    )
)
set_silent(model2)

@variable(model2, lb[i] <= xn12[i = 1:n] <= ub[i])  
@variable(model2, xk)
@objective(model2, Min, sum((xn12 .- xi).^2))     


# loop through to add a new constraint 
while any(stopCondition .>= epsilon) 

    # calculate f(xi), f'(xi):
    fxi, gxi = eval_gen_gradient(f, xi)     # req (1) for stopCondition

    # calculate fk^(xi) __________________________:
    @constraint(model1, fxi + gxi'*(xn1 .- xi) .<= t)         
    optimize!(model1)
    
    fHxk = value(t)                         # req (2) for stopCondition

    # calculate x(k+1) as euclidean projection:
    @constraint(model2, fxi + gxi'*(xn12 .- xi) .<= (1-alpha).*fHxk .+ alpha*fxi)
    optimize!(model2)

    # set x(k+1) as next loop's xi
    xi = value.(model2[:xn12])          

    # update stopCondition for next loop
    stopCondition = fxi .- fHxk 
    
    # max iteration condition 
    k = k + 1
    if k == 50
        break
    end #if 

end #while

print(k, " iterations \n")
return fHxk, xi

end #function 

# Example 1:
print("\nMethod 3: Level Method for solving f(x) = 0:\n")
f1(x) = x[1]^2 + x[2]^2

s1,x1 = method3(f1, [1.0, 2.0], [-0.5, -0.4], [3.0, 4.0])
print("\nFor f1(x), xk = ", k1, " and fHxk = ", s1, "\n")

# Example 2:
f2(x) = max([2.0, -1.0, 2.0]'*[x[1], x[2], x[3]] + 2.0,
[4.0, 5.0, -2.0]'*[x[2], x[2], x[2]] - 1.0,
[2.0, 2.0, -3.0]'*[x[3], x[2], x[1]] + 3.5)

s2,x2 = method3(f2, [1.0, 2.0, -2.0], [-2.0, -2.5, -4.0], [1.0, 2.0, 3.0])
print("\nFor f2(x), xk = ", x2, " and fHxk = ", s2, "\n")

# Example 3 | Save for Method 2:

# fStar(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 + x[3] + 3.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 6.0,
#     2.0*x[1]^2 + x[1] + x[2]^2 + 10.0*x[3] + 2.0*x[4] - 2.0,
#     3.0*x[1]^2 + x[1]*x[2] + 2.0*x[2]^2 + 2.0*x[3] + 9.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 9.0,
#     x[1]^2 + 3.0*x[2]^2 + 2.0*x[3] + 3.0*x[4] - 3.0]
# gStar(x) = hypot.(x, fStar(x)) .- (x .+ fStar(x))

# y, s, k = method3(gStar, [1.0, 1.0, 1.0, 1.0])
# print(y, "\n")
# print(s, "\n")
# print(k, "\n")
