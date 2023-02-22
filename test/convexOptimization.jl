
# RESTART TERMINAL FOR EACH TEST:

include("../src/NonsmoothFwdAD.jl")

using .NonsmoothFwdAD

using JuMP, Ipopt, LinearAlgebra

println("Initializing...")
_, test1 = eval_gen_gradient(x -> abs(x[1]) - abs(x[1]), [0.0])
_, test2 = eval_gen_derivative(x -> abs(x[1]) - abs(x[1]), [0.0])
println("Initializion complete")

##################################################################################

# function method1(
#     f::Function,
#     x0::Vector{Float64};
#     epsilon::Float64 = 0.05,    
#     delta::Float64 = 0.00005  
# )

# # set up initial condition
# x = x0              # initial guess
# n = length(x0)      # vector length
# k = 0               # number of iterations

# # define static stop condition
# stoppingCondition = epsilon + delta*norm(f(x0))

# while norm(f(x)) >= stoppingCondition
#     # calculate hessian of 'f' at 'x' to update value of 'x': 
#     if n == 1
#         y, H = eval_gen_gradient(f, x)
#         x = x .- H\[y]  
#     else
#         y, H = eval_gen_derivative(f, x)
#         x = x - H\y
#     end #if

#     # max iteration condition 
#     k = k + 1
#     if k > 1000
#         break
#     end #if
# end #while

# return x, norm(f(x)), k

# end #function

# print("\nMethod 1: Semismooth Newton method for solving f(x) = 0:\n")
# # Example 1:
# f3(x) = [2.0*(x[1]^2) + 5.0*x[1] - 10.0 + 2.0*abs(x[1] - 2.0)]
# x3, normFx3, k3 = method1(f3, [0.0])

# print("\nFor f3(x), x = ", x3, " and ||f(x)|| = ", normFx3)
# print("\nIn ", k3, " iterations\n")

# # Example 2:
# fStar(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 + x[3] + 3.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 6.0,
#     2.0*x[1]^2 + x[1] + x[2]^2 + 10.0*x[3] + 2.0*x[4] - 2.0,
#     3.0*x[1]^2 + x[1]*x[2] + 2.0*x[2]^2 + 2.0*x[3] + 9.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 9.0,
#     x[1]^2 + 3.0*x[2]^2 + 2.0*x[3] + 3.0*x[4] - 3.0]
# gStar(x) = hypot.(x, fStar(x)) .- (x .+ fStar(x))

# xStar, normFxStar, kStar = method1(gStar, [1.5, -1.0, 3.5, 0.25])

# print("\n\nFor fStar(x), x = ", xStar, " and ||f(x)|| = ", normFxStar)
# print("\nIn ", kStar, " iterations\n\n")

##################################################################################

# function method3(
#     f::Function,                    # continous convex function `f` of multiple inputs and single output 
#     xi::Vector{Float64},            # initial guess 
#     lb::Vector{Float64},            # coordinates for lower bound of box domain on which `f` is defined
#     ub::Vector{Float64};            # coordinates for upper bound of box domain on which `f` is defined
#     epsilon::Float64 = 1e-4,        # overall method tolerance 
#     TOLERANCE::Float64 = 1e-8,      # solver tolerance
#     alpha::Float64 = 1/(2+sqrt(2)),  # level coefficient in bounds (0,1)
#     A::Matrix{Float64} = ones(1, length(xi)),
#     b::Vector{Float64} = [1.0],
#     Aeq::Matrix{Float64} = ones(1, length(xi)),
#     beq::Vector{Float64} = [1.0]
# )

# # initialize system:
# k = 1                       # initial iteration value
# n = length(xi)              # length of initial guess vector 
# fHxk = 0                    # variable initialized f^(xk)
# stopCondition = 2*epsilon   # initial stopping condition
# xk = 0

# # set model to solver type:
# model1 = Model(optimizer_with_attributes(  
#     Ipopt.Optimizer,
#     "tol" => TOLERANCE,       
#     "max_iter" => 1000
#     )
# )     
# set_silent(model1)

# @variable(model1, lb[i] <= xn1[i = 1:n] <= ub[i])   #constraint (1)
# @constraint(model1, A*xn1 .<= b)                    #constraint (2)
# # @constraint(model1, Aeq*xn1 .== beq)                #constraint (3)

# @variable(model1, t)        # stand in variable for f^(xk) to avoid error
# @objective(model1, Min, t)

# model2 = Model(optimizer_with_attributes(  
#     Ipopt.Optimizer,
#     "tol" => TOLERANCE,
#     "max_iter" => 1000
#     )
# )
# set_silent(model2)

# @variable(model2, lb[i] <= xn12[i = 1:n] <= ub[i])  #constraint (1)
# @constraint(model2, con2, A*xn12 .<= b)             #constraint (2)
# # @constraint(model2, con3, Aeq*xn12 .== beq)         #constraint (3)

# # Norm Squared 
# # @objective(model2, Min, sum((xn12 .- xi).^2))         

# # Might handle infinity norm squared:
# @variable(model2, f_t[1:n])
# @variable(model2, f_t_max)

# @constraint(model2, [t=1:n], f_t_max >= f_t[t])
# @objective(model2, Min, f_t_max^2)

# # loop through to add a new constraint 
# while any(stopCondition .>= epsilon)    
    
#     # calculate f(xi), f'(xi):
#     fxi, gxi = eval_gen_gradient(f, xi)     

#     # MODEL 1 ###################################################################
#     # calculate fk^(xi) __________________________:
#     @constraint(model1, fxi + gxi'*(xn1 - xi) <= t)         
#     optimize!(model1)
    
#     # pull optimal values for iteration:
#     xk = value.(model1[:xn1])
#     fHxk = value(model1[:t])                         # req (2) for stopCondition
#     fStarxk = f(xk)                # req (1) for stopCondition
#     print(xk)

#     # MODEL 2 ###################################################################
#     # calculate x(k+1) as euclidean projection:
#     @constraint(model2, f_t .== xn12 - xi)  
#     @constraint(model2, fxi + gxi'*(xn12 - xi) <= (1-alpha)*fHxk + alpha*fStarxk)       #issue (1): not alpha*fxi
#     optimize!(model2)

#     # set x(k+1) as next loop's xi
#     xi = value.(model2[:xn12])   
#     fSmthg = objective_value(model2)

#     # update stopCondition for next loop
#     stopCondition = abs(fStarxk - fHxk)

#     print((xi, fSmthg, fStarxk, fHxk, stopCondition), "\n") 
    
#     # max iteration condition 
#     k = k + 1
#     if k == 15
#         break
#     end #if 

# end #while

# print(model1)

# print(k, " iterations")
# return fHxk, xk
# end #function 

# # Example 1:
# print("\nMethod 3: Level Method for finding minima:\n\n")                       
# f1(x) = x[1]^2 + x[2]^2

# s1,x1 = method3(f1, [2.0, 1.0], [-1.0, -1.0], [3.0, 4.0])                         #xk = [0.0, 0.0] for f(xk) = 0.0
# print("\nFor f1(x), xk = ", x1, " and fHxk = ", s1, "\n\n")    

# # Example 2:
# f2(x) = max([2.0, -1.0, 2.0]'*[x[1], x[2], x[3]] + 2.0,                           #xk = [-1.2000000100457517, -2.5000000124373436, -4.000000038746961] and fHxk = -5.900000100970536     
# [4.0, 5.0, -2.0]'*[x[2], x[2], x[2]] - 1.0,
# [2.0, 2.0, -3.0]'*[x[3], x[2], x[1]] + 3.5)

# s2,x2 = method3(f2, [1.0, 2.0, -2.0], [-2.0, -2.5, -4.0], [1.0, 2.0, 3.0])      
# print("\nFor f2(x), xk = ", x2, " and fHxk = ", s2, "\n\n") 

# f3(x) = 0.5*x[1]^2 + x[2]^2 - x[1]*x[2] - 2.0*x[1] - 6.0*x[2]                     #xk = [0.6667, 1.333] for f(xk) = -8.221

# A3 = [1.0 1.0; -1.0 2.0; 2.0 1.0]
# b3 = [2.0; 2.0; 3.0]
# lb3 = [0.0, 0.0]
# ub3 = [5.0, 5.0]

# s3,x3 = method3(f3, [1.0, 2.0], lb3, ub3, A = A3, b = b3)
# print("\nFor f3(x), xk = ", x3, " and fHxk = ", s3, "\n\n")

function method2(
    f::Function,                    # continous convex function `f` of multiple inputs and single output 
    z0::Vector{Float64},            # initial guess 
    lb::Vector{Float64},            # coordinates for lower bound of box domain on which `f` is defined
    ub::Vector{Float64};            # coordinates for upper bound of box domain on which `f` is defined
    epsilon::Float64 = 1e-2,        # overall method tolerance 
    TOLERANCE::Float64 = 1e-6,      # solver tolerance
    A::Matrix{Float64} = ones(1, length(z0)),
    b::Vector{Float64} = [1.0],
    Aeq::Matrix{Float64} = ones(1, length(z0)),
    beq::Vector{Float64} = [1.0]
)

# initialize system:
k = 1                       # initial iteration value
n = length(z0)              # length of initial guess vector 
s = z0                      # initial guess
GAMMA = 0

# set model to solver type:
model3 = Model(optimizer_with_attributes(  
    Ipopt.Optimizer,
    "tol" => TOLERANCE,       
    "max_iter" => 1000
    )
)     
set_silent(model3)

# Main model definition:
@variable(model3, lb[i] <= zk[i=1:n] <= ub[i])      # variable (1)          
@variable(model3, gamma >= 0)                       # variable (2)

@constraint(model3, c1, A*zk .<= b)                 # constraint (1) 

@objective(model3, Min, gamma)

while any(abs.(f(s)) .>= epsilon)    #otherwise, will not account for most recent "s" value

    # calculate f(xi), f'(xi):
    fsi, gsi = eval_gen_derivative(f, s) 

    # intermediate calculations:
    Nsi = norm(fsi, Inf) 

    # calculate fk^(xi) __________________________:
    @constraint(model3, [t=1:n], (fsi + gsi*(zk - s))[t] <= gamma*(Nsi^2))     # constraint (2) x k           #CHECK: zk - s or s - zk?
    @constraint(model3, [t=1:n], (zk - s)[t] <= gamma*Nsi)                     # constraint (3) x k           #CHECK: gsi or gsi'?

    optimize!(model3)

    # set s = zk for the next iteration
    s = value.(model3[:zk])   
    GAMMA = value.(model3[:gamma])

    print((k, s, f(s), GAMMA), "\n")

    # update iteration counter
    k = k + 1
    if k == 10
        break
    end #if

end #while

# return (s, f(s), GAMMA)
end # function 

# Example 3 | Save for Method 2:

print("Method 2: \n")

fStar(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 + x[3] + 3.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 6.0,
    2.0*x[1]^2 + x[1] + x[2]^2 + 10.0*x[3] + 2.0*x[4] - 2.0,
    3.0*x[1]^2 + x[1]*x[2] + 2.0*x[2]^2 + 2.0*x[3] + 9.0*x[4] + abs(x[3] - 2.0*x[4] - 3.0) - 9.0,
    x[1]^2 + 3.0*x[2]^2 + 2.0*x[3] + 3.0*x[4] - 3.0]
gStar(x) = hypot.(x, fStar(x)) .- (x .+ fStar(x))   #zk = [0.0, 0.0, 0.0, 1.0]            

lbStar = [0.0, 0.0, 0.0, 0.0]
ubStar = [5.0, 5.0, 5.0, 5.0]

# method2(gStar, [-5.0, 5.0, -5.0, 5.0], lbStar, ubStar)
# method2(gStar, [1.0, 1.0, 2.0, 0.0], lbStar, ubStar)

# fStarA(x) = [3.0*x[1]^2 + 2.0*x[1]*x[2] + 2.0*x[2]^2 - 2.0,
# 2.0*x[1]^2 + x[1] + x[2]^2 - 1.45]          #zk = [0.55622, 0.505]

# starA = [1.2  -1; 
#          -1.0  0.1;
#          -1.0  -2.0]
# starB = [0.5; -0.2; -1.0]

# method2(fStarA, [0.3, 0.1], [-2.0, -2.0], [0.8, 0.8], A=starA, b=starB, TOLERANCE=1E-6)

fStarB(x) = [x[1]*x[2] - x[3],
            x[1]^2 + x[2] - 1.0 - x[4],
            min(x[1], x[3]),
            min(x[2], x[4])]
method2(fStarB, [2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]) # zk = [1.0, 0.0, 0.0, 0.0], gamma = 0.0625
# method2(fStarB, [2.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]) # zk = [1.0, 0.0, 0.0, 0.0], gamma = 0.055
# method2(fStarB, [1.0, 2.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0], [3.0, 3.0, 3.0, 3.0]) # zk = [0.0, 1.0, 0.0, 0.0], gamma = 10.26       ANS is within 1e-1
