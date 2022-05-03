#= 
test.jl
=======
Uses .NonsmoothFwdAD to evaluate generalized derivative elements, replicating 
the results of calculations from the following article:

[1]: KA Khan and PI Barton (2013), https://doi.org/10.1145/2491491.2491493
=#

include("../src/NonsmoothFwdAD.jl")

using .NonsmoothFwdAD


println("Replicating Example 6.1 from [1]:")

_, f1Grad = eval_gen_gradient(x -> abs(x[1]) - abs(x[1]), [0.0])
println("  Gen. gradient element ",
        "of (x -> abs(x) - abs(x)) at 0.0:")
println("    ", f1Grad)

_, g1Grad = eval_gen_gradient(x -> max(x[1], 0.0) - min(x[1], 0.0), [0.0])
println("  Gen. gradient element ",
        "of (x -> max(x, 0.0) - min(x, 0.0)) at 0.0:")
println("    ", g1Grad)


println("\nReplicating Example 6.2 from [1]:")

f2(x) = max(min(x[1], -x[2]), x[2] - x[1])

_, f2Grad = eval_gen_gradient(f2, [0.0, 0.0])
println("  Gen. gradient element ",
        "of (x -> max(min(x[1], -x[2]), x[2] - x[1]) ",
        "at [0.0, 0.0]:")
println("    ", f2Grad)

_, f2Compass = eval_compass_difference(f2, [0.0, 0.0])
println("  Compass difference ",
        "of this function ",
        "at [0.0, 0.0]:")
println("    ", f2Compass)


println("\nReplicating Example 6.3 from [1]:")

_, f3Grad = eval_gen_gradient([0.0, 0.0]) do x
    return (1.0 + abs(x[1] - x[2]))*(x[1] - x[2])
end
println("  Gen. gradient element ",
        "of (x -> (1.0 + abs(x[1] - x[2]))*(x[1] - x[2])) at [0.0, 0.0]:")
println("    ", f3Grad)


println("\nReplicating Example 6.5 from [1]:")

tInVal = [160.0, 170.0, 60.0, 116.0]
tOutVal = [93.0, 126.0, 160.0, 260.0]
t0Val = Float64[]
for (tI, tO) in zip(tInVal, tOutVal)
    append!(t0Val, [tI, tO])
end

qVal, qGrad = eval_gen_derivative(t0Val) do t
    # stream parameters
    fcP = [8.79, 10.55, 7.62, 6.08]
    deltaT = 10.0
    nStreams = length(fcP)

    # recover inlet/outlet temperatures from input
    tIn = [t[2*i - 1] for i in 1:nStreams]
    tOut = [t[2*i] for i in 1:nStreams]

    # subtract min. temperature approach from inlet/outlet temperatures of hot streams
    for j in 1:nStreams
        if tOut[j] < tIn[j]
            tIn[j] -= deltaT
            tOut[j] -= deltaT
        end
    end

    # compute minimum heating
    qH = 0.0
    for i in 1:nStreams
        zP = 0.0
        for j in 1:nStreams
            zP += fcP[j] * (max(0.0, tOut[j] - tIn[i]) - max(0.0, tIn[j] - tIn[i]))
        end
        qH = max(qH, zP)
    end

    # compute minimum cooling
    omega = 0.0
    for j in 1:nStreams
        omega += fcP[j] * (tIn[j] - tOut[j])
    end
    qC = qH + omega

    return [qH, qC]
end

println("  With Q^H and Q^C defined as in this example,")
println("  and with stream parameters and T defined in Table V in [1],")
println("  gen. derivative element of [Q^H, Q^C] at T:")
for row in eachrow(qGrad)
    println("    ", row)
end
