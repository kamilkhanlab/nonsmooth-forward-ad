push!(LOAD_PATH,"C:/Users/maha-/Documents/GitHub/nonsmooth-forward-ad")

using Documenter

include("../src/NonsmoothFwdAD.jl")

using .NonsmoothFwdAD
using .GeneralizedDiff
using .ConvexOptimization

makedocs(sitename="NonSmoothFwdAD",
        format = Documenter.HTML(prettyurls = false),
        pages = Any["Introduction" => "index.md",
                    "GeneralizedDiff" => Any["generalizedDiff/methodOverview.md"
                                             "generalizedDiff/implementationOverview.md"
                                             "generalizedDiff/functions.md"],
                    "ConvexOptimization" => Any["convexOptimization/implementationOverview.md"
                                                "convexOptimization/functions.md"]
                    ]
)