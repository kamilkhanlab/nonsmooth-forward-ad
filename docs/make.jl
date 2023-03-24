push!(LOAD_PATH,"C:/Users/maha-/Documents/GitHub/nonsmooth-forward-ad")

using Documenter

include("../src/NonsmoothFwdAD.jl")

using .NonsmoothFwdAD
using .GeneralizedDiff
using .ConvexOptimization

makedocs(sitename="NonSmoothFwdAD",
        format = Documenter.HTML(prettyurls = false),
        pages = Any["Introduction" => "index.md",
                    "Method Overview" => "methodOverview.md",
                    "ConvexOptimization Implementation Overview" => "convexOptimization.md",
                    "Exported Functions" => "functions.md"]
)