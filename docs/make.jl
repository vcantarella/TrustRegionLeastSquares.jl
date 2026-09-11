using TrustRegionLeastSquares
using Documenter

DocMeta.setdocmeta!(TrustRegionLeastSquares, :DocTestSetup, :(using TrustRegionLeastSquares); recursive = true)

makedocs(;
    modules = [TrustRegionLeastSquares],
    authors = "vcantarella and contributors",
    sitename = "TrustRegionLeastSquares.jl",
    format = Documenter.HTML(;
        canonical = "https://vcantarella.github.io/TrustRegionLeastSquares",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "API Reference" => "api.md"],
)

deploydocs(; repo = "github.com/vcantarella/TrustRegionLeastSquares", devbranch = "main")
