using TrustRegionLeastSquares
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(
    TrustRegionLeastSquares,
    :DocTestSetup,
    :(using TrustRegionLeastSquares);
    recursive = true,
)

# Add titles of sections and overrides page titles
const titles = Dict(
    # "10-tutorials" => "Tutorials", # example folder title
    "10-theory" => "Theory",
    "15-benchmarks.md" => "Benchmarks",
    "20-bibliography.md" => "Bibliography",
    "91-developer.md" => "Developer docs",
    "95-reference.md" => "API reference",
)

function recursively_list_pages(folder; path_prefix = "")
    pages_list = Any[]
    for file in readdir(folder)
        if file == "index.md"
            # We add index.md separately to make sure it is the first in the list
            continue
        end
        # this is the relative path according to our prefix, not @__DIR__, i.e., relative to `src`
        relpath = joinpath(path_prefix, file)
        # full path of the file
        fullpath = joinpath(folder, relpath)

        if isdir(fullpath)
            # If this is a folder, enter the recursion case
            subsection = recursively_list_pages(fullpath; path_prefix = relpath)

            # Ignore empty folders
            if length(subsection) > 0
                title = if haskey(titles, relpath)
                    titles[relpath]
                else
                    @error "Bad usage: '$relpath' does not have a title set. Fix in 'docs/make.jl'"
                    relpath
                end
                push!(pages_list, title => subsection)
            end

            continue
        end

        if splitext(file)[2] != ".md" # non .md files are ignored
            continue
        elseif haskey(titles, relpath) # case 'title => path'
            push!(pages_list, titles[relpath] => relpath)
        else # case 'title'
            push!(pages_list, relpath)
        end
    end

    return pages_list
end

function list_pages()
    root_dir = joinpath(@__DIR__, "src")
    pages_list = recursively_list_pages(root_dir)

    return ["index.md"; pages_list]
end

makedocs(;
    modules = [TrustRegionLeastSquares],
    authors = "vcantarella <vcantarella@gmail.com> and contributors",
    repo = Remotes.GitHub("vcantarella", "TrustRegionLeastSquares.jl"),
    sitename = "TrustRegionLeastSquares.jl",
    format = Documenter.HTML(;
        canonical = "https://vcantarella.github.io/TrustRegionLeastSquares.jl",
    ),
    pages = list_pages(),
)

deploydocs(; repo = "github.com/vcantarella/TrustRegionLeastSquares.jl", devbranch = "main")
