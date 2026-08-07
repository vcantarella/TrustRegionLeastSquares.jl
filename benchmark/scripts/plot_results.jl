# Rebuild the poster figures from saved benchmark results — no solver runs, so plot
# styling can be iterated in seconds. Run the benchmark scripts first to produce the
# CSVs, then:
#   julia --project=benchmark benchmark/scripts/plot_results.jl
using DataFrames, CSV
include(joinpath(@__DIR__, "..", "evaluate.jl"))

results_dir = normpath(joinpath(@__DIR__, "..", "results"))

for (csvname, figname, legend) in [
    ("nlls_results.csv", "nlls_solver_performance", true),
    ("nlls_results_delay.csv", "nlls_solver_performance_delay", false),
]
    path = joinpath(results_dir, csvname)
    if !isfile(path)
        println("skipping $figname: $path not found (run the benchmark script first)")
        continue
    end
    df_proc = compare_with_best(CSV.read(path, DataFrame))
    # White-background PNG for previewing (ink text is unreadable when a viewer shows
    # transparency as dark); transparent SVG for the poster.
    fig_png = build_performance_plots(df_proc; legend = legend, background = :white)
    save(joinpath(plots_dir(), figname * ".png"), fig_png)
    fig_svg = build_performance_plots(df_proc; legend = legend)
    save(joinpath(plots_dir(), figname * ".svg"), fig_svg)
    println("wrote $figname.{png,svg} from $csvname")
end
