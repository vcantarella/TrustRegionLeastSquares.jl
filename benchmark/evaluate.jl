using DataFrames
using CairoMakie

# Directory where benchmark performance plots are written: the repo-root `test_plots/`,
# resolved relative to THIS file (not the working directory) so scripts save to the same
# place no matter where julia is launched from. Created on demand.
function plots_dir()
    dir = normpath(joinpath(@__DIR__, "..", "test_plots"))
    isdir(dir) || mkpath(dir)
    return dir
end

function compare_with_best(df::DataFrame)
    # Use standard DataFrames - no macro BS
    df_proc = copy(df)

    # Find minimum solution for each problem
    min_solutions =
        combine(groupby(df_proc, :problem), :final_cost => minimum => :min_solution)
    df_proc = leftjoin(df_proc, min_solutions, on = :problem)

    # Add comparison columns
    df_proc.final_close = abs.(df_proc.final_cost .- df_proc.min_solution) .<= 1e-4
    df_proc.final_close_abs =
        abs.(df_proc.final_cost .- df_proc.min_solution) ./ abs.(df_proc.min_solution) .<
        1e-4
    df_proc.is_success = df_proc.final_close .|| df_proc.final_close_abs
    df_proc.gap_to_best = df_proc.final_cost .- df_proc.min_solution

    return df_proc
end

function evaluate_solvers(df_proc::DataFrame)
    # Use standard DataFrames - no more Tidier headaches
    grouped_df = groupby(df_proc, :solver)
    # Median time and iterations are computed over SUCCESSFUL solves only. Failed runs
    # may carry time = Inf and iterations = 0 (see the catch branch in dispatch.jl), and runs
    # that converge to a worse-than-best point are not successes either, so including them
    # would make the timing comparison meaningless. A solver with no successes reports NaN.
    median_success(v, mask) = any(mask) ? median(v[mask]) : NaN
    summary_df = combine(
        grouped_df,
        :is_success => (x -> sum(x) / length(x)) => :percentage_success,
        [:iterations, :is_success] => median_success => :median_iterations_success,
        [:time, :is_success] => median_success => :median_time_success,
    )
    summary_df = sort(summary_df, :percentage_success, rev = true)
    return summary_df
end

function build_performance_plots(df_proc::DataFrame)
    solvers = sort(unique(df_proc.solver))
    n_solvers = length(solvers)
    num_problems = length(unique(df_proc.problem))

    # One distinct color per solver (tab20 gives up to 20 well-separated categorical hues),
    # reused in BOTH panels so a solver looks identical in the profile and the bar chart.
    # Colors alone aren't enough to tell ~13 curves apart, so we also cycle the line style.
    palette = Makie.resample_cmap(:tab20, max(n_solvers, 2))
    color_of = Dict(s => palette[i] for (i, s) in enumerate(solvers))
    linestyles = [:solid, :dash, :dot, :dashdot]

    fig = Figure(size = (1100, 850))

    # Panel 1: performance profile — for each solver, the fraction of all problems it has
    # SOLVED within a given wall-clock time. Only successful, finite-time solves count; we
    # plot a proper right-continuous step (a solver's curve jumps as each problem is solved).
    ax1 = Axis(
        fig[1, 1],
        xlabel = "Time (s, log scale)",
        ylabel = "Fraction of problems solved",
        title = "Performance profile",
        xscale = log10,
    )
    for (i, solver_n) in enumerate(solvers)
        df_s = filter(
            r -> r.solver == solver_n && r.is_success && isfinite(r.time) && r.time > 0,
            df_proc,
        )
        isempty(df_s) && continue
        sort!(df_s, :time)
        frac = (1:nrow(df_s)) ./ num_problems
        stairs!(
            ax1,
            df_s.time,
            frac;
            label = solver_n,
            color = color_of[solver_n],
            linestyle = linestyles[mod1(i, length(linestyles))],
            linewidth = 2,
            step = :post,
        )
    end
    ylims!(ax1, 0, 1)

    # Panel 2: success rate as a HORIZONTAL bar chart, sorted best-first, so the full solver
    # names sit on the y-axis with room (no rotated/truncated/overlapping x labels). Bars are
    # colored to match each solver's profile curve.
    summary_df = evaluate_solvers(df_proc)   # already sorted by percentage_success desc
    n = nrow(summary_df)
    ax2 = Axis(
        fig[2, 1],
        xlabel = "Success rate (%)",
        title = "Success rate by solver",
        yticks = (1:n, summary_df.solver),
        yreversed = true,   # highest success rate on top
    )
    barplot!(
        ax2,
        1:n,
        summary_df.percentage_success .* 100;
        direction = :x,
        color = [color_of[s] for s in summary_df.solver],
    )
    xlims!(ax2, 0, 100)

    Legend(fig[1, 2], ax1, "Solvers", merge = true, framevisible = true)
    resize_to_layout!(fig)
    return fig
end
