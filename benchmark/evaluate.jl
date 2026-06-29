using DataFrames
using CairoMakie

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
    summary_df = combine(
        grouped_df,
        :is_success => (x -> sum(x) / length(x)) => :percentage_success,
        :iterations => median => :iterations,
        :time => median => :mean_execution_time,
    )
    summary_df = sort(summary_df, :percentage_success, rev = true)
    return summary_df
end

function build_performance_plots(df_proc::DataFrame)
    colormap = :Paired_12
    fig = Figure()
    # Plot 1: Fraction solved vs time
    ax1 = Axis(
        fig[1, 1],
        xlabel = "Time (seconds, log scale)",
        ylabel = "Fraction of Problems Solved",
        ylabelsize = 12,
        title = "Performance Profile",
        xscale = log10,
    )
    solvers = sort(unique(df_proc.solver))
    problems = unique(df_proc.problem)
    num_problems = length(problems)
    c = 1 # color index
    for solver_n in solvers
        df_solver = filter(row -> row.solver == solver_n, df_proc)
        sort!(df_solver, :time)
        df_solver.cumulative_success = cumsum(df_solver.is_success) ./ num_problems
        lines!(
            ax1,
            df_solver.time,
            df_solver.cumulative_success,
            label = solver_n,
            linewidth = 2,
            colormap = colormap,
        )
    end
    # Plot 2: Success rate comparison
    ax2 = Axis(
        fig[2, 1],
        xlabel = "Solver",
        ylabel = "Success Rate (%)",
        title = "Success Rate by Solver",
        xticklabelrotation = (30/180)*π,
    )
    summary_df = evaluate_solvers(df_proc)
    n_solvers = nrow(summary_df)
    truncated_names = [s[1:min(10, length(s))] for s in summary_df.solver]
    ax2.xticks = (1:n_solvers, truncated_names)
    barplot!(
        ax2,
        1:n_solvers,
        summary_df.percentage_success .* 100,
        color = :steelblue,
        alpha = 0.7,
    )
    Legend(fig[1:2, 2], ax1, "Solvers", merge = true)
    resize_to_layout!(fig)
    return fig
end
