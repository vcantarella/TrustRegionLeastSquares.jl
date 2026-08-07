using DataFrames
using CairoMakie
using Printf
using Statistics

# Directory where benchmark performance plots are written: the repo-root `test_plots/`,
# resolved relative to THIS file (not the working directory) so scripts save to the same
# place no matter where julia is launched from. Created on demand.
function plots_dir()
    dir = normpath(joinpath(@__DIR__, "..", "test_plots"))
    isdir(dir) || mkpath(dir)
    return dir
end

# Best (minimum) objective over a problem's solvers, ignoring failures. A failed solve
# carries final_cost = Inf or NaN (see dispatch.jl); plain `minimum` PROPAGATES NaN, so a
# single failing solver would poison the group minimum and mark every solver — including the
# ones that actually solved the problem — as unsuccessful. We take the min over finite costs
# only; if no solver produced a finite cost, the best is Inf (nobody solved it).
function finite_min(v)
    fin = filter(isfinite, v)
    return isempty(fin) ? Inf : minimum(fin)
end

function compare_with_best(df::DataFrame)
    # Use standard DataFrames - no macro BS
    df_proc = copy(df)

    # Find minimum (best) solution for each problem, ignoring failed/NaN runs.
    min_solutions =
        combine(groupby(df_proc, :problem), :final_cost => finite_min => :min_solution)
    df_proc = leftjoin(df_proc, min_solutions, on = :problem)

    # Add comparison columns. A run is "close" to best in absolute OR relative terms; NaN
    # final costs compare false everywhere, so failed runs never count as successes.
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
    total_success(v, mask) = any(mask) ? sum(v[mask]) : NaN
    summary_df = combine(
        grouped_df,
        :is_success => (x -> sum(x) / length(x)) => :percentage_success,
        [:iterations, :is_success] => median_success => :median_iterations_success,
        [:time, :is_success] => median_success => :median_time_success,
        [:time, :is_success] => total_success => :total_time_success,
    )
    summary_df = sort(summary_df, :percentage_success, rev = true)
    return summary_df
end

# Fixed per-solver styling. Color and marker follow the SOLVER, not its position in a
# particular DataFrame: the jacobian-delay figure is drawn without a legend and read
# against the main figure's legend, so a solver must look identical across figures even
# though the two scripts run different solver subsets. Known solvers get a fixed slot;
# names not listed here (other scripts: TRF, PRIMA-BOBYQA, ...) get stable fallback
# slots appended in sorted order.
const SOLVER_ORDER = [
    "This work",
    "Optim-BFGS",
    "Optim-L-BFGS",
    "JSO-TRON",
    "LSO-Levenberg-QR",
    "LsqFit-LM",
    "NLLSsolver-LM",
    "NonlinearSolve-LM",
    "NonlinearSolve-TR",
    "PRIMA-NEWUOA",
    "Scipy-LeastSquares",
]
const SOLVER_MARKERS = [
    :circle,
    :rect,
    :utriangle,
    :diamond,
    :cross,
    :xcross,
    :star5,
    :pentagon,
    :hexagon,
    :ltriangle,
    :rtriangle,
]

# CVD-validated 11-slot categorical palette (adjacent-pair gates all pass: worst
# adjacent protan ΔE 9.1, worst adjacent normal-vision ΔE 19.6). The ORDER is the
# safety mechanism — don't shuffle it. Markers are the secondary encoding for the
# low-contrast slots.
const SOLVER_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#14b8d4",  # cyan
    "#9c5410",  # brown
    "#a0a424",  # olive
]

function solver_styles(solvers)
    palette = Makie.to_color.(SOLVER_COLORS)
    order = copy(SOLVER_ORDER)
    for s in sort(collect(solvers))
        s in order || push!(order, s)
    end
    return Dict(
        s => (
            color = palette[mod1(i, length(palette))],
            marker = SOLVER_MARKERS[mod1(i, length(SOLVER_MARKERS))],
        ) for (i, s) in enumerate(order)
    )
end

function build_performance_plots(
    df_proc::DataFrame;
    legend::Bool = true,
    background = :transparent,
)
    solvers = sort(unique(df_proc.solver))
    num_problems = length(unique(df_proc.problem))
    styles = solver_styles(solvers)

    # Two-column layout: performance profile on the left, legend + summary table
    # stacked on the right. `legend = false` drops the legend (the delay figure reuses
    # the main figure's legend on the poster). Transparent everywhere (figure, axes,
    # legend) so the figure sits directly on the poster background; exported as SVG.
    # Initial size is intentionally SMALLER than the content: every panel has a fixed
    # size, and resize_to_layout! grows the figure to fit exactly — starting larger
    # leaves slack that the auto column dumps between profile and legend as dead air.
    fig = Figure(
        size = (900, 620),
        fontsize = 16,
        backgroundcolor = background,
        figure_padding = 10,
    )

    # Panel 1: Dolan–Moré performance profile (Dolan & Moré 2002). For each solver s and
    # problem p the performance ratio r_ps = t_ps / (best solver's time on p), over
    # SUCCESSFUL solves only (failures: r = ∞, the curve simply never rises for them).
    # ρ_s(τ) = fraction of ALL problems solved within τ × the best time. Unlike absolute
    # wall-clock, the ratio is invariant to per-problem difficulty: ρ_s(1) reads as "how
    # often is s the fastest solver" and the right-hand asymptote as robustness.
    ok = filter(r -> r.is_success && isfinite(r.time) && r.time > 0, df_proc)
    best_time = Dict(g.problem[1] => minimum(g.time) for g in groupby(ok, :problem))
    ratios_of = Dict(
        s => sort([r.time / best_time[r.problem] for r in eachrow(ok) if r.solver == s]) for s in solvers
    )
    τmax = maximum(rs -> isempty(rs) ? 1.0 : rs[end], values(ratios_of); init = 1.0)

    ax1 = Axis(
        fig[1, 1],
        xlabel = "Performance ratio τ = time / best solver's time (log scale)",
        ylabel = "Fraction of problems solved within τ",
        title = "Performance profile",
        xscale = log10,
        yticks = 0:0.2:1,
        backgroundcolor = :transparent,
        # Fixed SIZE (not just height) so the profile panel is pixel-identical in both
        # figures: the delay figure has no legend, and auto sizing would otherwise let
        # its profile grow wider and its row collapse to the table's height.
        height = 540,
        width = 560,
    )
    for solver_n in solvers
        ratios = ratios_of[solver_n]
        isempty(ratios) && continue
        frac = (1:length(ratios)) ./ num_problems
        st = styles[solver_n]
        # Step line from τ = 1 (value 0 until the first solve) out to the global τmax so
        # every curve spans the full axis; markers sit at the actual solve ratios. Line
        # and markers share a label so Legend(merge = true) fuses them into one entry —
        # the marker (not linestyle) is what disambiguates solvers with similar colors.
        stairs!(
            ax1,
            [1.0; ratios; τmax],
            [0.0; frac; frac[end]];
            label = solver_n,
            color = st.color,
            linewidth = 2,
            step = :post,
        )
        scatter!(
            ax1,
            ratios,
            frac;
            label = solver_n,
            color = st.color,
            marker = st.marker,
            markersize = 10,
        )
    end
    ylims!(ax1, 0, 1)

    right = fig[1, 2] = GridLayout()
    if legend
        # Same 3-column arrangement as the single-column layout had (11 entries in 4
        # banks); tellwidth=true so the legend, not the narrower table, sets the
        # right column's width.
        Legend(
            right[1, 1],
            ax1,
            merge = true,
            framevisible = true,
            orientation = :horizontal,
            nbanks = 4,
            labelsize = 12,
            colgap = 12,
            patchlabelgap = 4,
            tellwidth = true,
            halign = :right,   # right-align with the table below (its row labels protrude left)
            backgroundcolor = :transparent,
        )
    end

    # Panel 2: summary TABLE — one row per solver (best success rate on top), a colored
    # cell per metric with the value printed in it: success rate (%), and SPEED-UP of the
    # solver's cumulative time (successful solves only) against the SciPy baseline —
    # "12×" = twelve times faster than SciPy. In both columns darker = better; speed-up
    # is colored on a log scale since it spans orders of magnitude. Translucent
    # single-hue ramps keep the cells soft. Read speed-up together with success: a
    # solver that solves fewer problems accumulates less time by doing less work.
    summary_df = evaluate_solvers(df_proc)   # already sorted by percentage_success desc
    n = nrow(summary_df)
    succ = summary_df.percentage_success .* 100
    times = summary_df.total_time_success
    tfin = filter(isfinite, times)
    # Baseline: SciPy when present; otherwise the slowest solver (other scripts reuse
    # this panel without SciPy), so the column header names whichever is in effect.
    bidx = findfirst(==("Scipy-LeastSquares"), summary_df.solver)
    baseline = bidx === nothing ? (isempty(tfin) ? NaN : maximum(tfin)) : times[bidx]
    baseline_label = bidx === nothing ? "slowest" : "SciPy"
    speedup = baseline ./ times
    lsp = log10.(speedup)
    # Uniform color scaling: min–max normalize each column onto [0, 1] (success on
    # raw %, speed-up on log10) so both columns share ONE heatmap, one ramp, one
    # colorrange — equal tint means equal standing within its column.
    function normcol(v)
        fin = filter(isfinite, v)
        length(unique(fin)) < 2 && return fill(0.5, length(v))
        return (v .- minimum(fin)) ./ (maximum(fin) - minimum(fin))
    end
    pct(v) = !isfinite(v) ? "—" : string(round(Int, v))
    spd(v) =
        !isfinite(v) ? "—" :
        v < 0.095 ? @sprintf("%.3f×", v) :
        v < 0.95 ? @sprintf("%.2f×", v) : v < 9.5 ? @sprintf("%.1f×", v) : @sprintf("%.0f×", v)

    ax2 = Axis(
        right[legend ? 2 : 1, 1],
        title = "Solver summary",
        titlesize = 16,
        titlegap = 10,
        # Cells ~150pt × 26pt. Fills are capped to LIGHT tints of one blue ramp (the
        # value never maps past ~70% of the colormap) so every number reads in plain
        # ink — the tint is a cue for scanning, the printed value is the datum. White
        # hairlines separate cells (surface gaps, not strokes).
        # Solver names are drawn INSIDE the axis (text! below), not as y-tick labels:
        # tick labels are protrusions, and Makie reserves the whole inter-column gap
        # for them — which showed up as dead air between profile and legend.
        height = 26 * n,
        width = 500,
        halign = :right,
        yticklabelsvisible = false,
        yreversed = true,   # highest success rate on top
        xticks = ([1, 2], ["Success %", "× vs $(baseline_label)"]),
        xticklabelsize = 14,
        xticklabelcolor = :gray30,
        xaxisposition = :top,
        xgridvisible = false,
        ygridvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        backgroundcolor = :transparent,
    )
    hidespines!(ax2)
    cells = permutedims(hcat(normcol(succ), normcol(lsp)))   # 2 × n, one value per cell
    heatmap!(
        ax2,
        1:2,
        1:n,
        cells;
        colormap = (:GnBu, 0.7),
        colorrange = (0, 1 / 0.7),   # best-in-column ⇒ ~70% of the ramp: strong tint, ink still reads
        nan_color = :transparent,
    )
    hlines!(ax2, collect(0.5:1:(n + 0.5)); xmin = 0.4, color = :white, linewidth = 2)
    vlines!(ax2, [1.5]; color = :white, linewidth = 2)
    for i in 1:n
        text!(
            ax2,
            0.42,
            i;
            text = summary_df.solver[i],
            align = (:right, :center),
            fontsize = 14,
            color = :black,
        )
        text!(
            ax2,
            1,
            i;
            text = pct(succ[i]),
            align = (:center, :center),
            fontsize = 15,
            color = :black,
        )
        text!(
            ax2,
            2,
            i;
            text = spd(speedup[i]),
            align = (:center, :center),
            fontsize = 15,
            color = :black,
        )
    end
    xlims!(ax2, -0.85, 2.5)   # left zone holds the solver names, cells at x ∈ [0.5, 2.5]

    colgap!(fig.layout, 20)
    rowgap!(right, 14)
    resize_to_layout!(fig)
    return fig
end
