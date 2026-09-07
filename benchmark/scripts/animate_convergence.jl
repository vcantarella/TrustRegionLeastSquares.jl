# Step-by-step convergence animations on 2-parameter NLS problems.
#
# For each (problem, starting point) case, every solver ("This work" + the
# NonlinearSolve selection from compare_unconstrained.jl + pure Gauss-Newton) is
# animated over a log10 cost map: the actual step next to the Gauss-Newton
# direction at the same point, the trust-region radius for TR methods, and a
# per-solver summary card. One MP4 per case.
#
# Note on divergence: on mgh01 pure GN converges in exactly 2 steps from ANY
# start (r2 is linear in x1, then r1 is linear in x2; det J = 10 everywhere).
# mgh02 (Freudenstein-Roth) is the interesting case: J is singular along
# x2 ≈ 2.23 and x2 ≈ -0.90, so GN takes chaotic excursions (up to ‖x‖ ~ 3e4)
# before being captured, while globalized methods can converge to the
# nonzero-residual local minimum instead of the root (5, 4).
#
# Usage: julia --project=benchmark benchmark/scripts/animate_convergence.jl
# Output: benchmark/results/convergence_<tag>.mp4

include(joinpath(@__DIR__, "..", "harness.jl"))
include(joinpath(@__DIR__, "..", "evaluate.jl")) # solver_styles / poster palette

const MAX_ITER = 100
const MAX_ANIM = 30 # animate at most this many steps per solver (chaos gets long)

# (problem, x0, known root, window bound, tag)
CASES = [
    (:mgh01, [-1.2, 1.0], [1.0, 1.0], 2.5, "mgh01_standard"),
    (:mgh01, [3.0, -3.0], [1.0, 1.0], 4.0, "mgh01_alt"),
    (:mgh02, [0.5, -2.0], [5.0, 4.0], 15.0, "mgh02_standard"),
    (:mgh02, [0.5, -0.9], [5.0, 4.0], 15.0, "mgh02_nearsingular"),
]

# ---------------------------------------------------------------------------
# Trace collection. Uniform step record:
#   (x, x_trial, accepted, radius, λ)  — radius/λ = NaN where not applicable
# ---------------------------------------------------------------------------

# Gauss-Newton reference direction at any point (QR handles rank deficiency)
gauss_newton_dir(pd, x) = -(qr(pd.jacobian_func(x)) \ pd.residual_func(x))

function trace_thiswork(pd, x0)
    steps = NamedTuple[]
    nonlinearlstr.lm_trust_region!(
        pd.residual_func!,
        pd.jacobian_func!,
        copy(x0),
        pd.n,
        nonlinearlstr.QRSolve(),
        nonlinearlstr.NoScaling();
        max_iter = MAX_ITER,
        gtol = 1e-8,
        ftol = 1e-8,
        callback = s -> push!(
            steps,
            (; x = s.x, x_trial = s.x_trial, accepted = s.accepted,
                radius = s.radius, λ = NaN),
        ),
    )
    return steps
end

# Defensive cache introspection (@concrete structs; fields vary by algorithm)
trcache(cache) = hasproperty(cache, :trustregion_cache) ? cache.trustregion_cache : nothing
function tr_radius(cache)
    tc = trcache(cache)
    tc !== nothing && hasproperty(tc, :trust_region) || return NaN
    r = tc.trust_region
    return r isa Real ? Float64(r) : NaN
end
function step_accepted(cache)
    tc = trcache(cache)
    tc !== nothing && hasproperty(tc, :last_step_accepted) || return true
    return tc.last_step_accepted
end
function rejected_trial(cache, x_from)
    tc = trcache(cache)
    tc !== nothing && hasproperty(tc, :u_cache) || return copy(x_from)
    return copy(tc.u_cache)
end
# Walk the descent-cache chain to the LM damping cache (geodesic acceleration wraps it)
function lm_lambda(c)
    hasproperty(c, :damping_fn_cache) && return Float64(c.damping_fn_cache.λ)
    hasproperty(c, :descent_cache) && return lm_lambda(c.descent_cache)
    return NaN
end

function trace_nonlinearsolve(pd, alg, x0)
    n_res!(du, u, p) = pd.residual_func!(du, u)
    nl_jac!(J, u, p) = pd.jacobian_func!(J, u)
    nl_func = NonlinearFunction(n_res!, jac = nl_jac!, resid_prototype = zeros(pd.n))
    prob_nl = NonlinearLeastSquaresProblem(nl_func, copy(x0))
    cache = SciMLBase.init(prob_nl, alg; maxiters = MAX_ITER, abstol = 1e-8)
    steps = NamedTuple[]
    while !cache.force_stop && cache.nsteps < cache.maxiters
        x_from = copy(cache.u)
        # read BEFORE step!: the values actually used for this step (λ mutates at step end)
        radius = tr_radius(cache)
        λ = lm_lambda(cache)
        SciMLBase.step!(cache)
        accepted = step_accepted(cache)
        x_trial = accepted ? copy(cache.u) : rejected_trial(cache, x_from)
        push!(steps, (; x = x_from, x_trial, accepted, radius, λ))
        any(!isfinite, cache.u) && break # diverged: stop tracing
    end
    return steps
end

solver_runs(pd, x0) = [
    ("This work", () -> trace_thiswork(pd, x0)),
    ("NonlinearSolve-TR", () -> trace_nonlinearsolve(pd, NonlinearSolve.TrustRegion(), x0)),
    ("NonlinearSolve-LM",
        () -> trace_nonlinearsolve(pd, NonlinearSolve.LevenbergMarquardt(), x0)),
    ("NonlinearSolve-GN", # pure GN: full step, no globalization
        () -> trace_nonlinearsolve(pd, NonlinearSolve.GaussNewton(), x0)),
    ("NonlinearSolve-GNBK",
        () -> trace_nonlinearsolve(pd,
            NonlinearSolve.GaussNewton(linesearch = BackTracking()), x0)),
    ("NonlinearSolve-GNLF",
        () -> trace_nonlinearsolve(pd,
            NonlinearSolve.GaussNewton(linesearch = LiFukushimaLineSearch()), x0)),
]

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------
function fmt(v)
    isnan(v) && return "–"
    v == 0 && return "0"
    av = abs(v)
    1e-3 <= av < 1e4 && return string(round(v; sigdigits = 3))
    e = floor(Int, log10(av))
    return "$(round(v / 10.0^e; sigdigits = 3))e$e" # avoids 5.2599999999999997e-23 repr
end
circle_pts(c, r) =
    Point2f[Point2f(c[1] + r * cos(t), c[2] + r * sin(t)) for t in range(0, 2π; length = 65)]

results_dir = normpath(joinpath(@__DIR__, "..", "results"))
isdir(results_dir) || mkpath(results_dir)
frames_dir = joinpath(results_dir, "animation_frames")
isdir(frames_dir) || mkpath(frames_dir)

# ---------------------------------------------------------------------------
# One animation per (problem, start) case
# ---------------------------------------------------------------------------
function animate_case(prob_name, x0, solution, winlim, tag)
    println("=== case $tag: $prob_name from $x0 ===")
    nlp = getfield(NLSProblems, prob_name)()
    pd = create_nls_functions(nlp)
    draw_lim = 4 * winlim
    clamp2(p) = clamp.(p, -draw_lim, draw_lim)

    traces = Tuple{String,Vector{NamedTuple}}[]
    for (name, runner) in solver_runs(pd, x0)
        try
            steps = runner()
            last_x = isempty(steps) ? x0 :
                     (steps[end].accepted ? steps[end].x_trial : steps[end].x)
            println("  $name: $(length(steps)) steps, final cost = $(pd.obj_func(last_x))")
            isempty(steps) || push!(traces, (name, steps))
        catch e
            println("  $name failed to trace: $e")
        end
    end

    # window from the visited points, ignoring wild excursions (a pure-GN overshoot
    # gets clipped at the axis edge instead of stretching the whole view)
    pts = Point2f[Point2f(x0...), Point2f(solution...)]
    for (_, steps) in traces, s in steps
        for p in (s.x, s.x_trial)
            all(abs.(p) .<= winlim) && push!(pts, Point2f(p...))
        end
    end
    xlo, xhi = extrema(p[1] for p in pts)
    ylo, yhi = extrema(p[2] for p in pts)
    mx, my = 0.15 * (xhi - xlo) + 0.1, 0.15 * (yhi - ylo) + 0.1
    xs = range(xlo - mx, xhi + mx; length = 400)
    ys = range(ylo - my, yhi + my; length = 400)
    Zlog = [log10(max(pd.obj_func([xv, yv]), 1e-8)) for xv in xs, yv in ys]

    styles = solver_styles(first.(traces))
    # GNLF and pure GN are not in SOLVER_ORDER, so solver_styles wraps them onto the
    # first palette slots — colliding with "This work". Pin them to free slots instead.
    if haskey(styles, "NonlinearSolve-GNLF")
        styles["NonlinearSolve-GNLF"] =
            (color = Makie.to_color("#9c5410"), marker = :rtriangle)
    end
    if haskey(styles, "NonlinearSolve-GN")
        styles["NonlinearSolve-GN"] = (color = Makie.to_color("#e87ba4"), marker = :ltriangle)
    end

    fig = Figure(size = (900, 680), fontsize = 16)
    ax = Axis(fig[1, 1]; xlabel = "x₁", ylabel = "x₂", title = String(prob_name))
    hm = heatmap!(ax, xs, ys, Zlog; colormap = :viridis)
    # keep the axis pinned to the cost map; long GN arrows get clipped instead of
    # stretching the view. Equal data scales so a TR radius renders as a true circle.
    limits!(ax, xs[1], xs[end], ys[1], ys[end])
    ax.aspect = DataAspect()
    Colorbar(fig[1, 2], hm; label = "log₁₀ cost 0.5‖r‖²")
    scatter!(ax, [Point2f(x0...)]; marker = :circle, markersize = 12,
        color = :white, strokecolor = :black, strokewidth = 1.5)
    scatter!(ax, [Point2f(solution...)]; marker = :star5, markersize = 20,
        color = :white, strokecolor = :black, strokewidth = 1.5)

    # Dynamic elements
    solver_color = Observable(Makie.to_color(:black))
    path = Observable(Point2f[])       # accepted iterates so far
    rej = Observable(Point2f[])        # rejected trial points
    step_seg = Observable(Point2f[])   # actual step (with manual arrowhead)
    gn_seg = Observable(Point2f[])     # Gauss-Newton direction
    circ = Observable(Point2f[])       # trust-region circle

    lines!(ax, path; color = solver_color, linewidth = 2.5)
    scatter!(ax, path; color = solver_color, markersize = 8)
    scatter!(ax, rej; color = (:white, 0.7), marker = :xcross, markersize = 10)
    lines!(ax, circ; color = :white, linestyle = :dot, linewidth = 2)
    lines!(ax, gn_seg; color = :black, linestyle = :dash, linewidth = 2.5)
    lines!(ax, step_seg; color = solver_color, linewidth = 3.5)

    # In-plot annotations: labels at the arrow tips, an info block in the corner,
    # and a per-solver summary card between segments
    span_x = Float32(xs[end] - xs[1])
    span_y = Float32(ys[end] - ys[1])
    off = 0.015f0 * span_x
    # keep labels inside the view even when the arrow tip is clipped at the edge
    labpos(p) = Point2f(
        clamp(p[1], xs[1] + 0.08f0 * span_x, xs[end] - 0.08f0 * span_x),
        clamp(p[2], ys[1] + 0.05f0 * span_y, ys[end] - 0.05f0 * span_y),
    )
    step_lab_pos = Observable(Point2f(xs[1], ys[1]))
    step_lab = Observable("")
    gn_lab_pos = Observable(Point2f(xs[1], ys[1]))
    gn_lab = Observable("")
    info_text = Observable("")
    card_text = Observable("")
    text!(ax, step_lab_pos; text = step_lab, color = :white, font = :bold,
        fontsize = 15, align = (:center, :bottom))
    text!(ax, gn_lab_pos; text = gn_lab, color = :white, font = :bold,
        fontsize = 15, align = (:center, :top))
    text!(ax, Point2f(xs[1] + 0.02f0 * span_x, ys[end] - 0.02f0 * span_y);
        text = info_text, color = :white, font = :bold, fontsize = 15,
        align = (:left, :top))
    text!(ax, Point2f(xs[1] + 0.5f0 * span_x, ys[1] + 0.5f0 * span_y);
        text = card_text, color = :white, font = :bold, fontsize = 24,
        align = (:center, :center))

    # Manual arrow: segment plus arrowhead (version-proof vs Makie's arrows API)
    function arrow_pts(a, b)
        a, b = Point2f(clamp2(a)...), Point2f(clamp2(b)...)
        v = b - a
        L = sqrt(v[1]^2 + v[2]^2)
        L < 1e-12 && return Point2f[]
        h = min(0.25f0 * L, 0.03f0 * span_x)
        u = v ./ L
        w = Point2f(-u[2], u[1])
        return Point2f[a, b, b - h * (Point2f(u...) - 0.5f0 * w), b,
            b - h * (Point2f(u...) + 0.5f0 * w)]
    end

    # arrows are labeled in-plot; the legend covers only the unlabeled symbols
    Legend(fig[2, 1:2],
        [LineElement(color = :white, linestyle = :dot, linewidth = 2),
            MarkerElement(marker = :xcross, color = :gray50, markersize = 10),
            MarkerElement(marker = :star5, color = :white, strokecolor = :black,
                strokewidth = 1, markersize = 14)],
        ["trust region", "rejected trial", "root"];
        orientation = :horizontal, framevisible = false, padding = (0, 0, 0, 0))
    resize_to_layout!(fig)

    verdict_of(last_x) =
        any(!isfinite, last_x) || norm(last_x) > draw_lim ? "diverged" :
        pd.obj_func(last_x) < 1e-10 ? "converged to root" :
        norm(pd.grad_func(last_x)) < 1e-6 ? "converged to local minimum (cost > 0)" :
        "stopped"

    outfile = joinpath(results_dir, "convergence_$tag.mp4")
    record(fig, outfile; framerate = 5) do io
        for (name, steps) in traces
            solver_color[] = styles[name].color
            path[] = [Point2f(clamp2(steps[1].x)...)]
            rej[] = Point2f[]
            step_seg[] = Point2f[]
            gn_seg[] = Point2f[]
            circ[] = Point2f[]
            ax.title = "$prob_name — $name"
            recordframe!(io)
            anim_steps = steps[1:min(end, MAX_ANIM)]
            for (k, s) in enumerate(anim_steps)
                gn = gauss_newton_dir(pd, s.x)
                step_seg[] = arrow_pts(s.x, s.x_trial)
                gn_seg[] = arrow_pts(s.x, s.x + gn)
                circ[] = (isnan(s.radius) || s.radius <= 0) ? Point2f[] :
                         circle_pts(s.x, s.radius)
                tip = Point2f(clamp2(s.x_trial)...)
                gtip = Point2f(clamp2(s.x + gn)...)
                step_lab_pos[] = labpos(tip + Point2f(0, off))
                step_lab[] = s.accepted ? "step $k" : "step $k (rejected)"
                gn_lab_pos[] = labpos(gtip + Point2f(0, -off))
                gn_lab[] = "Gauss-Newton"
                info_text[] = "step $k/$(length(steps))\n" *
                              "cost = $(fmt(pd.obj_func(s.x)))" *
                              (isnan(s.radius) ? (isnan(s.λ) ? "" : "\nλ = $(fmt(s.λ))") :
                               "\nradius = $(fmt(s.radius))")
                # hold so the reader can compare the step against the GN direction
                for _ = 1:4
                    recordframe!(io)
                end
                if s.accepted
                    path[] = push!(path[], Point2f(clamp2(s.x_trial)...))
                else
                    rej[] = push!(rej[], Point2f(clamp2(s.x_trial)...))
                end
                recordframe!(io)
                # spot-check stills for the first solver
                if name == first(traces)[1] && k in (1, 8)
                    save(joinpath(frames_dir, "frame_$(tag)_step$k.png"), fig)
                end
            end
            step_seg[] = Point2f[]
            gn_seg[] = Point2f[]
            circ[] = Point2f[]
            step_lab[] = ""
            gn_lab[] = ""
            info_text[] = ""
            last_x = steps[end].accepted ? steps[end].x_trial : steps[end].x
            n_rej = count(s -> !s.accepted, steps)
            card_text[] = "$name\n$(verdict_of(last_x))\n" *
                          "$(length(steps)) steps ($n_rej rejected)\n" *
                          "final cost = $(fmt(pd.obj_func(last_x)))" *
                          (length(steps) > MAX_ANIM ?
                           "\n(animated first $MAX_ANIM steps)" : "")
            for _ = 1:8
                recordframe!(io)
            end
            card_text[] = ""
        end
        # Final overlay: all paths + per-solver legend with summary stats
        path[] = Point2f[]
        rej[] = Point2f[]
        for (name, steps) in traces
            accepted = Point2f[Point2f(clamp2(steps[1].x)...)]
            for s in steps
                s.accepted && push!(accepted, Point2f(clamp2(s.x_trial)...))
            end
            last_x = steps[end].accepted ? steps[end].x_trial : steps[end].x
            n_rej = count(s -> !s.accepted, steps)
            stats = "$name — $(length(steps)) steps ($n_rej rej.), " *
                    "cost $(fmt(pd.obj_func(last_x)))"
            lines!(ax, accepted; color = styles[name].color, linewidth = 2.5,
                label = stats)
            scatter!(ax, accepted; color = styles[name].color,
                marker = styles[name].marker, markersize = 8)
        end
        axislegend(ax; position = :lb, framevisible = true,
            backgroundcolor = (:white, 0.7))
        ax.title = "$prob_name — all solver trajectories"
        save(joinpath(frames_dir, "frame_$(tag)_overlay.png"), fig)
        for _ = 1:10
            recordframe!(io)
        end
    end
    println("  -> $outfile")
    finalize(nlp)
end

for (prob_name, x0, solution, winlim, tag) in CASES
    animate_case(prob_name, x0, solution, winlim, tag)
end
