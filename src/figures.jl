###############################################################################
# The combined 3x3 robustness figure (CairoMakie).
#
#   row 1  read depth        — labels / energies / ranking vs sequencing depth
#   row 2  read bootstrap    — per-mode agreement / specificity plane / overlap
#   row 3  cross-validation  — per-mode agreement / label stability  / overlap
#
# Everything is drawn from `results/perturbations.jld2`, a ~200 KB file holding
# only what the panels need: the 369 designed candidates, the reference model's
# per-mode energies on them, and one 369x5 `E` per perturbation run. The fitted
# models and the 20 438-row evaluation panel of the original runs are not kept.
#
# SVG output is fully vector, but NOT editable text: Cairo's SVG surface subsets
# every glyph into an outline definition referenced by <use>, and no CairoMakie
# option changes that. The original matplotlib figure set `svg.fonttype = "none"`
# and did emit real <text>. If a label needs changing, change it here and re-run.
###############################################################################

const CLASS_COLOR = Dict(:mut_specific => "#C44E52", :wt_specific => "#4C72B0",
                         :cross_specific => "#DD8452", :none => "#999999")
const COL_PEARSON = "#4C72B0"
const COL_SPEARMAN = "#C44E52"

"""
Refitting the reference model on *identical* data already loses ~5 % of the
published labels to minibatch noise. 95 % is therefore the ceiling of the
depth panel, not 100 %; only the shortfall below it is a sampling effect.
"""
const SAME_DATA_CONTROL = 0.95

"""
    load_perturbations(; path)

The compact artifact: `(sequence, E_reference, runs)`, where each run is
`(kind, frac, rep, E)` with `kind ∈ ("resample", "cv")` and `E` the 369x5
per-mode energy matrix of that refit.

Grouped for the figure by `kind` and, for the resamples, by read fraction:
row 1 uses every fraction, row 2 the `frac == 1` bootstrap, row 3 the CV folds.
"""
function load_perturbations(; path = joinpath(repo_root(), "results", "perturbations.jld2"))
    isfile(path) || error("perturbation results not found: $path\n" *
                          "run scripts/02_perturbations.jl, or use the committed copy in results/")
    d = JLD2.load(path)
    return (sequence = d["sequence"], E_reference = d["E_reference"], runs = d["runs"])
end

# score by which candidates are ranked: mutant-selectivity, E[wt] - E[mutation]
_score(E) = Float64.(E[:, MODES[:wt]]) .- Float64.(E[:, MODES[:mutation]])

#------------------------------------------------------------------------------
# row 1 — rarefaction: three curves against read depth
#------------------------------------------------------------------------------
"""
    _panel_depth(ax, by_frac, Eref, what; topn)

One rarefaction curve. `what` selects the quantity plotted against the retained
read fraction:

* `:labels`  — share of published specificity calls reproduced (min/max whiskers)
* `:energy`  — Spearman of `E[mutation]` and of `E[wt]` against the reference
* `:ranking` — top-`topn` Jaccard of the selectivity ranking

They are drawn together because they are the same sweep read three ways, and
the point of the row is that they disagree: the labels collapse at 25 % depth
while the rankings hold, because a small systematic shift sweeps candidates
across fixed absolute cutoffs.
"""
function _panel_depth(ax, by_frac, Eref, what::Symbol; topn::Int = 100)
    fr = sort(collect(keys(by_frac)))
    x = 100 .* fr
    if what === :labels
        ref = published_labels(Eref)
        m = [mean(mean(published_labels(r.E) .== ref) for r in by_frac[f]) for f in fr]
        lo = [minimum(mean(published_labels(r.E) .== ref) for r in by_frac[f]) for f in fr]
        hi = [maximum(mean(published_labels(r.E) .== ref) for r in by_frac[f]) for f in fr]
        c = CLASS_COLOR[:mut_specific]
        errorbars!(ax, x, 100 .* m, 100 .* (m .- lo), 100 .* (hi .- m); color = c, whiskerwidth = 12)
        scatterlines!(ax, x, 100 .* m; color = c, markersize = 13, linewidth = 2.5)
        hlines!(ax, [100SAME_DATA_CONTROL]; color = :black, linestyle = :dash, linewidth = 1.3)
        text!(ax, x[1], 100SAME_DATA_CONTROL + 1.5; text = "same-data control", fontsize = 14)
        ax.ylabel = "published labels\nreproduced (%)"
        ax.title = "specificity calls vs read depth"
        ylims!(ax, 20, 103)
    elseif what === :energy
        for (mode, col, lab) in ((:mutation, CLASS_COLOR[:mut_specific], "E[mutation]"),
                                 (:wt, CLASS_COLOR[:wt_specific], "E[wt]"))
            i = MODES[mode]
            y = [mean(corspearman(Float64.(Eref[:, i]), Float64.(r.E[:, i])) for r in by_frac[f])
                 for f in fr]
            scatterlines!(ax, x, y; color = col, markersize = 13, linewidth = 2.5, label = lab)
        end
        ax.ylabel = "Spearman vs published"
        ax.title = "energy agreement vs read depth"
        ylims!(ax, 0, 1.04)
        axislegend(ax; position = :rb, framevisible = false)
    else
        base = _score(Eref)
        y = [mean(topn_overlap([base, _score(r.E)], topn)[1, 2] for r in by_frac[f]) for f in fr]
        scatterlines!(ax, x, y; color = "#55A868", markersize = 13, linewidth = 2.5)
        ax.ylabel = "top-$topn Jaccard\nvs published"
        ax.title = "ranking overlap vs read depth"
        ylims!(ax, 0, 1.04)
    end
    ax.xlabel = "sequencing reads retained (%)"
    return ax
end

#------------------------------------------------------------------------------
# rows 2 and 3 — per-mode agreement, the plane, ranking overlap, label stability
#------------------------------------------------------------------------------
"Pearson and Spearman of every refit against the reference, one column per mode."
function _panel_mode_agreement(ax, runs, Eref; title)
    tab = agreement_table(Eref, [r.E for r in runs], ["r$i" for i in eachindex(runs)])
    modes = unique(tab.mode)
    for (k, m) in enumerate(modes)
        sub = tab[tab.mode .== m, :]
        scatter!(ax, fill(k - 0.15, nrow(sub)), sub.pearson; color = (COL_PEARSON, 0.85),
                 markersize = 11, label = "Pearson")
        scatter!(ax, fill(k + 0.15, nrow(sub)), sub.spearman; color = (COL_SPEARMAN, 0.85),
                 markersize = 11, marker = :rect, label = "Spearman")
    end
    ax.xticks = (collect(1:length(modes)), string.(modes))
    ax.xticklabelrotation = π / 9
    ax.ylabel = "agreement with\npublished energies"
    ax.title = title
    ylims!(ax, 0.6, 1.02)
    axislegend(ax; position = :lb, framevisible = false, merge = true)
    return ax
end

"""
The `E[mutation]` / `E[wt]` plane: every refit in grey behind the reference
points, coloured by their published class, with the four classification cutoffs.
The spread of the grey cloud across a dashed line is the label instability of
row 1 made visible.
"""
function _panel_plane(ax, runs, Eref; title, show_legend = false)
    imut, iwt = MODES[:mutation], MODES[:wt]
    for r in runs
        scatter!(ax, r.E[:, imut], r.E[:, iwt]; markersize = 3.5, color = ("#bbbbbb", 0.18))
    end
    ref_lab = published_labels(Eref)
    for c in (:mut_specific, :cross_specific, :wt_specific, :none)
        k = findall(==(c), ref_lab)
        isempty(k) && continue
        scatter!(ax, Eref[k, imut], Eref[k, iwt]; markersize = 7, color = CLASS_COLOR[c],
                 label = replace(string(c), "_" => "-"))
    end
    th = PUBLISHED_THRESHOLDS
    vlines!(ax, [th.mut_hi]; color = :red, linestyle = :dash, linewidth = 1.2)
    vlines!(ax, [th.wt_mut_hi]; color = :orange, linestyle = :dash, linewidth = 1.2)
    hlines!(ax, [th.wt_lo]; color = :red, linestyle = :dash, linewidth = 1.2)
    hlines!(ax, [th.wt_wt_hi]; color = :blue, linestyle = :dash, linewidth = 1.2)
    ax.xlabel = "E[mutation]"; ax.ylabel = "E[wt]"; ax.title = title
    show_legend && axislegend(ax; position = :rt, framevisible = false, patchsize = (10, 10))
    return ax
end

"Pairwise top-`topn` Jaccard of the selectivity ranking, reference first."
function _panel_overlap(gl, runs, Eref; title, topn::Int = 100, labels = nothing)
    ax = Axis(gl[1, 1]; title = title, xticklabelrotation = π / 2)
    M = topn_overlap(vcat([_score(Eref)], [_score(r.E) for r in runs]), topn)
    n = size(M, 1)
    hm = heatmap!(ax, 1:n, 1:n, permutedims(M); colorrange = (0, 1), colormap = :viridis)
    labs = vcat(["pub."], labels === nothing ? string.(eachindex(runs)) : labels)
    ax.xticks = (1:n, labs); ax.yticks = (1:n, labs)
    ax.yreversed = true
    if n <= 6                      # annotate only where the cells can hold a number
        for i in 1:n, j in 1:n
            text!(ax, j, i; text = @sprintf("%.2f", M[i, j]), align = (:center, :center),
                  fontsize = 12, color = M[i, j] > 0.55 ? :black : :white)
        end
    end
    Colorbar(gl[1, 2], hm; label = "top-$topn Jaccard", width = 12)
    colgap!(gl, 8)
    return ax
end

"k of k refits agreeing -> dark green, 0 of k -> red."
_stability_color(cmap, j, k) = cmap[0.08 + 0.84 * (j / k)]

"""
Label stability by published class: one bar per class, split by how many of the
refits reproduce that candidate's published label (all … none), with the
absolute counts annotated. Grouping by class rather than by agreement count
makes the comparison that matters — mutant-specific vs cross-specific — readable
straight off the axis.
"""
function _panel_label_stability(gl, runs, Eref; title)
    ag = label_agreement(Eref, [r.E for r in runs])
    k = length(runs)
    classes = [c for c in (:mut_specific, :cross_specific, :wt_specific, :none)
               if any(ag.published .== c)]
    cmap = cgrad(:RdYlGn)
    ax = Axis(gl[1, 1]; title = title, ylabel = "candidates in class (%)")

    # Makie stacks by ASCENDING `stack` index, so full agreement must carry the
    # lowest one to sit at the bottom of the bar; `bottoms` is accumulated in
    # that same order, or the annotations land in the wrong band.
    xs = Int[]; hs = Float64[]; grp = Int[]
    labels = Tuple{Float64,Float64,Int}[]
    bottoms = zeros(length(classes))
    for (s, j) in enumerate(k:-1:0)
        for (i, c) in enumerate(classes)
            sub = ag[ag.published .== c, :]
            n = count(==(j), sub.n_agree)
            h = 100 * n / max(1, nrow(sub))
            push!(xs, i); push!(hs, h); push!(grp, s)
            h >= 3.0 && push!(labels, (float(i), bottoms[i] + h / 2, n))
            bottoms[i] += h
        end
    end
    barplot!(ax, xs, hs; stack = grp,
             color = [_stability_color(cmap, k - s + 1, k) for s in grp],
             width = 0.62, strokecolor = :white, strokewidth = 1.0)
    for (x, y, n) in labels
        text!(ax, x, y; text = string(n), align = (:center, :center), fontsize = 13)
    end

    ax.xticks = (1:length(classes),
                 [replace(string(c), "_" => "-\n") * "\nn=" * string(count(==(c), ag.published))
                  for c in classes])
    ylims!(ax, 0, 100)
    # legend built by hand: `barplot!` is one plot object, so it carries one entry
    elems = [PolyElement(color = _stability_color(cmap, j, k)) for j in k:-1:0]
    names = [j == k ? "$j/$k (all)" : (j == 0 ? "$j/$k (none)" : "$j/$k") for j in k:-1:0]
    Legend(gl[1, 2], elems, names, "refits reproducing\nthe published label";
           framevisible = false, labelsize = 13, titlesize = 13, patchsize = (11, 11))
    colgap!(gl, 6)
    return ax
end

#------------------------------------------------------------------------------
# assembly
#------------------------------------------------------------------------------
"""
    figure_combined(; path, topn=100, outfile=nothing)

Build the 3x3 robustness figure from the compact perturbation artifact and
return the `Figure`. If `outfile` is given (a path with or without extension)
the figure is also written as `.svg`, `.png` and `.pdf`.

```julia
fig = figure_combined(outfile = "figures/figure_combined")
```
"""
function figure_combined(; path = joinpath(repo_root(), "results", "perturbations.jld2"),
                         topn::Int = 100, outfile = nothing)
    (; E_reference, runs) = load_perturbations(; path = path)
    Eref = E_reference

    resample = filter(r -> r.kind == "resample", runs)
    folds = filter(r -> r.kind == "cv", runs)
    isempty(resample) && error("no resample runs in $path")
    isempty(folds) && error("no cross-validation runs in $path")
    by_frac = Dict(f => filter(r -> r.frac == f, resample) for f in unique(r.frac for r in resample))
    boot = by_frac[1.0]

    fig = Figure(size = (1620, 1500), fontsize = 15)
    gls = [GridLayout(fig[i, j]) for i in 1:3, j in 1:3]
    ax = Matrix{Any}(undef, 3, 3)

    for (j, what) in enumerate((:labels, :energy, :ranking))
        ax[1, j] = _panel_depth(Axis(gls[1, j][1, 1]), by_frac, Eref, what; topn = topn)
    end
    ax[2, 1] = _panel_mode_agreement(Axis(gls[2, 1][1, 1]), boot, Eref;
                                     title = "per-mode energy agreement")
    ax[2, 2] = _panel_plane(Axis(gls[2, 2][1, 1]), boot, Eref;
                            title = "specificity plane", show_legend = true)
    ax[2, 3] = _panel_overlap(gls[2, 3], boot, Eref; title = "ranking overlap", topn = topn)
    ax[3, 1] = _panel_mode_agreement(Axis(gls[3, 1][1, 1]), folds, Eref;
                                     title = "per-mode energy agreement")
    ax[3, 2] = _panel_label_stability(gls[3, 2], folds, Eref;
                                      title = "label stability by published class")
    ax[3, 3] = _panel_overlap(gls[3, 3], folds, Eref; title = "ranking overlap", topn = topn,
                              labels = ["f$i" for i in eachindex(folds)])

    for i in 1:3, j in 1:3
        Label(gls[i, j][1, 1, TopLeft()], string('a' + 3 * (i - 1) + (j - 1));
              fontsize = 22, font = :bold, padding = (0, 28, 8, 0), halign = :right)
    end
    for (i, txt) in enumerate(("Read depth — subsampling of sequencing reads",
                               "Read bootstrap at full depth — $(length(boot)) replicates",
                               "Cross-validation over sequences — $(length(folds)) folds"))
        Label(fig[i, 1:3, Top()], txt; fontsize = 20, font = :bold, halign = :left,
              padding = (0, 0, 34, 0))
    end
    rowgap!(fig.layout, 62)
    colgap!(fig.layout, 34)

    if outfile !== nothing
        stem = replace(String(outfile), r"\.(svg|png|pdf)$" => "")
        mkpath(dirname(abspath(stem)))
        for ext in ("svg", "png", "pdf")
            save("$stem.$ext", fig)
        end
    end
    return fig
end
