###############################################################################
# 03_figure.jl — rebuild the combined 3x3 robustness figure.
#
#   julia --project=. scripts/03_figure.jl [--out figures/figure_combined] [--topn 100]
#
# Reads only `results/perturbations.jld2` (committed), so this takes seconds and
# needs no training. Regenerate that file with scripts/02_perturbations.jl.
###############################################################################

using KrasPeptidePhageML, Printf

out = let i = findfirst(==("--out"), ARGS)
    i === nothing ? joinpath(@__DIR__, "..", "figures", "figure_combined") : ARGS[i + 1]
end
topn = let i = findfirst(==("--topn"), ARGS)
    i === nothing ? 100 : parse(Int, ARGS[i + 1])
end

figure_combined(; topn = topn, outfile = out)

println("\nwritten:")
for ext in ("svg", "png", "pdf")
    p = "$out.$ext"
    @printf("  %-52s %7.1f KB\n", p, filesize(p) / 1024)
end
