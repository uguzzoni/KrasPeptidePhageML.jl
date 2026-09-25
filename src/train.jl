###############################################################################
# Training.
#
# `learn!` here is the original `learn!` (`model.jl`:56-89) with ONE correction.
#
# The original builds its loss as
#
#     Flux.gradient(model) do m
#         ll = log_likelihood(m, data; ...)
#         return reg() - ll          # `reg` is a 0-argument closure over `model`
#     end
#
# Under Flux's explicit gradient mode (Flux 0.16 / `Flux.setup`), Zygote tracks by
# data flow from the argument `m`. Anything reached through the captured OUTER
# `model` is a constant, so `reg()` contributes EXACTLY ZERO GRADIENT and the
# penalty is silently ignored. Measured on the real model: 0.0 for the original
# form against 1.951370e1 (= 2*alpha*sum|W|) for the corrected one.
#
# Under Flux's older implicit mode (`gradient(() -> loss(), params(model))`)
# tracking was by parameter identity and the same closure DID contribute
# gradients — so the idiom was correct when written and broke on the Flux
# upgrade. Every training script in the original project is affected.
#
# The fix: take `regf(m)`, a function OF the differentiated model.
# `test/runtests.jl` pins this with a gradient test.
###############################################################################

"""
    l2_nn(model, α; γ=1.0)

L2 penalty on the dense layers of modes 1-4 (layers 2 and 3 of each `DeepEnergy`
chain; layer 1 is `Flux.flatten`). Modes 3-4 are weighted by `γ`.

This is the *fixed* form — `model.states[mode]` in both loops. The copy in
`training_08-2025/training_model.ipynb` indexes `states[1]` twice in the first
loop, so mode 1 is counted double and mode 2 is not regularized at all.
"""
function l2_nn(model, α; γ = 1.0)
    s = 0.0
    for mode = 1:2, layer = 2:3
        s += sum(abs2, model.states[mode].m.layers[layer].weight) +
             sum(abs2, model.states[mode].m.layers[layer].bias)
    end
    for mode = 3:4, layer = 2:3
        s += γ * sum(abs2, model.states[mode].m.layers[layer].weight) +
             sum(abs2, model.states[mode].m.layers[layer].bias)
    end
    return α * s
end

"""
    learn!(model, data; opt, epochs, regf, batchsize, callback, history, rare_binding)

Maximise the log-likelihood of `data` under `model`, minus `regf(model)`.

`regf` is a function of the model (or `nothing`). See the note at the top of this
file for why it cannot be a zero-argument closure.

`callback` is called after every optimiser update, which is how evaluation on a
held-out set is interleaved without disturbing the optimiser state.
"""
function learn!(model::Model, data::Data;
                opt = Flux.AdaBelief(),
                epochs = 1:100,
                regf = nothing,
                batchsize::Int = number_of_sequences(data),
                callback = nothing,
                history = TrainHistory(),
                rare_binding::Bool = false)

    opt_state = Flux.setup(opt, model)
    @assert number_of_rounds(model) == number_of_rounds(data)
    for epoch in epochs2range(epochs)
        for batch in minibatches(data, batchsize)
            grads = Flux.gradient(model) do m
                ll = log_likelihood(m, data; rare_binding = rare_binding, batch = batch)
                return (regf === nothing ? zero(ll) : regf(m)) - ll
            end
            Flux.update!(opt_state, model, grads[1])
            callback === nothing || callback()
        end
        push!(history, :loglikelihood,
              log_likelihood(model, data; rare_binding = rare_binding))
    end
    return history
end

"""
Training schedule shared by every run, as `(batchsize, epochs)` pairs.

A short small-batch warm-up is kept deliberately: the four per-mode networks
start near-symmetric and high-noise SGD is what separates them early — which is
exactly the `mutation`/`wt` axis the specificity calls depend on.

NB on epoch budget: held-out likelihood peaks within the first few epochs and
degrades monotonically afterwards, while training likelihood keeps improving. A
long schedule is appropriate for *fitting* the published model, not for a model
meant to generalise to unseen peptides. See `metrics.jl`.
"""
const DEFAULT_SCHEDULE = [(2^6, 10), (2^9, 250), (2^12, 90), (2^14, 50)]   # 400 epochs

"Shorter schedule for refits that start from an already-fitted model."
const WARM_SCHEDULE = [(2^9, 100), (2^12, 20)]                            # 120 epochs

"Two-epoch schedule for smoke tests."
const PILOT_SCHEDULE = [(2^9, 1), (2^12, 1)]

"""
    train_model!(model, data; schedule, α, γ, regularize, eval_data, evals_per_phase)

Run `schedule` through `learn!`, one call per phase so the optimiser state
persists across the epochs of a phase (as the original training scripts did).

If `eval_data` is given, the held-out log-likelihood is recorded
`evals_per_phase` times per phase through the `callback` hook, which does not
perturb training. Returns `(history, timings, curve)`.
"""
function train_model!(model::Model, data::Data;
                      schedule = DEFAULT_SCHEDULE,
                      α::Real = 0.05, γ::Real = 1.0,
                      regularize::Bool = true,
                      eval_data = nothing,
                      evals_per_phase::Int = 8,
                      verbose::Bool = true)

    history = TrainHistory()
    timings = NamedTuple[]
    curve = NamedTuple[]
    regf = regularize ? (m -> l2_nn(m, α; γ = γ)) : nothing

    for (ph, (batch, nep)) in enumerate(schedule)
        nsteps = minibatch_count(data, batch) * nep
        every = max(1, nsteps ÷ max(1, evals_per_phase))
        step = Ref(0)
        cb = eval_data === nothing ? nothing : function ()
            step[] += 1
            if step[] % every == 0
                push!(curve, (phase = ph, batch = batch, step = step[],
                              ll_test = log_likelihood(model, eval_data)))
            end
            return nothing
        end

        t0 = time()
        learn!(model, data; epochs = 1:nep, batchsize = batch, history = history,
               opt = Flux.AdaBelief(), regf = regf, callback = cb)
        dt = time() - t0
        push!(timings, (phase = ph, batch = batch, epochs = nep, steps = nsteps, seconds = dt))
        verbose && @info @sprintf("phase %d  batch=%-6d epochs=%-4d %8.1f s  ll=%.4f",
                                  ph, batch, nep, dt, last(history[:loglikelihood]))
    end

    return history, timings, curve
end
