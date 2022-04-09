"""
    discriminator_loss(d)

Computes the discriminator loss for discriminator network `d`.
""" # TODO: complete docstring
function discriminator_loss(d, Ỹ, y)
    Flux.logitbinarycrossentropy(d(Ỹ), y) # Warm up is handled by Seq2One object
end

"""
    generator_loss()


""" # TODO: complete docstring
function generator_loss(d, g, s, H, Z)
    Ê = [g(z) for z ∈ Z]
    Ĥ = [s(ê) for ê ∈ Ê]
    # Combine Y vectors
    Ỹ = [hcat(Ê[i], Ĥ[i]) for i ∈ 1:seqlen]
    y = ones(Float32, 1, 2batchsize)
    s_loss = supervised_loss(s, H)
    # TODO: first 2 moments losses
    g_loss_u = discriminator_loss(d, Ỹ, y)
    g_loss_u + 1f3sqrt(s_loss)
end

"""
    reconstruction_loss(e, r, X)

Computes the reconstruction loss for embedder network `e`, recovery network `r` and data `X`.
"""
function reconstruction_loss(e, r, X)
    r(e(X[1])) # Warm up models
    # Compute reconstruction loss on the whole sequence after t=1
    mean(sum(sqrt.(sum(abs2.(x .- r(e(x))), dims=1)) for x ∈ X[2:end]))
end

"""
    joint_reconstruction_loss(e, r, X)

""" # TODO: complete dosctring
function joint_reconstruction_loss(e, r, s, X, η)
    H = [e(x) for x ∈ X]
    r(H[1]) # Warm-up recovery
    g_loss = supervised_loss(s, H)
    e_loss = mean(
        sum(sqrt.(sum(abs2.(x .- r(h)), dims=1)) for (h, x) ∈ zip(H[2:end], X[2:end])))
    η * e_loss + 1f-1g_loss
end

"""
    supervised_loss(s, H)

Computes the supervised loss for supervisor network `s` and embedding data `H`.
"""
function supervised_loss(s, H)
    s(H[1]) # Warm up model
    # Compute supervised loss on the whole sequence after t=1
    mean(sum(sqrt.(sum(abs2.(h .- s(h)), dims=1)) for h ∈ H[2:end]))
end