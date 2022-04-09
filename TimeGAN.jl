using Flux
using Statistics
using UnPack
using Zygote: ignore

include("loss_functions.jl")

@Base.kwdef mutable struct TimeGAN
    embedder
    recovery
    generator
    supervisor
    discriminator
    opt_er0 = ADAM()
    opt_er = ADAM()
    opt_gs = ADAM()
    opt_s = ADAM()
    opt_d = ADAM()
    batchsize::Int = 128
    z_dim::Int = 5 # TODO: infer from data and match dim?
    seqlen::Int = 24
    λ::Float32 = 1f0 # Loss balancing hyperparameter
    η::Float32 = 1f1 # Loss balancing hyperparameter
    discriminator_loss::Float32 = 0f0
    generator_loss::Float32 = 0f0
    joint_reconstruction_loss::Float32 = 0f0
    reconstruction_loss::Float32 = 0f0
    supervised_loss::Float32 = 0f0
end

# ----- Full TimeGAN training loop ---------------------------------------------------------
"""
    train(tg, data; epochs, verbose, verbosity)

Full training loop for a TimeGAN network `tg`.
"""
function train(
    tg::TimeGAN, data; 
    epochs::Int = 50, verbose::Bool = false, verbosity::Int = 1_000
)
    verbose && println("===== Start TimeGAN training ", "="^51)
    d, e, g, r, s = tg.discriminator, tg.embedder, tg.generator, tg.recovery, tg.supervisor
    @unpack opt_d, opt_er0, opt_er, opt_gs, opt_s, batchsize, z_dim, seqlen, λ, η = tg
    Flux.trainmode!([d, e, g, r, s])
    # 1. Embedding network training --------------------------------------------------------
    verbose && println("----- Start Embedding network training ", "-"^41)
    θ_er = Flux.params([e, r]) # Embedder & Recovery parameters
    for epoch ∈ 1:epochs
        Flux.reset!([e, r])
        X = sample_batch(data, tg)
        ∇ = gradient(θ_er) do
            loss = η*reconstruction_loss(e, r, X)
            ignore() do
                tg.reconstruction_loss = loss
            end
            loss
        end
        Flux.update!(opt_er0, θ_er, ∇)
        if verbose # Checkpoint, print current loss
            if (epoch % verbosity == 0) || epoch == 1
                println("Epoch $epoch, reconstruction loss: ", tg.reconstruction_loss)
            end
        end
    end
    verbose && println("-"^80)

    # 2. Training only with supervised loss ------------------------------------------------
    verbose && println("----- Start Supervised network only training ", "-"^35)
    θ_s = Flux.params(s) # Supervisor parameters
    for epoch ∈ 1:epochs
        Flux.reset!([e, s]) # Embedder also needs to be reset to compute embedding
        X = sample_batch(data, tg)
        H = [e(x) for x ∈ X]
        ∇ = gradient(θ_s) do 
            loss = supervised_loss(s, H)
            ignore() do
                tg.supervised_loss = loss
            end
            loss
        end
        Flux.update!(opt_s, θ_s, ∇)
        if verbose # Checkpoint, print current loss
            if (epoch % verbosity == 0) || epoch == 1
                println("Epoch $epoch, supervised loss: ", tg.supervised_loss)
            end
        end
    end
    verbose && println("-"^80)

    # 3. Joint training --------------------------------------------------------------------
    verbose && println("----- Start Joint training ", "-"^53)
    θ_d = Flux.params(d)
    θ_gs = Flux.params([g, s])
    for epoch ∈ 1:epochs
        # Generator training ---------------------------------------------------------------
        for _ ∈ 1:2 # Train generator twice more than discriminator
            Flux.reset!([d, e, g, r, s])
            X = sample_batch(data, tg)
            Z = random_generator(batchsize, z_dim, seqlen)
            H = [e(x) for x ∈ X]
            # Traing generator
            ∇ = gradient(θ_gs) do
                loss = generator_loss(d, g, s, H, Z)
                ignore() do
                    tg.generator_loss = loss
                end
                loss
            end
            Flux.update!(opt_gs, θ_gs, ∇)
            # Train embedder
            Flux.reset!([e, r, s])
            ∇ = gradient(θ_er) do
                loss = joint_reconstruction_loss(e, r, s, X, η)
                ignore() do
                    tg.joint_reconstruction_loss = loss
                end
                loss
            end
            Flux.update!(opt_er, θ_er, ∇)
        end

        # Discriminator training -----------------------------------------------------------
        Flux.reset!([d, e, g, s])
        X = sample_batch(data, tg)
        Z = random_generator(batchsize, z_dim, seqlen)
        # Combine Y vectors
        H = [embedder(x) for x ∈ X]
        Ê = [generator(z) for z ∈ Z]
        Ĥ = [supervisor(ê) for ê ∈ Ê]
        Ỹ = [hcat(H[i], Ê[i], Ĥ[i]) for i ∈ 1:seqlen]
        y = hcat(ones(Float32, 1, batchsize), zeros(Float32, 1, 2batchsize))
        # Train discriminator
        ∇ = gradient(θ_d) do
            loss = discriminator_loss(d, Ỹ, y)
            ignore() do
                tg.discriminator_loss = loss
            end
            loss
        end
        # Only update discriminator if it does not perform well
        tg.discriminator_loss > .15 && Flux.update!(opt_d, θ_d, ∇)
        if verbose # Checkpoint, print current loss
            if (epoch % verbosity == 0) || epoch == 1
                println("Epoch $epoch, discriminator loss: ", tg.discriminator_loss)
            end
        end
    end
end

"""
    generate_data(tg)

Generates data from a TimeGAN network `tg`.
"""
function generate_data(tg::TimeGAN)
    g, r, s = tg.generator, tg.recovery, tg.supervisor
    @unpack batchsize, z_dim, seqlen = tg
    Z = random_generator(batchsize, z_dim, seqlen)
    Flux.reset!([g, r, s])
    [r(s(g(z))) for z ∈ Z]
end