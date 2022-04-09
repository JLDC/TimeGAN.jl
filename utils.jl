using Flux

"""
    Seq2One(rnn, fc)

Special neural network for sequence-to-one modeling, used for the discriminator.
"""
mutable struct Seq2One
    rnn
    fc
end

@Flux.functor Seq2One

function (m::Seq2One)(X)
    # Run RNN layers up to penultimate step
    [m.rnn(x) for x ∈ X[1:end-1]]
    # Apply FC layer to last step only
    m.fc(m.rnn(X[end]))
end

"""
    random_generator(batchsize, z_dim, seqlen)

Generates a batch of data from the latent space `Z`.
"""
random_generator(batchsize, z_dim, seqlen) = 
    [rand(Float32, z_dim, batchsize) for _ ∈ 1:seqlen]

"""
    MinMaxScaler(data)

Normalizes data to the range [0, 1] or back to its original range.
"""
struct MinMaxScaler
    min::Matrix{Float32}
    max::Matrix{Float32}
    MinMaxScaler(X) = new(minimum(X, dims=1), maximum(X, dims=1))
end

function (s::MinMaxScaler)(X; normalize=true)
    if normalize
        (X .- s.min) ./ max.(1f-10, s.max .- s.min)
    else
        X .* (s.max .- s.min) .+ s.min
    end
end

"""
    tabular2rnn(X)

Converts tabular data `X` into an RNN sequence format. 
`X` should have format T × K × M, where T is the number of time steps, K is the number 
of features, and M is the number of batches.
"""
tabular2rnn(X::AbstractArray{Float32, 3}) = [X[t, :, :] for t ∈ 1:size(X, 1)]

"""
    rnn2tabular(X)

Converts RNN sequence format `X` into tabular data.
"""
rnn2tabular(X::Vector{Matrix{Float32}}) = permutedims(cat(X..., dims=3), [3, 1, 2])