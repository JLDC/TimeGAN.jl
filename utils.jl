using Flux

"""
    Seq2One

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