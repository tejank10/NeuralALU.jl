include("../src/NeuralALU.jl")
using Main.NeuralALU, Flux, Distributions
using Flux:@epochs
import Flux.params

relu6(x) = min(max(x, 0.0f0), 6.0f0)

random_baseline = Chain(Dense(2,2), Dense(2,1))
ACTIVATIONS = [tanh, σ, relu6, softsign, selu, elu, relu, identity]

TEST_FN = Dict("+" => (a,b)->a+b,
               "-" => (a,b)->a-b,
               "*" => (a,b)->a*b,
               "/" => (a,b)->a/b,
               "²" => (a,b)->a^2,
               "√" => (a,b)->√a
	      )

INP_DIM = 2
HIDDEN_DIM = 2
OUT_DIM = 1
RANGE = [5, 10]

EPOCHS = 100000

function make_mlp(in_dim, hidden_dim, out_dim, non_lin)
    # hidden_dim is assumed to be an array of dimensions; empty if no hidden layer
    layer_dim = [in_dim, hidden_dim..., out_dim]
    if non_lin == nothing
        non_lin = identity
    end

    layers = []
    for i = 1:length(layer_dim) - 2
        push!(layers, Dense(layer_dim[i], layer_dim[i+1], non_lin))
    end
    push!(layers, Dense(layer_dim[end-1], layer_dim[end]))
    Chain(layers...)
end

function generate_data(dim, span, fn, train_size, test_size, sum_size)
    dist = Uniform(span...)
    data = rand(dist, dim)

    X = zeros(INP_DIM, train_size + test_size)
    Y = zeros(OUT_DIM, train_size + test_size)

    for i = 1:train_size+test_size
        idx_a = rand(1:dim, sum_size)
        idx_b = rand(setdiff(collect(1:dim), idx_a), sum_size)
        a, b = sum(data[idx_a]), sum(data[idx_b])

        X[:, i] = [a, b]
        Y[:, i] = [fn(a, b)]
    end

    idx = collect(1:train_size+test_size)
    shfl_idx = idx#shuffle(idx)

    X_train, Y_train = X[:, 1:train_size], Y[:,1:train_size]
    X_test, Y_test = X[:, train_size+1:end], Y[:, train_size+1:end]

    X_train, Y_train, X_test, Y_test
end

nets = []

for act_fn in ACTIVATIONS
    push!(nets, make_mlp(INP_DIM, [HIDDEN_DIM], OUT_DIM, act_fn))
end

push!(nets, Chain(NAC(2,2), NAC(2,1)))

push!(nets, Chain(NALU(2,2), NALU(2,1)))

loss(x, y, net) = Flux.mse(net(x), y)

for test_fn in keys(TEST_FN)
    X_train, Y_train, X_test, Y_test = generate_data(100, RANGE, TEST_FN[test_fn], 500, 50, 5)
    for (net,act_fn) in zip(nets, vcat(ACTIVATIONS, [NAC, NALU]))
	opt = RMSProp(params(net))
	for _ = 1:EPOCHS
	    Flux.train!(loss, [(X_train, Y_train, net)], opt)
	end
        l_train = loss(X_train, Y_train, net)
	l_test  = loss(X_test, Y_test, net)
        println(" Test Function: ", test_fn, " Activation: ", act_fn, " Loss: ", l_test.data) 
    end
    println()
end
