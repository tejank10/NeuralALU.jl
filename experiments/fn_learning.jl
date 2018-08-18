using NeuralALU, Flux, Distributions

relu6(x) = min(max(x, 0.0), 6.0)

random_baseline = Chain(Dense(2,2), Dense(2,1))
ACTIVATIONS = [tanh, σ, relu6, softsign, selu, elu, relu, nothing]

TEST_FN = ["+" => (a,b)->a+b,
           "-" => (a,b)->a-b,
           "*" => (a,b)->a*b,
           "/" => (a,b)->a/b,
           "²" => (a,b)->a^2,
           "√" => (a,b)->√a]

INP_DIM = 2
HIDDEN_DIM = 2
OUT_DIM = 1
RANGE = [-5, 5]

EPOCHS = 10000

function make_mlp(in_dim, hidden_dim, out_dim, non_lin)
    # hidden_dim is assumed to be an array of dimensions; empty if no hidden layer
    layer_dim = [in_dim, hidden_dim..., out_dim]
    if non_lin == nothing
        non_lin = identity
    end

    layers = []
    for i = 1:length(layer_dim) - 1
        push!(layers, Dense(layer_dim[i], layer_dim[i+1], non_lin))
    end
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
    shfl_idx = shuffle(idx)

    X_train, Y_train = X[:, shfl_idx[1:train_size]], Y[:, shfl_idx[1:train_size]]
    X_test, Y_test = X[:, shfl_idx[train_size:]], Y[:, shfl_idx[train_size:]]

    X_train, Y_train, X_test, Y_test
end

nets = []
optimizers = []

for act_fn in ACTIVATIONS
    push!(nets, make_mlp(INP_DIM, [HIDDEN_DIM], OUT_DIM, act_fn))
    push!(optimizers, SGD(params(nets[end])))
end

push!(nets, Chain(NAC(2,2), NAC(2,1)))
push!(optimizers, SGD(params(nets[end])))

push!(nets, Chain(NALU(2,2), NALU(2,1)))
push!(optimizers, SGD(params(nets[end])))

loss(x, y, net) = Flux.mse(net(x), y)

for test_fn in TEST_FN
    X_train, Y_train, X_test, Y_test = generate_data(100, RANGE, fn, 500, 50, 5)

    for (net, opt) in zip(nets, optimizers)
        @epochs EPOCHS Flux.train!(loss, [(X_train, Y_train, net)], opt)
        l = loss(X_test, Y_test, net)
        print(l, " ")
    end
    println()
end
