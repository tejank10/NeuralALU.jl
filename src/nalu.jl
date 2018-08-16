include("kaiming_init.jl")
using Flux
using Flux:@treelike
using NNlib: @fix

################################### NAC ########################################

struct NAC{F,S,T}
  Ŵ::S
  M̂::F
  b::T
end

NAC(in::Integer, out::Integer;
               initŴ = kaiming_uniform, initM̂ = kaiming_uniform, initb = zeros) =
    NAC(param(initŴ(out, in; a=√5.0)), param(initM̂(out, in; a=√5.0)), param(initb(out)))

@treelike NAC

function (a::NAC)(x)
  Ŵ, M̂, b = a.Ŵ, a.M̂, a.b
  W = tanh.(Ŵ) .* σ.(M̂)
  @fix W*x .+ b
end

Base.show(io::IO, l::NAC) = print(io, "NAC(", size(l.Ŵ, 2), ", ", size(l.Ŵ, 1), ")")

################################## NALU ########################################

struct NALU{N,S,T,E}
  nac::N
  G::S
  b::T
  ϵ::E
end

NALU(in::Integer, out::Integer;
               initĜ = kaiming_uniform, initb = zeros) =
  NALU(NAC(in, out), param(initĜ(out, in; a=√5.0)), param(initb(out)), 1e-10)

@treelike NALU

function (a::NALU)(x)
  nac, G, b, ϵ = a.nac, a.G, a.b, a.ϵ
  a = nac(x)
  log_inp = nac(abs.(x) .+ ϵ)
  m = exp.(log_inp)
  g = σ.(G*x .+ b)
  @fix g .* a + (1 .- g) .* m
end

Base.show(io::IO, l::NALU) = print(io, "NALU(", size(l.G, 2), ", ", size(l.G, 1), ")")
