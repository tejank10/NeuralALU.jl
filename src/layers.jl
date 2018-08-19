include("kaiming_init.jl")
using Flux
using Flux:@treelike
using NNlib: @fix, σ_stable

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

function (nac::NAC)(x)
  Ŵ, M̂, b = nac.Ŵ, nac.M̂, nac.b 
  W = tanh.(Ŵ) .* σ_stable.(M̂)  
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

function (nalu::NALU)(x)
  nac, G, b, ϵ = nalu.nac, nalu.G, nalu.b, nalu.ϵ
  a = nac(x)
  log_inp = log.(abs.(x) .+ ϵ)
  m = exp.(nac(log_inp))
  g = σ_stable.(G*x .+ b)
  @fix g .* a .+ (1.0 .- g) .* m
end

Base.show(io::IO, l::NALU) = print(io, "NALU(", size(l.G, 2), ", ", size(l.G, 1), ")")
