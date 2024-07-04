module Majoranas
using LinearAlgebra
using QuantumDots
using AffineRayleighOptimization
using Combinatorics
using TestItems

include("majorana_basis.jl")
include("utils.jl")
include("weak_majorana_utils.jl")
include("weak_majorana_constraints.jl")
include("weak_majorana_problem.jl")

export SingleParticleMajoranaBasis, ManyBodyMajoranaBasis

end
