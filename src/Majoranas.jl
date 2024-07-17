module Majoranas
using LinearAlgebra
using QuantumDots
using AffineRayleighOptimization
using LinearSolve
using Combinatorics
using TestItems
import AffineRayleighOptimization: solve

include("majorana_basis.jl")
include("utils.jl")
include("weak_majorana_utils.jl")
include("weak_majorana_constraints.jl")
include("weak_majorana_problem.jl")

export SingleParticleMajoranaBasis, ManyBodyMajoranaBasis, labels
export WeakMajoranaProblem
export solve

## "Trick" LSP so that stuff works in scripts files
@static if false
    include("../test/runtests.jl")
end

end
