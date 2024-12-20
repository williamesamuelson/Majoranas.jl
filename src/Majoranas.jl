module Majoranas
using LinearAlgebra
using QuantumDots
using AffineRayleighOptimization
using LinearSolve
using Combinatorics
using TestItems
using SparseArrays
using OrderedCollections

import AffineRayleighOptimization: solve
import QuantumDots: majorana_polarization

include("majorana_basis.jl")
include("hamiltonian.jl")
include("utils.jl")
include("weak_majorana_utils.jl")
include("weak_majorana_constraints.jl")
include("weak_majorana_problem.jl")
include("majorana_polarization.jl")
include("basisarray.jl")
include("majorana_metrics.jl")

export SingleParticleMajoranaBasis, ManyBodyMajoranaBasis, HamiltonianBasis, ProjectedHamiltonianBasis, labels, coeffs_to_dict
export WeakMajoranaProblem
export solve

## "Trick" LSP so that stuff works in scripts files
@static if false
    include("../test/runtests.jl")
end

end
