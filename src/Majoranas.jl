module Majoranas
using LinearAlgebra
using FermionicHilbertSpaces
using AffineRayleighOptimization
using LinearSolve
using Combinatorics
using TestItems
using SparseArrays
using OrderedCollections
using BlockDiagonals

import AffineRayleighOptimization: solve
import FermionicHilbertSpaces: AbstractHilbertSpace, MajoranaHilbertSpace

include("majorana_basis.jl")
include("hamiltonian.jl")
include("blockdiagonal.jl")
include("kitaev.jl")
include("utils.jl")
include("weak_majorana_utils.jl")
include("weak_majorana_constraints.jl")
include("weak_majorana_problem.jl")
include("majorana_polarization.jl")
include("majorana_metrics.jl")
include("basisarray.jl")

export blockdiagonal
export SingleParticleMajoranaBasis, ManyBodyMajoranaBasis, HamiltonianBasis, ProjectedHamiltonianBasis, labels, coeffs_to_dict
export WeakMajoranaProblem
export solve

## "Trick" LSP so that stuff works in scripts files
@static if false
    include("../test/runtests.jl")
end

end
