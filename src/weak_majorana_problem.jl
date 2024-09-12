import AffineRayleighOptimization: RQ_EIG, RQ_GENEIG
import LinearSolve: KrylovJL_MINRES

"""
    WeakMajoranaProblem(minimizer, γ, constraints, bs, projs)

A weak Majorana problem with
    minimizer: what to minimize (e.g., RayleighQuotient)
    γ: many body Majorana basis
    constraints: lhs matrix of contraint equation
    bs: rhs vectors of constraint equation
    projs: projection operators onto ground states/excited states
"""
struct WeakMajoranaProblem{M,G,C,B,PQ}
    minimizer::M
    γ::G
    constraints::C
    bs::B
    projs::PQ
end

_def_pauli_comps(::ManyBodyMajoranaBasis) = ([nothing, 1., 0., nothing], [nothing, 0., 1., nothing])
_def_minimizer(γ::ManyBodyMajoranaBasis) = RayleighQuotient(many_body_content_matrix(γ))
function WeakMajoranaProblem(γ::ManyBodyMajoranaBasis, oddvecs, evenvecs, minimizer=_def_minimizer(γ), gs_pauli_comps_vec=_def_pauli_comps(γ))
    P, Q = projection_ops(oddvecs, evenvecs)
    constraints = weak_majorana_constraint_matrix(γ, P, Q, gs_pauli_comps_vec[1]) # LHS matrix same for both Majoranas
    bs = [right_hand_side(σs, Q) for σs in gs_pauli_comps_vec]
    constraints, bs = _get_reduced_eqs(constraints, bs)
    WeakMajoranaProblem(minimizer, γ, constraints, bs, (P, Q))
end

# In the Hamiltonian problem, we might only have one parity
# Also, only one matrix to find
function WeakMajoranaProblem(γ::HamiltonianBasis, states, minimizer, gs_pauli_comps)
    P, Q = projection_ops(states)
    constraints = weak_majorana_constraint_matrix(γ, P, Q, gs_pauli_comps)
    bs = [right_hand_side(gs_pauli_comps, Q)]
    constraints, bs = _get_reduced_eqs(constraints, bs)
    WeakMajoranaProblem(minimizer, γ, constraints, bs, (P, Q))
end

function _zero_row_equations(constraints, bs, ϵ=1e-16)
    ind = norm.(eachrow(constraints)) .> ϵ # remove redundant constraints
    inc_eqs = !all([all(abs.(b[.!ind]) .< ϵ) for b in bs])
    inc_eqs ? error("equations inconsistent") : ind
end

@testitem "Zero row equations" begin
    constraints = vcat([1, 1, 1]', zeros(3)', rand(3)')
    b1 = [0, 1, 1] # not ok
    b2 = [1, 0, 0] # ok
    @test_throws ErrorException("equations inconsistent") Majoranas._zero_row_equations(constraints, [b1, b2])
    @test_throws ErrorException("equations inconsistent") Majoranas._zero_row_equations(constraints, [b1])
    @test Majoranas._zero_row_equations(constraints, [b2]) == [1, 0, 1]
end

function _get_reduced_eqs(constraints, bs)
    ind = _zero_row_equations(constraints, bs)
    constraints = constraints[ind, :]
    bs = [b[ind] for b in bs]
    return constraints, bs
end

@testitem "Hamiltonian weak Majorana problem" begin
    using QuantumDots, LinearAlgebra, AffineRayleighOptimization
    import QuantumDots: kitaev_hamiltonian
    c = FermionBasis(1:2; qn=QuantumDots.parity)
    pmmham = Hermitian(Matrix(kitaev_hamiltonian(c; μ=0.0, t=2.0, Δ=1.0)))
    eig = eigen(pmmham)
    δE = (eig.values[2] - eig.values[1]) / 2
    H = HamiltonianBasis(SingleParticleMajoranaBasis(c, (:a, :b)))
    prob = WeakMajoranaProblem(H, eig.vectors, nothing, [0, 0, 0, δE])
    sol = solve(prob, Majoranas.WM_BACKSLASH_SPARSE())
    corr = Majoranas.coeffs_to_matrix(H, sol)
    corrected_ham = blockdiagonal(pmmham + corr, c)
    oddvals, evenvals = Majoranas.parity_eigvals(corrected_ham)
    @test oddvals[1] ≈ evenvals[1]
end

function test_weak_majorana_solution(prob::WeakMajoranaProblem, sols)
    basis = Majoranas.pauli_basis()
    P, Q = prob.projs
    γs = map(sol -> Majoranas.coeffs_to_matrix(prob.γ, sol), sols)
    gs_test = map((γ, σ) -> norm(P'γ * P - σ), γs, basis[2:3])
    exc_test = map(γ -> norm(P' * γ * Q), γs)
    return gs_test, exc_test
end

function solve(prob::WeakMajoranaProblem{<:RayleighQuotient}, alg=AffineRayleighOptimization.RQ_EIG())
    sols = [solve(ConstrainedRayleighQuotientProblem(prob.minimizer, prob.constraints, b), alg) for b in prob.bs]
    _return_sol(prob, sols)
end

function solve(prob::WeakMajoranaProblem{<:QuadraticForm}, alg=KrylovJL_MINRES())
    sols = [solve(ConstrainedQuadraticFormProblem(prob.minimizer, prob.constraints, b), alg) for b in prob.bs]
    _return_sol(prob, sols)
end

struct WM_BACKSLASH end
struct WM_BACKSLASH_SPARSE end

function solve(prob::WeakMajoranaProblem{<:Nothing}, alg::WM_BACKSLASH)
    sols = [prob.constraints\b for b in prob.bs]
    _return_sol(prob, sols)
end

function solve(prob::WeakMajoranaProblem{<:Nothing}, alg::WM_BACKSLASH_SPARSE)
    sols = [sparse(prob.constraints)\b for b in prob.bs]
    _return_sol(prob, sols)
end

_return_sol(prob::WeakMajoranaProblem{M,<:ManyBodyMajoranaBasis}, sols) where M = sols
_return_sol(prob::WeakMajoranaProblem{M,<:HamiltonianBasis}, sols) where M = only(sols)
