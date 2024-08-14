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

function WeakMajoranaProblem(γ::ManyBodyMajoranaBasis, oddvecs, evenvecs, minimizer=RayleighQuotient(many_body_content_matrix(γ)))
    P, Q = projection_ops(oddvecs, evenvecs)
    constraints = weak_majorana_constraint_matrix(γ, P, Q)
    bs = right_hand_sides(Q)
    ind = norm.(eachrow(constraints)) .> 1e-16 # remove redundant constraints
    constraints = constraints[ind, :]
    bs = [b[ind] for b in bs]
    WeakMajoranaProblem(minimizer, γ, constraints, bs, (P, Q))
end

function test_weak_majorana_solution(prob::WeakMajoranaProblem, sols)
    σs = Majoranas.pauli_basis()
    P, Q = prob.projs
    γs = map(sol -> Majoranas.coeffs_to_matrix(prob.γ, sol), sols)
    gs_test = map((γ, σ) -> norm(P'γ * P - σ), γs, σs)
    exc_test = map(γ -> norm(P' * γ * Q), γs)
    return gs_test, exc_test
end

function solve(prob::WeakMajoranaProblem{<:RayleighQuotient}, alg=AffineRayleighOptimization.RQ_EIG())
    [solve(ConstrainedRayleighQuotientProblem(prob.minimizer, prob.constraints, b), alg) for b in prob.bs]
end

function solve(prob::WeakMajoranaProblem{<:QuadraticForm}, alg=KrylovJL_MINRES())
    [solve(ConstrainedQuadraticFormProblem(prob.minimizer, prob.constraints, b), alg) for b in prob.bs]
end

struct WM_BACKSLASH end

function solve(prob::WeakMajoranaProblem, alg::WM_BACKSLASH)
    [prob.constraints\b for b in prob.bs]
end
