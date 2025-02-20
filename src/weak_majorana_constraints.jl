function matrix_from_scalar_product(γ_basis, matrix_basis, lhs_projector, rhs_projector)
    return [hilbert_scalar_product(basis_element, lhs_projector'*γ*rhs_projector) for basis_element in matrix_basis, γ in γ_basis]
end

pauli_basis() = ([1 0; 0 1], [0 1; 1 0], [0 -1im; 1im 0], [1 0; 0 -1]) # should I do like this, change Eij basis, or have different scalar products?

off_diagonal_basis(exc_dims) = (sqrt(exc_dims)*reshape(col, 2, exc_dims) for col in eachcol(I(2*exc_dims))) # better way to create a basis of E_ij?

function groundstate_block(γ_basis::AbstractMajoranaBasis, gs_projector, gs_σ_comps)
    constraint_inds = _get_constraint_inds(gs_σ_comps)
    return matrix_from_scalar_product(γ_basis, pauli_basis()[constraint_inds], gs_projector, gs_projector)
end

function off_diagonal_block(γ_basis, gs_projector, exc_projector)
    exc_dims = size(exc_projector, 2)
    basis = off_diagonal_basis(exc_dims)
    return matrix_from_scalar_product(γ_basis, basis, gs_projector, exc_projector)
end

function construct_complex_matrices(γ::AbstractMajoranaBasis, gs_projector, exc_projector, gs_σ_comps)
    BP = groundstate_block(γ, gs_projector, gs_σ_comps)
    BPQ = off_diagonal_block(γ, gs_projector, exc_projector)
    return BP, BPQ
end

function weak_majorana_constraint_matrix(γ::AbstractMajoranaBasis, gs_projector, exc_projector, gs_σ_comps)
    BP, BPQ = construct_complex_matrices(γ, gs_projector, exc_projector, gs_σ_comps)
    B = [BP; BPQ]
    return [real.(B); imag.(B)]
end

function right_hand_side(gs_σ_comps, exc_projector)
    constraint_inds = _get_constraint_inds(gs_σ_comps)
    N = length(constraint_inds)
    complex_matrix_rows = N + 2 * size(exc_projector, 2) # x ground state equations and 2* #exc states eqs in exc states
    rhs = zeros(Float64, 2 * complex_matrix_rows) # twice the size after splitting into real and complex
    rhs[1:N] .= gs_σ_comps[constraint_inds]
    return rhs
end

_get_constraint_inds(gs_σ_comps) = findall(!isnothing, gs_σ_comps)

@testitem "Weak Majorana constraints" begin
    gs_σ_comps = [1, 1, nothing, 1]
    const_inds = Majoranas._get_constraint_inds(gs_σ_comps)
    @test const_inds == [1, 2, 4]
    tot_dims = 10
    exc_dims = tot_dims - 2
    exc_proj = ones(tot_dims, exc_dims)
    rhs = Majoranas.right_hand_side(gs_σ_comps, exc_proj)
    @test length(rhs) == 2*length(const_inds) + 2*2*exc_dims
    @test rhs[1:length(const_inds)] == gs_σ_comps[const_inds]
end
    
