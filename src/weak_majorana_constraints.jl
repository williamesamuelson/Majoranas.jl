function matrix_from_scalar_product(γ_basis, matrix_basis, lhs_projector, rhs_projector)
    return [hilbert_schmidt_scalar_product(basis_element, lhs_projector'*γ*rhs_projector) for basis_element in matrix_basis, γ in γ_basis]
end

pauli_basis() = ([1 0; 0 1], [0 1; 1 0], [0 -1im; 1im 0], [1 0; 0 -1]) # should I do like this, change Eij basis, or have different scalar products?

off_diagonal_basis(exc_dims) = (sqrt(exc_dims)*reshape(col, 2, exc_dims) for col in eachcol(I(2*exc_dims))) # better way to create a basis of E_ij?

function groundstate_block(γ_basis::AbstractMajoranaBasis, gs_projector)
    basis = pauli_basis()
    σx, σy = basis[2], basis[3]
    return matrix_from_scalar_product(γ_basis, (σx, σy), gs_projector, gs_projector)
end

function groundstate_block(γ_basis::HamiltonianBasis, gs_projector)
    basis = pauli_basis()
    σ0, σz = basis[1], basis[4]
    return matrix_from_scalar_product(γ_basis, (σ0, σz), gs_projector, gs_projector)
end

function off_diagonal_block(γ_basis, gs_projector, exc_projector)
    exc_dims = size(exc_projector, 2)
    basis = off_diagonal_basis(exc_dims)
    return matrix_from_scalar_product(γ_basis, basis, gs_projector, exc_projector)
end

function construct_complex_matrices(γ::AbstractMajoranaBasis, gs_projector, exc_projector)
    BP = groundstate_block(γ, gs_projector)
    BPQ = off_diagonal_block(γ, gs_projector, exc_projector)
    return BP, BPQ
end

function weak_majorana_constraint_matrix(γ::AbstractMajoranaBasis, gs_projector, exc_projector)
    BP, BPQ = construct_complex_matrices(γ, gs_projector, exc_projector)
    B = [BP; BPQ]
    return [real.(B); imag.(B)]
end

function right_hand_sides(exc_projector)
    complex_matrix_rows = 2 + 2 * size(exc_projector, 2)
    rhsx = zeros(Float64, 2 * complex_matrix_rows)
    rhsy = zeros(Float64, 2 * complex_matrix_rows)
    rhsx[1] = 1
    rhsy[2] = 1
    return rhsx, rhsy
end

function ham_right_hand_side(σ0_comp, σz_comp, exc_projector)
    complex_matrix_rows = 2 + 2 * size(exc_projector, 2)
    rhs = zeros(Float64, 2 * complex_matrix_rows)
    rhs[1] = σ0_comp
    rhs[2] = σz_comp
    return rhs
end
