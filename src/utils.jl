function parity_eigvals(H::QuantumDots.BlockDiagonal)
    eig = QuantumDots.diagonalize(H) # returns a DiagonalizedHamiltonian
    sectors = QuantumDots.blocks(eig; full=false) # used to be "fullsectors"
    oddvals = sectors[1].values
    evenvals = sectors[2].values
    return oddvals, evenvals
end

function coeffs_to_dict(γ::AbstractMajoranaBasis, coeffs::AbstractVector)
    return OrderedDict(zip(keys(γ), coeffs))
end

function single_particle_majoranas_matrix_els(γbasis::SingleParticleMajoranaBasis, oddvec, evenvec)
    return map(γ->ComplexF64(evenvec'*γ*oddvec), γbasis)
end

function single_particle_majoranas(γbasis::SingleParticleMajoranaBasis, matrix_els) # works if ham is real
    γx = mapreduce((z,γ)->real(z) * γ, +, matrix_els, γbasis)
    γy = mapreduce((z,γ)->imag(z) * γ, +, matrix_els, γbasis)
    return γx, γy
end

function single_particle_majoranas(γbasis::SingleParticleMajoranaBasis, oddvec, evenvec)
    coeffs = single_particle_majoranas_matrix_els(γbasis, oddvec, evenvec)
    return single_particle_majoranas(γbasis, coeffs)
end

function single_particle_majoranas(basis::FermionBasis, oddvec, evenvec)
    return single_particle_majoranas(SingleParticleMajoranaBasis(basis), oddvec, evenvec)
end

function strong_majoranas(oddvecs, evenvecs, signs)
    γx = mapreduce((odd, even, s) -> (-1)^s * (odd*even' + even*odd'), +, eachcol(oddvecs), eachcol(evenvecs), signs[1])
    γy = mapreduce((odd, even, s) -> (-1)^s * 1im * (odd*even' - even*odd'), +, eachcol(oddvecs), eachcol(evenvecs), signs[2])
    return γx, γy
end

function coeffs_to_matrix(γbasis::AbstractMajoranaBasis, coeffs)
    return mapreduce((coeff, γ)->coeff*γ, +, coeffs, γbasis)
end

function majorana_coefficients(γbasis::AbstractMajoranaBasis, mat::AbstractMatrix)
    map(γ->scalar_product(γ, mat, _get_basis_norm(γbasis)), γbasis)
end

scalar_product(A, B, ::HilbertNorm) = hilbert_scalar_product(A, B)
scalar_product(A, B, ::FrobeniusNorm) = tr(A'*B)
hilbert_scalar_product(A, B) = tr(A'*B)/size(B,2)

function matrix_to_dict(γbasis::AbstractMajoranaBasis, mat::AbstractMatrix)
    return coeffs_to_dict(γbasis, majorana_coefficients(γbasis, mat))
end

@testitem "Majorana utils" begin
    using QuantumDots, LinearAlgebra, OrderedCollections
    import QuantumDots: kitaev_hamiltonian
    import Majoranas: single_particle_majoranas, single_particle_majoranas_matrix_els, Hamiltonian, HilbertNorm, FrobeniusNorm
    c = FermionBasis(1:2; qn=QuantumDots.parity)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(c; μ=0.0, t=1.0, Δ=1.0)), c)
    eig = diagonalize(pmmham)
    fullsectors = QuantumDots.blocks(eig; full=true)
    oddvecs = fullsectors[1].vectors
    evenvecs = fullsectors[2].vectors
    γ = SingleParticleMajoranaBasis(c)
    sp_matrix_els = single_particle_majoranas_matrix_els(γ, oddvecs[:, 1], evenvecs[:,1])
    γx, γy = single_particle_majoranas(γ, sp_matrix_els)
    @test all(single_particle_majoranas(c, oddvecs[:,1], evenvecs[:,1]) .≈ (γx, γy))
    @test all(single_particle_majoranas(γ, oddvecs[:,1], evenvecs[:, 1]) .≈ (γx, γy))
    γ_mb = ManyBodyMajoranaBasis(γ, 1)
    prob = WeakMajoranaProblem(γ_mb, oddvecs, evenvecs, nothing)
    sols = solve(prob, Majoranas.WM_BACKSLASH())
    @test Majoranas.coeffs_to_dict(γ_mb, sols[1]) isa OrderedDict
    @test all(map(coeffs -> Majoranas.coeffs_to_matrix(γ_mb, coeffs), sols) .≈ (γx, γy))
    # strong majoranas
    signs_x = [0, 1, 1, 0] # no idea why these signs work
    signs_y = ones(size(oddvecs, 2))
    γx_s, γy_s = Majoranas.strong_majoranas(oddvecs, evenvecs, (signs_x, signs_y))
    #=@test all((γx_s, γy_s) .≈ (γx, γy))=#
    
    c = FermionBasis(1:3; qn=QuantumDots.parity)
    μ, t, Δ, V = rand(4)
    pmmham = blockdiagonal(kitaev_hamiltonian(c; μ, t, Δ, V), c)
    for basis_norm in (HilbertNorm(), FrobeniusNorm())
        local γ = SingleParticleMajoranaBasis(c, (:a, :b), basis_norm)
        γmb = ManyBodyMajoranaBasis(γ)
        Ham = Hamiltonian(pmmham; basis=c)
        o, e = Majoranas.ground_states(Ham)
        γ = o * e' + hc
        dict = Majoranas.matrix_to_dict(γmb, γ)
        @test sum(dict[key] * γmb[key] for key in keys(γmb)) ≈ γ
    end
end
