function hilbert_schmidt_scalar_product(A,B)
    return tr(A'*B)/size(B,2)
end

function coeffs_to_dict(γ::AbstractMajoranaBasis, coeffs::AbstractVector)
    return QuantumDots.dictionary(zip(keys(γ), coeffs))
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

function many_body_majorana(γbasis::AbstractMajoranaBasis, coeffs)
    return mapreduce((coeff, γ)->coeff*γ, +, coeffs, γbasis)
end

function majorana_coefficients(γbasis::AbstractMajoranaBasis, maj::AbstractMatrix)
    map(γ->hilbert_schmidt_scalar_product(γ, maj), γbasis)
end

function majorana_coefficients(fermion_basis::FermionBasis, maj::AbstractMatrix)
    majorana_coefficients(ManyBodyMajoranaBasis(fermion_basis), maj)
end

@testitem "Majorana utils" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    c = FermionBasis(1:2; qn=QuantumDots.parity)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(c; μ=0.0, t=1.0, Δ=1.0)), c)
    eig = diagonalize(pmmham)
    fullsectors = QuantumDots.blocks(eig; full=true)
    oddvecs = fullsectors[1].vectors
    evenvecs = fullsectors[2].vectors
    γ = SingleParticleMajoranaBasis(c)
    sp_matrix_els = Majoranas.single_particle_majoranas_matrix_els(γ, oddvecs[:, 1], evenvecs[:,1])
    γx, γy = Majoranas.single_particle_majoranas(γ, sp_matrix_els)
    @test all(Majoranas.single_particle_majoranas(γ, oddvecs[:,1], evenvecs[:, 1]) .≈ (γx, γy))
    γ_mb = ManyBodyMajoranaBasis(γ, 1)
    prob = WeakMajoranaProblem(γ_mb, oddvecs, evenvecs, nothing)
    sols = solve(prob)
    @test Majoranas.coeffs_to_dict(γ_mb, sols[1]) isa QuantumDots.Dictionary
    @test all(map(coeffs -> Majoranas.many_body_majorana(γ_mb, coeffs), sols) .≈ (γx, γy))
end
