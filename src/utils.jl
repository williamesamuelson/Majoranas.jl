function scalar_product(A,B)
    return tr(A'*B)/size(B,2)
end

function coeffs_to_dict(γ, coeffs)
    return QuantumDots.dictionary(zip(keys(γ), coeffs))
end

function single_particle_majoranas_coeffs(γbasis::SingleParticleMajoranaBasis, oddvec, evenvec)
    return QuantumDots.dictionary(zip(labels(γbasis), map(γ->ComplexF64(evenvec'*γ*oddvec), γbasis)))
end

function single_particle_majoranas_from_coeffs(γbasis::SingleParticleMajoranaBasis, coeffs) # works if ham is real
    γx = mapreduce((z,γ)->real(z) * γ, +, coeffs, γbasis)
    γy = mapreduce((z,γ)->imag(z) * γ, +, coeffs, γbasis) 
    return γx, γy
end

function single_particle_majoranas_from_ground_states(γbasis::SingleParticleMajoranaBasis, oddvec, evenvec)
    coeffs = single_particle_majoranas_coeffs(γbasis, oddvec, evenvec)
    return single_particle_majoranas_from_coeffs(γbasis, coeffs)
end

function many_body_majorana_from_coeffs(γbasis::ManyBodyMajoranaBasis, coeffs)
    return mapreduce((coeff, γ)->coeff*γ, +, coeffs, γbasis)
end

function majorana_coefficients(γbasis::ManyBodyMajoranaBasis, maj::AbstractMatrix)
    QuantumDots.dictionary(zip(labels(γbasis), map(γ->scalar_product(γ,maj), γbasis)))
end

function majorana_coefficients(fermion_basis::FermionBasis, maj::AbstractMatrix)
    majorana_coefficients(ManyBodyMajoranaBasis(fermion_basis), maj)
end