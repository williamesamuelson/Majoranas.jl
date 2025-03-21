struct Hamiltonian{H}
    ham::H
    dict::Dict{Symbol,Any}
    function Hamiltonian(ham, _dict=Dict{Symbol,Any}(); kwargs...)
        dict = merge(_dict, kwargs)
        return new{typeof(ham)}(ham, dict)
    end
end

get_basis(H::Hamiltonian) = H.dict[:basis]
get_ham(H::Hamiltonian) = H.ham
get_diag_ham(H::Hamiltonian) = isdiagonalized(H) ? H[:DH] : (diagonalize!(H); return H[:DH])
Base.haskey(H::Hamiltonian, key) = haskey(H.dict, key)
Base.getindex(H::Hamiltonian, key...) = getindex(H.dict, key...)
Base.setindex!(H::Hamiltonian, value, key...) = setindex!(H.dict, value, key...)

diagonalize!(H::Hamiltonian) = (H[:DH] = diagonalize(get_ham(H))) # maybe find a better key for the diagonalized hamiltonian
isdiagonalized(H::Hamiltonian) = haskey(H, :DH)

@testitem "Hamiltonian struct" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    import Majoranas: Hamiltonian, get_basis, get_ham, diagonalize!, isdiagonalized
    cpmm = FermionBasis(1:2; qn=ParityConservation())
    cpmm_noparity = FermionBasis(1:2)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(cpmm; μ=1.0, t=exp(1im * pi / 3), Δ=2.0, V=2.0)), cpmm)
    H = Hamiltonian(pmmham; basis=cpmm)
    @test get_basis(H) == cpmm
    @test get_ham(H) == pmmham == H.ham
    @test !haskey(H, :X)
    @test !isdiagonalized(H)
    diagonalize!(H)
    @test isdiagonalized(H)
end

function ground_states(H::Hamiltonian)
    haskey(H, :groundstates) && return H[:groundstates] # should they be stored?
    isdiagonalized(H) || diagonalize!(H)
    sectors = QuantumDots.blocks(get_diag_ham(H); full=true)
    oddvec = first(eachcol(sectors[1].vectors))
    evenvec = first(eachcol(sectors[2].vectors))
    H[:groundstates] = (oddvec, evenvec)
    return oddvec, evenvec
end

function energy_info(H::Hamiltonian) # user won't call this?
    isdiagonalized(H) || diagonalize!(H)
    sectors = QuantumDots.blocks(get_diag_ham(H))
    oddvals = sectors[1].values
    evenvals = sectors[2].values
    deg = first(oddvals) - first(evenvals)
    energies = (oddvals, evenvals)
    einfo = (; deg, energies, deg_ratio=deg_ratio(oddvals, evenvals), excgap=excgap(oddvals, evenvals))
    merge!(H.dict, Dict(pairs(einfo)))
    return einfo
end

function degeneracy(H::Hamiltonian)
    haskey(H, :deg) && return H[:deg]
    return energy_info(H).deg
end

function deg_ratio(oddvals, evenvals)
    δE = first(oddvals) - first(evenvals)
    length(oddvals) == 1 && return NaN
    Δ = min(oddvals[2], evenvals[2]) - min(first(oddvals), first(evenvals))
    return δE / Δ
end
function deg_ratio(H::Hamiltonian)
    haskey(H, :deg_ratio) && return H[:deg_ratio]
    return energy_info(H).deg_ratio
end

function excgap(oddvals, evenvals)
    length(oddvals) == 1 && return NaN
    return min(oddvals[2] - oddvals[1], evenvals[2] - evenvals[1])
end

function excgap(H::Hamiltonian)
    haskey(H, :excgap) && return H[:excgap]
    return energy_info(H).excgap
end

function spectrum_weakness(H::Hamiltonian)
    haskey(H, :energies) && return spectrum_weakness(H[:energies])
    return spectrum_weakness(energy_info(H).energies)
end
spectrum_weakness(energies) = norm(energies[1] .- energies[2])

function majorana_coefficients(H::Hamiltonian)
    haskey(H, :maj_coeffs) && return H[:maj_coeffs]
    isdiagonalized(H) || diagonalize!(H)
    basis = get_basis(H)
    oddvec, evenvec = ground_states(H)
    maj_coeffs = QuantumDots.majorana_coefficients(oddvec, evenvec, basis)
    H[:maj_coeffs] = maj_coeffs
    return maj_coeffs
end

function MP(R, H::Hamiltonian)
    majcoeffs = majorana_coefficients(H)
    return majorana_polarization(majcoeffs..., R).mp
end

function LD(red_basis::FermionBasis, H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    oddvec, evenvec = ground_states(H)
    δρ = oddvec * oddvec' - evenvec * evenvec'
    return norm(partial_trace(δρ, red_basis, get_basis(H)))
end
LD(R, H::Hamiltonian) = LD(FermionBasis(R), H)

function groundstate_majoranas(oddvec, evenvec)
    γ = oddvec * evenvec' + evenvec * oddvec'
    γtilde = 1im * (oddvec * evenvec' - evenvec * oddvec')
    return γ, γtilde
end

function LF_info(red_basis::FermionBasis, H::Hamiltonian, θ=0.0)
    isdiagonalized(H) || diagonalize!(H)
    oddvec, evenvec = ground_states(H)
    γ, γtilde = groundstate_majoranas(oddvec, evenvec)
    basis = get_basis(H)
    α, β = alpha_beta_matrices(oddvec, evenvec, red_basis, basis)
    θmin = - 1/2 * angle(tr(α * β'))
    θmax = θmin + π / 2
    γmin, γmax, γθ = map(θ -> cos(θ) * γ + sin(θ) * γtilde, (θmin, θmax, θ))
    α2_plus_β2 = norm(α)^2 + norm(β)^2
    LF, LFmax = map(sgn -> sqrt(α2_plus_β2 + sgn * 2 * abs(tr(α * β'))), (-1, 1))
    LFθ = norm(partial_trace(γθ, red_basis, basis))
    return (;LF, LFmax, LFθ, γmin, γmax, γθ, θmin)
end
LF_info(R, H::Hamiltonian) = LF_info(FermionBasis(R; qn=ParityConservation()), H)

LF(red_basis::FermionBasis, H::Hamiltonian) = LF_info(red_basis, H).LF
LF(R, H::Hamiltonian) = LF(FermionBasis(R; qn=ParityConservation()), H)

function alpha_beta_matrices(oddvec, evenvec, red_basis, basis)
    A_full = oddvec * evenvec'
    A_red = partial_trace(A_full, red_basis, basis)
    block_inds_red = QuantumDots.blockinds(QuantumDots.symmetry(red_basis))
    α = sqrt(2) * A_red[block_inds_red[1], block_inds_red[2]]
    β = -1 * sqrt(2) * A_red[block_inds_red[2], block_inds_red[1]]'
    return α, β
end

function creduced(red_basis::FermionBasis, H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    oddvec, evenvec = ground_states(H)
    γ, _ = groundstate_majoranas(oddvec, evenvec)
    basis = get_basis(H)
    P = QuantumDots.parityoperator(basis)
    c = (I + P) * γ
    return norm(partial_trace(c, red_basis, basis))
end

# perhaps useful to compare with GS LF, only for real ham
function single_particle_LF(R, H::Hamiltonian)
    eltype(get_ham(H)) <: Real || @warn "Hamiltonian must be real for single_particle_LF"
    isdiagonalized(H) || diagonalize!(H)
    γs = single_particle_majoranas(get_basis(H), ground_states(H)...)
    basis = get_basis(H)
    basis_red = FermionBasis(R; qn=ParityConservation())
    red_γs = map(γ -> partial_trace(γ, basis_red, basis), γs)
    return minimum(norm.(red_γs))
end

@testitem "Majorana metrics" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    import Majoranas: Hamiltonian, ground_states, energy_info, degeneracy, deg_ratio, excgap, spectrum_weakness
    import Majoranas: MP, LD, LF, LF_info, single_particle_LF
    cpmm = FermionBasis(1:2; qn=ParityConservation())
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(cpmm; μ=1.0, t=exp(1im * pi / 3), Δ=2.0, V=2.0)), cpmm)
    H = Hamiltonian(pmmham; basis=cpmm)
    regions = [[1], [2], [1, 2]]
    bases = [FermionBasis(R; qn=ParityConservation()) for R in regions]
    @test all(map(R -> MP(R, H), regions) .≈ [1.0, 1.0, 0.0])
    @test haskey(H, :maj_coeffs)
    @test all(map(bR -> LD(bR, H), bases) .≈ [0.0, 0.0, sqrt(2)])
    @test all(map(bR -> LF(bR, H), bases) .≈ [0.0, 0.0, sqrt(2)])
    H = Hamiltonian(pmmham; basis=cpmm)
    @test MP(regions[1], H) ≈ 1.0 # diagonalize! is called in MP
    e_info = energy_info(H)
    H = Hamiltonian(pmmham; basis=cpmm)
    @test e_info.deg_ratio ≈ deg_ratio(H) ≈ H[:deg_ratio] ≈ 0 # call diagonalize! again
    @test degeneracy(H) ≈ 0
    H = Hamiltonian(pmmham; basis=cpmm)
    @test e_info.excgap ≈ excgap(H) ≈ H[:excgap] ≈ 2.0
    H = Hamiltonian(pmmham; basis=cpmm)
    @test spectrum_weakness(e_info.energies) ≈ spectrum_weakness(H) ≈ 2.0
    pmmham_noint = blockdiagonal(Hermitian(kitaev_hamiltonian(cpmm; μ=0.0, t=1.0, Δ=1.0)), cpmm)
    H_noint = Hamiltonian(pmmham_noint; basis=cpmm)
    @test spectrum_weakness(H_noint) ≈ 0.0
    @test_warn "Hamiltonian must be real for single_particle_LF" single_particle_LF([1], H)
    @test single_particle_LF([1], H_noint) < 1e-10
end

@testitem "LF, γmin and alpha_beta_matrices" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    import Majoranas: Hamiltonian, LD, LF, LF_info, alpha_beta_matrices, ground_states
    # test LF and γmin for a 2-site complex interacting Kitaev chain
    qn = ParityConservation()
    basis = FermionBasis(1:2; qn)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(basis; μ=1.0, t=exp(1im * pi / 3), Δ=2.0, V=2.0)), basis)
    H = Hamiltonian(pmmham; basis)
    red_basis = FermionBasis([1]; qn)
    α, β = alpha_beta_matrices(ground_states(H)..., red_basis, basis)
    LFinfo = LF_info(red_basis, H)
    γmin, γmax, θmin = LFinfo.γmin, LFinfo.γmax, LFinfo.θmin
    θmax = θmin + π / 2
    @test γmin' ≈ γmin
    @test γmax' ≈ γmax
    @test norm(γmin) ≈ norm(γmax) ≈ sqrt(2)
    reduced_γmin = partial_trace(γmin, red_basis, basis)
    @test norm(reduced_γmin) < 1e-10 # small, but norm(reduced_γmin) !≈ LFinfo.LF
    reduced_γmax = partial_trace(γmax, red_basis, basis)
    @test norm(reduced_γmax) ≈ LFinfo.LFmax
    reconstructed_γmax = 1/sqrt(2) * ([0I exp(1im * θmax) * α - exp(-1im * θmax) * β; 0I 0I] + hc)
    @test reduced_γmax ≈ reconstructed_γmax

    # test α β matrices for a larger region
    basis = FermionBasis(1:3; qn)
    rand_ham = blockdiagonal(Hermitian(rand(2^3, 2^3)), basis)
    H = Hamiltonian(rand_ham; basis)
    red_basis = FermionBasis([1, 2]; qn)
    α, β = alpha_beta_matrices(ground_states(H)..., red_basis, basis)
    θ = 0.0
    LFinfo = LF_info(red_basis, H, θ)
    γmin, γmax, γθ, θmin = LFinfo.γmin, LFinfo.γmax, LFinfo.γθ, LFinfo.θmin
    @test abs(θmin) < 1e-10 || θmin ≈ - π / 2 # the Hamiltonian is real, so α and β are real
    @test γθ ≈ γmax
    @test LFinfo.LFθ ≈ LFinfo.LFmax
    @test γmin' ≈ γmin
    @test γmax' ≈ γmax
    @test norm(γmin) ≈ norm(γmax) ≈ sqrt(2)
    reduced_γmin = partial_trace(γmin, red_basis, basis)
    reduced_γmax = partial_trace(γmax, red_basis, basis)
    @test norm(reduced_γmax) ≈ LFinfo.LFmax
    @test norm(reduced_γmin) ≈ LFinfo.LF
    reconstructed_γmin = 1/sqrt(2) * ([0I exp(1im * θmin) * α - exp(-1im * θmin) * β; 0I 0I] + hc)
    @test reduced_γmin ≈ reconstructed_γmin
end
