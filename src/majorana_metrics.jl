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

function LD(R, H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    oddvec, evenvec = ground_states(H)
    δρ = oddvec * oddvec' - evenvec * evenvec'
    return norm(partial_trace(δρ, R, get_basis(H)))
end

function LF_info(R, H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    oddvec, evenvec = ground_states(H)
    γ = oddvec * evenvec' + evenvec * oddvec'
    basis = get_basis(H)
    block_inds = QuantumDots.blockinds(QuantumDots.symmetry(basis))
    γ[block_inds[2], block_inds[1]] .= 0
    basis_red = FermionBasis(R; qn=ParityConservation())
    A_red = partial_trace(γ, basis_red, basis)
    block_inds_red = QuantumDots.blockinds(QuantumDots.symmetry(basis_red))
    α = A_red[block_inds_red[1], block_inds_red[2]]
    β = -A_red[block_inds_red[2], block_inds_red[1]]'
    α2_plus_β2 = norm(α)^2 + norm(β)^2
    LFmin, LFmax = map(sgn -> sqrt(α2_plus_β2 + sgn * 2 * abs(tr(α * β'))), (-1, 1))
    prod = LFmin * LFmax
    return (;LFmin, prod)
end

LF(R, H::Hamiltonian) = LF_info(R, H).LFmin

function creduced(R, H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    oddvec, evenvec = ground_states(H)
    γ = oddvec * evenvec' + evenvec * oddvec'
    basis = get_basis(H)
    P = QuantumDots.parityoperator(basis)
    basis_red = FermionBasis(R; qn=ParityConservation())
    c = (I + P) * γ
    return norm(partial_trace(c, basis_red, basis))
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
    import Majoranas: MP, LD, LF, single_particle_LF
    cpmm = FermionBasis(1:2; qn=ParityConservation())
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(cpmm; μ=1.0, t=exp(1im * pi / 3), Δ=2.0, V=2.0)), cpmm)
    H = Hamiltonian(pmmham; basis=cpmm)
    regions = [[1], [2], [1, 2]]
    @test all(map(R -> MP(R, H), regions) .≈ [1.0, 1.0, 0.0])
    @test haskey(H, :maj_coeffs)
    @test all(map(R -> LD(R, H), regions) .≈ [0.0, 0.0, sqrt(2)])
    @test all(map(R -> LF(R, H), regions) .≈ [0.0, 0.0, 1.0])
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
