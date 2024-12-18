mutable struct Hamiltonian{B}
    ham::Union{QuantumDots.BlockDiagonal, QuantumDots.DiagonalizedHamiltonian}
    const basis::B
    function Hamiltonian(ham, basis)
        if !(QuantumDots.symmetry(basis) isa QuantumDots.AbelianFockSymmetry{<:Any, <:Any, <:Any, ParityConservation})
            throw(ArgumentError("Basis must have a parity quantum number"))
        end
        return new{typeof(basis)}(ham, basis)
    end
end

diagonalize!(H::Hamiltonian) = (H.ham = diagonalize(H.ham))
get_basis(H::Hamiltonian) = H.basis
isdiagonalized(H::Hamiltonian) = (H.ham isa QuantumDots.DiagonalizedHamiltonian)

function ground_states(H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    sectors = QuantumDots.blocks(H.ham; full=true)
    oddvec = first(eachcol(sectors[1].vectors))
    evenvec = first(eachcol(sectors[2].vectors))
    return oddvec, evenvec
end

function energy_info(H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    sectors = QuantumDots.blocks(H.ham)
    oddvals = sectors[1].values
    evenvals = sectors[2].values
    deg = first(oddvals) - first(evenvals)
    energies = (oddvals, evenvals)
    (; deg, energies, deg_ratio=deg_ratio(oddvals, evenvals), excgap=excgap(oddvals, evenvals))
end

function deg_ratio(oddvals, evenvals)
    δE = first(oddvals) - first(evenvals)
    Δ = min(oddvals[2], evenvals[2]) - min(first(oddvals), first(evenvals))
    return δE / Δ
end
deg_ratio(H::Hamiltonian) = energy_info(H).deg_ratio

excgap(oddvals, evenvals) = min(oddvals[2] - oddvals[1], evenvals[2] - evenvals[1])
excgap(H::Hamiltonian) = energy_info(H).excgap

function majorana_coefficients(H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    basis = get_basis(H)
    oddvec, evenvec = ground_states(H)
    return QuantumDots.majorana_coefficients(oddvec, evenvec, basis)
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

function LF(R, H::Hamiltonian)
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
    β = - A_red[block_inds_red[2], block_inds_red[1]]'
    return sqrt(norm(α)^2 + norm(β)^2 - 2 * abs(tr(α * β')))
end


@testitem "Hamiltonian struct and metrics" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    import Majoranas: Hamiltonian, MP, LD, LF, diagonalize!, ground_states, isdiagonalized, energy_info
    cpmm = FermionBasis(1:2; qn=ParityConservation())
    cpmm_noparity = FermionBasis(1:2)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(cpmm; μ=0.0, t=1.0, Δ=1.0)), cpmm)
    H = Hamiltonian(pmmham, cpmm)
    @test_throws ArgumentError Hamiltonian(pmmham, cpmm_noparity)
    @test !isdiagonalized(H)
    diagonalize!(H)
    @test isdiagonalized(H)

    regions = [[1], [2], [1, 2]]
    @test all(map(R -> MP(R, H), regions) .≈ [1.0, 1.0, 0.0])
    @test all(map(R -> LD(R, H), regions) .≈ [0.0, 0.0, sqrt(2)])
    @test all(map(R -> LF(R, H), regions) .≈ [0.0, 0.0, 1.0])
    H = Hamiltonian(pmmham, cpmm)
    @test MP(regions[1], H) ≈ 1.0 # diagonalize! is called in MP
    e_info = energy_info(H)
    H = Hamiltonian(pmmham, cpmm)
    @test e_info.deg_ratio ≈ Majoranas.deg_ratio(H) ≈ 0 # call diagonalize! again
    @test e_info.excgap ≈ Majoranas.excgap(H) ≈ 2.0
end

