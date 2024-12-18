mutable struct Hamiltonian{B}
    ham::QuantumDots.BlockDiagonal
    basis::B
    # QuantumDots.DiagonalizedHamiltonian also stores original ham :(
    diagham::Union{Nothing, QuantumDots.DiagonalizedHamiltonian}
    function Hamiltonian(ham, basis)
        if !(QuantumDots.symmetry(basis) isa QuantumDots.AbelianFockSymmetry{<:Any, <:Any, <:Any, ParityConservation})
            throw(ArgumentError("Basis must have a parity quantum number"))
        end
        return new{typeof(basis)}(ham, basis, nothing)
    end
end

diagonalize!(H::Hamiltonian) = (H.diagham = diagonalize(H.ham))
get_system_basis(H::Hamiltonian) = H.basis
isdiagonalized(H::Hamiltonian) = !isnothing(H.diagham)

function get_ground_states(H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    fullsectors = QuantumDots.blocks(H.diagham; full=true)
    oddvec = first(eachcol(fullsectors[1].vectors))
    evenvec = first(eachcol(fullsectors[2].vectors))
    return oddvec, evenvec
end

function majorana_coefficients(H::Hamiltonian)
    isdiagonalized(H) || diagonalize!(H)
    basis = get_system_basis(H)
    oddvec, evenvec = get_ground_states(H)
    return QuantumDots.majorana_coefficients(oddvec, evenvec, basis)
end

function MP(R, H)
    isdiagonalized(H) || diagonalize!(H)
    majcoeffs = majorana_coefficients(H)
    return majorana_polarization(majcoeffs..., R).mp
end


@testitem "Hamiltonian struct and metrics" begin
    using QuantumDots, LinearAlgebra
    import QuantumDots: kitaev_hamiltonian
    import Majoranas: Hamiltonian, MP, diagonalize!, get_ground_states, isdiagonalized
    cpmm = FermionBasis(1:2; qn=ParityConservation())
    cpmm_noparity = FermionBasis(1:2)
    pmmham = blockdiagonal(Hermitian(kitaev_hamiltonian(cpmm; μ=0.0, t=1.0, Δ=1.0)), cpmm)
    H = Hamiltonian(pmmham, cpmm)
    @test_throws ArgumentError Hamiltonian(pmmham, cpmm_noparity)
    @test !isdiagonalized(H)
    diagonalize!(H)
    @test isdiagonalized(H)
    R = [1]
    @test MP(R, H) ≈ 1.0
    H = Hamiltonian(pmmham, cpmm)
    @test MP(R, H) ≈ 1.0 # diagonalize! is called in MP
end

