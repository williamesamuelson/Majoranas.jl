struct HamiltonianBasis{D,B,N} <: AbstractMajoranaBasis
    dict::D
    fermion_basis::B
    basis_norm::N
end

function HamiltonianBasis(γ::SingleParticleMajoranaBasis; max_length::Int=length(γ))
    dict = ManyBodyMajoranaBasis(γ, 2:2:max_length).dict
    id_mat = copyto!(similar(first(dict)[2], size(first(dict)[2])), I)
    id_mat_normalized = only(_normalize_basis((id_mat,), γ.basis_norm))
    vd = collect(values(dict))
    kd = keys(dict)
    push!(vd, id_mat_normalized)
    kd = [collect(kd)..., empty(first(keys(dict)))]
    newdict = OrderedDict(zip(kd, vd))
    return HamiltonianBasis(newdict, γ.fermion_basis, γ.basis_norm)
end

_get_basis_norm(γ::HamiltonianBasis) = γ.basis_norm

function ProjectedHamiltonianBasis(γ::SingleParticleMajoranaBasis, parity::Int)
    H = HamiltonianBasis(γ)
    inds = γ.fermion_basis.symmetry.qntoinds[parity]
    projected_mats = map(mat->mat[inds, inds], values(H))
    # not all matrices are independent when projected
    independent_inds = [(1:div(length(H), 2)-1)..., length(H)] # is this really correct?
    independent_labels = collect(keys(H))[independent_inds]
    return HamiltonianBasis(OrderedDict(zip(independent_labels, projected_mats[independent_inds])), H.fermion_basis, H.basis_norm)
    # return HamiltonianBasis(QuantumDots.Dictionary(independent_labels, projected_mats[independent_inds]), H.fermion_basis)
end

@testitem "HamiltonianBasis" begin
    using QuantumDots
    import QuantumDots: kitaev_hamiltonian
    import Majoranas: HilbertNorm, FrobeniusNorm
    N = 3
    c = FermionBasis(1:N; qn=QuantumDots.parity)
    μ, t, Δ, V = rand(4)
    pmmham = kitaev_hamiltonian(c; μ, t, Δ, V)
    for basis_norm in (HilbertNorm(), FrobeniusNorm())
        γ = SingleParticleMajoranaBasis(c; basis_norm)
        factor = basis_norm == HilbertNorm() ? 1 : sqrt(2^N)
        ham_basis = HamiltonianBasis(γ; max_length=4)
        dict = Majoranas.matrix_to_dict(ham_basis, pmmham)
        @test dict[((1, :+), (1, :-))] ≈ factor * (μ/2 - V/4)
        @test dict[((2, :+), (2, :-))] ≈ factor * (μ/2 - V/2) # different value inside bulk
        @test dict[((3, :+), (3, :-))] ≈ factor * (μ/2 - V/4)
        @test dict[((1, :+), (2, :-))] ≈ factor * (1/2*(Δ + t))
        @test dict[((1, :-), (2, :+))] ≈ factor * (1/2*(Δ - t))
        @test dict[((1, :+), (1, :-), (2, :+), (2, :-))] ≈ factor * V/4 # sign is -1 on 4 length basis
        @test sum(dict[key] * ham_basis[key] for key in keys(ham_basis)) ≈ pmmham
    end
end

@testitem "Projected HamiltonianBasis" begin
    N = 12
    γ = SingleParticleMajoranaBasis(N)
    H = HamiltonianBasis(γ)
    Hp = ProjectedHamiltonianBasis(γ, -1)
    # check setdiff(fullmajorana, Hp_labels) and see that none come up in Hp_labels
    setdiffs = [setdiff(collect(keys(H))[end-1], lab) for lab in keys(Hp)]
    @test !all(setdiffs .∋ collect(keys(Hp)))
    @test length(Hp) == 2^(N-2)
end
