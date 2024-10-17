struct HamiltonianBasis{D,B} <: AbstractMajoranaBasis
    dict::D
    fermion_basis::B
end

function HamiltonianBasis(γ::SingleParticleMajoranaBasis)
    dict = ManyBodyMajoranaBasis(γ, 2:2:length(γ)).dict # 0 doesn't have to be in right?
    id_mat = copyto!(similar(first(dict)[2], size(first(dict)[2])), I)
    vd = collect(values(dict))
    kd = keys(dict)
    push!(vd, id_mat)
    kd = [collect(kd)..., empty(first(keys(dict)))]
    newdict = OrderedDict(zip(kd, vd))
    return HamiltonianBasis(newdict, γ.fermion_basis)
end

function ProjectedHamiltonianBasis(γ::SingleParticleMajoranaBasis, parity::Int)
    H = HamiltonianBasis(γ)
    inds = γ.fermion_basis.symmetry.qntoinds[parity]
    projected_mats = map(mat->mat[inds, inds], values(H.dict))
    # not all matrices are independent when projected
    independent_inds = [(1:div(length(H), 2)-1)..., length(H)] # is this really correct?
    independent_labels = labels(H)[independent_inds]
    return HamiltonianBasis(OrderedDict(zip(independent_labels, projected_mats[independent_inds])), H.fermion_basis)
    # return HamiltonianBasis(QuantumDots.Dictionary(independent_labels, projected_mats[independent_inds]), H.fermion_basis)
end

@testitem "HamiltonianBasis" begin
    using QuantumDots
    import QuantumDots: kitaev_hamiltonian
    c = FermionBasis(1:3; qn=QuantumDots.parity)
    γ = SingleParticleMajoranaBasis(c)
    ham_basis = HamiltonianBasis(γ)
    μ, t, Δ, V = rand(4)
    pmmham = kitaev_hamiltonian(c; μ, t, Δ, V)
    dict = Majoranas.matrix_to_dict(ham_basis, pmmham)
    @test dict[((1, :+), (1, :-))] ≈ μ/2 - V/4
    @test dict[((2, :+), (2, :-))] ≈ μ/2 - V/2 # different value inside bulk
    @test dict[((3, :+), (3, :-))] ≈ μ/2 - V/4
    @test dict[((1, :+), (2, :-))] ≈ 1/2*(Δ + t)
    @test dict[((1, :-), (2, :+))] ≈ 1/2*(Δ - t)
    @test dict[((1, :+), (1, :-), (2, :+), (2, :-))] ≈ V/4 # sign is -1 on 4 length basis
end

@testitem "Projected HamiltonianBasis" begin
    N = 12
    γ = SingleParticleMajoranaBasis(N)
    H = HamiltonianBasis(γ)
    Hp = ProjectedHamiltonianBasis(γ, -1)
    # check setdiff(fullmajorana, Hp_labels) and see that none come up in Hp_labels
    setdiffs = [setdiff(labels(H)[end-1], lab) for lab in labels(Hp)]
    @test !all(setdiffs .∋ labels(Hp))
    @test length(Hp) == 2^(N-2)
end
