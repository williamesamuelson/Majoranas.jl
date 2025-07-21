
function diagonalize(m::BlockDiagonal)
    BlockDiagonals.eigen_blockwise(m)
end

function diagonalize!(m::BlockDiagonal)
    eigen!_blockwise(m)
end
diagonalize(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize!(m)
diagonalize!(m::BlockDiagonal{<:Any,<:SparseMatrixCSC}) = diagonalize!(BlockDiagonal(Matrix.(m.blocks)))
diagonalize(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize!(m)
diagonalize!(m::BlockDiagonal{<:Any,<:Hermitian{<:Any,<:SparseMatrixCSC}}) = diagonalize!(BlockDiagonal((Hermitian ∘ Matrix).(m.blocks)))

function eigen!_blockwise(B::BlockDiagonal, args...; kwargs...)
    eigens = [eigen!(b, args...; kwargs...) for b in blocks(B)]
    values = [e.values for e in eigens]
    vectors = [e.vectors for e in eigens]
    vcat(values...), BlockDiagonal(vectors)
end


"""
    blockdiagonal(m::AbstractMatrix, basis::SymmetricFockHilbertSpace)

Construct a BlockDiagonal version of `m` using the symmetry of `basis`. No checking is done to ensure this is a faithful representation.
"""
blockdiagonal(m::AbstractMatrix, basis::FermionicHilbertSpaces.SymmetricFockHilbertSpace) = blockdiagonal(m, basis.symmetry)
blockdiagonal(m::AbstractMatrix, ::FermionicHilbertSpaces.SimpleFockHilbertSpace) = m
blockdiagonal(m::AbstractMatrix, ::FermionicHilbertSpaces.FockHilbertSpace) = m

function blockdiagonal(m::AbstractMatrix, sym::FermionicHilbertSpaces.FockSymmetry)
    blockinds = values(sym.qntoinds)
    BlockDiagonal([m[block, block] for block in blockinds])
end
function blockdiagonal(m::Hermitian, sym::FermionicHilbertSpaces.FockSymmetry)
    blockinds = [map(state -> sym.state_indexdict[state], block) for block in values(sym.qntofockstates)]
    Hermitian(BlockDiagonal([m[block, block] for block in blockinds]))
end

_blocks(m::AbstractMatrix) = [m]
_blocks(m::BlockDiagonal) = blocks(m)
function BlockDiagonals.blocks(eig::DiagonalizedHamiltonian; full=false)
    vals = eig.values
    vecs = eig.vectors
    bvecs = _blocks(vecs)
    sizes = size.(bvecs, 1)
    blockinds = map(i -> eachindex(vals)[i], sizestoinds(sizes))
    if full
        filteredinds = [map(i -> i in inds, eachindex(vals)) for inds in blockinds]
        map((inds, block) -> DiagonalizedHamiltonian(vals[inds], vecs[:, inds], original_hamiltonian(eig)), filteredinds, bvecs)
    else
        original_blocks = _blocks(original_hamiltonian(eig))
        map((inds, block, hblock) -> DiagonalizedHamiltonian(vals[inds], block, hblock), blockinds, bvecs, original_blocks)
    end
end
