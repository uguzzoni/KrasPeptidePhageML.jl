###############################################################################
# Monte-Carlo sequence design over the fitted model.
#
# VENDORED from the original MC implementation (the whole file is reachable), with
# one necessary change: the original hard-codes module-qualified names in the
# signatures (lines 56, 85, 104) and in one call (line 123). They are
# unqualified here.
#
# `ParetoAnneal!` and `isPareto` were not exported upstream (the generation
# notebook reached them by explicit import); they are exported here, since
# generation depends on them.
#
# Acceptance rule (this IS the scalarisation — there is no weighted objective):
# a flip is accepted if ALL ΔE over `modes` are <= 0 (a Pareto improvement), or
# with probability exp(-beta * maximum(ΔE)).
###############################################################################


const Sequence{V<:Real} = AbstractArray{V,2}

#################################################
#slow but general way

function flip(s::Sequence,a::Integer,i::Integer)

	A, L = size(s) 

	@assert 1 ≤ a ≤ A
	@assert 1 ≤ i ≤ L

	s2=copy(s)
	s2[:,i].=zeros(eltype(s),A)
	s2[a,i]=1

	return s2
end

function flip!(s::Sequence,a::Integer,i::Integer)

	A, L = size(s) 

	@assert 1 ≤ a ≤ A
	@assert 1 ≤ i ≤ L

	s[:,i].=zeros(eltype(s),A)
	s[a,i]=1

end

function energies(sequence::Sequence, model::Model)
    energies(reshape(sequence, (size(sequence)...,1)),model)
end

function Δenergies(sequence::Sequence, a::Integer, i::Integer, model::Model)

	A, L = size(sequence) 

	@assert 1 ≤ a ≤ A
	@assert 1 ≤ i ≤ L

	sequence2 = flip(sequence,a,i)

	return energies(sequence2,model) .- energies(sequence,model)

end



#################################################
## MC PARETO

function MCParetoTrial!(s::Sequence, a::Integer,i::Integer,
						model::Model,
						β::Real; 
						modes::Vector{Int64}=collect(1:length(model.states)))

	@assert β ≥ 0
	A, L = size(s) 

	@assert 1 ≤ a ≤ A && 1 ≤ i ≤ L
	ΔE = Δenergies(s,a,i,model)[modes]

	if all(ΔE .≤ 0) || rand() < exp(-β*maximum(ΔE))
		flip!(s,a,i)
		return ΔE
	end

	return zero(ΔE)
end

function MCPareto!(s::Sequence,model::Model,β::Real; kwargs... )

	@assert β ≥ 0
	A, L = size(s) 

	#@assert eltype(s) <: Integer
	a = rand(1:A); i = rand(1:L);
	return MCParetoTrial!(s,a,i,model,β;  kwargs...)
end

function ToParetoFront!(s::Sequence,
						model::Model; 
						modes::Vector{Int64}=collect(1:length(model.states)))

	A, L = size(s) 
	localmin = false
	while !localmin
		localmin = true
		for a=1:A, i=1:L
			ΔE = Δenergies(s,a,i,model)[modes]
			if all(ΔE .< 0)
	            flip!(s,a,i)
				localmin = false
				break
			end
		end
	end
end

function ParetoAnneal!(seq::Sequence,
                        model::Model,
                        schedule::AbstractVector{<:Real};  
						kwargs...)

	A, L = size(seq) 

	for β in schedule
        for i = 1 : length(seq)
            MCPareto!(seq, model, β;  kwargs...)
        end
    end
end

function isPareto(seq,model;modes=[1,2])

	A,L=size(seq)

    flag=true
    for a=1:A, i=1:L
                ΔE = Δenergies(seq,a,i,model)[modes]
                if all(ΔE .< 0)
                    flag = false
                    break
                end
    end
    return flag
end
###############################################################################à
