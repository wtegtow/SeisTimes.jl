module SeisTimes

using StaticArrays, LinearAlgebra, Einsum, NLsolve
using Printf, Test

include(joinpath(@__DIR__, "LxFS.jl"))
export fast_sweep, Solid2D, Solid3D

end