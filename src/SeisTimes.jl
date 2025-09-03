module SeisTimes

using StaticArrays, LinearAlgebra, Einsum, NLsolve
using Printf, Test

include(joinpath(@__DIR__, "tracer.jl"))
export ray_bending

include(joinpath(@__DIR__, "wavefront.jl"))
export fast_sweep, Solid2D, Solid3D

end