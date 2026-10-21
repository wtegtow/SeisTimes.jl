module SeisTimes

    using Dates
    include(joinpath(@__DIR__, "logger.jl"))

    # Lax-Friedrichs eikonal solver
    using NPZ, JLD2, HDF5, StaticArrays
    using LinearAlgebra, Printf, DelimitedFiles
    include(joinpath(@__DIR__, "LxFS.jl"));
    export fast_sweep, load_fast_sweep_result

end
