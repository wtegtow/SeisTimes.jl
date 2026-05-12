module SeisTimes

    export devmode!
    const _DEV = Ref(false)
    devmode!(on::Bool=true) = (_DEV[] = on)
    _log(msg) = _DEV[] && println("  [dev] ", msg)

    # Lax-Friedrichs eikonal solver
    using NPZ, JLD2, HDF5, StaticArrays
    using LinearAlgebra, Printf, DelimitedFiles
    include(joinpath(@__DIR__, "LxFS.jl")); 
    export fast_sweep, load_fast_sweep_result

end