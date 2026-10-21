const VP = 2000.0
const VS = 1000.0

function homogeneous_velmod(coords::NTuple{D}, vp, vs) where D
    velmod = zeros(D == 2 ? 7 : 13, length.(coords)...)
    for (I, x) in zip(CartesianIndices(length.(coords)), Iterators.product(coords...))
        velmod[1:D, I] .= x
    end
    selectdim(velmod, 1, D + 1) .= vp
    selectdim(velmod, 1, D + 2) .= vs
    return velmod
end

tt_analytic(coords, src, v) = [sqrt(sum(abs2, x .- src)) / v for x in Iterators.product(coords...)]

# boundary layers are extrapolated, not solved, so only the interior is compared
function interior_error(tt, tt_ana, scheme)
    ns    = SeisTimes.stencil_width(SeisTimes.scheme_from_symbol(scheme))
    inner = map(n -> 1+ns:n-ns, size(tt))
    return maximum(abs.(tt[inner...] .- tt_ana[inner...])) / maximum(tt_ana)
end

function test_homogeneous(coords, src, modes, tol)
    velmod = homogeneous_velmod(coords, VP, VS)
    for (mode, v) in modes, scheme in (:LxFS3, :LxFS5)
        @testset "$mode $scheme" begin
            res = only(fast_sweep(velmod, [src], mode, scheme; log_level=:error))
            @test res.converged
            @test interior_error(res.traveltimes, tt_analytic(coords, src, v), scheme) < tol
        end
    end
end

@testset "Homogeneous isotropic 2D" begin
    coords = ntuple(_ -> 0.0:10.0:500.0, 2)
    test_homogeneous(coords, (130.0, 370.0), (:P => VP, :S => VS), 0.01)
end

@testset "Homogeneous isotropic 3D" begin
    coords = ntuple(_ -> 0.0:25.0:500.0, 3)
    test_homogeneous(coords, (130.0, 250.0, 370.0), (:P => VP, :S1 => VS, :S2 => VS), 0.025)
end
