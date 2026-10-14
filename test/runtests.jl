using SignalAnalysis
using Test
using Random
using Statistics: mean

const FS = 256.0

@testset "SignalAnalysis" begin

@testset "windows" begin
    for k in (:rect, :hann, :hamming, :blackman)
        w = get_window(k, 64)
        @test length(w) == 64
        @test all(-1e-12 .<= w .<= 1 + 1e-12)
    end
    @test get_window(:hann, 1) == [1.0]
    @test mean(get_window(:hann, 1024)) ≈ 0.5 atol = 1e-12
    @test mean(abs2, get_window(:hann, 1024)) ≈ 0.375 atol = 1e-12
    @test_throws ArgumentError get_window(:bogus, 16)
    @test_throws ArgumentError get_window(:hann, 0)
end

@testset "frequency axis" begin
    f = get_freq(1024, FS)
    @test length(f) == 513
    @test f[1] == 0
    @test f[end] == FS / 2
    @test f[2] ≈ FS / 1024
    @test get_freq(zeros(9), FS) == get_freq(9, FS)
end

@testset "decomp_freq: on-bin tone amplitude and phase" begin
    n = 1024
    t = (0:n - 1) ./ FS
    for (A, f0, φ) in ((10.0, 3.0, 0.0), (2.5, 10.0, 1.0), (7.0, 40.0, -2.0))
        x = A .* cos.(2π .* f0 .* t .+ φ)
        for win in (:rect, :hann, :hamming, :blackman)
            ps = decomp_freq(x, FS; window = win)
            k = nearest_bin(ps, f0)
            @test ps.freq[k] ≈ f0
            @test ps.magnitude[k] ≈ A rtol = 1e-6
            @test ps.phase[k] ≈ φ atol = 1e-6
            @test ps.scaling === :amplitude
        end
    end
    @test_throws ArgumentError decomp_freq([1.0], FS)
    @test_throws ArgumentError decomp_freq(randn(64), 0.0)
    @test_throws ArgumentError decomp_freq(randn(64), FS; scaling = :bogus)
end

@testset "Parseval and scaling" begin
    rng = MersenneTwister(1)
    x = randn(rng, 4096)
    pr = decomp_freq(x, FS; window = :rect, detrend = false)
    @test band_power(pr, 0, FS / 2) ≈ power(x) rtol = 1e-12
    for win in (:hann, :hamming, :blackman)
        pp = decomp_freq(x, FS; window = win, scaling = :power, detrend = false)
        @test pp.scaling === :power
        @test band_power(pp, 0, FS / 2) ≈ power(x) rtol = 0.05
    end
    # amplitude-scaled Hann is refused: it would be 1.5x too high
    @test_throws ArgumentError band_power(decomp_freq(x, FS), 0, FS / 2)
    t = (0:4095) ./ FS
    pt = decomp_freq(10 .* sin.(2π .* 3 .* t), FS; scaling = :power)
    @test band_power(pt, 2, 4) ≈ 50 rtol = 1e-6
    @test band_power(pt, :delta) ≈ 50 rtol = 1e-6
    @test_throws ArgumentError band_power(pt, :bogus)
end

@testset "reconstruct round trip" begin
    rng = MersenneTwister(2)
    x = randn(rng, 501)
    for n in (500, 501)               # even and odd: Nyquist bin present or not
        xx = x[1:n]
        for sc in (:amplitude, :power, :none)
            ps = decomp_freq(xx, FS; window = :rect, detrend = false, scaling = sc)
            @test reconstruct(ps) ≈ xx atol = 1e-9
        end
    end
end

@testset "spectrum helpers" begin
    ps = decomp_freq(randn(MersenneTwister(3), 512), FS)
    @test resolution(ps) == FS / 512
    @test nyquist(ps) == FS / 2
    @test nearest_bin(ps, 0) == 1
    @test nearest_bin(ps, FS / 2) == length(ps)
    @test_throws ArgumentError nearest_bin(ps, FS)
    @test length(collect(ps)) == length(ps)
    @test ps[3].freq == ps.freq[3]
    sel = isolate_freq(ps, 8, 13)
    @test all(8 .<= sel.freq .<= 13)
    @test sel.bins == findall(f -> 8 <= f <= 13, ps.freq)
    @test_throws ArgumentError isolate_freq(ps, 13, 8)
    @test sprint(show, MIME("text/plain"), ps) isa String
end

@testset "find_peaks" begin
    v = [0, 1, 0, 3, 0, 2, 0.0]
    r = find_peaks(v)
    @test r.indices == [2, 4, 6]
    @test r.heights == [1, 3, 2]
    @test r.prominences ≈ [1, 3, 2]
    @test find_peaks(v; height = 1.5).indices == [4, 6]
    @test find_peaks(v; distance = 3).indices == [4]
    @test find_peaks([0, 1, 1, 0.0]).indices == [2]          # plateau reported once
    @test isempty(find_peaks([1.0, 2.0]).indices)             # too short
    # a bump on a flank is tall but not prominent
    big = [0, 2, 4, 6, 8, 10, 8, 6, 7, 2, 0.0]
    r = find_peaks(big)
    @test r.indices == [6, 9]
    @test r.prominences ≈ [10, 1]
    @test find_peaks(big; prominence = 2).indices == [6]
    # off-bin tones recovered by parabolic interpolation
    n = 1024
    t = (0:n - 1) ./ FS
    x = 10 .* sin.(2π .* 3.1 .* t) .+ 2 .* sin.(2π .* 10.3 .* t)
    pk = find_peaks(decomp_freq(x, FS); prominence = 1.0, interpolate = true)
    @test length(pk.freqs) == 2
    @test pk.freqs[1] ≈ 3.1 atol = 0.05
    @test pk.freqs[2] ≈ 10.3 atol = 0.05
    @test pk.heights[1] ≈ 10 rtol = 0.1
    @test pk.heights[2] ≈ 2 rtol = 0.1
end

@testset "filters" begin
    t = (0:8191) ./ FS
    x = sin.(2π .* 10 .* t) .+ sin.(2π .* 60 .* t)
    seg = 1001:7400                     # skip edges; 6400 samples puts 10 and 60 Hz on bins
    y = bandpass(x, FS, 0.5, 40.0)
    py = decomp_freq(y[seg], FS)
    @test py.magnitude[nearest_bin(py, 10.0)] ≈ 1 rtol = 1e-3
    @test py.magnitude[nearest_bin(py, 60.0)] < 0.05
    ref = decomp_freq(sin.(2π .* 10 .* t)[seg], FS)
    @test abs(py.phase[nearest_bin(py, 10.0)] - ref.phase[nearest_bin(ref, 10.0)]) < 1e-6
    z = notch(x, FS, 60.0)
    pz = decomp_freq(z[seg], FS)
    @test pz.magnitude[nearest_bin(pz, 60.0)] < 1e-6
    @test pz.magnitude[nearest_bin(pz, 10.0)] ≈ 1 rtol = 1e-3
    lp = lowpass(x, FS, 30.0)
    hp = highpass(x, FS, 30.0)
    pl = decomp_freq(lp[seg], FS)
    ph = decomp_freq(hp[seg], FS)
    @test pl.magnitude[nearest_bin(pl, 10.0)] ≈ 1 rtol = 1e-2
    @test pl.magnitude[nearest_bin(pl, 60.0)] < 0.05
    @test ph.magnitude[nearest_bin(ph, 60.0)] ≈ 1 rtol = 1e-2
    @test ph.magnitude[nearest_bin(ph, 10.0)] < 0.05
    @test_throws ArgumentError bandpass(x, FS, 40.0, 0.5)
    @test_throws ArgumentError lowpass(x, FS, 200.0)
    @test_throws ArgumentError highpass(x, FS, 0.0)
    @test_throws ArgumentError notch(x, FS, 60.0; bw = -1.0)
    @test_throws ArgumentError bandpass(x, FS, 0.5, 40.0; order = 0)
end

@testset "time domain" begin
    t = (0:2559) ./ FS                  # 10 s
    A = 3.0
    x = A .* sin.(2π .* 4 .* t)         # 40 whole cycles, 64 samples each
    @test power(x) ≈ A^2 / 2 rtol = 1e-12
    @test rms(x) ≈ A / √2 rtol = 1e-12
    @test energy(x, FS) ≈ power(x) * length(x) / FS rtol = 1e-12
    @test rectified_area(x, FS) ≈ 2A * 10 / π rtol = 1e-3
    @test line_length(x) ≈ 4A * 40 rtol = 2e-3
    @test line_length(x; normalize = true) ≈ line_length(x) / (length(x) - 1)
    @test line_length([1.0]) == 0
    @test_throws ArgumentError energy(x, 0.0)
    d = derivative(x, FS)
    @test d[100:2400] ≈ (2π * 4 * A) .* cos.(2π .* 4 .* t[100:2400]) rtol = 1e-2
    d2 = derivative(x, FS; order = 2)
    @test d2[100:2400] ≈ -(2π * 4)^2 .* x[100:2400] rtol = 1e-2
    @test_throws ArgumentError derivative(x, FS; order = 3)
    @test_throws ArgumentError derivative(x[1:2], FS)
    ts, e = windowed_energy(x, FS; win = 1.0, overlap = 0.5)
    @test length(ts) == length(e) == 19
    @test ts[1] ≈ (255 / 2) / FS
    @test all(isapprox.(e, A^2 / 2; rtol = 1e-9))       # 4 whole cycles per window
    _, ll = windowed_line_length(x, FS; win = 1.0, overlap = 0.5)
    @test all(isapprox.(ll, 4A * 4; rtol = 1e-2))
    @test_throws ArgumentError windowed_energy(x, FS; overlap = 1.0)
    @test_throws ArgumentError windowed_energy(x, FS; win = 20.0)
end

@testset "welch" begin
    rng = MersenneTwister(4)
    n = 256 * 30
    t = (0:n - 1) ./ FS
    x = randn(rng, n)
    w = welch(x, FS; win = 2.0, detrend = false)
    @test w.nseg == 29
    @test resolution(w) == 0.5
    @test nyquist(w) == FS / 2
    @test band_power(w, 0, FS / 2) ≈ power(x) rtol = 0.03
    wt = welch(10 .* sin.(2π .* 3 .* t), FS; win = 2.0)
    @test band_power(wt, 2, 4) ≈ 50 rtol = 1e-6
    @test band_power(wt, :delta) ≈ 50 rtol = 1e-6
    @test find_peaks(wt; prominence = 1.0).freqs == [3.0]
    @test length(collect(wt)) == length(wt)
    @test wt[7].freq == 3.0
    @test sprint(show, MIME("text/plain"), wt) isa String
    # one rectangular segment is the power-scaled periodogram
    w1 = welch(x, FS; win = n / FS, window = :rect, detrend = false)
    p1 = decomp_freq(x, FS; window = :rect, detrend = false, scaling = :power)
    @test w1.nseg == 1
    @test band_power(w1, 0, FS / 2) ≈ band_power(p1, 0, FS / 2) rtol = 1e-10
    @test_throws ArgumentError welch(x, FS; win = 40.0)
end

@testset "detrend" begin
    t = (0:1023) ./ FS
    ramp = 5 .+ 0.7 .* t
    s = 2 .* sin.(2π .* 8 .* t)          # 32 whole cycles
    x = ramp .+ s
    @test detrend(x; method = :none) == x
    @test mean(detrend(x; method = :mean)) ≈ 0 atol = 1e-12
    @test detrend(ramp) ≈ zeros(1024) atol = 1e-9
    @test maximum(abs.(detrend(x) .- s)) < 0.1
    @test_throws ArgumentError detrend(x; method = :quadratic)
    ps_mean = decomp_freq(x, FS; detrend = :mean)
    ps_lin = decomp_freq(x, FS; detrend = :linear)
    k = nearest_bin(ps_lin, 0.5)
    @test ps_lin.magnitude[k] < ps_mean.magnitude[k] / 5   # ramp leakage gone
    @test decomp_freq(x, FS; detrend = true).magnitude == ps_mean.magnitude
    @test decomp_freq(x, FS; detrend = false).magnitude ==
          decomp_freq(x, FS; detrend = :none).magnitude
end

@testset "multi-channel" begin
    rng = MersenneTwister(5)
    X = randn(rng, 256 * 30, 4) .* [1.0 2.0 3.0 4.0]
    @test rms(X) ≈ [1, 2, 3, 4] rtol = 0.05
    @test power(X) ≈ [1, 4, 9, 16] rtol = 0.1
    Y = bandpass(X, FS, 0.5, 40.0)
    @test size(Y) == size(X)
    @test Y[:, 2] ≈ bandpass(X[:, 2], FS, 0.5, 40.0)
    @test size(notch(X, FS, 60.0)) == size(X)
    W = welch(X, FS)
    @test length(W) == 4
    @test band_power(W, 0, FS / 2) ≈ power(X) rtol = 0.05
    S = decomp_freq(X, FS; scaling = :power)
    @test band_power(S, :alpha) isa Vector{Float64}
    @test length(band_power(S, :alpha)) == 4
    ts, E = windowed_energy(X, FS)
    @test size(E) == (59, 4)
    @test length(ts) == 59
    _, L = windowed_line_length(X, FS)
    @test size(L) == (59, 4)
    @test size(derivative(X, FS)) == size(X)
    @test length(line_length(X)) == 4
    @test length(energy(X, FS)) == 4
    @test length(rectified_area(X, FS)) == 4
    @test size(detrend(X)) == size(X)
    @test size(isolate_freq(X, FS, :alpha)) == size(X)
    @test_throws ArgumentError power(randn(1, 4))
end

@testset "connectivity" begin
    rng = MersenneTwister(42)
    fs2 = 250.0
    n2 = round(Int, 30 * fs2)
    t2 = (0:n2 - 1) ./ fs2
    # one generator seen at two electrodes with opposite polarity: zero lag
    src = 4 .* sin.(2π .* 3 .* t2)
    a1 = src .+ 1.2 .* randn(rng, n2)
    a2 = -0.6 .* src .+ 1.2 .* randn(rng, n2)
    # the same rhythm with a genuine 25 ms propagation delay
    lag = 0.025
    b1 = 4 .* sin.(2π .* 3 .* t2) .+ 1.2 .* randn(rng, n2)
    b2 = 4 .* sin.(2π .* 3 .* (t2 .- lag)) .+ 1.2 .* randn(rng, n2)
    function at3(u, v)
        f, C = coherence(u, v, fs2; win = 2.0)
        _, ic = imag_coherency(u, v, fs2; win = 2.0)
        _, pli = phase_lag_index(u, v, fs2; win = 2.0)
        _, wp = weighted_phase_lag_index(u, v, fs2; win = 2.0)
        k = argmin(abs.(f .- 3.0))
        return (coh = C[k], ic = abs(ic[k]), pli = pli[k], wpli = wp[k])
    end
    mA = at3(a1, a2)
    mB = at3(b1, b2)
    @test mA.coh > 0.8                  # coherence cannot tell the cases apart
    @test mB.coh > 0.8
    @test mA.ic < 0.15                  # lag-sensitive measures can
    @test mA.pli < 0.5
    @test mA.wpli < 0.5
    @test mB.ic > 0.3
    @test mB.pli > 0.6
    @test mB.wpli > 0.6
    @test_throws DimensionMismatch coherence(a1, a2[1:end - 1], fs2)
    @test_throws DimensionMismatch imag_coherency(a1, a2[1:end - 1], fs2)
    @test_throws ArgumentError coherence(a1[1:600], a2[1:600], fs2; win = 2.0)
end

end
