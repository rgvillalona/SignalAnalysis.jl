module SignalAnalysis

using FFTW
using Statistics: mean

export PolarSpectrum, resolution, nyquist, nearest_bin

export get_window, get_freq, decomp_freq, to_polar, to_complex, reconstruct,
       isolate_freq, find_peaks, band_power, EEG_BANDS
       
export derivative, energy, power, rms, rectified_area, line_length,
       windowed_energy, windowed_line_length

export cross_spectrum, coherence, imag_coherency, phase_lag_index,
       weighted_phase_lag_index

# =============================================================================
# UNITS
#
# Nothing here guesses units. Input in µV and fs in Hz gives:
#   energy µV²·s | power, variance µV² | rms, magnitude µV
#   rectified_area µV·s | line_length µV | derivative µV/s
#
# line_length has voltage units, not voltage/time, because it sums voltage
# differences rather than a rate. That is what makes it sensitive to amplitude
# and frequency simultaneously.
# =============================================================================

# =============================================================================
# TYPE
# =============================================================================

"""
    PolarSpectrum{T<:Real}

Single-sided spectrum in polar form. Bundling magnitude, phase, the frequency
axis, fs and n means downstream functions work in Hz instead of bin indices, and
you cannot pair a magnitude vector with the wrong frequency axis.

`n` is the ORIGINAL sample count, not the bin count — it cannot be recovered from
the bin count alone (n=8 and n=9 both give 5 bins) and `irfft` needs it.
`scale` records the amplitude normalisation so the transform can be inverted.
"""
struct PolarSpectrum{T<:Real}
    freq::Vector{T}
    magnitude::Vector{T}
    phase::Vector{T}
    fs::T
    n::Int
    scale::T

    function PolarSpectrum(freq::Vector{T}, magnitude::Vector{T},
                           phase::Vector{T}, fs::T, n::Integer,
                           scale::T) where {T<:Real}
        length(freq) == length(magnitude) == length(phase) ||
            throw(DimensionMismatch("freq, magnitude, phase must match in length"))
        fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
        n > 0 || throw(ArgumentError("sample count must be positive, got $n"))
        length(freq) == n ÷ 2 + 1 || throw(DimensionMismatch(
            "an $n-sample real signal gives $(n ÷ 2 + 1) bins, got $(length(freq))"))
        return new{T}(freq, magnitude, phase, fs, Int(n), scale)
    end
end

Base.length(ps::PolarSpectrum) = length(ps.freq)
Base.getindex(ps::PolarSpectrum, k::Integer) =
    (freq = ps.freq[k], magnitude = ps.magnitude[k], phase = ps.phase[k])

"""
    resolution(ps)

Bin spacing Δf = fs/n = 1/T, in Hz.

Set entirely by epoch DURATION, not sampling rate. A 2-second epoch gives 0.5 Hz
whether sampled at 256 Hz or 2 kHz — only a longer epoch helps, and only if the
signal is stationary across it. To separate 3 Hz from 2.75 Hz you need ≥4 s.
"""
resolution(ps::PolarSpectrum) = ps.fs / ps.n

"""
    nyquist(ps)

fs/2. Content above this is not lost, it is ALIASED — folded down onto lower
frequencies where it is indistinguishable from real content. The anti-aliasing
filter belongs in the amplifier and cannot be fixed here.
"""
nyquist(ps::PolarSpectrum) = ps.fs / 2

"""
    nearest_bin(ps, f)

Index of the bin centred closest to `f` Hz. Bins sit at k·fs/n.
"""
function nearest_bin(ps::PolarSpectrum, f::Real)
    0 <= f <= nyquist(ps) || throw(ArgumentError(
        "$f Hz is outside [0, $(nyquist(ps))] Hz"))
    return clamp(round(Int, f * ps.n / ps.fs) + 1, 1, length(ps))
end

function Base.show(io::IO, ::MIME"text/plain", ps::PolarSpectrum{T}) where {T}
    println(io, "PolarSpectrum{$T}")
    println(io, "  bins       : ", length(ps))
    println(io, "  samples    : ", ps.n)
    println(io, "  fs         : ", ps.fs, " Hz")
    println(io, "  resolution : ", round(resolution(ps), digits = 4), " Hz")
    if !isempty(ps.magnitude)
        k = argmax(ps.magnitude)
        println(io, "  peak       : ", round(ps.freq[k], digits = 3), " Hz")
    end
end

# =============================================================================
# BANDS
# =============================================================================

"""
    EEG_BANDS

Conventional clinical bands in Hz. These are shared vocabulary, not physics —
different labs use different cutoffs and the bands are not separate mechanisms.

Note 3 Hz spike-and-wave sits in :delta but its harmonics land in :theta (6 Hz)
and :alpha (9, 12 Hz) — band analysis splits one physiological event across four
bands, which is why harmonic structure is worth tracking explicitly.
"""
const EEG_BANDS = Dict(
    :delta => (0.5, 4.0),
    :theta => (4.0, 8.0),
    :alpha => (8.0, 13.0),
    :beta  => (13.0, 30.0),
    :gamma => (30.0, 100.0),
)

function _band_limits(band::Symbol)
    haskey(EEG_BANDS, band) || throw(ArgumentError(
        "unknown band $band; known: $(sort(collect(keys(EEG_BANDS))))"))
    return EEG_BANDS[band]
end

# =============================================================================
# WINDOWS
# =============================================================================

"""
    get_window(kind, n; periodic = true)

Window coefficients. `kind` ∈ (:rect, :hann, :hamming, :blackman).

WHY WINDOWING IS NOT OPTIONAL. The DFT analyses the infinite periodic extension
of your epoch — it assumes the samples repeat seamlessly forever. If the signal
does not complete a whole number of cycles, the last sample does not join onto
the first, and that splice is a discontinuity. Discontinuities are broadband, so
one clean sinusoid's energy smears across every bin. That is spectral leakage,
and it is an artefact of the epoch boundary, not of the signal.

Tapering to zero at both ends removes the splice. The cost is a wider main lobe:
you trade frequency resolution for dynamic range.

  :rect     narrowest main lobe, worst side lobes (−13 dB). Correct when the
            epoch genuinely holds whole cycles (synthetic tests).
  :hann     the default for EEG. −31 dB side lobes with steep roll-off.
  :hamming  lower first side lobe (−43 dB) but flat roll-off, so distant
            leakage is worse than Hann.
  :blackman −58 dB. Use when hunting something small next to something large.

`periodic = true` divides by n and is correct for spectral analysis (it makes
the window's own period equal n, matching the DFT's assumption). The symmetric
form divides by n−1 and is for FIR filter design.
"""
function get_window(kind::Symbol, n::Integer; periodic::Bool = true)
    n >= 1 || throw(ArgumentError("window length must be ≥ 1, got $n"))
    n == 1 && return ones(Float64, 1)

    denom = periodic ? n : n - 1
    k = 0:(n - 1)

    # Every window below is a raised cosine; the coefficients are chosen so
    # successive side lobes cancel against each other.
    if kind === :rect
        return ones(Float64, n)
    elseif kind === :hann
        return @. 0.5 - 0.5 * cos(2π * k / denom)
    elseif kind === :hamming
        return @. 0.54 - 0.46 * cos(2π * k / denom)
    elseif kind === :blackman
        return @. 0.42 - 0.5 * cos(2π * k / denom) + 0.08 * cos(4π * k / denom)
    else
        throw(ArgumentError(
            "unknown window $kind; expected :rect, :hann, :hamming, :blackman"))
    end
end

"""
    coherent_gain(w)

mean(w). Tapering removes energy — Hann has mean 0.5, so a windowed sinusoid
peaks at half amplitude. Dividing by this restores true amplitude. Correct for
deterministic tones; for broadband noise power use mean(w.^2) instead.
"""
coherent_gain(w::AbstractVector{<:Real}) = mean(w)

# =============================================================================
# FREQUENCY AXIS
# =============================================================================

"""
    get_freq(n, fs)
    get_freq(x, fs)

One-sided frequency axis in Hz: n÷2+1 values from 0 to fs/2, spaced fs/n.

Bin k correlates the signal against a complex sinusoid completing exactly k
cycles across the epoch. The epoch lasts n/fs seconds, so k cycles in n/fs
seconds is k·fs/n Hz. That is the entire derivation.

Only n÷2+1 bins because the input is real: X[n−k] = conj(X[k]), so the upper
half carries nothing new and rfft skips it.
"""
function get_freq(n::Integer, fs::Real)
    n > 0 || throw(ArgumentError("sample count must be positive, got $n"))
    fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
    return collect(0:(n ÷ 2)) .* (float(fs) / n)
end

get_freq(x::AbstractVector, fs::Real) = get_freq(length(x), fs)

# =============================================================================
# POLAR CONVERSION
# =============================================================================

"""
    to_polar(X, fs, n; scale = 1.0, single_sided = true)

Convert a complex half-spectrum (from `rfft`) to a `PolarSpectrum`.

Each X[k] holds two facts: `abs` = how much of that frequency is present,
`angle` = where in its cycle it starts. Real and imaginary parts alone mean
nothing physical — shifting the epoch by one sample reshuffles them completely
while leaving `abs` untouched. Polar form separates what is stable from what
depends on where you started counting.

THE DOUBLING. A real cosine is A·cos(θ) = (A/2)e^{iθ} + (A/2)e^{−iθ}: half its
amplitude sits at +f and half at −f. Discarding negative frequencies discards
half, so the rest is doubled. DC and Nyquist are exempt — they have no distinct
negative partner (−0 = 0, and at fs/2 both images land on the same bin).
"""
function to_polar(X::AbstractVector{<:Complex}, fs::Real, n::Integer;
                  scale::Real = 1.0, single_sided::Bool = true)
    nbins = length(X)
    nbins == n ÷ 2 + 1 || throw(DimensionMismatch(
        "expected $(n ÷ 2 + 1) bins for $n samples, got $nbins"))

    mag = abs.(X) .* float(scale)

    if single_sided && nbins > 1
        for k in 2:nbins
            mag[k] *= 2
        end
        # Nyquist exists only for even n and has no negative-frequency twin.
        if iseven(n)
            mag[nbins] /= 2
        end
    end

    phase = angle.(X)          # already wrapped to (-π, π]
    freq = get_freq(n, fs)

    T = promote_type(eltype(mag), eltype(phase), eltype(freq), typeof(float(fs)))
    return PolarSpectrum(Vector{T}(freq), Vector{T}(mag), Vector{T}(phase),
                         T(fs), n, T(scale))
end

"""
    to_complex(ps)

Rebuild the raw complex half-spectrum, undoing the doubling and the scaling.
"""
function to_complex(ps::PolarSpectrum)
    mag = copy(ps.magnitude)
    nb = length(mag)
    if nb > 1
        for k in 2:nb
            mag[k] /= 2
        end
        if iseven(ps.n)
            mag[nb] *= 2
        end
    end
    mag ./= ps.scale
    return mag .* cis.(ps.phase)     # cis(θ) = exp(im*θ)
end

"""
    reconstruct(ps)

Round-trip back to the time domain. Exact only for `window = :rect` with
`detrend = false` — tapering and mean removal are not invertible from the
spectrum alone.
"""
reconstruct(ps::PolarSpectrum) = irfft(to_complex(ps), ps.n)

# =============================================================================
# DECOMPOSITION
# =============================================================================

"""
    decomp_freq(x, fs; window = :hann, detrend = true, scaled = true)

Decompose a real signal into frequency components. Main entry point.

WHY detrend DEFAULTS TRUE. EEG amplifiers carry a DC offset and electrode drift
adds a slow ramp. Neither is neural, and both are enormous compared to the
signal — millivolts of offset against tens of microvolts of activity is routine.
Left in, the offset dumps into bin 1 and the window's side lobes spread a
fraction of that mountain across the low-frequency bins, contaminating exactly
the delta range where 3 Hz activity lives. (Mean removal does NOT fix a linear
trend — fit and subtract a line first if drift is substantial.)

THE SCALING, DERIVED. For x[j] = A·cos(2πf₀j/fs + φ) with f₀ on bin k₀:

    X[k₀] = Σ x[j]·exp(−2πi k₀ j/n) = (A·n/2)·exp(iφ)

The n/2 is n samples' accumulation of the half-amplitude at +f₀. So dividing by
n removes the length dependence, doubling (in to_polar) recovers the half lost
with the negative frequencies, and dividing by mean(w) undoes the taper — hence
scale = 1/(n·mean(w)), and magnitude[k₀] == A exactly.

Meanwhile angle(X[k₀]) == φ, so phase is referenced to a COSINE at the first
sample of the epoch. Shifting the epoch changes every phase; phase is meaningful
only relative to a fixed origin, or as a difference between channels sharing one.
"""
function decomp_freq(x::AbstractVector{<:Real}, fs::Real;
                     window::Union{Symbol,AbstractVector{<:Real}} = :hann,
                     detrend::Bool = true,
                     scaled::Bool = true)
    n = length(x)
    n >= 2 || throw(ArgumentError("need at least 2 samples, got $n"))
    fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))

    # Float64 regardless of input — integer ADC counts would truncate at detrend.
    xf = float.(collect(x))
    if detrend
        xf .-= mean(xf)
    end

    w = window isa Symbol ? get_window(window, n) : float.(collect(window))
    length(w) == n || throw(DimensionMismatch(
        "window has $(length(w)) coefficients but signal has $n samples"))

    cg = coherent_gain(w)
    cg > 0 || throw(ArgumentError("window has zero mean; cannot correct gain"))

    X = rfft(xf .* w)
    scale = scaled ? 1 / (n * cg) : 1.0
    return to_polar(X, fs, n; scale = scale, single_sided = scaled)
end

# =============================================================================
# BAND SELECTION / FILTERING
# =============================================================================

"""
    isolate_freq(ps, lo, hi)          -> NamedTuple   (SELECT bins)
    isolate_freq(ps, band::Symbol)    -> NamedTuple
    isolate_freq(x, fs, lo, hi)       -> Vector       (FILTER to time domain)
    isolate_freq(x, fs, band::Symbol) -> Vector

Two deliberately different behaviours.

On a PolarSpectrum this SELECTS bins — returns the bins inside [lo, hi] with
magnitude and phase untouched. Nothing is filtered, nothing returns to the time
domain. This is what you want for MEASURING.

On a raw signal this FILTERS — transforms, zeroes bins outside the range,
transforms back. This is what you want for LOOKING.

CAVEAT ON THE FILTER FORM. Zeroing bins is a brick wall: infinitely steep, which
sounds ideal and is not. A rectangular edge in frequency is a sinc in time, so
transients acquire ringing that oscillates BEFORE the event causing it. On an
EEG trace that looks exactly like a rhythmic precursor to a spike and is entirely
an artefact. Use a designed zero-phase filter for anything published.

Note `window` defaults to :rect here — a taper before an invertible transform
corrupts the reconstructed waveform.
"""
function isolate_freq(ps::PolarSpectrum, lo::Real, hi::Real)
    lo <= hi || throw(ArgumentError("lo ($lo) must not exceed hi ($hi)"))
    keep = findall(f -> lo <= f <= hi, ps.freq)
    isempty(keep) && throw(ArgumentError(
        "no bins in [$lo, $hi] Hz; resolution is " *
        "$(round(resolution(ps), digits = 4)) Hz — the band may be narrower " *
        "than one bin, which means you need a longer epoch"))

    # Returned as a NamedTuple, not a PolarSpectrum: a band-limited spectrum no
    # longer starts at 0 Hz and would violate the struct's bin-count invariant.
    return (freq = ps.freq[keep], magnitude = ps.magnitude[keep],
            phase = ps.phase[keep], fs = ps.fs, n = ps.n, bins = keep)
end

isolate_freq(ps::PolarSpectrum, band::Symbol) =
    isolate_freq(ps, _band_limits(band)...)

function isolate_freq(x::AbstractVector{<:Real}, fs::Real, lo::Real, hi::Real;
                      window::Union{Symbol,AbstractVector{<:Real}} = :rect,
                      detrend::Bool = true)
    n = length(x)
    xf = float.(collect(x))
    detrend && (xf .-= mean(xf))

    w = window isa Symbol ? get_window(window, n) : float.(collect(window))
    X = rfft(xf .* w)
    f = get_freq(n, fs)

    for k in eachindex(X)
        if !(lo <= f[k] <= hi)
            X[k] = 0
        end
    end

    return irfft(X, n)
end

isolate_freq(x::AbstractVector{<:Real}, fs::Real, band::Symbol; kwargs...) =
    isolate_freq(x, fs, _band_limits(band)...; kwargs...)

"""
    band_power(ps, lo, hi)
    band_power(ps, band::Symbol)

Total power in the band, in squared voltage units (µV²).

WHY A²/2 AND NOT A². A sinusoid of amplitude A has mean square A²/2, because the
average of cos² over a cycle is 1/2. Summing A_k²/2 across bins reproduces the
signal's mean square exactly — that is Parseval, and it is the strongest single
check on the whole package. DC and Nyquist are excluded from the halving because
they are not oscillations: a constant of size A has mean square A².

Band power scales with bandwidth, so comparing a 0.5 Hz band to a 70 Hz band is
meaningless unless you divide by width or express each as a fraction of total.
"""
function band_power(ps::PolarSpectrum, lo::Real, hi::Real)
    total = 0.0
    last_bin = length(ps)
    has_nyquist = iseven(ps.n)

    for k in eachindex(ps.freq)
        lo <= ps.freq[k] <= hi || continue
        a2 = ps.magnitude[k]^2
        is_edge = (k == 1) || (has_nyquist && k == last_bin)
        total += is_edge ? a2 : a2 / 2
    end
    return total
end

band_power(ps::PolarSpectrum, band::Symbol) = band_power(ps, _band_limits(band)...)

# =============================================================================
# PEAK FINDING
# =============================================================================

"""
    find_peaks(v; height, distance, prominence, interpolate = false)
    find_peaks(ps::PolarSpectrum; kwargs...)

Locate local maxima. On a vector returns (indices, heights, prominences,
offsets); on a spectrum returns (indices, freqs, heights, prominences) with
freqs in Hz and already offset-corrected.

`distance` is in SAMPLES for a vector but in HERTZ for a spectrum.

WHY HEIGHT ALONE IS NOT ENOUGH. Thresholding on height finds every wobble on the
flank of a large peak, because those wobbles are tall in absolute terms while
being nothing but texture on something else. PROMINENCE measures how far a peak
stands above the terrain connecting it to any higher peak: walk left and right
until the ground rises above the peak, note the lowest point reached each way,
subtract the higher of those two minima. A ripple on a flank has near-zero
prominence no matter how tall. (Same definition scipy uses.)

`distance` is applied last and greedily — sort by height, keep the tallest,
discard anything within `distance` of a survivor. This matters for harmonics:
with a 3 Hz fundamental, `distance` must stay well under 3 Hz or the 6/9/12 Hz
peaks get collapsed into the fundamental.

INTERPOLATION. Real components rarely sit on a bin centre; when one falls
between bins its energy splits and the tallest bin is up to half a bin off. With
Δf = 1 Hz a "3 Hz" reading could be anything from 2.5 to 3.5. Fitting a parabola
through the peak and its neighbours recovers the sub-bin location:

    δ = ½(y₋₁ − y₊₁) / (y₋₁ − 2y₀ + y₊₁),  corrected height y₀ − ¼(y₋₁ − y₊₁)δ

Typically cuts frequency error by an order of magnitude on a Hann spectrum.
"""
function find_peaks(v::AbstractVector{<:Real};
                    height::Union{Nothing,Real} = nothing,
                    distance::Union{Nothing,Real} = nothing,
                    prominence::Union{Nothing,Real} = nothing,
                    interpolate::Bool = false)
    n = length(v)
    n >= 3 && return _find_peaks_impl(v, height, distance, prominence, interpolate)
    return (indices = Int[], heights = Float64[], prominences = Float64[],
            offsets = Float64[])
end

function _find_peaks_impl(v, height, distance, prominence, interpolate)
    idx = _local_maxima(v)

    if height !== nothing
        filter!(i -> v[i] >= height, idx)
    end

    proms = [_prominence(v, i) for i in idx]
    if prominence !== nothing
        keep = findall(>=(prominence), proms)
        idx = idx[keep]
        proms = proms[keep]
    end

    # Greedy from the tallest: the survivor of any close pair is the taller one.
    if distance !== nothing && !isempty(idx)
        mask = falses(length(idx))
        kept = Int[]
        for j in sortperm(v[idx]; rev = true)
            if all(abs(idx[j] - p) >= distance for p in kept)
                mask[j] = true
                push!(kept, idx[j])
            end
        end
        idx = idx[mask]
        proms = proms[mask]
    end

    offsets = interpolate ? [_parabolic_offset(v, i) for i in idx] :
                            zeros(Float64, length(idx))
    heights = interpolate ?
        [_parabolic_height(v, idx[j], offsets[j]) for j in eachindex(idx)] :
        Float64[v[i] for i in idx]

    return (indices = idx, heights = heights, prominences = proms,
            offsets = offsets)
end

function find_peaks(ps::PolarSpectrum;
                    distance::Union{Nothing,Real} = nothing, kwargs...)
    df = resolution(ps)
    dist_bins = distance === nothing ? nothing : distance / df   # Hz -> bins
    r = find_peaks(ps.magnitude; distance = dist_bins, kwargs...)
    freqs = [ps.freq[i] for i in r.indices] .+ r.offsets .* df
    return (indices = r.indices, freqs = freqs, heights = r.heights,
            prominences = r.prominences)
end

# Strict local maxima, with plateau handling: a flat run that rises into it and
# falls out counts once, reported at its middle. Runs touching either end are
# not peaks — no evidence of what happens beyond the boundary.
function _local_maxima(v::AbstractVector{<:Real})
    n = length(v)
    peaks = Int[]
    i = 2
    while i < n
        if v[i] > v[i - 1]
            j = i
            while j < n && v[j] == v[j + 1]
                j += 1
            end
            j == n && break
            if v[j] > v[j + 1]
                push!(peaks, (i + j) ÷ 2)
            end
            i = j + 1
        else
            i += 1
        end
    end
    return peaks
end

# Topographic prominence. O(n) per peak worst case, so O(n·p) overall — fine for
# spectra, replace with a tree method if you ever run it on millions of samples.
function _prominence(v::AbstractVector{<:Real}, i::Integer)
    h = float(v[i])

    left_min = h
    j = i
    while j > 1
        j -= 1
        v[j] > h && break
        left_min = min(left_min, v[j])
    end

    right_min = h
    k = i
    while k < length(v)
        k += 1
        v[k] > h && break
        right_min = min(right_min, v[k])
    end

    # You escape a peak by the easier route, so the binding constraint is the
    # HIGHER of the two cols.
    return h - max(left_min, right_min)
end

function _parabolic_offset(v::AbstractVector{<:Real}, i::Integer)
    (i == 1 || i == length(v)) && return 0.0
    y0, y1, y2 = float(v[i - 1]), float(v[i]), float(v[i + 1])
    denom = y0 - 2y1 + y2
    denom == 0 && return 0.0
    return clamp(0.5 * (y0 - y2) / denom, -0.5, 0.5)
end

function _parabolic_height(v::AbstractVector{<:Real}, i::Integer, δ::Real)
    (i == 1 || i == length(v)) && return float(v[i])
    y0, y1, y2 = float(v[i - 1]), float(v[i]), float(v[i + 1])
    return y1 - 0.25 * (y0 - y2) * δ
end

# =============================================================================
# TIME DOMAIN
#
# No FFT needed. Cheap, no stationarity assumption, responds to SHAPE rather
# than periodic content. A spike is a poor sinusoid and a good discontinuity —
# a spectrum describes it badly, line length describes it well.
# =============================================================================

"""
    derivative(x, fs; order = 1)

Numerical time derivative, µV/s (order 2 gives µV/s²).

Interior points use the central difference (x[i+1] − x[i−1])·fs/2, which is
second-order accurate because the first-order error terms of the forward and
backward differences cancel. Endpoints use second-order ONE-SIDED formulae
rather than naive differences, so accuracy is uniform across the vector —
otherwise the two endpoints are quietly worse and any statistic over the whole
derivative inherits that.

WARNING FOR EEG. Differentiating multiplies each component by 2πf, so 100 Hz
noise is amplified 100× more than 1 Hz signal. On a raw trace with muscle
artefact and mains hum the derivative can be pure noise. Low-pass first
(`isolate_freq(x, fs, 0.5, 40)`). For order 2 the amplification goes as (2πf)².
"""
function derivative(x::AbstractVector{<:Real}, fs::Real; order::Integer = 1)
    n = length(x)
    order in (1, 2) || throw(ArgumentError("order must be 1 or 2, got $order"))
    n >= 3 || throw(ArgumentError("need at least 3 samples, got $n"))
    fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))

    h = 1 / float(fs)
    d = zeros(Float64, n)

    if order == 1
        for i in 2:(n - 1)
            d[i] = (x[i + 1] - x[i - 1]) / (2h)
        end
        d[1] = (-3x[1] + 4x[2] - x[3]) / (2h)
        d[n] = (3x[n] - 4x[n - 1] + x[n - 2]) / (2h)
    else
        for i in 2:(n - 1)
            d[i] = (x[i + 1] - 2x[i] + x[i - 1]) / h^2
        end
        # Conservative: the true one-sided second-derivative stencil needs four
        # points and is badly noise-sensitive. Endpoint curvature is rarely the
        # quantity of interest.
        d[1] = d[2]
        d[n] = d[n - 1]
    end

    return d
end

"""
    energy(x, fs)

∫x²dt ≈ Σx[j]²/fs. Units µV²·s.

WHY THE SQUARE IS NOT OPTIONAL. The obvious instinct — integrate the voltage —
is worthless for EEG, and worth understanding why. A neural signal is
approximately zero-mean: the trace spends as much time below baseline as above,
because what is measured is a fluctuation about a reference, not an absolute
quantity. So ∫V dt ≈ 0 for a resting trace AND ≈ 0 for a violent discharge. The
measure cannot tell them apart, because the positive and negative excursions
cancel exactly the deflections you care about.

Squaring destroys the sign. Every excursion contributes positively, and large
ones contribute disproportionately — double the amplitude, quadruple the energy.
That is what a seizure detector needs: an ictal discharge is not a slightly
busier resting trace and should not be reported as one.

The name is by analogy with physics (∫V²dt across a 1 Ω resistor). There is no
resistor and the units are not joules — it is a shape measure borrowing the
algebra.

Ties to the frequency domain: energy/duration == power(x) == Σ band_power over
the whole spectrum. If those ever disagree, a scaling factor is wrong.
"""
function energy(x::AbstractVector{<:Real}, fs::Real)
    fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
    return sum(abs2, x) / float(fs)
end

"""
    power(x)

Mean square, Σx²/n. Units µV². No fs needed — averaging over evenly spaced
samples is averaging over time, and the 1/fs cancels against the duration.

Unlike energy this does not grow with epoch length, so it is the right choice
when comparing epochs of different durations. Use energy when the total of a
discrete event matters and the event defines its own extent. For a zero-mean
signal this equals the variance.
"""
power(x::AbstractVector{<:Real}) = sum(abs2, x) / length(x)

"""
    rms(x)

√power(x). Units µV — same as the signal, which is the point of the root.

The amplitude a constant-magnitude signal would need to carry the same power.
For a sinusoid of amplitude A it is A/√2, from the mean of cos² being 1/2. That
factor belongs to the sinusoid, not to RMS: a square wave of amplitude A has RMS
exactly A.
"""
rms(x::AbstractVector{<:Real}) = sqrt(power(x))

"""
    rectified_area(x, fs)

∫|x|dt ≈ Σ|x[j]|/fs. Units µV·s.

The other way to defeat the cancellation problem: absolute value instead of
square. The difference is outlier weighting — energy weights quadratically so
one large spike dominates, rectified area weights linearly so it responds more
to sustained elevation than to isolated events.

Neither is universally better. Energy for a large discrete discharge, rectified
area for a diffuse amplitude change, both when unsure — their RATIO is itself
informative about whether a change is peaky or broad.

For a sinusoid of amplitude A over duration T this equals 2AT/π, since the mean
of |sin| is 2/π.
"""
function rectified_area(x::AbstractVector{<:Real}, fs::Real)
    fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
    return sum(abs, x) / float(fs)
end

"""
    line_length(x; normalize = false)

Σ|x[j+1] − x[j]| — total vertical distance the trace travels. Units µV.

WHY IT EARNS ITS PLACE. Line length is arc length, and it responds to amplitude
and frequency SIMULTANEOUSLY. Double the amplitude, it doubles; double the
frequency, it doubles too, because the trace covers the same vertical distance
twice as often. Since a seizure typically involves both, line length catches
both with one subtraction and one absolute value per sample — no transform, no
buffering. That is why it runs in implanted responsive-stimulation devices on a
battery.

CLOSED FORM. For A·sin(2πf₀t), over one period
    ∫|2πf₀A·cos(2πf₀t)|dt = 2πf₀A · 2/(πf₀) = 4A
so n complete periods give exactly 4An, INDEPENDENT of both f₀ and fs.

THE DISCRETISATION BIAS. The sum measures chords; the integral measures arc. A
chord is always shorter, so discrete line length systematically UNDERESTIMATES,
and the shortfall grows as sampling coarsens: ~0.45% low at 40 samples/cycle,
~5.0% low at 10. This is a bias, not noise — it does not average away.

Practical consequence: line length is only comparable across recordings at the
SAME sampling rate. Never compare a 256 Hz clinical value to a 1 kHz research
value without resampling first.
"""
function line_length(x::AbstractVector{<:Real}; normalize::Bool = false)
    n = length(x)
    n >= 2 || return 0.0

    total = 0.0
    for i in 2:n
        total += abs(x[i] - x[i - 1])
    end

    return normalize ? total / (n - 1) : total
end

# Sliding-window framing, shared by the windowed measures so an off-by-one is a
# single bug rather than one per measure. `win` in seconds, `overlap` in [0,1).
#
# Returned times are window CENTRES: a value summarising 1.0–2.0 s belongs at
# 1.5 s, and labelling it 1.0 s shifts every detected event half a window early.
#
# THE TRADE-OFF. Window length sets time/frequency resolution directly. Short
# windows localise sharply but average too little data, so output is noisy. Long
# windows are smooth but cannot say WHEN within themselves something started.
# For seizure onset, 0.5–2 s at 50% overlap is the usual compromise.
#
# Overlap adds no information — it interpolates. Consecutive outputs become
# correlated, so do not treat overlapping windows as independent samples in a
# statistical test.
function _sliding(f::Function, x::AbstractVector, fs::Real,
                  win::Real, overlap::Real)
    fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
    0 <= overlap < 1 || throw(ArgumentError(
        "overlap must be in [0, 1); got $overlap (1.0 would never advance)"))

    nwin = round(Int, win * fs)
    nwin >= 2 || throw(ArgumentError(
        "window of $win s at $fs Hz is $nwin samples; need at least 2"))
    nwin <= length(x) || throw(ArgumentError(
        "window of $nwin samples exceeds the $(length(x))-sample signal"))

    step = max(1, round(Int, nwin * (1 - overlap)))
    starts = 1:step:(length(x) - nwin + 1)

    times = [(s - 1 + (nwin - 1) / 2) / float(fs) for s in starts]
    values = [f(view(x, s:(s + nwin - 1))) for s in starts]

    return times, values
end

"""
    windowed_energy(x, fs; win = 1.0, overlap = 0.5) -> (times, values)

Energy over a sliding window. Times are window centres in seconds, so the output
plots directly against the raw trace with no shift. A discharge appears as a step
change. All windows share a duration, so values are directly comparable — divide
by `win` to read µV² instead.
"""
windowed_energy(x::AbstractVector{<:Real}, fs::Real;
                win::Real = 1.0, overlap::Real = 0.5) =
    _sliding(seg -> energy(seg, fs), x, fs, win, overlap)

"""
    windowed_line_length(x, fs; win = 1.0, overlap = 0.5, normalize = false)

Line length over a sliding window — the classic seizure-detection front end.

Compare each window against a running baseline from a preceding reference
period and flag sustained excursions above a MULTIPLE of that baseline. The
ratio matters rather than an absolute threshold, because absolute line length
varies between patients, between electrodes, and with impedance drift over a
long recording — a fixed threshold that works on day one of a monitoring
admission will not work on day three.
"""
windowed_line_length(x::AbstractVector{<:Real}, fs::Real;
                     win::Real = 1.0, overlap::Real = 0.5,
                     normalize::Bool = false) =
    _sliding(seg -> line_length(seg; normalize = normalize), x, fs, win, overlap)

    # Split x into overlapping windows and transform each. Segmenting is not an
# optimisation here — it is what makes coherence DEFINED. With one segment the
# numerator and denominator of coherence are algebraically equal and the answer
# is 1.0 at every frequency regardless of the data.
function _segment_ffts(x::AbstractVector{<:Real}, fs::Real, win::Real,
                       overlap::Real, window::Symbol)
    nwin = round(Int, win * fs)
    nwin >= 4 || throw(ArgumentError("window of $win s at $fs Hz is only $nwin samples"))
    nwin <= length(x) || throw(ArgumentError("window longer than signal"))
    0 <= overlap < 1 || throw(ArgumentError("overlap must be in [0, 1)"))
 
    step = max(1, round(Int, nwin * (1 - overlap)))
    starts = collect(1:step:(length(x) - nwin + 1))
    w = get_window(window, nwin)
 
    X = Matrix{ComplexF64}(undef, nwin ÷ 2 + 1, length(starts))
    for (m, s) in enumerate(starts)
        seg = float.(collect(view(x, s:(s + nwin - 1))))
        seg .-= mean(seg)
        X[:, m] = rfft(seg .* w)
    end
    return get_freq(nwin, fs), X
end
 
"""
    cross_spectrum(x, y, fs; win, overlap, window)
 
Per-segment Sxy[k,m] = X[k,m]·conj(Y[k,m]). Magnitude is the product of the two
amplitudes; ANGLE is the phase lead of x over y. Every measure below is a
different way of summarising that angle across segments.
"""
function cross_spectrum(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                        fs::Real; win::Real = 1.0, overlap::Real = 0.5,
                        window::Symbol = :hann)
    length(x) == length(y) || throw(DimensionMismatch("signals differ in length"))
    f, X = _segment_ffts(x, fs, win, overlap, window)
    _, Y = _segment_ffts(y, fs, win, overlap, window)
    return f, X .* conj.(Y)
end
 
"""
    coherence(x, y, fs; win, overlap, window)
 
Magnitude-squared coherence in [0,1]. CANNOT distinguish interaction from volume
conduction: one source seen at two electrodes gives 1.0 with zero lag, and since
that is the default situation on the scalp, high coherence between nearby
electrodes is the expected null result rather than a finding.
"""
function coherence(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                   fs::Real; win::Real = 1.0, overlap::Real = 0.5,
                   window::Symbol = :hann)
    f, X = _segment_ffts(x, fs, win, overlap, window)
    _, Y = _segment_ffts(y, fs, win, overlap, window)
    size(X, 2) >= 4 || throw(ArgumentError(
        "coherence needs ≥ 4 segments (got $(size(X,2))); it is identically 1 " *
        "for a single segment"))
    Sxy = vec(mean(X .* conj.(Y), dims = 2))
    Sxx = vec(mean(abs2.(X), dims = 2))
    Syy = vec(mean(abs2.(Y), dims = 2))
    return f, abs2.(Sxy) ./ (Sxx .* Syy)
end
 
"""
    imag_coherency(x, y, fs; kwargs...)
 
imag(⟨Sxy⟩)/√(⟨Sxx⟩⟨Syy⟩), in [-1,1]. Volume conduction is instantaneous and so
contributes only to the REAL part — discarding it removes the confound entirely.
Sign gives direction: positive means x leads y. Conservative by construction: a
real interaction at near-zero lag is suppressed along with the artefact.
"""
function imag_coherency(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                        fs::Real; win::Real = 1.0, overlap::Real = 0.5,
                        window::Symbol = :hann)
    f, X = _segment_ffts(x, fs, win, overlap, window)
    _, Y = _segment_ffts(y, fs, win, overlap, window)
    size(X, 2) >= 4 || throw(ArgumentError("needs ≥ 4 segments"))
    Sxy = vec(mean(X .* conj.(Y), dims = 2))
    Sxx = vec(mean(abs2.(X), dims = 2))
    Syy = vec(mean(abs2.(Y), dims = 2))
    return f, imag.(Sxy) ./ sqrt.(Sxx .* Syy)
end
 
"""
    phase_lag_index(x, y, fs; kwargs...)
 
|⟨sign(imag(Sxy))⟩| in [0,1]. Asks one question per segment — did x lead y? —
and measures how consistently the answer repeats. Magnitude is discarded, so a
single huge artefactual segment counts no more than any other. Zero-lag and
π-lag give imag(Sxy) ≈ 0, so their sign is noise and PLI averages toward 0.
"""
function phase_lag_index(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                         fs::Real; win::Real = 1.0, overlap::Real = 0.5,
                         window::Symbol = :hann)
    f, Sxy = cross_spectrum(x, y, fs; win=win, overlap=overlap, window=window)
    size(Sxy, 2) >= 4 || throw(ArgumentError("needs ≥ 4 segments"))
    return f, abs.(vec(mean(sign.(imag.(Sxy)), dims = 2)))
end
 
"""
    weighted_phase_lag_index(x, y, fs; kwargs...)
 
|⟨imag(Sxy)⟩| / ⟨|imag(Sxy)|⟩ in [0,1]. Same idea as PLI but each segment is
weighted by how far from zero its imaginary part is, so segments near zero lag
(where the sign is noise) contribute little. Removes PLI's discontinuity and is
both less biased and lower variance. Report this one.
"""
function weighted_phase_lag_index(x::AbstractVector{<:Real},
                                  y::AbstractVector{<:Real}, fs::Real;
                                  win::Real = 1.0, overlap::Real = 0.5,
                                  window::Symbol = :hann)
    f, Sxy = cross_spectrum(x, y, fs; win=win, overlap=overlap, window=window)
    size(Sxy, 2) >= 4 || throw(ArgumentError("needs ≥ 4 segments"))
    im_part = imag.(Sxy)
    num = abs.(vec(mean(im_part, dims = 2)))
    den = vec(mean(abs.(im_part), dims = 2))
    # 0/0 where there is no imaginary component at all: report 0, not NaN.
    return f, [d > 0 ? n / d : 0.0 for (n, d) in zip(num, den)]
end

end # module SignalAnalysis