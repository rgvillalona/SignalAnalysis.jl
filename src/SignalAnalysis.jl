module SignalAnalysis

using FFTW
using Statistics: mean
using DSP: digitalfilter, filtfilt, Bandpass, Lowpass, Highpass, Butterworth,
           iirnotch

export PolarSpectrum, PowerSpectrum, resolution, nyquist, nearest_bin

export get_window, get_freq, decomp_freq, welch, to_polar, to_complex, reconstruct,
       isolate_freq, find_peaks, band_power, EEG_BANDS
       
export bandpass, lowpass, highpass, notch

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
`scale` records the normalisation applied to |X| so the transform can be
inverted. `scaling` records WHICH normalisation (:amplitude, :power or :none),
because the right way to read `magnitude` depends on it — see `decomp_freq`.
`single_sided` records whether bins 2:end were doubled, so `to_complex` knows
whether to halve them again.
"""
struct PolarSpectrum{T<:Real}
    freq::Vector{T}
    magnitude::Vector{T}
    phase::Vector{T}
    fs::T
    n::Int
    scale::T
    scaling::Symbol
    single_sided::Bool

    function PolarSpectrum(freq::Vector{T}, magnitude::Vector{T},
                           phase::Vector{T}, fs::T, n::Integer, scale::T;
                           scaling::Symbol = :amplitude,
                           single_sided::Bool = true) where {T<:Real}
        length(freq) == length(magnitude) == length(phase) ||
            throw(DimensionMismatch("freq, magnitude, phase must match in length"))
        fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
        n > 0 || throw(ArgumentError("sample count must be positive, got $n"))
        length(freq) == n ÷ 2 + 1 || throw(DimensionMismatch(
            "an $n-sample real signal gives $(n ÷ 2 + 1) bins, got $(length(freq))"))
        scaling in (:amplitude, :power, :none) || throw(ArgumentError(
            "scaling must be :amplitude, :power or :none, got $scaling"))
        return new{T}(freq, magnitude, phase, fs, Int(n), scale, scaling,
                      single_sided)
    end
end

"""
    PowerSpectrum{T<:Real}

One-sided power spectral DENSITY from `welch`, in squared voltage per Hz
(µV²/Hz). `freq` and `psd` are the axis and values. `n` is the SEGMENT length in
samples, which sets the resolution exactly as in `PolarSpectrum`; `nseg` is how
many segments were averaged, which sets the variance.

Density, not power: the value at a bin is power per Hz, so it is comparable
across different segment lengths, and integrating it (Σ psd·Δf) gives power in
µV². `band_power` does that integration.
"""
struct PowerSpectrum{T<:Real}
    freq::Vector{T}
    psd::Vector{T}
    fs::T
    n::Int
    nseg::Int

    function PowerSpectrum(freq::Vector{T}, psd::Vector{T}, fs::T, n::Integer,
                           nseg::Integer) where {T<:Real}
        length(freq) == length(psd) ||
            throw(DimensionMismatch("freq and psd must match in length"))
        fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
        n > 0 || throw(ArgumentError("segment length must be positive, got $n"))
        nseg > 0 || throw(ArgumentError("segment count must be positive, got $nseg"))
        length(freq) == n ÷ 2 + 1 || throw(DimensionMismatch(
            "an $n-sample segment gives $(n ÷ 2 + 1) bins, got $(length(freq))"))
        return new{T}(freq, psd, fs, Int(n), Int(nseg))
    end
end

# Both spectrum types carry the same kind of frequency axis, so the axis
# helpers below accept either.
const AbstractSpectrum = Union{PolarSpectrum, PowerSpectrum}

_spectrum_values(ps::PolarSpectrum) = ps.magnitude
_spectrum_values(ps::PowerSpectrum) = ps.psd

Base.length(ps::AbstractSpectrum) = length(ps.freq)
Base.getindex(ps::PolarSpectrum, k::Integer) =
    (freq = ps.freq[k], magnitude = ps.magnitude[k], phase = ps.phase[k])
Base.getindex(ps::PowerSpectrum, k::Integer) = (freq = ps.freq[k], psd = ps.psd[k])

"""
    resolution(ps)

Bin spacing Δf = fs/n = 1/T, in Hz.

Set entirely by epoch DURATION, not sampling rate. A 2-second epoch gives 0.5 Hz
whether sampled at 256 Hz or 2 kHz — only a longer epoch helps, and only if the
signal is stationary across it. To separate 3 Hz from 2.75 Hz you need ≥4 s.

For a `PowerSpectrum` the relevant duration is the SEGMENT, not the record: a
30 s record analysed in 2 s segments has 0.5 Hz resolution, not 0.033 Hz.
"""
resolution(ps::AbstractSpectrum) = ps.fs / ps.n

"""
    nyquist(ps)

fs/2. Content above this is not lost, it is ALIASED — folded down onto lower
frequencies where it is indistinguishable from real content. The anti-aliasing
filter belongs in the amplifier and cannot be fixed here.
"""
nyquist(ps::AbstractSpectrum) = ps.fs / 2

"""
    nearest_bin(ps, f)

Index of the bin centred closest to `f` Hz. Bins sit at k·fs/n.
"""
function nearest_bin(ps::AbstractSpectrum, f::Real)
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
    println(io, "  scaling    : ", ps.scaling)
    if !isempty(ps.magnitude)
        k = argmax(ps.magnitude)
        println(io, "  peak       : ", round(ps.freq[k], digits = 3), " Hz")
    end
end

function Base.show(io::IO, ::MIME"text/plain", ps::PowerSpectrum{T}) where {T}
    println(io, "PowerSpectrum{$T}")
    println(io, "  bins       : ", length(ps))
    println(io, "  segment    : ", ps.n, " samples")
    println(io, "  segments   : ", ps.nseg)
    println(io, "  fs         : ", ps.fs, " Hz")
    println(io, "  resolution : ", round(resolution(ps), digits = 4), " Hz")
    if !isempty(ps.psd)
        k = argmax(ps.psd)
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
    to_polar(X, fs, n; scale = 1.0, single_sided = true, scaling = :amplitude)

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
                  scale::Real = 1.0, single_sided::Bool = true,
                  scaling::Symbol = :amplitude)
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
                         T(fs), n, T(scale);
                         scaling = scaling, single_sided = single_sided)
end

"""
    to_complex(ps)

Rebuild the raw complex half-spectrum, undoing the doubling (if it was applied)
and the scaling.
"""
function to_complex(ps::PolarSpectrum)
    mag = copy(ps.magnitude)
    nb = length(mag)
    if ps.single_sided && nb > 1
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
    decomp_freq(x, fs; window = :hann, detrend = true, scaling = :amplitude)

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

AMPLITUDE VERSUS POWER SCALING. The derivation above is for a TONE. It divides
by mean(w) because a taper shrinks a sinusoid's peak by exactly that factor.
But power goes as the SQUARE of the window, and mean(w²) ≠ mean(w)²: for Hann
they are 0.375 and 0.25, so amplitude scaling overstates broadband power by
1.5×. No single scale factor is right for both quantities.

  :amplitude  (default) magnitude[k₀] == A for a tone on bin k₀. Read peaks.
  :power      scale = 1/(n·√mean(w²)), so Σ magnitude²/2 == power(x) for any
              window (exactly for :rect, in expectation otherwise). Use this
              whenever you will call `band_power` or compare to time-domain
              power. Peak heights of tones will read √(mean(w²))/mean(w) high —
              1.22× for Hann — which is why it is not the default.
  :none       raw rfft output, no doubling. For inverting.

`PolarSpectrum.scaling` remembers the choice so downstream code can check it.
"""
function decomp_freq(x::AbstractVector{<:Real}, fs::Real;
                     window::Union{Symbol,AbstractVector{<:Real}} = :hann,
                     detrend::Bool = true,
                     scaling::Symbol = :amplitude)
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

    if scaling === :amplitude
        cg = coherent_gain(w)
        cg > 0 || throw(ArgumentError("window has zero mean; cannot correct gain"))
        scale = 1 / (n * cg)
    elseif scaling === :power
        pg = sqrt(mean(abs2, w))
        pg > 0 || throw(ArgumentError("window is all zeros"))
        scale = 1 / (n * pg)
    elseif scaling === :none
        scale = 1.0
    else
        throw(ArgumentError(
            "scaling must be :amplitude, :power or :none, got $scaling"))
    end

    X = rfft(xf .* w)
    return to_polar(X, fs, n; scale = scale, single_sided = scaling !== :none,
                    scaling = scaling)
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
an artefact. Use `bandpass` for anything you will measure or publish.

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
average of cos² over a cycle is 1/2. DC and Nyquist are excluded from the halving
because they are not oscillations: a constant of size A has mean square A².

PARSEVAL, AND WHEN IT HOLDS. Summing A_k²/2 across every bin reproduces
`power(x)` — the strongest single check on the package's scaling — but only if
the spectrum was made with `scaling = :power`. The default `:amplitude` scaling
is right for reading tone heights and overstates broadband power by
mean(w²)/mean(w)², which is 1.5 for Hann. For `:rect` the two scalings coincide
and the identity is exact; for other windows it holds in expectation. With
`detrend = true` the DC bin is zero regardless of the signal's offset, so
compare against the power of the detrended signal.

This function throws if given an `:amplitude` spectrum from a non-rectangular
window, because the number it would return is wrong in a way that is easy to
miss. Build the spectrum with `decomp_freq(x, fs; scaling = :power)`.

For a real recording longer than a few seconds, use `welch` and call
`band_power` on the result instead: a single periodogram is far too noisy to
read a band power from. See the `welch` docstring for why.

Band power scales with bandwidth, so comparing a 0.5 Hz band to a 70 Hz band is
meaningless unless you divide by width or express each as a fraction of total.
"""
function band_power(ps::PolarSpectrum, lo::Real, hi::Real)
    ps.scaling === :power || (ps.scaling === :amplitude && _is_rect_scaled(ps)) ||
        throw(ArgumentError(
            "band_power needs a :power-scaled spectrum (got :$(ps.scaling)); " *
            "rebuild with decomp_freq(x, fs; scaling = :power)"))
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

# With a rectangular window mean(w) == √mean(w²) == 1, so :amplitude scaling is
# already power-correct and scale == 1/n. Anything else means a taper was used.
_is_rect_scaled(ps::PolarSpectrum) = isapprox(ps.scale * ps.n, 1.0; rtol = 1e-9)

# =============================================================================
# FILTERS
#
# Designed IIR filters applied forward and backward. These are what you use on
# data you will MEASURE; `isolate_freq` on a raw signal is for looking only.
# =============================================================================

function _check_cutoff(fs::Real, f::Real, name::AbstractString)
    fs > 0 || throw(ArgumentError("sampling rate must be positive, got $fs"))
    0 < f < fs / 2 || throw(ArgumentError(
        "$name must lie strictly inside (0, fs/2) = (0, $(fs / 2)) Hz, got $f"))
end

function _check_order(order::Integer)
    order >= 1 || throw(ArgumentError("filter order must be ≥ 1, got $order"))
end

"""
    bandpass(x, fs, lo, hi; order = 4)
    lowpass(x, fs, cutoff; order = 4)
    highpass(x, fs, cutoff; order = 4)

Zero-phase Butterworth filter. Returns a new Float64 vector; `x` is untouched.

WHY ZERO-PHASE. Any causal filter delays each frequency by a different amount,
so a spike comes out smeared and SHIFTED — by several samples at 256 Hz, which
is enough to make two channels' onsets disagree about which fired first. Running
the filter forward and then backward over the whole epoch cancels the phase
shift exactly: what goes in at sample j comes out centred on sample j. The price
is that the filter becomes non-causal (it sees the future), so this is an
offline tool. For a real-time detector you accept the delay and compensate.

WHY BUTTERWORTH. Maximally flat passband — no ripple to modulate the amplitude
of the rhythm you are measuring — and a monotonic roll-off, so nothing in the
stopband is accidentally emphasised. The trade is a gentler transition than
Chebyshev or elliptic designs of the same order, which for EEG is the right
trade: a slightly leaky 40 Hz edge costs nothing, 1 dB of passband ripple at
10 Hz does.

ORDER. `order` is the Butterworth order per pass; two passes square the
magnitude response, so the effective roll-off is that of order 2·`order`
(48 dB/octave for the default). Higher orders are steeper but ring longer and
are less numerically stable at very low cutoffs. 4 is the usual compromise.

EDGES. The data is extended by reflection before filtering, so the first and
last few cycles of the LOWEST passband frequency are unreliable. With a 0.5 Hz
high-pass that is the first and last ~2–6 s. Either discard them or filter a
longer record than you need and trim afterwards.

TYPICAL EEG USE.
    y = bandpass(x, fs, 0.5, 40.0)        # clinical band
    y = notch(y, fs, 60.0)                # only if 60 Hz is inside the passband

For a sinusoid inside the passband the output amplitude equals the input
amplitude to within the passband flatness; for one in the stopband it is
attenuated by at least the stopband depth at that frequency.
"""
function bandpass(x::AbstractVector{<:Real}, fs::Real, lo::Real, hi::Real;
                  order::Integer = 4)
    _check_cutoff(fs, lo, "lower cutoff")
    _check_cutoff(fs, hi, "upper cutoff")
    lo < hi || throw(ArgumentError("lo ($lo) must be below hi ($hi)"))
    _check_order(order)
    f = digitalfilter(Bandpass(lo, hi), Butterworth(order); fs = fs)
    return filtfilt(f, float.(collect(x)))
end

function lowpass(x::AbstractVector{<:Real}, fs::Real, cutoff::Real;
                 order::Integer = 4)
    _check_cutoff(fs, cutoff, "cutoff")
    _check_order(order)
    f = digitalfilter(Lowpass(cutoff), Butterworth(order); fs = fs)
    return filtfilt(f, float.(collect(x)))
end

function highpass(x::AbstractVector{<:Real}, fs::Real, cutoff::Real;
                  order::Integer = 4)
    _check_cutoff(fs, cutoff, "cutoff")
    _check_order(order)
    f = digitalfilter(Highpass(cutoff), Butterworth(order); fs = fs)
    return filtfilt(f, float.(collect(x)))
end

# lowpass and highpass share bandpass's docstring.
@doc (@doc bandpass) lowpass
@doc (@doc bandpass) highpass

"""
    notch(x, fs, f0; bw = 1.0)

Zero-phase IIR notch at `f0` Hz with −3 dB bandwidth `bw` Hz. For mains hum:
60 Hz in the Americas, 50 Hz nearly everywhere else.

WHEN YOU NEED IT. Only when `f0` sits inside your passband. A 0.5–40 Hz
bandpass already attenuates 60 Hz by more than 40 dB, so notching after it
buys nothing. If you work up to 70 Hz for gamma, you need the notch — and
usually one at the second harmonic as well (`notch(y, fs, 120.0)`), which is
only possible if fs > 240 Hz.

WHY NARROW. A notch removes everything in its band, including any neural
activity that happens to be there. 1 Hz wide takes out 59.5–60.5 Hz and
nothing else. Widen it only if the hum visibly drifts, which on a stable
mains supply it does not.

WHY NOT JUST ZERO THE BIN. Mains hum is not stationary over a 30 s epoch — its
amplitude wanders with whatever else is on the circuit — so its energy spreads
across several bins and zeroing one leaves most of it. A time-domain notch
tracks the hum sample by sample.
"""
function notch(x::AbstractVector{<:Real}, fs::Real, f0::Real; bw::Real = 1.0)
    _check_cutoff(fs, f0, "notch frequency")
    bw > 0 || throw(ArgumentError("bandwidth must be positive, got $bw"))
    f0 - bw / 2 > 0 || throw(ArgumentError(
        "notch band [$(f0 - bw/2), $(f0 + bw/2)] Hz extends below 0 Hz"))
    f = iirnotch(f0, bw; fs = fs)
    return filtfilt(f, float.(collect(x)))
end

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

function find_peaks(ps::AbstractSpectrum;
                    distance::Union{Nothing,Real} = nothing, kwargs...)
    df = resolution(ps)
    dist_bins = distance === nothing ? nothing : distance / df   # Hz -> bins
    r = find_peaks(_spectrum_values(ps); distance = dist_bins, kwargs...)
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
the whole spectrum, provided the spectrum was built with `scaling = :power` and
`detrend = false` (Parseval). If those disagree, a scaling factor is wrong.
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

# =============================================================================
# WELCH POWER SPECTRAL DENSITY
# =============================================================================

"""
    welch(x, fs; win = 2.0, overlap = 0.5, window = :hann, detrend = true)

Averaged one-sided power spectral density in µV²/Hz, as a `PowerSpectrum`.

WHY ONE FFT IS NOT ENOUGH. The periodogram of a noisy signal is itself noisy:
each bin's estimate has a standard deviation about equal to its mean, and a
LONGER epoch does not help. More samples buy more bins, not better bins. On a
30 s EEG record the raw spectrum is a forest of spikes from which no band power
can be read reliably.

Welch's fix: cut the record into segments, transform each, average the squared
magnitudes. Averaging M roughly independent estimates cuts the scatter by about
√M. The price is resolution — each segment is shorter than the record, so
Δf = 1/win rather than 1/T. For EEG band power that trade is always right:
0.5 Hz from 2 s segments separates the clinical bands comfortably, and a 30 s
record then gives 29 segments at 50% overlap, a √29 ≈ 5× reduction in scatter.

THE NORMALISATION. Per segment |X[k]|² / (fs · Σw²), averaged across segments,
then doubled for k ≥ 2 (except Nyquist) because the negative frequencies are
folded in. Dividing by Σw² rather than n·mean(w)² is what makes this a POWER
estimate under a taper — the same reasoning as `scaling = :power` in
`decomp_freq`. Dividing by fs turns power per bin into power per Hz. Together
they give Σ psd·Δf == power(x) for any window, up to segment-mean removal.

WHY DENSITY. Power per bin depends on how wide the bins are, so two spectra with
different `win` cannot be compared bin for bin. Power per Hz can. It also makes
`band_power` an integral, Σ psd·Δf, which does not move when you change the
resolution — only its noisiness does.

CHOICES. `window = :hann` with `overlap = 0.5` is the standard pairing: Hann
tapers the segment edges to nothing, and 50% overlap puts those edges under the
middle of the neighbouring segment, so no sample is systematically
under-weighted. `detrend = true` removes each segment's mean so slow drift does
not leak into delta; set it false only for the Parseval check.

With `win` equal to the whole record and `window = :rect` this reduces to the
single periodogram with power scaling.
"""
function welch(x::AbstractVector{<:Real}, fs::Real; win::Real = 2.0,
               overlap::Real = 0.5, window::Symbol = :hann,
               detrend::Bool = true)
    f, X = _segment_ffts(x, fs, win, overlap, window; detrend = detrend)
    nwin = round(Int, win * fs)
    w = get_window(window, nwin)
    nseg = size(X, 2)

    psd = vec(mean(abs2.(X), dims = 2)) ./ (fs * sum(abs2, w))

    nb = length(psd)
    if nb > 1
        for k in 2:nb
            psd[k] *= 2
        end
        if iseven(nwin)
            psd[nb] /= 2
        end
    end

    T = promote_type(eltype(psd), eltype(f), typeof(float(fs)))
    return PowerSpectrum(Vector{T}(f), Vector{T}(psd), T(fs), nwin, nseg)
end

"""
    band_power(ps::PowerSpectrum, lo, hi)
    band_power(ps::PowerSpectrum, band::Symbol)

Σ psd·Δf over the bins in [lo, hi]: integrated power in µV². Because `psd` is a
density this is the same quantity `band_power` returns for a `:power`-scaled
`PolarSpectrum`, only far less noisy. This is the band power to report.
"""
function band_power(ps::PowerSpectrum, lo::Real, hi::Real)
    lo <= hi || throw(ArgumentError("lo ($lo) must not exceed hi ($hi)"))
    df = resolution(ps)
    total = 0.0
    for k in eachindex(ps.freq)
        lo <= ps.freq[k] <= hi || continue
        total += ps.psd[k] * df
    end
    return total
end

band_power(ps::PowerSpectrum, band::Symbol) = band_power(ps, _band_limits(band)...)

# =============================================================================
# CONNECTIVITY
# =============================================================================
# Split x into overlapping windows and transform each. Segmenting is not an
# optimisation here — it is what makes coherence DEFINED. With one segment the
# numerator and denominator of coherence are algebraically equal and the answer
# is 1.0 at every frequency regardless of the data.

function _segment_ffts(x::AbstractVector{<:Real}, fs::Real, win::Real,
                       overlap::Real, window::Symbol; detrend::Bool = true)
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
        detrend && (seg .-= mean(seg))
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
                   length(x) == length(y) || throw(DimensionMismatch("signals differ in length"))
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