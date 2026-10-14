# SignalAnalysis.jl

Spectral, time-domain and connectivity measures for EEG epochs, with
docstrings that explain the physics behind each number.

**Pre-1.0.** The API may change between minor versions.

## What it does

- **Spectrum** — `decomp_freq` returns a `PolarSpectrum` (magnitude, phase,
  frequency axis in Hz) with amplitude-correct scaling for the chosen window.
  `band_power`, `isolate_freq`, and `find_peaks` (with prominence and sub-bin
  interpolation) work on it directly in Hz. `welch` gives an averaged power
  spectral density in µV²/Hz, the right input for band power on real records.
- **Filters** — zero-phase Butterworth `bandpass`, `lowpass`, `highpass`, and a
  mains `notch`, via DSP.jl. These are what you use before measuring.
- **Time domain** — `energy`, `power`, `rms`, `rectified_area`, `line_length`,
  `derivative`, and sliding-window versions for seizure-onset detection.
- **Connectivity** — `coherence`, `imag_coherency`, `phase_lag_index`,
  `weighted_phase_lag_index` from Welch-style segment averaging.
- **Multi-channel** — every measure above also accepts a samples × channels
  matrix and returns one result per channel, so a 23-channel record needs no
  loop.

## Install

Not yet registered. From the Julia REPL:

```julia
using Pkg
Pkg.add(url = "https://github.com/rgvillalona/SignalAnalysis.jl")
```

## Example

```julia
using SignalAnalysis

fs = 256.0
t  = (0:4fs-1) ./ fs
x  = 10 .* sin.(2π .* 3 .* t) .+ 2 .* sin.(2π .* 10 .* t)   # µV

x  = bandpass(x, fs, 0.5, 40.0)         # zero-phase, 4th-order Butterworth
ps = decomp_freq(x, fs)                 # Hann window, mean removed
pk = find_peaks(ps; prominence = 1.0, interpolate = true)
pk.freqs                                # ≈ [3.0, 10.0]
pk.heights                              # ≈ [10.0, 2.0]

pp = decomp_freq(x, fs; scaling = :power)
band_power(pp, :alpha)                  # µV², 8–13 Hz, from one FFT
w  = welch(x, fs; win = 2.0)            # averaged PSD, µV²/Hz
band_power(w, :alpha)                   # same quantity, far less noisy

X = randn(256 * 30, 23)                 # 30 s, 23 channels: samples × channels
band_power(welch(X, fs), :alpha)        # 23-element Vector, one per channel
line_length(x)                          # µV
```

## Units

Input in µV and `fs` in Hz gives: energy µV²·s, power µV², rms and spectral
magnitude µV, rectified area µV·s, line length µV, derivative µV/s. Nothing
guesses units.

## Notes

- `isolate_freq` on a raw signal is a brick-wall FFT filter. Fine for looking,
  not for publishing; use `bandpass` for that.
- Two spectrum scalings: `:amplitude` (default) reads tone heights correctly,
  `:power` makes Σ band power == time-domain power. `band_power` refuses an
  amplitude-scaled Hann spectrum rather than return a number 1.5× too high.
  See the `decomp_freq` docstring.

## License

GPL-3.0 — see `LICENSE`. Copyleft: anyone who distributes a modified version
must publish their changes under the same terms.