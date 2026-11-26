# Architecture

The CLI has one small, local-only path:

~~~mermaid
flowchart LR
    CLI[CLI arguments] --> Demo[Demo pipeline]
    Demo --> Synthetic[Generated image]
    Synthetic --> Ops[Image operations]
    Ops --> PNG[PNG artifacts]
    Ops --> Sheet[Contact sheet]
    PNG --> Manifest[Local SHA-256 manifest]
    Sheet --> Manifest
~~~

`cli.py` validates command-line values and delegates to `demo.py`. The demo creates a deterministic image from a seed, applies functions from `operations.py`, and writes PNGs plus `manifest.json` to the selected local directory. The destination must be new or empty; an existing directory with files is left untouched.

The Fourier low-pass operation converts the generated image to grayscale, computes a two-dimensional FFT, applies a circular frequency mask centered on zero frequency, and converts the inverse transform back to an 8-bit image. The cutoff is bounded by the CLI before any artifact is written.

There are no network calls, user-image inputs, model weights, training steps, or external data sources in the demo path.
