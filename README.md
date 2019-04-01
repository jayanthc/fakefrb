# fakefrb

Program to generate fake fast radio bursts (FRBs) for training machine learning models.

_This program is currently under development, and is expected to go through various revisions, including changes to usage._

__Usage example:__ Generate 512 FRBs: `python fakefrb -n 512 -o frbs.npz`

The output dynamic spectra, along with metadata used for generation, is stored in a compressed `.npz` file.

For more usage information, run `python fakefrb -h`.

## TODO

The following needs to be modelled in the synthesis:

- Frequency-dependent flux variation (spectral index + other distortions).
- Frequency-dependent pulse-broadening due to scattering.
- Negative class (noise-only, noise + RFI).
- Positive class with more distortions (e.g., FRB + RFI).

Created by Jayanth Chennamangalam.
