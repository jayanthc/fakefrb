# fakefrb

Program to generate fake fast radio bursts (FRBs) for training machine learning models.

_This program is currently under development, and is expected to go through various revisions, including changes to usage._

__Usage example:__ Generate 512 FRBs: `python fakefrb -n 512 -o frbs.npz`

The output dynamic spectra, along with metadata used for generation, is stored in a compressed `.npz` file.

Detailed usage information can be obtained by running `python fakefrb -h`, which outputs the following:

```
usage: fakefrb.py [-h] [-n NUM_FRB] [-c NUM_CHAN] [-i NUM_SAMP] [-f FC]
                  [-b BW] [-t T_BIN_WIDTH] [-d DM_LOWER] [-D DM_UPPER]
                  [-w WIDTH_LOWER] [-W WIDTH_UPPER] [-s SNR_LOWER]
                  [-S SNR_UPPER] -o OUT_FILE [-g]

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_FRB, --num_frb NUM_FRB
                        number of FRBs to be generated (default: 1)
  -c NUM_CHAN, --num_chan NUM_CHAN
                        number of channels (default: 512)
  -i NUM_SAMP, --num_samp NUM_SAMP
                        number of time samples (default: 2048)
  -f FC, --fc FC        centre frequency in GHz (default: 1.44)
  -b BW, --bw BW        bandwidth in MHz (default: 100.0)
  -t T_BIN_WIDTH, --t_bin_width T_BIN_WIDTH
                        bin width in time, in ms (default: 0.064)
  -d DM_LOWER, --dm_lower DM_LOWER
                        dispersion measure, lower limit, in cm^-3 pc (default:
                        10)
  -D DM_UPPER, --dm_upper DM_UPPER
                        dispersion measure, upper limit, in cm^-3 pc (default:
                        3000)
  -w WIDTH_LOWER, --width_lower WIDTH_LOWER
                        pulse width, lower limit, in ms (default: 0.01)
  -W WIDTH_UPPER, --width_upper WIDTH_UPPER
                        pulse width, upper limit, in ms (default: 30)
  -s SNR_LOWER, --snr_lower SNR_LOWER
                        SNR, lower limit (default: 1)
  -S SNR_UPPER, --snr_upper SNR_UPPER
                        SNR, upper limit (default: 250)
  -o OUT_FILE, --out_file OUT_FILE
                        output file name (*.npz)
  -g, --graphics        plot a few sample FRBs
```

__Screenshot:__

![alt text](https://github.com/jayanthc/fakefrb/blob/master/screenshots/screenshot.png "Screenshot")

For ML training, the default options should create a file with the given number of FRBs forming the positive class. For the negative classes, the following are prescribed:

- Generate a file with 0 DM pulses
- Generate noise (can be done on the fly during training, using a generator)

__Reading the output:__

```
In [1]: import numpy as np

In [2]: frbs = np.load('foo.npz')

In [3]: frbs['specs'].shape     # 6 dynamic spectra with 512 channels and 2048 time samples
Out[3]: (6, 512, 2048)

In [4]: frbs['dm']
Out[4]:
array([[1838.17838998, 2337.37254581, 2834.51798503,  805.58037753,
        1513.67153543,  623.6867908 ]])
```

## TODO

The following needs to be modelled in the synthesis:

- Frequency-dependent flux variation (spectral index + distortions).
- Frequency-dependent pulse-broadening due to scattering.
- Negative classes (noise-only, noise + various kinds of RFI).
- Positive class with more distortions (e.g., FRB + RFI).

Created by Jayanth Chennamangalam.
