import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FRBGenerator:
    def __init__(self, num_chan, num_samp, fc, bw, dm_range):
        self.num_chan = num_chan
        self.num_samp = num_samp
        self.fc = fc
        self.bw = bw
        self.dm_range = dm_range

        # reference frequency, taken as the centre frequency of the highest
        # frequency channel
        self.f_ref = self.fc + (self.bw / 2)
        # centre frequencies of each channel
        self.f_chan = np.linspace(self.f_ref - bw, self.f_ref, self.num_chan)\
            .reshape((self.num_chan, 1))

    def generate(self, n):
        # generate Gaussian random noise as background
        self.specs = np.random.normal(loc=0.0,
                                      scale=1.0,
                                      size=(n, self.num_chan, self.num_samp))

        # generate random DMs
        dm = np.random.uniform(low=self.dm_range[0],
                               high=self.dm_range[1],
                               size=(1, n))

        # compute the dispersion delay per channel
        delta_t = np.abs(np.matmul(4.15e6 * (self.f_ref**-2 - self.f_chan**-2),
                                   dm))

        # generate start offsets
        t_start = np.random.randint(low=-self.num_samp // 2,
                                    high=self.num_samp // 2,
                                    size=n)

        # generate pulse and add it to the background
        for i, spec in enumerate(self.specs):
            for j in range(self.num_chan):
                sample_l = t_start[i]\
                    + int(np.round(delta_t[self.num_chan - 1 - j][i])) - 5
                sample_r = t_start[i]\
                    + int(np.round(delta_t[self.num_chan - 1 - j][i])) + 5
                for sample in range(sample_l, sample_r):
                    if sample >= 0 and sample < self.num_samp:
                        spec[self.num_chan - 1 - j][sample] = 10


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    # number of channels
    parser.add_argument('-c',
                        '--num_chan',
                        type=int,
                        default=512,
                        help='number of channels')
    # number of time samples
    parser.add_argument('-s',
                        '--num_samp',
                        type=int,
                        default=1024,
                        help='number of time samples')
    # dispersion measure, lower limit
    parser.add_argument('-d',
                        '--dm_lower',
                        type=float,
                        default=10,
                        help='dispersion measure, lower limit')
    # dispersion measure, upper limit
    parser.add_argument('-D',
                        '--dm_upper',
                        type=float,
                        default=1000,
                        help='dispersion measure, lower limit')
    args = parser.parse_args()

    frb_gen = FRBGenerator(args.num_chan,
                           args.num_samp,
                           100,
                           10,
                           (args.dm_lower, args.dm_upper))
    n_frb = 10
    frb_gen.generate(n_frb)

    # plot a few random dynamic spectra
    indices = np.random.randint(low=0, high=n_frb, size=9)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        if i < n_frb:
            plt.subplot(3, 3, i + 1)
            plt.imshow(frb_gen.specs[i], origin='lower', aspect='auto')
    plt.tight_layout()
    plt.show()
