import sys
import argparse
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns


class FRBGenerator:
    def __init__(self,
                 num_chan,
                 num_samp,
                 fc,
                 bw,
                 t_bin_width,
                 dm_range,
                 width_range,
                 snr_range):
        self.num_chan = num_chan
        self.num_samp = num_samp
        self.fc = fc * 1e3      # convert to MHz
        self.bw = bw
        self.t_bin_width = t_bin_width
        self.dm_range = dm_range
        self.width_range = width_range
        self.snr_range = snr_range

        # reference frequency, taken as the centre frequency of the highest
        # frequency channel
        self.f_ref = self.fc + (self.bw / 2)
        # centre frequencies of each channel
        self.f_chan = np.linspace(self.f_ref - bw, self.f_ref, self.num_chan)\
            .reshape((self.num_chan, 1))

    def generate(self, num_frb):
        # generate Gaussian random noise as background
        self.specs = np.random.normal(loc=0.0,
                                      scale=1.0,
                                      size=(num_frb,
                                            self.num_chan,
                                            self.num_samp))

        # generate random DMs
        dm = np.random.uniform(low=self.dm_range[0],
                               high=self.dm_range[1],
                               size=(1, num_frb))
        print('DM =', dm)

        # compute the dispersion delay per channel
        # (eq. 5.1 of Lorimer & Kramer, 2005)
        delta_t = np.abs(np.matmul(4.15e6 * (self.f_ref**-2 - self.f_chan**-2),
                                   dm))
        print(delta_t.shape)
        print(delta_t[0, :],  delta_t[-1, :])

        # generate Gaussian pulses
        pulse = self._generate_pulses(self.width_range,
                                      self.snr_range,
                                      self.t_bin_width,
                                      num_frb)

        # generate pulse and add it to the background
        for i, spec in enumerate(self.specs):
            t_start = np.random.uniform(low=-delta_t[0][i] / 2,
                                        high=self.num_samp - delta_t[0][i] / 2,
                                        size=1)[0]
            print(t_start)
            for j in range(self.num_chan):
                sample_lo = int(np.round(t_start
                                         + delta_t[self.num_chan - 1 - j][i]
                                         - len(pulse) / 2))
                sample_hi = int(np.round(t_start
                                         + delta_t[self.num_chan - 1 - j][i]
                                         + len(pulse) / 2))
                k = 0
                for sample in range(sample_lo, sample_hi):
                    if sample >= 0 and sample < self.num_samp:
                        spec[self.num_chan - 1 - j][sample] += pulse[k][i]
                    k += 1
                    if k == len(pulse):
                        assert True

    def _generate_pulses(self, width_range, snr_range, t_bin_width, num_frb):
        # convert width (full width at half max.) to standard deviation
        std_range = width_range / (2 * np.sqrt(2 * np.log(2)))
        std = np.random.uniform(low=std_range[0],
                                high=std_range[1],
                                size=num_frb)
        print('std =', std)

        snr = np.random.uniform(low=snr_range[0],
                                high=snr_range[1],
                                size=num_frb)
        print('snr =', snr)

        x_hi = 6 * std_range[1]
        x_lo = -x_hi
        x = np.linspace(x_lo, x_hi, 2 * std_range[1] / t_bin_width)
        x = x.reshape((len(x), 1))

        z = (x**2 / 2) * std**-2
        pulse = snr * np.exp(-z)

        return pulse


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    # number of FRBs to be generated
    parser.add_argument('-n',
                        '--num_frb',
                        type=int,
                        default=1,
                        help='number of FRBs to be generated')
    # number of channels
    parser.add_argument('-c',
                        '--num_chan',
                        type=int,
                        default=512,
                        help='number of channels')
    # number of time samples
    parser.add_argument('-i',
                        '--num_samp',
                        type=int,
                        default=1024,
                        help='number of time samples')
    # centre frequency in GHz
    parser.add_argument('-f',
                        '--fc',
                        type=float,
                        default=1.4204057517667,
                        help='centre frequency in GHz')
    # bandwidth in MHz
    parser.add_argument('-b',
                        '--bw',
                        type=float,
                        default=100.0,
                        help='bandwidth in MHz')
    # bin width in time, in ms
    parser.add_argument('-t',
                        '--t_bin_width',
                        type=float,
                        default=0.064,
                        help='bin width in time, in ms')
    # dispersion measure, lower limit, in cm^-3 pc
    parser.add_argument('-d',
                        '--dm_lower',
                        type=float,
                        default=10,
                        help='dispersion measure, lower limit, in cm^-3 pc')
    # dispersion measure, upper limit, in cm^-3 pc
    parser.add_argument('-D',
                        '--dm_upper',
                        type=float,
                        default=1000,
                        help='dispersion measure, upper limit, in cm^-3 pc')
    # pulse width, lower limit, in ms
    parser.add_argument('-w',
                        '--width_lower',
                        type=float,
                        default=0.1,
                        help='pulse width, lower limit, in ms')
    # pulse width, upper limit, in ms
    parser.add_argument('-W',
                        '--width_upper',
                        type=float,
                        default=5,
                        help='pulse width, upper limit, in ms')
    # SNR, lower limit
    parser.add_argument('-s',
                        '--snr_lower',
                        type=float,
                        default=3,
                        help='SNR, lower limit')
    # SNR, upper limit
    parser.add_argument('-S',
                        '--snr_upper',
                        type=float,
                        default=30,
                        help='SNR, upper limit')
    args = parser.parse_args()

    # input validation
    if args.width_upper / args.t_bin_width >= args.num_samp * 0.5:
        sys.stderr.write('ERROR: Pulse width upper limit is greater than 50% '
                         'of number of samples!\n')
        sys.exit(1)

    frb_gen = FRBGenerator(args.num_chan,
                           args.num_samp,
                           args.fc,
                           args.bw,
                           args.t_bin_width,
                           (args.dm_lower, args.dm_upper),
                           (args.width_lower, args.width_upper),
                           (args.snr_lower, args.snr_upper))
    frb_gen.generate(args.num_frb)

    # plot a few random dynamic spectra
    num_plot = 6
    indices = np.random.randint(low=0, high=args.num_frb, size=num_plot)
    plt.figure(figsize=(10, 6))
    for i in range(num_plot):
        if i < args.num_frb:
            plt.subplot(2, 3, i + 1)
            plt.imshow(frb_gen.specs[i], origin='lower', aspect='auto')
    plt.tight_layout()
    plt.show()
