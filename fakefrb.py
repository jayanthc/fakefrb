import sys
import argparse
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import axisplot as ap
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
        #self.specs = np.zeros((num_frb, self.num_chan, self.num_samp))

        # generate random DMs
        self.dm = np.random.uniform(low=self.dm_range[0],
                                    high=self.dm_range[1],
                                    size=(1, num_frb))
        print('DM =', self.dm)

        # compute the dispersion delay per channel
        # (eq. 5.1 of Lorimer & Kramer, 2005)
        delta_t = np.abs(np.matmul(4.15e6 * (self.f_ref**-2 - self.f_chan**-2),
                                   self.dm))
        print(delta_t.shape)
        print(delta_t[0, :],  delta_t[-1, :])

        # generate Gaussian pulses
        pulse = self.__generate_pulses(self.width_range,
                                       self.snr_range,
                                       self.t_bin_width,
                                       num_frb)

        t_obs = self.num_samp * self.t_bin_width
        for i, spec in enumerate(self.specs):
            t_start = np.random.uniform(low=-delta_t[0, i] * 0.75,
                                        high=t_obs - delta_t[0][i] * 0.25,
                                        size=1)[0]

            # per channel scale factor
            # TODO: make this some sort of polynomial
            #scale = np.random.uniform(low=0.9, high=1.1, size=self.num_chan)
            scale = np.ones((self.num_chan,))

            for j in range(self.num_chan):
                sample_mid = (t_start + delta_t[self.num_chan - 1 - j][i])\
                             / self.t_bin_width
                sample_lo = int(np.round(sample_mid - len(pulse) / 2))
                sample_hi = int(np.round(sample_mid + len(pulse) / 2))
                k = 0
                for sample in range(sample_lo, sample_hi):
                    if sample >= 0 and sample < self.num_samp:
                        spec[self.num_chan - 1 - j][sample]\
                            += scale[j] * pulse[k][i]
                    k += 1
                    if k == len(pulse):
                        assert True

    def __width_to_std(self, width):
        return width / (2 * np.sqrt(2 * np.log(2)))

    def __generate_pulses(self, width_range, snr_range, t_bin_width, num_frb):
        self.width = np.random.uniform(low=width_range[0],
                                       high=width_range[1],
                                       size=num_frb)
        # convert width (full width at half max.) to standard deviation
        std = self.__width_to_std(self.width)
        std_range = self.__width_to_std(width_range)

        # TODO: throw error if per-channel SNR is too low
        self.snr = np.random.uniform(low=snr_range[0],
                                     high=snr_range[1],
                                     size=num_frb)

        x_hi = 6 * std_range[1]
        x_lo = -x_hi
        x0 = np.linspace(x_lo, 0, std_range[1] / t_bin_width)
        x1 = np.linspace(0, x_hi, std_range[1] / t_bin_width)
        x = np.concatenate((x0, x1))
        x = x.reshape((len(x), 1))

        z = (x**2 / 2) * std**-2
        pulse = (self.snr / np.sqrt(self.num_chan)) * np.exp(-z)

        return pulse


# input validation functions
def validate_width(args):
    '''
    Validate pulse width. We would like the pulse to be visible within our
    dynamic spectrum window, and for that, its width needs to be less than some
    fraction of the width of the window (i.e., number of time samples).
    '''
    # compute per channel width
    width_lim_pct = 1.5
    if args.width_upper / args.t_bin_width >= args.num_samp * width_lim_pct:
        sys.stderr.write('ERROR: Pulse width upper limit {} ms ({} samples at '
                         'a time bin width of {} ms) is greater than {:.0%} '
                         'of number of samples ({})!\n'
                         .format(args.width_upper,
                                 args.width_upper / args.t_bin_width,
                                 args.t_bin_width,
                                 width_lim_pct,
                                 args.num_samp))
        return False
    else:
        return True


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    # number of FRBs to be generated
    parser.add_argument('-n',
                        '--num_frb',
                        type=int,
                        default=1,
                        help='number of FRBs to be generated '
                             '(default: %(default)s)')
    # number of channels
    parser.add_argument('-c',
                        '--num_chan',
                        type=int,
                        default=512,
                        help='number of channels '
                             '(default: %(default)s)')
    # number of time samples
    parser.add_argument('-i',
                        '--num_samp',
                        type=int,
                        default=1024,
                        help='number of time samples '
                             '(default: %(default)s)')
    # centre frequency in GHz
    parser.add_argument('-f',
                        '--fc',
                        type=float,
                        default=1.440,
                        help='centre frequency in GHz '
                             '(default: %(default)s)')
    # bandwidth in MHz
    parser.add_argument('-b',
                        '--bw',
                        type=float,
                        default=100.0,
                        help='bandwidth in MHz '
                             '(default: %(default)s)')
    # bin width in time, in ms
    parser.add_argument('-t',
                        '--t_bin_width',
                        type=float,
                        default=0.064,
                        help='bin width in time, in ms '
                             '(default: %(default)s)')
    # dispersion measure, lower limit, in cm^-3 pc
    parser.add_argument('-d',
                        '--dm_lower',
                        type=float,
                        default=10,
                        help='dispersion measure, lower limit, in cm^-3 pc '
                             '(default: %(default)s)')
    # dispersion measure, upper limit, in cm^-3 pc
    parser.add_argument('-D',
                        '--dm_upper',
                        type=float,
                        default=3000,
                        help='dispersion measure, upper limit, in cm^-3 pc '
                             '(default: %(default)s)')
    # pulse width, lower limit, in ms
    parser.add_argument('-w',
                        '--width_lower',
                        type=float,
                        default=0.01,
                        help='pulse width, lower limit, in ms '
                             '(default: %(default)s)')
    # pulse width, upper limit, in ms
    parser.add_argument('-W',
                        '--width_upper',
                        type=float,
                        default=30,
                        help='pulse width, upper limit, in ms '
                             '(default: %(default)s)')
    # SNR, lower limit
    parser.add_argument('-s',
                        '--snr_lower',
                        type=float,
                        default=1,
                        help='SNR, lower limit '
                             '(default: %(default)s)')
    # SNR, upper limit
    parser.add_argument('-S',
                        '--snr_upper',
                        type=float,
                        default=250,
                        help='SNR, upper limit '
                             '(default: %(default)s)')
    args = parser.parse_args()

    # input validation
    if not validate_width(args):
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
    plt.figure(figsize=(12, 7))
    for i in range(num_plot):
        if i < args.num_frb:
            plt.subplot(2, 3, i + 1)
            plt.imshow(frb_gen.specs[i], origin='lower', aspect='auto')
            plt.title('d = {:.3f}\nw = {:.3f}\ns = {:.3f}'
                      .format(frb_gen.dm[0, i],
                              frb_gen.width[i],
                              frb_gen.snr[i]))
    plt.tight_layout()
    plt.show()
