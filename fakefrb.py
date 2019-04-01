import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import frb_generator


# input validation functions
def is_width_valid(args):
    '''
    Validate pulse width input.
    '''
    if args.width_lower <= 0.0 or args.width_upper <= 0.0:
        sys.stderr.write('ERROR: Pulse width is negative!\n')
        return False

    if args.width_lower > args.width_upper:
        sys.stderr.write('ERROR: Pulse width lower limit {} is greater than '
                         'upper limit {}!\n'
                         .format(args.width_lower, args.width_upper))
        return False

    # we would like the pulse to be visible within our dynamic spectrum window,
    # and for that, its width needs to be less than some fraction of the width
    # of the window (i.e., number of time samples).

    # compute per channel width
    width_lim_pct = 1.5
    if args.width_upper / args.t_bin_width > args.num_samp * width_lim_pct:
        sys.stderr.write('ERROR: Pulse width upper limit {} ms ({} samples at '
                         'a time bin width of {} ms) is greater than {:.0%} '
                         'of number of samples ({})!\n'
                         .format(args.width_upper,
                                 args.width_upper / args.t_bin_width,
                                 args.t_bin_width,
                                 width_lim_pct,
                                 args.num_samp))
        return False

    return True


def is_dm_valid(args):
    '''
    Validate DM input.
    '''
    if args.dm_lower > args.dm_upper:
        sys.stderr.write('ERROR: DM lower limit {} is greater than upper '
                         'limit {}!\n'
                         .format(args.dm_lower, args.dm_upper))
        return False

    # reference frequency, taken as the centre frequency of the highest
    # frequency channel
    f_ref = args.fc * 1e3 + args.bw / 2
    # centre frequency of lowest frequency channel
    f_chan = args.fc * 1e3 - args.bw / 2

    # compute the highest dispersion delay per channel
    # (eq. 5.1 of Lorimer & Kramer, 2005)
    delta_t_upper = np.abs(4.15e6 * (f_ref**-2 - f_chan**-2) * args.dm_upper)

    dm_lim_pct = 0.10
    if (args.num_samp * args.t_bin_width) / delta_t_upper < dm_lim_pct:
        sys.stderr.write('ERROR: DM upper limit {} cm^-3 pc is too large for '
                         'the number of samples {}!\n'
                         .format(args.dm_upper, args.num_samp))
        return False

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
                        default=2048,
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
    if not is_width_valid(args):
        sys.exit(1)
    if not is_dm_valid(args):
        sys.exit(1)

    frb_gen = frb_generator.FRBGenerator(args.num_chan,
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
    if args.num_frb <= num_plot:
        indices = np.arange(args.num_frb)
    else:
        indices = np.sort(np.random.choice(np.arange(args.num_frb),
                                           size=num_plot,
                                           replace=False))
    plt.figure(figsize=(12, 7))
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i + 1)
        plt.imshow(frb_gen.specs[idx], origin='lower', aspect='auto')
        plt.title('idx = {}\nd = {:.3f}\nw = {:.3f}\ns = {:.3f}'
                  .format(idx,
                          frb_gen.dm[0, idx],
                          frb_gen.width[idx],
                          frb_gen.snr[idx]))
    plt.tight_layout()
    plt.show()
