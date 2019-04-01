import numpy as np


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
        self.dm = np.random.uniform(low=self.dm_range[0],
                                    high=self.dm_range[1],
                                    size=(1, num_frb))

        # compute the dispersion delay per channel
        # (eq. 5.1 of Lorimer & Kramer, 2005)
        delta_t = np.abs(np.matmul(4.15e6 * (self.f_ref**-2 - self.f_chan**-2),
                                   self.dm))

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

            for j in range(self.num_chan):
                sample_mid = (t_start + delta_t[self.num_chan - 1 - j][i])\
                             / self.t_bin_width
                sample_lo = int(np.round(sample_mid - len(pulse) / 2))
                sample_hi = int(np.round(sample_mid + len(pulse) / 2))
                k = 0
                for sample in range(sample_lo, sample_hi):
                    if sample >= 0 and sample < self.num_samp:
                        spec[self.num_chan - 1 - j][sample] += pulse[k][i]
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

