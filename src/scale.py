import numpy as np
import yaml
from plugin import Plugin


class plugin(Plugin):
    do_scale = False
    probability = 0
    amplitude = 0
    amplitude_max = 2
    amplitude_min = 0.1

    def __init__(self, param_string):
        if len(param_string) > 0:
            params = yaml.safe_load(param_string)
            if params.has_key("probability"):
                self.probability = params["probability"]
            if params.has_key("sample_freq_hz"):
                self.sample_freq_hz = params["sample_freq_hz"]
            if params.has_key("amplitude"):
                self.amplitude_min = params["amplitude"][0]
                self.amplitude_max = params["amplitude"][1]

    def prepare(self):
        self.do_scale = np.random.uniform() < self.probability
        self.amplitude = np.random.uniform(self.amplitude_min,
                                           self.amplitude_max)

    def augment_audio(self, mat):
        if self.do_scale:
            mat2 = (mat * self.amplitude).astype(np.int16)
            return mat2
        else:
            dst = mat
        return dst
