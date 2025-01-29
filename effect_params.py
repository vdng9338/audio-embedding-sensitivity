import numpy as np

effect_params_dict = {
    "bitcrush": np.arange(4, 16),
    "gain": np.arange(-40.0, 5.01, 1.5),
    "lowpass_cheby": np.round(np.geomspace(1600, 22000/1.2, num=31), 3),
    "reverb": np.linspace(0.01, 1, num=100)[::3] # so that everything fits in RAM
}


effect_params_str_dict = {
    "bitcrush": [str(i) for i in range(4, 16)],
    "gain": ["{:.1f}".format(g) for g in effect_params_dict["gain"]],
    "lowpass_cheby": ["{:.3f}".format(freq) for freq in effect_params_dict["lowpass_cheby"]],
    "reverb": ["{:.2f}".format(r) for r in effect_params_dict["reverb"]]
}