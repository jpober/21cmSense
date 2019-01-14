from __future__ import division
from builtins import str
from builtins import range
import aipy as aipy
import numpy as np


class AntennaArray(aipy.pol.AntennaArray):
    def __init__(self, *args, **kwargs):
        aipy.pol.AntennaArray.__init__(self, *args, **kwargs)
        self.array_params = {}

    def get_ant_params(self, ant_prms={"*": "*"}):
        prms = aipy.fit.AntennaArray.get_params(self, ant_prms)
        for k in ant_prms:
            top_pos = np.dot(self._eq2zen, self[int(k)].pos)
            if ant_prms[k] == "*":
                prms[k].update(
                    {"top_x": top_pos[0], "top_y": top_pos[1], "top_z": top_pos[2]}
                )
            else:
                for val in ant_prms[k]:
                    if val == "top_x":
                        prms[k]["top_x"] = top_pos[0]
                    elif val == "top_y":
                        prms[k]["top_y"] = top_pos[1]
                    elif val == "top_z":
                        prms[k]["top_z"] = top_pos[2]
        return prms

    def set_ant_params(self, prms):
        changed = aipy.fit.AntennaArray.set_params(self, prms)
        for i, ant in enumerate(self):
            ant_changed = False
            top_pos = np.dot(self._eq2zen, ant.pos)
            try:
                top_pos[0] = prms[str(i)]["top_x"]
                ant_changed = True
            except (KeyError):
                pass
            try:
                top_pos[1] = prms[str(i)]["top_y"]
                ant_changed = True
            except (KeyError):
                pass
            try:
                top_pos[2] = prms[str(i)]["top_z"]
                ant_changed = True
            except (KeyError):
                pass
            if ant_changed:
                ant.pos = np.dot(np.linalg.inv(self._eq2zen), top_pos)
            changed |= ant_changed
        return changed

    def get_arr_params(self):
        return self.array_params

    def set_arr_params(self, prms):
        for param in prms:
            self.array_params[param] = prms[param]
            if param == "dish_size_in_lambda":
                FWHM = 2.35 * (0.45 / prms[param])  # radians
                self.array_params["obs_duration"] = (
                    60.0 * FWHM / (15.0 * aipy.const.deg)
                )  # minutes it takes the sky to drift through beam FWHM
            if param == "antpos":
                bl_lens = np.sum(np.array(prms[param]) ** 2, axis=1) ** 0.5
        return self.array_params


def get_aa(freqs, prms):
    """Return the AntennaArray to be used for simulation."""
    antennas = []
    nants = len(prms["antpos"])
    for i in range(nants):
        beam = prms["beam"](
            freqs,
            xwidth=(0.45 / prms["dish_size_in_lambda"]),
            ywidth=(0.45 / prms["dish_size_in_lambda"]),
        )  # as it stands, the size of the beam as defined here is not actually used anywhere in this package, but is a necessary parameter for the aipy Beam2DGaussian object
        antennas.append(aipy.fit.Antenna(0, 0, 0, beam))
    aa = AntennaArray(prms["loc"], antennas)
    p = {}
    for i in range(nants):
        top_pos = prms["antpos"][i]
        p[str(i)] = {"top_x": top_pos[0], "top_y": top_pos[1], "top_z": top_pos[2]}
    aa.set_ant_params(p)
    aa.set_arr_params(prms)
    return aa


def get_catalog(*args, **kwargs):
    return aipy.src.get_catalog(*args, **kwargs)
