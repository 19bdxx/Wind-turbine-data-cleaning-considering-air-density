# -*- coding: utf-8 -*-
import numpy as np

class Scaler:
    def __init__(self, method="minmax", fixed=True, wind_range=(0,15), rho_range=(1.07,1.37)):
        self.method = method; self.fixed=fixed
        self.V_min, self.V_max = wind_range
        self.R_min, self.R_max = rho_range
        self.V_mu, self.V_sigma = 0.0, 1.0
        self.R_mu, self.R_sigma = 0.0, 1.0
    def fit_from_train(self, wind, rho, use_rho: bool):
        import pandas as pd
        if self.method == "zscore":
            v = pd.to_numeric(wind, errors="coerce").dropna().values
            self.V_mu = float(v.mean()) if v.size else 0.0
            self.V_sigma = float(v.std()) if v.size else 1.0
            if use_rho:
                r = pd.to_numeric(rho, errors="coerce").dropna().values
                self.R_mu = float(r.mean()) if r.size else 0.0
                self.R_sigma = float(r.std()) if r.size else 1.0
    def transform(self, v_arr, r_arr, use_rho: bool):
        import numpy as np
        if self.method == "zscore":
            v_std = (v_arr - self.V_mu) / max(self.V_sigma,1e-12)
            r_std = ((r_arr - self.R_mu)/max(self.R_sigma,1e-12) if use_rho else None)
            return v_std, r_std
        else:
            v_std = (v_arr - self.V_min) / max(self.V_max - self.V_min,1e-12)
            r_std = ((r_arr - self.R_min) / max(self.R_max - self.R_min,1e-12) if use_rho else None)
            return v_std, r_std
    def scale_ratio_r_over_v(self):
        if self.method == "zscore":
            return float(self.R_sigma / max(self.V_sigma,1e-12))
        return float((self.R_max-self.R_min)/max(self.V_max-self.V_min,1e-12))
