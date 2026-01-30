# -*- coding: utf-8 -*-
from dataclasses import dataclass

@dataclass
class ThresholdOutputs:
    thr_pos: object
    thr_neg: object
    is_abnormal: object

class ThresholdMethod:
    name = "base"
    def compute(self, *, train_X, train_zp, train_zn, query_X, D_all,
                idx_train_mask, idx_val_mask, taus, cfg, device=None):
        raise NotImplementedError
