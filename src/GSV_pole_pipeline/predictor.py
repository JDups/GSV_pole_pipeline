import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Predictor(ABC):
    @abstractmethod
    def predict(self, images):
        pass


class MarginPredictor(Predictor):
    def __init__(self, margin):
        if isinstance(margin, int) or isinstance(margin, float):
            self.mrgs = (margin,) * 4

        elif isinstance(margin, tuple):
            if any(not (isinstance(m, int) or isinstance(m, float)) for m in margin):
                raise TypeError("Margin tuple must contain only integers or floats.")

            if len(margin) == 2:
                self.mrgs = (margin[0],) * 2 + (margin[1],) * 2
            elif len(margin) == 4:
                self.mrgs = margin
            else:
                raise TypeError("Margin tuple must be either of length 2 or 4.")

        else:
            raise TypeError(
                "Margin must be specified as either integer, float, or tuples of those types."
            )

    def get_mask(self, dims):
        marg_msk = np.ones(dims)

        # I tried to be clever but I think this can be done in a better way
        mrg_yx = [
            int(dims[0] * m) if isinstance(m, float) else m for m in self.mrgs[:2]
        ]
        mrg_yx += [
            int(dims[1] * m) if isinstance(m, float) else m for m in self.mrgs[2:]
        ]

        marg_msk[mrg_yx[0] : dims[0] - mrg_yx[1], mrg_yx[2] : dims[1] - mrg_yx[3]] = 0

        return marg_msk

    def predict(self, images):
        return [
            {
                "fn": img["fn"],
                "out": {
                    "class": ["margin"],
                    "mask": [self.get_mask(img["img"].shape[:2])],
                },
            }
            for img in images
        ]


class MockPredictor(Predictor):
    def __init__(self, results):
        self.rslt_df = pd.DataFrame(results)

    def __get_mask_list(self, result):
        if result.masks:
            return list(result.masks.data.cpu().numpy().astype(bool))
        else:
            return []

    def predict(self, images):
        fns = [img["fn"] for img in images]

        return [
            {
                "fn": img["fn"],
                "out": {
                    "class": [
                        img["result"].names[c.item()]
                        for c in img["result"].boxes.cls.cpu()
                    ],
                    "mask": self.__get_mask_list(img["result"]),
                },
                "full": img["result"],
            }
            for img in self.rslt_df[self.rslt_df["fn"].isin(fns)].to_dict(
                orient="records"
            )
        ]


class CombinedPredictor(Predictor):
    def __init__(self, results):
        pass

    def predict(self, images):
        pass
