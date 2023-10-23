import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Predictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, images):
        pass

    def output_dict(self, fn=None, clss=None, mask=None, img=None, full=None):
        return {
            "fn": fn,
            "out": {
                "class": clss,
                "mask": mask,
            },
            "orig_img": img,
            "full": full,
        }


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

        return marg_msk.astype(bool)

    def predict(self, images):
        return [
            self.output_dict(
                fn=img["fn"],
                clss=["margin"],
                mask=[self.get_mask(img["img"].shape[:2])],
                img=img["img"],
            )
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
        return [
            self.output_dict(
                fn=img["fn"],
                clss=[
                    img["result"].names[c.item()] for c in img["result"].boxes.cls.cpu()
                ],
                mask=self.__get_mask_list(img["result"]),
                img=img["result"].orig_img,
                full=img["result"],
            )
            for img in [
                # [0] is because it returns a 1 element list
                self.rslt_df[self.rslt_df["fn"] == f].to_dict(orient="records")[0]
                for f in [i["fn"] for i in images]
            ]
        ]


class CombinedPredictor(Predictor):
    def __init__(self, predictors):
        self.predictors = predictors

    def predict(self, images):
        output_list = []
        # out_l = [self.output_dict(fn=fn) for fn in [i["fn"] for i in images]]
        # print(out_l)
        # for fn in [i["fn"] for i in images]:
        #     print(self.output_dict(fn=fn))

        for name, predictor in self.predictors:
            print([n["fn"] for n in images])
            print(name)
            preds = predictor.predict(images)
            print([n["fn"] for n in preds])

            if not output_list:
                output_list = [
                    self.output_dict(
                        fn=p["fn"],
                        clss=[name + "_" + cn for cn in p["out"]["class"]],
                        mask=p["out"]["mask"],
                        img=p["orig_img"],
                        full=p["full"],
                    )
                    for p in preds
                ]
            else:
                for i, p in enumerate(preds):
                    print(p["fn"])
                    print(output_list[i]["fn"])

                # output_list = [
                #     self.output_dict(
                #         fn=p["fn"],
                #         # clss=p["out"]["class"],
                #         clss=[name + "_" + cn for cn in p["out"]["class"]],
                #         mask=p["out"]["mask"],
                #         img=p["orig_img"],
                #         full=p["full"],
                #     )
                #     for p in output_list
                # ]

            # for m in preds:
            #     print(m['fn'])

        return output_list
