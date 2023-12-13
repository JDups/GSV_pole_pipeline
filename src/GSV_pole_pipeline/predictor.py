import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from ultralytics import YOLO
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


class Predictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, images):
        pass

    def output_dict(self, fn=None, clss=None, mask=None, img=None, full=None):
        # TODO: Decide whether out should be a dict of lists or list of dicts
        #       Was done teh current way because that's how the Yolo Result object has it.
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


class YOLOPredictor(Predictor):
    def __init__(self, weights_fp, device=None, conf=0.25):
        self.model = YOLO(weights_fp)
        self.device = device
        self.conf = conf

    def __get_mask_list(self, result):
        if result.masks:
            return list(result.masks.data.cpu().numpy().astype(bool))
        else:
            return []

    def predict(self, images):
        preds = []
        for i in images:
            result = self.model.predict(
                i["img"],
                device=self.device,
                conf=self.conf,
                imgsz=(i["img"].shape[0], i["img"].shape[1]),
            )[0]
            print([result.names[c.item()] for c in result.boxes.cls])
            print(result.boxes.conf)
            preds.append({"fn": i["fn"], "result": result})

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
            for img in preds
        ]


class CombinedPredictor(Predictor):
    def __init__(self, predictors):
        # predictors is a list of tuples of the format [("name", Predictor Object),]
        self.predictors = predictors

    def add_prefix(self, prefix, class_name):
        return prefix + "__" + class_name

    def predict(self, images):
        output_list = []

        for name, predictor in self.predictors:
            preds = predictor.predict(images)

            if not output_list:
                output_list = [
                    self.output_dict(
                        fn=p["fn"],
                        clss=[self.add_prefix(name, cn) for cn in p["out"]["class"]],
                        mask=p["out"]["mask"],
                        img=p["orig_img"],
                        full=[{name: p["full"]}],
                    )
                    for p in preds
                ]
                continue

            for i, p in enumerate(preds):
                # ol_idx is to get the correct index for the same filename.
                # To ensure that even if two predictors don't return predictions in
                # the same order that they get matched up correctly.
                # There is very possibly a more elegant solution.
                ol_idx = [
                    x
                    for x in range(len(output_list))
                    if output_list[x]["fn"] == p["fn"]
                ][0]

                output_list[ol_idx] = self.output_dict(
                    fn=p["fn"],
                    clss=output_list[ol_idx]["out"]["class"]
                    + [self.add_prefix(name, cn) for cn in p["out"]["class"]],
                    mask=output_list[ol_idx]["out"]["mask"] + p["out"]["mask"],
                    img=p["orig_img"],
                    full=output_list[ol_idx]["full"] + [{name: p["full"]}],
                )

        return output_list


class GroundedSAMPredictor(Predictor):
    def __init__(
        self,
        dino_weights_fp,
        dino_conf_fp,
        sam_weights_fp,
        sam_enc_version,
        class_prompts,
        box_thresh=0.35,
        text_thresh=0.25,
        device="0",
    ):
        self.device = device
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.class_prompts = class_prompts
        self.dino = Model(
            model_config_path=dino_conf_fp,
            model_checkpoint_path=dino_weights_fp,
            device=self.device,
        )
        self.sam = SamPredictor(
            sam_model_registry[sam_enc_version](checkpoint=sam_weights_fp).to(
                device=self.device
            )
        )

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam.predict(box=box, multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def __get_prompt_class(self, prompts, class_id):
        pred_class = []
        for pi in class_id:
            for c in self.class_prompts:
                if prompts[pi] in self.class_prompts[c]:
                    pred_class.append(c)
        return pred_class

    def predict(self, images):
        prompts = []
        for clss in self.class_prompts:
            prompts += self.class_prompts[clss]

        preds = []
        for i in images:
            detections = self.dino.predict_with_classes(
                image=i["img"],
                classes=prompts,
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
            )
            detections = detections[detections.class_id != None]
            detections.mask = self.segment(image=i["img"], xyxy=detections.xyxy)
            preds.append(
                {
                    "fn": i["fn"],
                    "classes": self.__get_prompt_class(prompts, detections.class_id),
                    "result": detections,
                    "img": i["img"],
                }
            )

        return [
            self.output_dict(
                fn=img["fn"],
                clss=img["classes"],
                mask=list(img["result"].mask),
                img=img["img"],
                full=img["result"],
            )
            for img in preds
        ]
