import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from ultralytics import YOLO
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

from omegaconf import OmegaConf
# from ldm.util import create_carvekit_interface
from PIL import Image
import cv2

import matplotlib.pyplot as plt  # fro p2g debugging

import sys
sys.path.append('/data/jonathandupuis/pix2gestalt/pix2gestalt') # for VM
import inference

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
    def __init__(self, predictors, links={}):
        # predictors is a list of tuples of the format [("name", Predictor Object),]
        self.predictors = predictors
        self.links = links

    def add_prefix(self, prefix, class_name):
        return prefix + "__" + class_name

    def predict(self, images):
        output_list = []

        for name, predictor in self.predictors:
            # print(name)
            # print(output_list)
            if name in self.links:
                pass_preds = output_list
                for i_p, p in enumerate(pass_preds):
                    new_class, new_mask = [], []
                    for i, c in enumerate(p["out"]["class"]):
                        if c.split("__")[0] == self.links[name]:
                            new_class.append(c)
                            new_mask.append(p["out"]["mask"][i])
                    pass_preds[i_p]["out"]["class"] = new_class
                    pass_preds[i_p]["out"]["mask"] = new_mask

                preds = predictor.predict(images, pass_preds)
            else:
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


class Pix2GestaltPredictor(Predictor):
    def __init__(
        self,
        conf_fp,
        weights_fp,
        target_class=[],
        device=None,
        guidance_scale=2.0,
        n_samples=1,
        ddim_steps=200,
        mask_type="single",
        vote_count=1,
    ):
        self.target_class = target_class
        self.guidance_scale = guidance_scale
        self.n_samples = n_samples
        self.ddim_steps = ddim_steps
        self.mask_type = mask_type
        self.vote_count = vote_count
        if self.mask_type == "single" and self.n_samples > 1:
            print(f"n_samples set to {self.n_samples} but mask_typeset to single, samples after 1st will be discarded.")
        conf = OmegaConf.load(conf_fp)
        self.p2g = inference.load_model_from_config(conf, weights_fp, device=device)

    def get_mask_from_pred(self, pred_image, thresholding=True):
        """
        Since pix2gestalt performs amodal completion and segmentation jointly,
        the whole (amodal) object is synthesized on a white background.

        We can either perform traditional thresholding or utilize a background removal /
        matting tool to extract the amodal mask (alpha channel) from pred_image.

        For evaluation, we use direct thresholding. Below, we implement both.
        While we didn't empirically verify this, matting should slightly improve
        the amodal segmentation performance.
        """
        if thresholding:
            # TODO: find better fix, pred_image should already be array at this point, unsure why it isn't
            gray_image = cv2.cvtColor(np.array(pred_image), cv2.COLOR_BGR2GRAY)
            _, pred_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)
        else:
            raise("Not yet implemented")
            # pred_image = Image.fromarray(pred_image)
            # interface = create_carvekit_interface()
            # amodal_rgba = np.array(interface([pred_image])[0])
            # alpha_channel = amodal_rgba[:, :, 3]
            # visible_mask = (alpha_channel > 0).astype(np.uint8) * 255

            # rgb_visible_mask = np.zeros(
            #     (visible_mask.shape[0], visible_mask.shape[1], 3), dtype=np.uint8
            # )
            # rgb_visible_mask[:, :, 0] = visible_mask
            # rgb_visible_mask[:, :, 1] = visible_mask
            # rgb_visible_mask[:, :, 2] = visible_mask  # (256, 256, 3)
            # pred_mask = rgb_visible_mask

        return pred_mask

    def resize_preds(self, original_image, pred_reconstructions):
        height, width = original_image.shape[:2]

        resized_images = []
        for image in pred_reconstructions:
            # Resize image to match the size of original_image using Lanczos interpolation
            resized_images.append(
                cv2.resize(
                    # image, (width, height), interpolation=cv2.INTER_NEAREST
                    image, (width, height), interpolation=cv2.INTER_LINEAR
                )
            )

        return resized_images

    def predict(self, images, prev_preds):
        preds = []

        for im in images:
            amodal_masks, v_mask_list, class_list, all_outs = [], [], [], []

            for p in prev_preds:
                if im["fn"] == p["fn"]:
                    v_mask_list = p["out"]["mask"]
                    class_list = p["out"]["class"]

            if self.target_class:
                target_idx = [
                    i
                    for i in range(len(class_list))
                    if class_list[i] in self.target_class
                ]
                v_mask_list = [v_mask_list[i] for i in target_idx]
                class_list = [class_list[i] for i in target_idx]

            pred_mask = []
            for v_mask in v_mask_list:
                print(v_mask)
                outs = inference.run_inference(
                    input_image=cv2.resize(im["img"], (256, 256)),
                    visible_mask=cv2.resize(
                        (v_mask * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST
                    ),
                    model=self.p2g,
                    guidance_scale=self.guidance_scale,
                    n_samples=self.n_samples,
                    ddim_steps=self.ddim_steps,
                )
                all_outs += outs

                if self.mask_type == "single":
                    pred_mask.append(self.get_mask_from_pred(outs[0]))
                    
                if self.mask_type == "voting":
                    sum_preds = np.zeros((256, 256))
                    for out in outs:
                        out_mask = self.get_mask_from_pred(out)
                        sum_preds = sum_preds + out_mask.astype(int)/255
                    vote_preds = sum_preds > self.vote_count
                    pred_mask.append(vote_preds)

            print(len(pred_mask))
            resized_masks = self.resize_preds(im["img"], pred_mask)
            amodal_masks = [m.astype(bool) for m in pred_mask]
            resized_masks = [m.astype(bool) for m in resized_masks]

            preds.append(
                {
                    "fn": im["fn"],
                    "classes": class_list,
                    "mask": resized_masks,
                    "img": im["img"],
                    "result": {
                        "amodal_masks": amodal_masks,
                        "resized_masks": resized_masks,
                        "outs": all_outs,
                        },
                }
            )

        return [
            self.output_dict(
                fn=img["fn"],
                clss=img["classes"],
                mask=list(img["mask"]),
                img=img["img"],
                full=img["result"],
            )
            for img in preds
        ]
