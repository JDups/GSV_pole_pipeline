import numpy as np
from utils import show_mask, get_overlay
import matplotlib.pyplot as plt
import math
import os
import cv2


def show_masks_indiv(preds, rules):
    for p in preds:
        print(f"Image: {p['fn']}")
        show_mask(p["orig_img"])

        if not p["out"]["mask"]:
            print("No masks for image")
            continue

        for i, m in enumerate(p["out"]["mask"]):
            prd_cls = p["out"]["class"][i]
            print(f"Prediction class: {prd_cls}")
            print(f"Mask area: {m.sum()}")
            if prd_cls in rules["interest"]:
                show_mask(p["orig_img"], p_msk=m)
            if prd_cls in rules["occluding"]:
                show_mask(p["orig_img"], n_msk=m)


class Pipeline:
    def __init__(self, loader, predictor, rules={}):
        self.lder = loader
        self.pder = predictor
        self.rls = rules


    def run(self, iterations=None, save_loc=None):
        counter = 0
        if save_loc:
            os.makedirs(save_loc, exist_ok=True)

        for pid in self.lder.pole_imgs_df["pole_id"].unique():
            counter += 1
            print(f"\nPole ID: {pid}")
            # batch = self.lder.get_batch(pid)
            preds = self.pder.predict(self.lder.get_batch(pid))

            # print(preds[0])

            # show_masks_indiv(preds, self.rls)
            # show_masks_comb(preds, self.rls)

            largest = {
                "fn": None,
                "interest": None,
                "occluding": None,
                "orig_img": None,
            }

            for p in preds:
                if save_loc:
                    cv2.imwrite(save_loc+"p"+p["fn"].split("_")[1]+"_s1__h"+p["fn"].split("_")[3]+".png", cv2.cvtColor(p["orig_img"], cv2.COLOR_BGR2RGB))
                occl = np.zeros(p["orig_img"].shape[:2], dtype=bool)

                mcntr = 0
                for clss, m in zip(p["out"]["class"], p["out"]["mask"]):
                    if save_loc:
                        cv2.imwrite(save_loc+"p"+p["fn"].split("_")[1]+"_s2_h"+p["fn"].split("_")[3]+f"_{mcntr}_"+clss+".png", cv2.cvtColor(show_mask(p["orig_img"], p_msk=m, show=False), cv2.COLOR_BGR2RGB))
                        mcntr += 1

                    if clss in self.rls["occluding"]:
                        occl = np.logical_or(occl, m)

                    if clss in self.rls["interest"]:
                        if largest["fn"] is None or m.sum() > largest["interest"].sum():
                            largest = {
                                "fn": p["fn"],
                                "interest": m,
                                "occluding": None,
                                "orig_img": p["orig_img"],
                            }

                if largest["fn"] == p["fn"]:
                    # print(largest["interest"].sum())
                    largest["occluding"] = occl
                # else:
                #     print("No pole yet")

                if save_loc:
                    if not p["out"]["mask"]:
                        cv2.imwrite(save_loc+"p"+p["fn"].split("_")[1]+"_s2_h"+p["fn"].split("_")[3]+"_no_masks.png", cv2.cvtColor(p["orig_img"], cv2.COLOR_BGR2RGB))

            if largest["fn"] is None:
                print("No pole found at location")
            else:
                cv2.imwrite(save_loc+"p"+largest["fn"].split("_")[1]+"_s3_h"+largest["fn"].split("_")[3]+".png", cv2.cvtColor(show_mask(largest["orig_img"], p_msk=largest["interest"], n_msk=largest["occluding"], show=False), cv2.COLOR_BGR2RGB))

                print(f"File: {largest['fn']}")
                overlap = np.logical_and(largest["interest"], largest["occluding"]).sum()
                print(overlap)
                if overlap:
                    column_sum = largest["interest"].sum(axis=0)
                    print(column_sum)
                    colums_hit = np.nonzero(column_sum)
                    print(colums_hit)
                    left_edge = np.min(colums_hit)
                    right_edge = np.max(colums_hit)
                    mid_point = (right_edge+left_edge)/2
                    print(mid_point)
                    # show_mask(
                    #     largest["orig_img"],
                    #     p_msk=largest["interest"],
                    #     n_msk=largest["occluding"],
                    # )
                    # https://stackoverflow.com/questions/28417604/plotting-a-line-from-a-coordinate-with-and-angle
                    fig, ax = plt.subplots()
                    heading = int(largest['fn'].split("_")[3])
                    ax.set_xlim(-10,10)
                    ax.set_ylim(-10,10)
                    print(left_edge/640*90)
                    print(right_edge/640*90)

                    ax.plot(0,0, 'ro', markersize=20, fillstyle='none')

                    for angle in [
                        heading-45,
                        heading+45,
                        heading+45-left_edge/640*90, 
                        heading+45-mid_point/640*90, 
                        heading+45-right_edge/640*90
                    ]: 
                        edge_length = 50
                        x, y = 0, 0
                        endx = x + edge_length * math.cos(math.radians(angle))
                        endy = y + edge_length * math.sin(math.radians(angle))
                        ax.plot([x,endx],[y,endy])
                    for angle in [heading-180+45-mid_point/640*90]: 
                        repo_length = 5
                        x, y = 0, 0
                        endx = x + repo_length * math.cos(math.radians(angle))
                        endy = y + repo_length * math.sin(math.radians(angle))
                        ax.plot([x,endx],[y,endy])
                        ax.plot(endx,endy, 'rx', markersize=20,)
                    plt.savefig(save_loc+"p"+largest["fn"].split("_")[1]+"_s4"+".png")
                    plt.close(fig)
                    # plt.show()
                else:
                    cv2.imwrite(save_loc+"p"+largest["fn"].split("_")[1]+"_sF"+".png", cv2.cvtColor(largest["orig_img"], cv2.COLOR_BGR2RGB))

            if counter == iterations:
                break


"""
    General idea will be
        - Loader object that returns batch of images that are to be processed at a time.
          Loader could also possibly be the one determine the end condition of the whole 
          process, with it keeping track internally. With a Pipeline level override.
        
        - Process objects that will return masks (or possibly other types of predictions)
        
        - Rules/Decison object that will take the the previous steps predictions and make decision.
          The Pipeline could be given a list of what prediction represent what we want and which represent what we don't want.
          It could then use that list to cross compare the outputs of multiple process objects
        
"""
