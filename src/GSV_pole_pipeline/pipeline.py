import numpy as np
from utils import show_mask, get_overlay
import matplotlib
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
    def __init__(self, loader, predictor, rules={}, log_fp=None):
        self.lder = loader
        self.pder = predictor
        self.rls = rules
        self.log_fp = log_fp

        if self.log_fp:
            os.makedirs(self.log_fp, exist_ok=True)
            self.lder.set_log_fp(self.log_fp)

    def save_log_img(self, fn, img, step_n, post_str=""):
        fn = (
            self.log_fp
            + f"p{fn.split('_')[1]}_s{step_n}_h{fn.split('_')[3]}{post_str}.png"
        )

        if isinstance(img, np.ndarray):
            cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif isinstance(img, matplotlib.figure.Figure):
            plt.savefig(fn)

    def run(self, iterations=None):
        counter = 0

        for pid in self.lder.data_df["pole_id"].unique():
            pid = 12390
            counter += 1
            print(f"\nPole ID: {pid}")
            batch = self.lder.get_batch(pid)
            preds = self.pder.predict(batch)

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
                if self.log_fp:
                    self.save_log_img(p["fn"], p["orig_img"], step_n=1)

                occl = np.zeros(p["orig_img"].shape[:2], dtype=bool)

                for mcntr, (clss, m) in enumerate(
                    zip(p["out"]["class"], p["out"]["mask"])
                ):
                    if self.log_fp:
                        self.save_log_img(
                            p["fn"],
                            show_mask(p["orig_img"], p_msk=m, show=False),
                            step_n=2,
                            post_str=f"_{mcntr}_{clss}.png",
                        )

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
                    largest["occluding"] = occl
                # else:
                #     print("No pole yet")

                if self.log_fp:
                    if not p["out"]["mask"]:
                        self.save_log_img(
                            p["fn"],
                            p["orig_img"],
                            step_n=2,
                            post_str="_no_masks.png",
                        )

            if largest["fn"] is None:
                print(f"No {self.rls['interest'][0]} found at location")
            else:
                if self.log_fp:
                    self.save_log_img(
                        largest["fn"],
                        show_mask(
                            largest["orig_img"],
                            p_msk=largest["interest"],
                            n_msk=largest["occluding"],
                            show=False,
                        ),
                        step_n=3,
                    )

                print(f"File: {largest['fn']}")
                overlap = np.logical_and(
                    largest["interest"], largest["occluding"]
                ).sum()
                print(overlap)
                if overlap == 0:
                    self.save_log_img(
                        largest["fn"],
                        largest["orig_img"],
                        step_n="F",
                    )
                else:
                    column_sum = largest["interest"].sum(axis=0)
                    colums_hit = np.nonzero(column_sum)
                    # print(colums_hit)
                    left_edge = np.min(colums_hit)
                    right_edge = np.max(colums_hit)
                    mid_point = (right_edge + left_edge) / 2
                    print(f"left_edge: {left_edge}")
                    print(f"right_edge: {right_edge}")
                    print(f"mid_point: {mid_point}")
                    # print(mid_point)
                    # show_mask(
                    #     largest["orig_img"],
                    #     p_msk=largest["interest"],
                    #     n_msk=largest["occluding"],
                    # )
                    # https://stackoverflow.com/questions/28417604/plotting-a-line-from-a-coordinate-with-and-angle
                    fig, ax = plt.subplots()
                    heading = -int(largest["fn"].split("_")[3]) + 90
                    print(f"heading: {heading}")
                    # ax.set_xlim(-10, 10)
                    # ax.set_ylim(-10, 10)
                    # print(left_edge / 640 * 90)
                    # print(right_edge / 640 * 90)

                    lat = batch[0]["metadata"]["location"]["lat"]
                    lng = batch[0]["metadata"]["location"]["lng"]
                    print(f"GSV lat: {lat} lng: {lng}")
                    plat = self.lder.data_df[self.lder.data_df["pole_id"] == pid][
                        "Latitude"
                    ].values[0]
                    plng = self.lder.data_df[self.lder.data_df["pole_id"] == pid][
                        "Longitude"
                    ].values[0]
                    print(f"CSV lat: {plat} lng: {plng}")
                    ax.plot(lng, lat, "ro", markersize=20, fillstyle="none")
                    ax.plot(plng, plat, "bo", markersize=20, fillstyle="none")

                    # ax.text(0,0, f"{lat} {lng}")

                    nlat, nlng = lat, lng
                    for angle in [
                        heading - 45,
                        heading + 45,
                        heading + 45 - left_edge / 640 * 90,
                        heading + 45 - mid_point / 640 * 90,
                        heading + 45 - right_edge / 640 * 90,
                    ]:
                        edge_length = 0.0005
                        # x, y = 0, 0
                        x, y = lng, lat
                        endx = x + edge_length * math.cos(math.radians(angle))
                        endy = y + edge_length * math.sin(math.radians(angle))
                        ax.plot([x, endx], [y, endy])

                    strat = "ortho"
                    if strat == "backup":
                        for angle in [heading - 180 + 45 - mid_point / 640 * 90]:
                            repo_length = 0.0005
                            # x, y = 0, 0
                            x, y = lng, lat
                            endx = x + repo_length * math.cos(math.radians(angle))
                            endy = y + repo_length * math.sin(math.radians(angle))
                            nlat, nlng = endy, endx
                            ax.plot([x, endx], [y, endy])
                            ax.plot(
                                endx,
                                endy,
                                "bx",
                                markersize=20,
                            )
                    if strat == "ortho":
                        for angle in [heading - 90 + 45 - mid_point / 640 * 90]:
                            repo_length = 0.0005
                            # x, y = 0, 0
                            x, y = lng, lat
                            endx = x + repo_length * math.cos(math.radians(angle))
                            endy = y + repo_length * math.sin(math.radians(angle))
                            nlat, nlng = endy, endx
                            ax.plot([x, endx], [y, endy])
                            ax.plot(
                                endx,
                                endy,
                                "bx",
                                markersize=20,
                            )

                    dlat = nlat - plat
                    dlng = nlng - plng
                    est_heading = int(-math.degrees(math.atan2(dlat, dlng)) - 90)

                    new_pic = self.lder.pic_from_loc_head(pid, nlat, nlng, est_heading)[0]

                    self.save_log_img(
                        new_pic["fn"],
                        new_pic["img"],
                        step_n=4,
                    )

                    clat = new_pic["metadata"]["location"]["lat"]
                    clng = new_pic["metadata"]["location"]["lng"]
                    ax.plot(clng, clat, "rx", markersize=20, fillstyle="none")

                    self.save_log_img(
                        largest["fn"],
                        fig,
                        step_n="P",
                    )
                    plt.close(fig)
                    # plt.show()


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
        
        - I think i will need an addition to the rules where I apply classifiers to certain masks/boxes
"""
