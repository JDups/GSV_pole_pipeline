import numpy as np
from utils import show_mask, get_overlay
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import cv2

"""
TODO: Marker drawing helper functions/methods
"""


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
        self.step_n = 0

        if self.log_fp:
            os.makedirs(self.log_fp, exist_ok=True)
            self.lder.log_fp = self.log_fp
            self.fig, self.ax = plt.subplots()
            plt.axis("equal")

    def __save_fn(self, fn, step_n, post_str=""):
        return (
            self.log_fp
            + f"p{fn.split('_')[1]}_s{step_n}_h{fn.split('_')[3]}{post_str}.png"
        )

    def __save_log_img(self, fn, img, step_n=self.step_n, post_str=""):
        if self.log_fp:
            fn = self.__save_fn(fn, step_n, post_str)
            cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __save_log_plt(self, fn, post_str=""):
        if self.log_fp:
            fn = self.__save_fn(fn, "P", post_str)
            plt.savefig(fn)

    def __draw_target(self, lng, lat, color="tab:red"):
        self.ax.plot(lng, lat, color=color, marker="o", markersize=20, fillstyle="none")
        self.ax.plot(lng, lat, color=color, marker="o", markersize=3)

    def __draw_cross(self, lng, lat, color="tab:red"):
        self.ax.plot(lng, lat, color=color, marker="x", markersize=20)

    def __run_reset():
        plt.cla()
        self.step_n = 0

    def run(self, iterations=None):
        for pcount, pid in enumerate(self.lder.data_df["pole_id"].unique()):
            # pid = 12390
            print(f"\nPole ID: {pid}")

            self.__run_reset()

            plat, plng = self.lder.data_df[self.lder.data_df["pole_id"] == pid][
                ["Latitude", "Longitude"]
            ].values[0]
            print(f"CSV lat: {plat} lng: {plng}")
            self.__draw_target(plng, plat)

            batch = self.lder.get_batch(pid)

            lat = batch[0]["metadata"]["location"]["lat"]
            lng = batch[0]["metadata"]["location"]["lng"]
            print(f"GSV lat: {lat} lng: {lng}")
            self.__draw_cross(lng, lat, "tab:blue")
            for b in batch:
                self.__save_log_img(b["fn"], b["img"])

            self.step_n = 1
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
                # self.__save_log_img(p["fn"], p["orig_img"], step_n=1)

                occl = np.zeros(p["orig_img"].shape[:2], dtype=bool)

                for mcntr, (clss, m) in enumerate(
                    zip(p["out"]["class"], p["out"]["mask"])
                ):
                    self.__save_log_img(
                        p["fn"],
                        show_mask(p["orig_img"], p_msk=m, show=False),
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

                if not p["out"]["mask"]:
                    self.__save_log_img(
                        p["fn"], p["orig_img"], post_str="_no_masks.png"
                    )

            if largest["fn"] is None:
                print(f"No {self.rls['interest'][0]} found at location")
            else:
                self.step_n = 2
                self.__save_log_img(
                    largest["fn"],
                    show_mask(
                        largest["orig_img"],
                        p_msk=largest["interest"],
                        n_msk=largest["occluding"],
                        show=False,
                    ),
                )

                # print(f"File: {largest['fn']}")
                overlap = np.logical_and(
                    largest["interest"], largest["occluding"]
                ).sum()

                img_w = largest["orig_img"].shape[0]
                column_sum = largest["interest"].sum(axis=0)
                colums_hit = np.nonzero(column_sum)
                left_edge = np.min(colums_hit)
                right_edge = np.max(colums_hit)
                mid_point = (right_edge + left_edge) / 2
                print(f"left_edge: {left_edge}")
                print(f"right_edge: {right_edge}")
                print(f"mid_point: {mid_point}")
                view_len = 0.0003

                # Turns gsv heading into angle from horizontal
                # 0->90, 90->0, 180->-90, 270->-180, 360->-270
                heading = -int(largest["fn"].split("_")[3]) + 90
                print(f"heading: {heading}")

                for angle in [-45, 45]:
                    angle = heading + angle
                    endx = lng + view_len * math.cos(math.radians(angle))
                    endy = lat + view_len * math.sin(math.radians(angle))
                    self.ax.plot([lng, endx], [lat, endy], "tab:blue")
                for angle in [left_edge, mid_point, right_edge]:
                    angle = heading + 45 - angle / img_w * 90
                    endx = lng + view_len * math.cos(math.radians(angle))
                    endy = lat + view_len * math.sin(math.radians(angle))
                    self.ax.plot(
                        [lng, endx],
                        [lat, endy],
                        "tab:red",
                        linewidth=0.5,
                        linestyle="--",
                    )

                if overlap == 1:
                    self.__save_log_img(largest["fn"], largest["orig_img"], step_n="F")

                else:
                    self.step_n = 3
                    nlat, nlng = lat, lng
                    strat = "ortho"
                    adj_angl = 0
                    repo_len = 0.0001
                    if strat == "backup":
                        adj_angl = heading - 180 + 45 - mid_point / img_w * 90
                    if strat == "ortho":
                        adj_angl = heading - 90  # + 45 - mid_point / img_w * 90
                    endx = lng + repo_len * math.cos(math.radians(adj_angl))
                    endy = lat + repo_len * math.sin(math.radians(adj_angl))
                    nlat, nlng = endy, endx
                    self.ax.plot([lng, endx], [lat, endy], "tab:brown")
                    self.__draw_cross(endx, endy, "tab:brown")

                    dlat = plat - nlat
                    dlng = plng - nlng
                    est_heading = int(
                        (-math.degrees(math.atan2(dlat, dlng)) + 90) % 360
                    )

                    new_pic = self.lder.pic_from_loc(pid, nlat, nlng, est_heading)[0]

                    self.__save_log_img(new_pic["fn"], new_pic["img"])

                    clat = new_pic["metadata"]["location"]["lat"]
                    clng = new_pic["metadata"]["location"]["lng"]
                    self.__draw_cross(clng, clat, "tab:cyan")

                    dlat = plat - clat
                    dlng = plng - clng
                    est_heading = int(
                        (-math.degrees(math.atan2(dlat, dlng)) + 90) % 360
                    )
                    self.step_n = 4
                    new_pic = self.lder.pic_from_loc(pid, clat, clng, est_heading)[0]

                    self.__save_log_img(new_pic["fn"], new_pic["img"])

                    est_heading = -est_heading + 90

                    for angle in [-45, 45]:
                        angle = est_heading + angle
                        endx = clng + view_len * math.cos(math.radians(angle))
                        endy = clat + view_len * math.sin(math.radians(angle))
                        self.ax.plot([clng, endx], [clat, endy], "tab:cyan")

                self.__save_log_plt(largest["fn"])

            if pcount + 1 == iterations:
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
