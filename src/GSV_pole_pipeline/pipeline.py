import numpy as np
from utils import show_mask, get_overlay, get_end_coords, get_est_heading
import matplotlib
import matplotlib.pyplot as plt
import math
import os
import cv2

"""
TODO: Fix plotting in jupyter notebooks
TODO: Add plotting for when no detection of interest is made
TODO: Make target selection into method
TODO: Loader and pipeline should be merged into same objects.
      The pipeline logic ends up depending on the loader anyways.

TODO: Make the drawfov method take an argument for the actual fov
      Then keep trying to fit the Dashcam loader in
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
        self.curr_step = 0

        if self.log_fp:
            os.makedirs(self.log_fp, exist_ok=True)
            self.lder.set_log_fp(self.log_fp)
            self.fig, self.ax = plt.subplots()
            plt.axis("equal")

    def __save_fn(self, fn, step_n, post_str=""):
        return (
            self.log_fp
            + f"p{fn.split('_')[1]}_s{step_n}_h{fn.split('_')[3]}{post_str}.png"
        )

    def __save_log_img(self, fn, img, step_n=None, post_str=""):
        if step_n == None:
            step_n = self.curr_step
        if self.log_fp:
            fn = self.__save_fn(fn, step_n, post_str)
            cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __save_log_plt(self, fn, post_str=""):
        if self.log_fp:
            fn = self.__save_fn(fn, "P", post_str)
            plt.savefig(fn)

    def __draw_target(self, lng, lat, color="tab:red"):
        if self.log_fp:
            self.ax.plot(
                lng, lat, color=color, marker="o", markersize=20, fillstyle="none"
            )
            self.ax.plot(lng, lat, color=color, marker="o", markersize=3)

    def __draw_cross(self, lng, lat, color="tab:red"):
        if self.log_fp:
            self.ax.plot(lng, lat, color=color, marker="x", markersize=20)

    def __draw_lines(self, x, y, angles, line_len, **kwargs):
        if self.log_fp:
            for a in angles:
                endx, endy = get_end_coords(x, y, a, line_len)
                self.ax.plot([x, endx], [y, endy], **kwargs)

    def __draw_fov(self, lng, lat, heading, color, fov=90, view_len=0.0003):
        if self.log_fp:
            kwargs = {"color": color}
            angles = [heading - fov / 2, heading + fov / 2]
            self.__draw_lines(lng, lat, angles, view_len, **kwargs)

    def __draw_obj_span(
        self, lng, lat, heading, edges, color="tab:red", view_len=0.0003
    ):
        if self.log_fp:
            kwargs = {"color": color, "linewidth": 0.5, "linestyle": "--"}
            angles = [heading + 45 - a for a in edges]
            self.__draw_lines(lng, lat, angles, view_len, **kwargs)

    def __run_reset(self):
        plt.cla()
        self.curr_step = 0

    def run(self, iterations=None):
        for pcount, (pid, batch) in enumerate(self.lder):
            self.__run_reset()
            # pid = 12390  # random good one
            # pid = 9633 # bad status
            # pid = 2569  # blurring on pole
            print(f"\nPole count: {pcount}, Pole ID: {pid}")

            plat, plng = self.lder.fetch_latlng(pid)
            print(f"CSV lat: {plat} lng: {plng}")
            self.__draw_target(plng, plat)

            if not batch:
                print("No response")
                continue

            lat = batch[0]["metadata"]["location"]["lat"]
            lng = batch[0]["metadata"]["location"]["lng"]
            print(f"GSV lat: {lat} lng: {lng}")
            self.__draw_cross(lng, lat, "tab:blue")
            for b in batch:
                self.__save_log_img(b["fn"], b["img"])

            self.curr_step = 1
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
                self.curr_step = 2
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

                # Turns gsv heading into angle from horizontal
                # 0->90, 90->0, 180->-90, 270->-180, 360->-270
                heading = -int(largest["fn"].split("_")[3]) + 90
                print(f"heading: {heading}")

                if self.lder.source == "Dashcam":
                    fov = 140
                else:
                    fov = 90

                self.__draw_fov(lng, lat, heading, fov=fov, color="tab:blue")

                img_w = largest["orig_img"].shape[1]
                column_sum = largest["interest"].sum(axis=0)
                colums_hit = np.nonzero(column_sum)
                left_edge = np.min(colums_hit)
                right_edge = np.max(colums_hit)
                mid_point = (right_edge + left_edge) / 2
                edges_angles = [
                    edge / img_w * fov for edge in [left_edge, mid_point, right_edge]
                ]
                print(f"left_edge: {left_edge}")
                print(f"right_edge: {right_edge}")
                print(f"mid_point: {mid_point}")

                self.__draw_obj_span(lng, lat, heading, edges_angles)

                if overlap == 0:
                    self.__save_log_img(largest["fn"], largest["orig_img"], step_n="F")

                else:
                    if self.lder.source == "GSV":
                        self.curr_step = 3
                        strat = "ortho"
                        adj_angl = 0
                        repo_len = 0.0001
                        if strat == "backup":
                            adj_angl = heading - 180 + 45 - mid_point / img_w * 90
                        if strat == "ortho":
                            adj_angl = heading - 90  # + 45 - mid_point / img_w * 90
                        nlng, nlat = get_end_coords(lng, lat, adj_angl, repo_len)

                        self.ax.plot([lng, nlng], [lat, nlat], "tab:brown")
                        self.__draw_cross(nlng, nlat, "tab:brown")

                        est_heading = get_est_heading(nlng, nlat, plng, plat)

                        _, new_loc = self.lder.results_from_loc(nlat, nlng, est_heading)

                        clat = new_loc.metadata[0]["location"]["lat"]
                        clng = new_loc.metadata[0]["location"]["lng"]

                        self.__draw_cross(clng, clat, "tab:cyan")

                        est_heading = get_est_heading(clng, clat, plng, plat)

                        self.curr_step = 4
                        new_pic = self.lder.pic_from_loc(pid, clat, clng, est_heading)[
                            0
                        ]

                        self.__save_log_img(new_pic["fn"], new_pic["img"], step_n="F")

                        est_heading = -est_heading + 90

                        self.__draw_fov(clng, clat, est_heading, "tab:cyan")

                    if self.lder.source == "Dashcam":
                        pass

            if largest["fn"]:
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
