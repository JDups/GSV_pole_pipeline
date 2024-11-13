import numpy as np
from utils import show_mask, get_overlay, get_end_coords, get_est_heading
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import pandas as pd

"""
TODO: Fix doubling .png in filename for masks
TODO: Fix plotting in jupyter notebooks
TODO: Add plotting for when no detection of interest is made
TODO: Loader and pipeline should be merged into same objects.
      The pipeline logic ends up depending on the loader anyways.
TODO: Move "if step_n is None" from __save_log_img to __save_fn
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
    def __init__(self, loader, predictor, decision=None, rules={}, log_fp=None, clf_fp=None, save_decision=True):
        self.lder = loader
        self.pder = predictor
        self.rls = rules
        self.log_fp = log_fp
        self.decision = decision
        self.curr_step = 0
        self.save_decision = save_decision

        if self.log_fp:
            os.makedirs(self.log_fp, exist_ok=True)
            self.lder.set_log_fp(self.log_fp)
            self.fig, self.ax = plt.subplots()
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
            plt.axis("equal")

        if clf_fp:
            with open(clf_fp, "rb") as f:
                self.clf = pickle.load(f)

    def __save_fn(self, fn, step_n, post_str=""):
        return (
            self.log_fp
            + f"p{fn.split('_')[1]}_s{step_n}_h{fn.split('_')[3]}{post_str}.png"
        )

    def __save_log_img(self, fn, img, step_n=None, post_str=""):
        if step_n is None:
            step_n = self.curr_step
        if self.log_fp:
            fn = self.__save_fn(fn, step_n, post_str)
            cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __save_log_batch(self, batch):
        for b in batch:
            self.__save_log_img(b["fn"], b["img"])

    def __save_log_biggest(self, biggest):
        self.__save_log_img(
            biggest["fn"],
            show_mask(
                biggest["orig_img"],
                p_msk=biggest["interest"],
                n_msk=biggest["occluding"],
                show=False,
            ),
        )

    def __save_log_plt(self, fn, post_str=""):
        if self.log_fp:
            fn = self.__save_fn(fn, "P", post_str)
            plt.savefig(fn, bbox_inches="tight")

    def __save_final(self, filename, image):
        self.__save_log_img(filename, image, step_n="F")
        self.__save_log_plt(filename)

    def __draw_target(self, lng, lat, color="tab:red", out_size=15, in_size=3):
        if self.log_fp:
            self.ax.plot(
                lng, lat, color=color, marker="o", markersize=out_size, fillstyle="none"
            )
            self.ax.plot(lng, lat, color=color, marker="o", markersize=in_size)

    def __draw_cross(self, lng, lat, color="tab:red"):
        if self.log_fp:
            self.ax.plot(
                lng, lat, color=color, marker="x", markersize=15, markeredgewidth=2
            )

    def __draw_lines(self, x, y, angles, line_len, **kwargs):
        if self.log_fp:
            for a in angles:
                endx, endy = get_end_coords(x, y, a, line_len)
                self.ax.plot([x, endx], [y, endy], **kwargs)

    def __draw_fov(
        self, lng, lat, heading, fov=None, color="tab:blue", view_len=0.0003
    ):
        if fov is None:
            fov = self.lder.fov
        if self.log_fp:
            kwargs = {"color": color, "linestyle": (0, (5, 10))}
            angles = [heading - fov / 2, heading + fov / 2]
            self.__draw_lines(lng, lat, angles, view_len, **kwargs)

    def __draw_batch_fov(self, lng, lat, batch):
        for b in batch:
            heading = -int(b["fn"].split("_")[3]) + 90
            self.__draw_fov(lng, lat, heading, color="tab:blue")

    def __draw_move_arrow(self, lng, nlng, lat, nlat):
        if self.log_fp:
            self.ax.plot([lng, nlng], [lat, nlat], "tab:orange")
            ang1 = -get_est_heading(lng, lat, nlng, nlat) - 90
            print(f"ANGLE: {ang1}")
            ang2 = ang1 + 25
            ang3 = ang1 - 25
            a1lng, a1lat = get_end_coords(nlng, nlat, ang2, 0.00002)
            self.ax.plot([nlng, a1lng], [nlat, a1lat], "tab:orange")
            a1lng, a1lat = get_end_coords(nlng, nlat, ang3, 0.00002)
            self.ax.plot([nlng, a1lng], [nlat, a1lat], "tab:orange")
            # self.__draw_cross(nlng, nlat, "tab:orange")

    def __draw_obj_span(
        self, lng, lat, heading, edges, fov=None, color="tab:red", view_len=0.0003
    ):
        if fov is None:
            fov = self.lder.fov
        if self.log_fp:
            kwargs = {"color": color, "linewidth": 0.5, "linestyle": "--"}
            angles = [heading + fov / 2 - a for a in edges]
            self.__draw_lines(lng, lat, angles, view_len, **kwargs)

    def __find_draw_obj(self, mask, lng, lat, heading, fov=None):
        if fov is None:
            fov = self.lder.fov
        img_w = mask.shape[1]
        column_sum = mask.sum(axis=0)
        colums_hit = np.nonzero(column_sum)
        left_edge = np.min(colums_hit)
        right_edge = np.max(colums_hit)
        mid_point = (right_edge + left_edge) / 2
        edges_angles = [
            edge / img_w * fov for edge in [left_edge, mid_point, right_edge]
        ]
        # self.__draw_obj_span(lng, lat, heading, edges_angles, fov)

        return mid_point

    def __run_reset(self):
        plt.cla()
        self.curr_step = 0

    def find_biggest(self, preds):
        biggest = {
            "fn": None,
            "interest": None,
            "occluding": None,
            "orig_img": None,
        }
        for p in preds:
            occl = np.zeros(p["orig_img"].shape[:2], dtype=bool)

            intrst_cnt = 0
            for mcntr, (clss, m) in enumerate(zip(p["out"]["class"], p["out"]["mask"])):
                self.__save_log_img(
                    p["fn"],
                    show_mask(p["orig_img"], p_msk=m, show=False),
                    post_str=f"_{mcntr}_{clss}.png",
                )

                if clss in self.rls["occluding"]:
                    occl = np.logical_or(occl, m)

                if clss in self.rls["interest"] and m.sum() > 0:
                    if biggest["fn"] is None or m.sum() > biggest["interest"].sum():
                        biggest = {
                            "fn": p["fn"],
                            "interest": m,
                            "occluding": None,
                            "orig_img": p["orig_img"],
                        }
                    for full in p["full"]:
                        if "p2g" in full:
                            biggest["v_mask"] = full["p2g"]["v_masks"][intrst_cnt]
                            biggest["amodal_masks"] = full["p2g"]["amodal_masks"][intrst_cnt]
                    intrst_cnt += 1

            if biggest["fn"] == p["fn"]:
                biggest["occluding"] = occl

            if not p["out"]["mask"]:
                self.__save_log_img(p["fn"], p["orig_img"], post_str="_no_masks.png")

        return biggest
    
    def move_decision(self, biggest):
        # return True if image is good and there is no more moves needed
        if self.decision == "overlap":
            return self.overlap_decision(biggest)
        elif self.decision == "classifier":
            return self.classifier_decision(biggest)
        else:
            if self.clf:
                if not self.classifier_decision(biggest):
                    return False
            if "occluding" in self.rls:
                if not self.overlap_decision(biggest):
                    return False
            return True

    def overlap_decision(self, biggest):
        overlap = np.logical_and(biggest["interest"], biggest["occluding"]).sum()
        if overlap == 0:
            return True
        else:
            return False
        
    def classifier_decision(self, biggest, log_dcs=False):
        mask_features = self.get_mask_features(biggest["v_mask"], biggest["amodal_masks"])

        clf_pred = self.clf.predict(mask_features)
        print(clf_pred)

        if log_dcs:
            fn = self.__save_fn(biggest["fn"], self.curr_step, "end_state")
            fn = fn.split(".png")[0] + ".txt"
            with open(fn, "w") as f:
                f.write(clf_pred)

        if clf_pred == "Clear":
            return True
        else:
            return False

    def get_mask_features(self, v_mask, pred_mask):

        v_mask_area = np.count_nonzero(v_mask)
        pred_mask_area = np.count_nonzero(pred_mask)

        vx1, vy1, vx2, vy2 = self.bbox2(v_mask)
        px1, py1, px2, py2 = self.bbox2(pred_mask)

        v_height = vy2 - vy1
        v_width = vx2 - vx1
        p_height = py2 - py1
        p_width = px2 - px1

        area_diff = pred_mask_area - v_mask_area
        if v_mask_area == 0:
            area_ratio = pred_mask_area/1
        else:
            area_ratio = pred_mask_area/v_mask_area

        v_ar = (v_width + 1) / (v_height + 1)
        p_ar = (p_width + 1) / (p_height + 1)
        ar_ratio = p_ar / v_ar

        
        return pd.DataFrame({
            "v_mask_area": v_mask_area, "pred_mask_area": pred_mask_area,
            "area_diff": area_diff, "area_ratio": area_ratio,
            "v_ar": v_ar, "p_ar": p_ar, "ar_ratio": ar_ratio,
            "v_height": v_height, "v_width": v_width, "p_height": p_height, "p_width": p_width
            }, index=[0])
    
    def bbox2(self, img):
        if np.count_nonzero(img) == 0:
            return 0, 0, 0, 0
        
        # https://stackoverflow.com/a/31402351
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # return rmin, rmax, cmin, cmax
        return rmin, cmin, rmax, cmax # changed to xyxy format

    def GSV_move(
        self, lng, lat, heading, strat="orthor", mid_point=0, img_w=640, repo_len=0.0001
    ):
        if strat == "backup":
            adj_angl = heading - 180 + 45 - mid_point / img_w * self.lder.fov
        if strat == "orthor":
            adj_angl = heading - 90
        if strat == "orthol":
            adj_angl = heading + 90

        return get_end_coords(lng, lat, adj_angl, repo_len)

    def run_GSV(self, pid, batch):
        # Original picture
        plat, plng = self.lder.fetch_latlng(pid)
        print(f"CSV lat: {plat} lng: {plng}")
        self.__draw_target(plng, plat)

        if not batch:
            print("No response")
            return

        lat = batch[0]["metadata"]["location"]["lat"]
        lng = batch[0]["metadata"]["location"]["lng"]
        print(f"{self.lder.source} lat: {lat} lng: {lng}")
        self.__draw_cross(lng, lat, "tab:blue")
        # self.__draw_target(lng, lat, "tab:blue", 10, 0)
        self.__save_log_batch(batch)

        self.curr_step = 1
        preds = self.pder.predict(batch)

        # print(preds[0])

        # show_masks_indiv(preds, self.rls)
        # show_masks_comb(preds, self.rls)

        biggest = self.find_biggest(preds)

        # If no picture is found
        if biggest["fn"] is None:
            print(f"No {self.rls['interest'][0]} found at location")
            self.__draw_batch_fov(lng, lat, batch)
            self.__save_log_plt(batch[0]["fn"])
            return

        self.curr_step = 2
        self.__save_log_biggest(biggest)

        # print(f"File: {largest['fn']}")
        # Turns gsv heading into angle from horizontal
        # 0->90, 90->0, 180->-90, 270->-180, 360->-270
        heading = -int(biggest["fn"].split("_")[3]) + 90
        print(f"heading: {heading}")

        self.__draw_fov(lng, lat, heading, color="tab:blue")

        self.__find_draw_obj(biggest["interest"], lng, lat, heading)

        good_image = self.move_decision(biggest)
        if good_image:
            self.__save_final(biggest["fn"], biggest["orig_img"])
            return

        # First move
        self.curr_step = 3
        nlng, nlat = self.GSV_move(lng, lat, heading)

        self.__draw_move_arrow(lng, nlng, lat, nlat)

        est_heading = get_est_heading(nlng, nlat, plng, plat)

        _, new_loc = self.lder.results_from_loc(nlat, nlng, est_heading)

        clat = new_loc.metadata[0]["location"]["lat"]
        clng = new_loc.metadata[0]["location"]["lng"]

        self.__draw_cross(clng, clat, "tab:cyan")

        est_heading = get_est_heading(clng, clat, plng, plat)

        self.curr_step = 4
        new_pic = self.lder.pic_from_loc(pid, clat, clng, est_heading)

        est_heading = -est_heading + 90
        self.__draw_fov(clng, clat, est_heading, color="tab:cyan")

        preds = self.pder.predict(new_pic)
        biggest = self.find_biggest(preds)

        self.curr_step = 5
        if biggest["fn"]:
            self.__save_log_biggest(biggest)
            good_image = self.move_decision(biggest)
            if good_image:
                self.__save_final(biggest["fn"], biggest["orig_img"])
                return

        # Second move
        self.curr_step = 6
        nlng, nlat = self.GSV_move(lng, lat, heading, "orthol")

        self.ax.plot([lng, nlng], [lat, nlat], "tab:brown")
        self.__draw_cross(nlng, nlat, "tab:brown")

        est_heading = get_est_heading(nlng, nlat, plng, plat)

        _, new_loc = self.lder.results_from_loc(nlat, nlng, est_heading)

        clat = new_loc.metadata[0]["location"]["lat"]
        clng = new_loc.metadata[0]["location"]["lng"]

        self.__draw_cross(clng, clat, "tab:cyan")

        est_heading = get_est_heading(clng, clat, plng, plat)

        self.curr_step = 7
        new_pic = self.lder.pic_from_loc(pid, clat, clng, est_heading)

        est_heading = -est_heading + 90
        self.__draw_fov(clng, clat, est_heading, color="tab:cyan")

        self.__save_final(new_pic[0]["fn"], new_pic[0]["img"])

        if self.save_decision:
            preds = self.pder.predict(new_pic)
            biggest = self.find_biggest(preds)
            if biggest["fn"]:
                _ = self.classifier_decision(biggest, True)
            else:
                fn = self.__save_fn(batch[0]["fn"], self.curr_step, "end_state")
                fn = fn.split(".png")[0] + ".txt"
                with open(fn, "w") as f:
                    f.write("No pole found")

    def run_DCam(self, pid, batch):
        plat, plng = self.lder.fetch_latlng(pid)
        print(f"CSV lat: {plat} lng: {plng}")
        self.__draw_target(plng, plat)

        if not batch:
            print("No response")
            return

        lat = batch[0]["metadata"]["location"]["lat"]
        lng = batch[0]["metadata"]["location"]["lng"]
        print(f"{self.lder.source} lat: {lat} lng: {lng}")
        self.__draw_cross(lng, lat, "tab:blue")
        self.__save_log_batch(batch)
        bfn = batch[0]["fn"]  # quick fix thing to save filename

        self.curr_step = 1
        preds = self.pder.predict(batch)

        # print(preds[0])

        # show_masks_indiv(preds, self.rls)
        # show_masks_comb(preds, self.rls)

        biggest = self.find_biggest(preds)

        if biggest["fn"] is None:
            print(f"No {self.rls['interest'][0]} found at location")
            self.__draw_batch_fov(lng, lat, batch)
            self.__save_log_plt(batch[0]["fn"])
            return

        self.curr_step = 2
        self.__save_log_biggest(biggest)

        # print(f"File: {largest['fn']}")

        # Turns gsv heading into angle from horizontal
        # 0->90, 90->0, 180->-90, 270->-180, 360->-270
        heading = -int(biggest["fn"].split("_")[3]) + 90
        print(f"heading: {heading}")

        self.__draw_fov(lng, lat, heading, color="tab:blue")

        self.__find_draw_obj(biggest["interest"], lng, lat, heading)

        good_image = self.move_decision(biggest)
        if good_image:
            self.__save_final(biggest["fn"], biggest["orig_img"])
            return

        track, curr = batch[0]["metadata"]["entry"][["Filename", "Point"]].values[0]
        while curr:
            self.curr_step += 1
            curr -= 1
            batch = self.lder.pic_from_track(pid, track, curr)
            clat = batch[0]["metadata"]["location"]["lat"]
            clng = batch[0]["metadata"]["location"]["lng"]
            self.__draw_cross(clng, clat, "tab:cyan")
            self.__save_log_batch(batch)
            if self.lder.get_distance(lng, lat, clng, clat) > 0.001:
                break

            self.curr_step += 1
            preds = self.pder.predict(batch)
            biggest = self.find_biggest(preds)

            if biggest["fn"] is None:
                print(f"No {self.rls['interest'][0]} found at location")
                break

            self.curr_step += 1
            self.__save_log_biggest(biggest)

            # print(f"File: {largest['fn']}")

            # Turns gsv heading into angle from horizontal
            # 0->90, 90->0, 180->-90, 270->-180, 360->-270
            heading = -int(biggest["fn"].split("_")[3]) + 90
            print(f"heading: {heading}")

            self.__draw_fov(clng, clat, heading, color="tab:cyan")
            self.__find_draw_obj(biggest["interest"], clng, clat, heading)

            good_image = self.move_decision(biggest)
            if good_image:
                self.__save_log_img(biggest["fn"], biggest["orig_img"], step_n="F")
                break

        self.__save_log_plt(bfn)

    def run(self, iterations=None):
        self.lder.iters = iterations
        for pcount, pid, batch in self.lder:
            self.__run_reset()
            print(f"\nPole count: {pcount}, Pole ID: {pid}")
            if self.lder.source == "GSV":
                self.run_GSV(pid, batch)
            elif self.lder.source == "Dashcam":
                self.run_DCam(pid, batch)


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
