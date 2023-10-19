# from utils import get_pole_id_from_filename
from glob import glob
import pandas as pd
import cv2


class ImgFetch:
    __supported_image_formats = ["png", "jpg"]

    def __init__(self, directory=None):
        self.directory = directory
        self.dir_div = "/"

        if self.directory:
            self.add_directory(self.directory)

    def add_directory(self, directory):
        if "\\" in self.directory:
            self.dir_div = "\\"
        fl_tp = [f"*.{f}" for f in ImgFetch.__supported_image_formats]

        pole_imgs = []
        for fl in fl_tp:
            pole_imgs.extend(glob(directory + fl))

        self.pole_imgs_df = pd.DataFrame({"img_fp": pole_imgs})
        self.pole_imgs_df["img_fn"] = (
            self.pole_imgs_df["img_fp"].str.split(self.dir_div).str[-1]
        )
        self.pole_imgs_df["pole_id"] = self.pole_imgs_df["img_fn"].str.split("_").str[1]

    def get_batch(self, idn):
        idn = str(idn)

        imgs = [
            cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            for fp in self.pole_imgs_df[self.pole_imgs_df["pole_id"] == idn]["img_fp"]
        ]

        return [
            {"img": img, "fn": fn}
            for img, fn in zip(
                imgs, self.pole_imgs_df[self.pole_imgs_df["pole_id"] == idn]["img_fn"]
            )
        ]
