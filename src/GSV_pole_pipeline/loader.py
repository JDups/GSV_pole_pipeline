from utils import get_pole_id_from_filename
from glob import glob
import pandas as pd
import cv2


class img_fetch:
    __supported_image_formats = ["png", "jpg"]

    def __init__(self, directory=None):
        self.directory = directory

        if self.directory:
            self.add_directory(self.directory)

    def add_directory(self, directory):
        fl_tp = [f"*.{f}" for f in self.__supported_image_formats]
        pole_pics = []
        for fl in fl_tp:
            pole_pics.extend(glob(directory + fl))

        self.pole_pics_df = pd.DataFrame({"pole_fp": pole_pics})
        self.pole_pics_df["pole_id"] = get_pole_id_from_filename(
            self.pole_pics_df["pole_fp"]
        )

    def get_batch(self, idn):
        idn = str(idn)
        imgs_fp = self.pole_pics_df[self.pole_pics_df["pole_id"] == idn][
            "pole_fp"
        ].tolist()
        imgs = [cv2.imread(fp) for fp in imgs_fp]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        return [
            {"img": img, "fp": fp.split("\\")[-1]} for img, fp in zip(imgs, imgs_fp)
        ]
