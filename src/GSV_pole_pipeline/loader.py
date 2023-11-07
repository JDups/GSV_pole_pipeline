# from utils import get_pole_id_from_filename
from glob import glob
import pandas as pd
import cv2
import google_streetview as gsv
import google_streetview.helpers
import google_streetview.api
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import math
from abc import ABC, abstractmethod

"""
TODO: Loader log_fp setter is useless, should probably just acess attribute directly

"""


class Loader(ABC):
    def __init__(self):
        self.log_fp = ""

    def set_log_fp(self, log_fp):
        self.log_fp = log_fp

    @abstractmethod
    def get_batch(self, idn):
        pass

    def output_dict(self, fn=None, img=None, mtdt=None):
        return {
            "fn": fn,
            "img": img,
            "metadata": mtdt,
        }


class ImgFetch(Loader):
    __supported_image_formats = ["png", "jpg"]

    def __init__(self, directory=None):
        super().__init__()
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

        self.data_df = pd.DataFrame({"img_fp": pole_imgs})
        self.data_df["img_fn"] = self.data_df["img_fp"].str.split(self.dir_div).str[-1]
        self.data_df["pole_id"] = self.data_df["img_fn"].str.split("_").str[1]

    def get_batch(self, idn):
        idn = str(idn)

        imgs = [
            cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
            for fp in self.data_df[self.data_df["pole_id"] == idn]["img_fp"]
        ]

        return [
            self.output_dict(fn=fn, img=img)
            for img, fn in zip(
                imgs, self.data_df[self.data_df["pole_id"] == idn]["img_fn"]
            )
        ]


class GSVFetch(Loader):
    def __init__(self, csv_file, API_key, full_360=False):
        super().__init__()
        self.data_df = pd.read_csv(csv_file)
        self.API_key = API_key
        self.full_360 = full_360
        self.api_defaults = {
            "size": "640x640",
            "fov": "90",
            "pitch": "10",
            "key": self.API_key,
            "source": "outdoor",
        }

    def pic_from_loc(self, idn, lat, lng, heading=None):
        apiargs = self.api_defaults.copy()
        apiargs["location"] = f"{lat},{lng}"
        apiargs["heading"] = str(heading)

        api_list = gsv.helpers.api_list(apiargs)
        api_results = gsv.api.results(api_list)
        if self.log_fp:
            api_results.save_links(self.log_fp + "links.txt")
            api_results.save_metadata(self.log_fp + "metadata.json")
        # print(api_results.metadata)

        rlat = api_results.metadata[0]["location"]["lat"]
        rlng = api_results.metadata[0]["location"]["lng"]
        dlat = lat - rlat
        dlng = lng - rlng
        est_heading = int((-math.degrees(math.atan2(dlat, dlng)) + 90) % 360)

        if not heading:
            api_list[0]["heading"] = est_heading

        fn = [
            f"pole_{idn}_heading_{args['heading']}_pitch_{args['pitch']}_zoom_0_fov_{args['fov']}"
            for args in api_list
        ]

        return [
            self.output_dict(
                fn=fn,
                img=np.array(Image.open(BytesIO(requests.get(link).content))),
                mtdt=mtdt,
            )
            for link, fn, mtdt in zip(api_results.links, fn, api_results.metadata)
        ]

    def pic_from_loc_360(self, idn, lat, lng):
        return self.pic_from_loc(idn, lat, lng, "0;90;180;270")

    def get_batch(self, idn):
        row = self.data_df[self.data_df["pole_id"] == idn]
        lat = row["Latitude"].values[0]
        lng = row["Longitude"].values[0]
        print(f"Lat: {lat}  Long: {lng}")

        if self.full_360:
            return self.pic_from_loc_360(idn, lat, lng)
        else:
            return self.pic_from_loc(idn, lat, lng)
