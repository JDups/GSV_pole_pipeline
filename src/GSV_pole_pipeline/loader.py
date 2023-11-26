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
from utils import get_est_heading
import os
import pickle

"""
TODO: Add OK status check to GSV API repsonses. I kinda added a check but it's jank
TODO: Use Path objects instead of strings for filepaths
"""


class Loader(ABC):
    def __init__(self):
        self.log_fp = ""

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
        self.log_fp = None
        self.saved_queries = {}
        self.saved_imgs = {}
        self.api_defaults = {
            "size": "640x640",
            "fov": "90",
            "pitch": "10",
            "key": self.API_key,
            "source": "outdoor",
        }

    def set_log_fp(self, fp):
        self.log_fp = fp
        self.cache_loc = self.log_fp + "saved_queries" + os.sep
        os.makedirs(self.cache_loc, exist_ok=True)
        try:
            with open(self.cache_loc + "saved_queries.pkl", "rb") as f:
                self.saved_queries = pickle.load(f)
        except:
            self.saved_queries = {}
        try:
            with open(self.cache_loc + "saved_imgs.pkl", "rb") as f:
                self.saved_imgs = pickle.load(f)
        except:
            self.saved_imgs = {}

    def __save_response(self, api_list, api_results):
        if self.log_fp:
            self.saved_queries[str(api_list)] = api_results
            with open(self.cache_loc + "saved_queries.pkl", "wb") as f:
                pickle.dump(self.saved_queries, f)

    def __save_image(self, link, img):
        if self.log_fp:
            self.saved_imgs[link] = img
            with open(self.cache_loc + "saved_imgs.pkl", "wb") as f:
                pickle.dump(self.saved_imgs, f)

    def results_from_loc(self, lat, lng, heading=None):
        apiargs = self.api_defaults.copy()
        apiargs["location"] = f"{lat},{lng}"
        apiargs["heading"] = str(heading)

        api_list = gsv.helpers.api_list(apiargs)

        if str(api_list) in self.saved_queries:
            api_results = self.saved_queries[str(api_list)]
        else:
            api_results = gsv.api.results(api_list)
            self.__save_response(api_list, api_results)

        return api_list, api_results

    def image_from_GSV(self, link):
        if link in self.saved_imgs:
            api_results = self.saved_imgs[link]
            img = self.saved_imgs[link]
        else:
            img = np.array(Image.open(BytesIO(requests.get(link).content)))
            self.__save_image(link, img)

        return img

    def pic_from_loc(self, idn, lat, lng, heading=None):
        api_list, api_results = self.results_from_loc(lat, lng, heading)
        # print(api_results.metadata)

        if api_results.metadata[0]["status"] != "OK":
            with open(self.log_fp + f"p{idn}_sN", "w") as f:
                f.write(api_results.metadata[0]["status"])
            return None

        rlat = api_results.metadata[0]["location"]["lat"]  # real Latitude
        rlng = api_results.metadata[0]["location"]["lng"]  # real Longitude
        est_heading = get_est_heading(rlng, rlat, lng, lat)

        if not heading:
            api_list[0]["heading"] = est_heading

        fn = [
            f"pole_{idn}_heading_{args['heading']}_pitch_{args['pitch']}_zoom_0_fov_{args['fov']}"
            for args in api_list
        ]

        return [
            self.output_dict(
                fn=fn,
                # img=np.array(Image.open(BytesIO(requests.get(link).content))),
                img=self.image_from_GSV(link),
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
