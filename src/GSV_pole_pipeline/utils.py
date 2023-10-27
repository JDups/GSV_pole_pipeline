import cv2
import numpy as np


def get_pole_id_from_filename(filename_series):
    return filename_series.str.split("_").str[1]


def show_mask(image, p_msk=None, n_msk=None, show=True):
    p_clr = np.array([0.90, 0.05, 0.05, 0.75])
    n_clr = np.array([0.05, 0.50, 0.50, 0.75])

    if p_msk is not None:
        if not isinstance(p_msk, list):
            p_msk = [p_msk]
        for msk in p_msk:
            overlay = get_overlay(msk, p_clr)
            image = cv2.addWeighted(image, 1, overlay, 0.55, 0)

    if n_msk is not None:
        if not isinstance(n_msk, list):
            n_msk = [n_msk]
        for msk in n_msk:
            overlay = get_overlay(msk, n_clr)
            image = cv2.addWeighted(image, 1, overlay, 0.55, 0)

    if show:
        cv2.imshow("img", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def get_overlay(msk, color):
    overlay = np.zeros((msk.shape[0], msk.shape[1], 4))
    overlay[msk] = color
    overlay[:, :, 3] = 0
    return (overlay[:, :, :-1] * 255).astype(np.uint8)
