def get_pole_id_from_filename(filepath_series):
    if "/" in filepath_series.iloc[0]:
        dir_div = "/"
    if "\\" in filepath_series.iloc[0]:
        dir_div = "\\"

    return filepath_series.str.split(dir_div).str[-1].str.split("_").str[1]
