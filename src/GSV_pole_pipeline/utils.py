def get_pole_id_from_filename(filename_series):
    return filename_series.str.split("_").str[1]
