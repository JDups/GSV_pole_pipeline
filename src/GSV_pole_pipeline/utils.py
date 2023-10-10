def get_pole_id_from_filename(filepath):
	return filepath.str.split("/").str[-1].str.split("_").str[1].astype(int)