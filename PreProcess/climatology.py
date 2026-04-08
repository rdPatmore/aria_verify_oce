import xarray as xr
import glob
import numpy as np

class make_glosat_climatology(object):
    """
    make climatology from glosat runs
    """

    def __init__(self):
        self.root = "/gws/ssde/j25a/verify_oce/NEMO/Preprocessing/"
        

    def surface_clim(self, y0, y1):
        """
        """

        var_list = [
                   "u10",
                   "v10",
                   "t1500mm",
                   "mslp",
                   "msdwlwrf",
                   "msdwswrf",
                   "msr",
                   "mtpr",
                   "sph"
                   ]
        
        for var in var_list:
            ds_set = []
            for y in range(y0, y1):
                path = glob.glob(self.root + "SBC/" + f"*{var}_y{y}.nc")[0]
                ds = xr.open_dataset(path)
                dates = np.arange(f"{y}-01",f"{y+1}-01",dtype="datetime64[M]")
                ds = ds.assign_coords(time=dates)
                ds_set.append(ds)
                

            ds = xr.concat(ds_set, "time")
            ds_mean = ds.groupby("time.month").mean()

            save_path = self.root + "SBC/" + f"glosat_clim_{var}_{y0}_{y1}.nc"
            ds_mean.to_netcdf(save_path)



mgc = make_glosat_climatology()
mgc.surface_clim(1850,1870)
