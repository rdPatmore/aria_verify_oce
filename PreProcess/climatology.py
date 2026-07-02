import xarray as xr
import glob
import numpy as np
from dask.diagnostics import ProgressBar

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

    def lbc_oce_clim(self, y0, y1):

        case = "MES_clean3"
        var_dict = {"T": ["votemper","vosaline","sossheig"],
                    "U": ["vobtcrtx","vozocrtx"],
                    "V": ["vobtcrty","vomecrty"]}
        for grid in ["T","U","V"]:
            ds_set = []
            for y in range(y0, y1):
                print (y)
                paths = glob.glob(self.root + f"LBC/OCE/{case}/" +
                                  f"*bdy{grid}_y{y}*.nc")
                ds_inner_set = []
                for path in paths:
                    ds = xr.open_dataset(path, chunks={"time_counter":1})
                    ds_inner_set.append(ds)
                ds = xr.concat(ds_inner_set, "time_counter", 
                               data_vars="minimal")
                #ds = xr.open_mfdataset(paths, chunks={"time":1, "z":1})
                dates = np.arange(f"{y}-01",f"{y+1}-01",dtype="datetime64[M]")
                ds = ds.assign_coords(time_counter=dates)
                ds_set.append(ds)
                

            ds = xr.concat(ds_set, "time_counter", data_vars="minimal")

            encoding = {}
            for var in var_dict[grid]:
                print (var)
                ds[var] = ds[var].groupby("time_counter.month").mean()
                encoding[var] = {"_FillValue":-1e20}
            ds = ds.drop_vars("time_counter")
            
            #ds = ds.rename({"month":"time_counter"})
            #ds = ds.drop_vars("time_counter")

            with ProgressBar():
                ds = ds.load()

            save_path = self.root + f"LBC/OCE/{case}/" + \
                               f"GloSat_NAARC_MES_clim_bdy{grid}_{y0}_{y1}.nc"
            ds.to_netcdf(save_path, unlimited_dims="month", encoding=encoding)

    def lbc_ice_clim(self, y0, y1):

        ds_set = []
        for y in range(y0, y1):
            paths = glob.glob(self.root + f"LBC/ICE/" +
                              f"*bdyT_y{y}*.nc")
            ds_inner_set = []
            for path in paths:
                ds = xr.open_dataset(path, chunks={"time_counter":1})
                ds_inner_set.append(ds)
            ds = xr.concat(ds_inner_set, "time_counter", 
                           data_vars="minimal")
            dates = np.arange(f"{y}-01",f"{y+1}-01",dtype="datetime64[M]")
            ds = ds.assign_coords(time_counter=dates)
            ds_set.append(ds)
            

        ds = xr.concat(ds_set, "time_counter", data_vars="minimal")

        encoding = {}
        for var in ["siconc","sithic","snthic"]:
            print (var)
            ds[var] = ds[var].groupby("time_counter.month").mean()
            encoding[var] = {"_FillValue":-1e20}
        ds = ds.drop_vars(["time_counter","votemper","vosaline","sossheig"])

        with ProgressBar():
            ds = ds.load()

        save_path = self.root + f"LBC/ICE/" + \
                           f"GloSat_NAARC_MES_clim_bdyI_{y0}_{y1}.nc"
        ds.to_netcdf(save_path, unlimited_dims="month", encoding=encoding)

mgc = make_glosat_climatology()
mgc.lbc_oce_clim(1950,1970)
mgc.lbc_ice_clim(1950,1970)
