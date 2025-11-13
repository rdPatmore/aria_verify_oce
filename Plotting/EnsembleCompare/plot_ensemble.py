import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
import glob
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

class glosat_ensemble_analysis(object):
    def __init__(self):
        self.glosat_path = "/gws/nopw/j04/glosat/production/UKESM/raw/"
        self.save_path = "/gws/nopw/j04/verify_oce/NEMO/PostProcessing/"
        
        self.ensemble_list=["u-ck651",
                            "u-co986",
                            "u-cp625",
                            "u-cu100",
                            "u-cu101",
                            "u-cu102"]
        self.t0_range=["1850","1870"]
        self.t1_range=["1940","1960"]

        self.dom_path = "/gws/nopw/j04/verify_oce/NEMO/Preprocessing/DOM/UKESM/domcfg_UKESM1p1_gdept.nc"

    def preprocess(self, ds):
        ds = ds.aice
        #ds = ds.aice.sum(["nj","ni"])
        return ds

    #def get_aice(self, path):
    #    print (path)
    #    ds = xr.open_dataset(path, chunks=-1, decode_times=False)
    #    print (ds)
    #    return ds.aice
    
    def create_sea_ice_area_sum(self):
        ensemble_datasets = []
        for ensemble_member in self.ensemble_list:
            print (ensemble_member)
            paths = glob.glob(self.glosat_path + ensemble_member + "/18*/cice*.nc")
            #futures = client.map(self.get_aice, paths)
            #ds = client.gather(futures)
            #print (ds)

            times = []
            for path in paths:
                print (path)
                ds = xr.open_dataset(path, chunks="auto",
                             decode_times=False)
                times.append(ds.aice.sum(["nj","ni"]).load())
            ds = xr.concat(times, "time")
            ds = ds.expand_dims(ensemble=[ensemble_member])

            ensemble_datasets.append(ds)
        ds_all = xr.concat(ensemble_datasets, "ensemble")
        
        with ProgressBar():
            ds_all.to_netcdf(self.save_path + "glosat_sea_ice_area_ensemble.nc")

    def render_ensemble_sea_ice(self):
        ds = xr.open_dataset(self.save_path + "glosat_sea_ice_area_ensemble.nc",
                decode_times=True)
        times = xr.open_dataset(self.save_path + "glosat_sea_ice_area_ensemble.nc",
                decode_times=False).time
        ds = ds.assign_coords({"time_num": ("time", times.data)})

        fig, axs = plt.subplots(1, figsize=(6.5,5))
        
        for ensemble, ds_member in ds.groupby("ensemble"):
            ice_ts_cycle = []
            for i, (year, ds_year) in enumerate(ds_member.groupby("time.year")):
                ice_min = ds_year.aice.min()
                ice_max = ds_year.aice.max()

                ice_ts_cycle.append(ice_max-ice_min)
                #ticks = plt.gca().get_xticks()
                #times = xr.DataArray(ticks, dims="time", attrs=ds.time.attrs,
                #                     name="time")
                #times = xr.decode_cf(times.to_dataset()).time.dt.strftime("%Y-%m-%d")
                #
                #plt.gca().set_xticklabels(times.data)
            axs.plot(np.arange(1850,1900), ice_ts_cycle)

    def get_mean_glosat_variable(self, var="tos",
             y0=1850, y1=1870, dir_str="18[5-7]"):

        def preProcess(ds):
            ds = ds.drop_vars("time_counter")
            ds = ds.swap_dims({"time_counter":"time_centered"})
            return ds
        path_set = []
        for y in range(y0, y1):
            print (y)
            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y) + f"*/*1m_{y}*grid-T.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y+1) + f"*/*1m_{y}*grid-T.nc")
            paths = paths0 + paths1

            path_set += paths

        def get_da(path):
            da = xr.open_dataset(path, chunks={"time_counter":1})[var]
            da = da.drop_vars("time_counter")
            da = da.swap_dims({"time_counter":"time_centered"})
            return da

        da_series = get_da(path_set[0])
        for path in path_set[1:]:
            da = get_da(path)
            da_series = xr.concat([da_series, da], dim="time_centered")

        with ProgressBar():
            da_mean = da_series.groupby("time_centered.month").mean().load()

        da_mean.to_netcdf(self.save_path + f"glosat_mean_{var}_{y0}_{y1}.nc")

    def plot_mean_glosat_change(self, var, label, unit, vmin, vmax):
        """
        plot surface temperature change
        """
        
        # get datasets
        da_t0 = xr.open_dataarray(self.save_path + f"glosat_mean_{var}_" + 
                              self.t0_range[0] + "_" + self.t0_range[1] + ".nc")
        da_t1 = xr.open_dataarray(self.save_path + f"glosat_mean_{var}_" + 
                              self.t1_range[0] + "_" + self.t1_range[1] + ".nc")

        # get seasonal means
        da_t0_DJF = da_t0.sel(month=[1,2,12]).mean("month")
        da_t1_DJF = da_t1.sel(month=[1,2,12]).mean("month")
        da_t0_JJA = da_t0.sel(month=[6,7,8]).mean("month")
        da_t1_JJA = da_t1.sel(month=[6,7,8]).mean("month")

        # get differences
        diff_DJF = da_t1_DJF - da_t0_DJF
        diff_JJA = da_t1_JJA - da_t0_JJA

        # initialise figure
        proj = ccrs.PlateCarree()
        proj_dict={"projection":ccrs.Orthographic(-30,60)}
        fig, axs = plt.subplots(3,2, figsize=(6.5,8), subplot_kw=proj_dict)
        plt.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.05)

        def render(ax, da, tmin, tmax, cmap, proj, label=""):
            pn = ax.pcolormesh(da.nav_lon, da.nav_lat, da, transform=proj,
                               vmin=tmin, vmax=tmax, cmap=cmap)
            cb = plt.colorbar(pn, ax=ax, extend="both")
            cb.ax.set_ylabel(label)

        # set colourbar lims and map
        cmap=plt.cm.viridis

        # render means
        render(axs[0,0], da_t0_DJF, vmin, vmax, cmap, proj,
               f"{label} 1850-1870")
        render(axs[1,0], da_t1_DJF, vmin, vmax, cmap, proj,
               f"{label} 1940-1960")
        render(axs[0,1], da_t0_JJA, 0, 50, cmap, proj,
               f"{label} 1850-1870")
        render(axs[1,1], da_t1_JJA, 0, 50, cmap, proj,
               f"{label} 1940-1960")

        # set colourbar lims and map
        tmin, tmax = -200, 200 
        cmap=cmocean.cm.balance
        
        # render differnces
        render(axs[2,0], diff_DJF, tmin, tmax, cmap, proj,
               f"Mean {label} Change")
        tmin, tmax = -20, 20 
        render(axs[2,1], diff_JJA, tmin, tmax, cmap, proj,
               f"Mean {label} Change")

        # add land 
        for i, ax in enumerate(axs.flatten()):
            ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        # set titles
        for ax in axs[:,0]:
            ax.set_title("DJF")
        for ax in axs[:,1]:
            ax.set_title("JJA")

        plt.savefig(self.save_path + f"glosat_{var}_change.png", dpi=600)

if __name__ == "__main__":
    gea = glosat_ensemble_analysis()
    #gea.get_mean_glosat_variable(y0=1940, y1=1960, var="somxl010")
    #gea.get_mean_glosat_variable(y0=1940, y1=1960, var="tos")
    gea.plot_mean_glosat_change(var="somxl010", label="Mixed Layer Depth",
                                unit="m", vmin=0, vmax=500)
    #gea.create_sea_ice_area_sum()

    #gea.render_ensemble_sea_ice()
    #plt.show()
#plt.savefig("sea_ice_area.png")
