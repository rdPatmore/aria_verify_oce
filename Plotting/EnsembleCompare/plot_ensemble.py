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
        self.t0_range=["1850-01","1869-12"]
        self.t1_range=["1940-01","1959-12"]

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

    def get_mean_surface_glosat_temperature(self, 
             date_range=["1850-01","1869-12"], dir_str="18[5-7]"):

        def preProcess(ds):
            ds = ds.tos
            ds = ds.drop_vars("time_counter")
            ds = ds.swap_dims({"time_counter":"time_centered"})
            print (ds)
            return ds
        #paths = glob.glob(self.glosat_path + self.ensemble_list[0] + "/18[5-6]*/*grid-T.nc")
        paths = glob.glob(self.glosat_path + self.ensemble_list[0] +"/" +
                          dir_str + "*/*_1m_*grid-T.nc")
        ds_series = xr.open_mfdataset(paths, chunks=dict(time_centered=1),
                        combine="nested",  preprocess=preProcess,
                        concat_dim="time_centered")
        ds_series = ds_series.sel(time_centered=slice(date_range[0],
                                                      date_range[1]))

        with ProgressBar():
            ds_mean = ds_series.groupby("time_centered.month").mean().load()

        ds_mean.to_netcdf(self.save_path + "glosat_mean_temp_" + date_range[0] + "_" + date_range[1] + ".nc")

    def plot_mean_surface_glosat_temperature_change(self):
        """
        plot surface temperature change
        """
        
        ds_t0 = xr.open_dataarray(self.save_path + "glosat_mean_temp_" + 
                              self.t0_range[0] + "_" + self.t0_range[1] + ".nc")
        ds_t1 = xr.open_dataarray(self.save_path + "glosat_mean_temp_" + 
                              self.t1_range[0] + "_" + self.t1_range[1] + ".nc")
        print (ds_t0)
        ds_t0_DJF = ds_t0.sel(month=[1,2,12]).mean("month")
        ds_t1_DJF = ds_t1.sel(month=[1,2,12]).mean("month")

        ds_t0_JJA = ds_t0.sel(month=[6,7,8]).mean("month")
        ds_t1_JJA = ds_t1.sel(month=[6,7,8]).mean("month")
        print (ds_t0_JJA)

        diff_DJF = ds_t1_DJF - ds_t0_DJF
        diff_JJA = ds_t1_JJA - ds_t0_JJA

        dom_cfg = xr.open_dataset(self.dom_path).squeeze()
        lon = dom_cfg.glamt
        lat = dom_cfg.gphit
        print (list(dom_cfg.data_vars))

        a = xr.open_dataset("/gws/nopw/j04/jmmp/GOSI9/eORCA1/ocean_annual/nemo_cw234o_1y_20031201-20041201_grid-T.nc")
        print (a)

        proj = ccrs.PlateCarree()
        proj_dict={"projection":ccrs.Orthographic(-30,60)}
        fig, axs = plt.subplots(3,2, figsize=(6.5,10),
                subplot_kw=proj_dict)
                #subplot_kw={"projection":ccrs.Orthographic(-10,45)})
        #ax = plt.axes(projection=ccrs.PlateCarree())

        #axs[0,0].pcolor(ds_t0_DJF.nav_lon, ds_t0_DJF.nav_lat, ds_t0_DJF,
        #p = ax.pcolor(a.nav_lon, a.nav_lat, a.zos.squeeze(), transform=proj)
        #axs[0,0].coastlines()
        #axs[0,1].pcolor(ds_t1_DJF.nav_lon, ds_t1_DJF.nav_lat, ds_t1_DJF,
        #    transform=ccrs.PlateCarree(), vmin=-10, vmax=10)
        #axs[0,2].pcolor(diff_DJF.nav_lon, diff_DJF.nav_lat, diff_DJF, 
        #               vmin=-2, vmax=2, cmap=plt.cm.RdBu_r,
        #    transform=ccrs.PlateCarree())
        #               

        #axs[1,0].pcolor(ds_t0_JJA.nav_lon, ds_t0_JJA.nav_lat, ds_t0_JJA,
        #    transform=ccrs.PlateCarree(), vmin=-10, vmax=10)
        #axs[1,1].pcolor(ds_t1_JJA.nav_lon, ds_t1_JJA.nav_lat, ds_t1_JJA,
        #    transform=ccrs.PlateCarree(), vmin=-10, vmax=10)
        #axs[1,2].pcolor(diff_JJA.nav_lon, diff_JJA.nav_lat, diff_JJA, 
        #               vmin=-2, vmax=2, cmap=plt.cm.RdBu_r,
        #    transform=ccrs.PlateCarree())

        tmin, tmax = -10, 30
        cmap=cmocean.cm.thermal
        def render(ax, ds, tmin, tmax, cmap):

        pn = axs[0,0].pcolormesh(ds_t0_DJF.nav_lon, ds_t0_DJF.nav_lat,
                                 ds_t0_DJF, transform=proj,
                                 vmin=tmin, vmax=tmax, cmap=cmap)
                #vmin=tmin, vmax=tmax, cmap=cmap, add_colorbar=False)
        #pn = ds_t0_DJF.plot(x="nav_lon", y="nav_lat", transform=proj, ax=axs[0,0],
        cb = plt.colorbar(pn, ax=axs[0,0], extend="both")
        cb.ax.set_ylabel("Surface Temperature 1850-1870")
        
        #pn = ds_t1_DJF.plot(x="nav_lon", y="nav_lat", transform=proj, ax=axs[1,0],
        #        vmin=tmin, vmax=tmax, cmap=cmap, add_colorbar=False)
        #cb = plt.colorbar(pn, ax=axs[1,0], extend="both")
        #cb.ax.set_ylabel("Surface Temperature 1940-1960")

        #pn = ds_t0_JJA.plot(x="nav_lon", y="nav_lat", transform=proj, ax=axs[0,1],
        #        vmin=tmin, vmax=tmax, cmap=cmap, add_colorbar=False)
        #cb = plt.colorbar(pn, ax=axs[0,1], extend="both")
        #cb.ax.set_ylabel("Surface Temperature 1850-1870")

        #pn = ds_t1_JJA.plot(x="nav_lon", y="nav_lat", transform=proj, ax=axs[1,1],
        #        vmin=tmin, vmax=tmax, cmap=cmap, add_colorbar=False)
        #cb = plt.colorbar(pn, ax=axs[1,1], extend="both")
        #cb.ax.set_ylabel("Surface Temperature 1940-1960")

        #pn = diff_DJF.plot(x="nav_lon", y="nav_lat", transform=proj, ax=axs[2,0],
        #        add_colorbar=False)
        #cb = plt.colorbar(pn, ax=axs[2,0], extend="both")
        #cb.ax.set_ylabel("Mean Surface Temperature Change")

        #pn = diff_JJA.plot(x="nav_lon", y="nav_lat", transform=proj, ax=axs[2,1],
        #        add_colorbar=False)
        #cb = plt.colorbar(pn, ax=axs[2,1], extend="both")
        #cb.ax.set_ylabel("Mean Surface Temperature Change")

        for i, ax in enumerate(axs.flatten()):
            ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')

        for ax in axs[:,0]:
            ax.set_title("DJF")
        for ax in axs[:,1]:
            ax.set_title("JJA")

        plt.show()
        #plt.savefig(self.save_path + "glosat_temperature_change.png", dpi=600)
        

if __name__ == "__main__":
    gea = glosat_ensemble_analysis()
    #gea.get_mean_surface_glosat_temperature()
    gea.plot_mean_surface_glosat_temperature_change()
    #gea.create_sea_ice_area_sum()

    #gea.render_ensemble_sea_ice()
    #plt.show()
#plt.savefig("sea_ice_area.png")
