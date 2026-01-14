import xarray as xr
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
import glob
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.colors as mcolors
import iris

import os
import dask
from dask.distributed import Client, LocalCluster

class glosat_ensemble_analysis(object):
    def __init__(self):
        self.glosat_path = "/gws/nopw/j04/glosat/production/UKESM/raw/"
        self.verify_root = "/gws/ssde/j25a/verify_oce/NEMO/"
        self.save_path = self.verify_root + "PostProcessing/"
        
        self.ensemble_list=["u-ck651",
                            "u-co986",
                            "u-cp625",
                            "u-cu100",
                            "u-cu101",
                            "u-cu102"]
        self.t0_range=["1850","1870"]
        self.t1_range=["1940","1960"]

        dom_path = "Preprocessing/DOM/UKESM/domcfg_UKESM1p1_gdept.nc"
        self.dom_path = self.verify_root + dom_path

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
            paths = glob.glob(self.glosat_path + ensemble_member + 
                              "/*/cice*1m*.nc")
            #futures = client.map(self.get_aice, paths)
            #ds = client.gather(futures)
            #print (ds)

            times = []
            for path in paths:
                print (path)
                try:
                    ds = xr.open_dataset(path, chunks="auto",
                                 decode_times=False)
                    times.append(ds.aice.sum(["nj","ni"]))
                except Exception as e:
                    print (e)
                    
            ds = xr.concat(times, "time")
            ds = ds.expand_dims(ensemble=[ensemble_member])

            ds = ds.drop_duplicates("time")

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

        fig, axs = plt.subplots(2, figsize=(6.5,5))
        plt.subplots_adjust(right=0.78)
        
        for j, (ensemble, ds_member) in enumerate(ds.groupby("ensemble")):
            ice_ts_cycle = []
            ice_ts_min = []
            ice_ts_max = []
            for i, (year, ds_year) in enumerate(ds_member.groupby("time.year")):
                ice_min = ds_year.aice.min()
                ice_max = ds_year.aice.max()

                ice_ts_cycle.append(ice_max-ice_min)
                ice_ts_min.append(ice_min)
                ice_ts_max.append(ice_max)
                #ticks = plt.gca().get_xticks()
                #times = xr.DataArray(ticks, dims="time", attrs=ds.time.attrs,
                #                     name="time")
                #times = xr.decode_cf(times.to_dataset()).time.dt.strftime("%Y-%m-%d")
                #
                #plt.gca().set_xticklabels(times.data)
            #axs.plot(np.arange(1850,1900), ice_ts_cycle)
            c = list(mcolors.TABLEAU_COLORS)[j]
            axs[0].plot(np.arange(1850,2015), ice_ts_min, lw=0.8, c=c,
                     label=self.ensemble_list[j])
            axs[0].plot(np.arange(1850,2015), ice_ts_max, lw=0.8, c=c)
            axs[1].plot(np.arange(1850,2015), ice_ts_cycle, lw=0.8, c=c)
        axs[0].legend(bbox_to_anchor=[1.01,1.05])
        for ax in axs:
            ax.set_xlim(1850,2014)
            ax.set_xlabel("Date")
        axs[0].set_ylabel(r"Sea Ice Area (m$^2$)")
        axs[1].set_ylabel(r"Sea Ice Area Cycle (m$^2$)")
        plt.savefig("sea_ice_area.png", dpi=600)

    def get_sea_ice_area_std(self):
        """ get sea ice standard deviation timeseries"""

    def get_da(self, path, var):
        da = xr.open_dataset(path, chunks={"time_counter":1})[var]
        da = da.drop_vars("time_counter")
        da = da.swap_dims({"time_counter":"time_centered"})
        return da

    def get_mld_mid(self, y0, y1):

        path_set = []
        for y in range(y0, y1):
            print (y)
            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                               "/" + str(y) + f"*/*1m_{y}*grid-T.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                               "/" + str(y+1) + f"*/*1m_{y}*grid-T.nc")
            paths = paths0 + paths1

            path_set += paths

        mld_series = self.get_da(path_set[0], "somxl010")
        for path in path_set[1:]:
            mld = self.get_da(path, "somxl010")
            mld_series = xr.concat([mld_series, mld], dim="time_centered")

        mld_mid = mld_series/2
        return mld_mid

    def get_mean_glosat_variable(self, y0=1850, y1=1870,
                                 var="tos", grid_str="T", mld_mid=False):

        path_set = []
        for y in range(y0, y1):
            print (y)
            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                               "/" + str(y) + f"*/*1m_{y}*grid-{grid_str}.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                               "/" + str(y+1) + f"*/*1m_{y}*grid-{grid_str}.nc")
            paths = paths0 + paths1

            path_set += paths

        da_series = self.get_da(path_set[0], var)
        for path in path_set[1:]:
            da = self.get_da(path, var)
            da_series = xr.concat([da_series, da], dim="time_centered")

        if mld_mid:
            dep = self.get_mld_mid(y0, y1)
            da_series = da_series.sel(depthw=dep, method="nearest")

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
        #proj_dict={"projection":ccrs.PlateCarree()}
        fig, axs = plt.subplots(3,2, figsize=(6.5,6), subplot_kw=proj_dict)
        plt.subplots_adjust(top=0.95, right=0.85, left=0.01, bottom=0.05,
                            wspace=0.3)

        def render(ax, da, tmin, tmax, cmap, proj, label=""):
            pn = ax.pcolormesh(da.nav_lon, da.nav_lat, da, transform=proj,
                               vmin=tmin, vmax=tmax, cmap=cmap)
            #levels = np.linspace(tmin,tmax, 11)
            #pn = ax.contourf(da.nav_lon, da.nav_lat, da, transform=proj,
            #                   vmin=tmin, vmax=tmax, cmap=cmap, levels=levels)
            cb = plt.colorbar(pn, ax=ax, extend="both")
            cb.ax.set_ylabel(label)

        # set colourbar lims and map
        cmap=plt.cm.viridis
        cmap=plt.cm.RdBu_r
        cmap=plt.cm.binary

        # render means
        render(axs[0,0], da_t0_DJF, vmin, vmax, cmap, proj,
               f"{label} 1850-1870")
        render(axs[1,0], da_t1_DJF, vmin, vmax, cmap, proj,
               f"{label} 1940-1960")
        render(axs[0,1], da_t0_JJA, vmin, vmax, cmap, proj,
               f"{label} 1850-1870")
        render(axs[1,1], da_t1_JJA, vmin, vmax, cmap, proj,
               f"{label} 1940-1960")

        # set colourbar lims and map
        tmin, tmax = -0.0001, 0.0001 
        cmap=cmocean.cm.balance
        
        # render differnces
        render(axs[2,0], diff_DJF, tmin, tmax, cmap, proj,
               f"Mean {label} Change")
        tmin, tmax = -0.0001, 0.0001 
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

    def barotropic_stream_function(self, y0, y1):
        """ calculate the barotropic streamfunction  """

        path_set = []
        for y in range(y0, y1):
            print (y)
            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y) + f"*/*1m_{y}*grid-U.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y+1) + f"*/*1m_{y}*grid-U.nc")
            paths = paths0 + paths1
            path_set += paths

        # get depth integrated velocities
        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()
        e3u = domcfg.e3u
        e3u = e3u.assign_coords({"nav_lon":domcfg.glamu,
                                 "nav_lat":domcfg.gphiu})
        e2u = domcfg.e2u
        e2u = e2u.assign_coords({"nav_lon":domcfg.glamu,
                                 "nav_lat":domcfg.gphiu})
        #e1u = e1u.where(domcfg.top_level == 1)

        uvel_series = self.get_da(path_set[0], "uo")
        for path in path_set[1:]:
            uvel = self.get_da(path, "uo")
            uvel_series = xr.concat([uvel_series, uvel], dim="time_centered")

        e3u_series = self.get_da(path_set[0], "thkcello")
        for path in path_set[1:]:
            e3u = self.get_da(path, "thkcello")
            e3u_series = xr.concat([e3u_series, e3u], dim="time_centered")

        with ProgressBar():
            uvel_mean = uvel_series.groupby("time_centered.month").mean().load()
            e3u_mean = e3u_series.groupby("time_centered.month").mean().load()

        transport = uvel_mean * e3u_mean


        baroU = transport.sum(dim="depthu") 
        baroU = baroU.stack(a=["x","y"])
        e2u = e2u.stack(a=["x","y"])

        baroU, e2u = xr.align(baroU, e2u)

        e2u = e2u.unstack("a")
        baroU = baroU.unstack("a")

        BSF = ( baroU * e2u ).sortby("y", ascending=False).cumsum(dim="y")/1e6
        BSF_masked = BSF.sortby("y").where(domcfg.top_level == 1)

        BSF_masked.name = "BSF"
        BSF.to_netcdf(self.save_path + f"glosat_mean_BSF_{y0}_{y1}.nc")

    def get_AMOC(self, path):
        da = xr.open_dataset(path, chunks={"time_counter":1}, 
                decode_times=False)["zomsfatl"]
        da = da.drop_vars("time_counter")
        da = da.swap_dims({"time_counter":"time_centered"})
        da = da.squeeze()
        return da

    def get_meridional_overturning_timeseries(self, y0, y1):
        """ plot overturning streamfuntion time series """

        ensemble_amoc = []
        for ensemble in self.ensemble_list:
            print (ensemble)
            path_set = []
            for y in range(y0, y1):
                paths0 = glob.glob(self.glosat_path + ensemble +
                                     "/" + str(y) + f"*/*1m_{y}*diaptr.nc")
                paths1 = glob.glob(self.glosat_path + ensemble +
                                     "/" + str(y+1) + f"*/*1m_{y}*diaptr.nc")
                paths = paths0 + paths1

                path_set += paths

            print ("paths set")
            da_series = self.get_AMOC(path_set[0])
            print ("first ts done")
            for path in path_set[1:]:
                print (path)
                da = self.get_AMOC(path)
                da_series = xr.concat([da_series, da], dim="time_centered")
            ind_26n = (abs(da.nav_lat - 26.5)).argmin("y").values
            ds_26n = da_series.isel(y=ind_26n)
            
            amoc = ds_26n.max("depthw")
            amoc.expand_dims(ensemble=[ensemble])
            ensemble_amoc.append(amoc)

        ensemble_amoc_ds = xr.concat(ensemble_amoc, "ensemble")
        ensemble_amoc_ds.to_netcdf(self.save_path + f"glosat_AMOC_{y0}_{y1}.nc")

    def plot_meridional_overturning_timeseries(self, y0, y1):

        fig, axs = plt.subplots(1)

        path_str = self.save_path + f"glosat_AMOC_{y0}_{y1}.nc"
        ensemble_amoc_ds = xr.open_dataarray(path_str)
        print (ensemble_amoc_ds)
        ensemble_amoc_ds["time_centered"] = np.arange(f"{y0}-01", f"{y1}-01",
                                           dtype="datetime64[M]")
        ensemble_amoc_ds = ensemble_amoc_ds.rolling(time_centered=120,
                                                    center=True).mean()

        for i, (ensemble_str, ensemble) in enumerate(
                                        ensemble_amoc_ds.groupby("ensemble")):
            plt.plot(ensemble.time_centered, ensemble.squeeze(),
                    label=self.ensemble_list[i])
        axs.set_xlim(ensemble.time_centered[0],ensemble.time_centered[-1])
        axs.set_xlabel("Date")
        axs.set_ylabel("AMOC (Sv)")
        plt.legend()
        
        plt.savefig("AMOC.png",dpi=600)
    
    def restrict_to_NA(self, da):
        """ restrict lat and lon on NAO definition """

        lat_lims = slice(20,70)
        lon_lims = slice(-90,40)

        # rebase lon
        da["longitude"] = xr.where(da.longitude > 180, da.longitude - 360,
                                   da.longitude)
        da = da.sortby("longitude")

        da = da.sel(longitude=lon_lims, latitude=lat_lims)

        return da

    def get_mean_NA_var(self, y0, y1, var="tos", grid_str="T"):
        """ get mean var in North Atlantic """

        path_set = []
        for y in range(y0, y1+1):
            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                               "/" + str(y) + f"*/*1m_{y}*grid-{grid_str}.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                               "/" + str(y+1) + f"*/*1m_{y}*grid-{grid_str}.nc")
            paths = paths0 + paths1

            path_set += paths

        da_s = self.get_da(path_set[0], var)
        for path in path_set[1:]:
            print (path)
            da = self.get_da(path, var)
            da_s = xr.concat([da_s, da], dim="time_centered")

        da_sr = da_s.where((da_s.nav_lon < 40) & (da_s.nav_lon > -90),
                        (da_s.nav_lat < 70) & (da_s.nav_lat > 20) )
        da_mean = da_sr.mean(["x","y"])
        with ProgressBar():
            da_mean.to_netcdf(self.save_path
                            + f"glosat_{var}_NA_mean_{y0}_{y1}.nc")


    def get_NAO_slp(self):
        """ get NAO index """

        y0 = 1850
        y1 = 2014

        year_range = np.arange(y0,y1)
     
        month_list = ["dec","jan","feb"]
     
        da_acum, da_djf_acum = [], []
        for year in year_range:
            for month in month_list:
     
                print ("year: ", year)
                print ("month: ", month)
                year_str = year

                # skip over missing data
                if (year == 1850) & (month == "dec"): continue
                if (year == 1900) & (month == "dec"): continue
     
                src_path = self.glosat_path + f'u-ck651/{year_str}0101T0000Z/'
                if month == 'dec': year_str = year - 1

                fn = src_path + f"ck651a.p5{year_str}{month}.pp"
                cube = iris.load_cube(fn, "m01s16i222")

                with ProgressBar():
                    da = xr.DataArray.from_iris(cube).load()

                if 'height' in da.coords:
                    da = da.drop('height')

                da = self.restrict_to_NA(da)

                da_acum.append(da)

            da_djf = xr.concat(da_acum, "time").mean("time")
            da_djf = da_djf.expand_dims(year=[year])
            da_djf_acum.append(da_djf) 

        da_djf_series = xr.concat(da_djf_acum, "year")

        da_djf_series.name = "air_pressure_at_sea_level"

        print (da_djf_series)
        da_djf_series.to_netcdf(self.save_path + f"glosat_NAO_slp_{y0}_{y1}.nc")

if __name__ == "__main__":

    # -- Initialise Dask Local Cluster -- #

    # Update temporary directory for Dask workers:
    dask.config.set({'temporary_directory': f"{os.getcwd()}/dask_tmp",
                     'local_directory': f"{os.getcwd()}/dask_tmp"
                     })
    
    # Create Local Cluster:
    #cluster = LocalCluster(n_workers=4, threads_per_worker=3, memory_limit='5GB')
    #client = Client(cluster)

    gea = glosat_ensemble_analysis()
    #gea.get_mean_NA_var(y0=1850, y1=2014, var="tos", grid_str="T")
    gea.get_NAO_slp()
    #gea.get_mean_glosat_variable(y0=1940, y1=1960, var="somxl010")
    #gea.get_mean_glosat_variable(y0=1850, y1=1870, var="obvfsq", grid_str="W")
    #gea.get_mean_glosat_variable(y0=1850, y1=1870, var="obvfsq", grid_str="W",
    #                             mld_mid=True)
    #gea.plot_mean_glosat_change(var="obvfsq", label=r"N$^2$",
    #                            unit=r"s$^{-1}$", vmin=0.0, vmax=0.001)
    #gea.plot_meridional_overturning_timeseries(y0=1850,y1=2014)
    #gea.barotropic_stream_function(y0=1850, y1=1870)
    #gea.barotropic_stream_function(y0=1940, y1=1960)
    #gea.create_sea_ice_area_sum()

    #gea.render_ensemble_sea_ice()
    #plt.show()
#plt.savefig("sea_ice_area.png")
