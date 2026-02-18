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
import xeofs as xe
from scipy import stats

import os
import dask
from dask.distributed import Client, LocalCluster

class glosat_ensemble_analysis(object):
    def __init__(self):
        self.glosat_path = "/gws/nopw/j04/glosat/production/UKESM/raw/"
        self.verify_root = "/gws/ssde/j25a/verify_oce/NEMO/"
        self.case = "GloSat"

        
        self.ensemble_list=["u-ck651",
                            "u-co986",
                            "u-cp625",
                            "u-cu100",
                            "u-cu101",
                            "u-cu102"]
        self.t0_range=["1850","1870"]
        self.t1_range=["1940","1960"]
        self.save_path = self.verify_root + "PostProcessing/" + self.case + \
                         "/" + self.ensemble_list[0] + "/"
        self.plot_path = self.verify_root + "PostProcessing/Plots/" 

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
                                 decode_times=False).load()
                    self.restrict_to_NA(ds, domain="seaice", drop=True)
                    print (ds)
                    times.append(ds.aice.sum(["nj","ni"]))
                except Exception as e:
                    print (e)
                    
            ds = xr.concat(times, "time")
            ds = ds.expand_dims(ensemble=[ensemble_member])

            ds = ds.drop_duplicates("time")

            ensemble_datasets.append(ds)
        ds_all = xr.concat(ensemble_datasets, "ensemble")
        
        with ProgressBar():
            ds_all.to_netcdf(self.save_path + 
                    "glosat_sea_ice_area_ensemble_NA.nc")

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

    def get_mfda(self, paths, var):
        def preprocess(da):
            da = da.drop_vars("time_counter")
            da = da.swap_dims({"time_counter":"time_centered"})
            return da
        da = xr.open_mfdataset(paths, preprocess=preprocess,
                           chunks={"time_counter":1})[var]
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

        def render(ax, da, tmin, tmax, cmap, proj, label="",
                   style="pcolormesh"):
            if style == "pcolormesh":
                pn = ax.pcolormesh(da.nav_lon, da.nav_lat, da.T, transform=proj,
                               vmin=tmin, vmax=tmax, cmap=cmap)
            if style == "contourf":
                levels = np.linspace(tmin,tmax, 11)
                pn = ax.contourf(da.nav_lon, da.nav_lat, da.T, transform=proj,
                               vmin=tmin, vmax=tmax, cmap=cmap, levels=levels)
            cb = plt.colorbar(pn, ax=ax, extend="both")
            cb.ax.set_ylabel(label)

        # set colourbar lims and map
        cmap=plt.cm.viridis
        cmap=plt.cm.RdBu_r
        cmap=cmocean.cm.balance
        #cmap=plt.cm.binary

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
        tmin, tmax = -10, 10
        cmap=cmocean.cm.balance
        
        # render differnces
        render(axs[2,0], diff_DJF, tmin, tmax, cmap, proj,
               f"Mean {label} Change")
        tmin, tmax = -10, 10
        render(axs[2,1], diff_JJA, tmin, tmax, cmap, proj,
               f"Mean {label} Change")

        # add land 
        for i, ax in enumerate(axs.flatten()):
            ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k',
                           facecolor="grey")

        # set titles
        for ax in axs[:,0]:
            ax.set_title("DJF")
        for ax in axs[:,1]:
            ax.set_title("JJA")

        plt.savefig(self.plot_path + 
                  f"glosat_{self.ensemble_list[0]}_{var}_change.png", dpi=600)

    def plot_mean_glosat_change_difference_only(self, 
            var, label, unit, slim, wlim):
        """ render differernce between time periods """

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
        fig, axs = plt.subplots(1,2, figsize=(6.5,4.5), subplot_kw=proj_dict)
        plt.subplots_adjust(top=0.90, right=0.98, left=0.02, bottom=0.05,
                            wspace=0.05)

        def render(ax, da, tmin, tmax, cmap, proj, label="",
                   style="pcolormesh"):
            if style == "pcolormesh":
                pn = ax.pcolormesh(da.nav_lon, da.nav_lat, da, transform=proj,
                               vmin=tmin, vmax=tmax, cmap=cmap)
            if style == "contourf":
                levels = np.linspace(tmin,tmax, 11)
                pn = ax.contourf(da.nav_lon, da.nav_lat, da, transform=proj,
                               vmin=tmin, vmax=tmax, cmap=cmap, levels=levels)
            cb = plt.colorbar(pn, ax=ax, extend="both", location="bottom",
                             pad=0.05)
            cb.ax.set_xlabel(label)

        # set colourbar lims and map
        wmin, wmax = -wlim, wlim
        smin, smax = -slim, slim
        cmap=cmocean.cm.balance
        
        # render differnces
        render(axs[0], diff_DJF, wmin, wmax, cmap, proj,
               f"Mean {label} Change ({unit})")
        render(axs[1], diff_JJA, smin, smax, cmap, proj,
               f"Mean {label} Change ({unit})")

        # add land 
        for i, ax in enumerate(axs.flatten()):
            ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k',
                           facecolor="grey")

        # set titles
        axs[0].set_title("DJF")
        axs[1].set_title("JJA")

        plt.savefig(self.plot_path + 
           f"glosat_{self.ensemble_list[0]}_{var}_change_difference_only.png", 
                  dpi=600)



    def get_altantic_mask_glosat(self):
        """ mask by atlantic """

        # Define directory path to ancillary files:
        domain_filepath = \
            "https://noc-msm-o.s3-ext.jc.rl.ac.uk/npd-eorca1-jra55v1/domain"
    
        # Open eORCA1 ocean basin masks:
        self.atlmsk = xr.open_zarr(f"{domain_filepath}/subbasins",
                        consolidated=True, chunks={}).atlmsk

    def get_barotropic_stream_function(self, y0, y1, averaging="clim"):
        """ 
        calculate the barotropic streamfunction 
        monthly climatalogical mean over window
        or annual mean
        """

        def get_annual_mean_depth_mean():

            y_set = []
            for y in range(y0, y1):
                print (y)
                paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                     "/" + str(y) + f"*/*1m_{y}*grid-V.nc")
                paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                     "/" + str(y+1) + f"*/*1m_{y}*grid-V.nc")
                year_paths = paths0 + paths1

                vvel_series = self.get_mfda(year_paths, "vo")
                e3v_series = self.get_mfda(year_paths, "thkcello")

                depth_weight_vvel_series = vvel_series * e3v_series

                baroV = depth_weight_vvel_series.sum(dim="depthv") 
                baroV_mean = baroV.mean("time_centered")
                baroV_mean = baroV_mean.expand_dims(year=[y])
                y_set.append(baroV_mean)

            with ProgressBar():
                baroV_mean_set = xr.concat(y_set, "year").load()

            return baroV_mean_set

        def get_clim_mean_depth_mean():

            path_set = []
            for y in range(y0, y1):
                paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                     "/" + str(y) + f"*/*1m_{y}*grid-V.nc")
                paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                     "/" + str(y+1) + f"*/*1m_{y}*grid-V.nc")
                year_paths = paths0 + paths1
                path_set += year_paths

            vvel_series = self.get_da(path_set[0], "vo")
            for path in path_set[1:]:
                print (path)
                vvel = self.get_da(path, "vo")
                vvel_series = xr.concat([vvel_series, vvel],
                        dim="time_centered")

            e3v_series = self.get_da(path_set[0], "thkcello")
            for path in path_set[1:]:
                e3v = self.get_da(path, "thkcello")
                e3v_series = xr.concat([e3v_series, e3v], dim="time_centered")

            # set averaging type
            ave_typ = "time_centered.month"

            with ProgressBar():
                vvel_mean = vvel_series.groupby(ave_typ).mean().load()
                e3v_mean = e3v_series.groupby(ave_typ).mean().load()

            depth_weight_vvel = vvel_mean * e3v_mean
            baroV = depth_weight_vvel.sum(dim="depthv") 

            return depth_int_vvel

        if averaging == "clim":
            baroV = get_clim_mean_depth_mean()
        if averaging == "annual":
            baroV = get_annual_mean_depth_mean() 

        # get depth integrated velocities
        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()
        #e3v = domcfg.e3v
        #e3v = e3v.assign_coords({"nav_lon":domcfg.glamu,
        #                         "nav_lat":domcfg.gphiu})
        e1v = domcfg.e1v
        e1v = e1v.assign_coords({"nav_lon":domcfg.glamu,
                                 "nav_lat":domcfg.gphiu})
        #e1u = e1u.where(domcfg.top_level == 1)

        baroV = baroV.stack(a=["x","y"])
        e1v = e1v.stack(a=["x","y"])

        baroU, e1v = xr.align(baroV, e1v)

        e1v = e1v.unstack("a")
        baroV = baroV.unstack("a")

        transport = ( baroV * e1v ).sortby("x",
                                     ascending=False).cumsum(dim="x")/1e6

        BSF_masked = transport.sortby("x").where(domcfg.top_level == 1)

        BSF_masked = BSF_masked.assign_coords({"nav_lon":domcfg.nav_lon,
                                               "nav_lat":domcfg.nav_lat})

        BSF_masked.name = "BSF"
        BSF_masked.to_netcdf(self.save_path + 
                             f"glosat_{averaging}_mean_BSF_{y0}_{y1}.nc")

    def plot_BSF_and_AMOC_single_ensemble(self):
        """ plot timeseries of BSF and AMOC over spg """

        # get data
        BSF = xr.open_dataarray(self.save_path +
                             "glosat_annual_mean_BSF_1850_2015.nc")
        AMOC_ensemble = xr.open_dataarray(self.save_path +
                             "glosat_AMOC_1850_2014.nc")
        AMOC = AMOC_ensemble.isel(ensemble=0)

        # get nao
        NAO = xr.open_dataarray(
                    f"{self.save_path}/NAO_eof_weighted_abs_scores.nc"
                    ).isel(mode=0)

        TAU = xr.open_dataarray(self.save_path +
                                 "glosat_tau_curl_1850_2015.nc")
        TOS = xr.open_dataarray(self.save_path +
                                 "glosat_tos_1850_2015_date_err.nc")

        # set time units
        TOS["time_counter"] = [np.datetime64(str(int(y)), "Y") for y in TAU.year]
        TOS = TOS.rename({"time_counter":"year"})
        TAU["year"] = [np.datetime64(str(int(y)), "Y") for y in TAU.year]
        NAO["year"] = [np.datetime64(str(int(y)), "Y") for y in NAO.year]
        BSF["year"] = [np.datetime64(str(int(y)), "Y") for y in BSF.year]
        AMOC["time_centered"] = [np.datetime64(str(m.values), "M") 
                                 for m in AMOC.time_centered]
        
        # find SPG and NAG max
        BSF_na = self.restrict_to_NA(BSF, domain="ocean")
        TAU_na = self.restrict_to_NA(TAU, domain="ocean")
        TOS_na = self.restrict_to_NA(TOS, domain="ocean")
        TAU_na = TAU_na.where(BSF_na.isel(x=slice(None,-1),
                                          y=slice(None,None)) > 10)
        TOS_na = TOS_na.where(BSF_na > 10)
        
        ## get SPG by mode (fixed position)
        #SPG_ind = BSF_na.argmax(["x","y"])
        #SPG_lon_mode = stats.mode(SPG_ind["x"]).mode
        #SPG_lat_mode = stats.mode(BSF_na.isel(x=SPG_lon_mode).argmax("y")).mode
        #SPG_alt = np.abs(BSF_na.isel(x=SPG_lon_mode,y=SPG_lat_mode))

        ## get NAG by mode (fixed position)
        #NAG_ind = BSF_na.argmin(["x","y"])
        #NAG_lon_mode = stats.mode(NAG_ind["x"]).mode
        #NAG_lat_mode = stats.mode(BSF_na.isel(x=NAG_lon_mode).argmin("y")).mode
        #NAG_alt = np.abs(BSF_na.isel(x=NAG_lon_mode,y=NAG_lat_mode))

        # get SPG and NAG max (moving position)
        SPG = BSF_na.max(["x","y"])
        NAG = np.abs(BSF_na.min(["x","y"]))

        #TAU = np.abs(TAU_na.isel(x=SPG_lon_mode,y=SPG_lat_mode))
        #TOS = np.abs(TOS_na.isel(x=SPG_lon_mode,y=SPG_lat_mode))
        TOS = TOS_na.mean(["x","y"])
        TAU = TAU_na.mean(["x","y"])
        
        # rolling mean AMOC
        AMOC = AMOC.rolling(time_centered=12, center=True).mean()

        # rolling mean NAO
        NAO = NAO / np.abs(NAO.max())
        NAO_rolling = NAO.rolling(year=5, center=True).mean()

        # add linear fit
        def fit_dim(da, dim, deg=1):
            p = da.polyfit(dim=dim, deg=deg)
            fit = xr.polyval(da[dim], p.polyfit_coefficients)
            return da-fit, fit

        AMOC_detrended, AMOC_fit = fit_dim(AMOC, "time_centered")
        SPG_detrended, SPG_fit = fit_dim(SPG, "year")
        NAG_detrended, NAG_fit = fit_dim(NAG, "year")
        TOS_detrended, TOS_fit = fit_dim(TOS, "year")
        TAU_detrended, TAU_fit = fit_dim(TAU, "year")

        # rolling window correlation
        def rolling_window_correlation(lag, ww):
            SPG_roll = SPG_detrended.rolling(year=ww, center=True).construct("window")
            NAO_roll = NAO_rolling.rolling(year=ww, center=True).construct("window")
            
            if lag != 0:
                x1 = NAO_roll.isel(window=slice(0, -lag))
                x2 = SPG_roll.isel(window=slice(lag, None))
            else:
                x1 = NAO_roll
                x2 = SPG_roll
            
            #x1 = x1.isel(year=slice(ww, None))
            #x2 = x2.isel(year=slice(ww, None)) 
            
            
            rac = xr.corr(x2, x1, dim="window")
            return rac

        fig, axs = plt.subplots(6, figsize=(6.5,8))
        plt.subplots_adjust(left=0.15, top=0.98, right=0.82, bottom=0.08)
        for lag in range(1,4):
            rac = rolling_window_correlation(lag, 20)
            axs[4].plot(rac.year.dt.year, rac, label=str(lag))
        axs[4].set_ylim(-1,1)
        axs[4].axhline()
        axs[4].set_ylabel("NAO-SPG\nlag corr")

        # plot positions
        #SPG_ind = BSF_na.argmin(["x","y"])
        #plt.scatter(BSF_na.x[SPG_ind["x"]], BSF_na.y[SPG_ind["y"]].data)
        print (AMOC.time_centered)
        print (AMOC.time_centered.dt.year)


        axs[0].plot(TOS.year.dt.year, TOS)
        axs[0].plot(TOS_fit.year.dt.year, TOS_fit, c="k")
        axs[0].set_ylabel("SPG surface\ntemperature\n" + r"(${^\circ}$C)")
        axs[1].plot(AMOC.time_centered, AMOC)
        axs[1].plot(AMOC_fit.time_centered, AMOC_fit, c="k")
        axs[1].set_ylabel("AMOC\nTransport\n(Sv)")
        axs[2].plot(NAG.year.dt.year, NAG, label="NAG")
        axs[2].plot(SPG.year.dt.year, SPG, label="SPG")
        axs[2].plot(NAG_fit.year.dt.year, NAG_fit, c="k")
        axs[2].plot(SPG_fit.year.dt.year, SPG_fit, c="k")
        axs[2].set_ylabel("Gyre\nTransport\n(Sv)")
        #axs[2].bar(NAO.year, NAO.where(NAO <0))
        #axs[2].bar(NAO.year, NAO.where(NAO >0))
        print (NAO.year.dt.year)
        axs[3].bar(NAO.year.dt.year, NAO, tick_label=NAO.year)
        axs[3].plot(NAO_rolling.year.dt.year, NAO_rolling , c='k')
        axs[3].set_ylabel("NAO")
        axs[3].set_xticks(np.arange(1850,2014,20))
        axs[5].plot(TAU.year.dt.year, TAU )
        axs[5].plot(TAU_fit.year.dt.year, TAU_fit, c='k')
        axs[5].set_ylabel("SPG Wind\nStress Curl\n"  + r"(N m$^{-3}$)")
        for ax in axs[2:]:
            ax.set_xlim(1850,2014)
        axs[0].set_xlim(1850,2014)
        axs[1].set_xlim(AMOC.time_centered.min(),AMOC.time_centered.max())
        for ax in axs[:5]:
            ax.set_xticklabels([])

        axs[-1].set_xlabel("Year")

        axs[2].legend(bbox_to_anchor=[1.01,1.05])
        axs[4].legend(bbox_to_anchor=[1.01,1.05], title="lag (years)")
        plt.savefig(self.save_path + "timeseries_master.png", dpi=1200)

    def curl(self, domcfg, u, v):

        dx = domcfg.e1t
        dy = domcfg.e2t
        area = (domcfg.e1f * domcfg.e2f).isel(x=slice(None,-1),
                                              y=slice(None,-1))
        u = u.isel(x=slice(1,-1), y=slice(None,-1))                
        v = v.isel(x=slice(1,-1), y=slice(None,-1))                

        print (u.shape)
        print (v.shape)
        udx = (u * dx).diff("y", label="lower")
        vdy = (v * dy).diff("x", label="lower")
        print (udx.shape)
        print (vdy.shape)
        
        udx = udx.isel(x=slice(None,-1))
        vdy = vdy.isel(y=slice(None,-1))

        curl = (-udx + vdy) / area

        return curl

    def get_wind_stress_curl(self, y0, y1):
        """ get_wind_stress_curl """

     
        domcfg = xr.open_dataset(self.dom_path, chunks="auto").squeeze()

        domcfg = domcfg.assign_coords({"nav_lon":domcfg.glamu,
                                       "nav_lat":domcfg.gphiu})

        y_set = []
        for y in range(y0, y1):
            print (y)
            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y) + f"*/*1m_{y}*grid-V.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y+1) + f"*/*1m_{y}*grid-V.nc")
            year_paths = paths0 + paths1
            tauv_series = self.get_mfda(year_paths, "tauvo")

            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y) + f"*/*1m_{y}*grid-U.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y+1) + f"*/*1m_{y}*grid-U.nc")
            year_paths = paths0 + paths1
            tauu_series = self.get_mfda(year_paths, "tauuo")

            with ProgressBar():
                tauu_mean = tauu_series.mean("time_centered").load()
                tauv_mean = tauv_series.mean("time_centered").load()

            #tauu_mean = self.restrict_to_NA(tauu_mean, ocean=True, drop=False)
            #tauv_mean = self.restrict_to_NA(tauv_mean, ocean=True, drop=False)

            # get curl
            tau_curl = self.curl(domcfg, tauu_mean, tauv_mean)

            tau_curl = tau_curl.expand_dims(year=[y])

            y_set.append(tau_curl)

        tau_series = xr.concat(y_set, "year")

        tau_series.name = "tau_curl"

        tau_series.to_netcdf(self.save_path + f"glosat_tau_curl_{y0}_{y1}.nc")

    def get_surface_annual_mean_temperature(self, y0, y1):

        y_set = []
        for y in range(y0, y1):
            print (y)
            paths0 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y) + f"*/*1m_{y}*grid-T.nc")
            paths1 = glob.glob(self.glosat_path + self.ensemble_list[0] +
                                 "/" + str(y+1) + f"*/*1m_{y}*grid-T.nc")
            year_paths = paths0 + paths1
            temp_series = self.get_mfda(year_paths, "tos")

            with ProgressBar():
                temp_mean = temp_series.mean("time_centered").load()

            temp_mean = temp_mean.expand_dims(year=[y])

            y_set.append(temp_mean)

        temp_series = xr.concat(y_set, "year")

        temp_series.name = "tos"

        temp_series.to_netcdf(self.save_path + f"glosat_tos_{y0}_{y1}.nc")

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
    
    def restrict_to_NA(self, da, domain="ocean", drop=True):
        """ restrict lat and lon on NAO definition """

        if domain == "ocean":
            lat_lims = [26,70]
            lon_lims = [-90,40]

            da = da.where((da.nav_lon > lon_lims[0]) &
                          (da.nav_lon < lon_lims[1]) &
                          (da.nav_lat > lat_lims[0]) &
                          (da.nav_lat < lat_lims[1]), drop=drop)

        elif domain == "seaice":
            lat_lims = [26,70]
            lon_lims = [-90,40]

            da = da.where((da.TLON > lon_lims[0]) &
                          (da.TLON < lon_lims[1]) &
                          (da.TLAT > lat_lims[0]) &
                          (da.TLAT < lat_lims[1]), drop=drop)

        else:
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
        y1 = 2015

        year_range = np.arange(y0,y1)
     
        month_list = ["dec","jan","feb"]
     
        da_djf_acum = []
        for year in year_range:
            da_acum = [] 
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

                # remove problematic coordinates
                drop_coord_list = ["height", "forecast_reference_time"]
                for coord in drop_coord_list:
                    if coord in da.coords:
                        da = da.drop(coord)

                da = self.restrict_to_NA(da)

                da_acum.append(da)

            da_djf = xr.concat(da_acum, "time").mean("time")
            da_djf = da_djf.expand_dims(year=[year])
            da_djf_acum.append(da_djf) 

        da_djf_series = xr.concat(da_djf_acum, "year")

        da_djf_series.name = "air_pressure_at_sea_level"

        print (da_djf_series)
        da_djf_series.to_netcdf(self.save_path + f"glosat_NAO_slp_{y0}_{y1}.nc")

    def get_eof(self, ds, fn, time_coord="time"):
        """ calculate eof of surface data """
    
        # initiate eof model
        # Note: use_coslat should be used to weight but latitude, but xeof
        # cannot handle 2d latitude variable - it searches for coordinate 
        # dimensions

        model = xe.single.EOF(n_modes=5, use_coslat=True)
    
        # calculate eof
        model.fit(ds, dim=time_coord)
    
        # save components to netcdf
        components = model.components(normalized=False)
        del components.attrs["solver_kwargs"]  # attr causes error
        components.to_netcdf(f"{self.save_path}/{fn}_eof_weighted_abs_components.nc")
    
        # save scores to netcdf
        scores = model.scores(normalized=False)
        del scores.attrs["solver_kwargs"]  # attr causes error
        scores.to_netcdf(f"{self.save_path}/{fn}_eof_weighted_abs_scores.nc")
    
        # save explained variance to netcdf
        var_exp = model.explained_variance_ratio()
        del var_exp.attrs["solver_kwargs"]  # attr causes error
        var_exp.to_netcdf(f"{self.save_path}/{fn}_eof_weighted_abs_var_explained_ratio.nc")
    def get_NAO(self, y0=1850, y1=2015):
        """ calculate eof based NAO """

        djf_slp = xr.open_dataarray(
                self.save_path + f"glosat_NAO_slp_{y0}_{y1}.nc")
        djf_slp = djf_slp.isel(year=slice(3,None))
        self.get_eof(djf_slp, "NAO", time_coord="year")

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
    #gea.get_NAO(y0=1850, y1=2015)
    #gea.get_NAO_slp()
    #gea.get_mean_glosat_variable(y0=1850, y1=1870, var="obvfsq", grid_str="W")
    #gea.get_mean_glosat_variable(y0=1850, y1=1870, var="obvfsq", grid_str="W",
    #                             mld_mid=True)
    gea.plot_mean_glosat_change_difference_only(var="somxl010", 
                                 label="Mixed Layer Depth",
                                unit="m", slim=15, wlim=200)
    gea.plot_mean_glosat_change_difference_only(var="tos",
                             label="Surface Temperature",
                                unit=r"$^{\circ}$C", slim=2.5, wlim=2.5)
    #gea.plot_meridional_overturning_timeseries(y0=1850,y1=2014)
    #gea.get_barotropic_stream_function(y0=1850, y1=2015, averaging="annual")
    #gea.get_surface_annual_mean_temperature(y0=1850, y1=2015)
    #gea.plot_BSF_and_AMOC_single_ensemble()
    
    #gea.get_barotropic_stream_function(y0=1940, y1=1960)
    #gea.create_sea_ice_area_sum()

    #gea.render_ensemble_sea_ice()
    #plt.show()
#plt.savefig("sea_ice_area.png")
