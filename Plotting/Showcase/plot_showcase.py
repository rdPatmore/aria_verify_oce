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

import os
import dask
from dask.distributed import Client, LocalCluster
import matplotlib 
matplotlib.rcParams.update({'font.size': 8})

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

        gs_dom_path = "Preprocessing/DOM/UKESM/domcfg_UKESM1p1_gdept.nc"
        na_dom_path = "Preprocessing/DOM/NAARC/domain_cfg_mes.nc"
        self.gs_dom_path = self.verify_root + gs_dom_path
        self.na_dom_path = self.verify_root + na_dom_path

    def preprocess(self, ds):
        ds = ds.aice
        #ds = ds.aice.sum(["nj","ni"])
        return ds

    #def get_aice(self, path):
    #    print (path)
    #    ds = xr.open_dataset(path, chunks=-1, decode_times=False)
    #    print (ds)
    #    return ds.aice
    
    def plot_naarc_glosat_compare(self):
        """ plot a resolution comparison between NAARC and GloSAT """

        # initialise figure
        proj = ccrs.PlateCarree()
        proj_dict={"projection":ccrs.Orthographic(-30,60)}
        #proj_dict={"projection":ccrs.PlateCarree()}
        fig, axs = plt.subplots(1,2, figsize=(6.5,3.5), subplot_kw=proj_dict)
        plt.subplots_adjust(top=0.95, right=0.85, left=0.01, bottom=0.05,
                            wspace=0.15)

        vmin, vmax = 0, 25
        cmap = cmocean.cm.thermal
        
        # get glosat data
        ds = xr.open_dataset(self.glosat_path + self.ensemble_list[0] + 
                   "/18510101T0000Z/nemo_ck651o_1m_18501201-18510101_grid-T.nc")
        ds_ice = xr.open_dataset(self.glosat_path + self.ensemble_list[0] + 
                   "/18510101T0000Z/cice_ck651i_1m_18501201-18510101.nc")

        ice_gs = ds_ice.aice.squeeze()
        tos_gs = ds.tos.squeeze()

        # get NAARC
        data_path = "Outputs/EXP_mes_LSM_new_radiation/1850/12/"
        fn = "VERIFY_6h_18501201_18501230_grid_T.nc"
        ds = xr.open_dataset(self.verify_root + data_path + fn)
        fn = "VERIFY_1m_18501201_18501230_icemod.nc"
        ds_ice = xr.open_dataset(self.verify_root + data_path + fn)

        tos_na = ds.sst_con.isel(time_counter=60)
        ice_na = ds_ice.siconc.squeeze()

        # plot glosat
        for i in [0,1]:
            pn = axs[i].pcolormesh(tos_gs.nav_lon, tos_gs.nav_lat, tos_gs,
                                transform=proj, vmin=vmin, vmax=vmax, cmap=cmap,
                                shading="nearest")

        # plot NAARC temp
        pn_temp = axs[1].pcolormesh(tos_na.nav_lon, tos_na.nav_lat, tos_na,
                            transform=proj, vmin=vmin, vmax=vmax, cmap=cmap,
                            shading="nearest")
        
        # plot sea-ice
        vmin, vmax = 0, 1
        ice_gs = ice_gs.where(ice_gs > 0) # mask ice free zone
        pn = axs[0].pcolormesh(ice_gs.TLON, ice_gs.TLAT, ice_gs,
                            transform=proj, vmin=vmin, vmax=vmax,
                            cmap=cmocean.cm.ice, shading="nearest")

        ice_na = ice_na.where(ice_na > 0) # mask ice free zone
        pn_ice = axs[1].pcolormesh(ice_na.nav_lon, ice_na.nav_lat, ice_na,
                            transform=proj, vmin=vmin, vmax=vmax,
                            cmap=cmocean.cm.ice, shading="nearest")

        # get domain cfg
        #cfg_gs = xr.open_dataset(self.gs_dom_path).tmask
        #cfg_na = xr.open_dataset(self.na_dom_path).tmask

        # add land 
        for i, ax in enumerate(axs):
            land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='grey')
            ax.add_feature(land_50m, zorder=100, lw=0.5)

        # colour bar
        pos = axs[1].get_position()
        cbar_ax = fig.add_axes([0.88, pos.y0, 
                                0.02, (pos.y1 - pos.y0)/2])
        cbar = fig.colorbar(pn_temp, cax=cbar_ax, orientation='vertical',
                           extend="both")
        cbar.ax.text(4.5, 0.5, r"Temperature ($^{\circ}$C)", fontsize=8,
                  rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left')

        cbar_ax = fig.add_axes([0.88, pos.y0 + (pos.y1 - pos.y0)/2, 
                                0.02, (pos.y1 - pos.y0)/2])
        cbar = fig.colorbar(pn_ice, cax=cbar_ax, orientation='vertical',
                           extend="both")
        cbar.ax.text(4.5, 0.5, r"Sea Ice Concentration (-)", fontsize=8,
                  rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left')

        # titles
        axs[0].set_title(r"GloSat $1^{\circ}$")
        axs[1].set_title(r"NAARC $1/12^{\circ}$")

        plt.savefig(self.save_path + "Plots/naarc_glosat_compare.png",
                    dpi=1200)

gea = glosat_ensemble_analysis()
gea.plot_naarc_glosat_compare()
