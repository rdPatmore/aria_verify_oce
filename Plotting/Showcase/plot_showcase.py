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
    
    def plot_naarc_glosat_compare(self):
        """ plot a resolution comparison between NAARC and GloSAT """

        # initialise figure
        proj = ccrs.PlateCarree()
        proj_dict={"projection":ccrs.Orthographic(-30,60)}
        #proj_dict={"projection":ccrs.PlateCarree()}
        fig, axs = plt.subplots(2, figsize=(6.5,6), subplot_kw=proj_dict)
        plt.subplots_adjust(top=0.95, right=0.85, left=0.01, bottom=0.05,
                            wspace=0.3)

        vmin, vmax = 0, 25
        cmap = cmocean.cm.thermal
        ds = xr.open_dataset(self.glosat_path + self.ensemble_list[0] + 
                   "/18510101T0000Z/nemo_ck651o_1m_18501201-18510101_grid-T.nc")
        da = ds.tos.squeeze()

        pn = axs[0].pcolormesh(da.nav_lon, da.nav_lat, da, transform=proj,
                            vmin=vmin, vmax=vmax, cmap=cmap, shading="nearest")
        cb = plt.colorbar(pn, ax=axs[0], extend="both")
        cb.ax.set_ylabel("")

        plt.savefig(self.save_path + "Plots/naarc_glosat_compare.png")

gea = glosat_ensemble_analysis()
gea.plot_naarc_glosat_compare()
