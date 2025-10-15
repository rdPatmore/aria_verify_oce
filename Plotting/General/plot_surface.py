import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import cmocean
import cartopy.feature as cfeature
import numpy as np
import matplotlib
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from dask.diagnostics import ProgressBar, Profiler 
import dask
import cftime

matplotlib.rcParams.update({'font.size': 8})

def plot_temperature():
    path = "/gws/nopw/j04/verify_oce/NEMO/"
    fn = path + "Outputs/ORCA2_1h_18500101_18500107_grid_T.nc"
    cfg = path + "Preprocessing/DOM/omain_cfg_zps_gdept.nc"
    
    # get surface temperature
    temp = xr.open_dataset(fn, chunks=-1).sst_con
    
    # set temperature limits and time choices
    tmin, tmax = -5, 30
    times = []
    
    # plot setup
    proj=ccrs.AlbersEqualArea()
    proj=ccrs.PlateCarree()
    plt_proj=ccrs.PlateCarree()
    proj_dict = {"projection": plt_proj}
    fig, axs = plt.subplots(1,2, figsize=(6.5,4.0), subplot_kw=proj_dict)
    plt.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.22)
    
    times_str = ["1850-01-01 01:00:00",
             "1850-01-02 01:00:00",
             "1850-01-04 01:00:00",
             "1850-01-07 23:00:00"]
    times = [cftime.datetime(1850,1,1,1, calendar="noleap"),
             cftime.datetime(1850,1,1,2, calendar="noleap"),
             cftime.datetime(1850,1,1,4, calendar="noleap"),
             cftime.datetime(1850,1,1,7, calendar="noleap")]
   
    times = [cftime.datetime(1850,1,1,1, calendar="noleap"),
             cftime.datetime(1850,1,1,7, calendar="noleap")]
    # render temperature
    for i, time in enumerate(times):
        with ProgressBar():
            temp_t = temp.sel(time_counter=time, method="nearest").load()
        print(temp_t)
        p = axs[i].pcolor(temp_t.nav_lon, temp_t.nav_lat, temp_t, 
                   vmin=tmin, vmax=tmax,
                   transform=proj, cmap=cmocean.cm.thermal, shading="nearest")

        date_str = "Date: " + temp_t.time_counter.dt.strftime("%Y-%m-%d").values
        print (date_str)
        axs[i].text(0.5,0.97, date_str, transform=fig.transFigure,
                 ha="center",va="top")
        
    
    
        axs[i].add_feature(cfeature.LAND, zorder=100, edgecolor='k')
        #axs[i].set_ylim(-90,90)
        #axs[i].set_xlim(-180,180)
    
    #axs.set_xticks([-25, -20, -15, -10, -5, 0, 5, 10], crs=ccrs.PlateCarree())
    #axs.set_yticks([45, 50, 55, 60], crs=ccrs.PlateCarree())
    #lon_formatter = LongitudeFormatter(zero_direction_label=True)
    #lat_formatter = LatitudeFormatter()

    pos = axs[-1].get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.12, 
                            pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -2.8, r"Temperature ($^{\circ}$C)", fontsize=8,
                 rotation=0, transform=cbar.ax.transAxes,
                 va='top', ha='center')
    
    for ax in axs:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    #plt.show()
    plt.savefig(path + "NAARC_spinup_surface_temp.png", dpi=600)


def render(fig, ax, cfg, da, vmin, vmax, cmap, proj, cbar_txt="",
           l_label_off=False, r_label_off=False):
    """
    Render surface map on subplot panel
    """

    # render
    p = ax.pcolor(da.nav_lon, da.nav_lat, da, vmin=vmin, vmax=vmax,
                  transform=proj, cmap=cmap, shading="nearest")

    # add land mask
    top_level = cfg.top_level
    top_level = top_level.where(top_level == 0)
    ax.pcolor(top_level.nav_lon, top_level.nav_lat, top_level, 
              vmin=-0.20, vmax=0.8,
               transform=proj, cmap=plt.cm.copper, shading="nearest")
    
    # set extent
    lon0 = da.nav_lon.isel(x=0, y=0).values
    lon1 = 9.8
    ax.set_extent([lon0, lon1, 46, 62], ccrs.PlateCarree())

    # format gridlines
    lon_grid = [-20,-10, 0, 10]
    lat_grid = [45, 50, 55, 60, 65]
    gl = ax.gridlines(draw_labels=True, xlocs=lon_grid, ylocs=lat_grid,
                     color='k', alpha=0.5)
    gl.xpadding = 2
    gl.ypadding = 2
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    if r_label_off:
        gl.right_labels = False
    if l_label_off:
        gl.left_labels = False
    gl.bottom_labels = False
    plt.draw()
    
    # add colourbar
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.12, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -3.5, cbar_txt, fontsize=8, rotation=0,
                 transform=cbar.ax.transAxes, va='top', ha='center')

if __name__ == "__main__":
    dask.config.set(scheduler="single-threaded")
    plot_temperature()
