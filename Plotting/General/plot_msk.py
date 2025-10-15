import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dask.diagnostics import ProgressBar

path = '/gws/nopw/j04/verify_oce/NEMO/Preprocessing/DOM/NAARC/'

cfg_path = path + 'domain_cfg_zps.nc'
cfg = xr.open_dataset(cfg_path, chunks=-1).squeeze()
top_lev = cfg.top_level.load()

# get mask
msk_path = path + 'bdy_msk_pybdy.nc'
msk = xr.open_dataarray(msk_path, chunks="auto")
import matplotlib.pyplot as plt

msk = xr.where((msk == 0) & (top_lev == 1),  -1, top_lev)
#msk_cut = msk[2660:2790,3765:3933]
#msk[2660:2790,3765:3933] = xr.where(msk_cut == -1, 0, msk_cut)

plt.figure(figsize=(3, 3))
ax = plt.axes(projection=ccrs.Orthographic())
plt.pcolor(msk.load(), transform=ccrs.PlateCarree())
ax.coastlines(resolution='110m')
ax.gridlines()


plt.show()
