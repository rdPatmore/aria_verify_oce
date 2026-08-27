import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import iris.analysis.geometry as iris_geom
import cartopy.geodesic as cgeo

class forcing_perturbation(object):
    """
    workflow for adjusting forcing
    """

    def __init__(self):
        self.root = "/gws/ssde/j25a/verify_oce/NEMO/Preprocessing/"
        self.output = "/gws/ssde/j25a/verify_oce/NEMO/Outputs/"

    def get_greenland_mask(self, ds, t_val, f_val):
        """
        get greenland mask according to lat-lon limits
        """

        msk = xr.where((ds.nav_lon > -66) &
                       (ds.nav_lon < -13) &
                       (ds.nav_lat > 58.92) &
                       (ds.nav_lat < 84.56),
                       t_val,
                       f_val)

        return msk

    def plot_greenland_mask(self):
        """ plot mask of greenland """

        # greenland mask
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1)
        landmask = domcfg.top_level.squeeze()
        landmask_alt = xr.where(landmask, -1, 0)
      
        #grnlnd_mask = self.get_greenland_mask(domcfg, landmask_alt, landmask)

        p = plt.pcolor(grnlnd_mask)
        plt.colorbar(p)
        plt.show()
        plt.savefig(self.root + "Plots/rnf_msk.png")
        
    def factor_greenland_runoff(self, factor):
        """ multiply the magnitude of greenland runoff by a factor """

        # get runoff data
        path = self.root + "RNF/runoff_1m_nomask.nc"
        rnf = xr.open_dataset(path, chunks=-1)

        # get Greenland Mask
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1).squeeze(drop=True)
        grnlnd_mask = self.get_greenland_mask(domcfg, 1, 0)

        # multiply region by factor
        rnf["sorunoff"] = xr.where(grnlnd_mask,
                                   rnf.sorunoff * factor,
                                   rnf.sorunoff).transpose("time_counter",
                                                           "y",
                                                           "x")

        # save
        fn_out = self.root + f"RNF/runoff_1m_nomask_grnlnd_f{str(factor)}.nc"
        rnf.to_netcdf(fn_out)

    def plot_runoff_magnitude(self):
        """ plot Greenland runoff climatology in Sv """

        # units for runoff kg/(m^2.s)
        # units for Sv 1e6 m^3/s

        # get runoff data
        path = self.root + "RNF/runoff_1m_nomask_grnlnd_f10.nc"
        rnf = xr.open_dataset(path, chunks=-1).sorunoff

        # get mask and domain_cfg
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1).squeeze(drop=True)
        grnlnd_mask = self.get_greenland_mask(domcfg, 1, 0)

        rho0=1026

        # mask greenland area 
        grnlnd_rnf = grnlnd_mask * rnf

        # convert units to Sv
        rnf_Sv = grnlnd_rnf * domcfg.e1t * domcfg.e2t / (rho0 * 1e6)

        # sum river points
        rnf_Sv_sum = rnf_Sv.sum(["x","y"])

        p = plt.plot(rnf_Sv_sum)
        plt.show()

    def plot_runoff_spatially(self):
        """ plot new and old runoff for greenland from model output"""

        year = 1858
        data_path = self.output + "Historical/"

        # get mask and domain_cfg
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1).squeeze()
        grnlnd_mask = self.get_greenland_mask(domcfg, 1, np.nan)

        # get old runoff
        case = "EXP_mes_climatology_1850_1870"
        fn = f"{data_path}/{case}/{year}/12/VERIFY_1m_{year}1201_{year}1230_grid_T.nc"
        old = xr.open_dataset(fn, chunks=-1).friver.squeeze()
        #grnlnd_mask = old.where(old>0)
        old = old.where(old>0)
        old_msk = self.get_greenland_mask(old, old, np.nan)
        old_msk = old_msk.dropna(dim="x", how="all").dropna(dim="y",
                                                            how="all")

        # get new runoff
        case = "EXP_mes_climatology_1850_1870_fw10"
        fn = f"{data_path}/{case}/{year}/12/VERIFY_1m_{year}1201_{year}1230_grid_T.nc"
        new = xr.open_dataset(fn, chunks=-1).friver.squeeze()
        print (new)
        print (grnlnd_mask)
        #new_msk = new * grnlnd_mask
        new = new.where(new>0)
        new_msk = self.get_greenland_mask(new, new, np.nan)
        new_msk = new_msk.dropna(dim="x", how="all").dropna(dim="y",
                                                            how="all")
        diff = old_msk - new_msk
        

        fig, axs = plt.subplots(1,3)

        p= axs[0].pcolor(old_msk)
        axs[1].pcolor(new_msk)
        axs[2].pcolor(diff)
        plt.colorbar(p)
        plt.show()

    def get_palaeo_routing(self, scenario):
        """ plot palaeo routing """

        # get runoff
        path = self.root + f"RNF/glac1d_freshwater_routed_{scenario}.nc"
        rnf = xr.load_dataarray(path)
        rnf["lon"] = rnf.lon - 360


        # add bounding box
        x, y = [-85, -120, 20, -45, -85], [35, 90, 90, 35, 35]
        #plt.plot(x, y, marker="o", transform=ccrs.PlateCarree(), color="coral")

        rnf = self.extract_box(x, y, rnf)

        rnf = rnf.stack(z=["lat","lon"], create_index=False)
        rnf = rnf.where(rnf<0, drop=True)

        return rnf

        #ax = plt.axes(projection=ccrs.PlateCarree())
        #ax.coastlines()
        #ax.scatter(rnf.lon, rnf.lat, transform=ccrs.PlateCarree())
        #plt.plot(x, y, marker="o", transform=ccrs.PlateCarree(), color="coral")
        #plt.show()

    def extract_box(self, x, y, da):
        
        # polygon: (longitude, latitude)
        polygon = Polygon(zip(x,y))
        
        da = da.rename({"lon":"x", "lat":"y"})
        da["x"].attrs["standard_name"] = "longitude"
        da["y"].attrs["standard_name"] = "latitude"
        da_cube = da.to_iris()
        for coord in da_cube.coords():
            print(coord.name(), coord.var_name, coord.standard_name, coord.bounds)

        x_diff_lower = (da.x - da.x.shift(x=1))/2
        x_diff_upper = (da.x - da.x.shift(x=-1))/2
        y_diff_lower = (da.y - da.y.shift(y=1))/2
        y_diff_upper = (da.y - da.y.shift(y=-1))/2
        x_diff_lower = x_diff_lower.fillna(0.25)
        x_diff_upper = x_diff_upper.fillna(-0.25)
        y_diff_lower = y_diff_lower.fillna(0.125)
        y_diff_upper = y_diff_upper.fillna(-0.125)
        x_lower = (da.x - x_diff_lower).values
        x_upper = (da.x - x_diff_upper).values
        y_lower = (da.y - y_diff_lower).values
        y_upper = (da.y - y_diff_upper).values
        da_cube.coord("longitude").bounds = list(zip(x_lower, x_upper))
        da_cube.coord("latitude").bounds = list(zip(y_lower, y_upper))
        # Quickest to do it using just one timestamp from PET series
        region_wgts = iris_geom.geometry_area_weights(da_cube, polygon)

        da = da.rename({"x":"lon", "y":"lat"})
        region_wgts = xr.DataArray(region_wgts, dims=["lat","lon"],
                                   coords=dict(lon=da.lat, lat=da.lon))

        # Select data inside polygon
        da_polygon = da.where(region_wgts > 0, drop=True)

        return da_polygon

    def get_NAARC_coastline(self, domcfg):
        """
        Find sea cells that touch land.
        Input value:
        0 = Land
        1 = ocean
        OBC also is 0


        Output values:
          1   = coastal sea point (the one cell that touch land)
          0   = non-coastal sea point
          NaN = land point
        """


        #land_sea = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
        #                         chunks=-1).top_level.squeeze()
        land_sea = domcfg.top_level.load()

        land_sea = land_sea.drop_vars(["x"])
        land_sea = (land_sea - 1) * -1 # set land to 1 and sea to 0
        
        land_neighbour_count = land_sea.shift(x=1)  +  \
                               land_sea.shift(x=-1) +  \
                               land_sea.shift(y=1)  +  \
                               land_sea.shift(y=-1) 

        coast = xr.where((land_neighbour_count > 0) & (land_sea == 0), 1, 0)
        coast.name = "coast"

        coast = coast.stack(z=["x","y"], create_index=False)
        coast = coast.where(coast==1, drop=True)

        return coast

    def nearest_coast_index(river_points, coast_points):
        """
        For each river point, find the nearest coastline point.

        This is intentionally a simple chunked search. It avoids adding extra
        dependencies and matches MATLAB's first-minimum behaviour.
        """
        nearest = np.empty(river_points.shape[0], dtype=int)

        for start in range(0, river_points.shape[0], CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, river_points.shape[0])

            lon_difference = river_points[start:stop, None, 0] \
                           - coast_points[None, :, 0]
            lat_difference = river_points[start:stop, None, 1] \
                           - coast_points[None, :, 1]
            distance_squared = lon_difference**2 + lat_difference**2

            nearest[start:stop] = np.argmin(distance_squared, axis=1)

        return nearest

    def nearest_coast(self, rnf, coast):

        
        coast_rnf = []
        print (rnf)
        for pt in rnf:
            dist = self.haversine_dist(coast.nav_lon, coast.nav_lat,
                                       pt.lon.values, pt.lat.values)
            argmin = dist.argmin()
            pt = pt.assign_coords(lat=dist.nav_lat[argmin].values,
                                  lon=dist.nav_lon[argmin].values)
            coast_rnf.append(pt)
        coast_rnf_all = xr.concat(coast_rnf, dim="rnf_pt")
        coast_rnf_all.name = "runoff"

        print (coast_rnf_all)
        coast_rnf_all = coast_rnf_all.groupby(["lat","lon"]).sum()
        coast_rnf_all = coast_rnf_all.stack(rnf_pt=["lat","lon"], create_index=False)
        coast_rnf_all = coast_rnf_all.dropna("rnf_pt")
        print (coast_rnf_all)
        #dist = cgeo.Geodesic().inverse(pt0, pt1)

        return coast_rnf_all

    def haversine_dist(self, lon0, lat0, lon1, lat1):
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon0 = lon0 * np.pi / 180
        lon1 = lon1 * np.pi / 180
        lat0 = lat0 * np.pi / 180
        lat1 = lat1 * np.pi / 180

        # haversine formula
        dlon = lon1 - lon0
        dlat = lat1 - lat0
        a = np.sin(dlat/2)**2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(a**0.5)
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        dist = c * r
        print (dist)
        return dist

    def map_rnf_to_coastline(self):
        domcfg = xr.open_dataset(self.root + "DOM/NAARC/domain_cfg_mes.nc",
                                 chunks=-1).squeeze()
        domcfg = domcfg.set_coords(["nav_lat","nav_lon"])
        top_lev = domcfg.top_level.load()

        print (domcfg)
        coast = self.get_NAARC_coastline(domcfg)
        rnf = self.get_palaeo_routing("8ka")
        rnf_coast = self.nearest_coast(rnf,coast)
        top_lev = top_lev.where((top_lev.nav_lat < rnf_coast.lat.max()) & \
                           (top_lev.nav_lat > rnf_coast.lat.min()) & \
                           (top_lev.nav_lon < rnf_coast.lon.max()) & \
                           (top_lev.nav_lon > rnf_coast.lon.min()), drop=True)

        ax = plt.axes(projection=ccrs.PlateCarree())
        #ax.coastlines()
        ax.pcolor(top_lev.nav_lon, top_lev.nav_lat, top_lev,
                  transform=ccrs.PlateCarree())
        ax.scatter(rnf.lon, rnf.lat, c="r", s=2, transform=ccrs.PlateCarree())
        ax.scatter(rnf_coast.lon, rnf_coast.lat, c="k",s=2,  transform=ccrs.PlateCarree())
        #plt.plot(x, y, marker="o", transform=ccrs.PlateCarree(), color="coral")
        plt.savefig("rnf_8ka.png", dpi=600)

#    coast_y, coast_x = np.where(coast_mask == 1)
#    coast_points = np.column_stack((lon_region[coast_y, coast_x], lat_region[coast_y, coast_x]))
#    river_points = np.column_stack((source_lon[inside_domain], source_lat[inside_domain]))
#
#    nearest = nearest_coast_index(river_points, coast_points)
#    river_values = river_outflow[inside_domain]
#
#    target_y = coast_y[nearest]
#    target_x = coast_x[nearest]
#
#    # Add together catchments that go to the same coast cell.
#    np.add.at(river_on_region, (target_y, target_x), river_values)
#
#    total_difference = np.nansum(river_values) - np.nansum(river_on_region)
#    if total_difference > 1.0e-6:
#        raise RuntimeError(f"Runoff was lost during remapping: {total_difference:g} m3/s")

forper =  forcing_perturbation()
#forper.factor_greenland_runoff(10)
#forper.plot_greenland_mask()
#forper.plot_palaeo_routing("9ka")
forper.map_rnf_to_coastline()
#forper.get_NAARC_coastline()
#forper.plot_runoff_spatially()
