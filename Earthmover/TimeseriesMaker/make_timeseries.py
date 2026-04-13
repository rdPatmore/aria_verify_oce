import xarray as xr
from Plotting.EnsembleCompare.plot_ensemble import glosat_ensemble_analysis

class timeseries(object):
    """ create timeseries from model object """

    def __init__(self, case, domcfg_fn):

        # set paths
        root = "/gws/ssde/j25a/verify_oce/NEMO/"
        model_path = root + "Outputs/" + case
        domcfg_path = root + "Preprocessing/DOM/" + domcfg_fn

        # get domcfg
        #self.domcfg = xr.open_dataset(domcfg_path, chunks="auto")

        a = glosat_ensemble_analysis()
        a.get_hadisst()

    def area_mean(self, da, weights):
        lat_lims = [45,65]
        lon_lims = [-60,10]

        # restrict to area
        da = da.where((da.nav_lon > lon_lims[0]) &
                      (da.nav_lon < lon_lims[1]) &
                      (da.nav_lat > lat_lims[0]) &
                      (da.nav_lat < lat_lims[1]), drop=False)
        da = da.isel(x=slice(1,-1), y=slice(None,-1))

        # area weighted mean
        da = da.weighted(weights).mean(["x","y"])

        return da

    def calc_SPG_temperature_timeseries(self, y0, y1, member):
        """ """

        gea = glosat_ensemble_analysis(ensemble_member=member)

        # get area
        domcfg = xr.open_dataset(gea.dom_path, chunks="auto").squeeze()
        area = (domcfg.e1f * domcfg.e2f)#.isel(x=slice(None,-1),
                                        #      y=slice(None,-1))

        y_set = []
        for y in range(y0, y1):
            print (y)
            year_paths = gea.get_year_paths(y)
            temp_series = gea.get_mfda(year_paths, "tos").load()
            temp_series_mean = self.area_mean(temp_series, area)

            y_set.append(temp_series_mean)

        temp_series_full = xr.concat(y_set, "time_centered")
        temp_series_full = temp_series_full.assign_attrs(
                   {"ensemble_member":gea.ens})

        fn = gea.save_path + f"glosat_SPG_tos_{y0}_{y1}.nc"
        temp_series_full.to_netcdf(fn)

    def calc_SPG_temperature_glosat_ens(self, y0, y1):
        """ get subpolar gyre temperature for all ensemble members """

        for i in range(6):
            self.calc_SPG_temperature_timeseries(y0, y1, i)

    def get_AMOC_glosat_ens(self, y0, y1):
        """ get AMOC strength for all ensemble members """

        for i in range(6):
            print (i)
            gea = glosat_ensemble_analysis(ensemble_member=i)
            gea.get_meridional_overturning_timeseries(y0, y1)

ts = timeseries("","")
ts.get_AMOC_glosat_ens(1850,2015)
#ts.calc_SPG_temperature_glosat_ens(1850,2015)
