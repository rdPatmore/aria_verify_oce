import xarray as xr
from Plotting.EnsembleCompare.plot_ensemble import glosat_ensemble_analysis
from Plotting.EnsembleCompare.plot_res_comp import NEMO_case
import matplotlib.pyplot as plt

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
        a.get_hadi(var="sst")


    def calc_SPG_temperature_timeseries_glosat(self, y0, y1, member):
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
            temp_series_mean = gea.area_mean(temp_series, area)

            y_set.append(temp_series_mean)

        temp_series_full = xr.concat(y_set, "time_centered")
        temp_series_full = temp_series_full.assign_attrs(
                   {"ensemble_member":gea.ens})

        fn = gea.save_path + f"glosat_SPG_tos_{y0}_{y1}.nc"
        temp_series_full.to_netcdf(fn)

    def calc_SPG_temperature_timeseries_naarc(self, y0, y1, case_dict):
        """ """

        gea = glosat_ensemble_analysis()

        # get area
        case = NEMO_case(case=case_dict["case"],
                         dom_cfg=case_dict["domcfg"],
                         zcoord=case_dict["zcoord"])
        domcfg = xr.open_dataset(case.dom_path, chunks="auto").squeeze()
        area = (domcfg.e1f * domcfg.e2f)#.isel(x=slice(None,-1),
                                        #      y=slice(None,-1))

        y_set = []
        for y in range(y0, y1):
            print (y)
            year_paths = case.get_paths(y)
            #temp_series = gea.get_mfda(year_paths, "tos_con")
            m_set = []
            for path in year_paths:
                print (path)
                month_da = xr.open_dataset(path, chunks="auto").thetao_con
                month_da_z0 = month_da.isel(deptht=0)
                m_set.append(case.area_mean(month_da_z0, area).load())
            temp_series_year = xr.concat(m_set, "time_counter")

            y_set.append(temp_series_year)

        temp_series_full = xr.concat(y_set, "time_counter")
        temp_series_full = temp_series_full.assign_attrs(
                   {"case":case.case_name})

        fn = case.save_path + f"naarc_SPG_tos_{y0}_{y1}.nc"
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
#ts.get_AMOC_glosat_ens(1850,2015)
case_dict = {"case": "EXP_mes_LSM_unlim_time",
              "domcfg":"domain_cfg_mes.nc",
              "zcoord":"MES"}
#case_dict = {"case": "EXP_zlevel_LSM_new_radiation",
#              "domcfg":"domain_cfg_zps_gdept.nc",
#              "zcoord":"ZPS"}
ts.calc_SPG_temperature_timeseries_naarc(1850, 1851, case_dict)
#ts.calc_SPG_temperature_glosat_ens(1850,2015)
