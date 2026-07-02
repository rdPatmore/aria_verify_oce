import xarray as xr
import Earthmover.add_data as add_data
from Plotting.EnsembleCompare.plot_ensemble import glosat_ensemble_analysis
import glob

def add_glosat_timeseries():
    gea = glosat_ensemble_analysis()
    ensemble_list = gea.ensemble_list

    for member in ensemble_list:
        print (member)
        # get glosat data
        path = f"/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/{member}/"
        ds = xr.open_dataset(path + "glosat_AMOC_1850_2015.nc")
        
        fn = f"GloSat/{member}/AMOC_1850_2014"
        repo = "ARIA-VERIFY/verify-benchmarking-repo"
        commit_str =  f"addtion of glosat {member} tos time series"
        
        add_data.add_data_to_cloud(ds, fn, repo, commit_str)

def add_verify_timeseries():

    data_path = "/gws/ssde/j25a/verify_oce/UKESM/wave1.5/earthmover_data/"
    member_list = list(set([i.removeprefix(data_path).split(".")[0]
                 for i in glob.glob(data_path + "*")]))
    
    var_list = ["amoc_max.26N", "sst.spg_region"]
    for var in var_list:
        print (var)
        for member in member_list:
            print (member)
            ds = xr.open_dataset(data_path + f"{member}.{var}.monthly.nc")
            var_alt = var.replace(".","_")

            fn = f"VERIFY/{member}/{var}_monthly"
            repo = "ARIA-VERIFY/verify-benchmarking-repo"
            commit_str =  f"addtion of verify {member} {var_alt} time series"
            
            add_data.add_data_to_cloud(ds, fn, repo, commit_str)

def add_dummy():
    gea = glosat_ensemble_analysis()
    member = gea.ensemble_list[0]

    # get glosat data
    path = f"/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/{member}/"
    da = xr.open_dataarray(path + "glosat_AMOC_1850_2015.nc")

    da_dummy = shift_and_scale(da)
    
    fn = f"GloSat/{member}_dummy/AMOC_1850_2014"
    repo = "ARIA-VERIFY/verify-benchmarking-repo"
    commit_str =  f"addtion of glosat dummy AMOC time series"
    
    add_data.add_data_to_cloud(da_dummy, fn, repo, commit_str)

def shift_and_scale(ds):
    """ shift and scale the maginitude of data """

    ds = ds * 100 + 1000

    return ds

def delete_errornous_dirs():
    gea = glosat_ensemble_analysis()
    ensemble_list = gea.ensemble_list
    
    for member in ensemble_list:
        commit_str =  f"delete {member} - path error"
        path = f"GloSat_{member}"
        repo = "ARIA-VERIFY/verify-benchmarking-repo"
        add_data.delete_cloud_dir(path, repo, commit_str)
add_dummy()
