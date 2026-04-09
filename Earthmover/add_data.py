from arraylake import Client
import xarray as xr

def add_data_to_cloud(ds, fn, repo, commit_str, branch="main"):
    """
    add xarray dataset to cloud storage

    ds: xarray dataset to be stored
    fn: directory and file name on cloud
    repo: repository name on cloud
    commit_str: comment for addition
    branch: branch of data repository
    """

    # Instantiate the Arraylake client
    client = Client()
    
    # Checkout the repo
    repo = client.get_repo(repo)
    session = repo.writable_session("main")
    
    ds.to_zarr(session.store, group=fn, zarr_format=3)
    
    # Make your first commit
    session.commit(commit_str)

if __name__ == "__main__":

    # get glosat data
    path = "/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/u-ck651/"
    ds = xr.open_dataset(path + "glosat_AMOC_1850_2014.nc")

    fn = "Glosat_u-ck651/AMOC_1850_2014"
    repo = "ARIA-VERIFY/verify-benchmarking-repo"
    commit_str "addtion of glosat AMOC time series"

    add_data_to_cloud(ds, fn, repo, commit_str)
