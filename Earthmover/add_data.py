from arraylake import Client
import xarray as xr

# Instantiate the Arraylake client
client = Client()

# Checkout the repo
repo = client.get_repo("ARIA-VERIFY/verify-benchmarking-repo")
session = repo.writable_session("main")

# get glosat data
path = "/gws/ssde/j25a/verify_oce/NEMO/PostProcessing/GloSat/u-ck651/"
ds = xr.open_dataset(path + "glosat_AMOC_1850_2014.nc")
ds.to_zarr(session.store, group="GloSat_u-ck651", zarr_format=3)

# Make your first commit
session.commit('Initial Commit')
