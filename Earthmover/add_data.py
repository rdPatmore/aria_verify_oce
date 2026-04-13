from arraylake import Client
import xarray as xr
import zarr


def add_data_to_cloud(ds, fn, repo_path, commit_str, branch="main"):
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
    repo = client.get_repo(repo_path)
    session = repo.writable_session("main")
    
    try:
        ds.to_zarr(session.store, group=fn, zarr_format=3, mode="w-")

        # Make your first commit
        session.commit(commit_str)
    except:
        print (f"failed {fn}")
    

def rename_cloud_data(src_path, dst_path, repo_path, commit_str, branch="main"):

    # Instantiate the Arraylake client
    client = Client()

    # Get repo
    repo = client.get_repo(repo_path)

    # Rename
    session = repo.rearrange_session(branch)
    session.move(src_path, dst_path)

    # Commit
    session.commit(commit_str)

def delete_cloud_dir(path, repo_path, commit_str, branch="main"):

    # Instantiate the Arraylake client
    client = Client()

    # Get repo
    repo = client.get_repo(repo_path)
    session = repo.writable_session(branch)
    group = zarr.open_group(session.store)
    del group[path]

    # Commit
    session.commit(commit_str)
