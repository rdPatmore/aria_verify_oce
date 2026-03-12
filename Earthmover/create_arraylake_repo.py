import icechunk
from arraylake import Client

client = Client()

config = icechunk.RepositoryConfig(
  storage = icechunk.StorageSettings(
      unsafe_use_conditional_update=False,
      unsafe_use_conditional_create=False,
  )
)

repo = client.create_repo(
  "ARIA-VERIFY/verify-benchmarking-repo",
  bucket_config_nickname="benchmarking-bucket",
  config=config,
)
