from arraylake import Client
import json

with open("credentials.json", "r") as cred:
    credentials = json.load(cred)

client = Client()

client.create_bucket_config(
  org="ARIA-VERIFY",
  nickname="benchmarking-bucket",
  uri="s3://verify-benchmarking",
  extra_config={
      'endpoint_url': 'https://verify-oce-o.s3-ext.jc.rl.ac.uk',
      'force_path_style': True,
  },
  auth_config={
      'access_key_id': credentials['access_key'],
      'secret_access_key': credentials['secret_access_key'],
      'method': 'hmac',
  }
)
