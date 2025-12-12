import paramiko
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

variable = "SOIL_M"

hostname = "merida-sftp.rse-web.it"
username = os.getenv("SFTP_USERNAME")
password = os.getenv("SFTP_PASSWORD")
remote_base = "/merida/METEO/Output/ope/WRFAMEZ/ECMWF/grib2/{ym}/{variable}/MERIDA_{variable}_{ym}.nc"
local_base = "Risikofaktoren/SoilMoisture/data/MERIDA_{variable}_{ym}.nc"
min_year = 2015
max_year = 2024


# Create SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    # Connect to SFTP server
    ssh.connect(hostname, username=username, password=password)
    sftp = ssh.open_sftp()

    for year in range(min_year, max_year):
        for month in range(1, 13):
            ym = f"{year}{month:02d}"
            remote_path = remote_base.format(ym=ym, variable=variable)
            local_path = local_base.format(ym=ym, variable=variable)
            try:
                print(f"Downloading {remote_path}...")
                sftp.get(remote_path, local_path)
                print(f"Successfully downloaded to {local_path}")
            except Exception as e:
                print(f"Failed to download {remote_path}: {e}")

    # Download for 202412
    ym = "202412"
    remote_path = remote_base.format(ym=ym, variable=variable)
    local_path = local_base.format(ym=ym, variable=variable)
    try:
        print(f"Downloading {remote_path}...")
        sftp.get(remote_path, local_path)
        print(f"Successfully downloaded to {local_path}")
    except Exception as e:
        print(f"Failed to download {remote_path}: {e}")

    sftp.close()
except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()
