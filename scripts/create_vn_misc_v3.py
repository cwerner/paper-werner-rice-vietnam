from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError, check_output

import pandas as pd
import xarray as xr


@lru_cache(maxsize=1)
def root():
    """ returns the absolute path of the repository root """
    try:
        base = check_output("git rev-parse --show-toplevel", shell=True)
    except CalledProcessError:
        raise IOError("Current working directory is not a git repository")
    return Path(base.decode("utf-8").strip())


def main(infile: Path, outfile: Path):

    cfile = root() / "data" / "raw" / "misc" / "counter_cc_correction.txt"
    changes = pd.read_csv(cfile, sep="\t")[["cellid", "nd_crop_new"]]

    with xr.open_dataset(infile) as misc:

        misc2 = misc.copy(deep=True)

        df = misc.siteid.to_dataframe().dropna().astype("int").reset_index()
        df = df[df.siteid.isin(set(changes.cellid))]
        df = df.merge(changes, left_on="siteid", right_on="cellid")

        for _, row in df.iterrows():
            misc2.rice_rot.loc[{"lat": row.lat, "lon": row.lon}] = row.nd_crop_new

        misc2.to_netcdf(outfile)


if __name__ == "__main__":
    main(
        root() / "data" / "raw" / "misc" / "VN_MISC5_V2.nc",
        root() / "data" / "raw" / "misc" / "VN_MISC5_V3.nc",
    )
