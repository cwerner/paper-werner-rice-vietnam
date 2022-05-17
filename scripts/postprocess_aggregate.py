import logging
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy
import pandas as pd
import rich_click.typer as typer
import xarray as xr
from dask.distributed import Client
from joblib import cpu_count, delayed, Parallel
from loguru import logger
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

warnings.filterwarnings("ignore")

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


vars_annual = [
    "surfacetemperature",
    "surfacewater",
    "irrigation",
    "precipitation",
    "DW_above",
    "C_stubble",
    "DW_fru_export",
    "C_plant_litter",
    "dC_ch4_emis",
    "dC_co2_emis_hetero",
    "dN_n2o_emis",
    "dN_n2_emis",
    "dN_nh3_emis",
    "dN_fertilizer",
    "dC_fertilizer",
]

vars_seasonal = [
    "surfacetemperature",
    "surfacewater",
    "irrigation",
    "precipitation",
    "DW_above",
    "dC_ch4_emis",
    "dC_co2_emis_hetero",
    "dN_n2o_emis",
]

merge_table = pd.read_csv(
    StringIO(
        """regionid,region,name,irri,mixed
        2,1,northern_mountains,0.6,0.4
        7,1,northern_mountains,0.6,0.4
        4,2,red_river_delta,0.0,1.0
        5,3,northern_central,0.4,0.6
        6,3,northern_central,0.4,0.6
        8,4,central_highlands,0.6,0.4
        3,5,east_southern_region,0.0,1.0
        1,6,mekong_delta,1.0,0.0
        """
    )
)


def convert_strpath(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


class ProgressParallel(Parallel):
    def __init__(self, label: str, total: Optional[int] = None, *args, **kwargs):

        self._pbar = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),  # [:-1],
            "Elapsed:",
            TimeElapsedColumn(),
        )
        self._task = self._pbar.add_task(description=label, total=total)
        self._total = total
        self._pbar.__enter__()
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.update(
            self._task, completed=self.n_completed_tasks, total=self.n_dispatched_tasks
        )

        if self.n_completed_tasks == self.n_dispatched_tasks:
            self._pbar.stop()


def aggregate_seasonal(ncfile: Path, *, outdir: Path, vars: Iterable[str]) -> None:
    comp = dict(zlib=True, complevel=5)
    encoding = {}

    with xr.open_dataset(ncfile)[vars] as ds:
        for var in ds.data_vars:
            encoding[var] = comp
        ds.groupby("time.dayofyear").mean(dim="time").to_netcdf(
            outdir / ncfile.name.replace(".nc", "_seasonal.nc"), encoding=encoding
        )

    return None


def merge_by_merge_table(
    ncfile1: Path,
    ncfile2: Path,
    *,
    outdir: Path,
    merge_table: pd.DataFrame,
    ref: xr.DataArray,
) -> None:

    comp = dict(zlib=True, complevel=5)
    encoding = {}

    with xr.open_dataset(ncfile1) as ds1, xr.open_dataset(ncfile2) as ds2:
        frac1 = ref.copy().where(ds1.surfacetemperature.isnull is False)
        frac2 = ref.copy().where(ds2.surfacetemperature.isnull is False)

        for _, row in merge_table.iterrows():
            frac1 = frac1.where(frac1 == int(row["regionid"]), int(row["irri"]))
            frac2 = frac2.where(frac2 == int(row["regionid"]), int(row["mixed"]))

        dsout = ds1 * frac1 + ds2 * frac2
        for var in dsout.data_vars:
            encoding[var] = comp

        dsout.to_netcdf(
            outdir
            / ncfile1.name.replace("_irrigated-", "_merged-").replace(
                "_upland-", "_merged-"
            ),
            encoding=encoding,
        )


def process_merge_mana_scens(
    workload1: Iterable[Any],
    workload2: Iterable[Any],
    *,
    outdir: Union[str, Path],
    ref: xr.DataArray,
    cores: int,
):
    outdir = convert_strpath(outdir)

    ftype = "seasonal" if "seasonal" in str(workload1[0]) else "yearly"

    logger.debug(workload1)
    logger.debug(workload2)

    func = partial(
        merge_by_merge_table, outdir=outdir, merge_table=merge_table, ref=ref
    )
    ProgressParallel(f"MERGE {ftype}", n_jobs=cores)(
        delayed(func)(ncfile1, ncfile2)
        for ncfile1, ncfile2 in zip(workload1, workload2)
    )


def process_seasonal(
    workload: Iterable[Any],
    *,
    outdir: Union[str, Path],
    vars: Iterable[str],
    cores: int,
):
    outdir = convert_strpath(outdir)
    func = partial(aggregate_seasonal, outdir=outdir, vars=vars)
    ProgressParallel("AGG seasonal", n_jobs=cores)(
        delayed(func)(ncfile) for ncfile in workload
    )


def aggregate_annual(ncfile: Path, *, outdir: Path, vars: Iterable[str]) -> None:
    comp = dict(zlib=True, complevel=5)
    encoding = {}

    with xr.open_dataset(ncfile)[vars] as ds:
        for var in ds.data_vars:
            encoding[var] = comp

        mean_vars = ["surfacewater", "surfacetemperature"]
        sum_vars = [x for x in ds.data_vars if x not in mean_vars]

        sum_ds = ds[sum_vars].groupby("time.year").sum(dim="time")
        mean_ds = ds[mean_vars].groupby("time.year").mean(dim="time")
        dsout = xr.merge([sum_ds, mean_ds])
        dsout.to_netcdf(
            outdir / ncfile.name.replace(".nc", "_yearly.nc"), encoding=encoding
        )

    return None


def process_annual(
    workload: Iterable[Any],
    *,
    outdir: Union[str, Path],
    vars: Iterable[str],
    cores: int,
):
    outdir = convert_strpath(outdir)

    func = partial(aggregate_annual, outdir=outdir, vars=vars)
    ProgressParallel("AGG annual  ", n_jobs=cores)(
        delayed(func)(ncfile) for ncfile in workload
    )


def calc_seasonal_pctl(
    workload: Iterable[Any], *, outdir: Union[str, Path], percentiles: Iterable[float]
):
    outdir = convert_strpath(outdir)
    comp = dict(zlib=True, complevel=5)
    encoding = {}

    with xr.open_mfdataset(
        workload, concat_dim="file", combine="nested", parallel=True
    ) as ds:
        dsout = xr.Dataset()

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),  # [:-1],
            "Elapsed:",
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("AGG pctl seasonal", total=len(ds.data_vars))

            for var in sorted(ds.data_vars):
                encoding[var] = comp

                with numpy.warnings.catch_warnings():
                    numpy.warnings.filterwarnings(
                        "ignore"
                    )  # , r'All-NaN (slice|axis) encountered')
                    dsout[var] = (
                        ds[var]
                        .chunk(dict(file=-1, dayofyear=-1, lat=10, lon=10))
                        .quantile(percentiles, dim="file", skipna=False)
                        .compute()
                    )

                progress.update(task, advance=1)

        dsout.to_netcdf(outdir / "seasonal_pctl.nc", encoding=encoding)


def calc_annual_pctl(
    workload: Iterable[Any], *, outdir: Union[str, Path], percentiles: Iterable[float]
):
    outdir = convert_strpath(outdir)
    comp = dict(zlib=True, complevel=5)
    encoding = {}

    with xr.open_mfdataset(
        workload, concat_dim="file", combine="nested", parallel=True
    ) as ds:
        dsout = xr.Dataset()

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),  # [:-1],
            "Elapsed:",
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("AGG pctl annual  ", total=len(ds.data_vars))

            for var in sorted(ds.data_vars):
                encoding[var] = comp

                # with numpy.warnings.catch_warnings():
                #     numpy.warnings.filterwarnings('ignore') #, r'All-NaN (slice|axis) encountered')
                dsout[var] = (
                    ds[var]
                    .chunk(dict(file=-1, year=-1, lat=10, lon=10))
                    .quantile(percentiles, dim="file", skipna=False)
                    .compute()
                )

                progress.update(task, advance=1)

        dsout.to_netcdf(outdir / "annual_pctl.nc", encoding=encoding)


def main(
    indir: Path = typer.Argument(None, exists=True, file_okay=False),
    outdir: Path = typer.Argument(None, exists=True, file_okay=False),
    reffile: Path = typer.Argument(None, exists=True, dir_okay=False),
    debug: bool = typer.Option(False, help="Print debugging output"),
    cores: int = typer.Option(cpu_count()),
):

    if not debug:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    if (Path.cwd() / "params.yaml").is_file():
        logger.debug("Loading parameter file")
        config = OmegaConf.load(Path.cwd() / "params.yaml")
    else:
        logger.debug("No parameter file found - using defaults")
        config = OmegaConf.create(
            {
                "vars_annual": vars_annual,
                "vars_seasonal": vars_seasonal,
            }
        )

    logger.debug(f"Using {cores} cores")
    logger.info("Stage 1: creating annual and seasonal source nc files")

    # use tmp dir for those files...
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        ptmp = Path(tmp)

        logger.debug(f"Using tmp folder {tmp}")
        logger.info("Stage 1a: Fully irrigated management files")

        workload = sorted(list(indir.glob("VN_arable_irrigated-*.nc")))
        process_annual(workload, outdir=tmp, vars=list(config.vars_annual), cores=cores)
        process_seasonal(
            workload, outdir=tmp, vars=list(config.vars_seasonal), cores=cores
        )

        logger.info("Stage 1b: Upland management files")

        # workload = list(indir.glob("*VN_arable_upland_*.nc"))
        # process_annual(workload, outdir=tmp, vars=config.vars_annual, cores=cores)
        # process_seasonal(workload, outdir=tmp, vars=config.vars_seasonal, cores=cores)

        # fake files
        import shutil

        [
            shutil.copyfile(f, ptmp / str(f.name).replace("irrigated-", "upland-"))
            for f in ptmp.glob("*irrigated*.nc")
        ]

        logger.info("Stage 2: merge scenarios based on merge table")

        ref_da: xr.DataArray = xr.open_dataset(reffile)["regionid"].load()

        workload1 = sorted(list(ptmp.glob("*irrigated*_yearly.nc")))
        workload2 = sorted(list(ptmp.glob("*upland*_yearly.nc")))
        process_merge_mana_scens(
            workload1, workload2, outdir=tmp, ref=ref_da, cores=cores
        )

        workload1 = sorted(list(ptmp.glob("*irrigated*_seasonal.nc")))
        workload2 = sorted(list(ptmp.glob("*upland*_seasonal.nc")))
        process_merge_mana_scens(
            workload1, workload2, outdir=tmp, ref=ref_da, cores=cores
        )

        logger.info("Stage 3: creating annual and seasonal percentile aggregations")
        pctl = [0.05, 0.25, 0.50, 0.75, 0.95]

        with Client(silence_logs=logging.ERROR) as client:
            logger.debug(f"Using dask during aggregations. See {client.dashboard_link}")

            workload = sorted(list(ptmp.glob("*merged*seasonal.nc")))
            calc_seasonal_pctl(workload, outdir=outdir, percentiles=pctl)

            workload = sorted(list(ptmp.glob("*merged*yearly.nc")))
            calc_annual_pctl(workload, outdir=outdir, percentiles=pctl)


if __name__ == "__main__":
    typer.run(main)

    # ref_da = xr.open_dataset("data/raw/misc/VN_MISC5_V2.nc")['regionid'].load()
    # tmp = "tmp"
    # ptmp = Path(tmp)
    # cores=8
    # workload1 = sorted(list(ptmp.glob("*irrigated*_yearly.nc")))
    # workload2 = sorted(list(ptmp.glob("*upland*_yearly.nc")))
    # process_merge_mana_scens(workload1, workload2, outdir=tmp, ref=ref_da, cores=cores)
