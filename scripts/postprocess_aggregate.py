import rich_click.typer as typer

from loguru import logger
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

import sys
from typing import Iterable, Any, Optional

from pathlib import Path
import xarray as xr

from functools import partial
import numpy
import logging
from dask.distributed import Client

from joblib import Parallel, delayed, cpu_count


import warnings

warnings.filterwarnings("ignore")

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from rich.progress import Progress


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


def aggregate_annual(ncfile: Path, *, outdir: Path, vars: Iterable[str]) -> None:
    comp = dict(zlib=True, complevel=5)
    encoding = {}

    with xr.open_dataset(ncfile)[vars] as ds:
        for var in ds.data_vars:
            encoding[var] = comp

        mean_vars = ["surfacewater", "surfacetemperature"]
        sum_vars = [x for x in ds.data_vars if x not in mean_vars]

        sum_ds = (
            ds[sum_vars].groupby("time.year").sum(dim="time")
        )  # .to_netcdf(outdir / ncfile.name, encoding=encoding)
        mean_ds = ds[mean_vars].groupby("time.year").mean(dim="time")
        dsout = xr.merge([sum_ds, mean_ds])
        dsout.to_netcdf(
            outdir / ncfile.name.replace(".nc", "_yearly.nc"), encoding=encoding
        )

    return None


def process_seasonal(
    workload: Iterable[Any], *, outdir: Path, vars: Iterable[str], cores: int
):
    func = partial(aggregate_seasonal, outdir=outdir, vars=vars)
    ProgressParallel("AGG seasonal", n_jobs=cores)(
        delayed(func)(ncfile) for ncfile in workload
    )


def process_annual(
    workload: Iterable[Any], *, outdir: Path, vars: Iterable[str], cores: int
):
    func = partial(aggregate_annual, outdir=outdir, vars=vars)
    ProgressParallel("AGG annual  ", n_jobs=cores)(
        delayed(func)(ncfile) for ncfile in workload
    )


def calc_seasonal_pctl(
    workload: Iterable[Any], *, outdir: Path, percentiles: Iterable[float]
):

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
                        .quantile(percentiles, dim="file", skipna=True)
                        .compute()
                    )

                progress.update(task, advance=1)

        dsout.to_netcdf(outdir / "seasonal_pctl.nc", encoding=encoding)


def calc_annual_pctl(
    workload: Iterable[Any], *, outdir: Path, percentiles: Iterable[float]
):

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
                    .quantile(percentiles, dim="file", skipna=True)
                    .compute()
                )

                progress.update(task, advance=1)

        dsout.to_netcdf(outdir / "annual_pctl.nc", encoding=encoding)


def main(
    indir: Path = typer.Argument(None, exists=True, file_okay=False),
    outdir: Path = typer.Argument(None, exists=True, file_okay=False),
    debug: bool = typer.Option(False, help="Print debugging output"),
    cores: int = typer.Option(cpu_count()),
):

    if not debug:
        logger.remove()
        logger.add(sys.stdout, level="INFO")

    workload = list(indir.glob("*.nc"))

    logger.debug(f"Using {cores} cores for {len(workload)} files")

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

    process_annual(workload, outdir=outdir, vars=vars_annual, cores=cores)

    process_seasonal(workload, outdir=outdir, vars=vars_seasonal, cores=cores)

    with Client(silence_logs=logging.ERROR) as client:
        logger.debug(
            f"Using dask for upcoming aggregations. See {client.dashboard_link}"
        )

        pctl = [0.05, 0.25, 0.50, 0.75, 0.95]

        workload = list(outdir.glob("*seasonal.nc"))
        calc_seasonal_pctl(workload, outdir=outdir, percentiles=pctl)

        workload = list(outdir.glob("*yearly.nc"))
        calc_annual_pctl(workload, outdir=outdir, percentiles=pctl)


if __name__ == "__main__":
    typer.run(main)
