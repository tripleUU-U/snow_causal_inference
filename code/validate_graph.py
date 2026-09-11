import pickle 
import logging
import time
import sys

import numpy as np
import pandas as pd

from pathlib import Path

import tigramite.data_processing as td
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr_wls import ParCorrWLS

logging.basicConfig(
    level=logging.INFO,              
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logging.getLogger("graph_validation_CI").setLevel(logging.CRITICAL)

def load_ts_df(
    file_path: Path, 
) -> pd.DataFrame:
    """Load q/climate ts df from LamaH-CE, and change index to datetime."""

    df = pd.read_csv(filepath_or_buffer=file_path, header=0, sep=";")

    df_date_index = (df
    .assign(date=pd.to_datetime(df.rename(columns={"DD": "day", "MM": "month", "YYYY": "year"})[["year", "month", "day"]]))
    .drop(columns=["DD", "MM", "YYYY"])
    .set_index("date")
    )
    return df_date_index

def calculate_snowmelt(
        climate_ts_df: pd.DataFrame
) -> pd.DataFrame: 
    """Calculate snowmelt as depletion of SWE from t-1 to t in basin."""

    snow_ts = climate_ts_df.swe

    # First day cannot have depletion.
    snow_depletion = [np.nan]
    prev = snow_ts.iloc[0]

    for day in range(1, len(snow_ts)):

        current = snow_ts.iloc[day]

        # Calculate delta swe from t-1 to t, if positive denote, else 0.
        depletion = prev - current

        if depletion > 0: 

            snow_depletion.append(depletion)

        else: 

            snow_depletion.append(0)

        prev = current

    # Sanity check. 
    logger.info(
        "Calculated snowmelt:"
        f"\n min  : {np.nanmin(snow_depletion)}"
        f"\n max  : {np.nanmax(snow_depletion)}"
        f"\n mean : {np.nanmean(snow_depletion)}"
    )

    # Add to df.
    climate_ts_df["snowmelt"] = snow_depletion

    return climate_ts_df


def construct_vars_tg(
        lamah_ce_path: Path, 
        basin_id: int
) -> pd.DataFrame: 

    # Load df's. 
    climate_ts_path = lamah_ce_path / f"B_basins_intermediate_all/2_timeseries/daily/ID_{basin_id}.csv"
    q_ts_path = lamah_ce_path / f"D_gauges/2_timeseries/daily/ID_{basin_id}.csv"

    climate_ts_df = load_ts_df(climate_ts_path) 
    q_ts_df = load_ts_df(q_ts_path) 
    logger.info(f"Loaded data from:\n{climate_ts_path}\n{q_ts_path}")

    climate_ts_df = calculate_snowmelt(climate_ts_df=climate_ts_df)

    # Select only relvant climate vars.
    climate_vars_df = climate_ts_df.loc[:, ["2m_temp_mean", "prec", "total_et", "volsw_123", "snowmelt"]]

    # Add discharge
    vars_df = pd.concat([climate_vars_df, q_ts_df["qobs"]], axis=1)

    # Rename for easier access.
    vars_df.columns = ["TEMP", "P", "ET", "SM", "SNO", "Q"]

    # Remove missing.
    vars_df = vars_df.dropna()

    if vars_df.isna().any(axis=None): 
        logger.error("vars_df still conains NA.")
        raise

    # Convert to tigramite form. 
    vars_tg = td.DataFrame(data=vars_df.to_numpy(), var_names=vars_df.columns)

    logger.info(f"Variable names in vars_tg: {vars_tg.var_names}")

    return vars_tg


def run_CI_tests(
        CI_config_dict: dict, 
        vars_tg: td.DataFrame,
        knn: int,
        out_path: Path
) -> None: 

    CI_test_dict = {
        "ParCorr" : ParCorr(),
        "CMIknn" : CMIknn(knn=knn)
    }

    var_names = vars_tg.var_names 

    CI_results_dict = CI_config_dict.copy()

    # For each Y variable run test with chosen X given Z (parents).
    for Y_tuple in CI_results_dict:

        # Tuples of Y as keys, and list holding CI_str, and later CI_results as values.
        X_dict = CI_results_dict[Y_tuple]["X_dict"]

        # Parents (list of tuples).
        Y_parents = CI_results_dict[Y_tuple]["Y_parents"]

        logger.info(f"\n --- Running CI tests for {var_names[Y_tuple[0]]} t_{Y_tuple[1]}: ---")

        for X_tuple in X_dict: 

            # Get the chosen CI test str.
            CI_str = X_dict[X_tuple][0]
            logger.info(f"Using {CI_str} for {var_names[X_tuple[0]]} t_{X_tuple[1]}")

            # Construct Z set from parents of X and Y. 
            X_parents = X_dict[X_tuple][1]

            Z = list(set(X_parents + Y_parents))
            logger.info(f"\tZ: {Z}, consisting of\n\tX_pa: {X_parents}\n\tY_pa: {Y_parents}")

            # Run test.
            CI_test = CI_test_dict[CI_str]
            CI_test.set_dataframe(dataframe=vars_tg)
            results = CI_test.run_test(X=[X_tuple], Y=[Y_tuple], Z=Z, alpha_or_thres=0.01)
            logger.info(f"\t {results}")

            # Save results to list of variable. 
            X_dict[X_tuple].append(results)

        # Replace pre-run X_dict with post-run X_dict. 
        CI_results_dict[Y_tuple]["X_dict"] = X_dict

    try: 

        with open(out_path, "wb") as out: 
            pickle.dump(CI_results_dict, file=out)

    except Exception as e: 
        logger.error(f"Exception occured:\n{e}")


def main(argv: list) -> None: 

    CI_config_dict_path = Path(argv[0])
    knn = int(argv[1])

    logger.info(f"Running CI test pipeline with {CI_config_dict_path}.")

    # Load CI config.
    with open(CI_config_dict_path, "rb") as f: 
        CI_config_dict = pickle.load(f)

    parent_dir_path = Path(CI_config_dict_path.parent)

    out_path = parent_dir_path / "CI_results_dict_knn_{knn}.pkl"

    # Load data.
    lamah_ce_path = Path("/home/wuhlmann/BA/data/raw_data/2_LamaH-CE_daily")

    basin_id = int(parent_dir_path.name)


    vars_tg = construct_vars_tg(
        lamah_ce_path=lamah_ce_path,
        basin_id=basin_id
    )


    run_CI_tests(
        CI_config_dict=CI_config_dict,
        vars_tg=vars_tg,
        knn=knn,
        out_path=out_path
    )


if __name__ == "__main__":

    start_time = time.time()

    main(sys.argv[1:])

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    logger.info(f"CI tests completed in {hours:02d}:{minutes:02d}:{seconds:05.2f}")

    