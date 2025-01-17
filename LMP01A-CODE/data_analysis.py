#!/bin/env python3

import click
import datetime as dt
from src.DataAnalysis import DataAnalysis, CacheData
from src.UnitConversions import UnitConversions


@click.command()
@click.option(
    "-s",
    "--start-timestamp-us",
    default=0,
    type=int,
    help="Start timestamp in us for filtering data. Default is 0.",
)
@click.option(
    "-e",
    "--end-timestamp-us",
    default=2**64,
    type=int,
    help="End timestamp in us for filtering data. Default is 2^64.",
)
@click.argument("csv-file", type=click.Path(exists=True))
@click.option("-p", "--plot", is_flag=True, help="Plot current vs timestamp")
@click.option(
    "-d", "--dont-calculate", is_flag=True, help="Don't calculate average current"
)
@click.option(
    "-c",
    "--no-cache",
    is_flag=True,
    help="Don't use cached values from the CSV file nor write cache data to the CSV file.",
)
@click.help_option("-h", "--help")
def main(
    start_timestamp_us: int,
    end_timestamp_us: int,
    csv_file: str,
    plot: bool,
    dont_calculate: bool,
    no_cache: bool,
):
    """Calculate average current consumption from a CSV file between start_timestamp_us and end_timestamp_us.
    If -p/--plot flag is used data will be plotted.

    This script will also cache the calculated data to the CSV file in form of a comments (starting with '#')
    if the whole file is used (option -s and -e are not used).

    Example usage:

    python data_analysis.py example.csv -s 3_000_000 -e 3_700_000 -p
    """

    da = DataAnalysis(
        csv_file, start_timestamp_us, end_timestamp_us, try_cache=(not no_cache)
    )
    uc = UnitConversions()

    cache_data = None

    # We can only read/write cache data if the whole file is used (option -s and -e are not used)
    if start_timestamp_us != 0 or end_timestamp_us != 2**64:
        no_cache = True

    if not dont_calculate:
        if not no_cache:
            cache_data = da.get_csv_cache_data()

        if cache_data == None:
            time_window_s = da.get_time_slice()
            num_values = da.get_number_of_used_values()
            average_current_Ah = da.calculate_average_current()
        else:
            print("============= Using cached data =============")
            time_window_s = cache_data.time_window_s
            num_values = cache_data.num_values
            average_current_Ah = cache_data.avg_current_Ah

        print(
            f"Selected time window: {time_window_s} s ({uc.s_to_ms(time_window_s)} ms)\n"
            f"Number of values: {num_values}\n"
            f"Average current consumption:\n"
            f"  - {average_current_Ah} Ah\n"
            f"  - {uc.A_to_mA(average_current_Ah)} mAh\n"
            f"  - {uc.A_to_uA(average_current_Ah)} uAh"
        )

    if not no_cache and cache_data == None and not dont_calculate:
        print("============= Writing cache data =============")
        cd = CacheData()
        cd.date = dt.datetime.now().strftime("%d-%m-%Y")
        cd.time = dt.datetime.now().strftime("%H:%M:%S")
        cd.time_window_s = time_window_s
        cd.time_window_ms = uc.s_to_ms(time_window_s)
        cd.num_values = num_values
        cd.avg_current_Ah = average_current_Ah
        cd.avg_current_mAh = uc.A_to_mA(average_current_Ah)
        da.write_csv_cache_data(cd)

    if plot:
        # Plot current vs timestamp
        da.plot_current_vs_timestamp(
            "Timestamp (ms)",
            "Current (uA)",
            "Current vs Timestamp",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Exiting...")
        exit(0)
