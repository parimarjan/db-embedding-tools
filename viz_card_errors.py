import argparse
import pandas as pd
from utils.utils import *
import pdb
from collections import defaultdict
import seaborn as sns

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='',
            type=str, required=False, default="exh.pd")
    parser.add_argument('--fig_name', help='',
            type=str, required=False, default="test.png")
    return parser.parse_args()

def num_differences(order1, order2):
    num_diff = 0
    for i,o1 in enumerate(order1):
        if set(order2[i]) != set(o1):
            num_diff += 1
    return num_diff

def main():
    df = load_object(args.file)
    assert df is not None
    # average results over each rep of same error + error_range
    # baseline =
    error_ranges = set(df["err_range"])
    reps = set(df["rep"])
    queries = set(df["query"])
    error_types = set()


    for k in df.keys():
        if "Error" in k:
            error_types.add(k)

    all_errors = defaultdict(list)
    for err_rng in error_ranges:
        # new bar plot for every one of these
        for rep in reps:
            # for this rep, index, and error_type
            for error_type in error_types:
                errors=df.loc[(df['rep']==rep)&(df['err_range']==err_rng)][[error_type,"baseline_order","cur_order"]]
                errors=errors.dropna(subset=[error_type])
                all_errors["error"].append(float(errors[error_type].mean()))
                all_errors["error_type"].append(str(error_type))
                all_errors["err_rng"].append(str(err_rng))
                cur_orders = errors["cur_order"].tolist()
                baseline_orders = errors["baseline_order"].tolist()
                diff = 0
                for i, order in enumerate(cur_orders):
                    diff += num_differences(order, baseline_orders[i])
                print("total differences: ", diff)
                if len(cur_orders) > 0:
                    all_errors["order_diff"].append(float(diff) / len(cur_orders))
                else:
                    all_errors["order_diff"].append(float(diff))

    plottable_df = pd.DataFrame(all_errors)
    bad_errors = plottable_df.loc[plottable_df["error"] >= 2.00]
    print(bad_errors)
    errors = (set(plottable_df["err_rng"]))
    errors_int = [int(e) for e in errors]
    errors_int = np.sort(np.array((errors_int)))
    errors_ord = [str(e) for e in errors_int]
    # hues = (set(plottable_df["error_type"]))
    hues_order = ["uniformPercentError", "uniformMixedPercentError",
                    "uniformPercentTablesError"]

    ax = sns.barplot(x="err_rng", y="error", hue="error_type",
            data=plottable_df, estimator=np.median, ci=75, order=errors_ord,
            hue_order=hues_order)
    # ax = sns.barplot(x="err_rng", y="error", data=plottable_df,
            # estimator="median")
    fig = ax.get_figure()
    fig.get_axes()[0].set_yscale('log')
    fig.savefig(args.fig_name)

args = read_flags()
main()
