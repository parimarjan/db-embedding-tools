import park
import pdb
import os
import traceback
import numpy as np
# from park.param import args
from park.param import parser
from collections import defaultdict
import pandas as pd
from utils.utils import *

def cleanup(env):
    env.clean()

def set_cardinality_error(env, card_type, err_range):
    env._send("setCardinalityError")
    env._send(card_type)
    env._send(str(err_range))

def run_all_eps(env, fixed_agent=None):
    '''
    @ret: dict: query : info,
        where info is returned at the end of the episode by park. info should
        contain all the neccessary facts about that run.
    '''
    queries = {}
    while True:
        # don't know precise episode lengths, changes based on query, so use
        # the done signal to stop the episode
        done = False
        state = env.reset()
        query = env.get_current_query()
        if query in queries.keys():
            break
        # episode loop
        num_ep = 0
        while not done:
            if fixed_agent is None:
                action = env.action_space.sample()
            else:
                action = tuple(fixed_agent[query][num_ep])
            new_state, reward, done, info = env.step(action)
            state = new_state
            num_ep += 1
        queries[query] = info
    return queries

def test(env, baseline="LEFT_DEEP", card_type="noError", reps=1,
        err_range=10):
    '''
    Runs the baseline algorithm, with the given cardinality estimation errors,
    and then uses the decisions made there as a fixed agent in the case with
    true cardinalities.
    @ret: pandas df summarizing the results.
    '''
    # key-val pairs: rep:[num], query:[str], baseline:[cost], card_type:[cost],
    # err_range:[num]
    all_data = defaultdict(list)
    set_cardinality_error(env, card_type, err_range)
    # set error type
    agents = []
    for i in range(reps):
        train_q = run_all_eps(env)
        fixed_agent = {}
        for query in train_q:
            info = train_q[query]
            actions = info["joinOrders"][baseline]
            fixed_agent[query] = actions
        agents.append(fixed_agent)

    set_cardinality_error(env, "noError", err_range)

    # TODO: optimize it so that exh search etc. don't have to rerun the
    # algorithm
    for rep, fixed_agent in enumerate(agents):
        print("rep! ", rep)
        test_q = run_all_eps(env, fixed_agent=fixed_agent)
        total_error = 0.00
        baseline_costs = []
        mod_card_costs = []
        for q in test_q:
            info = test_q[q]
            # TODO: not sure how we would compare these, since any difference might
            # make all future decisions to be different too?
            # heur_order = info["joinOrder"][baseline]

            bcost = info["costs"][baseline]
            card_cost = info["costs"]["RL"]
            cur_error = card_cost - bcost
            # if baseline == "EXHAUSTIVE":
                # if cur_error < 0:
                    # pdb.set_trace()
            total_error += card_cost - bcost
            baseline_costs.append(float(bcost))
            mod_card_costs.append(float(card_cost))

            cur_order = info["joinOrders"]["RL"]
            baseline_order = info["joinOrders"][baseline]

            # store stuff in the dict
            # key-val pairs: rep:[num], query:[str], baseline:[cost], card_type:[cost],
            # err_range:[num]
            all_data["rep"].append(rep)
            all_data["query"].append(q)
            all_data[baseline].append(bcost)
            all_data[card_type].append(float(card_cost) / bcost)
            all_data["err_range"].append(err_range)
            all_data["baseline_order"].append(baseline_order)
            all_data["cur_order"].append(cur_order)

        total_avg_err = np.mean(np.array(mod_card_costs)-np.array(baseline_costs))
        # print("total avg error: {}, {} : {}".format(baseline, card_type, total_avg_err))
        rel_error = np.mean(mod_card_costs) / np.mean(baseline_costs)
        print("total relative error {}, {}: {}".format(baseline, card_type, rel_error))
    return pd.DataFrame(all_data)


def main():
    env = park.make('query_optimizer')
    try:
        baselines = []
        if args.qopt_exh:
            baselines.append("EXHAUSTIVE")
        if args.qopt_left_deep:
            baselines.append("LEFT_DEEP")
        assert len(baselines) > 0
        card_types = args.card_types.split(",")
        err_ranges = args.err_ranges.split(",")
        all_df = load_object(args.output)
        if all_df is None:
            all_df = pd.DataFrame()
        for baseline in baselines:
            for err in err_ranges:
                if err == "":
                    continue
                err = int(err)
                for c in card_types:
                    df = test(env, baseline=baseline,card_type=c,err_range=err,
                            reps=args.reps)
                    all_df = all_df.append(df, ignore_index=True)
                    save_object(args.output, all_df)
        # print(all_df)

    except Exception as e:
        print(e)
        traceback.print_exc()
        cleanup(env)
    cleanup(env)

def read_flags():
    # parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='pandas dataframe',
            type=str, required=False, default="./card_err_exp.pd")
    parser.add_argument('--card_types', help='different cardinality error types',
            type=str, required=False, default="gaussianError,uniformError")
    parser.add_argument('--reps', help='repetitions for each fixed random err \
            agent', type=int, required=False, default=1)
    parser.add_argument('--err_ranges', help='ranges for random errors added \
            to cardinalities', type=str, required=False, default="100,")

    return parser.parse_args()

args = read_flags()
main()
