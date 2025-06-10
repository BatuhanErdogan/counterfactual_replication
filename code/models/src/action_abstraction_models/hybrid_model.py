import pandas as pd
import numpy as np
import csv
import os
from ast import literal_eval

# Hybrid model globals
JSON_INPUT_DIR = "gridworld_json"
CSV_OUTPUT_DIR = "model_output"
DISCOUNT_FACTOR = 0.9
MYSTERY_TREE_REWARD_OPTIMIST = 8
MYSTERY_TREE_REWARD_PESSIMIST = 2

EXPERIMENT_TRIALS = [
    "trial_13_v2", "trial_318", "trial_553", "trial_374", "trial_920",
    "trial_453", "trial_894", "trial_863", "trial_989", "trial_269",
    "trial_528", "trial_406", "trial_955", "trial_962", "trial_740",
    "trial_556", "trial_825", "trial_859", "trial_629", "trial_82"
]

def get_trial_info(trial_name, summary):
    row = summary[summary["name"] == trial_name].iloc[0]
    agent_type = row["agent_type"]
    start_pos = literal_eval(row["agent_start_position"])
    tree_visibility = literal_eval(row["tree_visibility"])
    tree_rewards = literal_eval(row["tree_rewards"])
    tree_positions = literal_eval(row["tree_positions"])
    best_path = literal_eval(row["best_path"])
    outcome = row["path_reached_reward_goal"]
    reward = row["path_true_reward"]
    return agent_type, start_pos, tree_visibility, tree_rewards, tree_positions, best_path, outcome, reward

def update_visibility(tree_visibility, tree_positions, path):
    visibility = tree_visibility[:]
    for loc in path:
        if loc in tree_positions:
            idx = tree_positions.index(loc)
            if visibility[idx] == 0:
                visibility[idx] = 1
    return visibility

def discounted_expected_reward(start_pos, visibility, rewards, positions, belief_mean, gamma=DISCOUNT_FACTOR):
    total = 0
    count = 0
    for i in range(len(rewards)):
        count += 1
        dist = abs(start_pos[0] - positions[i][0]) + abs(start_pos[1] - positions[i][1])
        if visibility[i] == 1:
            total += (gamma ** dist) * rewards[i]
        else:
            total += (gamma ** dist) * belief_mean
    return total / count

def get_trial_overview(trial_name, summary):
    agent_type, start_pos, visibility, rewards, positions, best_path, outcome, reward = get_trial_info(trial_name, summary)
    visibility = update_visibility(visibility, positions, best_path)

    belief_actual = MYSTERY_TREE_REWARD_OPTIMIST if agent_type == "optimist" else MYSTERY_TREE_REWARD_PESSIMIST
    belief_cf = MYSTERY_TREE_REWARD_PESSIMIST if agent_type == "optimist" else MYSTERY_TREE_REWARD_OPTIMIST
    start_cf = (1, 10) if start_pos == (10, 1) else (10, 1)

    H_actual = discounted_expected_reward(start_pos, visibility[:], rewards, positions, belief_actual)
    H_cf_trait = discounted_expected_reward(start_pos, visibility[:], rewards, positions, belief_cf)
    H_cf_start = discounted_expected_reward(start_cf, visibility[:], rewards, positions, belief_actual)

    C_trait = abs(H_cf_trait - H_actual)
    C_start = abs(H_cf_start - H_actual)
    w_trait = C_trait / (C_trait + C_start) if (C_trait + C_start) != 0 else 0.5

    return {
        "trial_name": trial_name,
        "agent": agent_type,
        "start_location": start_pos,
        "start_location_row": 1 if start_pos[0] == 1 else 10,
        "outcome": outcome,
        "path_true_reward": reward,
        "discounted_expected_reward": H_actual,
        "C_trait": C_trait,
        "C_start": C_start,
        "w_trait": w_trait,
        "w_start": 1 - w_trait
    }

def write_to_csv(trial_summary, filename):
    header = list(trial_summary[0].keys())
    vals = [[elem[key] for key in header] for elem in trial_summary]
    print(f'Writing to csv: {filename}')
    with open(filename, "w", newline="") as file:
        csv_writer = csv.writer(file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for idx, row in enumerate(vals):
            if idx == 0:
                csv_writer.writerow(header)
            print(f'Writing data row: {row}')
            csv_writer.writerow(row)

def main():
    summary_df = pd.read_csv(os.path.join(JSON_INPUT_DIR, "gridworld_summary.csv"))
    trial_summaries = [get_trial_overview(trial, summary_df) for trial in EXPERIMENT_TRIALS]
    write_to_csv(trial_summaries, os.path.join(CSV_OUTPUT_DIR, "hybrid_model_trial_summary.csv"))

if __name__ == "__main__":
    main()
