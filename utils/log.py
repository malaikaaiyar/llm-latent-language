import pickle
import os
import sys
import json
import gzip
import logging
import yaml
import json
from logging import Logger
import re
from datetime import datetime
import pprint

def write_log(layer_log2, cfg, info = {}):
    base_log_file_path = cfg.log_file.rsplit('.', 1)[0]  # Strip off the extension if provided

    # Ensure directory exists
    os.makedirs(os.path.dirname(base_log_file_path), exist_ok=True)

    with open(base_log_file_path + ".pkl", "wb") as pickle_file:
        pickle.dump(layer_log2, pickle_file)

    # pickle.dump(layer_log2, open(cfg.log_file + ".pkl", "wb"))

    log_legend = """
    Measuring 
    lp_out/p_out : logprobs/probs of correct answer
    lp_alt/p_alt logprobs/probs of alternate answer
    lp_diff/p_ratio: logprob_diff/probs ration of alt-correct or alt/correct
    """

    pp = pprint.PrettyPrinter(sort_dicts=False)
    # Save log_legend to the log file
    with open(base_log_file_path + ".log", "a") as f:
        f.write("Command: " + ' '.join(sys.argv) + "\n")
        f.write(pp.pformat(asdict(cfg)))
        f.write("\n==============\n")
        for key, val in info.items():
            f.write(f"{key}: {val}\n")
    print("Done!")



def yaml_to_dict(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)



def save_pickle(file, path):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_pickle(path):
    if path.endswith('gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'rb') as f:
        return pickle.load(f)


def printr(text):
    print(f'[running]: {text}')
    

def save_json(data: object, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def prepare_output_dir(base_dir: str = "./runs/") -> str:
    # create output directory based on current time (using zurich time zone)
    experiment_dir = os.path.join(
        base_dir, datetime.now(tz=pytz.timezone("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def get_logger(output_dir) -> Logger:
    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    # Log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Log to file
    file_path = os.path.join(LOG_DIR, f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.log')
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
