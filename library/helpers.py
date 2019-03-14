import colorsys # for get_N_HexCol
import os
import json
import csv
import sys
import subprocess
from multiprocessing import Process, Queue
import time
from datetime import datetime as dt
import importlib
import numpy as np

import pickle
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def list_equal(list1, list2):
    if not len(list1) == len(list2):
        print("not same length")
        return False
    for i, l1 in enumerate(list1):
        if not l1 == list2[i]:
            print(l1, list2[i])
            return False
    return True

def now_str():
    return '{:%Y%m%d-%H%M%S}'.format(dt.now() )



def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info("Created directory: %s"%path)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info("Created directory: %s"%path)
    create_dir_if_not_exists(path)


def pd(what, start_time):
    time_in_min = (time.time() - start_time) / 60.0
    time_in_h = time_in_min / 60.0
    print("%s took ~%0.2f min, ~%0.2f h"%(what, time_in_min, time_in_h))

def copy2clip(txt):
    cmd="echo '"+txt.strip()+"'| xsel --clipboard"
    return subprocess.check_call(cmd, shell=True)

# https://stackoverflow.com/questions/4760215/running-shell-command-from-python-and-capturing-the-outputhttps://stackoverflow.com/questions/4760215/running-shell-command-from-python-and-capturing-the-output
def execute_command(command_str):
    command_pieces = command_str.split(" ")
    return subprocess.check_output(command_pieces)

def save_to_pickle(obj, outfile_name):
    pickle.dump(obj, open(outfile_name,"wb"))
    logger.debug("Saved to pickle: %s."%outfile_name)

def load_from_pickle(outfile_name):
    loaded_obj = pickle.load(open(outfile_name, "rb"))
    logger.debug("Loaded %s from pickle"%outfile_name)
    return loaded_obj

# saves a file to the experiment directory
def save_to_json(data, outfile_name):
    with open(outfile_name, "w") as f:
        f.write( json.dumps( data ) )
    logger.debug("Saved to json: %s."%outfile_name)


# dump data array to file
def save_to_csv(data_rows, outfile_name, mode="w", convert_float=False):
    with open(outfile_name, mode) as f:
        cw =  csv.writer(f,delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for data_row in data_rows:
            if convert_float:
                data_row = ["%0.6f"%float(v) for v in data_row]         
            cw.writerow(data_row)


# load data from vocabulary
def load_from_json(infile_name):
    with open(infile_name, "r") as f:
        logger.debug("Loaded from json: %s"%infile_name)
        json_str = f.read()
        return json.loads(json_str)

# load data from csv
def load_from_csv(infile_name):
    with open(infile_name, "r") as f:
        read_rows =  csv.reader(f,delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = []
        for row in read_rows:
            rows.append(row)
        logger.debug("Loaded from csv: %s."%infile_name)
        return rows

def create_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def num_lines(file_name):
    return int(execute_command("wc -l %s"%file_name).split()[0])

# get a hex color range for number of parts
def get_N_HexCol(N=5):
    HSV_tuples = [(x*1.0/N, 1, 1) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = tuple(map(lambda x: int(x*255),colorsys.hsv_to_rgb(*rgb)))
        hex_out.append("#%.2X%.2X%.2X"%rgb )
    return hex_out
"""
def process_one_line(line_q, result_q, ):
    # get item blocking
    new_line = line_q.get(timeout=5)
    # do something
    result =  experiment_lib.extract_pattern_id(new_line)
    # get back result
    result_q.put(result)
"""
def multiprocess_file(file_name, process_one_line,  num_processes=8, max_size=10000):
    # define multithreading
    line_q = Queue(maxsize=max_size)
    result_q = Queue()

    # process for scooping lines to process on input q
    def load_line(line_q, file_name):
        for l in open(file_name, 'r'):
            line_q.put(l)
    # wrapper for processing the line in one loop
    def proccess_one_line_loop(line_q,result_q,pid):
        try:
            while True:
                process_one_line(line_q,result_q)
        except Exception as e:
            print(e)
            print("Shutting down processing thread %i"%pid)

    # define processes
    processes = []
    for pid in xrange(num_processes):
        processes.append(Process(target=proccess_one_line_loop, args=(line_q,result_q,pid)))

    line_load_p = Process(target=load_line, args=(line_q,file_name))

    # start threads
    [p.start() for p in processes]
    line_load_p.start()

    return result_q, line_q, processes, line_load_p

# Print iterations progress
def print_progress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")


def max_steps(log_name, batch_size, num_epochs=1):
    examples_per_epoch = num_lines(log_name)
    steps_per_epoch = examples_per_epoch/batch_size
    return int(steps_per_epoch * num_epochs)

def every_xth_steps(log_name, num_times, batch_size,  num_epochs=1):
    total_steps = max_steps(log_name, batch_size, num_epochs)
    return max(int(total_steps/num_times),1)

def import_all(module_name, glob):
    import importlib
    mod = importlib.import_module(module_name)
    module_dict = mod.__dict__
    try:
        to_import = mod.__all__
    except AttributeError:
        to_import = [name for name in module_dict if not name.startswith('_')]
    for name in to_import:
        glob.update({name:module_dict[name]})
    return mod

def print_one_example(batch):
    print("x_e:%s"%batch[0][0])
    print("x_d:%s"%batch[1][0])
    print("y_d:%s"%batch[2][0])
    print("s_e:%s"%batch[3][0])
    print("s_d:%s"%batch[4][0])

def save_result(experiment_id, result_name, value ):
    results_file = "results/all_results.json"
    if os.path.exists(results_file):
        all_results = load_from_json(results_file)
    else:
        all_results = {}

    if not experiment_id in all_results:
        all_results[experiment_id]={}
    if not result_name in all_results[experiment_id]:
        all_results[experiment_id][result_name]={}
    all_results[experiment_id][result_name]["value"]=value
    all_results[experiment_id][result_name]["time"]=time.time()

    save_to_json( all_results , results_file )

def cluster_name(name):
    return name.replace("050_","").replace(".py","")

def get_avg_seq_length(params):
    seq_length_file = os.path.join("data","sequence_lengths","%s_dec.idx"%params["log_name"])
    lines = [int(s) for s in list(open(seq_length_file,"r"))]
    return np.mean(lines), np.std(lines)

def get_used_signatures(params):
    assigned_signature_file = os.path.join("data","signatures","%s.sig"%params["log_name"])
    return len(set([int(s) for s in list(open(assigned_signature_file,"r"))]))

def file_exists(path):
    import os
    if not os.path.exists(path):
        return False
    return os.stat(path).st_size>1000

def get_max_seq_length(params):
    seq_length_file = os.path.join("data","sequence_lengths","%s_dec.idx"%params["log_name"])
    return max([int(s) for s in list(open(seq_length_file,"r"))])
