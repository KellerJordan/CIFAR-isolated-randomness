import os
import glob
import random
import pickle
import subprocess
import multiprocessing as mp
import pandas as pd

def run_one(proc_i, args):
    model_s, order_s, aug_s = args
    script_path = './train.py'
    cmd = 'python %s --model_seed %d --order_seed %d --aug_seed %d' % (script_path, model_s, order_s, aug_s)
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(proc_i)
    subprocess.run(cmd.split(), env=env)

def get_args_list():
    seeds_l = []
    for s1 in range(10):
        for s2 in range(10):
            for s3 in range(10):
                seeds_l.append((s1, s2, s3))
    random.shuffle(seeds_l)
    return seeds_l

def run_all(proc_i):
    for i, args in enumerate(arg_list):
        if i % 8 == proc_i:
            print(args)
            run_one(proc_i, args)

if __name__ == '__main__':
    arg_list = get_args_list()
    proc_l = []
    for i in range(8):
        p = mp.Process(target=run_all, args=(i,))
        p.start()
        proc_l.append(p)

    for p in proc_l:
        p.join()

    log_paths = glob.glob('./logs/*.pkl')
    logs = []
    for path in log_paths:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logs.append(obj)
    df = pd.DataFrame(logs)
    c = df['correct']
    print(c.min(), c.mean(), c.max())
    
