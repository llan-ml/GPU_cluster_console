# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:46:10 2016

@author: lanlin
"""

import os
import argparse
import yaml

def main(args):
    with open(args.config, "r") as f:
        config = yaml.load(f)
    try:
        k8s_config = config["kubernetes"]
    except KeyError:
        k8s_config = {}
    try:
        tf_config = config["tensorflow"]
    except KeyError:
        tf_config = {}
    try:
        data_config = config["data"]
    except KeyError:
        data_config = {}
    
    num_gpus = k8s_config["num_gpus"]
    k8s_config.pop("num_gpus")
    k8s_config["ps_replicas"] = num_gpus \
        if num_gpus % 2 != 0 else num_gpus - 1
    k8s_config["worker_replicas"] = num_gpus
    
    k8s_args = ["--{} \"{}\"".format(key, value)
        for key, value in k8s_config.iteritems()]
    k8s_args = " ".join(k8s_args) if k8s_args else ""
    
    tf_args = [(key, value) for key, value in tf_config.iteritems()]
    tf_args = "--parameters \"{}\"".format(tf_args) if tf_args else ""
    render_args = "{} {}".format(k8s_args, tf_args)
    
    os.system("python ./kubernetes/render.py {}".format(render_args))
    copy_files = ["./tensorflow/distribute_tf.py"]
    if "user_model_file" not in tf_config.keys() \
        or tf_config["user_model_file"] == "example_user_model.py":
        copy_files.append("./examples/example_user_model.py")
    else:
        copy_files.append(tf_config["user_model_file"])
    if "data_dir" not in data_config.keys() \
        or data_config["data_dir"] == "./examples/data":
        copy_files.append("./examples/data")
    else:
        copy_files.append(data_config["data_dir"])

    work_dir = ROOT_DIR + "../gpu_cluster_storage/{}/{}".format(
                                                k8s_config["namespace"],
                                                k8s_config["name"])

    os.system("mkdir -p {}".format(work_dir))
    os.system("cp -r {} {}".format(" ".join(copy_files), work_dir))
    os.system("chmod +x {}/start_job.sh".format(work_dir))
    os.system("chmod +x {}/stop_job.sh".format(work_dir))
#    os.system("chmod +x {}/start_tensorboard.sh".format(work_dir))
#    os.system("chmod +x {}/stop_tensorboard.sh".format(work_dir))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
        
    parser.add_argument(
        "--config",
        default="./examples/example_config.yaml")
    
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ROOT_DIR = "{}/".format(os.getcwd())
    config=main(args)
