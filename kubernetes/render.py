# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 20:18:45 2016

@author: lanlin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import jinja2

NAMESPACE_JINJA = "namespace.jinja"
NFS_JINJA = "nfs.jinja"
CLUSTER_SERVICE_JINJA = "cluster_svc.jinja"
CLUSTER_GROUP_JINJA = "cluster_job.jinja"
#NFS_TENSORBOARD_JINJA = "nfs_tensorboard.jinja"
#TENSORBOARD_JINJA = "tensorboard.jinja"
START_CLUSTER_JINJA = "start_job.jinja"
STOP_CLUSTER_JINJA = "stop_job.jinja"
#START_TENSORBOARD_JINJA = "start_tb.jinja"
#STOP_TENSORBOARD_JINJA = "stop_tb.jinja"

JINJA_FILE_0 = [NAMESPACE_JINJA, NFS_JINJA,
                CLUSTER_SERVICE_JINJA, CLUSTER_GROUP_JINJA]
#                NFS_TENSORBOARD_JINJA, TENSORBOARD_JINJA]
JINJA_FILE_1 = [START_CLUSTER_JINJA, STOP_CLUSTER_JINJA]
#                START_TENSORBOARD_JINJA, STOP_TENSORBOARD_JINJA]

def main(args):
    properties = dict(args._get_kwargs())
    dirname = ROOT_DIR + "../gpu_cluster_storage/{}/{}/k8s_config".format(
        properties["namespace"], properties["name"])
    os.system("mkdir -p {}".format(dirname))
    for jinja_file in JINJA_FILE_0:
        extension = "yaml"
        with open(jinja_file, "r") as f:
            template = jinja2.Template(f.read())
        with open("{}/{}.{}".format(
            dirname, os.path.splitext(jinja_file)[0], extension), "w") as f:
            f.write(template.render(properties=properties))

    for jinja_file in JINJA_FILE_1:
        extension = "sh"
        with open(jinja_file, "r") as f:
            template = jinja2.Template(f.read())
        with open("{}/{}.{}".format(
            os.path.dirname(dirname),
            os.path.splitext(jinja_file)[0], extension), "w") as f:
            f.write(template.render(properties=properties))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    
    parser.add_argument(
        "--namespace",
        default="llan")
    parser.add_argument(
        "--name",
        default="train")
    parser.add_argument(
        "--tf_image",
        default="tensorflow/tf-cluster-train:v1.0",
        required=False)
    parser.add_argument(
        "--train_entrypoint",
        required=True)
    parser.add_argument(
        "--ps_replicas",
        type=int,
        required=True)
    parser.add_argument(
        "--worker_replicas",
        type=eval,
        required=True)
    parser.add_argument(
        "--job_port",
        default="2222")
#    parser.add_argument(
#        "--tensorboard_port",
#        default="6006")
    parser.add_argument(
        "--nvidia_driver_version",
        default="375.26")
    parser.add_argument(
        "--nfs_image",
        default="gcr.io/google-samples/nfs-server:1.2")
        
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ROOT_DIR = "{}/../".format(os.getcwd())
    main(args)
