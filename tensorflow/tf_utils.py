#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 13:39:18 2017

@author: llan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

def get_cluster_spec():
    return eval(os.environ.get("CLUSTER_SPEC"))

def get_job_name():
    return os.environ.get("JOB_NAME")

def get_task_index():
    return int(os.environ.get("TASK_INDEX"))