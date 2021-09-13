import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import do_mpc
from do_mpc.opcmodules import RealtimeEstimator

def template_estimator(model, opc_opts):

    opc_opts['_opc_opts']['_client_type'] = "estimator"
    opc_opts['cycle_time'] = 2.0
    opc_opts['output_feedback'] = True

    estimator = RealtimeEstimator('SFB', model, opc_opts)

    return estimator
