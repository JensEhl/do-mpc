
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
from do_mpc.tools.timer import Timer

import matplotlib.pyplot as plt
import pickle
import time

from do_mpc.opcmodules import Server, Client
from do_mpc.opcmodules import RealtimeSimulator, RealtimeController, RealtimeEstimator
from do_mpc.opcmodules import RealtimeTrigger

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from template_estimator import template_estimator
from templat_opcua import template_opcua
""" User settings: """
show_animation = True
store_results = False


model = template_model()

opc_server, opc_opts = template_opcua(model)
#mpc = template_mpc(model)
#simulator = template_simulator(model)
#estimator = do_mpc.estimator.StateFeedback(model)


estimator = template_estimator(model, opc_opts)
mpc       = template_mpc(model, opc_opts)
simulator = template_simulator(model, opc_opts)


opc_opts['_opc_opts']['_client_type'] = 'ManualUser'
user = Client(opc_opts['_opc_opts'])
user.connect()

if opc_opts['_user_controlled']:
    user.updateSwitches(pos=-1, switchVal=[1, 1, 1, 0, 0])
    print("Switches updated manually!")

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0


simulator.init_server(x0.tolist())
mpc.init_server(x0.tolist())
estimator.init_server(x0.tolist())
print("Initial values set")

# Step 4: the cyclical operation can be safely started now
"""
Define triggers for each of the modules and start the parallel/asynchronous operation
"""
trigger_simulator  = RealtimeTrigger(simulator.cycle_time, simulator.asynchronous_step)

trigger_estimator  = RealtimeTrigger(estimator.cycle_time, estimator.asynchronous_step)

trigger_mpc        = RealtimeTrigger(mpc.cycle_time, mpc.asynchronous_step)
