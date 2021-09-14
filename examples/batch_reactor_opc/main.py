#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

from casadi.tools import *
import sys
sys.path.append('../../')

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from template_estimator import template_estimator
from template_opcua import template_opcua
from do_mpc.opcua.client import Client
from do_mpc.opcua.realtimemodules import RealtimeTrigger

""" User settings: """
show_animation = True
store_results = False

"""
User settings
"""
store_data     = False
plot_results   = True
plot_animation = True

'''Create the model and server'''
model = template_model()
opc_server, opc_opts = template_opcua(model)

rt_simulator = template_simulator(model, opc_opts)
rt_controller = template_mpc(model, opc_opts)
rt_estimator = template_estimator(model, opc_opts)
"""
Initialization and preparation of the server data base
"""
# The user is an object that lets the main thread access the OPCUA server for status and flag checks
opc_opts['_opc_opts']['_client_type'] = 'ManualUser'
user = Client(opc_opts['_opc_opts'])
user.connect()
# Start all modules if the manual mode is enabled and avoid delays
if opc_opts['_user_controlled']:
    user.updateSwitches(pos = -1, switchVal=[1,1,1,0,0])

X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
S_s_0 = 0.5 # This is the controlled variable [mol/l]
P_s_0 = 0.0 #[C]
V_s_0 = 120.0 #[C]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])


rt_controller.x0 = x0
rt_simulator.x0 = x0
rt_estimator.x0 = x0

rt_controller.set_initial_guess()
rt_controller.init_server(rt_controller._u0.cat.toarray().tolist())

# Step 4: the cyclical operation can be safely started now
"""
Define triggers for each of the modules and start the parallel/asynchronous operation
"""
trigger_simulator  = RealtimeTrigger(rt_simulator.cycle_time , rt_simulator.asynchronous_step)
trigger_estimator  = RealtimeTrigger(rt_estimator.cycle_time , rt_estimator.asynchronous_step)
trigger_controller = RealtimeTrigger(rt_controller.cycle_time, rt_controller.asynchronous_step)

"""
Setup graphic:
"""
# Once the main thread reaches this point, all real-time modules will be stopped
trigger_controller.stop()
trigger_simulator.stop()
trigger_estimator.stop()



