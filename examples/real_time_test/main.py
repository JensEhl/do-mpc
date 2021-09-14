from casadi.tools import *
import sys
sys.path.append('../../')
import do_mpc

import matplotlib.pyplot as plt
import time

from do_mpc.opcua.opcmodules import Client
from do_mpc.opcua.opcmodules import RealtimeTrigger

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from template_estimator import template_estimator
from templat_opcua import template_opcua
""" User settings: """
plot_animation = True
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


graphics = do_mpc.graphics.Graphics(mpc.data)


fig, ax = plt.subplots(2, sharex=True)
plt.ion()
# Configure plot:
graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
ax[0].set_ylabel('$C$ [mol/l]')
ax[1].set_ylabel('$T$ [K]')
fig.align_ylabels()
plt.ion()

max_iter = 100
manual_stop = False

while mpc.iter_count < max_iter and manual_stop == False:
    # The code below is executed on the main thread (e.g the Ipython console you're using to start do-mpc)
    print("Waiting on the main thread...Checking flags...Executing your main code...")

    if user.checkFlags(pos=0) == 1:
        print("The controller has failed! Better take backup action ...")
    if user.checkFlags(pos=0) == 0: print("All systems OK @ ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))

    # Checking the status of the modules
    switches = user.checkSwitches()
    print("The controller is:", 'ON' if switches[0] else 'OFF')
    print("The simulator  is:", 'ON' if switches[1] else 'OFF')
    print("The estimator  is:", 'ON' if switches[2] else 'OFF')

    # Check the 5th flag and stop all modules at once if the user has raised the flag
    # Alternatively, the user can individually stop modules by setting the switch to 0
    if user.checkFlags(pos=4) == 1:
        user.updateSwitches(pos=-1, switchVal=[0, 0, 0])
        manual_stop = True

    if plot_animation:
        graphics.plot_results()
        graphics.plot_predictions()
        graphics.reset_axes()
        plt.show()

    # The main thread sleeps for 10 seconds and repeats
    time.sleep(1)

# Once the main thread reaches this point, all real-time modules will be stopped
trigger_mpc.stop()
trigger_simulator.stop()
trigger_estimator.stop()

"""
All OPCUA services should be terminated and the communications closed, to prevent Python errors
"""
simulator.stop()
mpc.stop()
estimator.stop()
opc_server.stop()
del (opc_server)
