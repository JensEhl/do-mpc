import sys
sys.path.append('../../')
import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import pickle
import do_mpc
import opcua
import time
from do_mpc.simulator import Simulator
from do_mpc.estimator import StateFeedback, EKF, MHE
from do_mpc.controller import MPC
from do_mpc.opcua.client import Client

"""
The following represent the real-time MPC modules, which are inherited from the base do-mpc classes: Simulator, Estimator (SFB, EKF, MHE) and the NMPC Controller
"""


class RealtimeSimulator(Simulator):
    """The basic real-time, asynchronous simulator, which expands on the ::do-mpc class Simulator.
    This class implements an asynchronous operation, making use of a connection to a predefined OPCUA server, as
    a means of exchanging information with other modules, e.g. an NMPC controller or an estimator.
    """

    def __init__(self, model, opts):
        """

        :param model: Initial state
        :type model: numpy array
        :opts: a dictionary of parameters, mainly cycle_time and data structure of the server
        :type opts: cycle_time: float
                    opc_opts: dict, see Client settings

        :return: None
        :rtype: None
        """
        assert opts['_opc_opts'][
                   '_client_type'] == 'simulator', "You must define this module with asimulator OPC Client. Review your opts dictionary."

        super().__init__(model)
        self.enabled = False
        self.iter_count = 0
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        self.user_controlled = opts['_user_controlled']

        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time simulator could not connect to the server. Please check the server setup.")

        # The server must be initialized with the x0 and p0 values of the simulator
        tag = "ns=2;s=" + self.opc_client.namespace['PlantData']['x']
        self.opc_client.writeData(self._x0.cat.toarray().tolist(), tag)

    def init_server(self, dataVal):
        """Initializes the OPC-UA server with the first plant values. The simulator is typcially the one that starts first and writes state and output values.
        If the operation does not succeed, the simulator is deemed unable to carry on and the `self.enable` attribute is set to `False`.

        :param dataVal: the first optimal input vector at time t=0
        :type dataVal: list of float

        :return result: The result of server write operation
        :rtype result: boolean
        """
        tag = "ns=2;s=" + self.opc_client.namespace['PlantData']['x']
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time simulator could not connect to the server. Please correct the server setup.")

    def start(self):
        """Alternative method to start the simulator from the console. The client is usually automatically connected upon instantiation.

        :return result: The result of the connection attempt to the OPC-UA server.
        :rtype result: boolean
        """
        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time simulator could not connect to the server. Please check the server setup.")
        return self.enabled

    def stop(self):
        """ Stops the execution of the real-time estimator by disconnecting the OPC-UA client from the server.
        Throws an error if the operation cannot be performed.

        :return result: The result of the disconnect operation
        :rtype: boolean
        """
        try:
            self.opc_client.disconnect()
            self.enabled = False
        except RuntimeError:
            print(
                "The real-time simulator could not be stopped due to server issues. Please stop the client manually and delete the object!")
        return self.enabled

    def asynchronous_step(self):
        """ This function implements the server calls and simulator step. It must be used in combination
            with a real-time trigger of the type ::py:class:`RealtimeTrigger`, which calls this routine with a predefined frequency.
            The starting and stopping are also controlled via the trigger object.

        :param no params: because all information is stored in members of the object

        :return: none
        :rtype: none
        """

        tag_in = "ns=2;s=" + self.opc_client.namespace['ControllerData']['u_opt']
        tag_out_x = "ns=2;s=" + self.opc_client.namespace['PlantData']['x']
        tag_out_y = "ns=2;s=" + self.opc_client.namespace['PlantData']['y']
        tag_out_p = "ns=2;s=" + self.opc_client.namespace['PlantData']['p']

        # Read the latest control inputs from server and execute step
        uk = np.array(self.opc_client.readData(tag_in))

        yk = self.make_step(uk)
        xk = self._x0.cat.toarray()
        # The parameters can't be read using p_fun because the internal timestamp is not relevant
        pk = self.sim_p_num['_p'].toarray()

        # The full state vector is written back to the server (mainly for debugging and tracing)
        self.opc_client.writeData(xk.tolist(), tag_out_x)
        # The measurements are written to the server and will be used by the estimators
        self.opc_client.writeData(yk.tolist(), tag_out_y)
        # And also the current parameters, which are needed e.g by the estimator
        self.opc_client.writeData(pk.tolist(), tag_out_p)

        self.iter_count = self.iter_count + 1


class RealtimeController(MPC):
    """The basic real-time, asynchronous simulator, which expands on the ::do-mpc class Simulator.
    This class implements an asynchronous operation, making use of a connection to a predefined OPCUA server, as
    a means of exchanging information with other modules, e.g. an NMPC controller or an estimator.
    """

    def __init__(self, model, opts):
        """

        :param model: Initial state
        :type model: numpy array
        :opts: a dictionary of parameters, mainly cycle_time and data structure of the server
        :type opts: cycle_time: float
                    opc_opts: dict, see Client settings

        :return: None
        :rtype: None
        """
        assert opts['_opc_opts'][
                   '_client_type'] == 'controller', "You must define this module with a controller OPC Client. Review your opts dictionary."

        super().__init__(model)
        self.enabled = True
        self.iter_count = 0
        self.output_feedback = opts['_output_feedback']
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        self.user_controlled = opts['_user_controlled']

        try:
            self.opc_client.connect()
        except RuntimeError:
            print("The real-time controller could not connect to the server.")
            self.is_ready = False

        # The server must be initialized with the nonzero values for the inputs and the correct input structure
        if self.opc_client.connected:
            tag = "ns=2;s=" + self.opc_client.namespace['ControllerData']['u_opt']
            self.opc_client.writeData(model._u(0).cat.toarray().tolist(), tag)
            self.is_ready = True
        else:
            print("The server data could not be initialized by the controller!")
            self.is_ready = False

    def init_server(self, dataVal):
        """Initializes the OPC-UA server with the first MPC values for the optimal inputs to prevent simulator or estimator crashes.
        If the operation does not succeed, the controller is deemed unable to carry on and the `self.enable` attribute is set to `False`.

        :param dataVal: the first optimal input vector at time t=0
        :type dataVal: list of float

        :return result: The result of server write operation
        :rtype result: boolean
        """
        tag = "ns=2;s=" + self.opc_client.namespace['ControllerData']['u_opt']
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time controller could not connect to the server. Please correct the server setup.")
        # One optimizer iteration is called before starting the cyclical operation
        self.asynchronous_step()

        return self.enabled

    def start(self):
        """Alternative method to start the client from the console. The client is usually automatically connected upon instantiation.

        :return result: The result of the connection attempt to the OPC-UA server.
        :rtype result: boolean
        """
        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time controller could not connect to the server. Please check the server setup.")
        return self.enabled

    def stop(self):
        """ Stops the execution of the real-time estimator by disconnecting the OPC-UA client from the server.
        Throws an error if the operation cannot be performed.

        :return result: The result of the disconnect operation
        :rtype: boolean
        """
        try:
            self.opc_client.disconnect()
            self.enabled = False
        except RuntimeError:
            print(
                "The real-time controller could not be stopped due to server issues. Please stop the client manually and delete the object!")
        return self.enabled

    def check_status(self):
        """This function is called before every optimization step to ensure that the server data
        is sane and a call to the optimizer can be made in good faith, i.e. the plant state
        contains meaningful data and that no flags have been raised.

        :param no params: this function onyl needs internal data

        :return: check_result is the result of all the check done by the controller before executing the step
        :rtype: boolean
        """
        check_result = self.is_ready
        # Step 1: check that the server is running and the client is connected
        check_result = self.opc_client.connected and check_result
        if check_result == False:
            print("The controller check failed because: controller not connected to server.")
            return False

        # Step 2: check whether the user has requested to run the optimizer
        if self.user_controlled:
            check_result = check_result and self.opc_client.checkSwitches(pos=0)
            if check_result == False:
                print("The controller check failed because: controller not manually enabled on the server.")
                return False

        # Step 3: check whether the controller should run and no controller flags have been raised
        # flags = [0-controller, 1-simulator, 2-estimator, 3-monitoring, 4-extra]
        check_result = check_result and not self.opc_client.checkFlags(pos=0)
        if check_result == False:
            print("The controller check failed because: controller has raised a failure flag.")
            return False

        # Step 4: check that the plant/simulator is running and no simulator flags have been raised
        check_result = check_result and not (self.opc_client.checkFlags(pos=1) or self.opc_client.checkFlags(pos=2))
        self.is_ready = check_result
        if check_result == False:
            print(
                "The controller check failed because: either the simulator or estimator have reported crap data. Unsafe to run the controller!")
        return check_result

    def initialize_optimizer(self, style='static'):
        """This is an internal function meant to be called before each controller step to reinitialize the initial guess of
        the NLP solver with the most recent plant data.

        :param style: The type of initialization to be performed. A choice of three options is available. `static` denotes a crude
                    initialization using the current state values over the entire prediction horizon. `dynamic` uses an integrator
                    to simulate from the current state, over the prediction horizon, using the current inputs. Finally `predictive`
                    uses a one-step lookahead to pre-initialize the optimizer considering the average delay time.
        :type style: string in [`static`, `dynamic`, `predictive`]

        :return: none
        :rtype: none
        """
        if style == 'static':
            # This is a 'crude' initialization, which copies the same value over the entire prediction horizon
            x0 = np.array(self.opc_client.readData())
        if style == 'dynamic':
            # This is a dynamic initialization
            x0 = np.array(self.opc_client.readData())
        if style == 'predictive':
            # TODO: implement the step ahead prediction based on the cycle time delays
            x0 = np.array(self.opc_client.readData())

        # The NLP must be reinitialized with the most current data from the plant readings
        self.set_initial_state(x0, reset_history=True)
        self.set_initial_guess()

    def asynchronous_step(self):
        """This function implements the server calls and simulator step with a predefined frequency
        :param no params: because the cycle is stored by the object

        :return: time_left, the remaining time on the clock when the optimizer has finished the routine
        :rtype: float
        """
        if self.output_feedback == False:
            tag_in = "ns=2;s=" + self.opc_client.namespace['PlantData']['x']
        else:
            tag_in = "ns=2;s=" + self.opc_client.namespace['EstimatorData']['xhat']

        tag_out = "ns=2;s=" + self.opc_client.namespace['ControllerData']['u_opt']

        # Read the latest plant state from server and execute optimization step
        xk = np.array(self.opc_client.readData(tag_in))

        # The NLP must be reinitialized with the most current data from the plant readings

        self.x0 = (np.array(self.opc_client.readData(tag_in)))
        self.set_initial_guess()

        # Check the current status before running the optimizer step
        if self.check_status():
            # The controller can be executed
            uk = self.make_step(xk)
            # The iteration count is incremented regardless of the outcome
            self.iter_count = self.iter_count + 1

            if self.solver_stats['return_status'] == 'Solve_Succeeded':
                # The optimal inputs are written back to the server
                self.opc_client.writeData(uk.tolist(), tag_out)
            else:
                print("The controller failed at time ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
                print("The optimal inputs have not been updated on the server.")
            # The controller must wait for a predefined time
            time_left = self.cycle_time - self.solver_stats['t_wall_total']

        else:
            time_left = self.cycle_time
            print(
                "The controller is still waiting to be manually activated. When you're ready set the status bit to 1.")

        return time_left


class RealtimeEstimator():
    """The basic real-time, asynchronous estimator, which expands on the ::python:class:`do-mpc.Estimator`. The inheritance is done selectively,
    which means that this class prototype can inherit from any of the available estimators. The selection is done at runtime and must be specified by the user
    through a type variable (see the constructor of the class). This class implements an asynchronous operation, making use of a connection to a predefined OPCUA server, as
    a means of exchanging information with other modules, e.g. an NMPC controller or a simulator.
    """

    def __new__(cls, etype, model, opts):
        etype = {'SFB': StateFeedback, 'EKF': EKF, 'MHE': MHE}[etype]
        cls = type(cls.__name__ + '+' + etype.__name__, (cls, etype), {})
        return super(RealtimeEstimator, cls).__new__(cls)

    def __init__(self, etype, model, opts):
        """The constructor of the class. Creates a real-time estimator and sets the parameters.

        :param etype: The base estimator type to inherit from. Available types are `SFB`=::py:class:`do_mpc:Estimator:StateFeedback`,
        `EKF`=::py:class:`do_mpc:Estimator:EKF` and `MHE`=::py:class:`do_mpc:Estimator:MHE`
        :type etype: string

        :param model: A **do-mpc** model which contains the data used to initialize the estimator
        :type model: ::py:class:`do_mpc:Model`

        :opts: a dictionary of parameters, mainly cycle_time and data structure of the server
        :type opts: cycle_time: float
                    opc_opts: dict, see Client settings

        :return: None
        :rtype: None
        """
        assert opts['_opc_opts'][
                   '_client_type'] == 'estimator', "You must define this module with an estimator OPC Client. Please review the opts dictionary."

        super().__init__(model)
        self.etype = etype
        self.enabled = False
        self.iter_count = 0
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        self.user_controlled = opts['_user_controlled']

        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time estimator could not connect to the server. Please check the server setup.")

        # The server must be initialized with the x0 values of the simulator
        tag = "ns=2;s=" + self.opc_client.namespace['EstimatorData']['xhat']
        self.opc_client.writeData(self._x0.cat.toarray().tolist(), tag)

    def init_server(self, dataVal):
        """Initializes the OPC-UA server with the initial estimator values (states and parameters). If the operation does
        not succeed, the estimator is deemed unable to carry on and the `self.enable` attribute is set to `False`.
        """
        tag = "ns=2;s=" + self.opc_client.namespace['EstimatorData']['xhat']
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time estimator could not connect to the server. Please correct the server setup.")

    def start(self):
        """Alternative method to start the client from the console. The client is usually automatically connected upon instantiation.

        :return result: The result of the connection attempt to the OPC-UA server.
        :rtype result: boolean
        """
        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time estimator could not connect to the server. Please check the server setup.")
        return self.enabled

    def stop(self):
        """ Stops the execution of the real-time estimator by disconnecting the OPC-UA client from the server.
        Throws an error if the operation cannot be performed.

        :return result: The result of the disconnect operation
        :rtype: boolean
        """
        try:
            self.opc_client.disconnect()
            self.enabled = False
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print(
                "The real-time estimator could not be stopped because the connection to the server was interrupted. Please stop the client manually and delete the object!")
        return self.enabled

    def asynchronous_step(self):
        """This function implements one server call and estimator step, after which it writes output data back to the server.

        :param no params: all parameters are stored by the underlying estimator object

        :return: time_left, represents the leftover time until the max cycle time of the estimator
        :rtype: none
        """

        # Read the latest plant data from server
        tag_in_u = "ns=2;s=" + self.opc_client.namespace['ControllerData']['u_opt']
        tag_in_x = "ns=2;s=" + self.opc_client.namespace['PlantData']['x']
        tag_in_y = "ns=2;s=" + self.opc_client.namespace['PlantData']['y']
        tag_in_p = "ns=2;s=" + self.opc_client.namespace['PlantData']['p']
        tag_out_x = "ns=2;s=" + self.opc_client.namespace['EstimatorData']['xhat']
        tag_out_p = "ns=2;s=" + self.opc_client.namespace['EstimatorData']['phat']

        tic = time.time()

        if self.etype == 'SFB':
            xk = np.array(self.opc_client.readData(tag_in_x))
            xk_hat = self.make_step(xk)
            # The latest estimates are written back to the server
            self.opc_client.writeData(xk_hat.tolist(), tag_out_x)

        if self.etype == 'EKF':
            uk = np.array(self.opc_client.readData(tag_in_u))
            yk = np.array(self.opc_client.readData(tag_in_y))
            pk = np.array(self.opc_client.readData(tag_in_p))
            xk_hat, pk_hat = self.make_step(yk, uk, pk)

            # The latest estimates are written back to the server
            self.opc_client.writeData(xk_hat.tolist(), tag_out_x)
            try:
                self.opc_client.writeData(pk_hat.tolist(), tag_out_p)
            except RuntimeError:
                print("Cannot write parameter estimates because the server refused the write operation")

        if self.etype == 'MHE':
            # TODO: the call is yet to be implemente din real-time fashion
            xk_hat = self._x0
            self.opc_client.writeData(xk_hat.tolist(), tag_out_x)

        toc = time.time()
        # The estimator must wait for a predefined time
        time_left = self.cycle_time - (toc - tic)

        self.iter_count = self.iter_count + 1
        return time_left


"""
The trigger class implements a real-time mechanism that can be used to execute any code(Python function) with a predefined freqency.
The user must instantiate several objects of this class, one for each of the real-time modules employed. 
"""
from threading import Timer


class RealtimeTrigger(object):
    """
    This class is employed in timing the execution of your real-time ::do-mpc modules. One RealtimeTrigger is required
    for every module, i.e. one for the simulator, one for the controller and one for the estimator, if the latter is present.
    """

    def __init__(self, interval, function, *args, **kwargs):
        """This function implements the server calls and simulator step with a predefined frequency

        :param interval: the cycle time in seconds representing the frequency with which the target function is executed
        :type interval: integer

        :param function: a function to be called cyclically
        :type function: python function header

        :param args: arguments to pass to the target function
        :type args: python dict

        :return: none
        :rtype: none
        """
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False