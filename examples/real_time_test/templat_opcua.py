from do_mpc.opcmodules import Server, Client


def template_opcua(model):
    """
    --------------------------------------------------------------------------
    template_opcua: the parameters for the server and client
    --------------------------------------------------------------------------
    """
    # Defining the settings for the OPCUA server
    server_opts = {"_model": model,  # must be the model used to define the controller
                   "_name": "Poly Reactor OPCUA",  # give the server whatever name you prefer
                   "_address": "opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
                   "_port": 4840,  # does not need changing
                   "_server_type": "with_estimator",
                   # to use with either SFB or EKF estimators or select "basic" for no estimates
                   "_store_params": True,  # should always be set to True to spare yourself the headaches
                   "_store_predictions": False,  # set to True only if you plan to do fancy things with the predictions
                   "_n_steps_pred": 0,  # set to the number of predicted steps you want to store on the server
                   "_with_db": False}  # set to True if you plan to use the SQL database during/after the runtime
    opc_server = Server(server_opts)

    # The server can only be started if it hasn't been already created
    if opc_server.created == True and opc_server.running == False: opc_server.start()

    # Defining the settings for the OPCUA clients, which MUST match the server
    client_opts = {"_address": "opc.tcp://localhost:4840/freeopcua/server/",
                   "_port": 4840,
                   "_client_type": "simulator/estimator/controller",
                   "_namespace": opc_server.namespace}

    opc_opts = {'_cycle_time': 5.0,
                # the frequency with which a module is executed (can be different for each real-time module)
                '_output_feedback': False,
                # specifies the feedback scheme: False = state feedback, True = output feedback (implies an estimator must be running)
                '_user_controlled': True,  # specifies whether the user should start and stop the modules manually
                '_opc_opts': client_opts}  # passes on the client options for each real-time module to be able to connect

    return opc_server, opc_opts