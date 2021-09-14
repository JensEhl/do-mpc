from do_mpc.opcua.opcmodules import RealtimeEstimator

def template_estimator(model, opc_opts):

    opc_opts['_opc_opts']['_client_type'] = "estimator"
    opc_opts['cycle_time'] = 2.0
    opc_opts['output_feedback'] = True

    estimator = RealtimeEstimator('SFB', model, opc_opts)

    return estimator
