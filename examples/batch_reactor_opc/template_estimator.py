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

from do_mpc.opcua.realtimemodules import RealtimeEstimator


def template_estimator(model, opc_opts):
    """
    --------------------------------------------------------------------------
    template_estimator: no parameters to pass, this is a "mock" state estimator
    --------------------------------------------------------------------------
    """
    # The estimator is just a delayed state feedback estimator in this case

    opc_opts['_opc_opts']['_client_type'] = "estimator"
    opc_opts['_cycle_time'] = 3.0  # execute every 3 seconds
    opc_opts['_opc_opts']['_output_feedback'] = True

    estimator = RealtimeEstimator('SFB',model,opc_opts)

    return estimator