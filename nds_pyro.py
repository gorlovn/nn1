#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Доступ к nds_predict через pyro

Created on Thu Jul 25 12:26 2024

@author: gnv
"""
import os
import sys
import logging

import Pyro5.api

from helpers import setup_logger
from nds_predict_k import load_k_model
from nds_predict_k import main

if __name__ == "__main__":
    log = setup_logger('', '_nds_pyro_server.out', console_out=True)
else:
    log = logging.getLogger(__name__)

W_MODEL = load_k_model()
if W_MODEL is None:
    log.error("Unable to load model")
    sys.exit(1)


@Pyro5.api.expose
class NdsPredict(object):
    def predict_by_people_id(self, people_id):

        _r = main(people_id, W_MODEL)
        return _r


# make a Pyro daemon
daemon = Pyro5.api.Daemon()
# register the nds predict as a Pyro object
uri = daemon.register(NdsPredict)

# Locate the name server
ns = Pyro5.api.locate_ns()

# Register your object with the name server
ns.register("NdsPredict", uri)
print(f"Registered object with name server under the name 'NdsPredict'")

print("Ready.")
# start the event loop of the server to wait for calls
daemon.requestLoop()
