#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Клиент для доступа к nds_predict через pyro

Created on Thu Jul 25 12:45 2024

@author: gnv
"""
import os

import Pyro5.api
import Pyro5.errors

# Locate the name server
ns = Pyro5.api.locate_ns()
uri = ns.lookup("NdsPredict")
print(f"Found URI: {uri}")

if uri is not None:
    nds_predict = Pyro5.api.Proxy(uri)
    try:
        rr = nds_predict.predict_by_people_id(1)
    except Pyro5.errors.CommunicationError as _e:
        rr = f"{_e}"

    print(rr)
