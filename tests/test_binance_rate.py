#!/usr/bin/env python
import logging

from binance.cm_futures import CMFutures
from binance.lib.utils import config_logging

@pytest.mark.skip()
def test():
    config_logging(logging, logging.DEBUG)


    key = "isQR14OjrXeyl8Kzp1IJdXFw0hY7fli1cceJGpWeTAzFATn80hE5LIqe2N3fnU6P"

    # historical_trades requires api key in request header
    cm_futures_client = CMFutures(key=key)
    logging.info(cm_futures_client.historical_trades("BTCUSD_PERP", **{"limit": 10}))
    print(cm_futures_client.historical_trades("BTCUSD_PERP", **{"limit": 10}))
