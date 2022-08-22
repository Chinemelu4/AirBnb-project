import json
import uuid
from datetime import datetime
from time import sleep

import pyarrow.parquet as pq
import requests

table = pq.read_table("test_par.parquet")
data = table.to_pydict()


with open("target.csv", 'w') as f_target:
    for row in data:
        f_target.write(f"{row['id']},{'host_identity_verified'},{'neighbourhood group'},{'instant_bookable'},{'cancellation_policy'},{'room type'},{'Construction year'},{'service fee'},{'minimum nights'},{'number of reviews'}\n")
        resp = requests.post("http://localhost:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=json.dumps(row)).json()
        print(f"prediction: {resp['duration']}")
        sleep(1)
