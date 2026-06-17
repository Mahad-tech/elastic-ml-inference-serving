# simple_load.py

import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8001/add_to_queue"
IMG = "../imagenet-sample-images/Sasha.jpg"

def send(i):
    with open(IMG, "rb") as f:
        r = requests.post(URL, files={"image": f})
    print(i, r.status_code)

with ThreadPoolExecutor(max_workers=50) as ex:
    for i in range(500):
        ex.submit(send, i)