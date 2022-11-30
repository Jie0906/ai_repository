import os, sys
import json
from threading import Timer
from datetime import datetime
import time
data = []

# exp = 128
# path = f"C:/Users/user/Desktop/yolo_box/yolov5/runs/detect/exp{exp}/location_center/"
# dirlist = os.listdir( path )
# for i in dirlist:
#     with open(f"{path}{i}", 'r') as  f:
#         file = json.load(f)
#         data.append(file)

# for i in data:
#     for j in i :
#         print(j)


# def Time_threading(inc):
#     print(datetime.now(),"執行偵測中...")
#     t = Timer(inc,Time_threading,(inc,))
#     t.start()

# Time_threading(5)  #60s*60min*24h*10day


def sleep_time(hour, min, sec):
    return hour * 3600 + min *60 +sec

second = sleep_time(0,0,2)
i = 0
while True:
    
    print('hello test1')
    time.sleep(second)
    os.system('test2.py')
    print(i)
    i+=1