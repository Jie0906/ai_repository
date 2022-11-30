import time


def sleep_time(hour, min, sec):
    return hour * 3600 + min *60 +sec

second = sleep_time(0,0,5)

time.sleep(second)
print('hello test2')


