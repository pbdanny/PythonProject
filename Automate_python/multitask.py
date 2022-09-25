import time
import datetime
import threading

def take_a_nap():
    time.sleep(5)
    print('Awake!')

threadObj = threading.Thread(target=take_a_nap)
threadObj.start()

print('Program Ends')