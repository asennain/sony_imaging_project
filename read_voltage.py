from pyfirmata import Arduino, util
from pyfirmata.util import Iterator
import time

#! Don't read anything more than 5V! Board will fry. 
board = Arduino('COM6') #Select the correct port
it = util.Iterator(board)
it.start()
while True:
   board.analog[0].enable_reporting()
   time.sleep(.1)
   print((board.analog[0].read())*5.00)
   time.sleep(0.1)
   