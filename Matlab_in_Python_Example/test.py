import matlab.engine
import numpy as np


eng = matlab.engine.start_matlab()


# Ploting test #
# x = matlab.double([1,2,3,4,5])
# y = matlab.double([1,2,3,4,5])
# t = eng.plot(x,y)

robot = eng.loadrobot("frankaEmikaPanda")

eng.show(robot)

input("test")