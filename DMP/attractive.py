import numpy as np

def Att(x_target, x, spherepos, dT = 0.05, gamma_T = 2.5e3): # 2.5e3
    attraction = np.zeros((3))
    d = np.linalg.norm(spherepos - x)
    if d > dT:
        attraction = gamma_T * ( x_target - x ) * np.exp( d )
    return attraction