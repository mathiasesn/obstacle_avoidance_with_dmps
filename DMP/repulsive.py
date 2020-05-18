<<<<<<< HEAD
import numpy as np
from obstacle import Obstacle

def R_from_axis_angle(v : np.array, theta):
    R = np.diag([1]*3) * np.cos(theta)
    R += np.sin(theta) * skew(v)
    R += (1 - np.cos(theta)) * np.matmul(v, v.transpose())
    return R

def angle(v1 : np.array, v2 : np.array):
    return np.arccos(np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) ))

def skew(v : np.array):
    if len(v) != 3:
        print("ERROR: input to skew() is not a 3x1 vector")
        return -1
    mat = np.zeros((3, 3))
    mat[0,1] = -v[2]
    mat[0,2] =  v[1]
    mat[1,0] =  v[2]
    mat[1,2] = -v[0]
    mat[2,0] = -v[1]
    mat[2,1] =  v[0]
    return mat

def Ct(p,dp, sphere : Obstacle):
    def repulsive(steering_angle, gamma = 1e6, beta = 20/np.pi):
        return gamma *steering_angle * np.exp(-beta * abs(steering_angle))

    if np.linalg.norm(dp) < 0.01:
        return 0
    
    # Calculate steering_angle
    vec_obj = sphere.pos - p
    # Direction
    r   = np.cross(vec_obj, dp)
    R   = R_from_axis_angle(r, np.pi/2)
    steering_angle  = angle(vec_obj, dp)
    repuls          = np.matmul(R, dp) * repulsive(steering_angle)
    print(repuls)
    return repuls

def Ct_coupling(p, dp, sphere : Obstacle):
    if np.linalg.norm(dp) < 0.01:
        return 0

    if(sphere == None):
        return 0

    vec_obj = sphere.pos - p    # (o - x)
    d = np.linalg.norm(vec_obj) # ||o-x||

    theta   = angle(vec_obj, dp) # Steering angle
    beta    = 20 / np.pi         # Tuning parameter: angle      high beta   -> narrow region of influence of obstacle
    k       = 15                 # Tuning parameter: distance   high k      -> effect of obstacle decrease quickly with distance

    phi1    = theta * np.exp(-beta * abs(theta)) * np.exp(-k * d)
    phi2    = 0 # Not as relevant since object is a sphere.
    phi3    = np.exp(-k * d)    # To get repulsive effect when heading directly against obstacle, theta = 0.

    r   = np.cross(vec_obj, dp)
    R   = R_from_axis_angle(r, np.pi/2)
    Rdp = np.matmul(R, dp)

    # ORIGINAL VALUES
    # gamma1 = 1e6
    # gamma2 = 0
    # gamma3 = 1.25e4

    # MODIFIED VALUES FOR ONLINE DMP
    gamma1 = 3e6
    gamma2 = 0
    gamma3 = 2e4

    Ct = gamma1 * Rdp * phi1 + gamma2 * Rdp * phi2 + gamma3 * Rdp * phi3
=======
import numpy as np
from obstacle import Obstacle

def R_from_axis_angle(v : np.array, theta):
    R = np.diag([1]*3) * np.cos(theta)
    R += np.sin(theta) * skew(v)
    R += (1 - np.cos(theta)) * np.matmul(v, v.transpose())
    return R

def angle(v1 : np.array, v2 : np.array):
    return np.arccos(np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) ))

def skew(v : np.array):
    if len(v) != 3:
        print("ERROR: input to skew() is not a 3x1 vector")
        return -1
    mat = np.zeros((3, 3))
    mat[0,1] = -v[2]
    mat[0,2] =  v[1]
    mat[1,0] =  v[2]
    mat[1,2] = -v[0]
    mat[2,0] = -v[1]
    mat[2,1] =  v[0]
    return mat

def Ct(p,dp, sphere : Obstacle):
    def repulsive(steering_angle, gamma = 1e6, beta = 20/np.pi):
        return gamma *steering_angle * np.exp(-beta * abs(steering_angle))

    if np.linalg.norm(dp) < 0.01:
        return 0
    
    # Calculate steering_angle
    vec_obj = sphere.pos - p
    # Direction
    r   = np.cross(vec_obj, dp)
    R   = R_from_axis_angle(r, np.pi/2)
    steering_angle  = angle(vec_obj, dp)
    repuls          = np.matmul(R, dp) * repulsive(steering_angle)
    #print(repuls)
    return repuls

def Ct_coupling(p, dp, sphere : Obstacle):
    if np.linalg.norm(dp) < 0.01:
        return 0

    vec_obj = sphere.pos - p    # (o - x)
    d = np.linalg.norm(vec_obj) # ||o-x||

    theta   = angle(vec_obj, dp) # Steering angle
    beta    = 20 / np.pi         # Tuning parameter: angle      high beta   -> narrow region of influence of obstacle
    k       = 15                 # Tuning parameter: distance   high k      -> effect of obstacle decrease quickly with distance

    phi1    = theta * np.exp(-beta * abs(theta)) * np.exp(-k * d)
    phi2    = 0 # Not as relevant since object is a sphere.
    phi3    = np.exp(-k * d)    # To get repulsive effect when heading directly against obstacle, theta = 0.

    r   = np.cross(vec_obj, dp)
    R   = R_from_axis_angle(r, np.pi/2)
    Rdp = np.matmul(R, dp)

    gamma1 = 3e6
    gamma2 = 0
    gamma3 = 2e4

    Ct = gamma1 * Rdp * phi1 + gamma2 * Rdp * phi2 + gamma3 * Rdp * phi3
>>>>>>> c961f70e25f731f75ec6caf135d5c1b98b02ef89
    return Ct