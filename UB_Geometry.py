import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    
    #Convert degrees into Radians
    alpha= np.radians(alpha)
    beta= np.radians(beta)
    gamma=np.radians(gamma)

    Rotation_z = np.matrix([[ np.cos(alpha), -np.sin(alpha), 0 ],
                            [ np.sin(alpha), np.cos(alpha) , 0 ],
                             [ 0           , 0            , 1 ]])

    Rotation_x = np.matrix([[ 1, 0           , 0           ],
                            [ 0, np.cos(beta),-np.sin(beta)],
                            [ 0, np.sin(beta), np.cos(beta)]])

    Rotation_z1 = np.matrix([[np.cos(gamma),  -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0           , 0            , 1 ]])
                    
    rot_xyz2XYZ = Rotation_z1*Rotation_x*Rotation_z
    print(rot_xyz2XYZ)

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    
    #Convert degrees into Radians
    alpha= np.radians(alpha)
    beta= np.radians(beta)
    gamma=np.radians(gamma)
    
    Rotation_z = np.matrix([[ np.cos(alpha), np.sin(alpha), 0 ],
                          [ -np.sin(alpha), np.cos(alpha) , 0 ],
                          [ 0           , 0            , 1 ]])

    Rotation_x = np.matrix([[ 1,       0           , 0   ],
                            [ 0, np.cos(beta),np.sin(beta)],
                            [ 0, -np.sin(beta), np.cos(beta)]])

    Rotation_z1 = np.matrix([[np.cos(gamma),  np.sin(gamma), 0 ],
                            [-np.sin(gamma), np.cos(gamma), 0],
                            [0           , 0            , 1 ]])

    rot_XYZ2xyz = Rotation_z*Rotation_x*Rotation_z1
    
    print(rot_XYZ2xyz)
    
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1






#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2],dtype = float)
    
    # Your implementation
    CHECKERBOARD_SIZE = (4, 9)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE)

    if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    img_coord = corners2
    
    #Deleting the corners from the Z axis
    img_coord = np.delete(img_coord,[16,17,18,19],axis = 0)
    img_coord = np.array(img_coord)
    img_coord = img_coord.reshape(32,2)
    return img_coord

def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation
    world_coord = np.array([[40,0,40],[40,0,30],[40,0,20],[40,0,10],
                         [30,0,40],[30,0,30],[30,0,20],[30,0,10],
                         [20,0,40],[20,0,30],[20,0,20],[20,0,10],
                         [10,0,40],[10,0,30],[10,0,20],[10,0,10],
                         [0,10,40],[0,10,30],[0,10,20],[0,10,10],
                         [0,20,40],[0,20,30],[0,20,20],[0,20,10],
                         [0,30,40],[0,30,30],[0,30,20],[0,30,10],
                         [0,40,40],[0,40,30],[0,40,20],[0,40,10]])

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation
    M_matrix = create_MijMatrix(img_coord,world_coord)
    
    u, s, vt = np.linalg.svd(M_matrix, full_matrices=False)
    Mij_11 = []
    Mij_11 = vt[11]
    Mij_Reshaped = np.array(Mij_11).reshape((3,4))
    
    #Normalising the matrix
    rv = Mij_Reshaped[2,:3]
    sc = np.linalg.norm(rv)
    lamda = 1/sc
    Mij_Reshaped = lamda * Mij_Reshaped
    
    #Retrieving the rows of matrix M
    Mij_Reshaped_R1 = [Mij_Reshaped[0][0],Mij_Reshaped[0][1],Mij_Reshaped[0][2]]
    Mij_Reshaped_R2 = [Mij_Reshaped[1][0],Mij_Reshaped[1][1],Mij_Reshaped[1][2]]
    Mij_Reshaped_R3 = [Mij_Reshaped[2][0],Mij_Reshaped[2][1],Mij_Reshaped[2][2]]
    
    #Reshaping the Matrix
    Mij_Reshaped_m1 = np.array(Mij_Reshaped_R1).reshape((1,3))
    Mij_Reshaped_m2 = np.array(Mij_Reshaped_R2).reshape((1,3))
    Mij_Reshaped_m3 = np.array(Mij_Reshaped_R3).reshape((1,3))
    
    #Transpose of M
    Mij_Reshaped_m1_T = Mij_Reshaped_m1.transpose()
    Mij_Reshaped_m2_T = Mij_Reshaped_m2.transpose()
    Mij_Reshaped_m3_T = Mij_Reshaped_m3.transpose()

    cx = np.matmul(Mij_Reshaped_m3,Mij_Reshaped_m1_T)[0][0]
    cy = np.matmul(Mij_Reshaped_m3,Mij_Reshaped_m2_T)[0][0]
    fx = np.sqrt((np.matmul(Mij_Reshaped_m1,Mij_Reshaped_m1_T)) - (cx*cx))[0][0]
    fy = np.sqrt((np.matmul(Mij_Reshaped_m2,Mij_Reshaped_m2_T)) - (cy*cy))[0][0]
    
    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)
    
    # Your implementation
    M_matrix = create_MijMatrix(img_coord,world_coord)
    u, s, vt = np.linalg.svd(M_matrix, full_matrices=False)
    Mij_11 = []
    Mij_11 = vt[11]
    Mij_Reshaped = np.array(Mij_11).reshape((3,4))
    
    #Normalising the matrix
    rv = Mij_Reshaped[2,:3]
    sc = np.linalg.norm(rv)
    lamda = 1/sc
    Mij_Reshaped = lamda * Mij_Reshaped
    
    #Retrieving the rows of matrix M
    Mij_Reshaped_R1 = [Mij_Reshaped[0][0],Mij_Reshaped[0][1],Mij_Reshaped[0][2]]
    Mij_Reshaped_R2 = [Mij_Reshaped[1][0],Mij_Reshaped[1][1],Mij_Reshaped[1][2]]
    Mij_Reshaped_R3 = [Mij_Reshaped[2][0],Mij_Reshaped[2][1],Mij_Reshaped[2][2]]
    
    #Reshaping the Matrix
    Mij_Reshaped_m1 = np.array(Mij_Reshaped_R1).reshape((1,3))
    Mij_Reshaped_m2 = np.array(Mij_Reshaped_R2).reshape((1,3))
    Mij_Reshaped_m3 = np.array(Mij_Reshaped_R3).reshape((1,3))
    
    #Transpose of M
    Mij_Reshaped_m1_T = Mij_Reshaped_m1.transpose()
    Mij_Reshaped_m2_T = Mij_Reshaped_m2.transpose()
    Mij_Reshaped_m3_T = Mij_Reshaped_m3.transpose()

    #Calculating Intrinsic
    cx = np.matmul(Mij_Reshaped_m3,Mij_Reshaped_m1_T)[0][0]
    cy = np.matmul(Mij_Reshaped_m3,Mij_Reshaped_m2_T)[0][0]
    fx = np.sqrt((np.matmul(Mij_Reshaped_m1,Mij_Reshaped_m1_T)) - (cx*cx))[0][0]
    fy = np.sqrt((np.matmul(Mij_Reshaped_m2,Mij_Reshaped_m2_T)) - (cy*cy))[0][0]
    
    IM = [fx,0,cx,0,fy,cy,0,0,1]
    IM_Reshape = np.array(IM).reshape((3,3))
    EM =np.matmul(np.linalg.inv(IM_Reshape),Mij_Reshaped)
    
    R1 = [EM[0][0], EM[0][1], EM[0][2]]
    R2 = [EM[1][0], EM[1][1], EM[1][2]]
    R3 = [EM[2][0], EM[2][1], EM[2][2]]
    T_1 = [EM[0][3],EM[1][3],EM[2][3]]
    T = np.array(T_1).reshape((3,1))
    R = np.array([R1,R2,R3]).reshape((3,3))

    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2

def create_MijMatrix(img_coord: np.ndarray, world_coord: np.ndarray) -> np.ndarray:
    Mij_Matrix = np.empty([64, 12])
    M_append_index = 0
    Low =0
    High=32
    for i in range(Low, High):
        X_wC = world_coord[i][0]
        Y_wC = world_coord[i][1]
        Z_wC = world_coord[i][2]
        x_iC = img_coord[i][0]
        y_iC = img_coord[i][1]
        Mij_Matrix[M_append_index] = [X_wC, Y_wC, Z_wC, 1, 0, 0, 0, 0, (-x_iC * X_wC), (-x_iC * Y_wC), (-x_iC * Z_wC), -x_iC]
        Mij_Matrix[M_append_index + 1] = [0, 0, 0, 0, X_wC, Y_wC, Z_wC, 1, (-y_iC * X_wC), (-y_iC * Y_wC), (-y_iC * Z_wC), -y_iC]
        M_append_index = M_append_index+2
    return Mij_Matrix




#---------------------------------------------------------------------------------------------------------------------
