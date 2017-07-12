import matlab.engine as matlab
import sys
sys.path.append('/home/a/Downloads/myproject/camera_and_pose/release/python/')
import os 


def getPose(input_img, pose2d):
    pose2d_list = pose2d.tolist()
    eng = matlab.start_matlab()
    print 'aaaaaaaaaaaaa'
    X,R,t=eng.triarea(input_img, pose2d_list)
    return X,R,t