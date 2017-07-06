import matlab.engine
import sys 
sys.path.append('./camera_and_pose/release/python/')

def getPose(input_img,pose2d):
    pose2d_list=pose2d.tolist()
    eng = matlab.engine.start_matlab()
    print 'pose2d_list'
    print pose2d_list[1]

    eng.triarea(input_img,pose2d_list[1])
