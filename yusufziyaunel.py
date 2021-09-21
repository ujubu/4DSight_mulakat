import numpy as np
import cv2
import glob

image_points = np.load('vr2d.npy')
# image_points = np.array([
#                         (1358, 534), 
#                         (1358, 534),
#                         (1322, 534),
#                         (1322, 534),
#                         (578, 592),
#                         (578, 592),
#                         (696, 636),
#                         (696, 636),
#                         (1396, 154),
#                         (1332, 632),
#                         (1332, 632),
#                         (1084, 168),
#                         (860, 418),
#                         (860, 418),
#                         (1182, 380),
#                         (1182, 380),
#                         (1336, 316),
#                         (836, 616),
#                         (836, 616),
#                         (940, 612)
#                         ], dtype="double")

image_points = np.ascontiguousarray(image_points[:,:2]).reshape((-1, 1, 2))


model_points = np.load('vr3d.npy')
# model_points = np.array([
#                           (4.07699, -10.0051, 2.42278),
#                           (4.0786, -10.0016, 2.42466),
#                           (4.24954, -9.62041, 2.41662),
#                           (4.25343, -9.60823, 2.40936),
#                           (2.57967, -0.973489, 1.8305),
#                           (2.57579, -0.980851, 1.82587),
#                           (1.9782, -2.44128, 1.47028),
#                           (2.01343, -2.44332, 1.46808),
#                           (-0.428718, -7.9005, 4.98679),
#                           (2.45441, -8.90456, 1.44507),
#                           (2.32265, -8.86605, 1.46896),
#                           (6.54949, -6.90402, 8.06735),
#                           (8.26417, -3.20839, 4.44154),
#                           (8.27652, -3.19099, 4.43522),
#                           (5.92752, -8.28244, 4.66252),
#                           (5.92391, -8.27562, 4.6846),
#                           (4.01438, -9.63951, 5.10035),
#                           (2.19789, -3.79375, 1.62818),
#                           (2.20158, -3.79428, 1.63073),
#                           (4.15877, -4.77812, 1.44965)
#                           ], dtype=np.float32)



criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
frameSize = (960, 540)

dist = np.zeros((4,1)) # dist = 0
focal_length = 100 
center = (frameSize[0]/2, frameSize[1]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0,  focal_length, center[1]],
                         [0,             0,        1]], 
                         dtype = "double")


def drawBoxes(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


for fname in glob.glob('img*.png'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    # corners = np.int0(corners)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    
    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv2.solvePnP(model_points, image_points, camera_matrix, dist)  
    # project 3D spoints to image plane
    imgpts, jac = cv2.projectPoints(model_points, rvecs, tvecs, camera_matrix, dist)
    
    img = drawBoxes(img, corners2, imgpts)
    cv2.imshow('result', img)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'):
        cv2.imwrite(fname[:4] + '_changed.png', img)
      
        
cv2.destroyAllWindows()







