#Based on open3d examples/Python/ReconstructionSystem/sensors/realsense_recorder.py

import numpy as np
import cv2
from cv2 import aruco
import argparse
from os import makedirs
from os.path import exists, join
import shutil
import json
from enum import IntEnum
from CharucoBoardToSVG.arucoUtililties import toDict
import math
import glob 

def calibrate(image_folder,output_file,squaresX = 6, squaresY = 10, squareLength = 0.025, markerLength =0.02, dictionary = 'DICT_4X4_250'):
    image_filenames = glob.glob(image_folder + '*.jpg')
    image_filenames.sort()
    if len(image_filenames) == 0:
        return None
    else:
        dictionary = toDict(dictionary)# toDict converts string to OpenCV Charuco Dictionary integer

        # The calibration part is a modified version of https://github.co# toDict converts string to OpenCV Charuco Dictionary integerm/kyle-elsalhi/opencv-examples/blob/master/CalibrationByCharucoBoard/CalibrateCamera.py
        # Create constants to be passed into OpenCV and Aruco methods
        CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=squaresX,
            squaresY=squaresY,
            squareLength=squareLength,
            markerLength=markerLength,
            dictionary=dictionary )

        # Create the arrays and variables we'll use to store info like corners and IDs from images processed
        corners_all = []  # Corners discovered in all images processed
        ids_all = []  # Aruco ids corresponding to corners discovered
        image_size = None  # Determined at runtime

        # Loop through images 
        nrImages=len(image_filenames)
        for ii,iname in enumerate(image_filenames):
            print("Detecting markers: {:07.3f}%".format(100.0*ii/(nrImages-1)),end='\r')  

            # Open the image
            img = cv2.imread(iname)
            # imgscale the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find aruco markers in the query image
            corners, ids, _ = aruco.detectMarkers(
                image=gray,
                dictionary=dictionary)

            # Outline the aruco markers found in our query image
            img = aruco.drawDetectedMarkers(
                image=img,
                corners=corners)

            
            # Get charuco corners and ids from detected aruco markers
            response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=CHARUCO_BOARD)

            # If a Charuco board was found, let's collect image/corner points
            # Requiring at least 6 squares
            if response > 0.50*(len(CHARUCO_BOARD.ids)):
                # Add these corners and ids to our calibration arrays
                corners_all.append(charuco_corners)
                ids_all.append(charuco_ids)

                # If our image size is unknown, set it now
                if not image_size:
                    image_size = gray.shape[::-1]

                # Reproportion the image, maxing width or height at 1000
                proportion = max(img.shape) / 1000.0
                img = cv2.resize(img, (int(img.shape[1] / proportion), int(img.shape[0] / proportion)))

            else:
                print("Not able to detect a charuco board in image: {}".format(iname))


        # Make sure at least one image was found
        if len(image_filenames) < 1:
            print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
            exit()

        # Make sure we were able to calibrate on at least one charucoboard by checking
        # if we ever determined the image size
        if not image_size:
            # Calibration failed because we didn't see any charucoboards of the PatternSize used
            print(
                "Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
            # Exit for failure
            exit()

        # Now that we've seen all of our images, perform the camera calibration
        # based on the set of points we've discovered
        print("\nCalculating camera parameters ...")
        calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=corners_all,
            charucoIds=ids_all,
            board=CHARUCO_BOARD,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None)

        # Write results to json file and terminal
        camCal = {'cameraMatrix':cameraMatrix.tolist(),'distCoeffs':distCoeffs.tolist()}
        with open(output_file, 'w') as outfile:
            json.dump(camCal, outfile)
        print("Wrote {} to {}".format(camCal,output_file))

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Generate camera calibration with given charucoBoard parameters")
    # parser.add_argument('squaresX',help="squaresX parameter used for OpenCv charuco board initialization")
    # parser.add_argument('squaresY',help="squaresY parameter used for OpenCv charuco board initialization")
    # parser.add_argument('squareLength',help="squareLength parameter used for OpenCv charuco board initialization")
    # parser.add_argument('markerLength',help="markerLength parameter used for OpenCv charuco board initialization")
    # parser.add_argument('dictionary',help="dictionary parameter used for OpenCv charuco board initialization")
    parser.add_argument("--camCalJson",nargs="?",default='./charucoBoard.json',help="charuco json file")
    parser.add_argument("--output_file",nargs="?",default='./camCal.json',help="output file")
    parser.add_argument("--input_folder",nargs="?",default='./imageLogger/color/',help="input folder with images")

    args = parser.parse_args()

    data = []
    with open(args.camCalJson) as json_file:
        data = json.load(json_file)
    
    calibrate(args.input_folder,args.output_file,int(data['squaresX']),int(data['squaresY']),float(data['squareLength']),float(data['markerLength']),data['dictionary'])


    
