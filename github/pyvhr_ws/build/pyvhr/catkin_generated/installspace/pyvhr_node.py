#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge, CvBridgeError
import sys
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import math

# from PIL import Image
# from tqdm import tqdm
# from glob import glob
import torch
from torchvision.transforms import Normalize, ToTensor, Compose
# from cherenkov_utils import *

# Add your lib path
# sys.path.append('/home/luying')
# from lib.utils.visualizer import draw_boxes, draw_kpts
# from lib.yolo import Yolov7
#from lib.sam import SAM
#from lib.utils.imutils import crop, boxes_2_cs

# Add your pyVHR path
sys.path.append('/home/luying/pyVHR')
from pyVHR.analysis.pipeline import Pipeline, DeepPipeline
from pyVHR.extraction.sig_processing import SignalProcessing
from pyVHR.extraction.skin_extraction_methods import SkinExtractionHSV

class pyVHR:
    def __init__(self):
        rospy.init_node('pyvhr_node', anonymous=True)
        # print("pyVHR Node Initialized")
        rospy.Subscriber('/image/masked', Image, self.image_callback)
        self.hr_pub = rospy.Publisher('hr', Float32, queue_size=10)
        self.frames = []
        self.bridge = CvBridge()
        self.pipe = Pipeline
        self.sig_processing = SignalProcessing()
        self.skin_extraction = SkinExtractionHSV()
        rospy.spin()
        
    def decode_img_msg(self, msg: CompressedImage) -> np.ndarray:
        # np_arr = np.fromstring(msg.data, np.uint8)
        # img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        return img


    def image_callback(self, msg):
        '''
        Takes in the image and estimates heart rate.
        '''
        try:
            # Decompress the image
            # cv_image = self.decode_img_msg(msg)
            # print("Received Image")
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            data = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # apply hsv filter here, data is the cropped image
            cropped_skin_im, full_skin_im = self.skin_extraction.extract_skin(data, ldmks=None)
            # print(cropped_skin_im.shape)
            # append 300 frames
            if len(self.frames) < 250:
                self.frames.append(cropped_skin_im)

            if len(self.frames) == 250:               
                # Estimate heart rate (input single frame or list of frames?)
                #cropped_frames = self.person_cropping(self.frames)
                # self.frames = np.array(self.frames)
                # print(len(self.frames))
                # print(len(self.frames[0]))
                hr = self.pyvhr_est_dl(self.frames)
                rospy.loginfo("Heart Rate: %f" % hr)

                # Publish heart rate
                self.hr_pub.publish(Float32(hr))
                self.frames = []

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s" % e)
        except Exception as e:
            rospy.logerr("Error processing image: %s" % str(e))

    def pyvhr_est(self, frames):
        '''
        Processes the frames and estimates the heart rate.
        '''
        wsize = 6                # Window size in seconds
        roi_approach = 'holistic' # 'holistic' or 'patches'
        roi_method = 'hsv'        # 'convexhull' or 'faceparsing' or 'hsv'
        bpm_est = 'clustering'    # BPM final estimate, if patches choose 'median' or 'clustering'
        method = 'cupy_CHROM'      # One of the methods implemented in pyVHR

        # Run the pyVHR pipeline
        fps = 30  # Frames per second, adjust if neede
        bvps, timesES, bpmES = self.pipe.run_on_video(self, frames,
                                             winsize=wsize,
                                             ldmks_list=None,  # Set this if you have landmarks
                                             cuda=True,
                                             roi_method=roi_method,
                                             roi_approach=roi_approach,
                                             method=method,
                                             estimate=bpm_est,
                                             patch_size=150,
                                             RGB_LOW_HIGH_TH=(75,230),
                                             Skin_LOW_HIGH_TH=(75,230),
                                             pre_filt=True,
                                             post_filt=True,
                                             verb=True)

        # Filter and average BPM results
        bpm_list = [float(bpm) for bpm in bpmES]
        filtered_bpm = [bpm for bpm in bpm_list if 50 <= bpm <= 120]
        mean_bpm = np.mean(filtered_bpm)
        median_bpm = np.median(filtered_bpm)
        print(f"bpm_list: {filtered_bpm}, Mean BPM: {mean_bpm}, Median BPM: {median_bpm}")

        if np.isnan(mean_bpm):
            return 80
        else:
            return int(mean_bpm)
    
    def pyvhr_est_dl(self, frames):
        '''
        Processes the frames and estimates the heart rate using deep learning model.
        method: HR_CNN or MTTS_CAN
        '''
        pipe = DeepPipeline()
        timesES, bpmES = pipe.run_on_video(frames,cuda=True, method='MTTS_CAN', bpm_type='welch', post_filt=False, verb=True, crop_face=False)
        # Filter and average BPM results
        bpm_list = [float(bpm) for bpm in bpmES]
        filtered_bpm = [bpm for bpm in bpm_list if 50 <= bpm <= 130]
        mean_bpm = np.mean(filtered_bpm)
        median_bpm = np.median(filtered_bpm)

        return mean_bpm

    def person_cropping(self, frames):
        normalize_img = Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        device = 'cuda'

        # initialize yolov7 model adn SAM
        yolo = Yolov7(device=device, weights='/home/luying/lib/yolov7.pt', imgsz=640)
        sam = SAM(device=device, checkpoint="/mnt/dtc/perception_models/pretrain/sam_vit_h_4b8939.pth")

        cropped_frames = []

        for frame in frames:

            boxes = yolo(frame, conf=0.20)
            boxes = boxes.cpu()

            if len(boxes) == 0:
                print("No person detected")
                continue
            else:
                boxes_array = boxes.numpy()
                x1 = boxes_array[0, 0].item()
                y1 = boxes_array[0, 1].item()
                x2 = boxes_array[0, 2].item()
                y2 = boxes_array[0, 3].item()
                if x2 - x1 < 100 or y2 - y1 < 100:
                    print("No person detected")
                    continue

            centers, scales = boxes_2_cs(boxes)

            # calculate the mask
            mask = sam(img, boxes)
            # img_boxes = draw_boxes(img, boxes)

            mask_boolean = (mask==0)
            mask_index_of_zero = torch.nonzero(mask_boolean)
            list_of_index_of_zero = []
            for idx in mask_index_of_zero:
                t = tuple(idx.tolist())
                list_of_index_of_zero.append(t)

            mask_image = img.copy()
            for (y, x) in list_of_index_of_zero:
                mask_image[x, y] = (0, 0, 0)
            k = 0 # choose which human (goes up to len(boxes)-1)
            mask_image = crop(mask_image, centers[k], scales[k], [256, 256], rot=0).astype('uint8')

            cropped_frames.append(mask_image)

        return cropped_frames

if __name__ == '__main__':
    try:
        pyVHR()
    except rospy.ROSInterruptException:
        pass
