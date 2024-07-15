from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
from pyVHR.utils.errors import getErrors, printErrors, displayErrors
from pyVHR.analysis.stats import StatAnalysis
from pyVHR.extraction.sig_processing import SignalProcessing
from pyVHR.extraction.utils import *
import os



# params
wsize = 6                  # window size in seconds
roi_approach = 'patches'   # 'holistic' or 'patches'
roi_method='hsv'    # 'convexhull' or 'faceparsing' or 'hsv'
bpm_est = 'clustering'     # BPM final estimate, if patches choose 'medians' or 'clustering'
method = 'cpu_CHROM'       # one of the methods implemented in pyVHR
landmarks_list = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]

# run
pipe = Pipeline()          # object to execute the pipeline
sig_proc = SignalProcessing() # object to process the signal
videoFileName = '/home/luying/pyVHR/pyVHR/videos/meface.mp4'
# videoFileName = '/home/luying/EVM/PyEVM-master/src/lying2-deidentified.mp4'
fps = get_fps(videoFileName=videoFileName)

# res = pipe.run_on_dataset('/home/luying/pyVHR/results/cfg/ECG_Fitness_01-1_holistic.cfg', verb=True)

# sig_proc.set_landmarks(landmarks_list)
# sig_proc.set_square_patches_side(28.0)
# sig = sig_proc.extract_patches(videoFileName, 'square', 'mean')

# bpmGT = res.getData['bpmGT']
# timesGT = res.getData['timeGT']

bvps, timesES, bpmES = pipe.run_on_video(videoFileName,
                                        winsize=wsize, 
                                        ldmks_list=landmarks_list,
                                        cuda=True,
                                        roi_method=roi_method,
                                        roi_approach=roi_approach,
                                        method=method,
                                        estimate=bpm_est,
                                        patch_size=30, 
                                        RGB_LOW_HIGH_TH=(75,230),
                                        Skin_LOW_HIGH_TH=(75,230),
                                        pre_filt=True,
                                        post_filt=True,
                                        verb=True)

print("bvps: ", bvps)
print("timesES: ", timesES)
print("bpmES: ", bpmES)
print("finished")

# ERRORS
RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)
displayErrors(bpmES, bpmGT, timesES, timesGT)

