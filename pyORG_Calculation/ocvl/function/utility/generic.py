import os
from enum import Enum

import cv2
import numpy as np
import pandas as pd

from ocvl.function.preprocessing.improc import optimizer_stack_align
from ocvl.function.utility.resources import load_video, save_video


class PipeStages(Enum):
    RAW = 0,
    PROCESSED = 1,
    PIPELINED = 2,
    ANALYSIS_READY = 3

class GenericDataset:
    def __init__(self, video_path="", framestamp_path=None, coord_path=None, stimtrain_path=None, stage=PipeStages.RAW):

        # Paths to the data used here.
        self.video_path = video_path
        if framestamp_path is None:
            self.framestamp_path = self.video_path[0:-3] + "csv"
        else:
            self.framestamp_path = framestamp_path

        if coord_path is None:
            p_name = os.path.dirname(os.path.realpath(self.video_path))
            f_name = os.path.basename(os.path.realpath(self.video_path))
            self.coord_path = os.path.join(p_name, f_name[0:-4] + "_coords.csv")
        else:
            self.coord_path = coord_path

        # Information about the dataset
        self.stage = stage
        self.framerate = -1
        self.num_frames = -1
        self.width = -1
        self.height = -1
        self.framestamps = np.empty([1])
        self.reference_frame_idx = []
        self.stimtrain_frame_stamps = np.empty([1])

        # The data are roughly grouped by the following:
        # Base data
        self.coord_data = np.empty([1])
        self.reference_im = np.empty([1])
        self.metadata_data = np.empty([1])
        # Video data (processed or pipelined)
        self.video_data = np.empty([1])
        # Extracted data (temporal profiles
        self.raw_profile_data = np.empty([1])
        self.postproc_profile_data = np.empty([1])

    def clear_video_data(self):
        print("Deleting video data from "+self.video_path)
        del self.video_data

    def load_data(self):
        if self.stage is PipeStages.RAW:
            self.load_raw_data()
        elif self.stage is PipeStages.PROCESSED:
            self.load_processed_data()
        elif self.stage is PipeStages.PIPELINED:
            self.load_pipelined_data()
        elif self.stage is PipeStages.ANALYSIS_READY:
            self.load_analysis_ready_data()

    def load_raw_data(self):
        resource = load_video(self.video_path)

        self.video_data = resource.data

        self.framerate = resource.metadict["framerate"]
        self.metadata_data = resource.metadict
        self.width = resource.data.shape[1]
        self.height = resource.data.shape[0]
        self.num_frames = resource.data.shape[-1]

        if self.coord_path:
            self.coord_data = pd.read_csv(self.coord_path, delimiter=',', header=None,
                                          encoding="utf-8-sig").to_numpy()

    def load_pipelined_data(self):
        if self.stage is PipeStages.PIPELINED:
            resource = load_video(self.video_path)

            self.video_data = resource.data

            self.framerate = resource.metadict["framerate"]
            self.metadata_data = resource.metadict
            self.width = resource.data.shape[1]
            self.height = resource.data.shape[0]
            self.num_frames = resource.data.shape[-1]

            if self.coord_path:
                self.coord_data = pd.read_csv(self.coord_path, delimiter=',', header=None,
                                              encoding="utf-8-sig").to_numpy()
            if self.framestamp_path:
                # Load our text data.
                self.framestamps = pd.read_csv(self.framestamp_path, delimiter=',', header=None,
                                               encoding="utf-8-sig").to_numpy()

            if self.stimtrain_path:
                self.stimtrain_frame_stamps = np.cumsum(np.squeeze(pd.read_csv(self.stimtrain_path, delimiter=',', header=None,
                                                          encoding="utf-8-sig").to_numpy()))
            else:
                self.stimtrain_frame_stamps = self.num_frames-1

    def load_processed_data(self, force=False):
        # Establish our unpipelined filenames
        if self.stage is not PipeStages.RAW or force:
            resource = load_video(self.video_path)

            self.video_data = resource.data

            self.framerate = resource.metadict["framerate"]
            self.metadata_data = resource.metadict
            self.width = resource.data.shape[1]
            self.height = resource.data.shape[0]
            self.num_frames = resource.data.shape[-1]

            self.video_data, xforms, inliers = optimizer_stack_align(self.video_data,
                                                                         reference_idx=self.reference_frame_idx,
                                                                         dropthresh=0.0)

            print( "Keeping " +str(np.sum(inliers))+ " of " +str(self.num_frames)+"...")

            # Update everything with what's an inlier now.
            self.ref_video_data = self.ref_video_data[..., inliers]
            self.framestamps = self.framestamps[inliers]
            self.video_data = self.video_data[..., inliers]
            self.mask_data = self.mask_data[..., inliers]

            (rows, cols) = self.video_data.shape[0:2]

            for f in range(self.num_frames):
                if xforms[f] is not None:
                    self.video_data[..., f] = cv2.warpAffine(self.video_data[..., f], xforms[f],
                                                             (cols, rows),
                                                             flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP)
                    self.mask_data[..., f] = cv2.warpAffine(self.mask_data[..., f], xforms[f],
                                                            (cols, rows),
                                                            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)

            self.num_frames = self.video_data.shape[-1]

    def save_data(self, suffix):
        save_video(self.video_path[0:-4]+suffix+".avi", self.video_data, self.framerate)


