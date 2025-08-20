import importlib
import cv2
import argparse
import numpy as np
# from datasets import load_dataset
import os
from PIL import Image

def track_and_visualize(image1: np.ndarray, image2: np.ndarray, init_bbox: list, init_text_description: str, output_dir: str = "./demo_results"):

    # breakpoint()

    try:
        if isinstance(image1, Image.Image):  
            image1 = np.array(image1)
        if isinstance(image2, Image.Image):
            image2 = np.array(image2)

        parameter_name = "dutrack_256_full"  
        param_module = importlib.import_module(f'lib.test.parameter.dutrack')
        params = param_module.parameters(parameter_name, None)
        params.debug = False

        tracker_class = importlib.import_module(f'lib.test.tracker.dutrack').get_tracker_class()
        # breakpoint()
        tracker = tracker_class(params)

        init_info = {
            'init_bbox': init_bbox,
            'init_text_description': init_text_description
        }

        out = tracker.initialize(image1, init_info)

        info = {}
        out = tracker.track(image2, info)

        return True, out

    except Exception as e:
        breakpoint()
        print(f"Error during tracking and visualization: {str(e)}")
        return False, None

def batch_track_and_visualize(img1s: list, img2s: list, init_bboxes: list, init_text_descriptions: list):

    try:

        parameter_name = "dutrack_256_full" 
        param_module = importlib.import_module(f'lib.test.parameter.dutrack')
        params = param_module.parameters(parameter_name, None)
        params.debug = False

        tracker_class = importlib.import_module(f'lib.test.tracker.dutrack').get_tracker_class()
        tracker = tracker_class(params)

        results = []
        for image1, image2, init_bbox, init_text_description in zip(img1s, img2s, init_bboxes, init_text_descriptions):
            if isinstance(image1, Image.Image):  
                if image1.mode == 'RGBA':
                    image1 = image1.convert('RGB')
                    image2 = image2.convert('RGB')
                if image1.mode == 'L':
                    image1 = image1.convert('RGB')
                    image2 = image2.convert('RGB')
                image1 = np.array(image1)
                if len(image1.shape) == 2:
                    breakpoint()
            if isinstance(image2, Image.Image):
                image2 = np.array(image2)


            init_info = {
                'init_bbox': init_bbox,
                'init_text_description': init_text_description
            }

            out = tracker.initialize(image1, init_info)


            info = {}
            out = tracker.track(image2, info)

            results.append(out)

        return True, results

    except Exception as e:
        # breakpoint()
        print(f"Error during batch tracking and visualization: {str(e)}")
        return False, None
    
