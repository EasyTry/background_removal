# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend

class BackgroundRemoval:
    def __init__(self):

        cfg = get_cfg()
        # Add PointRend-specific config
        point_rend.add_pointrend_config(cfg)
        # Load a config from file
        current_dir = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(current_dir, "pointrend_rcnn_R_50_FPN_3x_coco.yaml")
        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
        self.predictor = DefaultPredictor(cfg)
        
    def __call__(self, image):
      outputs = self.predictor(image)
      mask = outputs["instances"].to("cpu").get('pred_masks')[0, :].numpy()
      return mask
