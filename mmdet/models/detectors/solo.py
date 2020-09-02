from .single_stage_ins import SingleStageInsDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class SOLO(SingleStageInsDetector):

    def __init__(self,
                 backbone, #resnet
                 neck, #FPN
                 bbox_head, #solo head
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SOLO, self).__init__(backbone, neck, bbox_head, None, train_cfg,
                                   test_cfg, pretrained)
