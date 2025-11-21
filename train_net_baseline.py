import os
import sys
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.data.datasets import register_coco_instances
from sparseinst import add_sparse_inst_config

# Register Datasets first
register_coco_instances("deepcrack_train", {}, "dataset/annotations/instances_train.json", "dataset/train_img")
register_coco_instances("deepcrack_test", {}, "dataset/annotations/instances_test.json", "dataset/test_img")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main():
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    cfg = setup(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    main()
