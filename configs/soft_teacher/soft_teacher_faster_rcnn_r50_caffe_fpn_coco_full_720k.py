_base_="base.py"

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(

        sup=dict(

            # ann_file="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/annotations/instances_train2017.json",
            # img_prefix="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/train2017",
            # ann_file="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/annotations/instances_val2017.json",
            # img_prefix="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/val2017",

            ann_file="/home/cvlab11/project/noisyDet/data/coco/annotations/newmixnoisy10_instances_train2017.json",
            img_prefix="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/train2017/",

        ),
        unsup=dict(
            # mira edit
            # ann_file="data/coco/annotations/instances_train2017.json",
            # ann_file="/home/cvlab11/project/noisyDet/data/coco/annotations/newmixnoisy40_instances_train2017.json",
            # img_prefix="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/train2017/",
            # ann_file="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/annotations/instances_val2017.json",
            # img_prefix="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/val2017",
            ann_file="/home/cvlab11/project/noisyDet/data/coco/annotations/newmixnoisy10_instances_train2017.json",
            img_prefix="/home/cvlab11/project/noisyDet/tmptest/mmdetection/data/coco/train2017/",

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],    # mira: sampler ratio 반으로 바꿈
        ),
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[12000 * 4, 16000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=18000 * 4)
