import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, BboxOverlaps2D, filter_invalid
import pdb

from mmcv.runner import get_dist_info

from collections import Counter

@DETECTORS.register_module()
class SoftTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight
        self.how_file = open('file_write.txt', 'w')
        self.bbox_overlaps_2d = BboxOverlaps2D()
        
    # train
    def forward_gmm(self, img, img_metas, **kwargs):
        return super().forward_gmm(img, img_metas, **kwargs)


    # train
    def forward_train(self, img, img_metas, **kwargs):
        rank, _ = get_dist_info()
        super().forward_train(img, img_metas, **kwargs)

        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        
        if 'rescale' in kwargs.keys():  
            kwargs.pop('rescale')       # TODO: rescale 앞에서 다시 살려주기!
        
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        # 여기서 gmm 뽑으면 -100 나올 수 있음. 무조건 bbox 안의 get_target_single로 들어가서 확인할 것!       
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            gmm_labels = data_groups["sup"]["gmm_labels"]
            GMM_GT_idx = data_groups["sup"]["GMM_GT_idx"]
            n_clf = data_groups["sup"]["n_clf"]
            n_loc = data_groups["sup"]["n_loc"] # True = 1. False = 0 -> noise - 1. clean - 0

            assert len(GMM_GT_idx) == len(n_loc)            
            # # TEMP
            # assert len(gt_bboxes) == len(n_loc)
            # assert len(gt_bboxes) == len(n_clf)


            '''
            # 1. GT sup - real GT C/N
            GT_sup_list = [0, 0, 0, 0]   # CC CN NC NN

            for noise_cls, noise_reg in zip(n_clf, n_loc):
                if len(noise_cls) > 0:
                    for n_cls, n_reg in zip(noise_cls, noise_reg):
                        ids = int(str(n_cls.item())+str(n_reg.item()), 2)    # 0부터 차례대로 CC, CN, NC, NN  0 = FALSE = noise가 없다는 뜻 = CLEAN
                        GT_sup_list[ids] += 1
            '''
            
            GT_sup_list = [0, 0, 0, 0]   
            gmm_gt_match_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]    # num_XX는 다 더하기

            for gmm_gt_idx, noise_cls, noise_reg in zip(GMM_GT_idx, n_clf, n_loc):
                if len(noise_cls) > 0:
                    for gmm_gt, n_cls, n_reg in zip(gmm_gt_idx, noise_cls, noise_reg):
                        ids = int(str(n_cls.item())+str(n_reg.item()), 2)    # 0부터 차례대로 CC, CN, NC, NN  0 = FALSE = noise가 없다는 뜻 = CLEAN
                        GT_sup_list[ids] += 1       #  비율 넣을 때
                        
                        decimal_ids = int(str(n_cls.item())+str(n_reg.item()))
                        if gmm_gt.item() == decimal_ids:   # CC, CN, NC, NN 처리  - strict 비교 
                            gmm_gt_match_list[ids] += 1
                        elif '9' in str(gmm_gt.item()): # X가 들어가있음 - 비벼볼 수는 있음
                            quotient, remainder = gmm_gt.item() // 10, gmm_gt.item() % 10   # q, r 둘 중 최소 1개는 
                            if quotient <= 1:
                                gmm_gt_match_list[4+quotient] += 1
                            elif remainder <= 1:
                                gmm_gt_match_list[6+remainder] += 1 
                            else:                               
                                gmm_gt_match_list[-1] += 1
            # GT - CC
            log_every_n(
                {"GT_sup_CC": GT_sup_list[0] / len(n_loc)}
            )
            # GT - CN
            log_every_n(
                {"GT_sup_CN": GT_sup_list[1] / len(n_loc)}
            )
            # GT - NC
            log_every_n(
                {"GT_sup_NC": GT_sup_list[2] / len(n_loc)}
            )
            # GT - NN
            log_every_n(
                {"GT_sup_NN": GT_sup_list[3] / len(n_loc)}
            )

            '''                                             
            # 2. GMM GT - GMM GT C/N
            # GMM_GT_idx_list = [item.item() for sublist in GMM_GT_idx for item in sublist]   # [0, 1, 9, 10, 11, 19, 90, 91, 99]
            # n_clf_list = [abs(1-item.item()) for sublist in n_clf for item in sublist]   # [0, 1, 9, 10, 11, 19, 90, 91, 99]
            # n_loc_list = [abs(1-item.item()) for sublist in n_loc for item in sublist]   # [0, 1, 9, 10, 11, 19, 90, 91, 99]
            # flattened_GMM_GT_idx_list = Counter(GMM_GT_idx_list) 
            '''
            
            # GT - CC -> 얘네 비율로 말고 accuracy로 다 바꾸기
            log_every_n(
                {"GMM_GT_CC": gmm_gt_match_list[0] / len(n_loc)}
            )
            # GT - CN
            log_every_n(
                {"GMM_GT_CN": gmm_gt_match_list[1] / len(n_loc)}
            )
            # GT - NC
            log_every_n(
                {"GMM_GT_NC": gmm_gt_match_list[2] / len(n_loc)}
            )
            # GT - NN
            log_every_n(
                {"GMM_GT_NN": gmm_gt_match_list[3] / len(n_loc)}
            )
            # GT - CX
            log_every_n(
                {"GMM_GT_CX": (gmm_gt_match_list[4] + gmm_gt_match_list[0] + gmm_gt_match_list[1]) / len(n_loc)}
            )
            # GT - NX
            log_every_n(
                {"GMM_GT_NX": (gmm_gt_match_list[5] + gmm_gt_match_list[2] + gmm_gt_match_list[3]) / len(n_loc)}
            )
            # GT - XC
            log_every_n(
                {"GMM_GT_XC": (gmm_gt_match_list[6] + gmm_gt_match_list[0] + gmm_gt_match_list[2]) / len(n_loc)}
            )
            # GT - XN
            log_every_n(
                {"GMM_GT_XN": (gmm_gt_match_list[7] + gmm_gt_match_list[1] + gmm_gt_match_list[3]) / len(n_loc)}
            )
            # GT - XX
            log_every_n(
                {"GMM_GT_XX": gmm_gt_match_list[8] / len(n_loc)}
            )


            sup_list = [0, 0, 0, 0]     # CC CN NC NN
            
            for gmm in gmm_labels:      
                if len(gmm) > 0:          
                    max_vals, max_ids = torch.max(gmm, dim=-1)              
                    pos_gt_ids = torch.where(max_vals >= 0.8)[0]    # 아하.. 어차피 여기서 -100인 애들은 무조건 걸러진다...!!
                
                    if len(pos_gt_ids) > 0:  
                        for ids in pos_gt_ids.tolist():
                            sup_list[max_ids[ids]] += 1


            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            
            log_every_n(
                {"sup_CC": sup_list[0] / len(gmm_labels)}
            )
            # GMM - CN
            log_every_n(
                {"sup_CN": sup_list[1] / len(gmm_labels)}
            )
            # GMM - NC
            log_every_n(
                {"sup_NC": sup_list[2] / len(gmm_labels)}
            )
            # GMM - NN
            log_every_n(
                {"sup_NN": sup_list[3] / len(gmm_labels)}
            )
            # 여기서 아예 gt bbox의 label과 기타 등등이 안들어가게 하는 방법은..? 
            sup_loss = self.student.forward_train(**data_groups["sup"], unsup=False)   # mmdet.two_stage.forward_train.py  # 여기에 gmm label 반영하기 - 앞에 4개만 들어가도록
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            # TODO - data_groups 둘다에서 gmm_label 제대로 들어가는지 확인->  file에 저장
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"],
                ),  # 여기 들어가는 gmm label이 -1이 아닌 다른 제대로 된 값인지 확인하기
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                img = teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                img_metas = [teacher_data["img_metas"][idx] for idx in tidx],
                proposals = [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                and (teacher_data["proposals"] is not None)
                else None,
                gmm_labels_list = teacher_data["gmm_labels"],
                gt_bboxes_list = teacher_data["gt_bboxes"],
                gt_label_list = teacher_data["gt_labels"],
            )   # 얘의 det_bboxes, det_labels 사수하기  # <- 이 이미지의 gt 정보는 load해오기 때문에 존재
        student_info = self.extract_student_info(**student_data)

        # (Pdb) student_info.keys()
        # dict_keys(['img', 'backbone_feature', 'rpn_out', 'img_metas', 'proposals', 'transform_matrix'])
        # (Pdb) teacher_info.keys()
        # dict_keys(['backbone_feature', 'proposals', 'det_bboxes', 'det_labels', 'transform_matrix', 'img_metas'])
        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)   # 여기는 rpn loss update하는 부분이라 일단 무시!
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        # import pdb 
        # pdb.set_trace() # rcnn cls, rcnn_reg 둘다 돌아가면서 확인하고 gmm_label이랑 bbox 잘 넘어가는지 확인
        # 여기서 해당 bbox들에 대한 gmm label을 가져와서, 그중에 threshold 넘는 애들만 반영하기
        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
                gmm_labels_list=teacher_info["gmm_labels"],
            )
        )
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
                gmm_labels_list=teacher_info["gmm_labels"],
            )
        )
        return loss

    def rpn_loss(
        self,
        rpn_out,
        pseudo_bboxes,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _, valid_idx_1, valid_idx_2 = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)

            # TODO - valid_indices에 gmm_labels를 넣기
            # 여기는 rpn딴이라 필요없는 듯.. 그리고 proposal list가 
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:, :4],
                bbox_tag="rpn_pseudo_label",
                scores=pseudo_bboxes[0][:, 4],
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        gmm_labels_list,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _, valid_idx_1, valid_idx_2 = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )

        gmm_labels = [gmm[idx_1][idx_2] for gmm, idx_1, idx_2 in zip(gmm_labels_list, valid_idx_1, valid_idx_2)]

        # 여기서 tau 계산해서 바로 걸러주는거 어떠셈 ㅇㅇ 너무좋아
        # max_vals, max_ids = torch.max(pos_gmm_labels, dim=-1)
        # tau_auto = (1 - max_vals) * 0.95 + max_vals * 0.5
        # pos_gt_ids_output = torch.where(max_vals >= tau_auto)
        # 

        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )   # 읭 여기에 gt bboxes, gt_labels 들어가는디..?
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn, gmm_labels=gmm_labels, unsup=True
        )   # train할 때도 여기서부터 gmm_label 들어감
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def unsup_rcnn_reg_loss(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        gmm_labels_list,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _, valid_idx_1, valid_idx_2 = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        gmm_labels = [gmm[idx_1][idx_2] for gmm, idx_1, idx_2 in zip(gmm_labels_list, valid_idx_1, valid_idx_2)]

        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, gmm_labels=gmm_labels, unsup=True, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}

    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        # import pdb 
        # pdb.set_trace() # assign과 sample 알아낸 뒤 거기에 gmm_labels 녹여넣기
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i],
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def extract_teacher_info(self, img, img_metas, gmm_labels_list, gt_bboxes_list, gt_label_list, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )   # {'nms_pre': 1000, 'max_per_img': 1000, 'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'min_bbox_size': 0}
            rpn_out = list(self.teacher.rpn_head(feat)) #  mmdet/models/dense_heads/anchor_head.py(152)forward()
            proposal_list = self.teacher.rpn_head.get_bboxes(   # <bound method BaseDenseHead.get_bboxes of RPNHead(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list

        # 여기 simple_test_bboxes도 봐보기. 이거가 teacher의 최종 output    # GT 넣고 ROI 굴리지 말고 가져오는 부분 반영하기
        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )   # <- 여기에서 rpn이 엄청 많았는데, simple_test_bboxes를 거치면서 소수의 갯수로 줄어듦. 
            # 이때 줄어들게 하는 기준이 뭔지, 그리고 여기 proposal_list에 gt bbox를 넣으면 어떻게 되는지..?

        # proposal_list 원본 저장해두기
        pseudo_bbox_list = []
        pseudo_label_list = []
        pseudo_gmm_labels_list = []        
        for idx, (prop_box, gt_box) in enumerate(zip(proposal_list, gt_bboxes_list)):
            max_overlaps, argmax_overlaps, thres_ids = self.bbox_overlaps_2d(prop_box, gt_box)   # suppose bboxes1 [M, 4], bboxes2 [N, 4]
            # assign result에 GT까지도 반영됨
            pseudo_bbox_list.append(proposal_list[idx][argmax_overlaps][thres_ids])
            pseudo_label_list.append(proposal_label_list[idx][argmax_overlaps][thres_ids])
            pseudo_gmm_labels_list.append(gmm_labels_list[idx][thres_ids])


        proposal_list = [p.to(feat[0].device) for p in pseudo_bbox_list]
        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in pseudo_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")

        proposal_list, proposal_label_list, _ , valid_idx_list_1, valid_idx_list_2 = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        # TODO - valid_idx를 teacher gmm_info에 넣기
        # pdb.set_trace()
        # gmm label 갯수가 original proposal 갯수보다 적음
        gmm_labels = [gmm[val_idx_1][val_idx_2] for gmm, val_idx_1, val_idx_2 in zip(pseudo_gmm_labels_list, valid_idx_list_1, valid_idx_list_2)]
        det_bboxes = proposal_list
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["gmm_labels"] = gmm_labels
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
