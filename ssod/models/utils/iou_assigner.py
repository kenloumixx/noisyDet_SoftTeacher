import torch
from mmdet.core import AssignResult


def assign_wrt_overlaps_ssod(overlaps, gt_labels=None):
    """Assign w.r.t. the overlaps of bboxes with gts.

    Args:
        overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
        gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

    Returns:
        :obj:`AssignResult`: The assign result.
    """
    num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

    if num_gts == 0 or num_bboxes == 0:
        return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)

    # for each anchor, which gt best overlaps with it
    # for each anchor, the max iou of all gts
    max_overlaps, argmax_overlaps = overlaps.max(dim=0) # 1000개의 pred box가 각각 어떤 gt와 overlap 되는지
    # for each gt, which anchor best overlaps with it   # ex. max(max_overlaps) = 0.8042    -> 1000개의 bbox가 각각 어떤 gt와 얼마만큼의 max iou로 결합됐는지
    # for each gt, the max iou of all proposals
    gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)  # 마찬가지, 몇번째 prediction이 가장 나은지
    thres_ids = torch.where(max_overlaps >= 0.5)[0]
    max_ids = argmax_overlaps[thres_ids]
    # iou 안넘으면 GT 자격 없음 
    return max_overlaps, argmax_overlaps, thres_ids

    neg_iou_thr = 0.5
    pos_iou_thr = 0.5

    # 2. assign negative: below
    # the negative inds are set to be 0
    assigned_gt_inds[(max_overlaps >= 0)
                        & (max_overlaps < neg_iou_thr)] = 0

    # 3. assign positive: above positive IoU threshold
    pos_inds = max_overlaps >= pos_iou_thr
    assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

    if gt_labels is not None:
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1]
    else:
        assigned_labels = None

    # TODO 여기서는 총 3개의 gt bbox에 대해 5개의 box가 매핑됨
    return AssignResult(
        num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
