import os.path as osp
import time 
import torch
import numpy as np 

import torch.distributed as dist
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, LoggerHook, WandbLoggerHook
from mmcv.runner.dist_utils import get_dist_info
from mmdet.core import DistEvalHook, GMMDistEvalHook 
from torch.nn.modules.batchnorm import _BatchNorm

from sklearn.mixture import GaussianMixture

import contextlib
import io
import logging
import os
import pickle
import time
import warnings
from datetime import timedelta
from typing import Callable, Dict, Optional, Tuple, Union, List

import torch.nn.functional as F


def all_gather(tensor: torch.Tensor, fixed_shape: Optional[List] = None) -> List[torch.Tensor]:
    def compute_padding(shape, new_shape):
        padding = []
        for dim, new_dim in zip(shape, new_shape):
            padding.insert(0, new_dim - dim)
            padding.insert(0, 0)
        return padding

    input_shape = tensor.shape
    if fixed_shape is not None:
        padding = compute_padding(tensor.shape, fixed_shape)
        if sum(padding) > 0:
            tensor = F.pad(tensor, pad=padding, mode='constant', value=0)
    
    output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)

    all_input_shapes = None
    if fixed_shape is not None:
        # gather all shapes
        tensor_shape = torch.tensor(input_shape, device=tensor.device)
        all_input_shapes = [torch.zeros_like(tensor_shape) for _ in range(dist.get_world_size())]
        dist.all_gather(all_input_shapes, tensor_shape)
        all_input_shapes = [t.tolist() for t in all_input_shapes]

    if all_input_shapes:
        for i, shape in enumerate(all_input_shapes):
            padding = compute_padding(output[i].shape, shape)
            if sum(padding) < 0:
                output[i] = F.pad(output[i], pad=padding)

    return output



def all_gather_object(object_list, obj, group=None):
    _pickler = pickle.Pickler
    _unpickler = pickle.Unpickler
    
    def _tensor_to_object(tensor, tensor_size):
        buf = tensor.numpy().tobytes()[:tensor_size]
        return _unpickler(io.BytesIO(buf)).load()

    def _object_to_tensor(obj):
        f = io.BytesIO()
        _pickler(f).dump(obj)
        byte_storage = torch.ByteStorage.from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will casue 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage)
        local_size = torch.LongTensor([byte_tensor.numel()])
        return byte_tensor, local_size


    input_tensor, local_size = _object_to_tensor(obj)
    current_device = torch.device("cuda", torch.cuda.current_device())
    input_tensor = input_tensor.to(current_device)
    local_size = local_size.to(current_device)

    group_size = torch.distributed.get_world_size()
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    group = torch.distributed.new_group(list(range(group_size)))
    torch.distributed.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device
    )

    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, input_tensor, group=group)

    k = 0
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()

        tensor_size = object_size_list[i]
        output = _tensor_to_object(tensor, tensor_size)
        for j in range(len(output)):
            object_list[k] = output[j]
            k += 1    

@HOOKS.register_module()
class SubModulesDistEvalHook(DistEvalHook):
    def __init__(self, *args, evaluated_modules=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluated_modules = evaluated_modules

    def before_run(self, runner):
        if is_module_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        assert hasattr(model, "submodules")
        assert hasattr(model, "inference_on")

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch and self._should_evaluate(runner):
            for hook in runner._hooks:
                if isinstance(hook, WandbLoggerHook):
                    _commit_state = hook.commit
                    hook.commit = False
                if isinstance(hook, LoggerHook):
                    hook.after_train_iter(runner)
                if isinstance(hook, WandbLoggerHook):
                    hook.commit = _commit_state
            runner.log_buffer.clear()

            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.

        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".eval_hook")

        if is_module_wrapper(runner.model):
            model_ref = runner.model.module
        else:
            model_ref = runner.model
        if not self.evaluated_modules:
            submodules = model_ref.submodules
        else:
            submodules = self.evaluated_modules
        key_scores = []
        from mmdet.apis import multi_gpu_test

        for submodule in submodules:    # inference를 teacher로 할건지, student로 할건지
            if submodule == 'student':
                break
            # change inference on
            model_ref.inference_on = submodule
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect,
            )   # 여기서 eval이 한번 돌아감
            if runner.rank == 0:
                key_score = self.evaluate(runner, results, prefix=submodule)
                if key_score is not None:
                    key_scores.append(key_score)

        if runner.rank == 0:
            runner.log_buffer.ready = True
            if len(key_scores) == 0:
                key_scores = [None]
            best_score = key_scores[0]
            for key_score in key_scores:
                if hasattr(self, "compare_func") and self.compare_func(
                    key_score, best_score
                ):
                    best_score = key_score

            print("\n")
            # runner.log_buffer.output["eval_iter_num"] = len(self.dataloader)
            if self.save_best:
                self._save_ckpt(runner, best_score)

    def evaluate(self, runner, results, prefix=""):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs
        )
        for name, val in eval_res.items():
            runner.log_buffer.output[(".").join([prefix, name])] = val

        if self.save_best is not None:
            if self.key_indicator == "auto":
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None


@HOOKS.register_module()
class GMMSubModulesDistEvalHook(GMMDistEvalHook):
    def __init__(self, *args, evaluated_modules=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluated_modules = evaluated_modules
        self.gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
        self.history_cls_loss = []
        self.history_bbox_loss = []
        self.history = [self.history_cls_loss, self.history_bbox_loss]

    def gmm_epoch(self, runner):
        """Called after every training iter to evaluate the results."""
        return self._do_evaluate(runner)    # output train data loaders

    def push_to_tensor_alternative(self, tensor_list, new_tensor):
        return torch.cat([tensor_list[1:5], new_tensor.unsqueeze(0)], dim=-1)


    def get_CN_label(self, loss, history, cls_idx):
        rank, world_size = get_dist_info()
        try:
            min_loss = min(loss).item()
            max_loss = max(loss).item()
        except:
            print(f'rank {rank} not pass in class {cls_idx}!')
            
        output_loss = (loss-min_loss)/(max_loss-min_loss)  # <- torch.tensor(5000) 

        if len(history) == 0:
            input_loss = output_loss.reshape(-1, 1).cpu()
        elif len(history) == 5:
            input_loss = self.push_to_tensor_alternative(history, output_loss).mean(0).reshape(-1, 1).cpu()
        else:
            history.append(loss)
            input_loss = torch.mean(torch.stack([history, loss], dim=-2)).cpu()

        # numpy를 받는건가..?
        # fit a two-component GMM to the loss
        self.gmm.fit(input_loss)
        prob = self.gmm.predict_proba(input_loss)        # TODO: prob가 input 갯수만큼 나오는지  
        prob = prob[:,self.gmm.means_.argmin()]         

        threshold = 0.5     # TODO
        CN_label = torch.from_numpy(prob > threshold)     # TODO: pred의 output type 확인하기  

        prob_var = np.var(prob)
        prob_mean = np.mean(prob)
        # dynamic thresold
        gmm_pro_max = 0.5 + 2*(0.25 - prob_var) * (1- prob_mean)
        gmm_pro_min = 0.5 - 2*(0.25 - prob_var) * (prob_mean)

        thre = max_total.ge(self.config.threshold1 * classwise_acc[
            max_idx_total]).detach().cpu().numpy()  # * classwise_acc[max_idx_total]
        
        # refine
        pred_zero = (prob >= gmm_pro_max) * thre
        label_zero_inedex = pred_zero.nonzero()[0]
        pred_one = (prob < gmm_pro_min)  * thre
        label_one_inedex = pred_one.nonzero()[0]
        print('dynamic threspld max: ', gmm_pro_max)
        print('dynamic threspld min: ', gmm_pro_min)

        # make train data
        train_logit_clean = logit_total[label_zero_inedex].detach().cpu().numpy()
        train_logit_delta_clean = logit_delta_total[label_zero_inedex].detach().cpu().numpy()
        train_noise_clean = target_total[label_zero_inedex].detach().cpu().numpy()
        # feature_clean = feature_total[label_zero_inedex].detach().cpu().numpy()
        train_logit_clean_label = np.zeros(len(train_logit_clean),
                                           dtype=np.float32)  # - losses[label_zero_inedex].detach().cpu().numpy() #np.zeros(len(train_logit_clean), dtype=np.float32) -

        return CN_label
        


    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.

        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, ".gmm_eval_hook")

        if is_module_wrapper(runner.model):
            model_ref = runner.model.module
        else:
            model_ref = runner.model
        if not self.evaluated_modules:
            submodules = model_ref.submodules
        else:
            submodules = self.evaluated_modules
        from mmdet.apis import gmm_multi_gpu_test

        # 1. submodules 어떤거 들어오는지 체크 
        # 2. results가 각 bbox별로 들어오는데, 어떻게 넣을지 체크
        for submodule in submodules:
            if submodule=='student':
                continue
            # change inference on
            model_ref.inference_on = submodule
            results = gmm_multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect,
                gmm=True,
            )

        bbox_ids = results[0] 
        loss_cls = results[1]
        loss_bbox = results[2] 
        logits_cls = results[3]
        gt_labels_cls = results[4]

        losses = [loss_cls, loss_bbox]  
        
        splitnet_data = [bbox_ids, loss_bbox, logits_cls, gt_labels_cls]
        gmm_original = False
        # if gmm_original:
        #     CN_label_list = []
        #     for type_idx, loss in enumerate(losses):
        #         CN_label = self.get_CN_label(loss, self.history[type_idx])
        #         CN_label_list.append(CN_label)
        #     clean_noise_label = CN_label_list[0] * CN_label_list[1]        
        # else:
        len_bbox_ids = torch.zeros(len(bbox_ids))
        clean_noise_label = torch.zeros(len(bbox_ids), dtype=torch.bool)
        for cls_idx in range(80):
            pos_cls_ids = torch.nonzero(gt_labels_cls == cls_idx, as_tuple=False).squeeze() # squeeze하면 어떻게 되는지 -> (dim, 1) -> (dim,)
            len_bbox_ids[pos_cls_ids] = True
            CN_label_list = []
            for type_idx, loss in enumerate(losses):
                new_loss = loss[pos_cls_ids]
                CN_label = self.get_CN_label(new_loss, self.history[type_idx], cls_idx)
                CN_label_list.append(CN_label)
            clean_noise_label[pos_cls_ids] = CN_label_list[0] * CN_label_list[1]
            len_bbox_ids[pos_cls_ids] = True
        splitnet_data.append(clean_noise_label)
        return splitnet_data
