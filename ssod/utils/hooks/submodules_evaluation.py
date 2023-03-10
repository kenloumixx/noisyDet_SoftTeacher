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
import json

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

        for submodule in submodules:
            # change inference on
            model_ref.inference_on = submodule
            results = multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect,
            )
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
        
    def gmm_epoch(self, runner):
        """Called after every training iter to evaluate the results."""
        return self._do_evaluate(runner)    # output train data loaders

    def push_to_tensor_alternative(self, tensor_list, new_tensor):
        return torch.stack([tensor_list[1:5], new_tensor.unsqueeze(0)], dim=-1)


    def get_CN_label(self, loss, type_idx):
        rank, world_size = get_dist_info()
        min_loss = min(loss).item()
        max_loss = max(loss).item()
        output_loss = (loss-min_loss)/(max_loss-min_loss)  # <- torch.tensor(5000)  # cuda ????????????
        output_loss = output_loss.cpu()
        
        if type_idx == 0:            
            if len(self.history_cls_loss) == 0:
                input_loss = output_loss.reshape(-1, 1)
                self.history_cls_loss.append(output_loss)
            else:
                if len(self.history_cls_loss) == 5:
                    self.history_cls_loss = self.history_cls_loss[1:]
                self.history_cls_loss.append(output_loss)
                input_loss = torch.mean(torch.stack(list(self.history_cls_loss), dim=-1), dim=-1).reshape(-1, 1)
        else:
            if len(self.history_bbox_loss) == 0:
                input_loss = output_loss.reshape(-1, 1)
                self.history_bbox_loss.append(output_loss)
            else:
                if len(self.history_bbox_loss) == 5:
                    self.history_bbox_loss = self.history_bbox_loss[1:]
                self.history_bbox_loss.append(output_loss)
                input_loss = torch.mean(torch.stack(list(self.history_bbox_loss), dim=-1), dim=-1).reshape(-1, 1)
                
        # numpy??? ????????????..?
        # fit a two-component GMM to the loss
        self.gmm.fit(input_loss)
        prob = self.gmm.predict_proba(input_loss)        # TODO: prob??? input ???????????? ????????????  
        prob = prob[:,self.gmm.means_.argmin()]         

        # thre = 0.5     # TODO
        # CN_label = torch.from_numpy(prob > thre)     # TODO: pred??? output type ????????????  
        
        CN_label = np.ones_like(prob, dtype=np.int64) * (9)    # TODO: pred??? output type ????????????  
        # len(CN_label) = len_bbox_ids

        # dynamic thresold        
        prob_var = np.var(prob)
        prob_mean = np.mean(prob)
        gmm_pro_max = 0.5 + 2*(0.25 - prob_var) * (1- prob_mean)
        gmm_pro_min = 0.5 - 2*(0.25 - prob_var) * (prob_mean)
 
        # gmm_pro_max = 0.8 + type_idx * 0.1 
        # gmm_pro_min = 0.1
        
        # refine
        pred_zero = (prob >= gmm_pro_max)     # clean
        label_zero_inedex = pred_zero.nonzero()[0]
        pred_one = (prob < gmm_pro_min)
        label_one_inedex = pred_one.nonzero()[0]   # noise 
        # label_X_inedex = np.setdiff1d(np.arange(len(input_loss)), np.union1d(label_zero_inedex, label_one_inedex))  # X
        CN_label[label_zero_inedex] = 0   
        CN_label[label_one_inedex] = 1
        label_total_inedex = np.union1d(label_zero_inedex, label_one_inedex)
        # 0 - clean. 1 - noise. 2 - do not belong to neither classes.         
        '''
        print('dynamic threspld max: ', gmm_pro_max)
        print('dynamic threspld min: ', gmm_pro_min)

        # make train data
        train_logit_clean = logit_total[label_zero_inedex].detach().cpu().numpy()
        train_logit_delta_clean = logit_delta_total[label_zero_inedex].detach().cpu().numpy()
        train_noise_clean = target_total[label_zero_inedex].detach().cpu().numpy()
        # feature_clean = feature_total[label_zero_inedex].detach().cpu().numpy()
        train_logit_clean_label = np.zeros(len(train_logit_clean),
                                           dtype=np.float32)  # - losses[label_zero_inedex].detach().cpu().numpy() #np.zeros(len(train_logit_clean), dtype=np.float32) -

        train_logit_dirty = logit_total[label_one_inedex].detach().cpu().numpy()
        train_logit_delta_dirty = logit_delta_total[label_one_inedex].detach().cpu().numpy()
        train_noise_dirty = target_total[label_one_inedex].detach().cpu().numpy()
        # feature_dirty = feature_total[label_one_inedex].detach().cpu().numpy()
        train_logit_dirty_label = np.ones(len(train_logit_dirty),
                                          dtype=np.float32)  # - losses[label_one_inedex].detach().cpu().numpy() # np.ones(len(train_logit_dirty), dtype=np.float32) -

        train_data_logit = np.concatenate((train_logit_clean, train_logit_dirty), axis=0)
        train_data_logit_delta = np.concatenate((train_logit_delta_clean, train_logit_delta_dirty), axis=0)
        train_data_noise = np.concatenate((train_noise_clean, train_noise_dirty), axis=0)
        train_data_label = np.concatenate((train_logit_clean_label, train_logit_dirty_label), axis=0)
        # train_data_feature = np.concatenate((feature_clean, feature_dirty), axis=0)

        print("splitnet clean data : ", len(label_zero_inedex))
        print("splitnet dirty data : ", len(label_one_inedex))
        '''
        return CN_label, label_total_inedex # ???.. ?????? splitnet label ????????? ???????????????..^^ ?????? ?????? ?????????..^^ <- ??? ??????????????????.. tensor ?????? ????????? ????????? ?????? ?????? ????????????.. 
        


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


        # 1. submodules ????????? ??????????????? ?????? 
        # 2. results??? ??? bbox?????? ???????????????, ????????? ????????? ??????
        for submodule in submodules:
            if submodule=='student':    # ???????????? teacher??? student?????? ?????? ??? ?????????!!
                continue
            # change inference on
            model_ref.inference_on = submodule
            results = gmm_multi_gpu_test(
                runner.model,
                self.dataloader,
                tmpdir=tmpdir,
                gpu_collect=self.gpu_collect,
                gmm=True,       # gmm->true - gt bbox roi??? feat??? ??????
            )

        bbox_ids = results[0] 
        loss_cls = results[1]
        loss_bbox = results[2] 
        logits_cls = results[3]
        gt_labels_cls = results[4]

        losses = [loss_cls, loss_bbox]  
    
        rank, world_size = get_dist_info()
        splitnet_data = [bbox_ids, loss_bbox, logits_cls, gt_labels_cls]
        
        # 1. ignore class 
        '''
        CN_label_list = []
        for type_idx, loss in enumerate(losses):
            CN_label = self.get_CN_label(loss, self.history[type_idx])
            CN_label_list.append(CN_label)
        clean_noise_label = CN_label_list[0] * CN_label_list[1]        
        # len_bbox_ids = torch.zeros(len(bbox_ids))
        '''
        if runner.start:
            if os.path.exists("history.json"):
                with open("history.json", "r") as json_file: 
                    data = json.load(json_file)
                self.history_cls_loss = [torch.tensor(cls_l).cpu() for cls_l in data['cls_loss']]
                self.history_bbox_loss = [torch.tensor(bbox_l).cpu() for bbox_l in data['bbox_loss']]
                print('\njson file loaded! \n')
                

        clean_noise_label = torch.ones_like(bbox_ids).long() * (-1)     # label??? training??? ?????? ?????????!
        CN_label_list = []
        label_total_idx_list = []
        for type_idx, loss in enumerate(losses):
            CN_label, label_total_idx = self.get_CN_label(loss, type_idx)
            CN_label_list.append(CN_label)
            label_total_idx_list.append(label_total_idx)    # C, N, X ??? C/N??? ????????????

        if rank == 0:
            if ((runner.iter + 1) % 7350 == 0) or ((runner.iter+1) == 2000):
                # with open("history.json", "w") as json_file: 
                with open("history.json", "w") as json_file: 
                    json.dump({'iter': runner.iter+1, 'cls_loss': [cls_l.numpy().tolist() for cls_l in self.history_cls_loss], 'bbox_loss': [bbox_l.numpy().tolist() for bbox_l in self.history_bbox_loss]}, json_file)
                    print('\njson file dumped! \n')
                
        class_label_total_idx = np.intersect1d(label_total_idx_list[0], label_total_idx_list[1])    # cls, reg ?????? ?????? ????????? ??? ?????????! -> type : np.ndarray
        class_ids_for_splitnet_train = np.arange(len(bbox_ids))[class_label_total_idx]  # ?????? ????????? ?????? ??? ??????????????? ??????????????? 

        class_CN_idx = [int(str(a)+str(b), 2) for a, b in zip(CN_label_list[0][class_label_total_idx], CN_label_list[1][class_label_total_idx])]    # 0?????? ???????????? CC, CN, NC, NN

        # for GMM-GT 
        GMM_GT_idx = torch.tensor([int(str(a)+str(b)) for a, b in zip(CN_label_list[0], CN_label_list[1])])
        
        class_CN_label = torch.tensor(class_CN_idx).to(rank)
        clean_noise_label[class_label_total_idx] = class_CN_label     # ???.. ????????? ?????? ?????????????????????.. ????????? ?????????.. ?????? ?????? ????????? ????????? ?????????.. 
        splitnet_data.append(clean_noise_label)
        return splitnet_data, class_ids_for_splitnet_train, GMM_GT_idx  # ids ??????????????? splitnet data for train????????? ??? ids??? ?????? ??? ??????!

        
        '''
        # 2. class-wise 
        clean_noise_label = torch.ones_like(bbox_ids).long() * (-1)     # label??? training??? ?????? ?????????!
        class_ids_for_splitnet_train_list = []
        # ?????? ??????.. class?????? ????????? ??? ?????????????????? ???????????? ??? ?????????..
        for cls_idx in range(80):
            pos_cls_ids = torch.nonzero(gt_labels_cls == cls_idx, as_tuple=False).squeeze() # squeeze?????? ????????? ????????? -> (dim, 1) -> (dim,)

            # len_bbox_ids[pos_cls_ids] = True
            CN_label_list = []
            label_total_idx_list = []
            for type_idx, loss in enumerate(losses):
                new_loss = loss[pos_cls_ids]    # ????????? clss??? ???????????? loss??? ?????? ???????????? ??????
                CN_label, label_total_idx = self.get_CN_label(new_loss, self.history[type_idx], cls_idx)
                CN_label_list.append(CN_label)
                label_total_idx_list.append(label_total_idx)    # C, N, X ??? C/N??? ????????????

            # ????????? ?????? ??????
            # import pdb 
            # pdb.set_trace() # ?????? ???????????? zip(CN_label_list[0][class_label_total_idx].int(), CN_label_list[1][class_label_total_idx].int())
            class_label_total_idx = np.intersect1d(label_total_idx_list[0], label_total_idx_list[1])    # cls, reg ?????? ?????? ????????? ??? ?????????! -> type : np.ndarray
            class_ids_for_splitnet_train = np.arange(len(bbox_ids))[pos_cls_ids.tolist()][class_label_total_idx]  # ?????? ????????? ?????? ??? ??????????????? ??????????????? 
            # (bbox_ids 10777????????? ?????? ?????????) pos_cls_ids??? ???????????? ids ???????????? ????????? class_label_total_idx??? ???????????? ids ???????????????
            # ????????? ?????? ???????????? zip(CN_label_list[0][class_label_total_idx].int(), CN_label_list[1][class_label_total_idx].int())
            # ????????? ?????? ??? -> ??? cls?????? ???????????? bbox??? ids??? ???????????? ?????? = class_ids_for_splitnet_train

            class_CN_idx = [int(str(a)+str(b), 2) for a, b in zip(CN_label_list[0][class_label_total_idx], CN_label_list[1][class_label_total_idx])]    # 0?????? ???????????? CC, CN, NC, NN
            # import pdb 
            # pdb.set_trace()
            # class_CN_idx = [int(str(a.item())+str(b.item()), 3) for a, b in zip(CN_label_list[0].int(), CN_label_list[1].int())]    # C/N
            class_CN_label = torch.tensor(class_CN_idx).to(rank)
            # class_CN_label[list(range(len(new_loss))), class_CN_idx] = 1
            clean_noise_label[pos_cls_ids[class_label_total_idx]] = class_CN_label     # ???.. ????????? ?????? ?????????????????????.. ????????? ?????????.. ?????? ?????? ????????? ????????? ?????????.. 
            class_ids_for_splitnet_train_list.extend(class_ids_for_splitnet_train)
            # len_bbox_ids[pos_cls_ids] = True
            # ????????? ????????? ???????????? ?????? ????????? ????????????   

        splitnet_data.append(clean_noise_label)
        return splitnet_data, class_ids_for_splitnet_train_list  # ids ??????????????? splitnet data for train????????? ??? ids??? ?????? ??? ??????!
        '''