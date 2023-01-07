import torch
from mmcv.runner import get_dist_info
import torch.distributed as dist

a = torch.tensor([1, 2, 3])


b = torch.tensor([4, 5, 6, 7])


c = torch.tensor([8])


d = torch.tensor([9, 10])


def total_tensor(tensor, max_num):      # 1. 패딩 2. 이어붙이기 3. 패딩 벗기기 -> max num 은 딱 패딩하기 좋은 사이즈로만..!
    rank, world_size = get_dist_info()
    # 1. padding하기 
    input_tensor_list = []
    delta = max_num.size(0) - tensor.size(0)            # batch 이후로 가장 앞단의 tensor size 가져오기
    # delta만큼 concat
    tensor = torch.cat(tensor, torch.zeros_like(tensor)[:delta])    # 전부 같은 사이즈
    
    # 2. 이어붙이기
    output = [torch.zeors_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    
    # 3. delta값들 가져오기
    total_delta = [None for _ in range(world_size)]
    rank_delta = [delta]    # 미리 저장해놓기
    all_gather_object(total_delta, rank_delta)
    
    # 4. 패딩 제거하기 
    for idx, delta_rank in enumerate(total_delta.reverse()):
        # max_num만큼 거꾸로 가면서, 원하는 갯수만큼 도려내기
        output = output[:-delta_rank]
    return output
    