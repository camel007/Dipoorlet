import math
import numpy as np
import torch
import torch.distributed as dist
from onnx import numpy_helper

from ..forward_net import ActivationCache
from ..quantize import quant_graph
from ..utils import ONNXGraph, logger


def _get_bias_shape(graph_bc, node):
    if len(node.input) > 2 and node.input[2] in graph_bc.initializer:
        return numpy_helper.to_array(graph_bc.initializer[node.input[2]][0]).shape
    weight = numpy_helper.to_array(graph_bc.initializer[node.input[1]][0])
    return (weight.shape[0],)


def _reduce_bias_diff(fp_activations, q_activations, node, graph_bc, args):
    axis = (0, 2, 3) if node.op_type == 'Conv' else (0,)
    if len(fp_activations) == 0:
        local_sum = np.zeros(_get_bias_shape(graph_bc, node), dtype=np.float64)
        count = 0
    else:
        bias_diff = np.stack(fp_activations, axis=0) \
            - np.stack(q_activations, axis=0)
        bias_diff = np.squeeze(bias_diff, axis=1)
        local_sum = bias_diff.sum(axis=axis)
        count = 1
        for ax in axis:
            count *= bias_diff.shape[ax]

    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda", args.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        sum_tensor = torch.from_numpy(local_sum).to(device)
        cnt_tensor = torch.tensor([count], device=device, dtype=torch.float32)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
        total = cnt_tensor.item()
        if total == 0:
            return local_sum
        return (sum_tensor / total).cpu().numpy()
    if count == 0:
        return local_sum
    return local_sum / count


def update_conv_node_bias(graph_bc, node, fp_activations, q_activations, args):
    bias_diff = _reduce_bias_diff(fp_activations, q_activations, node, graph_bc, args)
    if len(node.input) > 2:
        ori_bias = numpy_helper.to_array(graph_bc.initializer[node.input[2]][0])
        corrected_bias = ori_bias + bias_diff
        corrected_bias_name = graph_bc.initializer[node.input[2]][0].name
        graph_bc.set_initializer(corrected_bias_name, corrected_bias)
        graph_bc.tensor_name_shape_map[corrected_bias_name] = \
            graph_bc.tensor_name_shape_map.pop(graph_bc.initializer[node.input[2]][0].name)
        graph_bc.input.append(corrected_bias_name)
    else:
        bias = bias_diff
        bias_name = node.name + '_bias'
        graph_bc.set_initializer(bias_name, bias)
        graph_bc.tensor_name_shape_map[bias_name] = list(bias.shape)
        graph_bc.input.append(bias_name)
        for bc_node in graph_bc.graph.node:
            if bc_node.name == node.name:
                bc_node.input.append(bias_name)
                return


def bias_correction(graph, act_clip_val, weight_clip_val, args):
    bias_correction_node_type = ['Conv', 'Gemm']
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_bc = ONNXGraph()
    graph_bc.copy_from(graph)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank_num = int(math.ceil(args.data_num / float(world_size)))
    rank_st = rank * rank_num
    rank_ed = min(rank_st + rank_num, args.data_num)
    fp_cache = ActivationCache(graph, args, rank_st, rank_ed)
    prev_act = None
    for node in graph.graph.node:
        if node.op_type in bias_correction_node_type:
            logger.info("Update bias for node: {}".format(node.name))
            # We should do incremental update here.
            graph_q, _ = quant_graph(graph_bc, clip_val, args)
            q_cache = ActivationCache(graph_q, args, rank_st, rank_ed)
            if prev_act is not None:
                q_cache.activation_cache = prev_act
            _ = q_cache[node.input[0]]
            prev_act = q_cache.activation_cache.copy()
            update_conv_node_bias(graph_bc, node, fp_cache[node.output[0]], q_cache[node.output[0]], args)
            graph_bc.update_model()

    if rank == 0:
        graph_bc.save_onnx_model('update_bias_model')
