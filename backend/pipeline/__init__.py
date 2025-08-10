from enum import IntEnum
from utils.loss_utils import pearson_loss, NID_loss
from utils.opt_utils import grid_search_transform, random_search_transform, random_search_transform2
from utils.print_utils import print_transform_in_euler_and_translation
import numpy as np
import json
from functools import partial


class PipelineMode(IntEnum):
    DEFAULT = 0,
    FINETUNE_BOTH = 1,
    FINETUNE_RORATION = 2,
    FINETUNE_TRANSLATION = 3


class SearchMode(IntEnum):
    RANDOM_SEARCH = 0,
    GRID_SEARCH = 1


def get_pipeline(params):
    mode = PipelineMode(params["mode"])
    if mode == PipelineMode.DEFAULT:
        return get_default_pipeline(params)
    elif mode == PipelineMode.FINETUNE_BOTH:
        return get_finetune_both_pipeline(params)
    elif mode == PipelineMode.FINETUNE_RORATION or mode == PipelineMode.FINETUNE_TRANSLATION:
        return get_finetune_single_pipeline(params)
    else:
        raise NotImplementedError("not implemented pipeline!")


def get_default_pipeline(params):
    # unpack params
    patch_size = int(params["patch_size"])
    init_rot_range = params["init_rot_range"]
    init_rot_resolution = params["init_rot_resolution"]
    coarse_trans_range = params["coarse_trans_range"]
    coarse_iters = params["coarse_iters"]
    fine_trans_range = params["fine_trans_range"]
    fine_iters = params["fine_iters"]

    def pipeline(T_init, frame_data, socket=None):
        def func0(lidar_depths, mono_depth): return pearson_loss(
            lidar_depths, mono_depth, patch_size, shift=0, weight=None, set_zero_to=-1.0)

        def func1(lidar_depths, mono_depth): return pearson_loss(
            lidar_depths, mono_depth, patch_size, shift=patch_size // 2, weight=None, set_zero_to=-1.0)

        def func2(lidar_depths, mono_depth): return pearson_loss(
            lidar_depths, mono_depth, patch_size, shift=0, weight=None, set_zero_to=-1.0) / 5.0

        def func3(lidar_depths, mono_depth): return pearson_loss(lidar_depths, mono_depth,
                                                                 patch_size, shift=patch_size // 2, weight=None, set_zero_to=-1.0) / 5.0

        print_transform_in_euler_and_translation(T_init)

        if socket is not None:
            def socket_callback(step_name, freq, iter, total_iters, T_best):
                if iter != total_iters - 1 and iter % freq != 0:
                    return
                status = {
                    "type": 1,
                    "step": step_name,
                    "iter": iter,
                    "total_iters": total_iters
                }
                socket.send(json.dumps(status))
            callback0 = partial(socket_callback, "Initial Grid Search", 2)
            callback1 = partial(socket_callback, "Coarse Random Search", 10)
            callback2 = partial(socket_callback, "Fine Random Search", 10)
        else:
            callback0 = None
            callback1 = None
            callback2 = None

        T_est = grid_search_transform(T_init=T_init,
                                      loss_func=[func0, func1],
                                      frame_data=frame_data,
                                      rotation_ranges=[-init_rot_range,
                                                       init_rot_range],
                                      rotation_resolutions=[
                                          init_rot_resolution],
                                      callback=callback0)

        T_est = random_search_transform(T_init=T_est,
                                        loss_func=[func2, func3],
                                        frame_data=frame_data,
                                        iterations=coarse_iters,
                                        mode=0,
                                        translation_ranges=[-coarse_trans_range,
                                                            coarse_trans_range],
                                        translation_perturb_at_opt=False,
                                        loss_func_intensity=[NID_loss],
                                        callback=callback1)

        T_est = random_search_transform(T_init=T_est,
                                        loss_func=[func2, func3],
                                        frame_data=frame_data,
                                        iterations=fine_iters,
                                        mode=1,
                                        translation_ranges=[-fine_trans_range,
                                                            fine_trans_range],
                                        translation_perturb_at_opt=False,
                                        loss_func_intensity=[NID_loss],
                                        callback=callback2)

        print_transform_in_euler_and_translation(T_est)

        return T_est

    return pipeline


def get_finetune_both_pipeline(params):
    # unpack params
    patch_size = int(params["patch_size"])
    fine_rot_range = params["fine_rot_range"]
    fine_trans_range = params["fine_trans_range"]
    fine_iters = params["fine_iters"]

    def pipeline(T_init, frame_data, socket=None):
        def func0(lidar_depths, mono_depth): return pearson_loss(
            lidar_depths, mono_depth, patch_size, shift=0, weight=None, set_zero_to=-1.0) / 5.0

        def func1(lidar_depths, mono_depth): return pearson_loss(lidar_depths, mono_depth,
                                                                 patch_size, shift=patch_size // 2, weight=None, set_zero_to=-1.0) / 5.0

        print_transform_in_euler_and_translation(T_init)

        if socket is not None:
            def socket_callback(step_name, freq, iter, total_iters, T_best):
                if iter != total_iters - 1 and iter % freq != 0:
                    return
                status = {
                    "type": 1,
                    "step": step_name,
                    "iter": iter,
                    "total_iters": total_iters
                }
                socket.send(json.dumps(status))
            callback0 = partial(socket_callback, "Fine Random Search", 10)
        else:
            callback0 = None

        rotation_ranges = [-fine_rot_range,
                           fine_rot_range] if fine_rot_range > 0 else None
        translation_ranges = [-fine_trans_range,
                              fine_trans_range] if fine_trans_range > 0 else None
        T_est = random_search_transform2(T_init=T_init,
                                         loss_func=[func0, func1],
                                         frame_data=frame_data,
                                         iterations=fine_iters,
                                         rotation_ranges=rotation_ranges,
                                         rotation_perturb_at_opt=False,
                                         translation_ranges=translation_ranges,
                                         translation_perturb_at_opt=False,
                                         rotation_candidate_num=3,
                                         loss_func_intensity=[NID_loss],
                                         callback=callback0)

        return T_est

    return pipeline


def get_finetune_single_pipeline(params):
    # unpack params
    mode = PipelineMode(params["mode"])
    patch_size = int(params["patch_size"])
    search_mode = SearchMode(params["search_mode"])
    fine_iters = params["fine_iters"]

    if mode == PipelineMode.FINETUNE_RORATION:
        fine_rot_range = params["fine_rot_range"]
        fine_rot_resolution = params["fine_rot_resolution"]
        rotation_ranges = [-fine_rot_range,
                           fine_rot_range] if fine_rot_range > 0 else None
        translation_ranges = None
        fine_trans_resolution = 0
    else:
        fine_trans_range = params["fine_trans_range"]
        fine_trans_resolution = params["fine_trans_resolution"]
        translation_ranges = [-fine_trans_range,
                              fine_trans_range] if fine_trans_range > 0 else None
        rotation_ranges = None
        fine_rot_resolution = None

    def pipeline(T_init, frame_data, socket=None):
        def func0(lidar_depths, mono_depth): return pearson_loss(
            lidar_depths, mono_depth, patch_size, shift=0, weight=None, set_zero_to=-1.0) / 5.0

        def func1(lidar_depths, mono_depth): return pearson_loss(lidar_depths, mono_depth,
                                                                 patch_size, shift=patch_size // 2, weight=None, set_zero_to=-1.0) / 5.0

        print_transform_in_euler_and_translation(T_init)

        if socket is not None:
            def socket_callback(step_name, freq, iter, total_iters, T_best):
                if iter != total_iters - 1 and iter % freq != 0:
                    return
                status = {
                    "type": 1,
                    "step": step_name,
                    "iter": iter,
                    "total_iters": total_iters
                }
                socket.send(json.dumps(status))
            text = "Grid Search" if search_mode == SearchMode.GRID_SEARCH else "Fine Random Search"
            callback0 = partial(socket_callback, text, 10)
        else:
            callback0 = None

        if search_mode == SearchMode.GRID_SEARCH:
            T_est = grid_search_transform(T_init=T_init,
                                          loss_func=[func0, func1],
                                          frame_data=frame_data,
                                          rotation_ranges=rotation_ranges,
                                          rotation_resolutions=[
                                              fine_rot_resolution],
                                          translation_ranges=translation_ranges,
                                          translation_resolutions=[
                                              fine_trans_resolution],
                                          loss_func_intensity=[NID_loss],
                                          callback=callback0)
        else:
            T_est = random_search_transform2(T_init=T_init,
                                             loss_func=[func0, func1],
                                             frame_data=frame_data,
                                             iterations=fine_iters,
                                             rotation_ranges=rotation_ranges,
                                             rotation_perturb_at_opt=False,
                                             translation_ranges=translation_ranges,
                                             translation_perturb_at_opt=False,
                                             rotation_candidate_num=3,
                                             loss_func_intensity=[NID_loss],
                                             callback=callback0)

        return T_est

    return pipeline
