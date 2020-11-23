import torch
import torch.nn as nn
import numpy as np
import os
import copy
from misc.util import argmax_heatmap
from misc.metric import pixel2mm
import pickle
import time
from misc import util
from dataset import transforms

def run_hint_test(config, model, grad:bool, criterion, loader, optimizer=None, writer=None, split=None):

    with torch.no_grad():
        model.eval()
        save=False
        results = iter_epoch(config=config, model=model, grad=grad, criterion=criterion, loader=loader,
                             writer=writer, save=save, split=split)

    return results

def save_basic_item_info(batch, config, batch_idx, save,split):
    image_name = batch['img_path'][0].split('/')[-1]
    item_save_path = os.path.join(config.PATH.RUN.SAVE_HINT_RESULT,'{}_{}_{}'.format(split,batch_idx,image_name))
    item_results_dict = {}
    item_results_dict['batch_idx'] = batch_idx
    item_results_dict['pspace'] = batch['pspace'][0].data.cpu().numpy()
    item_results_dict['image_path'] = batch['img_path'][0]

    # label 정보 저장
    label_keypoint = batch['points_coord_gt']
    label_keypoint_mm = pixel2mm(config, copy.deepcopy(label_keypoint), batch['pspace'])
    if batch['bbox'] is None:
        batch['bbox'] = [-1]
    item_results_dict['label'] = {
        'bbox': batch['bbox'][0].data.cpu().numpy(),
        'opt': batch['opts_gt'][0].data.cpu().numpy(),
        'keypoint': batch['points_coord_gt'][0].data.cpu().numpy()
    }

    return item_results_dict, label_keypoint_mm, item_save_path, image_name

def iter_epoch(config, model, grad, criterion, loader, optimizer=None, writer=None, save=False, split=''):
    num_item = 0
    mae_mm_all_list = [0 for i in range(config.numkey+1)]
    mae_mm_post_processing_list = [0 for i in range(config.numkey+1)]
    mae_mm_weighted_sum_list = [0 for i in range(config.numkey+1)]
    mae_mm_weighted_sum_baseline_list = [0 for i in range(config.numkey+1)]
    mae_mm_weighted_sum_post_processing_list = [0 for i in range(config.numkey+1)]
    mre_mm_list = [0 for i in range(config.numkey+1)]
    mre_mm_baseline_list = [0 for i in range(config.numkey+1)]
    mre_mm_post_processing_list = [0 for i in range(config.numkey+1)]

    mae_mm_baseline_list = [0 for i in range(config.numkey+1)]
    success_rate = {standard : [0 for i in range(config.numkey+1)] for standard in config.success_standard}
    success_rate_baseline =  {standard : [0 for i in range(config.numkey+1)] for standard in config.success_standard}
    success_rate_post_processing = {standard : [0 for i in range(config.numkey+1)] for standard in config.success_standard}

    mean_hint_inference_time = []
    hint0_mae_mm_dict = {}
    for batch_idx, batch in enumerate(loader):
        print('=============== version : {} =============='.format(config.version))
        num_item+=1

        item_results_dict, label_keypoint_mm, item_save_path, image_name = save_basic_item_info(batch, config, batch_idx, save, split)
        hint_index_0to13 = {i:None for i in range(config.numkey+1)}
        batch['img'] = batch['img'].to(config.device)
        batch['points'] = batch['points'].to(config.device)

        hint_matrix = np.zeros((config.numkey+1,config.numkey)) #
        mae_mm_keypoint_matrix = np.ones((config.numkey+1,config.numkey)) * -1 #
        output_keypoint_matrix = np.ones((config.numkey+1,config.numkey,2)) * -1
        output_bbox_matrix = np.ones((config.numkey+1,4)) * -1
        success_dict = {standard:np.zeros((config.numkey+1,config.numkey)) for standard in config.success_standard} #
        success_dict_baseline = {standard: np.zeros((config.numkey+1, config.numkey)) for standard in config.success_standard}
        success_dict_post_processing = {standard: np.zeros((config.numkey+1, config.numkey)) for standard in config.success_standard}


        hint_indices = []
        for num_hint in range(6):
            # hint 만들기
            if num_hint == 0:
                hint_layer_flag = False
                batch['hints'] = torch.zeros_like(batch['points'])
                batch['hint_mask'] = torch.zeros(1, config.numkey, device=batch['points'].device)
                batch['hint_pixel'] = (torch.ones(1, config.numkey, 2, device=batch['points'].device) * (-1))
            else:
                hint_layer_flag = True
                if num_hint == 1:
                    sort_idx = (torch.tensor(mae_mm_keypoint_13by2.mean(-1)).sort()[1]).data.cpu().numpy()[::-1]

                for idx in sort_idx:
                    if idx not in hint_indices:
                        hint_indices.append(int(idx))
                        break

                batch['hints'][:, idx] = copy.deepcopy(batch['points'])[:, idx]
                batch['hint_mask'][:, idx] = 1 # hint = 1 / nohint = 0
                batch['hint_indices'] = torch.tensor(hint_indices).unsqueeze(0).long()




                batch['hint_pixel'][0,idx] = output_keypoint[0,idx]
                batch['hint_pixel'] = batch['hint_pixel'].long()


            hint_learning_start_time = time.time()
            _out = model(batch, hint_layer_flag)

            inference_time = time.time()-hint_learning_start_time
            util.log(config, '\nbatch [{}] image [{}] num hint [{}] inference time [{:.3f}sec]'.format(batch_idx, image_name, num_hint, inference_time))

            if num_hint>0:
                mean_hint_inference_time.append(inference_time)


            with torch.no_grad():
                out = {'bbox': _out[0], 'points': _out[1], 'opts': _out[2]}

                output_keypoint = argmax_heatmap(out['points'])  # (1,13,2)
                output_keypoint_matrix[num_hint] = output_keypoint[0].data.cpu().numpy()
                if out['bbox'] is None:
                    output_bbox_matrix[num_hint] = np.array([-1,-1,-1,-1])
                else:
                    output_bbox_matrix[num_hint] = out['bbox'][0].data.cpu().numpy()
                hint_matrix[num_hint] = batch['hint_mask'].squeeze().data.cpu().numpy() # mask 형태
                hint_index_0to13[num_hint] = np.array(hint_indices).astype(int) # index 형태
                output_keypoint_mm = pixel2mm(config, copy.deepcopy(output_keypoint), batch['pspace'])
                mae_mm_keypoint_13by2 = (nn.L1Loss(reduction='none')(output_keypoint_mm, label_keypoint_mm)[0]).data.cpu().numpy() # (13,2)
                mae_mm_keypoint_13_numpy = mae_mm_keypoint_13by2.mean(-1)
                mae_mm_keypoint_matrix[num_hint] = mae_mm_keypoint_13_numpy # (13,2) -> (13)
                if num_hint == 0:
                    mae_mm_baseline_13 = copy.deepcopy(mae_mm_keypoint_13_numpy)
                mae_mm_baseline_13[hint_matrix[num_hint].astype(bool)] = 0
                mae_mm_baseline = mae_mm_baseline_13.sum() / config.numkey
                mae_mm_baseline_list[num_hint] += mae_mm_baseline
                mae_mm_post_processing_13 = copy.deepcopy(mae_mm_keypoint_13_numpy)
                mae_mm_post_processing_13[hint_matrix[num_hint].astype(bool)] = 0
                mae_mm_post_processing = mae_mm_post_processing_13.sum() / config.numkey
                mae_mm_post_processing_list[num_hint] += mae_mm_post_processing
                weighted_sum_coord_pred = transforms.get_weighted_sum2(out['points']).data.cpu()  # batch, 13, 2 (row, column)
                weighted_sum_coord_pred_mm = pixel2mm(config, copy.deepcopy(weighted_sum_coord_pred), batch['pspace'])
                weighted_sum_mae_mm_keypoint_13by2 = (nn.L1Loss(reduction='none')(weighted_sum_coord_pred_mm, label_keypoint_mm)[0]).data.cpu().numpy()  # (13,2)
                weighted_sum_mae_mm_keypoint_13_numpy = weighted_sum_mae_mm_keypoint_13by2.mean(-1)
                mae_mm_weighted_sum_list[num_hint] += weighted_sum_mae_mm_keypoint_13_numpy.sum() / config.numkey
                if num_hint == 0:
                    mae_mm_weighted_sum_baseline_13 = copy.deepcopy(weighted_sum_mae_mm_keypoint_13_numpy)
                mae_mm_weighted_sum_baseline_13[hint_matrix[num_hint].astype(bool)] = 0 #
                mae_mm_weighted_sum_baseline = mae_mm_weighted_sum_baseline_13.sum() / config.numkey
                mae_mm_weighted_sum_baseline_list[num_hint] += mae_mm_weighted_sum_baseline
                mae_mm_weighted_sum_post_processing_13 = copy.deepcopy(weighted_sum_mae_mm_keypoint_13_numpy)
                mae_mm_weighted_sum_post_processing_13[hint_matrix[num_hint].astype(bool)] = 0  #
                mae_mm_weighted_sum_post_processing = mae_mm_weighted_sum_post_processing_13.sum() / config.numkey
                mae_mm_weighted_sum_post_processing_list[num_hint] += mae_mm_weighted_sum_post_processing
                y_diff_sq = (output_keypoint_mm[:, :, 0] - label_keypoint_mm[:, :, 0]) ** 2
                x_diff_sq = (output_keypoint_mm[:, :, 1] - label_keypoint_mm[:, :, 1]) ** 2
                sqrt_x2y2 = torch.sqrt(y_diff_sq + x_diff_sq)  # (1,13)
                mre_mm_13_numpy = sqrt_x2y2[0].data.cpu().numpy()
                mre_mm = mre_mm_13_numpy.sum() / config.numkey
                mre_mm_list[num_hint] += mre_mm
                if num_hint == 0:
                    mre_mm_baseline_13_numpy = copy.deepcopy(mre_mm_13_numpy)
                mre_mm_baseline_13_numpy[hint_matrix[num_hint].astype(bool)] = 0
                mre_mm_baseline = mre_mm_baseline_13_numpy.sum() / config.numkey
                mre_mm_baseline_list[num_hint] += mre_mm_baseline
                mre_mm_post_processing_13_numpy = copy.deepcopy(mre_mm_13_numpy)
                mre_mm_post_processing_13_numpy[hint_matrix[num_hint].astype(bool)] = 0
                mre_mm_post_processing = mre_mm_post_processing_13_numpy.sum() / config.numkey
                mre_mm_post_processing_list[num_hint] += mre_mm_post_processing
                if num_hint == 0:
                    hint0_mae_mm_dict[image_name] = mae_mm_keypoint_matrix[num_hint].mean()
                for standard in config.success_standard:
                    success_dict[standard][num_hint] = (mae_mm_keypoint_13_numpy < standard).astype('uint8')
                    success_dict_baseline[standard][num_hint] =(mae_mm_baseline_13 < standard).astype('uint8')
                    success_dict_post_processing[standard][num_hint] =(mae_mm_post_processing_13 < standard).astype('uint8')
                util.log(config, 'batch {} | num hint {}'.format(
                        batch_idx, num_hint))
                for standard in config.success_standard:
                    sentence = 'standard {} | success rate {:.3f} | success num {} | baseline success num {} | post processing success num {}'.format(
                        standard,
                        success_dict[standard][num_hint].astype(float).mean(),
                        success_dict[standard][num_hint].sum(),
                        success_dict_baseline[standard][num_hint].sum(),
                        success_dict_post_processing[standard][num_hint].sum()
                    )
                    util.log(config, sentence)
                if num_hint>0:
                    before = mae_mm_keypoint_matrix[num_hint - 1].mean()
                    after = mae_mm_keypoint_matrix[num_hint].mean()
                    util.log(config, 'before MAE_mm ({})[hint {}] {:.4f} | after MAE_mm({})[hint {}] {:.4f} | improved MAE_mm({}) {:.4f} | baseline MAE_mm({}) {:.4f} | post processing MAE_mm {:.4f} | MRE_mm {:.4f} | baseline MRE_mm {:.4f} | post processing MRE_mm {:.4f}'.format(
                            config.numkey,num_hint-1, before, config.numkey, num_hint, after, config.numkey, before-after, config.numkey, mae_mm_baseline,  mae_mm_post_processing, mre_mm, mre_mm_baseline, mre_mm_post_processing
                        ))
                if num_hint == 0:
                    hint0_mae_mm_dict[image_name] = mae_mm_keypoint_matrix[num_hint].mean()

        item_results_dict['output'] = {
            'keypoint':output_keypoint_matrix,
            'bbox':output_bbox_matrix,
            'opt':None,
            'hint_mask': hint_matrix,
            'hint_indices': hint_index_0to13
        }

        item_results_dict['metric'] = {}
        for standard in config.success_standard:
            item_results_dict['metric']['success<{}'.format(standard)] = success_dict[standard]

        save_name = os.path.join(item_save_path,'results.pickle')
        if not os.path.exists(item_save_path):
            os.makedirs(item_save_path)
        with open(save_name, 'wb') as f:
            pickle.dump(item_results_dict, f, pickle.HIGHEST_PROTOCOL)

        for num_hint in range(config.numkey+1):
            for standard in config.success_standard:
                success_rate[standard][num_hint] += success_dict[standard][num_hint].astype(float).mean()
                success_rate_baseline[standard][num_hint] += success_dict_baseline[standard][num_hint].astype(float).mean()
                success_rate_post_processing[standard][num_hint] += success_dict_post_processing[standard][num_hint].astype(float).mean()
            mae_mm_all_list[num_hint] += mae_mm_keypoint_matrix[num_hint].mean() #

    mean_hint_inference_time = np.mean(mean_hint_inference_time)
    util.log(config, '\nmean hint inference time (without hint 0 inference time) : {:.4f}sec'.format(mean_hint_inference_time))
    ranking_dict = {'mae_mm_best': sorted(hint0_mae_mm_dict.items(), key=(lambda x:x[1])) }
    with open('{}/ranking_dict.pickle'.format(config.PATH.RUN.SAVE), 'wb') as f:
        pickle.dump(ranking_dict, f , pickle.HIGHEST_PROTOCOL)

    mae_mm_all_list = [mae_mm_all_list[i]/num_item for i in range(config.numkey+1)] #(batch,numhint)
    mae_mm_baseline_list = [mae_mm_baseline_list[i]/num_item for i in range(config.numkey+1)]
    mae_mm_post_processing_list = [mae_mm_post_processing_list[i] / num_item for i in range(config.numkey+1)]
    mre_mm_list = [mre_mm_list[i] / num_item for i in range(config.numkey+1)]
    mre_mm_baseline_list = [mre_mm_baseline_list[i] / num_item for i in range(config.numkey+1)]
    mre_mm_post_processing_list = [mre_mm_post_processing_list[i] / num_item for i in range(config.numkey+1)]

    util.log(config, '')
    util.log(config, '============================= Summarize results ================================= ')
    util.log(config, '')

    test_metric = 'argmax'
    for num_hint in range(config.numkey+1):
        util.log(config, 'All ({}) | {} | Num hint {:02d} | MAE_mm: {:.6f} | baseline MAE_mm: {:.6f} | post_processing MAE_mm: {:.6f} | MRE_mm: {:.6f} | baseline MRE_mm: {:.6f} | post_processing MRE_mm: {:.6f}'.format(
            config.numkey,test_metric, num_hint,
                               mae_mm_all_list[num_hint], mae_mm_baseline_list[num_hint], mae_mm_post_processing_list[num_hint], mre_mm_list[num_hint], mre_mm_baseline_list[num_hint], mre_mm_post_processing_list[num_hint]
                                                                                                                                                                       ))

    util.log(config, '') # 줄바꿈
    util.log(config, '================================================================================= ')
    util.log(config, '') # 줄바꿈

    util.log(config,'Average success rates (MAE_MM<Xmm) | num_hint 0 1 2 3 4 5 6 7 8 9 10 11 12 13')
    for standard in config.success_standard:
        util.log(config,'Average success rates (MAE_MM<{}mm) | \nafter {} | \nbaseline {} | \npost_processing {}'.format(
                                standard,
                                np.array(success_rate[standard])/num_item,
                                np.array(success_rate_baseline[standard])/num_item,
                                np.array(success_rate_post_processing[standard])/num_item
        ))


    util.log(config, '')



    results = {}
    return results
