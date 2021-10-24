import torch
import numpy
import numpy as np
import scipy.io as sio
from train_and_test.tr_te import one_mc_trial
import argparse
import time
import os


if_fix_random_seed = True
random_seed = 1

def parse_args():
    parser = argparse.ArgumentParser(description='online scalar channel prediction')
    parser.add_argument('--window_length', type=int, default=5, help='window length for channel prediction (size of covariate vector (N))')
    parser.add_argument('--lag', type=int, default=3, help='prediction lag (delta)') 
    parser.add_argument('--num_samples_for_test', type=int, default=10000, help='number of test samples for computing the MSE during meta-testing') 
    parser.add_argument('--num_paths', type=int, default=1, help='dimension of the channel (should be 1 since we are considering scarlar now)') 
    parser.add_argument('--if_perfect_statistics', dest='if_perfect_statistics', action='store_true', default=False, help='for genie-aided')
    parser.add_argument('--if_ridge', dest='if_ridge', action='store_true', default=False, help='whether to consider ridge regression or ordinary least squares')
    parser.add_argument('--linear_ridge_mode', type=int, default=2, help='0: meta learning by closed form, 1: joint learning, 2: conventional learning, 6: meta-learning by implicit gradient theorem, 7: meta-learning by EP')
    parser.add_argument('--ridge_lambda_coeff', type=float, default=1, help='coefficient for the regularizor (lamdba)') 
    parser.add_argument('--meta_training_samples_per_task', type=int, default=None, help='number of samples for meta-training dataset (L = L^tr + L^te)') 
    parser.add_argument('--num_meta_tr_tasks', type=int, default=None, help='number of meta-training tasks (determined automatically)') 
    parser.add_argument('--num_meta_te_tasks', type=int, default=None, help='number of meta-testing tasks (determined automatically)') 
    parser.add_argument('--fading_mode', type=int, default=1, help='0: indep fading (deprecated), 1: rounded and jakes, 2: 3gpp standard channel') 
    parser.add_argument('--if_jakes_rounded_toy', dest='if_jakes_rounded_toy', action='store_true', default=False) # deprecated
    parser.add_argument('--if_mtr_fix_supp', dest='if_mtr_fix_supp', action='store_true', default=False,  help='whether to use fixed supp size (deprecated, we now control this in the main file)') # deprecated
    parser.add_argument('--meta_training_fixed_supp', type=int, default=8, help='fixed supp size for meta-training if args.if_mtr_fix_supp = True (deprecated now)')
    parser.add_argument('--if_wiener_filter_as_ols', dest='if_wiener_filter_as_ols', action='store_true', default=False, help ='consider ordinary least squares for regressor (should be always true)')
    parser.add_argument('--normalize_factor', type=int, default=1, help='normalize factor for ridge regression during meta-testing, we are not considering normalization now. if 1 no normalize')
    parser.add_argument('--normalize_factor_meta_ridge', type=int, default=1, help='normalize factor for ridge regression during meta-training, we are not considering normalization now. if 1 no normalize')
    parser.add_argument('--if_meta_noise_free', dest='if_meta_noise_free', action='store_true', default=False) # for experiment if meta-training with perfect channel would be useful -- not very effective so deprecated now
    parser.add_argument('--if_joint_noise_free', dest='if_joint_noise_free', action='store_true', default=False) # for experiment if joint training with perfect channel would be useful -- not very effective so deprecated now
    parser.add_argument('--if_simple_multivariate_extension', dest='if_simple_multivariate_extension', action='store_true', default=False) #deprecated
    parser.add_argument('--if_exp_over_noise', dest='if_exp_over_noise', action='store_true', default=False, help='run experiment over channel estimation noise')
    parser.add_argument('--if_exp_over_supp', dest='if_exp_over_supp', action='store_true', default=False, help='run experiment over support')
    parser.add_argument('--multivariate_expansion_dim_from_doppler_scalar', type=int, default=None, help='deprecated')
    parser.add_argument('--noise_var_for_exp_over_supp', type=float, default=0.0, help='channel estimation noise variance for experiment over supp')
    parser.add_argument('--fixed_lambda_value', type=float, default=None, help='if None, use grid search, else: fix with this lambda always')
    parser.add_argument('--total_round_online', type=int, default=400, help='total number of frames for online learning')
    parser.add_argument('--num_recent_saving_tasks_online', type=int, default=1, help='memory size for online learning (M)')
    parser.add_argument('--lr_EP', type=float, default=0.01, help='step size for EP')
    parser.add_argument('--lr_EP_decay', type=float, default=1, help='step size decay for EP (not used)')
    parser.add_argument('--if_meta_closed_form_keep_as_regul_online', dest='if_meta_closed_form_keep_as_regul_online', action='store_true', default=False) #deprecated
    parser.add_argument('--if_online_meta_update_when_memory_full', dest='if_online_meta_update_when_memory_full', action='store_true', default=False, help='whether not to update until memory gets full (almost similar on/off)')
    parser.add_argument('--if_for_online_meta_test_do_offline_meta_train', dest='if_for_online_meta_test_do_offline_meta_train', action='store_true', default=False, help='for comparision with offline meta-tarining')
    parser.add_argument('--offline_meta_training_total_tasks', type=int, default=None, help='number of tasks for offline meta-training when args.if_for_online_meta_test_do_offline_meta_train=True (F)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print('Called with args:')
    print(args)

    torch.set_default_dtype(torch.double)
    torch.set_default_tensor_type(torch.DoubleTensor)

    if args.linear_ridge_mode == 0:
        assert args.meta_training_samples_per_task == None # we are going to set as curr supp (L^new) + fixed query (100)
    else:
        pass

    assert args.if_jakes_rounded_toy == False
    assert args.fading_mode == 3

    curr_dir = '../../../results/online/total_round_400/offline_F_' +str(args.offline_meta_training_total_tasks) + '/meta_offline/memory_' + str(args.num_recent_saving_tasks_online) + '/fix_lambda_1/query_100/' + '_window_' + str(args.window_length) + '_lag_' + str(args.lag) + '/linear_ridge_mode_' + str(args.linear_ridge_mode) + '/if_ridge' + str(args.if_ridge) + '/if_perf' + str(args.if_perfect_statistics) + '/'
    if os.path.isdir(curr_dir):
        pass
    else:
        os.makedirs(curr_dir)

    if args.fading_mode == 3:
        args.num_paths = 1
        if args.if_jakes_rounded_toy:
            raise NotImplementedError
        else:
            args.num_meta_tr_tasks = None # -- it is increased by time
            args.num_meta_te_tasks = 1
    else:
        raise NotImplementedError

    eval_results_path = curr_dir + 'test_result.mat'

    ## we do not consider normalization at all for all cases
    assert args.normalize_factor_meta_ridge == 1
    assert args.normalize_factor == 1

    if args.multivariate_expansion_dim_from_doppler_scalar is not None:
        raise NotImplementedError
    else:
        args.toy_long_term = None

    if if_fix_random_seed:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy.random.seed(random_seed)
    else:
        pass

    available_velocity_list = [0] # deprecated
    num_supp = 8
    noise_var = args.noise_var_for_exp_over_supp
    args.noise_var = noise_var
    args.noise_var_meta = noise_var
    lambda_coeff = args.fixed_lambda_value
    args.ridge_lambda_coeff = lambda_coeff
    mc_num = 10
    if args.if_for_online_meta_test_do_offline_meta_train:
        assert mc_num < 100
    else:
        pass
    ind_velocity = 0 # deprecated
    mse_curr_net_per_supp_wiener = torch.zeros(mc_num, args.total_round_online)
    curr_rounds = torch.zeros(mc_num, args.total_round_online)
    available_supp = torch.zeros(mc_num, args.total_round_online)

    # actual running
    if args.if_for_online_meta_test_do_offline_meta_train:
        print('--------------------------------------------------------')
        print('number of total frames for offline meta-learning (F) ', args.offline_meta_training_total_tasks)
    else:
        print('--------------------------------------------')
        print('memory size for online meta-learning (M) ', args.num_recent_saving_tasks_online)
    for ind_mc in range(mc_num):
        print('----------ind_mc----------', ind_mc)
        args.ridge_meta = None # once we compute this, do not need to repeat this again!
        args.Jake_dataloader_meta = None
        args.Jake_dataloader = None
        args.common_mean_joint = None # for mode 3
        args.v_bar_curr = None ## for online gradient-based learning # if None start with zeros, else: keep updating -- like SGD 
        args.momentum = torch.zeros(1) # for momentum gradient! - not used now
        args.momentum_coeff = 0.0 
        args.prev_v_bar_for_online_for_closed_form = None # this is for closed form!! # if None -- original method , if not None, use this value as regularizer for next round when updating v_bar with current memory # deprecated feature now
        for ind_round in range(args.total_round_online):
            if args.if_meta_closed_form_keep_as_regul_online:
                pass
            else:
                args.prev_v_bar_for_online_for_closed_form = None
            if ind_round % 1000000000 == 0: #deprecated lr decay
                if ind_round == 0:
                    pass
                else:
                    args.lr_EP /= args.lr_EP_decay
            args.num_meta_tr_tasks = ind_round
            if args.num_meta_tr_tasks > args.num_recent_saving_tasks_online:
                args.num_meta_tr_tasks = args.num_recent_saving_tasks_online # only saving recent tasks
            if args.if_online_meta_update_when_memory_full:
                if args.num_meta_tr_tasks < args.num_recent_saving_tasks_online: 
                    print('memory not filled yet.. wait until the memory get fulled!')
                    args.num_meta_tr_tasks = 0
                else:
                    pass
            else:
                pass
            
            args.ind_round = ind_round
            # always renew meta-learned hyperparameter
            if args.if_for_online_meta_test_do_offline_meta_train:
                if args.offline_meta_training_total_tasks is None:
                    args.offline_meta_training_total_tasks = args.total_round_online-1
                else:
                    pass
                args.num_meta_tr_tasks = args.offline_meta_training_total_tasks
            else:
                args.ridge_meta = None
                args.Jake_dataloader_meta = None
            args.Jake_dataloader = None
            available_supp[ind_mc, ind_round] = num_supp
            velocity_kmph_actual = None
            # args.num_samples_for_test: number of samples to compute accuracy -- not necessarilly have to be same as online meta-trainig dataset 
            snr_curr, mse_wiener, mse_nn_best = one_mc_trial(args, curr_dir, num_supp, args.num_samples_for_test, velocity_kmph_actual, ind_mc)
            if ind_round % 200 == 0:
                print('ind_round', ind_round, 'mse', mse_wiener)
            mse_curr_net_per_supp_wiener[ind_mc, ind_round] = mse_wiener  
            curr_rounds[ind_mc, ind_round] = ind_round + 1
            eval_results = {}
            eval_results['mse_wiener'] = mse_curr_net_per_supp_wiener.data.numpy() # mean, first dim, second dim
            eval_results['supps'] = available_supp.data.numpy()
            eval_results['rounds'] = curr_rounds.data.numpy()
            sio.savemat(eval_results_path, eval_results)
        print('ind_round', ind_round, 'mse', mse_wiener)