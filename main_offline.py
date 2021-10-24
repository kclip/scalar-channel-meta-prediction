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
    parser = argparse.ArgumentParser(description='offline scalar channel prediction')
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
    parser.add_argument('--if_simple_multivariate_extension', dest='if_simple_multivariate_extension', action='store_true', default=False) # deprecated 
    parser.add_argument('--if_exp_over_noise', dest='if_exp_over_noise', action='store_true', default=False, help='run experiment over channel estimation noise')
    parser.add_argument('--if_exp_over_supp', dest='if_exp_over_supp', action='store_true', default=False, help='run experiment over support')
    parser.add_argument('--multivariate_expansion_dim_from_doppler_scalar', type=int, default=None, help='deprecated') # deprecated 
    parser.add_argument('--noise_var_for_exp_over_supp', type=float, default=0.0, help='channel estimation noise variance for experiment over supp')
    parser.add_argument('--fixed_lambda_value', type=float, default=None, help='if None, use grid search, else: fix with this lambda always')
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

    curr_dir = '../../../results/offline/' + 'window_' + str(args.window_length) + '_lag_' + str(args.lag) + '/linear_ridge_mode_' + str(args.linear_ridge_mode) + '/if_ridge' + str(args.if_ridge) + '/if_perf' + str(args.if_perfect_statistics) + '/'
    if os.path.isdir(curr_dir):
        pass
    else:
        os.makedirs(curr_dir)

    if args.fading_mode == 1 or args.fading_mode == 2:
        args.num_paths = 1
        if args.if_jakes_rounded_toy:
            args.num_meta_tr_tasks = 2
            args.num_meta_te_tasks = 2
        else:
            args.num_meta_tr_tasks = 40*2 # rounded & jakes
            args.num_meta_te_tasks = 10*2 # rounded & jakes
    else:
        pass

    eval_results_path = curr_dir + 'test_result.mat'

    ## we do not consider normalization at all for all cases
    assert args.normalize_factor_meta_ridge == 1
    assert args.normalize_factor == 1

    if args.multivariate_expansion_dim_from_doppler_scalar is not None:
        # deprecated
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

    available_velocity_list = [3.59751] #no meaning
    if args.if_jakes_rounded_toy:
        # deprecated
        raise NotImplementedError
    else:
        if args.if_exp_over_supp:
            available_supp_list = [8,10,15,20,50,100,1000, 10000]
        elif args.if_exp_over_noise:
            available_supp_list = [8]
        else:
            raise NotImplementedError
  
    if args.if_exp_over_supp:
        noise_var_list = [args.noise_var_for_exp_over_supp] #[1e-4] #[0.0]
    elif args.if_exp_over_noise:
        noise_var_list = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5] #[0.0]#[1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
    else:
        raise NotImplementedError
    lambda_coeff_list = [args.fixed_lambda_value]#[None] #[1] #[None] 
    # if this is None -> grid search
    # if it has certain value -> use that lambda without grid search
    mc_num = 1
    ind_velocity = 0
    mse_curr_net_per_supp_nn = torch.zeros(len(lambda_coeff_list), len(noise_var_list), mc_num, len(available_supp_list), len(available_velocity_list))
    mse_curr_net_per_supp_wiener = torch.zeros(len(lambda_coeff_list), len(noise_var_list), mc_num, len(available_supp_list), len(available_velocity_list))
    available_supp = torch.zeros(len(lambda_coeff_list), len(noise_var_list), mc_num, len(available_supp_list), len(available_velocity_list))
    available_snr = torch.zeros(len(lambda_coeff_list), len(noise_var_list), mc_num, len(available_supp_list), len(available_velocity_list))

    args.ridge_meta = None # once we compute this, do not need to repeat this again!
    args.common_mean_joint = None # for mode 3
    args.prev_v_bar_for_online_for_closed_form = None # this is for online -- for offline we do not consider continual closed form case # if this is None, original ridge meta
    
    ind_lambda = 0
    for lambda_coeff in lambda_coeff_list:
        args.ridge_meta = None # need to meta-train again for new snr!
        args.Jake_dataloader = None
        args.Jake_dataloader_meta = None
        args.ridge_lambda_coeff = lambda_coeff
        print('lambda coeff', args.ridge_lambda_coeff)
        ind_snr = 0
        for noise_var in noise_var_list:
            args.noise_var = noise_var
            if args.if_meta_noise_free: # deprecated
                args.noise_var_meta = 0 # adding noise directly at ridge_meta or OLS for joint
            else:
                args.noise_var_meta = noise_var
            args.ridge_meta = None # need to meta-train again for new snr!
            args.Jake_dataloader = None
            args.Jake_dataloader_meta = None
            print('noise var', noise_var)
            for velocity_kmph in available_velocity_list:
                ind_supp = 0
                for num_supp in available_supp_list:
                    if args.if_mtr_fix_supp:
                        pass # no need to meta-train again for each supp
                    else: # since supp number is changed, we need to do meta-training again for no discrepancy!
                        args.ridge_meta = None
                        args.Jake_dataloader_meta = None
                    print('number of training samples (L^new): ', num_supp - (args.window_length+args.lag -1))
                    for ind_mc in range(mc_num): # mc for different realization of training (adaptation) set. especially for small number of dataset
                        available_supp[ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = num_supp
                        if noise_var == 0:
                            available_snr[ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = 1e10
                        else:
                            available_snr[ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = 10*numpy.log10(1/noise_var)
                        velocity_kmph_actual = None
                        args.prev_v_bar_for_online_for_closed_form = None # this is for online -- for offline we do not consider continual closed form case # if this is None, original ridge meta
                        snr_curr, mse_wiener, mse_nn_best = one_mc_trial(args, curr_dir, num_supp, args.num_samples_for_test, velocity_kmph_actual, ind_mc)
                        print('mse', mse_wiener)
                        mse_curr_net_per_supp_nn[ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = mse_nn_best
                        mse_curr_net_per_supp_wiener[ind_lambda, ind_snr, ind_mc, ind_supp, ind_velocity] = mse_wiener              
                    ind_supp += 1
                    eval_results = {}
                    eval_results['mse_wiener'] = mse_curr_net_per_supp_wiener.data.numpy() # mean, first dim, second dim
                    eval_results['mse_nn'] = mse_curr_net_per_supp_nn.data.numpy()
                    eval_results['supps'] = available_supp.data.numpy()
                    eval_results['snr'] = available_snr.data.numpy()
                    sio.savemat(eval_results_path, eval_results)
                eval_results = {}
                eval_results['mse_wiener'] = mse_curr_net_per_supp_wiener.data.numpy() # mean, first dim, second dim
                eval_results['mse_nn'] = mse_curr_net_per_supp_nn.data.numpy()
                eval_results['supps'] = available_supp.data.numpy()
                eval_results['snr'] = available_snr.data.numpy()
                sio.savemat(eval_results_path, eval_results)
            eval_results = {}
            eval_results['mse_wiener'] = mse_curr_net_per_supp_wiener.data.numpy()
            eval_results['mse_nn'] = mse_curr_net_per_supp_nn.data.numpy()
            eval_results['supps'] = available_supp.data.numpy()
            eval_results['snr'] = available_snr.data.numpy()
            sio.savemat(eval_results_path, eval_results)
            print("--- %s seconds ---" % (time.time() - start_time))
            ind_snr += 1
        eval_results = {}
        eval_results['mse_wiener'] = mse_curr_net_per_supp_wiener.data.numpy()
        eval_results['mse_nn'] = mse_curr_net_per_supp_nn.data.numpy()
        eval_results['supps'] = available_supp.data.numpy()
        eval_results['snr'] = available_snr.data.numpy()
        sio.savemat(eval_results_path, eval_results)
        print("--- %s seconds ---" % (time.time() - start_time))
        ind_lambda += 1
    


