import torch
import numpy as np
from dataloader.data_loader_jakes_rounded import Doppler_Jake_dataloader_jakes_rounded
from dataloader.data_loader_standard_offline import Doppler_Jake_dataloader_standard
from dataloader.data_loader_standard_online import Doppler_Jake_dataloader_standard_online
from funcs.Ridge_meta import Ridge_meta
from funcs.Ridge_meta_EP import Ridge_meta_EP
from funcs.Ridge_meta_implicit import Ridge_meta_implicit
from funcs.ordinary_least_square import OLS
import copy

def one_mc_trial(args, curr_dir, num_supp, num_query, velocity_kmph, ind_mc):
    loss = torch.nn.MSELoss(reduction='sum')
    curr_supp_best_mse_nn = 999999999999
    if args.Jake_dataloader is None:
        if args.fading_mode == 1:
            Jake_dataloader = Doppler_Jake_dataloader_jakes_rounded(num_supp=num_supp, num_query=num_query, if_jakes_rounded_toy=args.if_jakes_rounded_toy, 
                                                                        if_meta_tr=False, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks, num_paths=args.num_paths, noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension, toy_long_term = args.toy_long_term)
        elif args.fading_mode == 2:
            Jake_dataloader = Doppler_Jake_dataloader_standard(num_supp=num_supp, num_query=num_query, if_jakes_rounded_toy=args.if_jakes_rounded_toy, 
                                                                        if_meta_tr=False, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks, num_paths=args.num_paths, noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension, toy_long_term = args.toy_long_term)
        elif args.fading_mode == 3: # online standard
            Jake_dataloader = Doppler_Jake_dataloader_standard_online(num_supp=num_supp, num_query=num_query, if_jakes_rounded_toy=args.if_jakes_rounded_toy, 
                                                                        if_meta_tr=False, ind_round = args.ind_round, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks, num_paths=args.num_paths, noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension, toy_long_term = args.toy_long_term, ind_mc=ind_mc, if_for_online_meta_test_do_offline_meta_train=args.if_for_online_meta_test_do_offline_meta_train, total_round_online=args.total_round_online)
        else:
            raise NotImplementedError
        args.Jake_dataloader = Jake_dataloader # since only supp is different for offline case -- we can assign is directly via get_supp_samples_total
        # for online, we need to renew always.
    else:
        Jake_dataloader = args.Jake_dataloader
    if args.Jake_dataloader_meta is None:
        if args.meta_training_samples_per_task is None:
            meta_training_total_samples_per_tasks = num_supp + 100  #L^tr + L^te
        else:
            meta_training_total_samples_per_tasks = args.meta_training_samples_per_task
        if args.fading_mode == 1:
            Jake_dataloader_meta =  Doppler_Jake_dataloader_jakes_rounded(num_supp=meta_training_total_samples_per_tasks, num_query= 0, if_jakes_rounded_toy=args.if_jakes_rounded_toy, if_meta_tr=True, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks, num_paths=args.num_paths, noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension, toy_long_term = args.toy_long_term)
        elif args.fading_mode == 2:
            Jake_dataloader_meta =  Doppler_Jake_dataloader_standard(num_supp=meta_training_total_samples_per_tasks, num_query= 0, if_jakes_rounded_toy=args.if_jakes_rounded_toy, if_meta_tr=True, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks, num_paths=args.num_paths, noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension, toy_long_term = args.toy_long_term)
        elif args.fading_mode == 3:
            Jake_dataloader_meta =  Doppler_Jake_dataloader_standard_online(num_supp=meta_training_total_samples_per_tasks, num_query= 0, if_jakes_rounded_toy=args.if_jakes_rounded_toy, if_meta_tr=True, ind_round = args.ind_round, num_meta_tr_tasks=args.num_meta_tr_tasks, num_meta_te_tasks=args.num_meta_te_tasks, num_paths=args.num_paths, noise_var = args.noise_var, noise_var_meta = args.noise_var_meta, if_simple_multivariate_extension=args.if_simple_multivariate_extension, toy_long_term = args.toy_long_term, ind_mc=ind_mc, if_for_online_meta_test_do_offline_meta_train=args.if_for_online_meta_test_do_offline_meta_train, total_round_online=args.total_round_online)
        else:
            raise NotImplementedError
        args.Jake_dataloader_meta = Jake_dataloader_meta # it cannot directly controlled via get_supp_samples_total when num_supp changes -- since actual dataset is different. we need to redefine per num_supp.
    else:
        Jake_dataloader_meta = args.Jake_dataloader_meta
    ## snr computation by samples
    snr_curr = -999
    # WF
    if args.linear_ridge_mode == 0: # 0: meta, 1: joint, 2: tfs
        # meta-learning!!!
        beta_in_seq_total = []
        for ind_task_mtr in range(args.num_meta_tr_tasks):
            curr_mtr_data = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)
            beta_in_seq_total.append(curr_mtr_data)
        if args.ridge_meta is None:
            if args.if_mtr_fix_supp:
                num_supp_ridge_meta = args.meta_training_fixed_supp # not the number of training samples (L^tr) -- this is number of total availble samples = L^tr + N+delta-1
            else:
                num_supp_ridge_meta = num_supp
            ridge_meta = Ridge_meta(beta_in_seq_total=beta_in_seq_total, supp_mb=num_supp_ridge_meta, window_length=args.window_length, lag=args.lag, ar_from_psd=None, noise_var_meta= args.noise_var, if_meta_noise_free = args.if_meta_noise_free, lambda_coeff=args.ridge_lambda_coeff, normalize_factor=args.normalize_factor_meta_ridge) # use noise_var_meta as same as that will be used at meta-te
            adapted_common_mean, adapted_lambda = ridge_meta.grid_search(args.fixed_lambda_value, args.prev_v_bar_for_online_for_closed_form)
            ridge_lambda_coeff = float(adapted_lambda)
            args.ridge_meta = ridge_meta # deprecated..
            args.prev_v_bar_for_online_for_closed_form = adapted_common_mean
            args.adapted_common_mean = adapted_common_mean
            args.adapted_lambda = adapted_lambda
        else:
            if args.if_for_online_meta_test_do_offline_meta_train:
                adapted_common_mean = args.adapted_common_mean
                adapted_lambda = args.adapted_lambda # we are not adpating lambda -- just for writing consistency.
                ridge_lambda_coeff = float(adapted_lambda)
            else:
                raise NotImplementedError # now we are only considering adapting lambda for each case!
        mse_wiener = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            if args.if_perfect_statistics:
                ar_from_psd = Jake_dataloader.ar_from_psd # deprecated
                perf_stat = (Jake_dataloader.meta_te_dict_true_w[ind_task_mte], Jake_dataloader.delta_t)
            else:
                ar_from_psd = None
                perf_stat = None

            if args.if_ridge:
                lambda_coeff = ridge_lambda_coeff # may be adapted
            else:
                lambda_coeff = None
            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (curr_training_seq, curr_training_seq), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, W_common_mean=adapted_common_mean, noise_var=args.noise_var, toy_long_term=args.toy_long_term)
            else:
                raise NotImplementedError
            WF.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, perf_stat=perf_stat, normalize_factor=args.normalize_factor)
            mse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks

    elif args.linear_ridge_mode == 1:
        mse_wiener = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            curr_training_seq_noiseless = curr_training_seq # use noisy for meta-te
            # concatenate all available data
            beta_in_seq_total = []
            for ind_task_mtr in range(args.num_meta_tr_tasks):
                curr_mtr_data = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)
                beta_in_seq_total.append(curr_mtr_data)
            beta_in_seq_total.append(curr_training_seq)
            beta_in_seq_joint = torch.cat(beta_in_seq_total, dim=0)
            # concatenate all available data - noiseless
            beta_in_seq_total_noiseless = []
            for ind_task_mtr in range(args.num_meta_tr_tasks):
                if args.if_joint_noise_free:
                    curr_mtr_data_noiseless = Jake_dataloader_meta.get_supp_samples_total_noiseless(if_meta_tr=True, ind_task=ind_task_mtr)
                else:
                    curr_mtr_data_noiseless = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)
                beta_in_seq_total_noiseless.append(curr_mtr_data_noiseless)
            beta_in_seq_total_noiseless.append(curr_training_seq_noiseless) # in fact noisy for meta-te task
            beta_in_seq_joint_noiseless = torch.cat(beta_in_seq_total_noiseless, dim=0)

            beta_in_seq_joint_total = (beta_in_seq_joint, beta_in_seq_joint_noiseless)

            if args.if_ridge:
                lambda_coeff = args.ridge_lambda_coeff
            else:
                lambda_coeff = None
            if args.if_wiener_filter_as_ols:
                WF_joint = OLS(beta_in_seq = beta_in_seq_joint_total, window_length=args.window_length, lag=args.lag, ar_from_psd=None, lambda_coeff = lambda_coeff, noise_var=args.noise_var, toy_long_term=args.toy_long_term)
            else:
                raise NotImplementedError
            WF_joint.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, normalize_factor=args.normalize_factor)
            W_joint = (WF_joint.W)
            if args.if_perfect_statistics:
                ar_from_psd = Jake_dataloader.ar_from_psd
            else:
                ar_from_psd = None
            if args.if_ridge:
                lambda_coeff = args.ridge_lambda_coeff
            else:
                lambda_coeff = None
            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (None, None), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, noise_var=args.noise_var, toy_long_term=args.toy_long_term)
            else:
                raise NotImplementedError
            WF.W = W_joint # use obtained W from joint training instead
            mse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                # maybe we need to squeeze -> win, mb_size, num_paths
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks
    elif args.linear_ridge_mode == 2:
        mse_wiener = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            if args.if_perfect_statistics:
                ar_from_psd = None
                if args.fading_mode == 0:
                    raise NotImplementedError # deprecated
                    perf_stat = (Jake_dataloader.meta_te_dict_true_w[ind_task_mte], Jake_dataloader.delta_t)
                elif args.fading_mode == 1:
                    perf_stat = Jake_dataloader.gt_R[ind_task_mte]
                else:
                    raise NotImplementedError
            else:
                ar_from_psd = None
                perf_stat = None
            if args.if_ridge:
                lambda_coeff = args.ridge_lambda_coeff
            else:
                lambda_coeff = None
            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (curr_training_seq, curr_training_seq), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, noise_var=args.noise_var, toy_long_term=args.toy_long_term)
            else:
                raise NotImplementedError
            WF.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, perf_stat=perf_stat, normalize_factor=args.normalize_factor)
            mse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks
    elif args.linear_ridge_mode == 6:
        # meta-learning with implicit gradient!!
        beta_in_seq_total = []
        for ind_task_mtr in range(args.num_meta_tr_tasks):
            curr_mtr_data = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)
            beta_in_seq_total.append(curr_mtr_data)
        if args.ridge_meta is None:
            if args.if_mtr_fix_supp:
                num_supp_ridge_meta = args.meta_training_fixed_supp # not the number of training samples (L^tr) -- this is number of total availble samples = L^tr + N+delta-1
            else:
                num_supp_ridge_meta = num_supp
            ridge_meta = Ridge_meta_implicit(beta_in_seq_total=beta_in_seq_total, supp_mb=num_supp_ridge_meta, window_length=args.window_length, lag=args.lag, ar_from_psd=None, noise_var_meta= args.noise_var, if_meta_noise_free = args.if_meta_noise_free, lambda_coeff=args.ridge_lambda_coeff, normalize_factor=args.normalize_factor_meta_ridge) # use noise_var_meta as same as that will be used at meta-te
            adapted_common_mean, adapted_lambda = ridge_meta.grid_search(args.fixed_lambda_value)
            ridge_lambda_coeff = float(adapted_lambda)
            args.ridge_meta = ridge_meta
        else:
            raise NotImplementedError # now we are only considering adapting lambda for each case!
            ridge_meta = args.ridge_meta
        mse_wiener = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            if args.if_perfect_statistics:
                ar_from_psd = Jake_dataloader.ar_from_psd # deprecated
                perf_stat = (Jake_dataloader.meta_te_dict_true_w[ind_task_mte], Jake_dataloader.delta_t)
            else:
                ar_from_psd = None
                perf_stat = None
            if args.if_ridge:
                lambda_coeff = ridge_lambda_coeff # may be adapted
            else:
                lambda_coeff = None
            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (curr_training_seq, curr_training_seq), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, W_common_mean=adapted_common_mean, noise_var=args.noise_var, toy_long_term=args.toy_long_term)
            else:
                raise NotImplementedError
            WF.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, perf_stat=perf_stat, normalize_factor=args.normalize_factor)
            mse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks
    elif args.linear_ridge_mode == 7:
        # meta-learning with EP!!!
        beta_in_seq_total = []
        for ind_task_mtr in range(args.num_meta_tr_tasks):
            curr_mtr_data = Jake_dataloader_meta.get_supp_samples_total(if_meta_tr=True, ind_task=ind_task_mtr)
            beta_in_seq_total.append(curr_mtr_data)
        if args.ridge_meta is None:
            if args.if_mtr_fix_supp:
                num_supp_ridge_meta = args.meta_training_fixed_supp # not the number of training samples (L^tr) -- this is number of total availble samples = L^tr + N+delta-1
            else:
                num_supp_ridge_meta = num_supp
            ridge_meta = Ridge_meta_EP(beta_in_seq_total=beta_in_seq_total, supp_mb=num_supp_ridge_meta, window_length=args.window_length, lag=args.lag, ar_from_psd=None, noise_var_meta= args.noise_var, if_meta_noise_free = args.if_meta_noise_free, lambda_coeff=args.ridge_lambda_coeff, normalize_factor=args.normalize_factor_meta_ridge, lr_EP = args.lr_EP) # use noise_var_meta as same as that will be used at meta-te
            adapted_common_mean, adapted_lambda, momentum = ridge_meta.grid_search(args.fixed_lambda_value, args.v_bar_curr, args.momentum, args.momentum_coeff)
            args.v_bar_curr = adapted_common_mean #this enables online learning
            ridge_lambda_coeff = float(adapted_lambda)
            args.ridge_meta = ridge_meta
            args.momentum = momentum
        else:
            raise NotImplementedError # now we are only considering adapting lambda for each case!
            ridge_meta = args.ridge_meta
        mse_wiener = 0
        for ind_task_mte in range(args.num_meta_te_tasks):   
            curr_training_seq = Jake_dataloader.get_supp_samples_total(if_meta_tr=False, ind_task=ind_task_mte, num_supp=num_supp)  
            if args.if_perfect_statistics:
                ar_from_psd = Jake_dataloader.ar_from_psd # deprecated
                perf_stat = (Jake_dataloader.meta_te_dict_true_w[ind_task_mte], Jake_dataloader.delta_t)
            else:
                ar_from_psd = None
                perf_stat = None

            if args.if_ridge:
                lambda_coeff = ridge_lambda_coeff # may be adapted
            else:
                lambda_coeff = None
            if args.if_wiener_filter_as_ols:
                WF = OLS(beta_in_seq = (curr_training_seq, curr_training_seq), window_length=args.window_length, lag=args.lag, ar_from_psd=ar_from_psd, lambda_coeff = lambda_coeff, W_common_mean=adapted_common_mean, noise_var=args.noise_var, toy_long_term=args.toy_long_term)
            else:
                raise NotImplementedError
            WF.coefficient_compute(window_length=args.window_length, lag=args.lag, if_ridge=args.if_ridge, perf_stat=perf_stat, normalize_factor=args.normalize_factor)
            mse_wiener_curr_mte_task = 0
            all_possible_test = Jake_dataloader.get_all_possible_querys(window_length=args.window_length, lag=args.lag, ind_mte_task=ind_task_mte)
            for test_ch in all_possible_test:
                pred_ch = WF.prediction(input_seq = test_ch[0], window_length=args.window_length, lag=args.lag)
                pred_ch_in_real = torch.flatten(torch.view_as_real(pred_ch))
                true_ch_in_real = torch.flatten(torch.view_as_real(test_ch[2]))
                mse_wiener_curr_mte_task += loss(pred_ch_in_real, true_ch_in_real) # compare to perfect channel
            mse_wiener_curr_mte_task /= len(all_possible_test)
            mse_wiener += mse_wiener_curr_mte_task
        mse_wiener /= args.num_meta_te_tasks
    else:
        raise NotImplementedError

    return snr_curr, mse_wiener, curr_supp_best_mse_nn


