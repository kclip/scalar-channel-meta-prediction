import torch
import numpy as np
import scipy.io as sio

class Doppler_Jake_dataloader_standard:
    def __init__(self, num_supp, num_query, if_jakes_rounded_toy, if_meta_tr, num_meta_tr_tasks, num_meta_te_tasks, num_paths, noise_var, noise_var_meta, if_simple_multivariate_extension, toy_long_term): # num_supp working as training data, num_query working as test data for conven. learning
        self.num_supp = num_supp 
        self.num_query= num_query
        self.num_paths = num_paths
        self.noise_var = noise_var
        self.noise_var_meta = noise_var_meta
        #print('noise meta actual', self.noise_var_meta)
        # num_paths are in complex dim.  -> 1 in scalar case

        if if_jakes_rounded_toy:
            raise NotImplementedError
        else:              
            saved_path_meta_tr = '../../../../generated_channels/3gpp_meta_training_offline.mat' # use first 80
            total_channels_meta_tr = sio.loadmat(saved_path_meta_tr) 
            saved_path_meta_te = '../../../../generated_channels/3gpp_meta_training_offline.mat' # use last 20
            total_channels_meta_te = sio.loadmat(saved_path_meta_te) 
        assert num_paths == 1

        if if_meta_tr:
            self.meta_tr_dict = {}
            self.meta_tr_dict_noiseless = {}
            for ind_task in range(num_meta_tr_tasks):
                if if_jakes_rounded_toy:
                    raise NotImplementedError
                else:
                    supp_query_samples_total = torch.from_numpy(total_channels_meta_tr['meta_te_dataset'][ind_task,0])
                ## simple multivariate!!
                if toy_long_term is not None:
                    supp_query_samples_total = supp_query_samples_total @ torch.transpose(toy_long_term[ind_task],1,0)
                else:
                    pass
                ## add noise!
                curr_task_noise = torch.randn(supp_query_samples_total.shape, dtype=torch.cdouble) # CN(0,1)
                curr_task_noise *= np.sqrt(self.noise_var_meta) # if var = 0 then noise is considerd during meta-learning or joint learning just use perfect samples with meta-te noisy samples jointly
                supp_query_samples_total_noisy = supp_query_samples_total + curr_task_noise
                self.meta_tr_dict[ind_task] = supp_query_samples_total_noisy
                self.meta_tr_dict_noiseless[ind_task] = supp_query_samples_total
        else:
            self.meta_te_dict = {}
            self.meta_te_dict_without_noise = {} # for only computing MSE performance
            self.gt_R = {}
            for ind_task in range(num_meta_te_tasks):
                if if_jakes_rounded_toy:
                    raise NotImplementedError
                else:
                    supp_query_samples_total = torch.from_numpy(total_channels_meta_te['meta_te_dataset'][ind_task+80,0])
                    self.ar_from_psd = None
                if toy_long_term is not None:
                    supp_query_samples_total = supp_query_samples_total @ torch.transpose(toy_long_term[ind_task],1,0)
                else:
                    pass
                curr_task_noise = torch.randn(supp_query_samples_total.shape, dtype=torch.cdouble) # CN(0,1)
                curr_task_noise *= np.sqrt(self.noise_var)
                supp_query_samples_total_noisy = supp_query_samples_total + curr_task_noise
                self.meta_te_dict[ind_task] = supp_query_samples_total_noisy
                self.meta_te_dict_without_noise[ind_task] = supp_query_samples_total
                self.gt_R[ind_task] = self.ar_from_psd

    def get_supp_samples_total(self, if_meta_tr, ind_task, num_supp=None):
        if num_supp is None:
            num_supp_curr = self.num_supp # for meta-tr
        else:
            num_supp_curr = num_supp # for varying exp. over num_supp
        if if_meta_tr:
            return self.meta_tr_dict[ind_task][:num_supp_curr]
        else: # should not use query set (test channels)
            assert num_supp_curr + self.num_query <= self.meta_te_dict[ind_task].shape[0] # total samples # unless we are using more pilots than available!
            return self.meta_te_dict[ind_task][:num_supp_curr]
    
    def get_supp_samples_total_noiseless(self, if_meta_tr, ind_task, num_supp=None): # for noise-free joint
        if num_supp is None:
            num_supp_curr = self.num_supp # for meta-tr
        else:
            num_supp_curr = num_supp # for varying exp. over num_supp
        if if_meta_tr:
            return self.meta_tr_dict_noiseless[ind_task][:num_supp_curr]
        else: # should not use query set (test channels)
            raise NotImplementedError # should not be used here! # noiseless only for meta-tr
            assert num_supp_curr + self.num_query <= self.meta_te_dict_noiseless[ind_task].shape[0] # total samples # unless we are using more pilots than available!
            return self.meta_te_dict[ind_task][:num_supp_curr]

    def get_all_possible_querys(self, window_length, lag, ind_mte_task): # since we are not using in-batch compute. it is fine to use single mb to get the test result # mb only works for averaging grads
        query_samples_total = self.meta_te_dict[ind_mte_task][-self.num_query:] # use last self.num_query channels for test
        query_samples_total_without_noise = self.meta_te_dict_without_noise[ind_mte_task][-self.num_query:]
        all_possible_querys = []
        total_queries = len(query_samples_total)-window_length-lag+1
        for curr_ind in range(total_queries):
            curr_input_mb = query_samples_total[curr_ind:curr_ind+window_length]
            curr_output_mb = query_samples_total[curr_ind+window_length-1+lag] # noisy target which may not be used at all...
            curr_output_mb_without_noise = query_samples_total_without_noise[curr_ind+window_length-1+lag]
            all_possible_querys.append([curr_input_mb, curr_output_mb, curr_output_mb_without_noise])
        return all_possible_querys

    