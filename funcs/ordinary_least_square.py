import torch
import numpy as np
from numpy.linalg import inv

class OLS:
    def __init__(self, beta_in_seq, window_length, lag, ar_from_psd, if_ols_compute_via_direct_avg = False, lambda_coeff=None, W_common_mean=None, noise_var=None, toy_long_term=None): 
        assert ar_from_psd == None # ar_from_psd deprecated
        self.beta_in_seq_complex = beta_in_seq[0]
        self.beta_in_seq_complex_noiseless = beta_in_seq[1] # except joint pure, same as self.beta_in_seq_complex
        self.autocorr_time_diff_dict = None # we do not use this but direct computation of X'X
        self.lambda_coeff = lambda_coeff
        self.if_ols_compute_via_direct_avg = if_ols_compute_via_direct_avg
        self.noise_var = noise_var
        self.toy_long_term = toy_long_term
        if W_common_mean is None:
            self.W_common_mean = 0
        else:
            self.W_common_mean = W_common_mean
    def coefficient_compute(self, window_length, lag, if_ridge, perf_stat=None, normalize_factor=None, x_te=None): #x_te is used for adapting lambda for meta-learning
        curr_beta = self.beta_in_seq_complex
        curr_beta_noiseless = self.beta_in_seq_complex_noiseless
        num_paths = curr_beta.shape[1]
        if perf_stat is not None:
            if len(perf_stat) == 100: # R value itself
                ar_from_psd = perf_stat
                R_dict = {}
                for time_diff in range(len(ar_from_psd)):  
                    R_dict[time_diff] = ar_from_psd[time_diff]
            else:
                raise NotImplementedError  
            H_herm_H = torch.zeros(num_paths*window_length,num_paths*window_length,dtype=torch.cdouble)
            H_herm_h = torch.zeros(num_paths*window_length,num_paths,dtype=torch.cdouble)
            if self.toy_long_term is not None: # deprecated
                tmp_diag = torch.zeros((self.toy_long_term.shape[0], self.toy_long_term.shape[0]), dtype=torch.cdouble)
                for ind_col in range(window_length):
                    for ind_row in range(window_length):
                        if ind_row > ind_col:
                            curr_R = Herm(R_dict[ind_row-ind_col])
                        else:
                            curr_R = R_dict[ind_col-ind_row]
                        if ind_row == ind_col:
                            for ind in range(self.toy_long_term.shape[0]):
                                tmp_diag[ind, ind] = self.noise_var
                            H_herm_H[ind_row*num_paths:ind_row*num_paths+num_paths, ind_col*num_paths:ind_col*num_paths+num_paths] = curr_R * Herm_tensor(self.toy_long_term)@self.toy_long_term   + tmp_diag#*0.00001 # braodcasting if R is matrix
                        else:
                            H_herm_H[ind_row*num_paths:ind_row*num_paths+num_paths, ind_col*num_paths:ind_col*num_paths+num_paths] = curr_R * Herm_tensor(self.toy_long_term)@self.toy_long_term #+ self.noise_var # braodcasting if R is matrix
                for ind_col in range(1):
                    for ind_row in range(window_length):
                        curr_R = Herm(R_dict[ind_row-ind_col+lag])
                        H_herm_h[ind_row*num_paths:ind_row*num_paths+num_paths, ind_col*num_paths:ind_col*num_paths+num_paths] = curr_R * Herm_tensor(self.toy_long_term)@self.toy_long_term #+ self.noise_var
            else:
                for ind_col in range(window_length):
                    for ind_row in range(window_length):
                        if ind_row > ind_col:
                            curr_R = Herm(R_dict[ind_row-ind_col])
                        else:
                            curr_R = R_dict[ind_col-ind_row]
                        if ind_row == ind_col:
                            H_herm_H[ind_row*num_paths:ind_row*num_paths+num_paths, ind_col*num_paths:ind_col*num_paths+num_paths] = curr_R + self.noise_var#*0.00001 # braodcasting if R is matrix
                        else:
                            H_herm_H[ind_row*num_paths:ind_row*num_paths+num_paths, ind_col*num_paths:ind_col*num_paths+num_paths] = curr_R #+ self.noise_var # braodcasting if R is matrix
                for ind_col in range(1):
                    for ind_row in range(window_length):
                        curr_R = Herm(R_dict[ind_row-ind_col+lag])
                        H_herm_h[ind_row*num_paths:ind_row*num_paths+num_paths, ind_col*num_paths:ind_col*num_paths+num_paths] = curr_R #+ self.noise_var
        else:
            supp_mb = len(curr_beta)-window_length-lag+1 # number of total samples in the beta
            prev_channels = []
            future_channels = []
            if normalize_factor is not None:
                #normalize_factor = 1
                pass
            else: # unless specified, use normal approach
                normalize_factor = supp_mb
            for ind_row in range(supp_mb): # use whole
                curr_row = []
                rand_start_ind = ind_row + window_length-1
                for ind_col in range(window_length):
                    curr_channel = Herm(curr_beta[rand_start_ind-ind_col]) # input should be noisy
                    curr_row.append(curr_channel.unsqueeze(dim=0))
                curr_row = torch.cat((curr_row), dim=1)
                prev_channels.append(curr_row)
                future_channel = Herm(curr_beta_noiseless[rand_start_ind+lag]) # only noiseless when args.if_joint_noise_free and in that case, only for meta-training set 
                future_channels.append(future_channel.unsqueeze(dim=0))
            X = torch.cat(prev_channels, dim=0)
            Y = torch.cat(future_channels, dim=0)
            if if_ridge:
                H_herm_H = (Herm(X)@X)/normalize_factor 
                H_herm_h = (Herm(X)@Y)/normalize_factor
            else:
                if (supp_mb < X.shape[1]):
                    # if toy_long_term is not None, we are forcing multivariate from scalar, so we cannot have inverse of H_herm_H
                    #print('under determined system!!, consider least-norm solution!')
                    # due to under detemined system, we consider elast-norm solution instead
                    H_H_herm = (X@Herm(X))/normalize_factor 
                else:
                    H_herm_H = (Herm(X)@X)/normalize_factor # instead Herm(X)@X it would be better to directly average 
                    H_herm_h = (Herm(X)@Y)/normalize_factor # instead Herm(X)@Y it would be better to directly average 
        if if_ridge:
            H_herm_H += self.lambda_coeff * np.eye(H_herm_H.shape[0])
        else:
            pass
        if if_ridge:
            H_herm_h += self.lambda_coeff * self.W_common_mean 
        else:
            pass
        if perf_stat is not None:
            self.W = Herm(torch.from_numpy(inv(H_herm_H)) @ H_herm_h)
        else:
            if if_ridge:
                self.W = Herm(torch.from_numpy(inv(H_herm_H)) @ H_herm_h)
            else:
                if (supp_mb < X.shape[1]): ## need to consider gen inv also for the underdetermined case!!! # maybe if we change X.shape[1] to window_length, we may not have to worry about this!!
                    self.W = Herm(Herm(X) @inv(H_H_herm) @ Y)
                else:
                    self.W = Herm(torch.from_numpy(inv(H_herm_H)) @ H_herm_h)
    def prediction(self, input_seq, window_length, lag):
        assert input_seq.shape[0] == window_length
        input_seq_complex = input_seq
        num_complex_paths = input_seq_complex.shape[1]
        pred_channel = 0
        for ind_window in range(window_length):
            ind_window_reverse = window_length-1-ind_window
            pred_channel += self.W[:, ind_window*num_complex_paths:ind_window*num_complex_paths+num_complex_paths] @ input_seq_complex[ind_window_reverse].unsqueeze(dim=1).numpy()        
        return pred_channel

def Herm(vector):
    return np.transpose(np.conjugate(vector))

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)