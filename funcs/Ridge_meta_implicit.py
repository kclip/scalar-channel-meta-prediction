import torch
import numpy as np
import scipy.io as sio
from numpy.linalg import inv

class Ridge_meta_implicit:
    def __init__(self, beta_in_seq_total, supp_mb, window_length, lag, ar_from_psd, noise_var_meta, if_meta_noise_free, lambda_coeff=None, normalize_factor=None, total_iter_implicit=15000, lr_implicit=0.0001): 
        self.window_length = window_length
        self.lag = lag
        supp_mb = supp_mb-window_length-lag+1 # actual number of supp_mb -- actual number of training pairs -> this is L^tr -- before supp_mb is number of total available channels
        self.supp_mb = supp_mb
        self.lambda_coeff = lambda_coeff
        self.beta_in_seq_total = beta_in_seq_total
        if if_meta_noise_free:
            raise NotImplementedError # deprecated
            self.noise_var_meta = noise_var_meta # unless if_meta_noise_free, the noise is already being added prior
        else:
            self.noise_var_meta = 0.0  # if force this to be 0 regardless of input, we will just use pure meta and then directly use to meta-te .. similar to joint but quite naive..
        self.normalize_factor = normalize_factor
        print('noise working at meta phase by addition at ridge meta phase', self.noise_var_meta)
        print('meta training supp size: ', supp_mb)
        self.lambda_dict = {}
        if self.lambda_coeff is None:
            for i in range(13):
                self.lambda_dict[i] = 10**(i-6)
        else:
            self.lambda_dict[0] = self.lambda_coeff # no grid search!! use given lambda!
        assert self.noise_var_meta == 0
        assert self.normalize_factor == 1
        self.total_iter_implicit = total_iter_implicit
        self.lr_implicit = lr_implicit

    def grid_search(self, fixed_lambda_value):
        if fixed_lambda_value is not None:
            assert fixed_lambda_value == self.lambda_coeff 
            curr_lambda = self.lambda_coeff 
            self.curr_lambda = torch.tensor(curr_lambda, dtype=torch.double, requires_grad=False)
            curr_common_mean = self.v_bar_update()
            return curr_common_mean.detach(), curr_lambda
        else:
            print('currenly only consider fixed lambda for online case')
            raise NotImplementedError

    def v_bar_update(self):
        ### get Z_tr and Z_te
        X_tr_dict = {}
        Y_tr_dict = {}
        X_te_dict = {}
        Y_te_dict = {}
        total_possible_supp_query_pairs = 1
        for ind_task in range(len(self.beta_in_seq_total)):
            curr_beta = self.beta_in_seq_total[ind_task]
            for ind_possible_slot in range(total_possible_supp_query_pairs):
                prev_channels = []
                future_channels = []
                for ind_row in range(self.supp_mb):
                    curr_row = []
                    supp_start_ind = ind_possible_slot + self.window_length-1 + ind_row
                    for ind_col in range(self.window_length):
                        curr_channel = Herm(curr_beta[supp_start_ind-ind_col])
                        curr_row.append(curr_channel.unsqueeze(dim=0))
                    curr_row = torch.cat((curr_row), dim=1)
                    prev_channels.append(curr_row)
                    future_channel = Herm(curr_beta[supp_start_ind+self.lag])
                    future_channels.append(future_channel.unsqueeze(dim=0))
                X = torch.cat(prev_channels, dim=0)
                Y = torch.cat(future_channels, dim=0)
                noise_X = torch.randn(X.shape, dtype=torch.cdouble)*np.sqrt(self.noise_var_meta) 
                X += noise_X
                if self.normalize_factor is not None:
                    pass
                else: # unless specified, use normal approach
                    self.normalize_factor = self.supp_mb

                X_tr_dict[ind_task] = X
                Y_tr_dict[ind_task] = Y
                
                start_ind_for_XY = ind_possible_slot + self.window_length-1
                last_ind_for_XY = ind_possible_slot + self.window_length-1 + self.supp_mb -1
                query_start_ind = 0 + self.window_length-1 # from very first, except for the supp channels
                prev_channels = []
                future_channels = []
                while query_start_ind <= len(self.beta_in_seq_total[0])-1-self.lag:
                    if start_ind_for_XY <= query_start_ind <= last_ind_for_XY:
                        query_start_ind += 1
                    else:
                        x_tmp = []
                        y_tmp = []
                        for ind_col in range(self.window_length):
                            curr_channel = Herm(curr_beta[query_start_ind-ind_col]) # row vector
                            x_tmp.append(curr_channel.unsqueeze(dim=0))
                        x = torch.cat((x_tmp), dim=1) # one row
                        noise_x = torch.randn(x.shape, dtype=torch.cdouble)*np.sqrt(self.noise_var_meta)
                        x += noise_x # may be disregard this part also?
                        future_channel = Herm(curr_beta[query_start_ind+self.lag])
                        y = future_channel.unsqueeze(dim=0)
                        prev_channels.append(x)
                        future_channels.append(y)
                        query_start_ind += 1
                X_te = torch.cat(prev_channels, dim=0)
                Y_te = torch.cat(future_channels, dim=0)
                X_te_dict[ind_task] = X_te
                Y_te_dict[ind_task] = Y_te
        ### now do implicit gradient
        # start from 0 matrix
        v_bar_curr = torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.cdouble)
        for ind_iter in range(self.total_iter_implicit):
            curr_meta_grad = torch.zeros(v_bar_curr.shape, dtype=torch.cdouble) # zero_grad
            for ind_task in range(len(self.beta_in_seq_total)):
                # get V^* (fixed point) based on v_bar_curr
                C_inv = torch.inverse((Herm(X_tr_dict[ind_task])@X_tr_dict[ind_task]) + self.curr_lambda*torch.eye(X_tr_dict[ind_task].shape[1])) #self.lambda_coeff*np.eye(window_length))
                rhs_for_v_star = Herm(X_tr_dict[ind_task])@Y_tr_dict[ind_task] + self.curr_lambda*v_bar_curr
                v_star = C_inv@rhs_for_v_star

                tmp = (1/self.curr_lambda)*Herm(X_tr_dict[ind_task])@X_tr_dict[ind_task] + torch.eye(X_tr_dict[ind_task].shape[1])
                meta_grad_lhs = torch.inverse(tmp)
                meta_grad_rhs = 2*Herm(X_te_dict[ind_task])@(X_te_dict[ind_task]@v_star - Y_te_dict[ind_task])
                curr_task_meta_grad = meta_grad_lhs@meta_grad_rhs
                curr_meta_grad += curr_task_meta_grad
            # update
            v_bar_curr -= self.lr_implicit * curr_meta_grad
        return v_bar_curr

def Herm(vector):
    return np.transpose(np.conjugate(vector))

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)

