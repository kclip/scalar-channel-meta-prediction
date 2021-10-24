import torch
import numpy as np
import scipy.io as sio
from numpy.linalg import inv

class Ridge_meta_EP:
    def __init__(self, beta_in_seq_total, supp_mb, window_length, lag, ar_from_psd, noise_var_meta, if_meta_noise_free, lambda_coeff=None, normalize_factor=None, total_iter_EP=200, lr_EP=None, coeff_EP=0.01): 
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
        self.lambda_dict = {}
        if self.lambda_coeff is None:
            for i in range(13):
                self.lambda_dict[i] = 10**(i-6)
        else:
            self.lambda_dict[0] = self.lambda_coeff # no grid search!! use given lambda!
        assert self.noise_var_meta == 0
        assert self.normalize_factor == 1
        self.total_iter_EP = total_iter_EP
        self.lr_EP = lr_EP
        self.coeff_EP = coeff_EP


    def grid_search(self, fixed_lambda_value, v_bar_curr, momentum, momentum_coeff):
        if fixed_lambda_value is not None:
            assert fixed_lambda_value == self.lambda_coeff 
            curr_lambda = self.lambda_coeff 
            self.curr_lambda = torch.tensor(curr_lambda, dtype=torch.double, requires_grad=False)
            curr_common_mean, momentum = self.v_bar_update(v_bar_curr, momentum, momentum_coeff)
            return curr_common_mean, curr_lambda, momentum
        else:
            print('currenly only consider fixed lambda for online case')
            raise NotImplementedError

    def v_bar_update(self, v_bar_curr, momentum, momentum_coeff):
        total_possible_supp_query_pairs = 1
        ### get Z_tr and Z_te
        X_tr_dict = {}
        Y_tr_dict = {}
        X_te_dict = {}
        Y_te_dict = {}
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
                        x += noise_x # deprecated part
                        future_channel = Herm(curr_beta[query_start_ind+self.lag])
                        y = future_channel.unsqueeze(dim=0)
                        prev_channels.append(x)
                        future_channels.append(y)
                        query_start_ind += 1
                X_te = torch.cat(prev_channels, dim=0)
                Y_te = torch.cat(future_channels, dim=0)
                X_te_dict[ind_task] = X_te
                Y_te_dict[ind_task] = Y_te
        ### now do EP
        if len(self.beta_in_seq_total) == 0:
            pass
        else:
            # start from 0 matrix
            if v_bar_curr is None:
                v_bar_curr = torch.zeros((X.shape[1], Y.shape[1]), dtype=torch.cdouble)
            else: # use prev. round's v_bar to keep updating in online manner
                pass
            for ind_iter in range(self.total_iter_EP):
                curr_meta_grad = torch.zeros(v_bar_curr.shape, dtype=torch.cdouble) # zero_grad
                for ind_task in range(len(self.beta_in_seq_total)):
                    # get V^* (fixed point) based on v_bar_curr
                    C_inv = torch.inverse((Herm(X_tr_dict[ind_task])@X_tr_dict[ind_task]) + self.curr_lambda*torch.eye(X_tr_dict[ind_task].shape[1])) #self.lambda_coeff*np.eye(window_length))
                    rhs_for_v_0 = Herm(X_tr_dict[ind_task])@Y_tr_dict[ind_task] + self.curr_lambda*v_bar_curr
                    v_0 = C_inv@rhs_for_v_0 # same as v_star

                    ## new_X and new_Y for total loss (multiply coeff_EP to the test dataset and concat. with training dataset)
                    new_X = torch.cat( [X_tr_dict[ind_task], self.coeff_EP * X_te_dict[ind_task]]  , dim=0)
                    new_Y = torch.cat( [Y_tr_dict[ind_task], self.coeff_EP * Y_te_dict[ind_task]]  , dim=0)
                    C_inv_new = torch.inverse((Herm(new_X)@new_X) + self.curr_lambda*torch.eye(new_X.shape[1])) #self.lambda_coeff*np.eye(window_length))
                    rhs_for_v_new = Herm(new_X)@new_Y + self.curr_lambda*v_bar_curr
                    v_new = C_inv_new@rhs_for_v_new # same as v_star

                    curr_task_meta_grad = (1/self.coeff_EP) * (2*self.curr_lambda*(v_0 - v_new))
                    curr_meta_grad += curr_task_meta_grad
                curr_meta_grad /= len(self.beta_in_seq_total)
                # update
                v_bar_prev = v_bar_curr.clone()
                v_bar_curr -= self.lr_EP * curr_meta_grad + momentum_coeff*momentum
                momentum = v_bar_curr - v_bar_prev
        return v_bar_curr, momentum

def Herm(vector):
    return np.transpose(np.conjugate(vector))

def Herm_tensor(vector):
    return torch.transpose(torch.conj(vector),0,1)

