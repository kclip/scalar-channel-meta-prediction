clc;
clear all;
max_mc = 100; % for online
%max_mc = 1; % for offline
for ind_mc = 1:max_mc
    %% custom 
    %% simulation set-up
    TOTAL_FRAME = 100; % 2^para.mu*10*TOTAL_FRAME = number of total dataset (should be larger than training size + testing size)
    RS_pos = 1; % OFDM symbol index for RS
    % in order to make channel smallar for increase taps...
    % we need to decrease DS... W??
    %% basic init. setting for all tasks 
    init.DS = 30e-9;% Short delay spread
    init.mu = 1; 
    init.num_RB = 11; % 5MHz, smallest num_RB
    init.fc = 30e9; 

    init.num_Tx_antenna_horizontal = 1;            
    init.num_Tx_antenna_vertical = 1;
    init.num_Rx_antenna_horizontal = 1;
    init.num_Rx_antenna_vertical = 1;
    init.Tx_pol = 1;
    init.Rx_pol = 1;

    init.ch_type = 'CDL_C';
    init.total_rs_num = init.num_RB*12; % full RS for now...
    init.num_FFT = 256; %[128 256 512 1024 1596 2048 4096]; % this should be larger than init.num_RB*12
    init.InitialTime = 0; % input to para.CurrentTime

    TOTAL_META_DATASET = 500; %10000; %20000; # maybe due to saving issue...

    meta_te_dataset = cell(TOTAL_META_DATASET,1);
    
    % this is fixed over meta-dataset (in the notion from the paper -- should be fixed over frames!)
    rand_coup_AOD = NaN;
    rand_coup_ZOD = NaN;
    rand_coup_AOA = NaN;
    rand_coup_ZOA = NaN;

    for ind_meta_dataset = 1:TOTAL_META_DATASET
        disp('current task')
        disp(ind_meta_dataset)
        % for different doppler frequency
        init.user_speed = rand*0.9 + 0.1; % 0.1 ~ 1 -> 10~100
        init.theta_v = 45;
        init.phi_v = 45;
        if ind_meta_dataset > 1
            init.InitialTime = para.CurrentTime; % time keeps going!
        end

        para = module_Parameter_MIMO(init);
        num_Tx_antenna = para.num_Tx_antenna;
        num_Rx_antenna = para.num_Rx_antenna;
        % for each task, at the very first, we need to generate long-term
        % features
        very_first = true;
        Initial_Phase_for_all_rays = NaN;
        XPR = NaN;
        Rx_patterns = NaN;
        Tx_patterns = NaN;
        rhat_rx = NaN;
        %vbar = NaN;
        wl = NaN;
        a_MS = NaN;
        a_BS = NaN;
        Num_ray = -1; % full ray
        nTap = 1; % one cluster
        % currently XPR is fixed!
        ind_data = 1;
        scalar_channel = zeros(2^para.mu*10*TOTAL_FRAME, 1);
        for ind_frame = 1:TOTAL_FRAME
            for ind_subframe = 1:10
                for ind_slot =  1:2^para.mu
                    for ind_OFDM_symbol = 1:14
                        if ind_OFDM_symbol == 1 | ind_OFDM_symbol == 8
                            symbol_duration = para.symbol_duration(1);
                        else
                            symbol_duration = para.symbol_duration(2);
                        end
                        % generate channel 
                        if ind_OFDM_symbol == RS_pos
                            % cdl channel
                            [H_actual_tap, Initial_Phase_for_all_rays, XPR, Rx_patterns, Tx_patterns, rhat_rx, wl, a_MS, a_BS, Num_ray, nTap, rand_coup_AOD, rand_coup_AOA, rand_coup_ZOD, rand_coup_ZOA] =  CDL_generation(para, very_first, Initial_Phase_for_all_rays, XPR, Rx_patterns, Tx_patterns, rhat_rx, wl, a_MS, a_BS, Num_ray, nTap, rand_coup_AOD, rand_coup_AOA, rand_coup_ZOD, rand_coup_ZOA);
                            if very_first == true
                                very_first = false;
                            else
                                ;
                            end
                            curr_channel = zeros(num_Tx_antenna*num_Rx_antenna,para.L); %zeros(para.L,num_Tx_antenna*num_Rx_antenna);
                            for ind_tx_antenna = 1:num_Tx_antenna
                                curr_channel((ind_tx_antenna-1)*num_Rx_antenna+1:(ind_tx_antenna-1)*num_Rx_antenna+num_Rx_antenna,:) = H_actual_tap(:, ind_tx_antenna, :);
                            end 
                            scalar_channel(ind_data) = curr_channel(:,1);
                            ind_data = ind_data + 1;
                        else
                            ;
                        end
                        % debug with this!
                        para.CurrentTime = para.CurrentTime + symbol_duration;
                    end
                end
            end
        end
        meta_te_dataset{ind_meta_dataset} = scalar_channel;
    end
    file_name = strcat('./online_dataset/3gpp_meta_training_online_mc',string(ind_mc),'.mat'); % for online 
    % for online but to get comparison w.r.t. offline meta-training, choose held-out ind_mc number (e.g., larger than max_mc), then generate from this ind_mc, and name it as '3gpp_meta_training_online_for_offline.mat'
    % file_name = strcat('3gpp_meta_training_offline.mat'); for offline
    save(file_name, 'meta_te_dataset');
end





