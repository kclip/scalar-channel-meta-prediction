clc;
clear all;
rayleighchan = comm.RayleighChannel;

%% Rounded generation
num_tasks = 50; % num diff doppler freq.
num_samples = 1000000;
x = ones(num_samples,1);
curr_fading = cell(num_tasks, 1);
curr_ar_from_psd = cell(num_tasks,1);
for doppler_type_ind = 1:num_tasks
    rayleighchan.release()
    rayleighchan.SampleRate = 1;
    rayleighchan.PathDelays = 0;
    rayleighchan.MaximumDopplerShift = rand*0.05 + 0.05; %rand*0.099 + 0.001;%(0.1)-0.099*(doppler_type_ind-1)/(num_tasks-1); % 0.1 ~ 0.01
    rayleighchan.AveragePathGains = 0;
    rayleighchan.PathGainsOutputPort = true;
    rayleighchan.FadingTechnique = 'Filtered Gaussian noise';
    rayleighchan.DopplerSpectrum = doppler('Rounded');
    [y, pathgains] =  rayleighchan(x);
    curr_fading{doppler_type_ind, 1} = pathgains;  
    rayleighchan.reset;
    [y, pathgains] =  rayleighchan(x);
    curr_fading{doppler_type_ind, 2} = pathgains;
    %% ar from psd
    a_0 = 1;
    a_2 = -1.72;
    a_4 = 0.785;
    f_d = rayleighchan.MaximumDopplerShift;
    C_r = 1/((2*f_d)*(a_0+a_2/3 + a_4/5));
    max_lag = 100;
    ar_from_psd = [];
    lags = linspace(0,max_lag-1, max_lag);
    f_sample = linspace(-f_d, f_d, 1000);
    power_spectral_density = (1/((2*f_d)*(a_0+a_2/3 + a_4/5)))*((a_0 + a_2*(f_sample/f_d).^2 + a_4*(f_sample/f_d).^4));
    fun = @(f,t) (1/((2*f_d)*(a_0+a_2/3 + a_4/5)))*((a_0 + a_2*(f/f_d).^2 + a_4*(f/f_d).^4)).*exp(1j*2*pi*f*t);
    for t = 0:max_lag-1
        q = integral(@(f) fun(f,t),-f_d,f_d);
        ar_from_psd = [ar_from_psd, q];
    end
    curr_ar_from_psd{doppler_type_ind, 1} = ar_from_psd;
end
file_name = strcat('rounded_multi_w.mat');
save(file_name, 'curr_fading');
file_name = 'rounded_multi_w_ar_from_psd.mat';
save(file_name, 'curr_ar_from_psd');

