## Channel Prediction via Meta-Learning and EP

This repository contains code for "[Predicting Flat-Fading Channels via Meta-Learned Closed-Form Linear Filters and Equilibrium Propagation](https://arxiv.org/abs/2110.00414)" - Sangwoo Park and Osvaldo Simeone.

### Dependencies
This program is written in python 3.8 and uses PyTorch 1.8.1.

### Essential codes
- Closed-form meta-learning for linear filters can be found at `funcs/Ridge_meta.py`.
- Gradient-based meta-learning for linear filters via Equilibrium Propagation (EP) can be found at `funcs/Ridge_meta_EP.py`.
- Offline meta-learning scenario can be found at `main_offline.py`. Detailed usage can be found below.
- Online meta-learning scenario can be found at `main_online.py`. Detailed usage can be found below.
- Channel dataset generation can be found at `channel_gen` folder.

### How to run the codes

#### Prerequisites (data generation)
- For *Random Doppler Frequency (Offline)* (Fig. 2), run `channel_gen/Jakes_Rounded/jakes_multi_w.m,rounded_multi_w.m` and place the resulting .mat files into `../generated_channels/`.
- For *Standard Channel Model (Offline)* (Fig. 3), run `channel_gen/5G_standard_CDL/main_custom.m` and place the resulting `3gpp_meta_training_offline.mat` file into `../generated_channels/`.
- For *Gradient-Based Meta-Learning (Onlline)* (Fig. 4), run `channel_gen/5G_standard_CDL/main_custom.m` and place the resulting .mat files into `../generated_channels/online_dataset/`.

#### 1) *Random Doppler Frequency (Offline)* (Fig. 2)
- For genie-aided performance, execute `runs/fig_234/offline_jakes_rounded_fig_2/genie_aided.sh`

- For conventional learning, execute `runs/fig_234/offline_jakes_rounded_fig_2/conven.sh`

- For joint learning, execute `runs/fig_234/offline_jakes_rounded_fig_2/joint.sh`

- For meta-learning, execute `runs/fig_234/offline_jakes_rounded_fig_2/meta.sh`

#### 2) *Standard Channel Model (Offline)* (Fig. 3)
- For conventional learning, execute `runs/fig_234/offline_standard_fig_3/conven.sh`

- For joint learning, execute `runs/fig_234/offline_standard_fig_3/joint.sh`

- For meta-learning, execute `runs/fig_234/offline_standard_fig_3/meta.sh`
#### 3) *Gradient-Based Meta-Learning (Onlline)* (Fig. 4)
- For EP-based online meta-learning, execute `runs/fig_234/online_fig_4/online_meta_EP.sh`

- For offline joint learning, execute `runs/fig_234/online_fig_4/offline_joint.sh`

- For offline meta-learning, execute `runs/fig_234/online_fig_4/offline_meta.sh`
