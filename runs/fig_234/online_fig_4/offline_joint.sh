python ../../../main_online.py --window_length 5 --lag 3 --num_paths 5   --linear_ridge_mode 1  --if_wiener_filter_as_ols   --fading_mode 3  --num_samples_for_test 1000 --if_exp_over_supp --noise_var_for_exp_over_supp 1e-4 --fixed_lambda_value 1 --if_for_online_meta_test_do_offline_meta_train --offline_meta_training_total_tasks 1;
python ../../../main_online.py --window_length 5 --lag 3 --num_paths 5   --linear_ridge_mode 1  --if_wiener_filter_as_ols   --fading_mode 3  --num_samples_for_test 1000 --if_exp_over_supp --noise_var_for_exp_over_supp 1e-4 --fixed_lambda_value 1 --if_for_online_meta_test_do_offline_meta_train --offline_meta_training_total_tasks 5;
python ../../../main_online.py --window_length 5 --lag 3 --num_paths 5   --linear_ridge_mode 1  --if_wiener_filter_as_ols   --fading_mode 3  --num_samples_for_test 1000 --if_exp_over_supp --noise_var_for_exp_over_supp 1e-4 --fixed_lambda_value 1 --if_for_online_meta_test_do_offline_meta_train --offline_meta_training_total_tasks 100;