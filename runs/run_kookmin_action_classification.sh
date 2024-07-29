# ## train
# train kookmin S01 with clip aumentation / test kookmin others
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01_augment_243_50.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01_Augment_243_50

# train kookmin S01, 2 with clip aumentation / test kookmin others 
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01-2_augment_243_50.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01-2_Augment_243_50

# train kookmin S01 / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01_test_augment_243_50.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01_Test_Augment_243_50

# train kookmin S01, 2 / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01-2_test_augment_243_50.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01-2_Test_Augment_243_50

# train kookmin S01 with clip aumentation / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01_augment_243_50_test_augment_243_50.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01_Augment_243_50_Test_Augment_243_50

# train kookmin S01, 2 with clip aumentation / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01-2_augment_243_50_test_augment_243_50.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01-2_Augment_243_50_Test_Augment_243_50

# ## test
# train kookmin S01 with clip aumentation / test kookmin others
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01_augment_243_50.yaml \
--evaluate checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01_Augment_243_50/best_epoch.bin

# train kookmin S01, 2 with clip aumentation / test kookmin others 
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01-2_augment_243_50.yaml \
--evaluate checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01-2_Augment_243_50/best_epoch.bin

# train kookmin S01 / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01_test_augment_243_50.yaml \
--evaluate checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01_Test_Augment_243_50/best_epoch.bin

# train kookmin S01, 2 / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01-2_test_augment_243_50.yaml \
--evaluate checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01-2_Test_Augment_243_50/best_epoch.bin

# train kookmin S01 with clip aumentation / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01_augment_243_50_test_augment_243_50.yaml \
--evaluate checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01_Augment_243_50_Test_Augment_243_50/best_epoch.bin

# train kookmin S01, 2 with clip aumentation / test kookmin others with clip augmentation
python train_action_kookmin.py \
--config configs/action/MB_ft_Kookmin_train_s01-2_augment_243_50_test_augment_243_50.yaml \
--evaluate checkpoint/action/FT_MB_release_MB_Kookmin_Action_Train_S01-2_Augment_243_50_Test_Augment_243_50/best_epoch.bin
