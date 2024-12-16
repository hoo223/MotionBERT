# python train_custom.py \
# --config configs/pose3d/DHDSTformer_torso2_h36m_depth3.yaml \
# --evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_depth3/best_epoch.bin

# python train_custom.py \
# --config configs/pose3d/DHDSTformer_torso2_h36m_depth1.yaml \
# --evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_depth1/best_epoch.bin

# python train_custom.py \
# --config configs/pose3d/DHDSTformer_torso2_h36m_head4.yaml \
# --evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_head4/best_epoch.bin

# python train_custom.py \
# --config configs/pose3d/DHDSTformer_torso2_h36m_depth3_head4.yaml \
# --evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_depth3_head4/best_epoch.bin

# mmpose test
python train_kookmin.py \
--config configs/pose3d_kookmin/FT_fit3d_ateta_kookmin_total_with_kookmin_setting2_s0178_test_with_mmpose_test.yaml \
--evaluate checkpoint/pose3d/FT_fit3d_ateta_kookmin_total_with_kookmin_setting2_s0178_test/best_epoch.bin