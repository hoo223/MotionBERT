python train_custom.py \
--config configs/pose3d/DHDSTformer_torso2_h36m_depth3.yaml \
--evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_depth3/best_epoch.bin

python train_custom.py \
--config configs/pose3d/DHDSTformer_torso2_h36m_depth1.yaml \
--evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_depth1/best_epoch.bin

python train_custom.py \
--config configs/pose3d/DHDSTformer_torso2_h36m_head4.yaml \
--evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_head4/best_epoch.bin

python train_custom.py \
--config configs/pose3d/DHDSTformer_torso2_h36m_depth3_head4.yaml \
--evaluate checkpoint/pose3d/DHDSTformer_torso2_h36m_depth3_head4/best_epoch.bin