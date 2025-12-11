from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class DisplacementPredictor(nn.Module):
    """
    从观测序列预测轨迹的位移向量
    obs: (B, n_obs_steps, obs_dim) -> displacement: (B, action_dim)
    """
    def __init__(self, obs_dim, n_obs_steps, output_dim, hidden_dims=[256, 128]):
        super().__init__()
        input_dim = obs_dim * n_obs_steps  # 展平观测
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, obs):
        """
        Args:
            obs: (B, n_obs_steps, obs_dim) - 归一化后的观测序列
        Returns:
            displacement: (B, output_dim) - 预测的位移向量
        """
        B = obs.shape[0]
        obs_flat = obs.reshape(B, -1)  # (B, n_obs_steps * obs_dim)
        return self.mlp(obs_flat)


class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # ===== REPA 对齐 loss 参数 =====
            use_alignment_loss=False,           # 是否启用对齐 loss
            alignment_loss_weight=0.5,          # 对齐 loss 权重
            projection_hidden_dims=[512, 256],  # MLP 隐藏层维度
            alignment_loss_type='mse',          # 'mse' 或 'cosine'
            # ===== Cross Attention 参数 =====
            use_cross_attention=False,          # 是否使用 Cross Attention 注入 displacement
            displacement_predictor_hidden_dims=[256, 128],  # Displacement Predictor MLP 隐藏层
            displacement_loss_weight=1.0,       # Displacement 预测 loss 权重
            # ================================
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

         # ===== 新增：初始化投影头 =====
        self.use_alignment_loss = use_alignment_loss
        self.alignment_loss_weight = alignment_loss_weight
        self.alignment_loss_type = alignment_loss_type
    
        self.projection_head = None
        if use_alignment_loss:
            from diffusion_policy.model.common.projection_head import REPAProjectionHead
            # mid_dim 是 UNet 最深层的通道数（down_dims 的最后一个）
            mid_dim = model.mid_modules[0].out_channels
            self.projection_head = REPAProjectionHead(
                input_dim=mid_dim,
                output_dim=action_dim,  # displacement 与 action 维度相同
                hidden_dims=projection_hidden_dims
            )
        
        # ===== 新增：Cross Attention 和 Displacement Predictor =====
        self.use_cross_attention = use_cross_attention
        self.displacement_loss_weight = displacement_loss_weight
        
        # 初始化 Displacement Predictor MLP
        self.displacement_predictor = None
        if use_cross_attention:
            self.displacement_predictor = DisplacementPredictor(
                obs_dim=obs_dim,
                n_obs_steps=n_obs_steps,
                output_dim=action_dim,  # displacement 维度与 action 相同
                hidden_dims=displacement_predictor_hidden_dims
            )
        # ================================================================
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            nobs=None,  # 新增：用于预测 displacement 的观测
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        # ===== 推理时：使用 Displacement Predictor 预测 displacement =====
        cross_attention_cond = None
        if self.use_cross_attention and self.displacement_predictor is not None and nobs is not None:
            # nobs: (B, n_obs_steps, obs_dim)
            cross_attention_cond = self.displacement_predictor(nobs)
        # =================================================================

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            # 推理时使用 Displacement Predictor 的输出
            model_output, _ = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond,
                cross_attention_cond=cross_attention_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def _compute_displacement_gt(self, action):
        """
        计算动作序列的位移 GT
    
        Args:
            action: (B, T, Da) - 归一化后的动作序列
        Returns:
            displacement: (B, Da) - 从第一帧到最后一帧的累积位移
        """
        # 方案1: 直接计算首尾差
        displacement = action[:, -1, :] - action[:, 0, :]
    
        # 方案2: 计算累积和（如果需要）
        # displacement = action.sum(dim=1)  # (B, Da)
    
        return displacement


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        # 传递 nobs[:,:To] 给 Displacement Predictor
        nobs_for_predictor = nobs[:,:To]  # (B, n_obs_steps, obs_dim)
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            nobs=nobs_for_predictor,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # ===== 计算 displacement_gt（用于 Cross Attention 或 REPA） =====
        displacement_gt = None
        displacement_pred = None
        displacement_loss = torch.tensor(0.0, device=trajectory.device)
        
        if self.use_cross_attention or self.use_alignment_loss:
            displacement_gt = self._compute_displacement_gt(action)
        
        # ===== 使用 Displacement Predictor 预测并计算 loss =====
        if self.use_cross_attention and self.displacement_predictor is not None:
            assert displacement_gt is not None, "displacement_gt should be computed when use_cross_attention is True"
            # 获取用于预测的 obs: (B, n_obs_steps, obs_dim)
            obs_for_predictor = obs[:, :self.n_obs_steps, :]
            displacement_pred = self.displacement_predictor(obs_for_predictor)
            
            # 计算 Displacement Predictor 的监督 loss
            displacement_loss = F.mse_loss(displacement_pred, displacement_gt)
        
        # ===== Predict the noise residual =====
        # 训练时使用 displacement_gt 注入 Cross Attention（方案 A）
        cross_attention_cond = displacement_gt if self.use_cross_attention else None
        pred, mid_feature = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond,
            cross_attention_cond=cross_attention_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # ===== 原有的 diffusion loss =====
        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = diffusion_loss * loss_mask.type(diffusion_loss.dtype)
        diffusion_loss = reduce(diffusion_loss, 'b ... -> b (...)', 'mean')
        diffusion_loss = diffusion_loss.mean()

        # ===== 计算总 loss =====
        total_loss = diffusion_loss
        
        # 添加 Displacement Predictor loss
        if self.use_cross_attention and self.displacement_predictor is not None:
            total_loss = total_loss + self.displacement_loss_weight * displacement_loss
    
        if self.use_alignment_loss and self.projection_head is not None:
            # displacement_gt 已在前面计算，此时一定不为 None
            assert displacement_gt is not None, "displacement_gt should be computed when use_alignment_loss is True"
            
            # MLP 投影（梯度会回传到 UNet，alignment_loss 影响 UNet 训练）
            projected_displacement = self.projection_head(mid_feature)
        
            # 计算对齐 loss (REPA-style: 归一化 + 负余弦相似度)
            if self.alignment_loss_type == 'mse':
                alignment_loss = F.mse_loss(
                    projected_displacement, 
                    displacement_gt
                )
            elif self.alignment_loss_type == 'cosine':
                # REPA 原始实现: 先归一化，再计算负内积
                projected_norm = F.normalize(projected_displacement, dim=-1)
                gt_norm = F.normalize(displacement_gt, dim=-1)
                # 负余弦相似度: -(z · z_tilde)
                alignment_loss = -(projected_norm * gt_norm).sum(dim=-1).mean()
            else:
                raise ValueError(f"Unknown alignment_loss_type: {self.alignment_loss_type}")
        
            # 4. 加权组合
            total_loss = diffusion_loss + self.alignment_loss_weight * alignment_loss
            
            # 5. 计算相对误差: sqrt(alignment_loss) / mean(|displacement_gt|)
            rmse = torch.sqrt(alignment_loss)
            gt_magnitude = torch.norm(displacement_gt, dim=-1).mean()
            alignment_relative_error = rmse / (gt_magnitude + 1e-8)
            
            # 返回字典，包含各项 loss
            return {
                'total_loss': total_loss,
                'diffusion_loss': diffusion_loss.detach(),
                'alignment_loss': alignment_loss.detach(),
                'alignment_relative_error': alignment_relative_error.detach(),
                'displacement_loss': displacement_loss.detach() if isinstance(displacement_loss, torch.Tensor) else displacement_loss
            }
        # ===============================
    
        # 如果不使用 alignment loss，返回兼容格式
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss.detach(),
            'alignment_loss': torch.tensor(0.0, device=diffusion_loss.device),
            'alignment_relative_error': torch.tensor(0.0, device=diffusion_loss.device),
            'displacement_loss': displacement_loss.detach() if isinstance(displacement_loss, torch.Tensor) else displacement_loss
        }
