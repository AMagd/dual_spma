for eta in 0.3 0.5 0.7 0.9 1.0; do
    echo "Running experiment with SPMA eta: $eta"
    python cleanrl/spma_minigrid.py \
      --env_id MiniGrid-Empty-5x5-v0 \
      --spma_eta $eta \
      --total_timesteps 200000 \
      --exp_name "spma_eta_${eta}"
done