# Notebook-to-script mapping

| Original notebook section | Server script |
|---|---|
| Environment installation and validation | `requirements.txt`, `environment.yml`, `scripts/check_environment.py` |
| Stage-1 training | `scripts/train_stage1.py` |
| Export official Stage-1 topology | `scripts/export_stage1_topology.py` |
| Stage-1 and reduced-topology rollout | `scripts/run_external_topology_inference.py` |
| Online node error | `scripts/compute_online_node_error.py` |
| Neighbor-only node clustering | `scripts/generate_node_cluster_topology.py` |
| Reduced-to-full trajectory reconstruction | `scripts/reconstruct_node_cluster_trajectory.py` |
| Online stiffness and BT-guided spring pruning | `scripts/generate_online_spring_topologies.py` |
| Complete ordered execution | `scripts/run_strategy2_pipeline.py` |
| CD and tracking error | `scripts/evaluate_geometry.py` |
| Gaussian rendering and rendering FPS | `scripts/render_methods.py` |
| PSNR, SSIM, LPIPS and IoU | `scripts/evaluate_rendering.py` |
| Simulation FPS | `scripts/benchmark_simulation.py` |
| GT / prediction / error videos | `scripts/make_visual_videos.py` |

The notebook's exploratory print/check cells were consolidated into input
validation, JSON metadata, CSV outputs, and the standalone smoke test.
