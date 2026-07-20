# SLAM领域最新论文 (2026.05.17)

> 每日自动更新SLAM领域的最新arXiv论文

> 使用说明: [点击查看](./docs/README.md#usage)

<details>
<summary>分类目录</summary>
<ol>
<li><a href='#lidar-slam'>LiDAR SLAM</a></li>
<li><a href='#visual-slam'>Visual SLAM</a></li>
<li><a href='#loop-closure'>Loop Closure</a></li>
<li><a href='#image-matching'>Image Matching</a></li>
<li><a href='#3dgs'>3DGS</a></li>
<li><a href='#depth-estimation'>Depth Estimation</a></li>
</ol>
</details>

<h2 id='lidar-slam'>LiDAR SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2026-05-14</td><td>SOCC-ICP: Semantics-Assisted Odometry based on Occupancy Grids and ICP</td><td>[2605.15074](http://arxiv.org/pdf/2605.15074)</td><td>◆ Reliable pose estimation in previously unseen environments is a fundamental capability of autonomous systems.
◆ Existing LiDAR odometry methods typically employ point-, surfel-, or NDT-based map representations, which are distinct from the semantic occupancy grids commonly used for downstream tasks such as motion planning.
◆ We introduce SOCC-ICP, a semantics-assisted odometry framework that jointly performs Semantic OCCupancy grid mapping and LiDAR scan alignment.</td></tr>
<tr><td>2026-05-13</td><td>LEXI-SG: Monocular 3D Scene Graph Mapping with Room-Guided Feed-Forward Reconstruction</td><td>[2605.13741](http://arxiv.org/pdf/2605.13741)</td><td>◆ Scene graphs are becoming a standard representation for robot navigation, providing hierarchical geometric and semantic scene understanding.
◆ However, most scene graph mapping methods rely on depth cameras or LiDAR sensors.
◆ In this work, we present LEXI-SG, the first dense monocular visual mapping system for open-vocabulary 3D scene graphs using only RGB camera input.</td></tr>
<tr><td>2026-05-12</td><td>WildPose: A Unified Framework for Robust Pose Estimation in the Wild</td><td>[2605.12774](http://arxiv.org/pdf/2605.12774)</td><td>◆ Estimating camera pose in dynamic environments is a critical challenge, as most visual SLAM and SfM methods assume static scenes.
◆ While recent dynamic-aware methods exist, they are often not unified: semantic-based approaches are brittle, per-sequence optimization methods fail on short sequences, and other learned models may degrade on static-only scenes.
◆ We present WildPose, a unified monocular pose estimation framework that is robust in dynamic environments while maintaining state-of-the-art performance on static and low-ego-motion datasets.</td></tr>
<tr><td>2026-05-11</td><td>MAGS-SLAM: Monocular Multi-Agent Gaussian Splatting SLAM for Geometrically and Photometrically Consistent Reconstruction</td><td>[2605.10760](http://arxiv.org/pdf/2605.10760)</td><td>◆ Collaborative photorealistic 3D reconstruction from multiple agents enables rapid large-scale scene capture for virtual production and cooperative multi-robot exploration.
◆ While recent 3D Gaussian Splatting (3DGS) SLAM algorithms can generate high-fidelity real-time mapping, most of the existing multi-agent Gaussian SLAM methods still rely on RGB-D sensors to obtain metric depth and simplify cross-agent alignment, which limits the deployment on lightweight, low-cost, or power-constrained robotic platforms.
◆ To address this challenge, we propose MAGS-SLAM, the first RGB-only multi-agent 3DGS SLAM framework for collaborative scene reconstruction.</td></tr>
<tr><td>2026-05-10</td><td>Above and Below: Heterogeneous Multi-robot SLAM Across Surface and Underwater Domains</td><td>[2605.09811](http://arxiv.org/pdf/2605.09811)</td><td>◆ Multi-robot simultaneous localization and mapping (SLAM) is a fundamental task in multi-robot operations.
◆ Robots must have a common understanding of their location and that of their team members to complete coordinated actions.
◆ However, multi-robot SLAM between Uncrewed Surface Vessels (USVs) and Autonomous Underwater Vehicles (AUVs) has primarily been achieved through acoustic pinging between robots to retrieve range measurements; a measurement technique requires that robots to be in similar locations simultaneously, have an uninterrupted path for signal propagation, and may necessitate synchronized clocks.</td></tr>
<tr><td>2026-05-10</td><td>Safety-Critical LiDAR-Inertial Odometry with On-Manifold Deterministic Protection Level</td><td>[2605.09383](http://arxiv.org/pdf/2605.09383)</td><td>◆ In safety-critical scenarios, the protection level of the autonomous navigation system is crucial for enabling mobile robots to perform safe tasks.
◆ However, existing studies on probabilistic navigation systems for robots usually perform offline accuracy evaluations using limited datasets and assume that the results can be applied to unknown real-world environments.
◆ We further introduce GeomPrompt-Recovery, an adaptation module that compensates for degraded depth by predicting the fourth channel correction relevant for the frozen segmenter.</td></tr>
<tr><td>2026-04-13</td><td>CDPR: Cross-modal Diffusion with Polarization for Reliable Monocular Depth Estimation</td><td>[2604.11097](http://arxiv.org/pdf/2604.11097)</td><td>◆ Monocular depth estimation is a fundamental yet challenging task in computer vision, especially under complex conditions such as textureless surfaces, transparency, and specular reflections.
◆ Recent diffusion-based approaches have significantly advanced performance by reformulating depth prediction as a denoising process in the latent space.
◆ However, existing methods rely solely on RGB inputs, which often lack sufficient cues in challenging regions.</td></tr>
<tr><td>2026-04-11</td><td>SMFormer: Empowering Self-supervised Stereo Matching via Foundation Models and Data Augmentation</td><td>[2604.10218](http://arxiv.org/pdf/2604.10218)</td><td>◆ Recent self-supervised stereo matching methods have made significant progress.
◆ They typically rely on the photometric consistency assumption, which presumes corresponding points across views share the same appearance.
◆ However, this assumption could be compromised by real-world disturbances, resulting in invalid supervisory signals and a significant accuracy gap compared to supervised methods.</td></tr>
<tr><td>2026-04-10</td><td>FF3R: Feedforward Feature 3D Reconstruction from Unconstrained views</td><td>[2604.09862](http://arxiv.org/pdf/2604.09862)</td><td>◆ Recent advances in vision foundation 
◆ We address this challenge by reframing MIE detection as a data-driven learning problem that assumes no prior knowledge of state preparation.</td></tr>
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2026.05.17
