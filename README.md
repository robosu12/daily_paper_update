# SLAM领域最新论文 (2025.12.09)

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
<tr><td>2025-12-08</td><td>OptMap: Geometric Map Distillation via Submodular Maximization</td><td>[2512.07775](http://arxiv.org/pdf/2512.07775)</td><td>◆ Autonomous robots rely on geometric maps to inform a diverse set of perception and decision-making algorithms.
◆ As autonomy requires reasoning and planning on multiple scales of the environment, each algorithm may require a different map for optimal performance.
◆ Light Detection And Ranging (LiDAR) sensors generate an abundance of geometric data to satisfy these diverse requirements, but selecting informative, size-constrained maps is computationally challenging as it requires solving an NP-hard combinatorial optimization.</td></tr>
<tr><td>2025-12-08</td><td>Spatiotemporal Calibration and Ground Truth Estimation for High-Precision SLAM Benchmarking in Extended Reality</td><td>[2512.07221](http://arxiv.org/pdf/2512.07221)</td><td>◆ Simultaneous localization and mapping (SLAM) plays a fundamental role in extended reality (XR) applications.
◆ As the standards for immersion in XR continue to increase, the demands for SLAM benchmarking have become more stringent.
◆ Trajectory accuracy is the key metric, and marker-based optical motion capture (MoCap) systems are widely used to generate ground truth (GT) because of their drift-free and relatively accurate measurements.</td></tr>
<tr><td>2025-12-07</td><td>Dynamic Visual SLAM using a General 3D Prior</td><td>[2512.06868](http://arxiv.org/pdf/2512.06868)</td><td>◆ Reliable incremental estimation of camera poses and 3D reconstruction is key to enable various applications including robotics, interactive visualization, and augmented reality.
◆ However, this task is particularly challenging in dynamic natural environments, where scene dynamics can severely deteriorate camera pose estimation accuracy.
◆ In this work, we propose a novel monocular visual SLAM system that can robustly estimate camera poses in dynamic scenes.</td></tr>
<tr><td>2025-12-04</td><td>ARCAS: An Augmented Reality Collision Avoidance System with SLAM-Based Tracking for Enhancing VRU Safety</td><td>[2512.05299](http://arxiv.org/pdf/2512.05299)</td><td>◆ Vulnerable road users (VRUs) face high collision risks in mixed traffic, yet most existing safety systems prioritize driver or vehicle assistance over direct VRU support.
◆ This paper presents ARCAS, a real-time augmented reality collision avoidance system that provides personalized spatial alerts to VRUs via wearable AR headsets.
◆ By fusing roadside 360-degree 3D LiDAR with SLAM-based headset tracking and an automatic 3D calibration procedure, ARCAS accurately overlays world-locked 3D bounding boxes and directional arrows onto approaching hazards in the user&#x27;s passthrough view.</td></tr>
<tr><td>2025-12-04</td><td>TEMPO-VINE: A Multi-Temporal Sensor Fusion Dataset for Localization and Mapping in Vineyards</td><td>[2512.04772](http://arxiv.org/pdf/2512.04772)</td><td>◆ In recent years, precision agriculture has been introducing groundbreaking innovations in the field, with a strong focus on automation.
◆ However, research studies in robotics and autonomous navigation often rely on controlled simulations or isolated field trials.
◆ The absence of a realistic common benchmark represents a significant limitation for the diffusion of robust autonomous systems under real complex agricultural conditions.</td></tr>
<tr><td>2025-12-03</td><td>What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models</td><td>[2512.03422](http://arxiv.org/pdf/2512.03422)</td><td>◆ In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models.
◆ While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance.
◆ Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence.</td></tr>
<tr><td>2025-12-04</td><td>Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing</td><td>[2512.03397](http://arxiv.org/pdf/2512.03397)</td><td>◆ LiDAR-inertial odometry (LIO) is an active research area, as it enables accurate real-time state estimation in GPS-denied environments.
◆ Recent advances in map data structures and spatial indexing have significantly improved the efficiency of LIO systems.
◆ Nevertheless, we observe that two aspects may still leave room for improvement: (1) nearest neighbor search often requires examining multiple spatial units to gather sufficient points for plane fitting, and (2) plane parameters are typically recomputed at every iteration despite unchanged map geometry.</td></tr>
<tr><td>2025-12-02</td><td>VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</td><td>[2512.02293](http://arxiv.org/pdf/2512.02293)</td><td>◆ We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction.
◆ Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations.
◆ Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states.</td></tr>
<tr><td>2025-12-01</td><td>KM-ViPE: Online Tightly Coupled Vision-Language-Geometry Fusion for Open-Vocabulary Semantic SLAM</td><td>[2512.01889](http://arxiv.org/pdf/2512.01889)</td><td>◆ We present KM-ViPE (Knowledge Mapping Video Pose Engine), a real-time open-vocabulary SLAM framework for uncalibrated monocular cameras in dynamic environments.
◆ Unlike systems requiring depth sensors and offline calibration, KM-ViPE operates directly on raw RGB streams, making it ideal for ego-centric applications and harvesting internet-scale video data for training.
◆ KM-ViPE tightly couples DINO visual features with geometric constraints through a high-level features based adaptive robust kernel that handles both moving objects and movable static objects (e.g., moving furniture in ego-centric views).</td></tr>
<tr><td>2025-12-01</td><td>Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching</td><td>[2512.01850](http://arxiv.org/pdf/2512.01850)</td><td>◆ Point cloud registration aligns multiple unposed point clouds into a common frame, and is a core step for 3D reconstruction and robot localization.
◆ In this work, we cast registration as conditional generation: a learned continuous, point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered.
◆ Unlike previous methods that conduct correspondence matching to estimate the transformation between a pair of point clouds and then optimize the pairwise transformations to realize multi-view registration, our model directly generates the registered point cloud.</td></tr>
<tr><td>2025-12-01</td><td>AgriLiRa4D: A Multi-Sensor UAV Dataset for Robust SLAM in Challenging Agricultural Fields</td><td>[2512.01753](http://arxiv.org/pdf/2512.01753)</td><td>◆ Multi-sensor Simultaneous Localization and Mapping (SLAM) is essential for Unmanned Aerial Vehicles (UAVs) performing agricultural tasks such as spraying, surveying, and inspection.
◆ However, real-world, multi-modal agricultural UAV datasets that enable research on robust operation remain scarce.
◆ To address this gap, we present AgriLiRa4D, a multi-modal UAV dataset designed for challenging outdoor agricultural environments.</td></tr>
<tr><td>2025-12-01</td><td>EGG-Fusion: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly</td><td>[2512.01296](http://arxiv.org/pdf/2512.01296)</td><td>◆ Real-time 3D reconstruction is a fundamental task in computer graphics.
◆ Recently, differentiable-rendering-based SLAM system has demonstrated significant potential, enabling photorealistic scene rendering through learnable scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS).
◆ Current differentiable rendering methods face dual challenges in real-time computation and sensor noise sensitivity, leading to degraded geometric fidelity in scene reconstruction and limited practicality.</td></tr>
<tr><td>2025-12-01</td><td>Design loads for wave impacts -- introducing the Probabilistic Adaptive Screening (PAS) method for predicting extreme non-linear loads on maritime structures</td><td>[2511.23156](http://arxiv.org/pdf/2511.23156)</td><td>◆ To ensure the safety of marine and coastal structures, extreme (design) values should be known at the design stage.
◆ But for such complex systems, estimating the magnitude of events which are both non-linear and rare is extremely challenging, and involves considerable computational cost to capture the high-fidelity physics.
◆ To address this challenge, we offer a new multi-fidelity screening method, Probabilistic Adaptive Screening (PAS), which accurately predicts extreme values of strongly non-linear wave-induced loads while minimising the required high-fidelity simulation duration.</td></tr>
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='visual-slam'>Visual SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-12-07</td><td>Dynamic Visual SLAM using a General 3D Prior</td><td>[2512.06868](http://arxiv.org/pdf/2512.06868)</td><td>◆ Reliable incremental estimation of camera poses and 3D reconstruction is key to enable various applications including robotics, interactive visualization, and augmented reality.
◆ However, this task is particularly challenging in dynamic natural environments, where scene dynamics can severely deteriorate camera pose estimation accuracy.
◆ In this work, we propose a novel monocular visual SLAM system that can robustly estimate camera poses in dynamic scenes.</td></tr>
<tr><td>2025-12-02</td><td>VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</td><td>[2512.02293](http://arxiv.org/pdf/2512.02293)</td><td>◆ We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction.
◆ Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations.
◆ Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states.</td></tr>
<tr><td>2025-12-01</td><td>KM-ViPE: Online Tightly Coupled Vision-Language-Geometry Fusion for Open-Vocabulary Semantic SLAM</td><td>[2512.01889](http://arxiv.org/pdf/2512.01889)</td><td>◆ We present KM-ViPE (Knowledge Mapping Video Pose Engine), a real-time open-vocabulary SLAM framework for uncalibrated monocular cameras in dynamic environments.
◆ Unlike systems requiring depth sensors and offline calibration, KM-ViPE operates directly on raw RGB streams, making it ideal for ego-centric applications and harvesting internet-scale video data for training.
◆ KM-ViPE tightly couples DINO visual features with geometric constraints through a high-level features based adaptive robust kernel that handles both moving objects and movable static objects (e.g., moving furniture in ego-centric views).</td></tr>
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='loop-closure'>Loop Closure</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-12-07</td><td>On Memory: A comparison of memory mechanisms in world models</td><td>[2512.06983](http://arxiv.org/pdf/2512.06983)</td><td>◆ World models enable agents to plan within imagined environments by predicting future states conditioned on past observations and actions.
◆ However, their ability to plan over long horizons is limited by the effective memory span of the backbone architecture.
◆ This limitation leads to perceptual drift in long rollouts, hindering the model&#x27;s capacity to perform loop closures within imagined trajectories.</td></tr>
<tr><td>2025-12-06</td><td>General Computation using Slidable Tiles with Deterministic Global Forces</td><td>[2512.06574](http://arxiv.org/pdf/2512.06574)</td><td>◆ We study the computational power of the Full-Tilt model of motion planning, where slidable polyominos are moved maximally around a board by way of a sequence of directional ``tilts.&#x27;&#x27; We focus on the deterministic scenario in which the tilts constitute a repeated clockwise rotation.
◆ We show that general-purpose computation is possible within this framework by providing a direct and efficient simulation of space-bounded Turing machines in which one computational step of the machine is simulated per $O(1)$ rotations.
◆ We further show that the initial tape of the machine can be programmed by an initial tilt-sequence preceding the rotations.</td></tr>
<tr><td>2025-12-06</td><td>Innovation, Spillovers and Economic Geography</td><td>[2512.06402](http://arxiv.org/pdf/2512.06402)</td><td>◆ We develop a Schumpeterian quality-ladder spatial model in which innovation arrivals depend on regional knowledge spillovers.
◆ A parsimonious reduced-form diffusion mechanism induces the convergence of regions&#x27; average distance to the global frontier quality.
◆ As a result, regional differences in knowledge levels stem residually from asymmetries in the spatial distribution of researchers and firms.</td></tr>
<tr><td>2025-12-05</td><td>GuideNav: User-Informed Development of a Vision-Only Robotic Navigation Assistant For Blind Travelers</td><td>[2512.06147](http://arxiv.org/pdf/2512.06147)</td><td>◆ While commendable progress has been made in user-centric research on mobile assistive systems for blind and low-vision (BLV) individuals, references that directly inform robot navigation design remain rare.
◆ To bridge this gap, we conducted a comprehensive human study involving interviews with 26 guide dog handlers, four white cane users, nine guide dog trainers, and one O\&amp;M trainer, along with 15+ hours of observing guide dog-assisted walking.
◆ After de-identification, we open-sourced the dataset to promote human-centered development and informed decision-making for assistive systems for BLV people.</td></tr>
<tr><td>2025-12-04</td><td>Shared Multi-modal Embedding Space for Face-Voice Association</td><td>[2512.04814](http://arxiv.org/pdf/2512.04814)</td><td>◆ The FAME 2026 challenge comprises two demanding tasks: training face-voice associations combined with a multilingual setting that includes testing on languages on which the model was not trained.
◆ Our approach consists of separate uni-modal processing pipelines with general face and voice feature extraction, complemented by additional age-gender feature extraction to support prediction.
◆ The resulting single-modal features are projected into a shared embedding space and trained with an Adaptive Angular Margin (AAM) loss.</td></tr>
<tr><td>2025-12-04</td><td>TEMPO-VINE: A Multi-Temporal Sensor Fusion Dataset for Localization and Mapping in Vineyards</td><td>[2512.04772](http://arxiv.org/pdf/2512.04772)</td><td>◆ In recent years, precision agriculture has been introducing groundbreaking innovations in the field, with a strong focus on automation.
◆ However, research studies in robotics and autonomous navigation often rely on controlled simulations or isolated field trials.
◆ The absence of a realistic common benchmark represents a significant limitation for the diffusion of robust autonomous systems under real complex agricultural conditions.</td></tr>
<tr><td>2025-12-02</td><td>MagicQuillV2: Precise and Interactive Image Editing with Layered Visual Cues</td><td>[2512.03046](http://arxiv.org/pdf/2512.03046)</td><td>◆ We propose MagicQuill V2, a novel system that introduces a \textbf{layered composition} paradigm to generative image editing, bridging the gap between the semantic power of diffusion models and the granular control of traditional graphics software.
◆ While diffusion transformers excel at holistic generation, their use of singular, monolithic prompts fails to disentangle distinct user intentions for content, position, and appearance.
◆ To overcome this, our method deconstructs creative intent into a stack of controllable visual cues: a content layer for what to create, a spatial layer for where to place it, a structural layer for how it is shaped, and a color layer for its palette.</td></tr>
<tr><td>2025-12-02</td><td>Polar Perspectives: Evaluating 2-D LiDAR Projections for Robust Place Recognition with Visual Foundation Models</td><td>[2512.02897](http://arxiv.org/pdf/2512.02897)</td><td>◆ This work presents a systematic investigation into how alternative LiDAR-to-image projections affect metric place recognition when coupled with a state-of-the-art vision foundation model.
◆ We introduce a modular retrieval pipeline that controls for backbone, aggregation, and evaluation protocol, thereby isolating the influence of the 2-D projection itself.
◆ Using consistent geometric and structural channels across multiple datasets and deployment scenarios, we identify the projection characteristics that most strongly determine discriminative power, robustness to environmental variation, and suitability for real-time autonomy.</td></tr>
<tr><td>2025-12-02</td><td>VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</td><td>[2512.02293](http://arxiv.org/pdf/2512.02293)</td><td>◆ We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction.
◆ Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations.
◆ Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states.</td></tr>
<tr><td>2025-12-01</td><td>Magnetoelectric effect in the mixed valence polyoxovanadate cage V$_{12}$</td><td>[2512.02215](http://arxiv.org/pdf/2512.02215)</td><td>◆ Development of spintronic and quantum computing devices increases demand for efficient, energy saving method of spin manipulation at molecular scale.
◆ Polyoxovanadate molecular magnets being susceptible to both electric and magnetic fields may serve here as a good base material.
◆ In this paper two isostructural anions [V$_{12}$As$_8$O$_{40}$(HCO$_2$)]$^{n-}$ (with $n=3,5$) featuring two different mixed-valence states with itinerant and localized valence electrons are studied.</td></tr>
<tr><td>2025-12-04</td><td>The Dependence of Earth Milankovitch Cycles on Martian Mass</td><td>[2512.02108](http://arxiv.org/pdf/2512.02108)</td><td>◆ The Milankovitch cycles of Earth result from gravitational interactions with other bodies in the Solar System.
◆ These interactions lead to slow changes in the orbit and angular momentum vector of Earth, and correspondingly influence Earth&#x27;s climate evolution.
◆ Several studies have shown that Mars may play a significant role in these Milankovitch cycles, such as the 2.4 Myr eccentricity cycle related to perihelion precession dynamics.</td></tr>
<tr><td>2025-12-01</td><td>Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching</td><td>[2512.01850](http://arxiv.org/pdf/2512.01850)</td><td>◆ Point cloud registration aligns multiple unposed point clouds into a common frame, and is a core step for 3D reconstruction and robot localization.
◆ In this work, we cast registration as conditional generation: a learned continuous, point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered.
◆ Unlike previous methods that conduct correspondence matching to estimate the transformation between a pair of point clouds and then optimize the pairwise transformations to realize multi-view registration, our model directly generates the registered point cloud.</td></tr>
<tr><td>2025-12-01</td><td>Exciton-Polariton hybrid skin-topological states</td><td>[2512.01768](http://arxiv.org/pdf/2512.01768)</td><td>◆ The non Hermitian skin effect, where bulk states accumulate at system boundaries, challenges the conventional bulk boundary correspondence.
◆ Here we propose a scheme to realize hybrid skin topological states in exciton polariton honeycomb lattices by introducing sublattice dependent gain and loss.
◆ This non Hermiticity couples with the intrinsic topological edge modes, leading to relocalization of edge states.</td></tr>
<tr><td>2025-12-01</td><td>RoboLoc: A Benchmark Dataset for Point Place Recognition and Localization in Indoor-Outdoor Integrated Environments</td><td>[2512.01194](http://arxiv.org/pdf/2512.01194)</td><td>◆ Robust place recognition is essential for reliable localization in robotics, particularly in complex environments with fre- quent indoor-outdoor transitions.
◆ However, existing LiDAR-based datasets often focus on outdoor scenarios and lack seamless domain shifts.
◆ In this paper, we propose RoboLoc, a benchmark dataset designed for GPS-free place recognition in indoor-outdoor environments with floor transitions.</td></tr>
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='image-matching'>Image Matching</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-12-08</td><td>UnCageNet: Tracking and Pose Estimation of Caged Animal</td><td>[2512.07712](http://arxiv.org/pdf/2512.07712)</td><td>◆ Animal tracking and pose estimation systems, such as STEP (Simultaneous Tracking and Pose Estimation) and ViTPose, experience substantial performance drops when processing images and videos with cage structures and systematic occlusions.
◆ We present a three-stage preprocessing pipeline that addresses this limitation through: (1) cage segmentation using a Gabor-enhanced ResNet-UNet architecture with tunable orientation filters, (2) cage inpainting using CRFill for content-aware reconstruction of occluded regions, and (3) evaluation of pose estimation and tracking on the uncaged frames.
◆ Our Gabor-enhanced segmentation model leverages orientation-aware features with 72 directional kernels to accurately identify and segment cage structures that severely impair the performance of existing methods.</td></tr>
<tr><td>2025-12-04</td><td>Value Gradient Guidance for Flow Matching Alignment</td><td>[2512.05116](http://arxiv.org/pdf/2512.05116)</td><td>◆ While methods exist for aligning flow matching models--a popular and effective class of generative models--with human preferences, existing approaches fail to achieve both adaptation efficiency and probabilistically sound prior preservation.
◆ In this work, we leverage the theory of optimal control and propose VGG-Flow, a gradient-matching-based method for finetuning pretrained flow matching models.
◆ The key idea behind this algorithm is that the optimal difference between the finetuned velocity field and the pretrained one should be matched with the gradient field of a value function.</td></tr>
<tr><td>2025-12-04</td><td>Deep infant brain segmentation from multi-contrast MRI</td><td>[2512.05114](http://arxiv.org/pdf/2512.05114)</td><td>◆ Segmentation of magnetic resonance images (MRI) facilitates analysis of human brain development by delineating anatomical structures.
◆ However, in infants and young children, accurate segmentation is challenging due to development and imaging constraints.
◆ Pediatric brain MRI is notoriously difficult to acquire, with inconsistent availability of imaging modalities, substantial non-head anatomy in the field of view, and frequent motion artifacts.</td></tr>
<tr><td>2025-12-04</td><td>Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression</td><td>[2512.05081](http://arxiv.org/pdf/2512.05081)</td><td>◆ Recent advances in autoregressive video diffusion have enabled real-time frame streaming, yet existing solutions still suffer from temporal repetition, drift, and motion deceleration.
◆ We find that naively applying StreamingLLM-style attention sinks to video diffusion leads to fidelity degradation and motion stagnation.
◆ To overcome this, we introduce Deep Forcing, which consists of two training-free mechanisms that address this without any fine-tuning.</td></tr>
<tr><td>2025-12-04</td><td>Improving Posterior Inference of Galaxy Properties with Image-Based Conditional Flow Matching</td><td>[2512.05078](http://arxiv.org/pdf/2512.05078)</td><td>◆ Estimating physical properties of galaxies from wide-field surveys remains a central challenge in astrophysics.
◆ While spectroscopy provides precise measurements, it is observationally expensive, and photometry discards morphological information that correlates with mass, star formation history, metallicity, and dust.
◆ We present a conditional flow matching (CFM) framework that leverages pixel-level imaging alongside photometry to improve posterior inference of galaxy properties.</td></tr>
<tr><td>2025-12-04</td><td>Generative Neural Video Compression via Video Diffusion Prior</td><td>[2512.05016](http://arxiv.org/pdf/2512.05016)</td><td>◆ We present GNVC-VD, the first DiT-based generative neural video compression framework built upon an advanced video generation foundation model, where spatio-temporal latent compression and sequence-level generative refinement are unified within a single codec.
◆ Existing perceptual codecs primarily rely on pre-trained image generative priors to restore high-frequency details, but their frame-wise nature lacks temporal modeling and inevitably leads to perceptual flickering.
◆ To address this, GNVC-VD introduces a unified flow-matching latent refinement module that leverages a video diffusion transformer to jointly enhance intra- and inter-frame latents through sequence-level denoising, ensuring consistent spatio-temporal details.</td></tr>
<tr><td>2025-12-04</td><td>Environment-Aware Channel Inference via Cross-Modal Flow: From Multimodal Sensing to Wireless Channels</td><td>[2512.04966](http://arxiv.org/pdf/2512.04966)</td><td>◆ Accurate channel state information (CSI) underpins reliable and efficient wireless communication.
◆ However, acquiring CSI via pilot estimation incurs substantial overhead, especially in massive multiple-input multiple-output (MIMO) systems operating in high-Doppler environments.
◆ By leveraging the growing availability of environmental sensing data, this treatise investigates pilot-free channel inference that estimates complete CSI directly from multimodal observations, including camera images, LiDAR point clouds, and GPS coordinates.</td></tr>
<tr><td>2025-12-04</td><td>LatentFM: A Latent Flow Matching Approach for Generative Medical Image Segmentation</td><td>[2512.04821](http://arxiv.org/pdf/2512.04821)</td><td>◆ Generative models have achieved remarkable progress with the emergence of flow matching (FM).
◆ It has demonstrated strong generative capabilities and attracted significant attention as a simulation-free flow-based framework capable of learning exact data densities.
◆ Motivated by these advances, we propose LatentFM, a flow-based model operating in the latent space for medical image segmentation.</td></tr>
<tr><td>2025-12-04</td><td>Unveiling gravitational waves from core-collapse supernovae with MUSE</td><td>[2512.04804](http://arxiv.org/pdf/2512.04804)</td><td>◆ The core collapse of a massive star at the end of its life can give rise to one of the most powerful phenomena in the Universe.
◆ Because of violent mass motions that take place during the explosion, core-collapse supernovae have been considered a potential source of detectable gravitational waveforms for decades.
◆ However, their intrinsic stochasticity makes ineffective the use of modelled techniques such as matched filtering, forcing us to develop model independent technique to unveil their nature.</td></tr>
<tr><td>2025-12-04</td><td>Live Avatar: Streaming Real-time Audio-Driven Avatar Generation with Infinite Length</td><td>[2512.04677](http://arxiv.org/pdf/2512.04677)</td><td>◆ Existing diffusion-based video generation methods are fundamentally constrained by sequential computation and long-horizon inconsistency, limiting their practical adoption in real-time, streaming audio-driven avatar synthesis.
◆ We present Live Avatar, an algorithm-system co-designed framework that enables efficient, high-fidelity, and infinite-length avatar generation using a 14-billion-parameter diffusion model.
◆ Our approach introduces Timestep-forcing Pipeline Parallelism (TPP), a distributed inference paradigm that pipelines denoising steps across multiple GPUs, effectively breaking the autoregressive bottleneck and ensuring stable, low-latency real-time streaming.</td></tr>
<tr><td>2025-12-04</td><td>Spectral micro-CT for quantitative analysis of calcification in fibrocartilage</td><td>[2512.04662](http://arxiv.org/pdf/2512.04662)</td><td>◆ This work introduces a quantitative method for assessing calcification in fibrocartilage using spectral micro-computed tomography ($μ$CT).
◆ Tissue samples of hip acetabular labrum from patients with osteoarthritis and femoroacetabular impingement were imaged with a laboratory-based spectral $μ$CT system equipped with a small-pixel photon-counting detector.
◆ The detector operated with two energy thresholds, allowing the simultaneous acquisition of two CT datasets at different X-ray energies.</td></tr>
<tr><td>2025-12-03</td><td>Leveraging topological data analysis to estimate bone strength from micro-CT as a surrogate for advanced imaging</td><td>[2512.03880](http://arxiv.org/pdf/2512.03880)</td><td>◆ Accurate bone strength prediction is essential for assessing fracture risk, particularly in aging populations and individuals with osteoporosis.
◆ Bone imaging has evolved from X-rays and DXA to clinical computed tomography (CT), and now to advanced modalities such as high-resolution peripheral quantitative CT and synchrotron radiation CT, which offer unprecedented resolution of bone microarchitecture.
◆ However, analytical methods have not kept pace with these imaging advances.</td></tr>
<tr><td>2025-12-03</td><td>DINO-RotateMatch: A Rotation-Aware Deep Framework for Robust Image Matching in Large-Scale 3D Reconstruction</td><td>[2512.03715](http://arxiv.org/pdf/2512.03715)</td><td>◆ This paper presents DINO-RotateMatch, a deep-learning framework designed to address the chal lenges of image matching in large-scale 3D reconstruction from unstructured Internet images.
◆ The   method integrates a dataset-adaptive image pairing strategy with rotation-aware keypoint extraction and   matching.
◆ DINO is employed to retrieve semantically relevant image pairs in large collections, while   rotation-based augmentation captures orientation-dependent local features using ALIKED and Light Glue.</td></tr>
<tr><td>2025-12-03</td><td>A Novel Approach to Tomato Harvesting Using a Hybrid Gripper with Semantic Segmentation and Keypoint Detection</td><td>[2512.03684](http://arxiv.org/pdf/2512.03684)</td><td>◆ This paper presents an autonomous tomato-harvesting system built around a hybrid robotic gripper that combines six soft auxetic fingers with a rigid exoskeleton and a latex basket to achieve gentle, cage-like grasping.
◆ The gripper is driven by a servo-actuated Scotch--yoke mechanism, and includes separator leaves that form a conical frustum for fruit isolation, with an integrated micro-servo cutter for pedicel cutting.
◆ For perception, an RGB--D camera and a Detectron2-based pipeline perform semantic segmentation of ripe/unripe tomatoes and keypoint localization of the pedicel and fruit center under occlusion and variable illumination.</td></tr>
<tr><td>2025-12-03</td><td>Linking Aneurysmal Geometry and Hemodynamics Using Computational Fluid Dynamics</td><td>[2512.03660](http://arxiv.org/pdf/2512.03660)</td><td>◆ The development and progression of abdominal aortic aneurysms (AAA) are related to complex flow patterns and wall-shear-driven mechanobiological stimuli, yet the quantitative relationship between aneurysmal geometry and hemodynamics remains poorly defined.
◆ In this study, we conducted a comprehensive hemodynamic analysis of 74 patient-specific abdominal aortas, representing one of the largest Computational Fluid Dynamics (CFD) cohorts reported to date.
◆ A multiscale framework coupling 0D-1D systemic circulation models with 3D stabilized finite-element simulations is used to generate physiologically consistent boundary conditions and high-fidelity flow fields.</td></tr>
<tr><td>2025-12-03</td><td>Memory-Guided Point Cloud Completion for Dental Reconstruction</td><td>[2512.03598](http://arxiv.org/pdf/2512.03598)</td><td>◆ Partial dental point clouds often suffer from large missing regions caused by occlusion and limited scanning views, which bias encoder-only global features and force decoders to hallucinate structures.
◆ We propose a retrieval-augmented framework for tooth completion that integrates a prototype memory into standard encoder--decoder pipelines.
◆ After encoding a partial input into a global descriptor, the model retrieves the nearest manifold prototype from a learnable memory and fuses it with the query feature through confidence-gated weighting before decoding.</td></tr>
<tr><td>2025-12-03</td><td>Hierarchical Attention for Sparse Volumetric Anomaly Detection in Subclinical Keratoconus</td><td>[2512.03346](http://arxiv.org/pdf/2512.03346)</td><td>◆ The detection of weak, spatially distributed anomalies in volumetric medical imaging remains a major challenge.
◆ The subtle, non-adjacent nature of early disease signals is often lost due to suboptimal architectural inductive biases: 2D/3D CNNs impose strong locality, while ViTs diffuse unconstrained global attention.
◆ This conflict leaves the optimal inductive structure for robust, sparse volumetric pattern recognition unresolved.</td></tr>
<tr><td>2025-12-02</td><td>The Convex Matching Distance in Multiparameter Persistence</td><td>[2512.02944](http://arxiv.org/pdf/2512.02944)</td><td>◆ We introduce the convex matching distance, a novel metric for comparing functions with values in the real plane.
◆ This metric measures the maximal bottleneck distance between the persistence diagrams associated with the convex combinations of the two function components.
◆ Similarly to the traditional matching distance, the convex matching distance aggregates the information provided by two real-valued components.</td></tr>
<tr><td>2025-12-02</td><td>Learning Multimodal Embeddings for Traffic Accident Prediction and Causal Estimation</td><td>[2512.02920](http://arxiv.org/pdf/2512.02920)</td><td>◆ We consider analyzing traffic accident patterns using both road network data and satellite images aligned to road graph nodes.
◆ Previous work for predicting accident occurrences relies primarily on road network structural features while overlooking physical and environmental information from the road surface and its surroundings.
◆ In this work, we construct a large multimodal dataset across six U.S.</td></tr>
<tr><td>2025-12-02</td><td>Terahertz Emission from Spintronic Stack Nanodecorated with Drop-Cast Core-Shell Plasmonic Nanoparticles</td><td>[2512.02889](http://arxiv.org/pdf/2512.02889)</td><td>◆ Spintronic emitters promise to revolutionise terahertz (THz) sources by converting ultrafast optical pulses into broadband THz radiation without phase-matching constraints.
◆ Because the conversion relies on spin-current injection across a nanometre-thin magnetic layer, its efficiency is ordinarily limited by weak optical coupling.
◆ Here, we present a demonstration of a drop-casting based approach to introduce ultrafast plasmonic-mediated coupling: a sparse-layer of silica-gold core-shell nanoparticles is deposited directly onto a W/Fe/Pt spintronic trilayer.</td></tr>
<tr><td>2025-12-02</td><td>A Comparative Study on How Data Normalization Affects Zero-Shot Generalization in Time Series Foundation Models</td><td>[2512.02833](http://arxiv.org/pdf/2512.02833)</td><td>◆ We investigate input normalization methods for Time-Series Foundation Models (TSFMs).
◆ While normalization is well-studied in dataset-specific time-series models, it remains overlooked in TSFMs where generalization is critical.
◆ Time-series data, unlike text or images, exhibits significant scale variation across domains and channels, coupled with non-stationarity, can undermine TSFM performance regardless of architectural complexity.</td></tr>
<tr><td>2025-12-02</td><td>From Navigation to Refinement: Revealing the Two-Stage Nature of Flow-based Diffusion Models through Oracle Velocity</td><td>[2512.02826](http://arxiv.org/pdf/2512.02826)</td><td>◆ Flow-based diffusion models have emerged as a leading paradigm for training generative models across images and videos.
◆ However, their memorization-generalization behavior remains poorly understood.
◆ In this work, we revisit the flow matching (FM) objective and study its marginal velocity field, which admits a closed-form expression, allowing exact computation of the oracle FM target.</td></tr>
<tr><td>2025-12-02</td><td>Direct observational evidence that higher-luminosity type 1 active galactic nuclei are most commonly triggered by galaxy mergers</td><td>[2512.02805](http://arxiv.org/pdf/2512.02805)</td><td>◆ We examine the connection between galaxy mergers and the triggering of active galactic nuclei (AGNs) using a sample of 614 type 1 AGNs at $z&lt;0.07$, along with a control sample of inactive galaxies matched to the AGNs for comparison.
◆ We used tidal features, detected in deep images from the DESI Legacy Imaging Survey, as direct evidence of recent mergers.
◆ We find that the fraction of type 1 AGN hosts with tidal features ($f_T$) is higher for AGNs with higher luminosities and (to a lesser extent) more massive black holes.</td></tr>
<tr><td>2025-12-02</td><td>Exploring Definitions of Quality and Diversity in Sonic Measurement Spaces</td><td>[2512.02783](http://arxiv.org/pdf/2512.02783)</td><td>◆ Digital sound synthesis presents the opportunity to explore vast parameter spaces containing millions of configurations.
◆ Quality diversity (QD) evolutionary algorithms offer a promising approach to harness this potential, yet their success hinges on appropriate sonic feature representations.
◆ Existing QD methods predominantly employ handcrafted descriptors or supervised classifiers, potentially introducing unintended exploration biases and constraining discovery to familiar sonic regions.</td></tr>
<tr><td>2025-12-02</td><td>Diffusion-Prior Split Gibbs Sampling for Synthetic Aperture Radar Imaging under Incomplete Measurements</td><td>[2512.02768](http://arxiv.org/pdf/2512.02768)</td><td>◆ Synthetic aperture radar (SAR) imaging plays a critical role in all-weather, day-and-night remote sensing, yet reconstruction is often challenged by noise, undersampling, and complex scattering scenarios.
◆ Conventional methods, including matched filtering and sparsity-based compressed sensing, are limited in capturing intricate scene structures and frequently suffer from artifacts, elevated sidelobes, and loss of fine details.
◆ Recent diffusion models have demonstrated superior capability in representing high-order priors; however, existing diffusion-based SAR methods still yield degraded reconstructions due to oversimplified likelihood approximations in guided sampling.</td></tr>
<tr><td>2025-12-02</td><td>Beyond Paired Data: Self-Supervised UAV Geo-Localization from Reference Imagery Alone</td><td>[2512.02737](http://arxiv.org/pdf/2512.02737)</td><td>◆ Image-based localization in GNSS-denied environments is critical for UAV autonomy.
◆ Existing state-of-the-art approaches rely on matching UAV images to geo-referenced satellite images; however, they typically require large-scale, paired UAV-satellite datasets for training.
◆ Such data are costly to acquire and often unavailable, limiting their applicability.</td></tr>
<tr><td>2025-12-02</td><td>GeoBridge: A Semantic-Anchored Multi-View Foundation Model Bridging Images and Text for Geo-Localization</td><td>[2512.02697](http://arxiv.org/pdf/2512.02697)</td><td>◆ Cross-view geo-localization infers a location by retrieving geo-tagged reference images that visually correspond to a query image.
◆ However, the traditional satellite-centric paradigm limits robustness when high-resolution or up-to-date satellite imagery is unavailable.
◆ It further underexploits complementary cues across views (e.g., drone, satellite, and street) and modalities (e.g., language and image).</td></tr>
<tr><td>2025-12-01</td><td>SARL: Spatially-Aware Self-Supervised Representation Learning for Visuo-Tactile Perception</td><td>[2512.01908](http://arxiv.org/pdf/2512.01908)</td><td>◆ Contact-rich robotic manipulation requires representations that encode local geometry.
◆ Vision provides global context but lacks direct measurements of properties such as texture and hardness, whereas touch supplies these cues.
◆ Modern visuo-tactile sensors capture both modalities in a single fused image, yielding intrinsically aligned inputs that are well suited to manipulation tasks requiring visual and tactile information.</td></tr>
<tr><td>2025-12-01</td><td>Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching</td><td>[2512.01850](http://arxiv.org/pdf/2512.01850)</td><td>◆ Point cloud registration aligns multiple unposed point clouds into a common frame, and is a core step for 3D reconstruction and robot localization.
◆ In this work, we cast registration as conditional generation: a learned continuous, point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered.
◆ Unlike previous methods that conduct correspondence matching to estimate the transformation between a pair of point clouds and then optimize the pairwise transformations to realize multi-view registration, our model directly generates the registered point cloud.</td></tr>
<tr><td>2025-12-01</td><td>Envision: Benchmarking Unified Understanding &amp; Generation for Causal World Process Insights</td><td>[2512.01816](http://arxiv.org/pdf/2512.01816)</td><td>◆ Current multimodal models aim to transcend the limitations of single-modality representations by unifying understanding and generation, often using text-to-image (T2I) tasks to calibrate semantic consistency.
◆ However, their reliance on static, single-image generation in training and evaluation leads to overfitting to static pattern matching and semantic fusion, while fundamentally hindering their ability to model dynamic processes that unfold over time.
◆ To address these constraints, we propose Envision-a causal event progression benchmark for chained text-to-multi-image generation.</td></tr>
<tr><td>2025-12-01</td><td>ViT$^3$: Unlocking Test-Time Training in Vision</td><td>[2512.01643](http://arxiv.org/pdf/2512.01643)</td><td>◆ Test-Time Training (TTT) has recently emerged as a promising direction for efficient sequence modeling.
◆ TTT reformulates attention operation as an online learning problem, constructing a compact inner model from key-value pairs at test time.
◆ This reformulation opens a rich and flexible design space while achieving linear computational complexity.</td></tr>
<tr><td>2025-12-01</td><td>Depth Matching Method Based on ShapeDTW for Oil-Based Mud Imager</td><td>[2512.01611](http://arxiv.org/pdf/2512.01611)</td><td>◆ In well logging operations using the oil-based mud (OBM) microresistivity imager, which employs an interleaved design with upper and lower pad sets, depth misalignment issues persist between the pad images even after velocity correction.
◆ This paper presents a depth matching method for borehole images based on the Shape Dynamic Time Warping (ShapeDTW) algorithm.
◆ The method extracts local shape features to construct a morphologically sensitive distance matrix, better preserving structural similarity between sequences during alignment.</td></tr>
<tr><td>2025-12-01</td><td>Semantic-aware Random Convolution and Source Matching for Domain Generalization in Medical Image Segmentation</td><td>[2512.01510](http://arxiv.org/pdf/2512.01510)</td><td>◆ We tackle the challenging problem of single-source domain generalization (DG) for medical image segmentation.
◆ To this end, we aim for training a network on one domain (e.g., CT) and directly apply it to a different domain (e.g., MR) without adapting the model and without requiring images or annotations from the new domain during training.
◆ We propose a novel method for promoting DG when training deep segmentation networks, which we call SRCSM.</td></tr>
<tr><td>2025-12-01</td><td>Non-Markovian dynamics in ice nucleation</td><td>[2512.01479](http://arxiv.org/pdf/2512.01479)</td><td>◆ In simulation studies of crystallisation, the size of the largest crystalline nucleus is often used as a reaction coordinate to monitor the progress of the nucleation process.
◆ Here, we investigate, for the case of homogeneous ice nucleation, whether the nucleus size exhibits Markovian dynamics, as assumed in classical nucleation theory.
◆ Using 300 independent nucleation trajectories generated by molecular dynamics, we evaluate the mean recurrence time required to reach selected values of the largest nucleus size.</td></tr>
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='3dgs'>3DGS</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-12-08</td><td>From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images</td><td>[2512.07527](http://arxiv.org/pdf/2512.07527)</td><td>◆ City-scale 3D reconstruction from satellite imagery presents the challenge of extreme viewpoint extrapolation, where our goal is to synthesize ground-level novel views from sparse orbital images with minimal parallax.
◆ This requires inferring nearly $90^\circ$ viewpoint gaps from image sources with severely foreshortened facades and flawed textures, causing state-of-the-art reconstruction engines such as NeRF and 3DGS to fail.
◆ To address this problem, we propose two design choices tailored for city structures and satellite inputs.</td></tr>
<tr><td>2025-12-08</td><td>AdLift: Lifting Adversarial Perturbations to Safeguard 3D Gaussian Splatting Assets Against Instruction-Driven Editing</td><td>[2512.07247](http://arxiv.org/pdf/2512.07247)</td><td>◆ Recent studies have extended diffusion-based instruction-driven 2D image editing pipelines to 3D Gaussian Splatting (3DGS), enabling faithful manipulation of 3DGS assets and greatly advancing 3DGS content creation.
◆ However, it also exposes these assets to serious risks of unauthorized editing and malicious tampering.
◆ Although imperceptible adversarial perturbations against diffusion models have proven effective for protecting 2D images, applying them to 3DGS encounters two major challenges: view-generalizable protection and balancing invisibility with protection capability.</td></tr>
<tr><td>2025-12-08</td><td>STRinGS: Selective Text Refinement in Gaussian Splatting</td><td>[2512.07230](http://arxiv.org/pdf/2512.07230)</td><td>◆ Text as signs, labels, or instructions is a critical element of real-world scenes as they can convey important contextual information.
◆ 3D representations such as 3D Gaussian Splatting (3DGS) struggle to preserve fine-grained text details, while achieving high visual fidelity.
◆ Small errors in textual element reconstruction can lead to significant semantic loss.</td></tr>
<tr><td>2025-12-08</td><td>SUCCESS-GS: Survey of Compactness and Compression for Efficient Static and Dynamic Gaussian Splatting</td><td>[2512.07197](http://arxiv.org/pdf/2512.07197)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a powerful explicit representation enabling real-time, high-fidelity 3D reconstruction and novel view synthesis.
◆ However, its practical use is hindered by the massive memory and computational demands required to store and render millions of Gaussians.
◆ These challenges become even more severe in 4D dynamic scenes.</td></tr>
<tr><td>2025-12-08</td><td>COREA: Coarse-to-Fine 3D Representation Alignment Between Relightable 3D Gaussians and SDF via Bidirectional 3D-to-3D Supervision</td><td>[2512.07107](http://arxiv.org/pdf/2512.07107)</td><td>◆ We present COREA, the first unified framework that jointly learns relightable 3D Gaussians and a Signed Distance Field (SDF) for accurate geometry reconstruction and faithful relighting.
◆ While recent 3D Gaussian Splatting (3DGS) methods have extended toward mesh reconstruction and physically-based rendering (PBR), their geometry is still learned from 2D renderings, leading to coarse surfaces and unreliable BRDF-lighting decomposition.
◆ To address these limitations, COREA introduces a coarse-to-fine bidirectional 3D-to-3D alignment strategy that allows geometric signals to be learned directly in 3D space.</td></tr>
<tr><td>2025-12-07</td><td>RAVE: Rate-Adaptive Visual Encoding for 3D Gaussian Splatting</td><td>[2512.07052](http://arxiv.org/pdf/2512.07052)</td><td>◆ Recent advances in neural scene representations have transformed immersive multimedia, with 3D Gaussian Splatting (3DGS) enabling real-time photorealistic rendering.
◆ Despite its efficiency, 3DGS suffers from large memory requirements and costly training procedures, motivating efforts toward compression.
◆ Existing approaches, however, operate at fixed rates, limiting adaptability to varying bandwidth and device constraints.</td></tr>
<tr><td>2025-12-07</td><td>RDSplat: Robust Watermarking Against Diffusion Editing for 3D Gaussian Splatting</td><td>[2512.06774](http://arxiv.org/pdf/2512.06774)</td><td>◆ 3D Gaussian Splatting (3DGS) has enabled the creation of digital assets and downstream applications, underscoring the need for robust copyright protection via digital watermarking.
◆ However, existing 3DGS watermarking methods remain highly vulnerable to diffusion-based editing, which can easily erase embedded provenance.
◆ This challenge highlights the urgent need for 3DGS watermarking techniques that are intrinsically resilient to diffusion-based editing.</td></tr>
<tr><td>2025-12-06</td><td>AGORA: Adversarial Generation Of Real-time Animatable 3D Gaussian Head Avatars</td><td>[2512.06438](http://arxiv.org/pdf/2512.06438)</td><td>◆ The generation of high-fidelity, animatable 3D human avatars remains a core challenge in computer graphics and vision, with applications in VR, telepresence, and entertainment.
◆ Existing approaches based on implicit representations like NeRFs suffer from slow rendering and dynamic inconsistencies, while 3D Gaussian Splatting (3DGS) methods are typically limited to static head generation, lacking dynamic control.
◆ We bridge this gap by introducing AGORA, a novel framework that extends 3DGS within a generative adversarial network to produce animatable avatars.</td></tr>
<tr><td>2025-12-05</td><td>TED-4DGS: Temporally Activated and Embedding-based Deformation for 4DGS Compression</td><td>[2512.05446](http://arxiv.org/pdf/2512.05446)</td><td>◆ Building on the success of 3D Gaussian Splatting (3DGS) in static 3D scene representation, its extension to dynamic scenes, commonly referred to as 4DGS or dynamic 3DGS, has attracted increasing attention.
◆ However, designing more compact and efficient deformation schemes together with rate-distortion-optimized compression strategies for dynamic 3DGS representations remains an underexplored area.
◆ Prior methods either rely on space-time 4DGS with overspecified, short-lived Gaussian primitives or on canonical 3DGS with deformation that lacks explicit temporal control.</td></tr>
<tr><td>2025-12-04</td><td>RobustSplat++: Decoupling Densification, Dynamics, and Illumination for In-the-Wild 3DGS</td><td>[2512.04815](http://arxiv.org/pdf/2512.04815)</td><td>◆ 3D Gaussian Splatting (3DGS) has gained significant attention for its real-time, photo-realistic rendering in novel-view synthesis and 3D modeling.
◆ However, existing methods struggle with accurately modeling in-the-wild scenes affected by transient objects and illuminations, leading to artifacts in the rendered images.
◆ We identify that the Gaussian densification process, while enhancing scene detail capture, unintentionally contributes to these artifacts by growing additional Gaussians that model transient disturbances and illumination variations.</td></tr>
<tr><td>2025-12-04</td><td>Gaussian Entropy Fields: Driving Adaptive Sparsity in 3D Gaussian Optimization</td><td>[2512.04542](http://arxiv.org/pdf/2512.04542)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a leading technique for novel view synthesis, demonstrating exceptional rendering efficiency.
◆ \replaced[]{Well-reconstructed surfaces can be characterized by low configurational entropy, where dominant primitives clearly define surface geometry while redundant components are suppressed.}{The key insight is that well-reconstructed surfaces naturally exhibit low configurational entropy, where dominant primitives clearly define surface geometry while suppressing redundant components.} Three complementary technical contributions are introduced: (1) entropy-driven surface modeling via entropy minimization for low configurational entropy in primitive distributions; (2) adaptive spatial regularization using the Surface Neighborhood Redundancy Index (SNRI) and image entropy-guided weighting; (3) multi-scale geometric preservation through competitive cross-scale entropy alignment.
◆ Extensive experiments demonstrate that GEF achieves competitive geometric precision on DTU and T\&amp;T benchmarks, while delivering superior rendering quality compared to existing methods on Mip-NeRF 360.</td></tr>
<tr><td>2025-12-03</td><td>ReCamDriving: LiDAR-Free Camera-Controlled Novel Trajectory Video Generation</td><td>[2512.03621](http://arxiv.org/pdf/2512.03621)</td><td>◆ We propose ReCamDriving, a purely vision-based, camera-controlled novel-trajectory video generation framework.
◆ While repair-based methods fail to restore complex artifacts and LiDAR-based approaches rely on sparse and incomplete cues, ReCamDriving leverages dense and scene-complete 3DGS renderings for explicit geometric guidance, achieving precise camera-controllable generation.
◆ To mitigate overfitting to restoration behaviors when conditioned on 3DGS renderings, ReCamDriving adopts a two-stage training paradigm: the first stage uses camera poses for coarse control, while the second stage incorporates 3DGS renderings for fine-grained viewpoint and geometric guidance.</td></tr>
<tr><td>2025-12-03</td><td>What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models</td><td>[2512.03422](http://arxiv.org/pdf/2512.03422)</td><td>◆ In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models.
◆ While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance.
◆ Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence.</td></tr>
<tr><td>2025-12-02</td><td>Flux4D: Flow-based Unsupervised 4D Reconstruction</td><td>[2512.03210](http://arxiv.org/pdf/2512.03210)</td><td>◆ Reconstructing large-scale dynamic scenes from visual observations is a fundamental challenge in computer vision, with critical implications for robotics and autonomous systems.
◆ While recent differentiable rendering methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have achieved impressive photorealistic reconstruction, they suffer from scalability limitations and require annotations to decouple actor motion.
◆ Existing self-supervised methods attempt to eliminate explicit annotations by leveraging motion cues and geometric priors, yet they remain constrained by per-scene optimization and sensitivity to hyperparameter tuning.</td></tr>
<tr><td>2025-12-02</td><td>EGGS: Exchangeable 2D/3D Gaussian Splatting for Geometry-Appearance Balanced Novel View Synthesis</td><td>[2512.02932](http://arxiv.org/pdf/2512.02932)</td><td>◆ Novel view synthesis (NVS) is crucial in computer vision and graphics, with wide applications in AR, VR, and autonomous driving.
◆ While 3D Gaussian Splatting (3DGS) enables real-time rendering with high appearance fidelity, it suffers from multi-view inconsistencies, limiting geometric accuracy.
◆ In contrast, 2D Gaussian Splatting (2DGS) enforces multi-view consistency but compromises texture details.</td></tr>
<tr><td>2025-12-02</td><td>PolarGuide-GSDR: 3D Gaussian Splatting Driven by Polarization Priors and Deferred Reflection for Real-World Reflective Scenes</td><td>[2512.02664](http://arxiv.org/pdf/2512.02664)</td><td>◆ Polarization-aware Neural Radiance Fields (NeRF) enable novel view synthesis of specular-reflection scenes but face challenges in slow training, inefficient rendering, and strong dependencies on material/viewpoint assumptions.
◆ However, 3D Gaussian Splatting (3DGS) enables real-time rendering yet struggles with accurate reflection reconstruction from reflection-geometry entanglement, adding a deferred reflection module introduces environment map dependence.
◆ We address these limitations by proposing PolarGuide-GSDR, a polarization-forward-guided paradigm establishing a bidirectional coupling mechanism between polarization and 3DGS: first 3DGS&#x27;s geometric priors are leveraged to resolve polarization ambiguity, and then the refined polarization information cues are used to guide 3DGS&#x27;s normal and spherical harmonic representation.</td></tr>
<tr><td>2025-12-02</td><td>VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</td><td>[2512.02293](http://arxiv.org/pdf/2512.02293)</td><td>◆ We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction.
◆ Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations.
◆ Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states.</td></tr>
<tr><td>2025-12-01</td><td>SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting</td><td>[2512.02172](http://arxiv.org/pdf/2512.02172)</td><td>◆ 3D Gaussian Splatting (3DGS) enables high-quality novel view synthesis, motivating interest in generating higher-resolution renders than those available during training.
◆ A natural strategy is to apply super-resolution (SR) to low-resolution (LR) input views, but independently enhancing each image introduces multi-view inconsistencies, leading to blurry renders.
◆ Prior methods attempt to mitigate these inconsistencies through learned neural components, temporally consistent video priors, or joint optimization on LR and SR views, but all uniformly apply SR across every image.</td></tr>
<tr><td>2025-12-01</td><td>EGG-Fusion: Efficient 3D Reconstruction with Geometry-aware Gaussian Surfel on the Fly</td><td>[2512.01296](http://arxiv.org/pdf/2512.01296)</td><td>◆ Real-time 3D reconstruction is a fundamental task in computer graphics.
◆ Recently, differentiable-rendering-based SLAM system has demonstrated significant potential, enabling photorealistic scene rendering through learnable scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS).
◆ Current differentiable rendering methods face dual challenges in real-time computation and sensor noise sensitivity, leading to degraded geometric fidelity in scene reconstruction and limited practicality.</td></tr>
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='depth-estimation'>Depth Estimation</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-12-08</td><td>More than Segmentation: Benchmarking SAM 3 for Segmentation, 3D Perception, and Reconstruction in Robotic Surgery</td><td>[2512.07596](http://arxiv.org/pdf/2512.07596)</td><td>◆ The recent Segment Anything Model (SAM) 3 has introduced significant advancements over its predecessor, SAM 2, particularly with the integration of language-based segmentation and enhanced 3D perception capabilities.
◆ SAM 3 supports zero-shot segmentation across a wide range of prompts, including point, bounding box, and language-based prompts, allowing for more flexible and intuitive interactions with the model.
◆ In this empirical evaluation, we assess the performance of SAM 3 in robot-assisted surgery, benchmarking its zero-shot segmentation with point and bounding box prompts and exploring its effectiveness in dynamic video tracking, alongside its newly introduced language prompt segmentation.</td></tr>
<tr><td>2025-12-07</td><td>Generalized Geometry Encoding Volume for Real-time Stereo Matching</td><td>[2512.06793](http://arxiv.org/pdf/2512.06793)</td><td>◆ Real-time stereo matching methods primarily focus on enhancing in-domain performance but often overlook the critical importance of generalization in real-world applications.
◆ In contrast, recent stereo foundation models leverage monocular foundation models (MFMs) to improve generalization, but typically suffer from substantial inference latency.
◆ To address this trade-off, we propose Generalized Geometry Encoding Volume (GGEV), a novel real-time stereo matching network that achieves strong generalization.</td></tr>
<tr><td>2025-12-07</td><td>CoT4Det: A Chain-of-Thought Framework for Perception-Oriented Vision-Language Tasks</td><td>[2512.06663](http://arxiv.org/pdf/2512.06663)</td><td>◆ Large Vision-Language Models (LVLMs) have demonstrated remarkable success in a broad range of vision-language tasks, such as general visual question answering and optical character recognition (OCR).
◆ However, their performance on perception-centric tasks -- such as object detection, semantic segmentation, and depth estimation -- remains significantly inferior to that of task-specific expert models.
◆ For example, Qwen2.5-VL-7B-Instruct achieves only 19% mAP on COCO2017 val, particularly struggling with dense scenes and small object recall.</td></tr>
<tr><td>2025-12-06</td><td>Human3R: Incorporating Human Priors for Better 3D Dynamic Reconstruction from Monocular Videos</td><td>[2512.06368](http://arxiv.org/pdf/2512.06368)</td><td>◆ Monocular dynamic video reconstruction faces significant challenges in dynamic human scenes due to geometric inconsistencies and resolution degradation issues.
◆ Existing methods lack 3D human structural understanding, producing geometrically inconsistent results with distorted limb proportions and unnatural human-object fusion, while memory-constrained downsampling causes human boundary drift toward background geometry.
◆ To address these limitations, we propose to incorporate hybrid geometric priors that combine SMPL human body models with monocular depth estimation.</td></tr>
<tr><td>2025-12-05</td><td>See in Depth: Training-Free Surgical Scene Segmentation with Monocular Depth Priors</td><td>[2512.05529](http://arxiv.org/pdf/2512.05529)</td><td>◆ Pixel-wise segmentation of laparoscopic scenes is essential for computer-assisted surgery but difficult to scale due to the high cost of dense annotations.
◆ We propose depth-guided surgical scene segmentation (DepSeg), a training-free framework that utilizes monocular depth as a geometric prior together with pretrained vision foundation models.
◆ DepSeg first estimates a relative depth map with a pretrained monocular depth estimation network and proposes depth-guided point prompts, which SAM2 converts into class-agnostic masks.</td></tr>
<tr><td>2025-12-05</td><td>YOLO and SGBM Integration for Autonomous Tree Branch Detection and Depth Estimation in Radiata Pine Pruning Applications</td><td>[2512.05412](http://arxiv.org/pdf/2512.05412)</td><td>◆ Manual pruning of radiata pine trees poses significant safety risks due to extreme working heights and challenging terrain.
◆ This paper presents a computer vision framework that integrates YOLO object detection with Semi-Global Block Matching (SGBM) stereo vision for autonomous drone-based pruning operations.
◆ Our system achieves precise branch detection and depth estimation using only stereo camera input, eliminating the need for expensive LiDAR sensors.</td></tr>
<tr><td>2025-12-05</td><td>Genetic Algorithms For Parameter Optimization for Disparity Map Generation of Radiata Pine Branch Images</td><td>[2512.05410](http://arxiv.org/pdf/2512.05410)</td><td>◆ Traditional stereo matching algorithms like Semi-Global Block Matching (SGBM) with Weighted Least Squares (WLS) filtering offer speed advantages over neural networks for UAV applications, generating disparity maps in approximately 0.5 seconds per frame.
◆ However, these algorithms require meticulous parameter tuning.
◆ We propose a Genetic Algorithm (GA) based parameter optimization framework that systematically searches for optimal parameter configurations for SGBM and WLS, enabling UAVs to measure distances to tree branches with enhanced precision while maintaining processing efficiency.</td></tr>
<tr><td>2025-12-04</td><td>Ground state energy and phase transitions of Long-range XXZ using VQE</td><td>[2512.04615](http://arxiv.org/pdf/2512.04615)</td><td>◆ The variational quantum eigen solver (VQE), has been widely used to find the ground state energy of different Hamiltonians with no analytical solutions and are classically difficult to compute.
◆ In our work, we have used VQE to identify the phase transition boundary for an infinite order phase transition.
◆ We use long-range XXZ (LRXXZ) chain for our study.</td></tr>
<tr><td>2025-12-04</td><td>Supramolecular approach-based intermolecular interaction energy calculations using quantum phase estimation algorithm</td><td>[2512.04587](http://arxiv.org/pdf/2512.04587)</td><td>◆ Accurate computation of non-covalent, intermolecular interaction energies is important to understand various chemical phenomena, and quantum computers are anticipated to accelerate it.
◆ Although the state-of-the-art quantum computers are still noisy and intermediate-scale ones, development of theoretical frameworks those are expected to work on a fault-tolerant quantum computer is an urgent issue.
◆ In this work, we explore resource-efficient implementation of the quantum phase estimation-based complete active space configuration interaction (QPE-CASCI) calculations, with the aid of the second-order Møller--Plesset perturbation theory (MP2)-based active space selection with Boys localized orbitals.</td></tr>
<tr><td>2025-12-04</td><td>COOPER: A Unified Model for Cooperative Perception and Reasoning in Spatial Intelligence</td><td>[2512.04563](http://arxiv.org/pdf/2512.04563)</td><td>◆ Visual Spatial Reasoning is crucial for enabling Multimodal Large Language Models (MLLMs) to understand object properties and spatial relationships, yet current models still struggle with 3D-aware reasoning.
◆ Existing approaches typically enhance either perception, by augmenting RGB inputs with auxiliary modalities such as depth and segmentation, or reasoning, by training on spatial VQA datasets and applying reinforcement learning, and thus treat these two aspects in isolation.
◆ In this work, we investigate whether a unified MLLM can develop an intrinsic ability to enhance spatial perception and, through adaptive interleaved reasoning, achieve stronger spatial intelligence.</td></tr>
<tr><td>2025-12-04</td><td>MASE: Interpretable NLP Models via Model-Agnostic Saliency Estimation</td><td>[2512.04386](http://arxiv.org/pdf/2512.04386)</td><td>◆ Deep neural networks (DNNs) have made significant strides in Natural Language Processing (NLP), yet their interpretability remains elusive, particularly when evaluating their intricate decision-making processes.
◆ Traditional methods often rely on post-hoc interpretations, such as saliency maps or feature visualization, which might not be directly applicable to the discrete nature of word data in NLP.
◆ Addressing this, we introduce the Model-agnostic Saliency Estimation (MASE) framework.</td></tr>
<tr><td>2025-12-04</td><td>MAFNet:Multi-frequency Adaptive Fusion Network for Real-time Stereo Matching</td><td>[2512.04358](http://arxiv.org/pdf/2512.04358)</td><td>◆ Existing stereo matching networks typically rely on either cost-volume construction based on 3D convolutions or deformation methods based on iterative optimization.
◆ The former incurs significant computational overhead during cost aggregation, whereas the latter often lacks the ability to model non-local contextual information.
◆ These methods exhibit poor compatibility on resource-constrained mobile devices, limiting their deployment in real-time applications.</td></tr>
<tr><td>2025-12-03</td><td>Gamma-from-Mono: Road-Relative, Metric, Self-Supervised Monocular Geometry for Vehicular Applications</td><td>[2512.04303](http://arxiv.org/pdf/2512.04303)</td><td>◆ Accurate perception of the vehicle&#x27;s 3D surroundings, including fine-scale road geometry, such as bumps, slopes, and surface irregularities, is essential for safe and comfortable vehicle control.
◆ However, conventional monocular depth estimation often oversmooths these features, losing critical information for motion planning and stability.
◆ To address this, we introduce Gamma-from-Mono (GfM), a lightweight monocular geometry estimation method that resolves the projective ambiguity in single-camera reconstruction by decoupling global and local structure.</td></tr>
<tr><td>2025-12-03</td><td>Unique Lives, Shared World: Learning from Single-Life Videos</td><td>[2512.04085](http://arxiv.org/pdf/2512.04085)</td><td>◆ We introduce the &quot;single-life&quot; learning paradigm, where we train a distinct vision model exclusively on egocentric videos captured by one individual.
◆ We leverage the multiple viewpoints naturally captured within a single life to learn a visual encoder in a self-supervised manner.
◆ Our experiments demonstrate three key findings.</td></tr>
<tr><td>2025-12-03</td><td>SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL</td><td>[2512.04069](http://arxiv.org/pdf/2512.04069)</td><td>◆ Vision Language Models (VLMs) demonstrate strong qualitative visual understanding, but struggle with metrically precise spatial reasoning required for embodied applications.
◆ The agentic paradigm promises that VLMs can use a wide variety of tools that could augment these capabilities, such as depth estimators, segmentation models, and pose estimators.
◆ Yet it remains an open challenge how to realize this vision without solely relying on handcrafted prompting strategies or enforcing fixed, predefined tool pipelines that limit VLMs&#x27; ability to discover optimal tool-use patterns.</td></tr>
<tr><td>2025-12-03</td><td>Approximate Optimal Active Learning of Decision Trees</td><td>[2512.03971](http://arxiv.org/pdf/2512.03971)</td><td>◆ We consider the problem of actively learning an unknown binary decision tree using only membership queries, a setting in which the learner must reason about a large hypothesis space while maintaining formal guarantees.
◆ Rather than enumerating candidate trees or relying on heuristic impurity or entropy measures, we encode the entire space of bounded-depth decision trees symbolically in SAT formulas.
◆ We propose a symbolic method for active learning of decision trees, in which approximate model counting is used to estimate the reduction of the hypothesis space caused by each potential query, enabling near-optimal query selection without full model enumeration.</td></tr>
<tr><td>2025-12-03</td><td>MDE-AgriVLN: Agricultural Vision-and-Language Navigation with Monocular Depth Estimation</td><td>[2512.03958](http://arxiv.org/pdf/2512.03958)</td><td>◆ Agricultural robots are serving as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily relying on manual operations or railway systems for movement.
◆ The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling a robot to navigate to a target position following a natural language instruction.
◆ Unlike human binocular vision, most agricultural robots are only given a single camera for monocular vision, which results in limited spatial perception.</td></tr>
<tr><td>2025-12-03</td><td>Three-dimensional modelling of drag anchor penetration using the material point method</td><td>[2512.03632](http://arxiv.org/pdf/2512.03632)</td><td>◆ Drag embedment anchors are a key threat to buried subsea linear infrastructure, such as power/data cables and pipelines.
◆ For cables, selecting a burial depth is a compromise between protecting the cable from anchor strike and the increased cost of deeper installation.
◆ This presents an efficient large deformation, elasto-plastic Material Point Method-based soil-structure interaction predictive tool for the estimation of anchor penetration based on Cone Penetration Test (CPT) site investigation data.</td></tr>
<tr><td>2025-12-03</td><td>Postseismicity of slow-slip doublets discerned on the outermost of the Nankai Trough subduction megathrust</td><td>[2512.03559](http://arxiv.org/pdf/2512.03559)</td><td>◆ Despite dissimilar slip rates, slow earthquakes are faulting as ordinary earthquakes are.
◆ It is therefore physically natural that slow earthquakes also cause postseismic motions similarly to ordinary earthquakes, even though coseismic and postseismic slips remain undifferentiated for slow earthquakes.
◆ We pursue the slow-earthquake postseismicity based on the analysis of a fault slip beneath the Bungo Channel, the westernmost region of the Nankai Trough subduction zone in southwestern Japan.</td></tr>
<tr><td>2025-12-03</td><td>Generalization Evaluation of Deep Stereo Matching Methods for UAV-Based Forestry Applications</td><td>[2512.03427](http://arxiv.org/pdf/2512.03427)</td><td>◆ Autonomous UAV forestry operations require robust depth estimation methods with strong cross-domain generalization.
◆ However, existing evaluations focus on urban and indoor scenarios, leaving a critical gap for specialized vegetation-dense environments.
◆ We present the first systematic zero-shot evaluation of eight state-of-the-art stereo methods--RAFT-Stereo, IGEV, IGEV++, BridgeDepth, StereoAnywhere, DEFOM (plus baseline methods ACVNet, PSMNet, TCstereo)--spanning iterative refinement, foundation model, and zero-shot adaptation paradigms.</td></tr>
<tr><td>2025-12-03</td><td>A three-dimensional model for the reversal in the local large-scale interstellar magnetic field</td><td>[2512.03332](http://arxiv.org/pdf/2512.03332)</td><td>◆ We probe the three-dimensional geometry of the large-scale Galactic magnetic field within 1 kpc of the Sun using the Dominion Radio Astrophysical Observatory (DRAO) Global Magneto-Ionic Medium Survey (GMIMS) of the Northern Sky (DRAGONS).
◆ DRAGONS is a new full polarization survey of the Northern sky from 350 to 1030 MHz covering declinations -20° &lt; $δ$ &lt; 90° and a component of GMIMS.
◆ The first moment of the Faraday depth spectra produced from DRAGONS above 500 MHz reveals large-angular-scale Faraday depth structures with signs that alternate only once in the Southern Galactic hemisphere and twice in the Northern hemisphere, patterns shared by other Faraday rotation datasets.</td></tr>
<tr><td>2025-12-03</td><td>DynamicVerse: A Physically-Aware Multimodal Framework for 4D World Modeling</td><td>[2512.03000](http://arxiv.org/pdf/2512.03000)</td><td>◆ Understanding the dynamic physical world, characterized by its evolving 3D structure, real-world motion, and semantic content with textual descriptions, is crucial for human-agent interaction and enables embodied agents to perceive and act within real environments with human-like capabilities.
◆ However, existing datasets are often derived from limited simulators or utilize traditional Structurefrom-Motion for up-to-scale annotation and offer limited descriptive captioning, which restricts the capacity of foundation models to accurately interpret real-world dynamics from monocular videos, commonly sourced from the internet.
◆ To bridge these gaps, we introduce DynamicVerse, a physical-scale, multimodal 4D world modeling framework for dynamic real-world video.</td></tr>
<tr><td>2025-12-02</td><td>BEVDilation: LiDAR-Centric Multi-Modal Fusion for 3D Object Detection</td><td>[2512.02972](http://arxiv.org/pdf/2512.02972)</td><td>◆ Integrating LiDAR and camera information in the bird&#x27;s eye view (BEV) representation has demonstrated its effectiveness in 3D object detection.
◆ However, because of the fundamental disparity in geometric accuracy between these sensors, indiscriminate fusion in previous methods often leads to degraded performance.
◆ In this paper, we propose BEVDilation, a novel LiDAR-centric framework that prioritizes LiDAR information in the fusion.</td></tr>
<tr><td>2025-12-02</td><td>DF-Mamba: Deformable State Space Modeling for 3D Hand Pose Estimation in Interactions</td><td>[2512.02727](http://arxiv.org/pdf/2512.02727)</td><td>◆ Modeling daily hand interactions often struggles with severe occlusions, such as when two hands overlap, which highlights the need for robust feature learning in 3D hand pose estimation (HPE).
◆ To handle such occluded hand images, it is vital to effectively learn the relationship between local image features (e.g., for occluded joints) and global context (e.g., cues from inter-joints, inter-hands, or the scene).
◆ However, most current 3D HPE methods still rely on ResNet for feature extraction, and such CNN&#x27;s inductive bias may not be optimal for 3D HPE due to its limited capability to model the global context.</td></tr>
<tr><td>2025-12-01</td><td>DepthScape: Authoring 2.5D Designs via Depth Estimation, Semantic Understanding, and Geometry Extraction</td><td>[2512.02263](http://arxiv.org/pdf/2512.02263)</td><td>◆ 2.5D effects, such as occlusion and perspective foreshortening, enhance visual dynamics and realism by incorporating 3D depth cues into 2D designs.
◆ However, creating such effects remains challenging and labor-intensive due to the complexity of depth perception.
◆ We introduce DepthScape, a human-AI collaborative system that facilitates 2.5D effect creation by directly placing design elements into 3D reconstructions.</td></tr>
<tr><td>2025-12-01</td><td>KM-ViPE: Online Tightly Coupled Vision-Language-Geometry Fusion for Open-Vocabulary Semantic SLAM</td><td>[2512.01889](http://arxiv.org/pdf/2512.01889)</td><td>◆ We present KM-ViPE (Knowledge Mapping Video Pose Engine), a real-time open-vocabulary SLAM framework for uncalibrated monocular cameras in dynamic environments.
◆ Unlike systems requiring depth sensors and offline calibration, KM-ViPE operates directly on raw RGB streams, making it ideal for ego-centric applications and harvesting internet-scale video data for training.
◆ KM-ViPE tightly couples DINO visual features with geometric constraints through a high-level features based adaptive robust kernel that handles both moving objects and movable static objects (e.g., moving furniture in ego-centric views).</td></tr>
<tr><td>2025-12-01</td><td>BlinkBud: Detecting Hazards from Behind via Sampled Monocular 3D Detection on a Single Earbud</td><td>[2512.01366](http://arxiv.org/pdf/2512.01366)</td><td>◆ Failing to be aware of speeding vehicles approaching from behind poses a huge threat to the road safety of pedestrians and cyclists.
◆ In this paper, we propose BlinkBud, which utilizes a single earbud and a paired phone to online detect hazardous objects approaching from behind of a user.
◆ The core idea is to accurately track visually identified objects utilizing a small number of sampled camera images taken from the earbud.</td></tr>
<tr><td>2025-12-01</td><td>Data-Driven Learnability Transition of Measurement-Induced Entanglement</td><td>[2512.01317](http://arxiv.org/pdf/2512.01317)</td><td>◆ Measurement-induced entanglement (MIE) captures how local measurements generate long-range quantum correlations and drive dynamical phase transitions in many-body systems.
◆ Yet estimating MIE experimentally remains challenging: direct evaluation requires extensive post-selection over measurement outcomes, raising the question of whether MIE is accessible with only polynomial resources.
◆ We address this challenge by reframing MIE detection as a data-driven learning problem that assumes no prior knowledge of state preparation.</td></tr>
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2025.12.09
