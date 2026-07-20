# SLAM领域最新论文 (2026.07.20)

> 每日自动更新SLAM领域的最新arXiv论文

> 使用说明: [点击查看](./docs/README.md#usage)

<details>
<summary>分类目录</summary>
<ol>
<li><a href='#lidar-slam'>LiDAR SLAM</a></li>
<li><a href='#visual-slam'>Visual SLAM</a></li>
<li><a href='#loop-closure'>Loop Closure</a></li>
<li><a href='#image-matching'>Image Matching</a></li>
<li><a href='#depth-estimation'>Depth Estimation</a></li>
<li><a href='#3dgs'>3DGS</a></li>
</ol>
</details>

<h2 id='lidar-slam'>LiDAR SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2026-07-16</td><td>Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency</td><td>[2607.14481](http://arxiv.org/pdf/2607.14481)</td><td>◆ 3D Gaussian Splatting (3DGS) has become the method of choice for reconstructing and real-time rendering of captured scenes.
◆ To capture a scene with good visual quality, continuous image sequences are usually combined with out-of-order shots for better scene coverage.
◆ Structure from motion can reconstruct such captures, but only after they are all available and often with high computational cost.</td></tr>
<tr><td>2026-07-15</td><td>Improving Map Consistency in Graph-Based LiDAR SLAM Through Information-Aware Odometry and Retroactive Loop Closure</td><td>[2607.13516](http://arxiv.org/pdf/2607.13516)</td><td>◆ High-quality maps are fundamental for robotics tasks such as navigation and planning.
◆ Although modern graph-based LiDAR SLAM systems achieve good trajectory accuracies, a low trajectory error alone does not guarantee geometrically consistent maps, particularly at revisit locations where missed loop closures and residual drift can produce local misalignments.
◆ In this work, we address the problem of jointly improving global trajectory estimation and local map quality in 3D LiDAR SLAM.</td></tr>
<tr><td>2026-07-15</td><td>WNOJ-LIO: A White-Noise-on-Jerk Motion-Prior EKF for High-Dynamic LiDAR-IMU Fusion</td><td>[2607.13405](http://arxiv.org/pdf/2607.13405)</td><td>◆ LiDAR-inertial odometry (LIO) is a key component of autonomous navigation, but high-dynamic driving exposes two coupled challenges: intra-scan motion distortion and vibration-contaminated inertial measurements.
◆ Most real-time LiDAR-inertial pipelines propagate the system state by integrating raw IMU measurements and then use the propagated trajectory for point cloud de-distortion, thereby propagating inertial noise into both the corrected scan and the subsequent scan-to-map registration.
◆ This paper presents WNOJ-LIO, a LiDAR-IMU fusion framework based on a White-Noise-on-Jerk (WNOJ) Extended Kalman Filter (EKF).</td></tr>
<tr><td>2026-07-15</td><td>Breaking Déjà Vu: Independent Auditing of Visual Place Recognition through Vision-Language Reasoning</td><td>[2607.12818](http://arxiv.org/pdf/2607.12818)</td><td>◆ Visual place recognition (VPR) is a key enabler of accurate localization and long-term autonomous navigation in robotics applications, such as loop closure detection for simultaneous localisation and mapping (SLAM).
◆ However, real-world VPR deployment relies on selecting an image matching threshold that balances precision and recall.
◆ These thresholds are typically tuned using labeled validation data and fixed during deployment, making them unreliable under environmental changes where ground truth is unavailable.</td></tr>
<tr><td>2026-07-14</td><td>PixelLoop: Shortcut Topological Navigation with Pixel-Level Loops</td><td>[2607.12811](http://arxiv.org/pdf/2607.12811)</td><td>◆ Although topological mapping and navigation have been studied extensively, the specific role and downstream effect of loop closures in purely topological representations has received relatively little attention.
◆ Importantly, loop closure over topological maps is distinct from loop closure over globally referenced trajectories and metric maps.
◆ Building on recent denser topologies grounded in pixel-level, relative 3D geometry, we propose PixelLoop which introduces loop closures directly in pixel space.</td></tr>
<tr><td>2026-07-14</td><td>DiffRadar: Differentiable Physics-Aware Radar SLAM with Gaussian Fields</td><td>[2607.12265](http://arxiv.org/pdf/2607.12265)</td><td>◆ Radar sensing is increasingly used in mobile systems because it operates reliably under poor lighting, adverse weather, and privacy-sensitive settings where cameras and LiDAR often fail.
◆ However, most existing radar SLAM systems estimate motion through scan matching on discretized radar heatmaps, which breaks geometric continuity and fails to capture key radar sensing properties, often leading to unstable pose estimation and degraded mapping in regenerate or dynamically changing environments.
◆ We present DiffRadar, a real-time radar SLAM system that models radar observations as a differentiable, physics-aware Gaussian field rather than discrete scans.</td></tr>
<tr><td>2026-07-13</td><td>GeoGS-SLAM: Online Monocular Reconstruction Using Gaussian Splatting with Geometric Priors</td><td>[2607.11184](http://arxiv.org/pdf/2607.11184)</td><td>◆ SLAM methods based on 3D Gaussian Splatting (3DGS) have demonstrated impressive tracking and mapping performance, but typically require additional geometric information from external depth sensors.
◆ Meanwhile, recent SLAM systems that leverage geometric priors from pre-trained feed-forward models enable real-time dense reconstruction, yet often discard original RGB information during optimization, thus degrading overall reconstruction quality.
◆ We present GeoGS-SLAM, an online monocular dense reconstruction system that combines the 3DGS-based map representation with learned geometric priors.</td></tr>
<tr><td>2026-07-13</td><td>Desc++: Efficient Descriptor Enhancement for Data Association in Existing Visual SLAM Systems</td><td>[2607.11099](http://arxiv.org/pdf/2607.11099)</td><td>◆ Reliable visual data association is fundamental to visual SLAM (V-SLAM), as it directly determines the quality of the camera pose estimation and map consistency.
◆ However, the handcrafted descriptors used by most mature real-time systems degrade under illumination and viewpoint changes, while learning-based front-ends that address this weakness typically require replacing the extraction-and-matching pipeline and introduce substantial computational overhead.
◆ Descriptor enhancement offers a compromise by refining existing descriptors within their original format, yet current methods rely on simplified attention mechanisms whose limited contextual modeling constrains the achievable matching quality.</td></tr>
<tr><td>2026-07-12</td><td>Mapping Pamir: Multi-Session Visual-Inertial SLAM and 3D Reconstruction of an Underwater Shipwreck</td><td>[2607.10925](http://arxiv.org/pdf/2607.10925)</td><td>◆ This paper presents a framework for multi-session mapping of underwater environments utilizing an affordable action camera.
◆ The Visual-Inertial data are augmented by water depth recordings from a dive computer.
◆ SVIn2, an open-source VI-SLAM framework, is utilized to generate a trajectory and a sparse reconstruction for each session.</td></tr>
<tr><td>2026-07-11</td><td>CSI-Assisted Edge SLAM Testbed Platform for 5G Connected Unmanned Autonomous Vehicles</td><td>[2607.10394](http://arxiv.org/pdf/2607.10394)</td><td>◆ The evolution from 5G towards 6G reinforces interest in connected robotics, where mobile robots offload compute-intensive tasks to edge servers over ultra-reliable low-latency communication (URLLC) links.
◆ Simultaneous localization and mapping (SLAM), a fundamental yet demanding robotics function, is increasingly considered for edge deployment within mobile edge computing (MEC) frameworks.
◆ In parallel, integrated sensing and communications (ISAC) enables the use of radio channel information, such as channel state information (CSI), as an additional sensing modality in radio-based SLAM.</td></tr>
</tbody>
</table>
</div>

<h2 id='visual-slam'>Visual SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2026-07-15</td><td>SeeSE3: Emergence of 3D Space in Vision Features</td><td>[2607.14228](http://arxiv.org/pdf/2607.14228)</td><td>◆ In this paper, we ask whether vision foundation models construct representations that reflect the intrinsic properties of 3D Euclidean space.
◆ Unlike previous works that probe 3D awareness of vision features by regressing image-centric quantities such as depth or normals, we investigate the relation between the structure of the space of visual features and the group of Euclidean transformations $SE(3)$.
◆ We propose a set of probes to evaluate this relation from both topological and geometric perspectives: a mutual neighborhood metric that measures the alignment between feature neighborhoods and spatial topology, and a Poincaré Adapter to test the linear accessibility of the geometry of camera motion from latent displacements in static scenes.</td></tr>
<tr><td>2026-07-14</td><td>Attitude Estimation Using Inertial and Barometric Measurements</td><td>[2607.13254](http://arxiv.org/pdf/2607.13254)</td><td>◆ Accurate and robust attitude estimation is a key challenge for autonomous vehicles, particularly in GNSS-denied conditions and during highly accelerated flight.
◆ In such conditions, Inertial Measurement Units (IMUs) alone are insufficient for reliable tilt estimation due to the ambiguity between gravitational and inertial accelerations.
◆ Although auxiliary velocity sensors such as GNSS, Pitot tubes, Doppler radar, or Visual Inertial Odometry are commonly used, they may be unavailable, intermittent, or costly.</td></tr>
<tr><td>2026-07-13</td><td>Self-Healing Visual Recovery for Autonomous Ground Vehicles Using Camera-Only Visual Odometry</td><td>[2607.11686](http://arxiv.org/pdf/2607.11686)</td><td>◆ Low-cost unmanned ground vehicles are often used in indoor places like warehouses, inspection corridors, and farm rows, where painted floor lines guide the robot.
◆ Line following is useful because it only needs one camera and little computing power, but it can fail when the line is blocked or turns sharply and goes out of view.
◆ Sensor-rich platforms tolerate this through hardware redundancy (LiDAR, GPS, multiple cameras), but camera-only systems must recover at runtime with no additional infrastructure.</td></tr>
<tr><td>2026-07-13</td><td>Desc++: Efficient Descriptor Enhancement for Data Association in Existing Visual SLAM Systems</td><td>[2607.11099](http://arxiv.org/pdf/2607.11099)</td><td>◆ Reliable visual data association is fundamental to visual SLAM (V-SLAM), as it directly determines the quality of the camera pose estimation and map consistency.
◆ However, the handcrafted descriptors used by most mature real-time systems degrade under illumination and viewpoint changes, while learning-based front-ends that address this weakness typically require replacing the extraction-and-matching pipeline and introduce substantial computational overhead.
◆ Descriptor enhancement offers a compromise by refining existing descriptors within their original format, yet current methods rely on simplified attention mechanisms whose limited contextual modeling constrains the achievable matching quality.</td></tr>
<tr><td>2026-07-07</td><td>SASGeo: Stability-Aware Semantic Map Localization for GNSS-Denied UAVs -- A Framework and Synthetic Proof of Concept</td><td>[2607.07737](http://arxiv.org/pdf/2607.07737)</td><td>◆ GNSS-denied unmanned aerial vehicles require occasional absolute position fixes to bound the drift of visual-inertial odometry.
◆ Cross-view image retrieval can provide such fixes, but raw appearance is sensitive to season, illumination, viewpoint, map age, and sensor modality.
◆ We propose \sas, a semantic map-localization framework that represents the environment through persistent structures such as roads, buildings, waterways, railways, intersections, and field boundaries.</td></tr>
<tr><td>2026-07-08</td><td>GeoGS-SLAM: Geometry-Only Gaussian Splatting for Dense Monocular SLAM</td><td>[2607.07452](http://arxiv.org/pdf/2607.07452)</td><td>◆ Dense visual SLAM is a fundamental problem in robotics.
◆ Recent advances in 3DGS have demonstrated its potential for dense SLAM.
◆ Existing 3DGS frameworks focus on both appearance and geometry modeling.</td></tr>
<tr><td>2026-07-08</td><td>PLED-VINS: A Point-Line Event-Based Visual Inertial SLAM for Dynamic Environments</td><td>[2607.07374](http://arxiv.org/pdf/2607.07374)</td><td>◆ Dynamic environments remain a fundamental challenge for visual SLAM, where unreliable observations from moving objects and rapid motion degrade state estimation accuracy.
◆ Although event cameras preserve fine-grained spatio-temporal information, most existing event-based SLAM frameworks still assume static scenes and lack approaches to estimate the reliability of features.
◆ To this end, we propose PLED-VINS, a monocular event camera-based visual-inertial SLAM framework that enables robust state estimation in dynamic environments.</td></tr>
<tr><td>2026-07-07</td><td>MiLSD: A Micro Line-Segment Detector for Resource-Constrained Devices</td><td>[2607.06600](http://arxiv.org/pdf/2607.06600)</td><td>◆ Line segment detection is a key building block in visual SLAM, 3D reconstruction, and industrial inspection.
◆ Recent deep learning methods have greatly improved accuracy, yet even the smallest models require several megabytes of memory, exceeding low-cost MCU capacity.
◆ This work investigates the maximum achievable accuracy under a sub-megabyte budget.</td></tr>
<tr><td>2026-07-07</td><td>Hilti-Trimble-Oxford Dataset: 360 Visual-Inertial Benchmark with Floor Plan Priors for SLAM and Localization</td><td>[2607.06464](http://arxiv.org/pdf/2607.06464)</td><td>◆ Automated progress monitoring on construction sites is an active area of research and development.
◆ Robot and human-carried mapping systems have been developed to build 3D maps of building and infrastructure projects.
◆ While LiDAR-based mapping systems achieve high accuracy, the cost of LiDAR can be prohibitive.</td></tr>
<tr><td>2026-07-07</td><td>Why does Deep Learning Improve Visual SLAM?</td><td>[2607.06023](http://arxiv.org/pdf/2607.06023)</td><td>◆ Visual SLAM is a well-established technology utilized in a wide range of real-world applications.
◆ However, its performance still degrades under challenging visual conditions, such as low texture, severe motion blur, and poor illumination.
◆ Systems based on deep learning outperform classical geometry-based ones and achieve state-of-the-art results by combining learned 2D data association and uncertainty with differentiable geometric optimization in recurrent architectures.</td></tr>
</tbody>
</table>
</div>

<h2 id='loop-closure'>Loop Closure</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2026-07-17</td><td>Are All Tokens Necessary for Visual Place Recognition? An Empirical Study of Token Reduction for Efficient Inference</td><td>[2607.15563](http://arxiv.org/pdf/2607.15563)</td><td>◆ Recent visual place recognition (VPR) methods based on vision transformers, particularly foundation models, have achieved remarkable recognition performance.
◆ However, these models process all visual tokens throughout the entire network, resulting in substantial computational overhead, which hinders their deployment in real-time and resource-constrained scenarios.
◆ A natural question thus arises: are all visual tokens necessary for VPR?</td></tr>
<tr><td>2026-07-16</td><td>Selectivity Drives Efficiency: Dataset Pruning for Visual Place Recognition</td><td>[2607.14897](http://arxiv.org/pdf/2607.14897)</td><td>◆ Recent visual place recognition (VPR) studies have increasingly relied on large-scale datasets to train more robust and discriminative models.
◆ Although this trend significantly improves recognition performance, it also introduces substantial storage and training costs, especially when new architectures or training strategies need to be repeatedly developed and evaluated.
◆ Dataset pruning (DP) provides a promising way to improve data efficiency by retaining only informative training data.</td></tr>
<tr><td>2026-07-16</td><td>Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency</td><td>[2607.14481](http://arxiv.org/pdf/2607.14481)</td><td>◆ 3D Gaussian Splatting (3DGS) has become the method of choice for reconstructing and real-time rendering of captured scenes.
◆ To capture a scene with good visual quality, continuous image sequences are usually combined with out-of-order shots for better scene coverage.
◆ Structure from motion can reconstruct such captures, but only after they are all available and often with high computational cost.</td></tr>
<tr><td>2026-07-15</td><td>Visual Place Recognition Using Rate-Encoded Spiking Neural Networks with Discrete STDP Learning</td><td>[2607.13584](http://arxiv.org/pdf/2607.13584)</td><td>◆ Spiking Neural Networks (SNNs) trained through unsupervised Spike-Timing-Dependent Plasticity (STDP) have been explored as solutions to visual loop closure problems, driven by the prospect of efficient on-device inference on neuromorphic devices.
◆ State-of-the-art STDP-based models deliver high classification accuracy but fail to reach the high Recall at 100% Precision (R@100P) needed for reliable autonomous navigation.
◆ We present a discrete, tensor-native implementation of the STDP-based SNN-VPR pipeline using PyTorch with snnTorch and evaluate it on a 100-place Nordland dataset using 15 independently-trained networks.</td></tr>
<tr><td>2026-07-15</td><td>Improving Map Consistency in Graph-Based LiDAR SLAM Through Information-Aware Odometry and Retroactive Loop Closure</td><td>[2607.13516](http://arxiv.org/pdf/2607.13516)</td><td>◆ High-quality maps are fundamental for robotics tasks such as navigation and planning.
◆ Although modern graph-based LiDAR SLAM systems achieve good trajectory accuracies, a low trajectory error alone does not guarantee geometrically consistent maps, particularly at revisit locations where missed loop closures and residual drift can produce local misalignments.
◆ In this work, we address the problem of jointly improving global trajectory estimation and local map quality in 3D LiDAR SLAM.</td></tr>
<tr><td>2026-07-15</td><td>Marker-free deformable registration and fusion for augmented reality-guided positive margin localization during tumor resection surgery</td><td>[2607.13343](http://arxiv.org/pdf/2607.13343)</td><td>◆ Positive margins in head and neck oncologic surgery require mapping specimen-side pathology findings to the patient resection bed.
◆ This is challenging because pathologists identify the positive margin on slices of the resected, deformed specimen, while surgeons must relocate the corresponding site on the resection bed using only verbal descriptions and no visual guidance.
◆ We present a marker-free augmented reality (AR) workflow for mapping a margin label from a three-dimensional specimen scan to the resection bed.</td></tr>
<tr><td>2026-07-15</td><td>Breaking Déjà Vu: Independent Auditing of Visual Place Recognition through Vision-Language Reasoning</td><td>[2607.12818](http://arxiv.org/pdf/2607.12818)</td><td>◆ Visual place recognition (VPR) is a key enabler of accurate localization and long-term autonomous navigation in robotics applications, such as loop closure detection for simultaneous localisation and mapping (SLAM).
◆ However, real-world VPR deployment relies on selecting an image matching threshold that balances precision and recall.
◆ These thresholds are typically tuned using labeled validation data and fixed during deployment, making them unreliable under environmental changes where ground truth is unavailable.</td></tr>
<tr><td>2026-07-14</td><td>PixelLoop: Shortcut Topological Navigation with Pixel-Level Loops</td><td>[2607.12811](http://arxiv.org/pdf/2607.12811)</td><td>◆ Although topological mapping and navigation have been studied extensively, the specific role and downstream effect of loop closures in purely topological representations has received relatively little attention.
◆ Importantly, loop closure over topological maps is distinct from loop closure over globally referenced trajectories and metric maps.
◆ Building on recent denser topologies grounded in pixel-level, relative 3D geometry, we propose PixelLoop which introduces loop closures directly in pixel space.</td></tr>
<tr><td>2026-07-14</td><td>DiffRadar: Differentiable Physics-Aware Radar SLAM with Gaussian Fields</td><td>[2607.12265](http://arxiv.org/pdf/2607.12265)</td><td>◆ Radar sensing is increasingly used in mobile systems because it operates reliably under poor lighting, adverse weather, and privacy-sensitive settings where cameras and LiDAR often fail.
◆ However, most existing radar SLAM systems estimate motion through scan matching on discretized radar heatmaps, which breaks geometric continuity and fails to capture key radar sensing properties, often leading to unstable pose estimation and degraded mapping in regenerate or dynamically changing environments.
◆ We present DiffRadar, a real-time radar SLAM system that models radar observations as a differentiable, physics-aware Gaussian field rather than discrete scans.</td></tr>
<tr><td>2026-07-15</td><td>SalientGS: Unified SfM-to-3DGS with Importance-Guided MCMC Gaussian Allocation</td><td>[2607.11285](http://arxiv.org/pdf/2607.11285)</td><td>◆ Reconstructing 3D scenes from unordered images remains bottlenecked by expensive Structure-from-Motion (SfM) preprocessing and frozen pose interfaces.
◆ We present SalientGS, a unified SfM-to-3D Gaussian Splatting (3DGS) pipeline.
◆ Its central contribution is importance-guided Markov Chain Monte Carlo (MCMC) Gaussian allocation, which aggregates multi-view residuals into per-Gaussian underfit and redundancy signals.</td></tr>
</tbody>
</table>
</div>

<h2 id='image-matching'>Image Matching</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2026-06-19</td><td>An Empirical Study of Handcrafted Feature Learning and Convolutional Neural Networks for Facial Expression Recognition</td><td>[2607.15288](http://arxiv.org/pdf/2607.15288)</td><td>◆ Facial expression recognition is an important computer vision task with applications in human--computer interaction, mental health monitoring, driver alert systems, and behavioral analysis.
◆ While convolutional neural networks (CNNs) dominate modern facial expression recognition, handcrafted feature descriptors such as Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP) remain useful classical baselines.
◆ This study compares HOG with Support Vector Machine (SVM), LBP with Logistic Regression, and a lightweight CNN across three facial expression datasets: FER-2013, CK+, and KDEF.</td></tr>
<tr><td>2026-07-15</td><td>Breaking Déjà Vu: Independent Auditing of Visual Place Recognition through Vision-Language Reasoning</td><td>[2607.12818](http://arxiv.org/pdf/2607.12818)</td><td>◆ Visual place recognition (VPR) is a key enabler of accurate localization and long-term autonomous navigation in robotics applications, such as loop closure detection for simultaneous localisation and mapping (SLAM).
◆ However, real-world VPR deployment relies on selecting an image matching threshold that balances precision and recall.
◆ These thresholds are typically tuned using labeled validation data and fixed during deployment, making them unreliable under environmental changes where ground truth is unavailable.</td></tr>
<tr><td>2026-07-06</td><td>Hybrid Deep Learning for Traceability and Classification of Industrial Slate Tiles</td><td>[2607.04811](http://arxiv.org/pdf/2607.04811)</td><td>◆ Applying deep learning to instance-aware reidentification of slate tiles and extraction site classification can improve production efficiency and quality control in the slate tile industry.
◆ These tasks are particularly important for handling natural materials where visual variability can make manual inspection costly and error-prone.
◆ We present a lightweight, hybrid deep learning approach that combines image matching and classification within a single framework.</td></tr>
<tr><td>2026-07-03</td><td>A Vision Based System for Guided and Collaborative Reconstruction of Fragmented Documents</td><td>[2607.03621](http://arxiv.org/pdf/2607.03621)</td><td>◆ This paper presents the development and evaluation of a collaborative system for real-time reconstruction of fragmented paper documents in the context of cultural heritage preservation.
◆ The developed system includes a collaborative robot, or cobot, that can fully manage the positioning of paper fragments using a specially designed vacuum-based suction attachment.
◆ This attachment enables gentle and precise positioning, ensuring the preservation of fragile materials.</td></tr>
<tr><td>2026-07-01</td><td>GKDT: General Keypoint Detection Transformer</td><td>[2607.00752](http://arxiv.org/pdf/2607.00752)</td><td>◆ With the emergence of various pre-trained vision and language models, computer vision is shifting from narrow-domain to open-domain recognition.
◆ The construction of a more powerful yet general keypoint detection (GKD) model to support diverse tasks has become increasingly important in the field.
◆ To this end, we firstly present a large-scale unified keypoint dataset called MegaKPT.</td></tr>
<tr><td>2026-07-01</td><td>AnyMatch: Supercharging Universal Multi-Modal Image Matching with Large-Scale Single-View Images</td><td>[2606.31077](http://arxiv.org/pdf/2606.31077)</td><td>◆ Multi-modal image matching is essential for visual localization and multi-sensor fusion, but it is hindered by the scarcity of large-scale training data with precise geometric annotations.
◆ Existing real-world datasets suffer from prohibitive costs, limited scene diversity, and errors in SfM-MVS pipelines, while synthetic methods struggle to maintain 3D geometric consistency or achieve photorealistic appearance.
◆ To address this, we propose AnyMatch, a novel framework that leverages abundant, easily accessible single-view images at minimal cost to generate rich multi-modal training data.</td></tr>
<tr><td>2026-06-29</td><td>MF-UAVPose6D: A Model-Free Monocular 6-DoF Pose Estimation Framework for Fixed-Wing UAVs</td><td>[2606.29697](http://arxiv.org/pdf/2606.29697)</td><td>◆ For uncrewed aerial vehicles (UAVs), estimating six-degree-of-freedom (6-DoF) poses is essential for airspace situational awareness, target tracking, and counter-UAV operations.
◆ However, non-cooperative targets usually lack computer-aided design (CAD) models and keypoint priors, making existing model-based or keypoint-matching methods difficult to apply reliably.
◆ To address these challenges, this paper proposes MF-UAVPose6D, a model-free monocular 6-DoF pose estimation framework for fixed-wing UAVs.</td></tr>
<tr><td>2026-06-26</td><td>KM-Speaker: Keypoint-Based Style Control for High-Quality Speech-Driven 3D Facial Animation and Dialogue Localization</td><td>[2606.28568](http://arxiv.org/pdf/2606.28568)</td><td>◆ Speech-driven 3D facial animation methods face significant challenges in simultaneously achieving high-fidelity motion and precise artistic control at production quality.
◆ Existing controllable models typically learn global style control by relying on large-scale, low-quality \emph{in-the-wild} datasets that compromise overall animation realism.
◆ Furthermore, these frameworks often lack the fine-grained temporal precision required for demanding tasks such as dialogue localization (e.g., dubbing), where matching specific facial expressions is as critical as lip synchronization.</td></tr>
<tr><td>2026-06-22</td><td>ISOPoT: Imaging Sonar Odometry by Point Tracking</td><td>[2606.23006](http://arxiv.org/pdf/2606.23006)</td><td>◆ Reliable navigation in underwater environments remains a key challenge in marine robotics.
◆ In such scenarios, forward-looking sonars are a natural choice for long-range perception, offering wide coverage even in turbid, low-visibility conditions.
◆ However, sonar images are inherently noisy, contain artifacts, and lack rich semantic structure, causing standard computer vision methods for keypoint detection and matching to perform poorly.</td></tr>
<tr><td>2026-06-22</td><td>G-MASt3R-SfM: Graph-based View Pruning and Multi-stage Optimization for Robust SfM</td><td>[2606.22856](http://arxiv.org/pdf/2606.22856)</td><td>◆ Structure from Motion (SfM) is essential for multi-view 3D reconstruction, however, its accuracy heavily relies on the accuracy of image matching.
◆ While the recent correspondence matching method, MASt3R, enables robust matching even under challenging conditions, it tends to generate incorrect correspondences for non-overlapping image pairs.
◆ Consequently, existing SfM methods using MASt3R, such as MASt3R-SfM, suffer from significant degradation in pose estimation accuracy as they incorporate these unreliable matches directly into optimization.</td></tr>
</tbody>
</table>
</div>

<h2 id='depth-estimation'>Depth Estimation</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2026-07-17</td><td>Dimension-invariant uniform consistency of the empirical spatial distribution function and its associated spatial depth estimator</td><td>[2607.16092](http://arxiv.org/pdf/2607.16092)</td><td>◆ We provide a proof that the empirical spatial distribution estimator in $\mathbb R^d$ as well as the corresponding plug-in estimator of the spatial depth are uniformly $L^1$-consistent.
◆ The consistency rate only depends on the sample size $n$, not on the dimension $d$ or any tuning or regularization parameters.
◆ This is a rare property.</td></tr>
<tr><td>2026-07-17</td><td>DPNeXt: A Lightweight Multi-Scale Feature Fusion Framework for Efficient ViT-Based Multi-Task Dense Prediction</td><td>[2607.16012](http://arxiv.org/pdf/2607.16012)</td><td>◆ Multi-Task Learning (MTL) in robotics perception systems supports comprehensive 3D spatial scene understanding by integrating semantic segmentation and depth estimation.
◆ While Vision Foundation Models (VFMs) are increasingly adopted as robust feature encoders, existing decoding strategies present a critical bottleneck.
◆ To address this, we propose DPNeXt, a streamlined multi-scale feature fusion decoder and efficient alternative to the standard Dense Prediction Transformer (DPT).</td></tr>
<tr><td>2026-07-17</td><td>Geometric Distillation from Rectified Stereo: Leveraging Epipolar Cues for Monocular Depth</td><td>[2607.15600](http://arxiv.org/pdf/2607.15600)</td><td>◆ Monocular depth foundation models have demonstrated remarkable generalization capabilities across diverse environments.
◆ However, they continue to struggle with metric depth estimation in diverse environments.
◆ This limitation stems from the inherent scale ambiguity of single-view inference, leading to misaligned scale predictions even when the relative geometry is accurate.</td></tr>
<tr><td>2026-07-15</td><td>WAVE-Stereo: Warp-Aligned Volume Encoding for Stereo Matching</td><td>[2607.13674](http://arxiv.org/pdf/2607.13674)</td><td>◆ Existing iterative stereo matching methods primarily adopt two types of correspondence representation: explicit matching search via correlation volumes and local residual refinement via warped features, yet the two remain separately modeled.
◆ We propose WAVE-Stereo, built on a core insight: correlation volumes and feature warping provide complementary matching cues.
◆ \textbf{GeoWarp Correspondence Encoder (GWCE)} encodes matching search, residual alignment, and disparity prior in parallel at the ConvGRU input.</td></tr>
<tr><td>2026-07-15</td><td>X-Lens: Real-Time Metric Depth Estimation with Heterogeneous Cameras</td><td>[2607.12993](http://arxiv.org/pdf/2607.12993)</td><td>◆ We present X-lens, a compact feed-forward model for metric depth estimation from a variable number of calibrated fisheye and pinhole views.
◆ To support real-time downstream perception, X-lens is built around a geometry-aware heterogeneous camera formulation with two key components.
◆ Learnable calibration tokens provide a coarse alignment between fisheye and pinhole projective spaces, while a Jacobian-parameterized distortion bias injected into cross-attention models local projection changes and promotes cross-camera consistency, enabling robust generalization with only 0.04B parameters and up to 41 FPS.</td></tr>
<tr><td>2026-07-14</td><td>Let RGB Be the Language of Vision</td><td>[2607.12450](http://arxiv.org/pdf/2607.12450)</td><td>◆ This work introduces a unified formulation for vision models, where diverse forms of visual information beyond natural images, such as masks, depth maps, and other structured visual signals, are all represented as RGB images, while general visual tasks can be converted into a common RGB-to-RGB image editing problem.
◆ In this paradigm, different types of visual information internally share the same encoding and decoding architecture and parameters as natural images, enabling a single model to transfer across tasks through a unified visual interface, in a way analogous to how language models operate over text.
◆ We refer to this formulation as RGB In and RGB Out (RINO).</td></tr>
<tr><td>2026-07-14</td><td>ARDepth: Auto-regressive Monocular Depth Estimation with Progressive Visual Conditioning</td><td>[2607.12433](http://arxiv.org/pdf/2607.12433)</td><td>◆ Diffusion models have recently become the dominant paradigm for monocular depth estimation (MDE).
◆ However, they implicitly assume that depth can be recovered as a globally smooth field through iterative denoising, which does not explicitly reflect the piecewise and scale-dependent organization of scene geometry.
◆ In practice, geometric structure emerges progressively across spatial scales, where coarse layout, surfaces, and boundaries are constructed in a hierarchical manner.</td></tr>
<tr><td>2026-07-14</td><td>DM-KG: A Novel Method for Boosting Spatial Cognition of Vision-Language Models in Street View Imagery</td><td>[2607.12319](http://arxiv.org/pdf/2607.12319)</td><td>◆ As vision-language models (VLMs) are increasingly deployed in geospatial question answering and visual scene understanding, improving their spatial cognition capability on street view imagery for complex logical reasoning has emerged as a key research priority.
◆ However, existing VLMs frequently suffer from &quot;spatial semantic hallucinations&quot; when perceiving object locations, distances, and directions in real-world street view scenes.
◆ Furthermore, such errors are often recalcitrant to tracing and calibration, posing a critical bottleneck for their practical deployment in geospatial tasks.</td></tr>
<tr><td>2026-07-13</td><td>When Depth Is Better Told Than Shown: Depth-Ordinal Prompting for Vision-Language Spatial Reasoning</td><td>[2607.11173](http://arxiv.org/pdf/2607.11173)</td><td>◆ Vision-language models (VLMs) are expected to reason about physical space -- which object is closer, what lies behind what, and how objects are arranged in 3D -- yet they still struggle with such spatial judgments.
◆ A natural remedy is to show the model a depth map, but we find that this can make performance worse.
◆ We show that depth is not absent: it reaches the language model, but becomes difficult to access for downstream reasoning, while rendered pseudo-depth maps act as noisy auxiliary images that frozen VLMs cannot easily regulate.</td></tr>
<tr><td>2026-07-13</td><td>GHOST: Geometry-Guided Hallucination of Opaque Surface Textures</td><td>[2607.11118](http://arxiv.org/pdf/2607.11118)</td><td>◆ Transparent objects pose a fundamental challenge for depth estimation and 3D reconstruction due to their violation of Lambertian assumptions, leading to severe geometry degradation in downstream tasks.
◆ To address this, we propose a novel geometry-guided preprocessing framework \textbf{GHOST} that leverages visual foundation models to transform transparent regions into opaque, structurally consistent representations without requiring downstream model retraining.
◆ Specifically, our pipeline utilizes (1) \textbf{TransDINO} and (2) \textbf{TransDecomp} to disentangle masks and transparency physical properties, while (3) \textbf{DAF-Net} recovers surface normal priors to encode geometric curvature.</td></tr>
</tbody>
</table>
</div>

<h2 id='3dgs'>3DGS</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2026-07-17</td><td>Rendering 3D Gaussians on a Graph Processor</td><td>[2607.15951](http://arxiv.org/pdf/2607.15951)</td><td>◆ We present the first implementation of a 3D Gaussian renderer on an Intelligence Processing Unit (IPU), comprising 1,472 independent tiles with only on-chip SRAM; constraints that approximate properties of efficient sensor-processor architectures.
◆ Our input scenes are 3D Gaussian maps from real-world sequences.
◆ Each tile &#x27;owns&#x27; a screen-space region of the framebuffer; Gaussian primitives are routed to destination tiles via Manhattan-distance hops on a north-east-west-south (NEWS) grid, then distributed to overlapping neighbours in an expanding tree pattern.</td></tr>
<tr><td>2026-07-17</td><td>HybridSim: A Physics-Learning Hybrid Digital Twin for mmWave Human Sensing</td><td>[2607.15806](http://arxiv.org/pdf/2607.15806)</td><td>◆ High-fidelity simulation of mmWave radar signals for dynamic human motion is valuable for developing radar-based human sensing models; yet collecting accurately labeled measurements for a specific deployment site remains expensive.
◆ We present HybridSim, a physics-learning hybrid simulator that synthesizes mmWave radar signals from dynamic human meshes under a fixed indoor room configuration, explicitly decoupling propagation into two components.
◆ To parameterize the human subject, we use a tri-plane representation to extract human features and a Graph Convolutional Network to stabilize optimization and mitigate gradient instability.</td></tr>
<tr><td>2026-07-17</td><td>ImprovedVBGS: Real-time Continual Variational Bayes Gaussian Splatting</td><td>[2607.15542](http://arxiv.org/pdf/2607.15542)</td><td>◆ On-the-fly reconstruction is a key requirement for many applications in robotics and autonomous navigation.
◆ Variational Bayes Gaussian Splatting (VBGS) enables continual learning without replay buffers using Coordinate Ascent Variational Inference (CAVI), but its per-frame iterations over all observed points make it too slow for real-time use with strict memory and latency requirements.
◆ We present ImprovedVBGS, an accelerated framework for on-the-fly continual reconstruction.</td></tr>
<tr><td>2026-07-17</td><td>E3DGS: Unified Geometric-Photometric Equivariance for 3D Gaussian Splatting via Color-as-Geometry Embedding</td><td>[2607.15536](http://arxiv.org/pdf/2607.15536)</td><td>◆ 3D Gaussian Splatting (3DGS) captures scenes by coupling explicit geometry (position, covariance) with view-dependent photometry (Spherical Harmonics).
◆ However, building $\mathrm{SE}(3)$-equivariant architectures on these primitives presents a fundamental representation bottleneck.
◆ Color has been treated as a signal rather than a geometric entity, making it nontrivial to unify symmetry across geometry and appearance as the camera frame changes.</td></tr>
<tr><td>2026-07-16</td><td>AeroAct: Action-Centered World-Action Models for Language-Conditioned Quadrotor Flight</td><td>[2607.14997](http://arxiv.org/pdf/2607.14997)</td><td>◆ Language-conditioned quadrotor flight requires a policy to ground semantic goals, anticipate the visual consequences of ego-motion, and output control references that remain smooth and dynamically executable under rapidly changing first-person views.
◆ Existing aerial vision-language navigation and vision-language-action methods commonly use discrete actions, high-level waypoints, or instantaneous velocity commands, which provide limited supervision about how flight actions change future observations.
◆ We present AeroAct, an action-centered world-action model (WAM) for quadrotor navigation.</td></tr>
<tr><td>2026-07-16</td><td>JADE-GS: Joint Alternating Deblurring Guided by Events in 3D Gaussian Splatting</td><td>[2607.14990](http://arxiv.org/pdf/2607.14990)</td><td>◆ When a camera moves fast during exposure, blur destroys the intra-exposure motion a 3D model needs to recover the sharp scene, while event cameras capture exactly this signal at microsecond resolution.
◆ Turning them into reliable 3D supervision faces two obstacles.
◆ First, the two restoration priors fail in opposite ways: physics-based event-integration priors preserve edges but accumulate drift; learned networks recover texture but distort boundaries.</td></tr>
<tr><td>2026-07-16</td><td>Compression of 3D Gaussian Splatting Data Using GPU-friendly Graphics Texture Coding</td><td>[2607.14513](http://arxiv.org/pdf/2607.14513)</td><td>◆ Techniques for modeling 3D scenes from image collections, such as 3D Gaussian Splatting (3DGS), are capable of generating high-quality novel views by leveraging graphics primitives with view-dependent appearance.
◆ In 3DGS, spherical harmonic (SH) are employed to model view-dependent color, resulting in a large number of SH coefficients per primitive and large memory requirements.
◆ While compression approaches have been proposed to mitigate this problem, they do not exploit the capabilities of modern Graphics Processing Units (GPUs) for parallel decoding and rendering.</td></tr>
<tr><td>2026-07-16</td><td>Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency</td><td>[2607.14481](http://arxiv.org/pdf/2607.14481)</td><td>◆ 3D Gaussian Splatting (3DGS) has become the method of choice for reconstructing and real-time rendering of captured scenes.
◆ To capture a scene with good visual quality, continuous image sequences are usually combined with out-of-order shots for better scene coverage.
◆ Structure from motion can reconstruct such captures, but only after they are all available and often with high computational cost.</td></tr>
<tr><td>2026-07-16</td><td>G$^2$SR: Geometric Methods for Fast and Memory-Efficient Gaussian-based Surface Reconstruction</td><td>[2607.14470](http://arxiv.org/pdf/2607.14470)</td><td>◆ Few-view surface reconstruction recovers the visible surfaces of a scene from a few posed RGB images, providing the 3D models that robots need to explore and interact online.
◆ On mobile platforms, the reconstruction must be fast and geometrically accurate while keeping a small memory footprint to ensure safe and efficient operation.
◆ 3D Gaussian Splatting (3DGS) offers a high-fidelity scene representation, but building it from a few views is ill-posed, as many distinct surfaces reproduce the same images, making traditional photometric methods prone to &quot;floater&quot; artifacts.</td></tr>
<tr><td>2026-07-15</td><td>Instant NuRec: Feed-Forward 3D Gaussian Reconstruction for Driving Scene Simulation</td><td>[2607.14203](http://arxiv.org/pdf/2607.14203)</td><td>◆ 3D simulation platforms are critical for autonomous driving because they enable end-to-end policy evaluation, thereby reducing development costs and improving safety.
◆ In recent years, neural simulation has become predominant, with methods such as NuRec playing a central role; however, these methods remain relatively slow and typically require per-scene tuning.
◆ In this work, we present Instant NuRec, a feed-forward neural reconstruction model that turns a short multi-view driving log into a fully simulatable 3D Gaussian Splatting (3DGS) world in a single forward pass.</td></tr>
</tbody>
</table>
</div>

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2026.07.20
