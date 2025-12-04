# SLAM领域最新论文 (2025.12.04)

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
<tr><td>2025-12-03</td><td>What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models</td><td>[2512.03422](http://arxiv.org/pdf/2512.03422)</td><td>◆ In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models.
◆ While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance.
◆ Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence.</td></tr>
<tr><td>2025-12-03</td><td>Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing</td><td>[2512.03397](http://arxiv.org/pdf/2512.03397)</td><td>◆ LiDAR-inertial odometry (LIO) is an active research area, as it enables accurate real-time state estimation in GPS-denied environments.
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

<h2 id='visual-slam'>Visual SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-12-02</td><td>VIGS-SLAM: Visual Inertial Gaussian Splatting SLAM</td><td>[2512.02293](http://arxiv.org/pdf/2512.02293)</td><td>◆ We present VIGS-SLAM, a visual-inertial 3D Gaussian Splatting SLAM system that achieves robust real-time tracking and high-fidelity reconstruction.
◆ Although recent 3DGS-based SLAM methods achieve dense and photorealistic mapping, their purely visual design degrades under motion blur, low texture, and exposure variations.
◆ Our method tightly couples visual and inertial cues within a unified optimization framework, jointly refining camera poses, depths, and IMU states.</td></tr>
<tr><td>2025-12-01</td><td>KM-ViPE: Online Tightly Coupled Vision-Language-Geometry Fusion for Open-Vocabulary Semantic SLAM</td><td>[2512.01889](http://arxiv.org/pdf/2512.01889)</td><td>◆ We present KM-ViPE (Knowledge Mapping Video Pose Engine), a real-time open-vocabulary SLAM framework for uncalibrated monocular cameras in dynamic environments.
◆ Unlike systems requiring depth sensors and offline calibration, KM-ViPE operates directly on raw RGB streams, making it ideal for ego-centric applications and harvesting internet-scale video data for training.
◆ KM-ViPE tightly couples DINO visual features with geometric constraints through a high-level features based adaptive robust kernel that handles both moving objects and movable static objects (e.g., moving furniture in ego-centric views).</td></tr>
</tbody>
</table>
</div>

<h2 id='loop-closure'>Loop Closure</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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
<tr><td>2025-12-01</td><td>The Dependence of Earth Milankovitch Cycles on Martian Mass</td><td>[2512.02108](http://arxiv.org/pdf/2512.02108)</td><td>◆ The Milankovitch cycles of Earth result from gravitational interactions with other bodies in the Solar System.
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

<h2 id='image-matching'>Image Matching</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

<h2 id='3dgs'>3DGS</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

<h2 id='depth-estimation'>Depth Estimation</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2025.12.04
