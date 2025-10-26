# SLAM领域最新论文 (2025.10.26)

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
<tr><td>2025-10-23</td><td>Deep Learning-Powered Visual SLAM Aimed at Assisting Visually Impaired Navigation</td><td>[2510.20549](http://arxiv.org/pdf/2510.20549)</td><td>◆ Despite advancements in SLAM technologies, robust operation under challenging conditions such as low-texture, motion-blur, or challenging lighting remains an open challenge.
◆ Such conditions are common in applications such as assistive navigation for the visually impaired.
◆ These challenges undermine localization accuracy and tracking stability, reducing navigation reliability and safety.</td></tr>
<tr><td>2025-10-21</td><td>Underwater Dense Mapping with the First Compact 3D Sonar</td><td>[2510.18991](http://arxiv.org/pdf/2510.18991)</td><td>◆ In the past decade, the adoption of compact 3D range sensors, such as LiDARs, has driven the developments of robust state-estimation pipelines, making them a standard sensor for aerial, ground, and space autonomy.
◆ Unfortunately, poor propagation of electromagnetic waves underwater, has limited the visibility-independent sensing options of underwater state-estimation to acoustic range sensors, which provide 2D information including, at-best, spatially ambiguous information.
◆ This paper, to the best of our knowledge, is the first study examining the performance, capacity, and opportunities arising from the recent introduction of the first compact 3D sonar.</td></tr>
<tr><td>2025-10-21</td><td>DeepDetect: Learning All-in-One Dense Keypoints</td><td>[2510.17422](http://arxiv.org/pdf/2510.17422)</td><td>◆ Keypoint detection is the foundation of many computer vision tasks, including image registration, structure-from motion, 3D reconstruction, visual odometry, and SLAM.
◆ Traditional detectors (SIFT, SURF, ORB, BRISK, etc.) and learning based methods (SuperPoint, R2D2, LF-Net, D2-Net, etc.) have shown strong performance yet suffer from key limitations: sensitivity to photometric changes, low keypoint density and repeatability, limited adaptability to challenging scenes, and lack of semantic understanding, often failing to prioritize visually important regions.
◆ We present DeepDetect, an intelligent, all-in-one, dense keypoint detector that unifies the strengths of classical detectors using deep learning.</td></tr>
<tr><td>2025-10-18</td><td>LightGlueStick: a Fast and Robust Glue for Joint Point-Line Matching</td><td>[2510.16438](http://arxiv.org/pdf/2510.16438)</td><td>◆ Lines and points are complementary local features, whose combination has proven effective for applications such as SLAM and Structure-from-Motion.
◆ The backbone of these pipelines are the local feature matchers, establishing correspondences across images.
◆ Traditionally, point and line matching have been treated as independent tasks.</td></tr>
<tr><td>2025-10-17</td><td>VAR-SLAM: Visual Adaptive and Robust SLAM for Dynamic Environments</td><td>[2510.16205](http://arxiv.org/pdf/2510.16205)</td><td>◆ Visual SLAM in dynamic environments remains challenging, as several existing methods rely on semantic filtering that only handles known object classes, or use fixed robust kernels that cannot adapt to unknown moving objects, leading to degraded accuracy when they appear in the scene.
◆ We present VAR-SLAM (Visual Adaptive and Robust SLAM), an ORB-SLAM3-based system that combines a lightweight semantic keypoint filter to deal with known moving objects, with Barron&#x27;s adaptive robust loss to handle unknown ones.
◆ The shape parameter of the robust kernel is estimated online from residuals, allowing the system to automatically adjust between Gaussian and heavy-tailed behavior.</td></tr>
<tr><td>2025-10-17</td><td>Dynamic Recalibration in LiDAR SLAM: Integrating AI and Geometric Methods with Real-Time Feedback Using INAF Fusion</td><td>[2510.15803](http://arxiv.org/pdf/2510.15803)</td><td>◆ This paper presents a novel fusion technique for LiDAR Simultaneous Localization and Mapping (SLAM), aimed at improving localization and 3D mapping using LiDAR sensor.
◆ Our approach centers on the Inferred Attention Fusion (INAF) module, which integrates AI with geometric odometry.
◆ Utilizing the KITTI dataset&#x27;s LiDAR data, INAF dynamically adjusts attention weights based on environmental feedback, enhancing the system&#x27;s adaptability and measurement accuracy.</td></tr>
<tr><td>2025-10-17</td><td>LVI-Q: Robust LiDAR-Visual-Inertial-Kinematic Odometry for Quadruped Robots Using Tightly-Coupled and Efficient Alternating Optimization</td><td>[2510.15220](http://arxiv.org/pdf/2510.15220)</td><td>◆ Autonomous navigation for legged robots in complex and dynamic environments relies on robust simultaneous localization and mapping (SLAM) systems to accurately map surroundings and localize the robot, ensuring safe and efficient operation.
◆ While prior sensor fusion-based SLAM approaches have integrated various sensor modalities to improve their robustness, these algorithms are still susceptible to estimation drift in challenging environments due to their reliance on unsuitable fusion strategies.
◆ Therefore, we propose a robust LiDAR-visual-inertial-kinematic odometry system that integrates information from multiple sensors, such as a camera, LiDAR, inertial measurement unit (IMU), and joint encoders, for visual and LiDAR-based odometry estimation.</td></tr>
<tr><td>2025-10-16</td><td>3D Scene Prompting for Scene-Consistent Camera-Controllable Video Generation</td><td>[2510.14945](http://arxiv.org/pdf/2510.14945)</td><td>◆ We present 3DScenePrompt, a framework that generates the next video chunk from arbitrary-length input while enabling precise camera control and preserving scene consistency.
◆ Unlike methods conditioned on a single image or a short clip, we employ dual spatio-temporal conditioning that reformulates context-view referencing across the input video.
◆ Our approach conditions on both temporally adjacent frames for motion continuity and spatially adjacent content for scene consistency.</td></tr>
<tr><td>2025-10-15</td><td>Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU</td><td>[2510.13546](http://arxiv.org/pdf/2510.13546)</td><td>◆ Feature detection is a common yet time-consuming module in Simultaneous Localization and Mapping (SLAM) implementations, which are increasingly deployed on power-constrained platforms, such as drones.
◆ Graphics Processing Units (GPUs) have been a popular accelerator for computer vision in general, and feature detection and SLAM in particular.
◆ On the other hand, System-on-Chips (SoCs) with integrated Field Programmable Gate Array (FPGA) are also widely available.</td></tr>
<tr><td>2025-10-15</td><td>Through the Lens of Doubt: Robust and Efficient Uncertainty Estimation for Visual Place Recognition</td><td>[2510.13464](http://arxiv.org/pdf/2510.13464)</td><td>◆ Visual Place Recognition (VPR) enables robots and autonomous vehicles to identify previously visited locations by matching current observations against a database of known places.
◆ However, VPR systems face significant challenges when deployed across varying visual environments, lighting conditions, seasonal changes, and viewpoints changes.
◆ Failure-critical VPR applications, such as loop closure detection in simultaneous localization and mapping (SLAM) pipelines, require robust estimation of place matching uncertainty.</td></tr>
<tr><td>2025-10-15</td><td>DAMM-LOAM: Degeneracy Aware Multi-Metric LiDAR Odometry and Mapping</td><td>[2510.13287](http://arxiv.org/pdf/2510.13287)</td><td>◆ LiDAR Simultaneous Localization and Mapping (SLAM) systems are essential for enabling precise navigation and environmental reconstruction across various applications.
◆ Although current point-to-plane ICP algorithms perform effec- tively in structured, feature-rich environments, they struggle in scenarios with sparse features, repetitive geometric structures, and high-frequency motion.
◆ This leads to degeneracy in 6- DOF pose estimation.</td></tr>
<tr><td>2025-10-11</td><td>FORM: Fixed-Lag Odometry with Reparative Mapping utilizing Rotating LiDAR Sensors</td><td>[2510.09966](http://arxiv.org/pdf/2510.09966)</td><td>◆ Light Detection and Ranging (LiDAR) sensors have become a de-facto sensor for many robot state estimation tasks, spurring development of many LiDAR Odometry (LO) methods in recent years.
◆ While some smoothing-based LO methods have been proposed, most require matching against multiple scans, resulting in sub-real-time performance.
◆ Due to this, most prior works estimate a single state at a time and are ``submap&#x27;&#x27;-based.</td></tr>
<tr><td>2025-10-09</td><td>ARTDECO: Towards Efficient and High-Fidelity On-the-Fly 3D Reconstruction with Structured Scene Representation</td><td>[2510.08551](http://arxiv.org/pdf/2510.08551)</td><td>◆ On-the-fly 3D reconstruction from monocular image sequences is a long-standing challenge in computer vision, critical for applications such as real-to-sim, AR/VR, and robotics.
◆ Existing methods face a major tradeoff: per-scene optimization yields high fidelity but is computationally expensive, whereas feed-forward foundation models enable real-time inference but struggle with accuracy and robustness.
◆ In this work, we propose ARTDECO, a unified framework that combines the efficiency of feed-forward models with the reliability of SLAM-based pipelines.</td></tr>
<tr><td>2025-10-09</td><td>RTGS: Real-Time 3D Gaussian Splatting SLAM via Multi-Level Redundancy Reduction</td><td>[2510.06644](http://arxiv.org/pdf/2510.06644)</td><td>◆ 3D Gaussian Splatting (3DGS) based Simultaneous Localization and Mapping (SLAM) systems can largely benefit from 3DGS&#x27;s state-of-the-art rendering efficiency and accuracy, but have not yet been adopted in resource-constrained edge devices due to insufficient speed.
◆ Addressing this, we identify notable redundancies across the SLAM pipeline for acceleration.
◆ While conceptually straightforward, practical approaches are required to minimize the overhead associated with identifying and eliminating these redundancies.</td></tr>
<tr><td>2025-10-07</td><td>Human3R: Everyone Everywhere All at Once</td><td>[2510.06219](http://arxiv.org/pdf/2510.06219)</td><td>◆ We present Human3R, a unified, feed-forward framework for online 4D human-scene reconstruction, in the world frame, from casually captured monocular videos.
◆ Unlike previous approaches that rely on multi-stage pipelines, iterative contact-aware refinement between humans and scenes, and heavy dependencies, e.g., human detection, depth estimation, and SLAM pre-processing, Human3R jointly recovers global multi-person SMPL-X bodies (&quot;everyone&quot;), dense 3D scene (&quot;everywhere&quot;), and camera trajectories in a single forward pass (&quot;all-at-once&quot;).
◆ Our method builds upon the 4D online reconstruction model CUT3R, and uses parameter-efficient visual prompt tuning, to strive to preserve CUT3R&#x27;s rich spatiotemporal priors, while enabling direct readout of multiple SMPL-X bodies.</td></tr>
<tr><td>2025-10-07</td><td>Dropping the D: RGB-D SLAM Without the Depth Sensor</td><td>[2510.06216](http://arxiv.org/pdf/2510.06216)</td><td>◆ We present DropD-SLAM, a real-time monocular SLAM system that achieves RGB-D-level accuracy without relying on depth sensors.
◆ The system replaces active depth input with three pretrained vision modules: a monocular metric depth estimator, a learned keypoint detector, and an instance segmentation network.
◆ Dynamic objects are suppressed using dilated instance masks, while static keypoints are assigned predicted depth values and backprojected into 3D to form metrically scaled features.</td></tr>
<tr><td>2025-10-07</td><td>Coordinate-Consistent Localization via Continuous-Time Calibration and Fusion of UWB and SLAM Observations</td><td>[2510.05992](http://arxiv.org/pdf/2510.05992)</td><td>◆ Onboard simultaneous localization and mapping (SLAM) methods are commonly used to provide accurate localization information for autonomous robots.
◆ However, the coordinate origin of SLAM estimate often resets for each run.
◆ On the other hand, UWB-based localization with fixed anchors can ensure a consistent coordinate reference across sessions; however, it requires an accurate assignment of the anchor nodes&#x27; coordinates.</td></tr>
<tr><td>2025-10-06</td><td>OKVIS2-X: Open Keyframe-based Visual-Inertial SLAM Configurable with Dense Depth or LiDAR, and GNSS</td><td>[2510.04612](http://arxiv.org/pdf/2510.04612)</td><td>◆ To empower mobile robots with usable maps as well as highest state estimation accuracy and robustness, we present OKVIS2-X: a state-of-the-art multi-sensor Simultaneous Localization and Mapping (SLAM) system building dense volumetric occupancy maps, while scalable to large environments and operating in realtime.
◆ Our unified SLAM framework seamlessly integrates different sensor modalities: visual, inertial, measured or learned depth, LiDAR and Global Navigation Satellite System (GNSS) measurements.
◆ Unlike most state-of-the-art SLAM systems, we advocate using dense volumetric map representations when leveraging depth or range-sensing capabilities.</td></tr>
<tr><td>2025-10-02</td><td>RSV-SLAM: Toward Real-Time Semantic Visual SLAM in Indoor Dynamic Environments</td><td>[2510.02616](http://arxiv.org/pdf/2510.02616)</td><td>◆ Simultaneous Localization and Mapping (SLAM) plays an important role in many robotics fields, including social robots.
◆ Many of the available visual SLAM methods are based on the assumption of a static world and struggle in dynamic environments.
◆ In the current study, we introduce a real-time semantic RGBD SLAM approach designed specifically for dynamic environments.</td></tr>
<tr><td>2025-10-02</td><td>EC3R-SLAM: Efficient and Consistent Monocular Dense SLAM with Feed-Forward 3D Reconstruction</td><td>[2510.02080](http://arxiv.org/pdf/2510.02080)</td><td>◆ The application of monocular dense Simultaneous Localization and Mapping (SLAM) is often hindered by high latency, large GPU memory consumption, and reliance on camera calibration.
◆ To relax this constraint, we propose EC3R-SLAM, a novel calibration-free monocular dense SLAM framework that jointly achieves high localization and mapping accuracy, low latency, and low GPU memory consumption.
◆ This enables the framework to achieve efficiency through the coupling of a tracking module, which maintains a sparse map of feature points, and a mapping module based on a feed-forward 3D reconstruction model that simultaneously estimates camera intrinsics.</td></tr>
<tr><td>2025-10-02</td><td>Non-Rigid Structure-from-Motion via Differential Geometry with Recoverable Conformal Scale</td><td>[2510.01665](http://arxiv.org/pdf/2510.01665)</td><td>◆ Non-rigid structure-from-motion (NRSfM), a promising technique for addressing the mapping challenges in monocular visual deformable simultaneous localization and mapping (SLAM), has attracted growing attention.
◆ We introduce a novel method, called Con-NRSfM, for NRSfM under conformal deformations, encompassing isometric deformations as a subset.
◆ Our approach performs point-wise reconstruction using 2D selected image warps optimized through a graph-based framework.</td></tr>
<tr><td>2025-10-01</td><td>Instant4D: 4D Gaussian Splatting in Minutes</td><td>[2510.01119](http://arxiv.org/pdf/2510.01119)</td><td>◆ Dynamic view synthesis has seen significant advances, yet reconstructing scenes from uncalibrated, casual video remains challenging due to slow optimization and complex parameter estimation.
◆ In this work, we present Instant4D, a monocular reconstruction system that leverages native 4D representation to efficiently process casual video sequences within minutes, without calibrated cameras or depth sensors.
◆ Our method begins with geometric recovery through deep visual SLAM, followed by grid pruning to optimize scene representation.</td></tr>
<tr><td>2025-10-01</td><td>Semantic Visual Simultaneous Localization and Mapping: A Survey on State of the Art, Challenges, and Future Directions</td><td>[2510.00783](http://arxiv.org/pdf/2510.00783)</td><td>◆ Semantic Simultaneous Localization and Mapping (SLAM) is a critical area of research within robotics and computer vision, focusing on the simultaneous localization of robotic systems and associating semantic information to construct the most accurate and complete comprehensive model of the surrounding environment.
◆ Since the first foundational work in Semantic SLAM appeared more than two decades ago, this field has received increasing attention across various scientific communities.
◆ Despite its significance, the field lacks comprehensive surveys encompassing recent advances and persistent challenges.</td></tr>
<tr><td>2025-09-30</td><td>Benchmarking Egocentric Visual-Inertial SLAM at City Scale</td><td>[2509.26639](http://arxiv.org/pdf/2509.26639)</td><td>◆ Precise 6-DoF simultaneous localization and mapping (SLAM) from onboard sensors is critical for wearable devices capturing egocentric data, which exhibits specific challenges, such as a wider diversity of motions and viewpoints, prevalent dynamic visual content, or long sessions affected by time-varying sensor calibration.
◆ While recent progress on SLAM has been swift, academic research is still driven by benchmarks that do not reflect these challenges or do not offer sufficiently accurate ground truth poses.
◆ In this paper, we introduce a new dataset and benchmark for visual-inertial SLAM with egocentric, multi-modal data.</td></tr>
<tr><td>2025-09-30</td><td>Graphite: A GPU-Accelerated Mixed-Precision Graph Optimization Framework</td><td>[2509.26581](http://arxiv.org/pdf/2509.26581)</td><td>◆ We present Graphite, a GPU-accelerated nonlinear graph optimization framework.
◆ It provides a CUDA C++ interface to enable the sharing of code between a realtime application, such as a SLAM system, and its optimization tasks.
◆ The framework supports techniques to reduce memory usage, including in-place optimization, support for multiple floating point types and mixed-precision modes, and dynamically computed Jacobians.</td></tr>
<tr><td>2025-09-30</td><td>Radio-based Multi-Robot Odometry and Relative Localization</td><td>[2509.26558](http://arxiv.org/pdf/2509.26558)</td><td>◆ Radio-based methods such as Ultra-Wideband (UWB) and RAdio Detection And Ranging (radar), which have traditionally seen limited adoption in robotics, are experiencing a boost in popularity thanks to their robustness to harsh environmental conditions and cluttered environments.
◆ This work proposes a multi-robot UGV-UAV localization system that leverages the two technologies with inexpensive and readily-available sensors, such as Inertial Measurement Units (IMUs) and wheel encoders, to estimate the relative position of an aerial robot with respect to a ground robot.
◆ The first stage of the system pipeline includes a nonlinear optimization framework to trilaterate the location of the aerial platform based on UWB range data, and a radar pre-processing module with loosely coupled ego-motion estimation which has been adapted for a multi-robot scenario.</td></tr>
<tr><td>2025-09-30</td><td>DEPTHOR++: Robust Depth Enhancement from a Real-World Lightweight dToF and RGB Guidance</td><td>[2509.26498](http://arxiv.org/pdf/2509.26498)</td><td>◆ Depth enhancement, which converts raw dToF signals into dense depth maps using RGB guidance, is crucial for improving depth perception in high-precision tasks such as 3D reconstruction and SLAM.
◆ However, existing methods often assume ideal dToF inputs and perfect dToF-RGB alignment, overlooking calibration errors and anomalies, thus limiting real-world applicability.
◆ This work systematically analyzes the noise characteristics of real-world lightweight dToF sensors and proposes a practical and novel depth completion framework, DEPTHOR++, which enhances robustness to noisy dToF inputs from three key aspects.</td></tr>
<tr><td>2025-09-30</td><td>Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring</td><td>[2509.26121](http://arxiv.org/pdf/2509.26121)</td><td>◆ The transition of seaweed farming to an alternative food source on an industrial scale relies on automating its processes through smart farming, equivalent to land agriculture.
◆ Key to this process are autonomous underwater vehicles (AUVs) via their capacity to automate crop and structural inspections.
◆ However, the current bottleneck for their deployment is ensuring safe navigation within farms, which requires an accurate, online estimate of the AUV pose and map of the infrastructure.</td></tr>
<tr><td>2025-09-30</td><td>User-Centric Communication Service Provision for Edge-Assisted Mobile Augmented Reality</td><td>[2509.25905](http://arxiv.org/pdf/2509.25905)</td><td>◆ Future 6G networks are envisioned to facilitate edge-assisted mobile augmented reality (MAR) via strengthening the collaboration between MAR devices and edge servers.
◆ In order to provide immersive user experiences, MAR devices must timely upload camera frames to an edge server for simultaneous localization and mapping (SLAM)-based device pose tracking.
◆ In this paper, to cope with user-specific and non-stationary uplink data traffic, we develop a digital twin (DT)-based approach for user-centric communication service provision for MAR.</td></tr>
<tr><td>2025-09-29</td><td>PROFusion: Robust and Accurate Dense Reconstruction via Camera Pose Regression and Optimization</td><td>[2509.24236](http://arxiv.org/pdf/2509.24236)</td><td>◆ Real-time dense scene reconstruction during unstable camera motions is crucial for robotics, yet current RGB-D SLAM systems fail when cameras experience large viewpoint changes, fast motions, or sudden shaking.
◆ Classical optimization-based methods deliver high accuracy but fail with poor initialization during large motions, while learning-based approaches provide robustness but lack sufficient accuracy for dense reconstruction.
◆ We address this challenge through a combination of learning-based initialization with optimization-based refinement.</td></tr>
<tr><td>2025-09-28</td><td>GRS-SLAM3R: Real-Time Dense SLAM with Gated Recurrent State</td><td>[2509.23737](http://arxiv.org/pdf/2509.23737)</td><td>◆ DUSt3R-based end-to-end scene reconstruction has recently shown promising results in dense visual SLAM.
◆ However, most existing methods only use image pairs to estimate pointmaps, overlooking spatial memory and global consistency.To this end, we introduce GRS-SLAM3R, an end-to-end SLAM framework for dense scene reconstruction and pose estimation from RGB images without any prior knowledge of the scene or camera parameters.
◆ Unlike existing DUSt3R-based frameworks, which operate on all image pairs and predict per-pair point maps in local coordinate frames, our method supports sequentialized input and incrementally estimates metric-scale point clouds in the global coordinate.</td></tr>
<tr><td>2025-09-28</td><td>From Fields to Splats: A Cross-Domain Survey of Real-Time Neural Scene Representations</td><td>[2509.23555](http://arxiv.org/pdf/2509.23555)</td><td>◆ Neural scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have transformed how 3D environments are modeled, rendered, and interpreted.
◆ NeRF introduced view-consistent photorealism via volumetric rendering; 3DGS has rapidly emerged as an explicit, efficient alternative that supports high-quality rendering, faster optimization, and integration into hybrid pipelines for enhanced photorealism and task-driven scene understanding.
◆ This survey examines how 3DGS is being adopted across SLAM, telepresence and teleoperation, robotic manipulation, and 3D content generation.</td></tr>
<tr><td>2025-09-27</td><td>EKF-Based Fusion of Wi-Fi/LiDAR/IMU for Indoor Localization and Navigation</td><td>[2509.23118](http://arxiv.org/pdf/2509.23118)</td><td>◆ Conventional Wi-Fi received signal strength indicator (RSSI) fingerprinting cannot meet the growing demand for accurate indoor localization and navigation due to its lower accuracy, while solutions based on light detection and ranging (LiDAR) can provide better localization performance but is limited by their higher deployment cost and complexity.
◆ To address these issues, we propose a novel indoor localization and navigation framework integrating Wi-Fi RSSI fingerprinting, LiDAR-based simultaneous localization and mapping (SLAM), and inertial measurement unit (IMU) navigation based on an extended Kalman filter (EKF).
◆ Specifically, coarse localization by deep neural network (DNN)-based Wi-Fi RSSI fingerprinting is refined by IMU-based dynamic positioning using a Gmapping-based SLAM to generate an occupancy grid map and output high-frequency attitude estimates, which is followed by EKF prediction-update integrating sensor information while effectively suppressing Wi-Fi-induced noise and IMU drift errors.</td></tr>
<tr><td>2025-09-26</td><td>Good Weights: Proactive, Adaptive Dead Reckoning Fusion for Continuous and Robust Visual SLAM</td><td>[2509.22910](http://arxiv.org/pdf/2509.22910)</td><td>◆ Given that Visual SLAM relies on appearance cues for localization and scene understanding, texture-less or visually degraded environments (e.g., plain walls or low lighting) lead to poor pose estimation and track loss.
◆ However, robots are typically equipped with sensors that provide some form of dead reckoning odometry with reasonable short-time performance but unreliable long-time performance.
◆ The Good Weights (GW) algorithm described here provides a framework to adaptively integrate dead reckoning (DR) with passive visual SLAM for continuous and accurate frame-level pose estimation.</td></tr>
<tr><td>2025-09-26</td><td>IMU-Preintegrated Radar Factors for Asynchronous Radar-LiDAR-Inertial SLAM</td><td>[2509.22288](http://arxiv.org/pdf/2509.22288)</td><td>◆ Fixed-lag Radar-LiDAR-Inertial smoothers conventionally create one factor graph node per measurement to compensate for the lack of time synchronization between radar and LiDAR.
◆ For a radar-LiDAR sensor pair with equal rates, this strategy results in a state creation rate of twice the individual sensor frequencies.
◆ This doubling of the number of states per second yields high optimization costs, inhibiting real-time performance on resource-constrained hardware.</td></tr>
<tr><td>2025-09-26</td><td>An Adaptive ICP LiDAR Odometry Based on Reliable Initial Pose</td><td>[2509.22058](http://arxiv.org/pdf/2509.22058)</td><td>◆ As a key technology for autonomous navigation and positioning in mobile robots, light detection and ranging (LiDAR) odometry is widely used in autonomous driving applications.
◆ The Iterative Closest Point (ICP)-based methods have become the core technique in LiDAR odometry due to their efficient and accurate point cloud registration capability.
◆ However, some existing ICP-based methods do not consider the reliability of the initial pose, which may cause the method to converge to a local optimum.</td></tr>
<tr><td>2025-09-25</td><td>Real-Time Indoor Object SLAM with LLM-Enhanced Priors</td><td>[2509.21602](http://arxiv.org/pdf/2509.21602)</td><td>◆ Object-level Simultaneous Localization and Mapping (SLAM), which incorporates semantic information for high-level scene understanding, faces challenges of under-constrained optimization due to sparse observations.
◆ Prior work has introduced additional constraints using commonsense knowledge, but obtaining such priors has traditionally been labor-intensive and lacks generalizability across diverse object categories.
◆ We address this limitation by leveraging large language models (LLMs) to provide commonsense knowledge of object geometric attributes, specifically size and orientation, as prior factors in a graph-based SLAM framework.</td></tr>
<tr><td>2025-09-25</td><td>AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation</td><td>[2509.21006](http://arxiv.org/pdf/2509.21006)</td><td>◆ We address natural language pick-and-place in unseen, unpredictable indoor environments with AnywhereVLA, a modular framework for mobile manipulation.
◆ A user text prompt serves as an entry point and is parsed into a structured task graph that conditions classical SLAM with LiDAR and cameras, metric semantic mapping, and a task-aware frontier exploration policy.
◆ An approach planner then selects visibility and reachability aware pre grasp base poses.</td></tr>
<tr><td>2025-09-29</td><td>MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM</td><td>[2509.20757](http://arxiv.org/pdf/2509.20757)</td><td>◆ Visual SLAM is a cornerstone technique in robotics, autonomous driving and extended reality (XR), yet classical systems often struggle with low-texture environments, scale ambiguity, and degraded performance under challenging visual conditions.
◆ Recent advancements in feed-forward neural network-based pointmap regression have demonstrated the potential to recover high-fidelity 3D scene geometry directly from images, leveraging learned spatial priors to overcome limitations of traditional multi-view geometry methods.
◆ However, the widely validated advantages of probabilistic multi-sensor information fusion are often discarded in these pipelines.</td></tr>
<tr><td>2025-09-25</td><td>SLAM-Free Visual Navigation with Hierarchical Vision-Language Perception and Coarse-to-Fine Semantic Topological Planning</td><td>[2509.20739](http://arxiv.org/pdf/2509.20739)</td><td>◆ Conventional SLAM pipelines for legged robot navigation are fragile under rapid motion, calibration demands, and sensor drift, while offering limited semantic reasoning for task-driven exploration.
◆ To deal with these issues, we propose a vision-only, SLAM-free navigation framework that replaces dense geometry with semantic reasoning and lightweight topological representations.
◆ A hierarchical vision-language perception module fuses scene-level context with object-level cues for robust semantic inference.</td></tr>
<tr><td>2025-09-24</td><td>Optical Ocean Recipes: Creating Realistic Datasets to Facilitate Underwater Vision Research</td><td>[2509.20171](http://arxiv.org/pdf/2509.20171)</td><td>◆ The development and evaluation of machine vision in underwater environments remains challenging, often relying on trial-and-error-based testing tailored to specific applications.
◆ This is partly due to the lack of controlled, ground-truthed testing environments that account for the optical challenges, such as color distortion from spectrally variant light attenuation, reduced contrast and blur from backscatter and volume scattering, and dynamic light patterns from natural or artificial illumination.
◆ Additionally, the appearance of ocean water in images varies significantly across regions, depths, and seasons.</td></tr>
<tr><td>2025-09-23</td><td>Bioinspired SLAM Approach for Unmanned Surface Vehicle</td><td>[2509.19522](http://arxiv.org/pdf/2509.19522)</td><td>◆ This paper presents OpenRatSLAM2, a new version of OpenRatSLAM - a bioinspired SLAM framework based on computational models of the rodent hippocampus.
◆ OpenRatSLAM2 delivers low-computation-cost visual-inertial based SLAM, suitable for GPS-denied environments.
◆ Our contributions include a ROS2-based architecture, experimental results on new waterway datasets, and insights into system parameter tuning.</td></tr>
<tr><td>2025-09-23</td><td>CU-Multi: A Dataset for Multi-Robot Collaborative Perception</td><td>[2509.19463](http://arxiv.org/pdf/2509.19463)</td><td>◆ A central challenge for multi-robot systems is fusing independently gathered perception data into a unified representation.
◆ Despite progress in Collaborative SLAM (C-SLAM), benchmarking remains hindered by the scarcity of dedicated multi-robot datasets.
◆ Many evaluations instead partition single-robot trajectories, a practice that may only partially reflect true multi-robot operations and, more critically, lacks standardization, leading to results that are difficult to interpret or compare across studies.</td></tr>
<tr><td>2025-09-23</td><td>Towards Robust LiDAR Localization: Deep Learning-based Uncertainty Estimation</td><td>[2509.18954](http://arxiv.org/pdf/2509.18954)</td><td>◆ LiDAR-based localization and SLAM often rely on iterative matching algorithms, particularly the Iterative Closest Point (ICP) algorithm, to align sensor data with pre-existing maps or previous scans.
◆ However, ICP is prone to errors in featureless environments and dynamic scenes, leading to inaccurate pose estimation.
◆ Accurately predicting the uncertainty associated with ICP is crucial for robust state estimation but remains challenging, as existing approaches often rely on handcrafted models or simplified assumptions.</td></tr>
<tr><td>2025-09-22</td><td>Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation</td><td>[2509.18342](http://arxiv.org/pdf/2509.18342)</td><td>◆ Accurate localisation is critical for mobile robots in structured outdoor environments, yet LiDAR-based methods often fail in vineyards due to repetitive row geometry and perceptual aliasing.
◆ We propose a semantic particle filter that incorporates stable object-level detections, specifically vine trunks and support poles into the likelihood estimation process.
◆ Detected landmarks are projected into a birds eye view and fused with LiDAR scans to generate semantic observations.</td></tr>
<tr><td>2025-09-22</td><td>ProDyG: Progressive Dynamic Scene Reconstruction via Gaussian Splatting from Monocular Videos</td><td>[2509.17864](http://arxiv.org/pdf/2509.17864)</td><td>◆ Achieving truly practical dynamic 3D reconstruction requires online operation, global pose and map consistency, detailed appearance modeling, and the flexibility to handle both RGB and RGB-D inputs.
◆ However, existing SLAM methods typically merely remove the dynamic parts or require RGB-D input, while offline methods are not scalable to long video sequences, and current transformer-based feedforward methods lack global consistency and appearance details.
◆ To this end, we achieve online dynamic scene reconstruction by disentangling the static and dynamic parts within a SLAM system.</td></tr>
<tr><td>2025-09-21</td><td>SLAM-Former: Putting SLAM into One Transformer</td><td>[2509.16909](http://arxiv.org/pdf/2509.16909)</td><td>◆ We present SLAM-Former, a novel neural approach that integrates full SLAM capabilities into a single transformer.
◆ Similar to traditional SLAM systems, SLAM-Former comprises both a frontend and a backend that operate in tandem.
◆ The frontend processes sequential monocular images in real-time for incremental mapping and tracking, while the backend performs global refinement to ensure a geometrically consistent result.</td></tr>
<tr><td>2025-09-21</td><td>ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM</td><td>[2509.16863](http://arxiv.org/pdf/2509.16863)</td><td>◆ We introduce ConfidentSplat, a novel 3D Gaussian Splatting (3DGS)-based SLAM system for robust, highfidelity RGB-only reconstruction.
◆ Addressing geometric inaccuracies in existing RGB-only 3DGS SLAM methods that stem from unreliable depth estimation, ConfidentSplat incorporates a core innovation: a confidence-weighted fusion mechanism.
◆ This mechanism adaptively integrates depth cues from multiview geometry with learned monocular priors (Omnidata ViT), dynamically weighting their contributions based on explicit reliability estimates-derived predominantly from multi-view geometric consistency-to generate high-fidelity proxy depth for map supervision.</td></tr>
<tr><td>2025-09-19</td><td>SLaM-DiMM: Shared Latent Modeling for Diffusion Based Missing Modality Synthesis in MRI</td><td>[2509.16019](http://arxiv.org/pdf/2509.16019)</td><td>◆ Brain MRI scans are often found in four modalities, consisting of T1-weighted with and without contrast enhancement (T1ce and T1w), T2-weighted imaging (T2w), and Flair.
◆ Leveraging complementary information from these different modalities enables models to learn richer, more discriminative features for understanding brain anatomy, which could be used in downstream tasks such as anomaly detection.
◆ However, in clinical practice, not all MRI modalities are always available due to various reasons.</td></tr>
<tr><td>2025-09-19</td><td>Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion</td><td>[2509.15673](http://arxiv.org/pdf/2509.15673)</td><td>◆ Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but most existing LiDAR-inertial-visual odometry (LIVO) systems rely on a single camera, leading to limited spatial coverage and degraded robustness.
◆ We present Omni-LIVO, the first tightly coupled multi-camera LIVO system that bridges the FoV mismatch between wide-angle LiDAR and conventional cameras.
◆ Omni-LIVO introduces a Cross-View direct tracking strategy that maintains photometric consistency across non-overlapping views, and extends the Error-State Iterated Kalman Filter (ESIKF) with multi-view updates and adaptive covariance weighting.</td></tr>
<tr><td>2025-09-18</td><td>Human Interaction for Collaborative Semantic SLAM using Extended Reality</td><td>[2509.14949](http://arxiv.org/pdf/2509.14949)</td><td>◆ Semantic SLAM (Simultaneous Localization and Mapping) systems enrich robot maps with structural and semantic information, enabling robots to operate more effectively in complex environments.
◆ However, these systems struggle in real-world scenarios with occlusions, incomplete data, or ambiguous geometries, as they cannot fully leverage the higher-level spatial and semantic knowledge humans naturally apply.
◆ We introduce HICS-SLAM, a Human-in-the-Loop semantic SLAM framework that uses a shared extended reality environment for real-time collaboration.</td></tr>
<tr><td>2025-09-18</td><td>Event-LAB: Towards Standardized Evaluation of Neuromorphic Localization Methods</td><td>[2509.14516](http://arxiv.org/pdf/2509.14516)</td><td>◆ Event-based localization research and datasets are a rapidly growing area of interest, with a tenfold increase in the cumulative total number of published papers on this topic over the past 10 years.
◆ Whilst the rapid expansion in the field is exciting, it brings with it an associated challenge: a growth in the variety of required code and package dependencies as well as data formats, making comparisons difficult and cumbersome for researchers to implement reliably.
◆ To address this challenge, we present Event-LAB: a new and unified framework for running several event-based localization methodologies across multiple datasets.</td></tr>
<tr><td>2025-09-17</td><td>MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping</td><td>[2509.14191](http://arxiv.org/pdf/2509.14191)</td><td>◆ Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage.
◆ We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS).
◆ Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map.</td></tr>
<tr><td>2025-09-17</td><td>BIM Informed Visual SLAM for Construction Monitoring</td><td>[2509.13972](http://arxiv.org/pdf/2509.13972)</td><td>◆ Simultaneous Localization and Mapping (SLAM) is a key tool for monitoring construction sites, where aligning the evolving as-built state with the as-planned design enables early error detection and reduces costly rework.
◆ LiDAR-based SLAM achieves high geometric precision, but its sensors are typically large and power-demanding, limiting their use on portable platforms.
◆ Visual SLAM offers a practical alternative with lightweight cameras already embedded in most mobile devices.</td></tr>
<tr><td>2025-09-16</td><td>Semantic 3D Reconstructions with SLAM for Central Airway Obstruction</td><td>[2509.13541](http://arxiv.org/pdf/2509.13541)</td><td>◆ Central airway obstruction (CAO) is a life-threatening condition with increasing incidence, caused by tumors in and outside of the airway.
◆ Traditional treatment methods such as bronchoscopy and electrocautery can be used to remove the tumor completely; however, these methods carry a high risk of complications.
◆ Recent advances allow robotic interventions with lesser risk.</td></tr>
<tr><td>2025-09-16</td><td>MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM</td><td>[2509.13536](http://arxiv.org/pdf/2509.13536)</td><td>◆ Recent advancements in 3D Gaussian Splatting (3DGS) have made a significant impact on rendering and reconstruction techniques.
◆ Current research predominantly focuses on improving rendering performance and reconstruction quality using high-performance desktop GPUs, largely overlooking applications for embedded platforms like micro air vehicles (MAVs).
◆ These devices, with their limited computational resources and memory, often face a trade-off between system performance and reconstruction quality.</td></tr>
<tr><td>2025-09-18</td><td>MATTER: Multiscale Attention for Registration Error Regression</td><td>[2509.12924](http://arxiv.org/pdf/2509.12924)</td><td>◆ Point cloud registration (PCR) is crucial for many downstream tasks, such as simultaneous localization and mapping (SLAM) and object tracking.
◆ This makes detecting and quantifying registration misalignment, i.e.,~{\it PCR quality validation}, an important task.
◆ All existing methods treat validation as a classification task, aiming to assign the PCR quality to a few classes.</td></tr>
<tr><td>2025-09-16</td><td>Match Chat: Real Time Generative AI and Generative Computing for Tennis</td><td>[2509.12592](http://arxiv.org/pdf/2509.12592)</td><td>◆ We present Match Chat, a real-time, agent-driven assistant designed to enhance the tennis fan experience by delivering instant, accurate responses to match-related queries.
◆ Match Chat integrates Generative Artificial Intelligence (GenAI) with Generative Computing (GenComp) techniques to synthesize key insights during live tennis singles matches.
◆ The system debuted at the 2025 Wimbledon Championships and the 2025 US Open, where it provided about 1 million users with seamless access to streaming and static data through natural language queries.</td></tr>
<tr><td>2025-09-15</td><td>Adaptive Motorized LiDAR Scanning Control for Robust Localization with OpenStreetMap</td><td>[2509.11742](http://arxiv.org/pdf/2509.11742)</td><td>本文提出了一种结合OpenStreetMap（OSM）先验的自适应激光雷达扫描控制方法，旨在提升机器人在不完整或过时地图环境中的定位鲁棒性。其核心贡献与创新点如下：

◆ 首次将OSM全局先验信息与局部可观测性预测相结合，构建了主动感知框架，优化了激光雷达扫描策略。  
◆ 提出了一种基于不确定性感知的模型预测控制方法，其中创新地加入了OSM感知项，实现了扫描资源按需分配。  
◆ 通过场景依赖的可观测性与OSM特征空间分布的自适应协调，有效减少特征稀疏区域的扫描浪费，提升定位精度。  
◆ 在校园、室内走廊和城市等多种真实场景中验证了该方法，相比恒定转速基线显著降低了轨迹误差，同时保持了扫描完整性。  
◆ 该工作展示了开源地图与自适应感知硬件协同的潜力，为复杂环境下实现高效、鲁棒的定位提供了新思路。</td></tr>
<tr><td>2025-09-15</td><td>See What I Mean? Mobile Eye-Perspective Rendering for Optical See-through Head-mounted Displays</td><td>[2509.11653](http://arxiv.org/pdf/2509.11653)</td><td>本文针对光学透射头显（OST HMD）中因世界相机与用户视角未对准导致的注册偏差问题，提出了移动式眼视角渲染（EPR）技术框架。  
◆ 实现并系统评估了三种基于软件的EPR方法：平面代理、网格代理和新型注视代理渲染。  
◆ 创新性地提出注视深度对齐的Gaze-Proxy EPR方法，通过眼动追踪动态适配用户注视深度。  
◆ 在商用设备HoloLens 2上实现无需额外硬件的轻量化解决方案，显著提升虚实注册精度。  
◆ 通过用户研究验证了精确EPR在真实任务中的重要性，证明注视代理法可作为几何重建方法的有效替代方案。  
◆ 开源完整的EPR框架，为后续研究提供重要技术基础。</td></tr>
<tr><td>2025-09-15</td><td>Gaussian-Plus-SDF SLAM: High-fidelity 3D Reconstruction at 150+ fps</td><td>[2509.11574](http://arxiv.org/pdf/2509.11574)</td><td>本文提出了一种结合高斯模型与符号距离场（SDF）的混合表示方法，用于实现高速高保真RGB-D SLAM。其核心贡献在于显著提升了三维重建的效率和实时性。  
◆ 提出高斯-SDF混合场景表示，利用SDF表达平滑几何与外观，再以高斯模型补充细节，兼顾重建质量与计算效率。  
◆ 通过SDF辅助减少高斯模型数量，比现有方法减少50%的高斯元件，降低计算和存储开销。  
◆ 优化过程仅针对高斯模型进行外观细化，所需迭代次数减少75%，大幅加速优化过程。  
◆ 基于该表示构建了GPS-SLAM系统，在真实Azure Kinect序列上达到150+ fps，相比现有方法提速一个数量级，同时保持相当的重建质量。  
该系统为实时高质量三维重建提供了新的解决方案，代码和数据将开源以促进后续研究。</td></tr>
<tr><td>2025-09-13</td><td>FastTrack: GPU-Accelerated Tracking for Visual SLAM</td><td>[2509.10757](http://arxiv.org/pdf/2509.10757)</td><td>该论文的核心贡献是提出了一种利用GPU加速来显著提升视觉-惯性SLAM系统跟踪模块性能的新方法FastTrack。  
◆ 创新性地利用GPU并行计算能力来加速跟踪过程中最耗时的组件，包括立体特征匹配和局部地图跟踪。  
◆ 在主流开源SLAM系统ORB-SLAM3的跟踪流程中实现了基于CUDA的GPU加速设计方案。  
◆ 在立体-惯性模式下，使用EuRoC和TUM-VI等标准数据集进行验证，在桌面平台和Jetson Xavier NX嵌入式平台上均实现了最高2.8倍的跟踪性能提升。  
◆ 通过GPU加速有效减少了每帧处理时间，从而降低了系统延迟与跟踪丢失风险，提升了SLAM系统的实时性与鲁棒性。</td></tr>
<tr><td>2025-09-12</td><td>Robust Localization in Modern Cellular Networks using Global Map Features</td><td>[2509.10433](http://arxiv.org/pdf/2509.10433)</td><td>该论文提出了一种增强型多路径同时定位与建图（MP-SLAM）方法，旨在利用现代蜂窝网络实现复杂环境下的高精度鲁棒定位。  
◆ 引入全局地图特征（GMF）存储库，整合历史遍历过程中收集的高质量地图特征（如虚拟锚点），提升系统对环境的全局感知能力。  
◆ 通过概率假设密度（PHD）滤波器将GMF动态集成到SLAM框架中，实现地图特征强度函数的时序传播与融合。  
◆ 在密集城市场景中利用LTE信号进行真实实验验证，系统在严重多路径传播和小区间干扰条件下仍保持定位精度与鲁棒性。  
该方法优于传统基于自感知传感器的定位方案及常规MP-SLAM方法，尤其适用于5G/6G网络中非视距、多路径等恶劣信号环境。</td></tr>
<tr><td>2025-09-11</td><td>SMapper: A Multi-Modal Data Acquisition Platform for SLAM Benchmarking</td><td>[2509.09509](http://arxiv.org/pdf/2509.09509)</td><td>SMapper的核心贡献是提出了一种专为SLAM研究设计的开源多模态数据采集平台，以解决现有数据集在传感器多样性、环境覆盖和实验可复现性方面的不足。其创新点包括：
◆ 采用开放式硬件设计，集成了同步的LiDAR、多相机和惯性测量单元，提供了丰富的多模态传感能力。
◆ 提供了一套可靠的标定与同步流程，确保了跨模态数据在时空上的精确对齐。
◆ 平台设计兼具手持和机器人搭载的灵活性，支持研究社区扩展硬件功能并复现实验。
◆ 公开发布了名为SMapper-light的数据集，包含室内外典型场景的多模态同步数据及高精度真值轨迹。
◆ 基于该数据集对主流LiDAR与视觉SLAM算法进行了性能评估，为算法开发与评测提供了坚实基础。</td></tr>
<tr><td>2025-09-11</td><td>S-BEVLoc: BEV-based Self-supervised Framework for Large-scale LiDAR Global Localization</td><td>[2509.09110](http://arxiv.org/pdf/2509.09110)</td><td>S-BEVLoc提出了一种基于鸟瞰图（BEV）的自监督激光雷达全局定位框架，其核心贡献在于摆脱了对高精度真值位姿的依赖。  
◆ 首次构建了无需地面真值位姿的自监督学习框架，通过利用关键点之间的已知地理距离构建训练三元组，大幅降低了数据采集成本。  
◆ 提出基于BEV图像的单帧数据构造三元组的方法，利用以关键点为中心的BEV图像块之间的空间距离关系生成监督信号。  
◆ 采用CNN提取局部特征并结合NetVLAD聚合全局描述符，有效提升了特征表达能力和检索效率。  
◆ 创新性地引入SoftCos损失函数，优化三元组学习过程，增强了模型对相似场景的区分能力。  
在KITTI和NCLT大规模数据集上的实验表明，该框架在位置识别、回环检测和全局定位任务上达到了领先性能，同时展现出极强的可扩展性。</td></tr>
<tr><td>2025-09-10</td><td>Behaviorally Heterogeneous Multi-Agent Exploration Using Distributed Task Allocation</td><td>[2509.08242](http://arxiv.org/pdf/2509.08242)</td><td>本文针对行为异构多机器人系统，提出了一种分布式探索与任务分配框架。  
◆ 引入行为熵（BE）量化机器人探索前沿区域（AoI）的异构效用，有效融合了不同机器人的行为偏好。  
◆ 将任务分配问题转化为非合作博弈，并设计分布式算法d-PRGRA收敛至纳什均衡，证明该均衡即最优分配方案。  
◆ 针对效用未知情况提出近似奖励方法，提供具有鲁棒性的性能边界保证。  
◆ 算法通信成本低、收敛快，并通过仿真验证了感知半径、精度和团队异构性对探索效率的积极影响。  
研究表明，行为异构团队在探索时间和路径长度上均优于同构团队。</td></tr>
<tr><td>2025-09-10</td><td>Online Dynamic SLAM with Incremental Smoothing and Mapping</td><td>[2509.08197](http://arxiv.org/pdf/2509.08197)</td><td>本文首次将增量优化技术应用于动态SLAM领域，实现了在线实时估计能力。  
◆ 提出了一种新颖的因子图模型和系统架构，充分利用现有增量优化方法。  
◆ 在保证精度的同时显著提升计算效率，相机位姿和物体运动估计精度达到或超过现有最优方法。  
◆ 通过分析问题结构，证明了该方法具有良好的可扩展性并揭示了增量求解的挑战。  
◆ 所构建的问题结构特别适合增量求解器，系统架构进一步带来性能提升。  
最终实现了相比现有方法5倍的加速比，为动态SLAM的在线应用提供了实用解决方案。</td></tr>
<tr><td>2025-09-09</td><td>SVN-ICP: Uncertainty Estimation of ICP-based LiDAR Odometry using Stein Variational Newton</td><td>[2509.08069](http://arxiv.org/pdf/2509.08069)</td><td>SVN-ICP是一种用于激光雷达里程计的新型ICP算法，其核心贡献在于提供了精确的位姿估计和可靠的协方差不确定性。  
◆ 首次将流形上的Stein变分牛顿法(SVN)应用于ICP问题，以推断后验分布和噪声参数。  
◆ 无需对噪声进行显式建模或手动调参，通过粒子近似即可自动学习不确定性。  
◆ 即使在激光雷达退化环境中（如长走廊或开阔地），也能保持一致的精度和不确定性估计。  
该方法通过一个误差状态卡尔曼滤波器与IMU融合，在多数据集和不同机器人平台上验证了其优越性。  
实验表明，其在挑战性场景中的性能超越了同类最佳方法。</td></tr>
<tr><td>2025-09-09</td><td>Sensing with Mobile Devices through Radio SLAM: Models, Methods, Opportunities, and Challenges</td><td>[2509.07775](http://arxiv.org/pdf/2509.07775)</td><td>本文探讨了无线电SLAM作为6G通感一体化的关键技术，其核心贡献在于系统性地构建了通过移动设备实现环境感知与定位的理论框架。  
◆ 提出将无线电SLAM确立为6G通感一体化（ISAC）的关键实现路径，利用通信信号同时完成定位与建图。  
◆ 系统分析了不同频段下无线电SLAM的性能权衡，包括覆盖范围、分辨率与硬件需求之间的关联。  
◆ 强调了与传感、定位及协同网络融合的新机遇，为复杂场景下的协同感知提供了方向。  
◆ 研究成果为自动驾驶、工业机器人等6G应用领域的标准化解决方案奠定了理论基础。  
论文最终为未来移动设备实现高精度环境感知提供了重要的模型、方法与挑战分析。</td></tr>
<tr><td>2025-09-10</td><td>Robust Radar SLAM for Vehicle Parking Applications</td><td>[2509.07683](http://arxiv.org/pdf/2509.07683)</td><td>该论文针对自动泊车场景提出了一种高精度的雷达SLAM系统，重点解决了恶劣天气下的鲁棒定位问题。其核心创新点包括：

◆ 提出多普勒增强的雷达SLAM方法，通过融合特征点位置和多普勒速度信息提升数据关联和滤波器收敛的鲁棒性  
◆ 采用以机器人为中心的建模方式（robocentric formulation），有效降低了运动线性化误差  
◆ 支持多雷达传感器融合架构，增强系统感知能力和环境覆盖范围  
◆ 引入基于信息理论的特征筛选策略，优化计算效率并维持地图稀疏性  
实验表明该方法在定位精度和鲁棒性方面优于现有技术，能够满足自动泊车对厘米级精度的严苛要求，同时支持在线校准以降低部署成本。</td></tr>
<tr><td>2025-09-08</td><td>Co-Located VR with Hybrid SLAM-based HMD Tracking and Motion Capture Synchronization</td><td>[2509.06582](http://arxiv.org/pdf/2509.06582)</td><td>该论文提出了一种多用户协同定位VR框架，通过混合跟踪与同步技术实现高精度共享沉浸体验。  
◆结合外部动捕系统与SLAM内置跟踪，兼顾高帧率、低延迟与长期稳定性，克服传统外追延迟或单次校准漂移问题。  
◆动态重对齐机制在保持SLAM本地响应速度的同时，支持按需与外部系统实时校正，消除累积误差。  
◆实现跨设备实时姿态共享，确保多用户间空间对齐一致性与交互自然性。  
◆系统在评估中展现出优于现有方案的舒适性、可扩展性与鲁棒性，满足自然多用户交互的空间精度需求。</td></tr>
<tr><td>2025-09-08</td><td>Real-time Photorealistic Mapping for Situational Awareness in Robot Teleoperation</td><td>[2509.06433](http://arxiv.org/pdf/2509.06433)</td><td>该论文核心贡献是提出了一种实时逼真地图构建系统，显著提升了机器人在未知环境中的远程遥操作效率。  
◆ 创新性地将高斯溅射SLAM技术与在线地图遥操作系统进行模块化集成  
◆ 采用基于GPU的高效计算架构，解决了传统方法计算成本高的问题  
◆ 实现了实时生成视觉精确的3D地图，克服了现有系统实时性差的缺陷  
◆ 通过真实无人机实验验证了系统能提升决策速度和环境交互准确性  
◆ 为陌生环境下的远程操作提供了沉浸式视觉反馈和情境感知支持</td></tr>
<tr><td>2025-09-07</td><td>DVLO4D: Deep Visual-Lidar Odometry with Sparse Spatial-temporal Fusion</td><td>[2509.06023](http://arxiv.org/pdf/2509.06023)</td><td>DVLO4D提出了一种新颖的视觉-激光雷达里程计框架，通过稀疏时空融合显著提升了定位的精度与鲁棒性。其核心创新点包括：
◆ 提出稀疏查询融合方法，利用稀疏的LiDAR查询实现高效的多模态数据融合。
◆ 设计了时序交互与更新模块，将历史预测位姿与当前帧数据融合，为位姿估计提供更优初始值，有效抑制累积误差。
◆ 引入时序片段训练策略和集体平均损失机制，通过多帧损失聚合实现全局优化，减少长序列中的尺度漂移。
该框架在KITTI和Argoverse数据集上实现了最先进的性能，同时具备高效推理能力（82毫秒/帧），为实时部署提供了可能。</td></tr>
<tr><td>2025-09-06</td><td>LiDAR-BIND-T: Improving SLAM with Temporally Consistent Cross-Modal LiDAR Reconstruction</td><td>[2509.05728](http://arxiv.org/pdf/2509.05728)</td><td>本文提出了LiDAR-BIND-T，通过增强时间一致性改进了多模态融合SLAM系统。其核心贡献与创新点包括：
◆ 引入时间嵌入相似性机制，显式对齐连续时刻的潜在特征以保持时序连贯。
◆ 提出运动对齐变换损失函数，确保预测点云与真实LiDAR点云之间的位移一致性。
◆ 设计专用时序融合模块，采用窗口时序融合策略整合历史信息。
◆ 优化模型架构以更好地保留空间结构，提升跨模态（雷达/声纳到LiDAR）重建质量。
◆ 提出基于Frèchet视频运动距离（FVMD）和相关峰值距离的实用时序评价指标，为SLAM性能提供量化依据。
该系统保持即插即用的多模态融合能力，显著提升了下游SLAM的轨迹精度和地图构建鲁棒性。</td></tr>
<tr><td>2025-09-04</td><td>Stitching the Story: Creating Panoramic Incident Summaries from Body-Worn Footage</td><td>[2509.04370](http://arxiv.org/pdf/2509.04370)</td><td>该论文的核心贡献是开发了一个自动化计算机视觉流程，将执法或急救人员佩戴的相机所拍摄的长视频，转化为简洁的全景图像摘要，以支持快速态势感知和事件分析。

◆ 提出利用单目SLAM技术从视频中估计相机运动轨迹并重建场景空间布局，为创建空间连贯的摘要奠定基础。
◆ 通过沿相机轨迹对位姿进行聚类来识别关键视点，确保摘要能全面覆盖场景的重要区域。
◆ 从每个视点簇中选取代表性帧，并采用多帧图像拼接技术将其融合成高质量、信息丰富的全景图像。
◆ 最终生成的全景摘要图像能直观展示复杂事件现场的全貌，显著提升了视频审查和决策效率。</td></tr>
<tr><td>2025-09-03</td><td>Efficient Active Training for Deep LiDAR Odometry</td><td>[2509.03211](http://arxiv.org/pdf/2509.03211)</td><td>本文提出了一种用于深度激光雷达里程计的高效主动训练框架，核心贡献在于显著减少训练数据需求的同时提升模型泛化能力。  
◆ 设计了主动训练框架，通过选择性提取多样化环境中的数据来降低训练负担。  
◆ 提出初始训练集选择策略（ITSS），通过分析通用天气下的运动序列节点和边缘来构建多样性丰富的初始数据集。  
◆ 针对复杂场景（如雪天），引入主动增量选择策略（AIS），利用场景重建和预测不一致性迭代筛选样本以优化模型。  
实验表明，仅使用52%的序列数据即可达到全数据集训练性能，验证了该框架的高效性和鲁棒性。  
该方法为激光雷达里程计系统在多变环境中实现更精准可靠的导航奠定了基础。</td></tr>
<tr><td>2025-09-03</td><td>IL-SLAM: Intelligent Line-assisted SLAM Based on Feature Awareness for Dynamic Environments</td><td>[2509.02972](http://arxiv.org/pdf/2509.02972)</td><td>本文提出了一种基于特征感知的智能线辅助动态SLAM系统IL-SLAM，其核心贡献在于解决了动态环境下特征管理的效率与质量问题。
◆ 提出了一种特征感知机制，能够主动评估当前点特征的充足性，以此智能判断是否需要引入线特征进行辅助，从而避免不必要的特征引入。
◆ 系统仅在必要时激活线特征支持，显著降低了因持续引入额外特征带来的计算开销，提升了系统运行效率。
◆ 通过选择性引入机制，有效减少了低质量线特征和噪声的累积，避免了其对系统长期性能的潜在负面影响。
◆ 在线特征的后端处理上，创新地将其用于辅助跟踪、局部建图与回环检测以优化初始位姿，但将其排除在全局优化之外，确保了全局地图的稳定性。
在TUM数据集上的实验表明，该方案在ATE和RPE精度上均优于ORB-SLAM3基线及其他动态SLAM与多特征方法。</td></tr>
<tr><td>2025-09-02</td><td>Coral: A Unifying Abstraction Layer for Composable Robotics Software</td><td>[2509.02453](http://arxiv.org/pdf/2509.02453)</td><td>本文提出了Coral，一个用于组合式机器人软件的统一抽象层，旨在解决机器人系统集成困难的核心问题。  
◆引入了一个高层抽象层，专注于构建、部署和协调独立软件组件，最大化组合性以实现快速系统集成，无需修改底层代码。  
◆通过语义化约束集成过程，减少配置负担，同时保持对不同领域、系统和任务的适应能力，而不替代现有工具。  
◆实现了实际可用的组合性，提高了组件可重用性、系统可重构性，并降低专家和非专家用户的访问门槛。  
◆在多种复杂场景（如基于LiDAR的SLAM和多机器人防腐任务）中验证了其有效性，证明了广泛应用的潜力。  
Coral以开源形式发布，为机器人软件集成挑战提供了可扩展的解决方案。</td></tr>
<tr><td>2025-09-02</td><td>Generalizing Unsupervised Lidar Odometry Model from Normal to Snowy Weather Conditions</td><td>[2509.02011](http://arxiv.org/pdf/2509.02011)</td><td>该论文的核心贡献是提出了一种无监督激光雷达里程计模型，能够从正常天气泛化到雪天等恶劣环境。其创新点包括：
◆ 提出Patch Spatial Measure（PSM）模块，通过评估点云块内点的离散程度来有效检测稀疏噪声。
◆ 设计了Patch Point Weight Predictor（PPWP），通过预测自适应点级权重增强局部区域的特征判别能力。
◆ 采用强度阈值掩码快速抑制激光雷达附近的密集雪花噪声簇，保障实时性。
◆ 结合多模态特征融合优化点级权重预测，提升系统在恶劣天气下的整体鲁棒性。
该模型仅在正常天气数据上训练，但在雪天和动态场景中表现出强劲性能，显著提高了跨环境泛化能力。</td></tr>
<tr><td>2025-09-02</td><td>Doctoral Thesis: Geometric Deep Learning For Camera Pose Prediction, Registration, Depth Estimation, and 3D Reconstruction</td><td>[2509.01873](http://arxiv.org/pdf/2509.01873)</td><td>该论文的核心贡献是开发了结合几何先验与深度学习的几何深度学习方法，以解决三维视觉中的关键任务。  
◆ 针对相机位姿估计、点云配准、深度预测及三维重建等任务，提出了定制化的几何深度学习模型。  
◆ 通过引入深度信息、表面法线和等变性等几何约束，增强了模型的表示准确性与鲁棒性。  
◆ 克服了传统方法（如SfM和SLAM）在非结构化环境中特征模糊和几何表示不足的局限性。  
◆ 在文化遗产数字化保护和VR/AR等实际应用中验证了方法的有效性，推动了三维映射与场景重建技术的发展。</td></tr>
<tr><td>2025-09-01</td><td>ViSTA-SLAM: Visual SLAM with Symmetric Two-view Association</td><td>[2509.01584](http://arxiv.org/pdf/2509.01584)</td><td>ViSTA-SLAM是一个无需已知相机内参即可实时运行的单目视觉SLAM系统，其核心贡献在于通过一个轻量化的对称式双视图关联模型显著提升了系统性能与适用性。

◆ 提出一种轻量级对称双视图关联（STA）模型作为前端，仅需两张RGB图像即可同时估计相对相机位姿并回归局部点云地图。
◆ 该前端设计极大降低了模型复杂度，其大小仅为同类先进方法的35%，同时生成了更高质量的双视图约束用于后续优化。
◆ 在后端构建了一个特殊的Sim(3)位姿图，通过融入回环检测来有效处理累积的尺度漂移问题。
◆ 整个系统不依赖预先标定的相机内参，使其能广泛适用于各种不同的相机设置，提升了实用性与便捷性。
大量实验表明，该系统在相机跟踪精度和稠密三维重建质量上均优于当前主流方法。</td></tr>
<tr><td>2025-09-01</td><td>FGO-SLAM: Enhancing Gaussian SLAM with Globally Consistent Opacity Radiance Field</td><td>[2509.01547](http://arxiv.org/pdf/2509.01547)</td><td>FGO-SLAM的核心贡献在于提出了一种基于全局一致不透明度辐射场的新型高斯SLAM系统，显著提升了场景几何重建的质量和系统的追踪精度。

◆ 采用不透明度辐射场作为场景表示，有效增强了系统的几何建图性能。
◆ 在初始位姿估计后，引入全局调整优化策略，同时优化相机位姿和稀疏点云，确保了鲁棒的追踪效果。
◆ 维护了一个基于3D高斯且全局一致的不透明度辐射场，并创新性地增加了深度失真和法向一致性约束项来优化场景表示。
◆ 通过构建四面体网格并提取等值面，实现了直接从3D高斯中提取物体表面，简化了表面重建流程。
实验结果表明，该方法在多个真实和大型合成数据集上均实现了最先进的追踪与建图性能。</td></tr>
<tr><td>2025-09-01</td><td>SR-SLAM: Scene-reliability Based RGB-D SLAM in Diverse Environments</td><td>[2509.01111](http://arxiv.org/pdf/2509.01111)</td><td>SR-SLAM提出了一种基于场景可靠性的RGB-D SLAM框架，旨在提升视觉SLAM在不同环境下的精度与鲁棒性。其核心创新在于引入了一个统一的多指标场景可靠性评估机制，并基于该评估开发了四项关键技术。  
◆ 提出自适应动态区域选择方法，采用灵活几何约束以提升动态环境适应性。  
◆ 开发深度辅助的自调整聚类算法，在高维环境下实现高效的动态特征剔除。  
◆ 设计可靠性感知的位姿优化机制，在特征不足时动态融合直接法以提升估计稳定性。  
◆ 提出基于可靠性的关键帧选择与加权优化策略，在保证精度的同时显著降低计算开销。  
实验证明该系统在公开数据集和真实场景中优于现有动态SLAM方法，精度和鲁棒性提升最高达90%。</td></tr>
<tr><td>2025-08-31</td><td>DyPho-SLAM : Real-time Photorealistic SLAM in Dynamic Environments</td><td>[2509.00741](http://arxiv.org/pdf/2509.00741)</td><td>DyPho-SLAM是一种能够在动态环境中实时运行的视觉SLAM系统，其核心贡献是解决了动态物体导致的相机跟踪漂移和地图模糊问题，并实现了高保真的密集地图重建。

◆ 系统创新性地整合了先验图像信息来生成精细化掩码，有效减少了因动态物体误判而产生的噪声。
◆ 设计了自适应特征提取策略，在移除动态障碍后为优化过程增强了约束，显著提升了系统的鲁棒性。
◆ 成功将高保真高斯泼溅表示应用于动态场景，在实现实时运行的同时，保持了资源的高效利用。
实验结果表明，该系统在公开动态RGB-D数据集上的相机位姿估计和密集地图重建精度达到了领先水平。</td></tr>
<tr><td>2025-08-30</td><td>AGS: Accelerating 3D Gaussian Splatting SLAM via CODEC-Assisted Frame Covisibility Detection</td><td>[2509.00433](http://arxiv.org/pdf/2509.00433)</td><td>该论文提出了AGS，一个算法-硬件协同设计框架，旨在显著加速基于3D高斯泼溅的SLAM系统。其核心创新点在于利用视频帧间的相似性来避免冗余计算，并通过专用硬件引擎实现高效处理。

◆ 在软件层面，提出了一种根据机器人运动速度调整的由粗到精的位姿跟踪方法，优化了计算效率。
◆ 创新地通过跨帧共享高斯点的贡献信息，避免了大量重复计算，减少了处理负担。
◆ 在硬件层面，设计了一个帧共视性检测引擎，能够直接从视频编解码器（CODEC）中提取中间数据来快速判断帧间相似性。
◆ 实现了带有工作负载调度器的位姿跟踪引擎和建图引擎，以高效部署整个AGS算法。
实验结果表明，AGS相比移动和高端GPU以及现有加速器方案，取得了最高达17.12倍、6.71倍和5.41倍的加速效果。</td></tr>
<tr><td>2025-08-29</td><td>The Rosario Dataset v2: Multimodal Dataset for Agricultural Robotics</td><td>[2508.21635](http://arxiv.org/pdf/2508.21635)</td><td>本文介绍了Rosario Dataset v2，一个专为农业机器人设计的综合性多模态数据集。其核心贡献在于提供了一个高度真实且技术严谨的农业环境数据基准。

◆ 数据集包含超过两小时在大豆田采集的多传感器数据，涵盖红外/彩色相机、IMU、多种GNSS模式和轮式里程计。
◆ 它精准捕捉了农业环境的典型挑战，如光照变化、运动模糊、颠簸地形和视觉相似的长序列，填补了该领域数据空白。
◆ 数据集设计满足了多模态SLAM系统评测的关键需求，其突出特点是实现了硬件同步和提供了长轨迹上的6自由度精确真值。
◆ 作者利用该数据集对前沿SLAM算法进行了基准测试，揭示了它们在农业应用中的现有局限性，验证了数据集的实用价值。
该数据集旨在推动农业机器人技术在定位、建图、感知与导航方面的算法研发和性能评估。</td></tr>
<tr><td>2025-08-28</td><td>Adam SLAM - the last mile of camera calibration with 3DGS</td><td>[2508.20526](http://arxiv.org/pdf/2508.20526)</td><td>该论文提出了一种利用3D高斯泼溅（3DGS）模型优化相机标定的新方法。  
◆ 首创通过新视图合成的颜色损失反向传播，直接优化相机参数（如位姿和焦距），突破了传统标定依赖专用采集流程的限制。  
◆ 将标定问题转化为可微分优化问题，无需真实场景真值，仅依靠渲染图像质量即可评估和提升标定精度。  
◆ 在3DGS标准数据集上实现了平均0.4 dB的PSNR提升，显著提高了新视图合成质量。  
该方法尤其适用于对标定精度要求极高的参考场景（如Mip-NeRF 360），虽计算耗时较长，但为高精度神经渲染提供了关键技术支持。</td></tr>
<tr><td>2025-08-24</td><td>SEER-VAR: Semantic Egocentric Environment Reasoner for Vehicle Augmented Reality</td><td>[2508.17255](http://arxiv.org/pdf/2508.17255)</td><td>SEER-VAR提出了一个用于车载增强现实（AR）的创新框架，其核心贡献在于将语义理解、环境感知SLAM与大语言模型推荐系统进行了统一。  
◆首创了通过深度引导的视觉-语言基础模型，动态分离车辆驾驶舱与外部道路场景，解决了传统系统假设静态或单一视角的局限。  
◆设计了双分支的上下文感知SLAM（CASB），分别对舱内和车外环境进行自我运动跟踪，提升了复杂驾驶情境下的空间对齐鲁棒性。  
◆引入基于GPT的推荐模块，生成情境感知的AR叠加内容，如仪表盘提示和危险警报，增强了信息的实用性和场景贴合度。  
◆发布了一个名为EgoSLAM-Drive的真实世界数据集，包含同步的自中心视角图像、6DoF真实姿态和多样驾驶场景下的AR标注，支持系统评估与后续研究。  
该框架在实验中表现出优越的空间对齐能力和AR渲染一致性，并通过用户研究验证了其在提升场景理解、信息相关性和驾驶舒适度方面的有效性。</td></tr>
<tr><td>2025-08-24</td><td>VROOM - Visual Reconstruction over Onboard Multiview</td><td>[2508.17172](http://arxiv.org/pdf/2508.17172)</td><td>VROOM提出了一种仅依靠F1赛车上车载摄像头视频来重建赛道三维模型的系统。
◆ 创新性地利用真实世界高速运动场景（2023年摩纳哥大奖赛）的极端车载视频数据，应对高动态和镜头骤变的挑战。
◆ 系统化地分析并融合了多种先进方法（如DROID-SLAM、AnyCam、Monst3r），构建了一个完整的处理流程。
◆ 开发了针对性的预处理技术，包括多种掩蔽方法、时间分块和分辨率缩放，以处理剧烈运动并满足计算限制。
实验表明，VROOM能在复杂环境中部分恢复赛道和车辆轨迹，验证了仅用车载视频进行可扩展4D重建的现实可行性。</td></tr>
<tr><td>2025-08-23</td><td>DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration</td><td>[2508.17034](http://arxiv.org/pdf/2508.17034)</td><td>本文提出了一种新颖的双空间刚性配准方法DualReg，有效结合了基于特征匹配和局部几何匹配的优势。  
◆ 创新性地引入双空间范式，协同利用特征匹配处理大变换差异和几何匹配实现精细对齐的能力。  
◆ 设计高效过滤机制，采用轻量级单点RANSAC算法与细化模块，快速剔除不可靠的特征对应关系。  
◆ 提出将过滤后的对应点作为锚点，提取几何代理并构建高效目标函数，配合定制求解器优化变换估计。  
◆ 在KITTI数据集上实现最高32倍CPU时间加速，且在保持精度的同时显著提升计算效率。  
该方法为噪声环境下部分重叠数据的实时刚性配准提供了实用解决方案。</td></tr>
<tr><td>2025-08-23</td><td>A Workflow for Map Creation in Autonomous Vehicle Simulations</td><td>[2508.16856](http://arxiv.org/pdf/2508.16856)</td><td>本文针对自动驾驶仿真中高精度地图创建困难且耗时的问题，提出了一套定制化工作流。其核心贡献与创新点包括：
◆ 设计了一个简化且高效的制图流程，显著降低了创建仿真就绪地图的难度和资源消耗。
◆ 以CARLA等主流仿真器为应用背景，成功生成了安大略理工大学停车场的3D地图实例，验证了工作流的可行性。
◆ 该工作流为开发者提供了更灵活的解决方案，减少了对特定仿真平台或昂贵计算资源的依赖。
未来研究方向包括集成SLAM技术、提升多仿真器兼容性以及优化经纬度数据处理精度。</td></tr>
<tr><td>2025-08-22</td><td>COSMO-Bench: A Benchmark for Collaborative SLAM Optimization</td><td>[2508.16731](http://arxiv.org/pdf/2508.16731)</td><td>该论文的核心贡献是创建并发布了首个专门用于评估多机器人协同SLAM优化算法的基准测试套件COSMO-Bench。其创新点包括：
◆ 填补了多机器人协同SLAM领域缺乏标准评估基准的空白，解决了该领域研究难以公平比较的痛点。
◆ 提供了24个高质量数据集，这些数据源自真实世界的LiDAR数据和最先进的C-SLAM前端算法，确保了数据的真实性和可靠性。
◆ 所有数据集均开源，并配有永久可访问的DOI，极大地方便了研究社区的使用和后续研究。
该基准的发布有望像传统单机器人SLAM基准那样，推动多机器人协同SLAM优化算法的标准化测试和性能提升。</td></tr>
<tr><td>2025-08-22</td><td>GPL-SLAM: A Laser SLAM Framework with Gaussian Process Based Extended Landmarks</td><td>[2508.16459](http://arxiv.org/pdf/2508.16459)</td><td>本文提出了一种基于高斯过程（GP）地标的新型激光SLAM框架GPL-SLAM。  
◆ 采用高斯过程对环境中物体的轮廓进行建模，替代传统的栅格地图或点云配准方法。  
◆ 提出在线递归更新方案，能够高效更新轮廓并显著减少内存使用。  
◆ 在完全贝叶斯框架下联合推断机器人位姿与基于物体的地图，支持语义信息提取。  
◆ 提供物体形状的置信边界，为安全导航和探索等下游任务提供关键信息。  
实验证明该方法在合成与真实场景中均能实现精确的定位与建图。</td></tr>
<tr><td>2025-08-21</td><td>GelSLAM: A Real-time, High-Fidelity, and Robust 3D Tactile SLAM System</td><td>[2508.15990](http://arxiv.org/pdf/2508.15990)</td><td>GelSLAM提出了一种仅依靠触觉感知即可实现实时、高保真和鲁棒三维SLAM的系统。  
◆ 首创完全基于触觉传感的实时三维SLAM系统，摆脱对视觉的依赖，适用于遮挡场景。  
◆ 采用触觉衍生的表面法线和曲率进行位姿跟踪与回环检测，替代传统点云方法，提升鲁棒性。  
◆ 实现了长时间、低误差、低漂移的实时物体运动跟踪，即使对木质工具等低纹理物体也有效。  
◆ 能够以亚毫米级精度高保真重建物体形状，显著提升触觉感知的空间范围和全局能力。  
该系统将触觉感知从局部接触扩展至全局长时序空间感知，为高精度灵巧操作任务奠定了基础。</td></tr>
<tr><td>2025-08-19</td><td>SLAM-based Safe Indoor Exploration Strategy</td><td>[2508.14235](http://arxiv.org/pdf/2508.14235)</td><td>本文提出了一种基于SLAM的室内安全探索策略，主要面向具有圆形底盘的差速驱动机器人。其核心贡献在于将安全性作为最高优先级，同时兼顾未知空间的高效探索。
◆ 针对非点式且无法瞬时调整位姿的圆形机器人，设计了专用的安全探索框架。
◆ 采用多传感器融合方案，结合IMU、3D-LiDAR和RGB-D相机，通过RTAB-SLAM实现精准定位与建图。
◆ 提出了基于“安全骨架”的路径规划方法，使机器人始终尽可能远离静态障碍物。
◆ 探索过程中朝向空间中的开放区域前进，优先保障安全，再最大化探索未测区域。
实验通过ROS移动机器人平台验证了该策略的有效性与实用性。</td></tr>
<tr><td>2025-08-19</td><td>Online 3D Gaussian Splatting Modeling with Novel View Selection</td><td>[2508.14014](http://arxiv.org/pdf/2508.14014)</td><td>该研究针对仅使用RGB图像序列进行在线3D高斯泼溅（3DGS）建模的挑战，提出了创新解决方案。  
◆ 提出了自适应视图选择新机制，通过在线分析重建质量，智能选取最优的非关键帧进行补充训练。  
◆ 突破了传统方法仅依赖关键帧的局限，通过融合关键帧和精选的非关键帧，从多视角优化不完整区域。  
◆ 设计了集成在线多视图立体（MVS）技术的框架，确保3D信息在整个建模过程中的一致性。  
◆ 在在线处理的计算约束下，实现了更完整的场景重建，尤其在复杂户外场景中显著优于现有方法。  
该方法无需大量帧或迭代训练，即可高效生成高质量且更完整的3DGS模型。</td></tr>
<tr><td>2025-08-19</td><td>ROVER: Robust Loop Closure Verification with Trajectory Prior in Repetitive Environments</td><td>[2508.13488](http://arxiv.org/pdf/2508.13488)</td><td>该论文提出了一种用于重复环境下回环闭合验证的鲁棒方法ROVER。其核心创新在于利用机器人运动轨迹的先验知识来增强验证可靠性。  
◆ 首次将历史轨迹作为空间-时间先验约束引入回环验证框架，突破传统仅依赖外观特征的局限。  
◆ 提出通过位姿图优化生成候选回环的轨迹假设，并设计评分机制评估其与先验轨迹的一致性。  
◆ 在高度重复环境中有效剔除虚假回环，解决了外观相似性导致的误检测难题。  
实验表明，该方法在公开数据集和真实场景中均显著提升验证准确率，并可无缝集成至现有SLAM系统。开源代码与数据集为相关研究提供了重要基准。</td></tr>
<tr><td>2025-08-14</td><td>Super LiDAR Reflectance for Robotic Perception</td><td>[2508.10398](http://arxiv.org/pdf/2508.10398)</td><td>◆ 提出了一种创新框架，能够从稀疏LiDAR数据生成密集的反射率图像，解决了低成本LiDAR因数据稀疏性导致的应用受限问题。  
◆ 针对非重复扫描LiDAR（NRS-LiDAR）的特性，设计了专用的稠密化网络，优化了反射率图像的生成效果。  
◆ 解决了反射率校准和从静态场景到动态场景转换的关键技术难题，实现了真实场景下的密集反射率图像重建。  
◆ 构建了一个全面的LiDAR反射率图像稠密化数据集，为后续研究提供了重要基础。  
◆ 展示了稠密反射率图像在闭环检测和交通车道检测等多种机器人感知任务中的实际应用价值。  
◆ 通过主动光学感知重新定义了视觉边界，为机器人感知领域提供了新的技术思路和方法。</td></tr>
<tr><td>2025-08-12</td><td>Transient Noise Removal via Diffusion-based Speech Inpainting</td><td>[2508.08890](http://arxiv.org/pdf/2508.08890)</td><td>◆ 提出PGDI框架，基于扩散模型实现语音修复，能有效恢复长达1秒的严重受损或缺失语音段。  
◆ 突破传统方法局限，在保持说话人身份、韵律和环境因素（如混响）的同时，处理不同说话人和长间隙的挑战。  
◆ 引入分类器引导机制，特别是音素级引导，显著提升重建保真度。  
◆ 实现说话人无关的鲁棒性，即使语音段被强瞬态噪声（如烟花、关门声）完全掩盖，仍能有效修复。  
◆ 在实验中验证了PGDI的优越性能，无论是已知还是未知文本的场景下均表现良好，文本辅助可进一步提升效果。</td></tr>
<tr><td>2025-08-09</td><td>EGS-SLAM: RGB-D Gaussian Splatting SLAM with Events</td><td>[2508.07003](http://arxiv.org/pdf/2508.07003)</td><td>EGS-SLAM的核心贡献是通过融合事件相机与RGB-D数据，解决了传统高斯泼溅SLAM在运动模糊场景下的性能退化问题。其创新点包括：

◆ 提出首个结合事件数据与RGB-D输入的高斯泼溅SLAM框架，通过事件流补偿运动模糊，同时利用RGB-D数据弥补事件流的稀疏性。

◆ 创新性地建模了相机曝光期间的连续运动轨迹，在统一的高斯泼溅场景中实现事件感知与模糊感知的联合跟踪与建图。

◆ 设计了可学习的相机响应函数，有效对齐事件与图像的动态范围差异，提升数据融合精度。

◆ 引入无事件损失函数，有效抑制重建过程中的振铃伪影，提高三维重建质量。

◆ 构建了包含合成与真实场景的新数据集，验证了方法在严重运动模糊下的优越性能，轨迹精度与重建质量均超越现有方法。</td></tr>
<tr><td>2025-08-07</td><td>Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the Impact of Pretraining on High-Resource Languages</td><td>[2508.05149](http://arxiv.org/pdf/2508.05149)</td><td>◆ 研究了语音大语言模型(Speech LLMs)在低资源自动语音识别(ASR)场景下的数据量需求，填补了该领域的研究空白。  
◆ 提出采用SLAM-ASR框架（轻量级可训练投影器连接语音编码器与大模型），验证了即使小规模训练数据也能达到Whisper-only模型的性能基准。  
◆ 首次证明利用高资源语言预训练的单一或多语言投影器能显著缓解数据稀缺问题，尤其在极小训练集时效果更突出。  
◆ 通过多语言大模型(EuroLLM/Salamandra)与whisper-large-v3-turbo的组合实验，为低资源语言的跨语言迁移提供了实证依据。  
◆ 在多个公开基准测试中系统评估性能，为优化低资源多语言语音LLMs的研究方向提供了新见解。</td></tr>
<tr><td>2025-08-06</td><td>Pseudo Depth Meets Gaussian: A Feed-forward RGB SLAM Baseline</td><td>[2508.04597](http://arxiv.org/pdf/2508.04597)</td><td>◆ 提出了一种基于3D高斯映射的RGB SLAM方法，通过结合深度估计器和3D高斯重建技术，解决了传统方法在长序列处理中的几何细节不准确问题。  
◆ 引入前馈循环预测模块，直接从光流推断相机位姿，替代了耗时的测试时优化，使跟踪速度提升90%以上。  
◆ 设计了局部图渲染技术，增强了前馈位姿预测的鲁棒性，提高了系统在复杂场景中的稳定性。  
◆ 在Replica和TUM-RGBD数据集上的实验表明，该方法性能与当前最先进的SplaTAM相当，同时大幅降低了计算开销。  
◆ 通过实际部署验证了方法的实用性，展示了其在实时3D重建中的高效性和可靠性。</td></tr>
<tr><td>2025-08-05</td><td>Inland-LOAM: Voxel-Based Structural Semantic Mapping for Inland Waterways</td><td>[2508.03672](http://arxiv.org/pdf/2508.03672)</td><td>◆ 提出Inland-LOAM框架，针对内河航道环境优化LiDAR SLAM，解决传统方法在垂直漂移和语义缺失上的局限性。  
◆ 改进特征提取方法并引入水面平面约束，有效抑制SLAM系统的垂直漂移问题。  
◆ 创新性采用体素化几何分析流程，将3D点云实时转换为结构化2D语义地图，支持桥梁净空等导航参数计算。  
◆ 开发自动化模块提取岸线轮廓，并输出轻量化、兼容内河电子航道图(IENC)的标准格式。  
◆ 实测验证表明，该系统定位精度优于现有先进方法，生成的语义地图与岸线数据符合真实场景需求。  
◆ 公开代码与数据集，为内河自主航行提供可靠的环境感知解决方案。</td></tr>
<tr><td>2025-08-04</td><td>A Moment Matching-Based Method for Sparse and Noisy Point Cloud Registration</td><td>[2508.02187](http://arxiv.org/pdf/2508.02187)</td><td>◆ 提出基于矩匹配的点云配准框架，将点云视为同分布独立采样点，通过匹配广义高斯径向基矩估计刚体变换，避免传统方法在稀疏噪声场景下的性能下降。  
◆ 无需显式建立点云间的点对点对应关系，克服了ICP等依赖对应点匹配的局限性，提升了算法鲁棒性。  
◆ 理论证明了方法的收敛一致性，为算法可靠性提供数学保障。  
◆ 在合成与真实数据集实验中，精度和鲁棒性显著优于ICP、NDT等传统方法，尤其在稀疏高噪声条件下优势明显。  
◆ 成功将框架集成至4D雷达SLAM系统，定位性能达到与激光雷达系统相当水平，拓展了毫米波雷达在复杂环境中的应用潜力。</td></tr>
<tr><td>2025-08-04</td><td>AID4AD: Aerial Image Data for Automated Driving Perception</td><td>[2508.02140](http://arxiv.org/pdf/2508.02140)</td><td>◆ 提出AID4AD数据集，首次将高分辨率航拍图像与nuScenes自动驾驶数据集的空间坐标系精确对齐，填补了航拍数据在车辆感知任务中的空白。  
◆ 开发基于SLAM点云地图的自动对齐流程，通过定位和投影畸变校正技术确保空间保真度，并人工筛选高质量对齐样本作为基准真值。  
◆ 验证航拍图像在自动驾驶两大任务中的实用价值：在线地图构建精度提升15-23%，运动预测性能提高2%，证明其可替代高精地图。  
◆ 提出航拍图像作为可扩展的环境上下文来源，特别适用于高精地图缺失、过时或维护成本高的场景，增强系统适应性。  
◆ 公开数据集、评估代码与预训练模型，推动航拍数据在自动驾驶感知领域的后续研究。</td></tr>
<tr><td>2025-07-31</td><td>The Monado SLAM Dataset for Egocentric Visual-Inertial Tracking</td><td>[2508.00088](http://arxiv.org/pdf/2508.00088)</td><td>◆ 提出了Monado SLAM数据集，专门针对头戴式设备的视觉-惯性跟踪挑战，填补了现有数据集的空白。  
◆ 数据集包含真实场景下的多序列数据，覆盖了高动态运动、动态遮挡、长时间跟踪等复杂场景。  
◆ 解决了现有数据集在低纹理区域、恶劣光照条件和传感器饱和等问题上的不足。  
◆ 数据来源于多款虚拟现实头显设备，更贴近实际应用场景。  
◆ 采用CC BY 4.0许可协议公开数据集，推动VIO/SLAM领域的研究与发展。  
◆ 通过真实数据验证了当前先进跟踪系统在实际应用中的局限性，为改进算法提供了重要参考。</td></tr>
<tr><td>2025-07-31</td><td>Stereo 3D Gaussian Splatting SLAM for Outdoor Urban Scenes</td><td>[2507.23677](http://arxiv.org/pdf/2507.23677)</td><td>◆ 提出了首个面向户外场景的双目3D高斯泼溅SLAM系统（BGS-SLAM），填补了现有3DGS-SLAM技术主要针对室内环境且依赖主动深度传感器的空白。  
◆ 仅需RGB立体图像对即可运行，无需LiDAR或主动传感器，降低了硬件成本和复杂度。  
◆ 利用预训练深度立体网络的深度估计指导3D高斯优化，结合多损失策略提升几何一致性与视觉质量。  
◆ 针对复杂户外环境设计了高效的系统架构，在跟踪精度和建图性能上超越现有基于3DGS的解决方案。  
◆ 通过多数据集实验验证了系统在大型户外场景中的鲁棒性和优越性，为户外SLAM提供了新范式。</td></tr>
<tr><td>2025-07-31</td><td>DRACo-SLAM2: Distributed Robust Acoustic Communication-efficient SLAM for Imaging Sonar EquippedUnderwater Robot Teams with Object Graph Matching</td><td>[2507.23629](http://arxiv.org/pdf/2507.23629)</td><td>◆ 提出DRACo-SLAM2框架，改进原有系统，专为配备多波束成像声纳的水下机器人团队设计，实现分布式SLAM。  
◆ 创新性地将声纳地图表示为物体图，通过物体图匹配实现高效跨机器人回环检测，无需依赖先验几何信息。  
◆ 针对水下扫描匹配特点，提出增量式群组一致测量集最大化（GCM）方法，改进原有PCM算法，有效处理相邻跨机器人回环共享相似配准误差的场景。  
◆ 通过物体图匹配显著提升跨机器人回环检测的时间效率，解决了水下环境中传统方法的计算瓶颈。  
◆ 在仿真和真实数据集上进行广泛对比实验，验证了所提方法的优越性和实用性。</td></tr>
<tr><td>2025-07-31</td><td>GSFusion:Globally Optimized LiDAR-Inertial-Visual Mapping for Gaussian Splatting</td><td>[2507.23273](http://arxiv.org/pdf/2507.23273)</td><td>◆ 提出GSFusion系统，首次将激光雷达（LiDAR）、惯性测量单元（IMU）和视觉传感器融合到3D高斯泼溅（3DGS）框架中，实现高精度实时建图。  
◆ 引入全局位姿图优化中的面元到面元（surfel-to-surfel）约束，显著提升地图的全局一致性和对齐精度，解决传统3DGS因累积误差导致的失真问题。  
◆ 设计像素感知的高斯初始化策略，有效利用稀疏LiDAR数据快速生成高质量高斯表示，大幅降低优化时间。  
◆ 提出有界Sigmoid约束机制，防止高斯分布因数据稀疏性导致的过度扩散，提升重建稳定性和渲染质量。  
◆ 在公开和自建数据集上验证，系统在渲染质量和建图效率上均优于现有3DGS SLAM方案，尤其在弱纹理或光照不足环境中表现突出。</td></tr>
<tr><td>2025-07-30</td><td>Modality-Aware Feature Matching: A Comprehensive Review of Single- and Cross-Modality Techniques</td><td>[2507.22791](http://arxiv.org/pdf/2507.22791)</td><td>◆ 全面综述了单模态与跨模态特征匹配技术，涵盖RGB图像、深度图像、3D点云、LiDAR扫描、医学图像及视觉-语言交互等多种模态，填补了该领域系统性总结的空白。  
◆ 对比传统手工方法（如Harris角点、SIFT和ORB描述子）与深度学习方法，指出后者在跨模态鲁棒性和适应性上的显著优势，例如CNN-based SuperPoint和Transformer-based LoFTR等无检测器策略。  
◆ 提出“模态感知”创新方向，针对不同模态设计专用特征匹配方案，如深度图像的几何与深度感知描述子、3D点云的稀疏与稠密学习方法、LiDAR的注意力增强神经网络等。  
◆ 重点分析了医学图像匹配的特殊性，引入MIND描述子等专用解决方案，解决复杂组织结构的匹配挑战。  
◆ 强调跨模态应用的突破性进展，如医学图像配准和视觉-语言任务，展示了特征匹配技术在多模态数据交互中的前沿发展。</td></tr>
<tr><td>2025-07-30</td><td>UAVScenes: A Multi-Modal Dataset for UAVs</td><td>[2507.22412](http://arxiv.org/pdf/2507.22412)</td><td>◆ 提出了首个支持多模态（相机图像和LiDAR点云）帧级标注的大规模无人机数据集UAVScenes，填补了现有数据集仅支持定位或地图级语义分割的空白。  
◆ 基于MARS-LVIG数据集进行升级，新增了人工标注的逐帧图像和点云语义标签，以及高精度6自由度位姿数据，显著扩展了数据用途。  
◆ 首次支持无人机场景下的多任务联合评测，包括分割、深度估计、6-DoF定位、地点识别和新视角合成等高级场景理解任务。  
◆ 提供了严格校准的多模态传感器数据（相机+LiDAR），解决了现有无人机数据集中模态对齐不足的问题。  
◆ 通过开源数据集和基准任务，为无人机多模态感知研究提供了标准化评测平台，推动该领域技术发展。  
◆ 特别关注实际无人机应用场景的需求，数据采集涵盖复杂环境，增强了数据集的实用性和泛化能力。</td></tr>
<tr><td>2025-07-29</td><td>Impact of Underwater Image Enhancement on Feature Matching</td><td>[2507.21715](http://arxiv.org/pdf/2507.21715)</td><td>◆ 提出了局部匹配稳定性和最远可匹配帧数两项量化指标，用于评估水下图像增强效果。  
◆ 针对水下环境设计了一种新颖的评估框架，专门分析增强技术对帧匹配性能的影响。  
◆ 通过指标驱动的分析方法，揭示了现有方法的优势、局限性和实际应用评估中的不足。  
◆ 结合实用匹配策略，建立了首个面向水下场景的上下文感知增强方法基准测试体系。  
◆ 首次验证了图像视觉质量提升与SLAM算法性能的直接关联，证明了框架在实际水下作业中的实用价值。  
◆ 系统解决了水下光吸收、散射、生物附着等退化因素对特征匹配任务的干扰问题。</td></tr>
<tr><td>2025-07-29</td><td>Adaptive Prior Scene-Object SLAM for Dynamic Environments</td><td>[2507.21709](http://arxiv.org/pdf/2507.21709)</td><td>◆ 提出基于场景-物体的可靠性评估框架，通过当前帧质量指标和与可靠参考帧的场景变化，全面评估SLAM系统的稳定性。  
◆ 针对现有系统在姿态估计不可靠时缺乏纠错机制的问题，引入姿态优化策略，利用可靠帧信息优化相机位姿估计，有效减少动态干扰的影响。  
◆ 在动态环境中显著提升了定位精度，尤其在视角突变和运动物体特征不明确的挑战性场景中表现突出。  
◆ 通过结合当前帧质量与场景变化分析，增强了系统对动态干扰的鲁棒性，避免了传统SLAM因静态环境假设导致的定位漂移。  
◆ 在TUM RGB-D数据集上的大量实验验证了该方法在动态场景中的优越性，定位准确性和系统鲁棒性均有显著提升。</td></tr>
<tr><td>2025-08-01</td><td>Multi-robot LiDAR SLAM: a practical case study in underground tunnel environments</td><td>[2507.21553](http://arxiv.org/pdf/2507.21553)</td><td>◆ 针对多机器人LiDAR SLAM在隧道环境中的实际应用问题，论文系统分析了现有去中心化系统的技术瓶颈，首次明确指出&quot;回环检测误报率过高&quot;是导致系统失效的核心原因。  
◆ 提出了一种新型启发式算法，通过优化回环检测机制显著降低了误报率，解决了地下隧道等极端场景下SLAM稳定性差的难题。  
◆ 创新性地选择地下隧道作为验证环境，这种长走廊、低特征、高对称性的极端场景为SLAM研究提供了更具挑战性的测试基准。  
◆ 揭示了当前多机器人SLAM系统在动态环境适应性方面的不足，为后续研究指明了改进方向。  
◆ 通过实证研究发现了现有技术框架中未被充分探索的研究空白，特别是多机协同建图时的数据融合优化问题。  
◆ 提出的解决方案不依赖额外硬件，仅通过算法改进即可提升系统鲁棒性，具有工程实用价值。</td></tr>
<tr><td>2025-07-28</td><td>$S^3$LAM: Surfel Splatting SLAM for Geometrically Accurate Tracking and Mapping</td><td>[2507.20854](http://arxiv.org/pdf/2507.20854)</td><td>◆ 提出S³LAM系统，采用2D面元（surfel）作为基本单元，替代传统3D高斯椭球体，实现更高效的场景几何表示。  
◆ 创新性地利用2D高斯面元进行场景表面建模，专注于物体表面几何，显著提升重建质量和精度。  
◆ 设计自适应表面渲染策略，有效解决SLAM中视角受限下的实时优化问题，兼顾计算效率与建图准确性。  
◆ 首次从2D面元渲染公式直接推导相机位姿雅可比矩阵，通过几何精确表示提升跟踪收敛性。  
◆ 在合成与真实数据集上验证了S³LAM的优越性，性能达到当前最优水平。</td></tr>
<tr><td>2025-07-28</td><td>Large-Scale LiDAR-Inertial Dataset for Degradation-Robust High-Precision Mapping</td><td>[2507.20516](http://arxiv.org/pdf/2507.20516)</td><td>◆ 提出首个面向复杂场景的大规模LiDAR-惯性数据集，填补现有研究在真实环境验证不足的空白。  
◆ 覆盖4类差异化场景（6万-75万平方米），包含长轨迹、复杂结构及工业级传感器数据（多线LiDAR+IMU+RTK-GNSS）。  
◆ 创新采用背包式移动平台实现高灵活性数据采集，兼顾大范围与高精度需求。  
◆ 提出SLAM优化与RTK-GNSS锚定融合的基准真值生成方法，并通过倾斜摄影联合验证轨迹精度（厘米级）。  
◆ 为LIO系统在实际测绘中的退化鲁棒性、泛化能力评估提供标准化测试基准。</td></tr>
<tr><td>2025-07-26</td><td>DOA: A Degeneracy Optimization Agent with Adaptive Pose Compensation Capability based on Deep Reinforcement Learning</td><td>[2507.19742](http://arxiv.org/pdf/2507.19742)</td><td>◆ 提出基于PPO深度强化学习的自适应退化优化智能体（DOA），解决传统粒子滤波SLAM在长直走廊等退化环境中的定位失效问题。  
◆ 创新性设计系统化方法解决监督学习的三大瓶颈：退化数据集获取困难、训练样本质量下降以及标注协议模糊性问题。  
◆ 设计专用奖励函数引导智能体感知退化环境，通过退化因子动态调节多传感器在位姿优化中的贡献权重。  
◆ 提出基于退化因子的线性插值公式，将观测分布向运动模型分布自适应偏移，实现位姿补偿步长的动态调整。  
◆ 引入迁移学习模块赋予跨环境泛化能力，解决退化环境训练效率低下问题，并通过消融实验验证其关键作用。  
◆ 实验证明DOA在多种环境中优于现有方法，尤其在退化检测和位姿优化精度方面达到SOTA水平。</td></tr>
<tr><td>2025-07-25</td><td>DINO-SLAM: DINO-informed RGB-D SLAM for Neural Implicit and Explicit Representations</td><td>[2507.19474](http://arxiv.org/pdf/2507.19474)</td><td>◆ 提出DINO-SLAM框架，通过DINO特征增强策略提升神经隐式（NeRF）和显式（3DGS）SLAM系统的场景表示能力。  
◆ 设计场景结构编码器（SSE），将原始DINO特征升级为增强版EDINO，有效捕捉场景层次化元素及其结构关系。  
◆ 首次构建两种基于EDINO特征的SLAM范式，分别适配NeRF和3DGS两种主流三维表示方法。  
◆ 在Replica、ScanNet和TUM数据集上验证了方案优越性，性能超越现有最优方法。  
◆ 通过统一特征增强策略，实现了对复杂场景更全面的几何与语义理解，为SLAM系统提供新思路。</td></tr>
<tr><td>2025-07-25</td><td>The Eloquence team submission for task 1 of MLC-SLM challenge</td><td>[2507.19308](http://arxiv.org/pdf/2507.19308)</td><td>◆ 评估了官方基线模型的多语言ASR性能，通过训练线性投影器和qformer两种投影器，结合不同基础模型，分析了其优势与局限性。  
◆ 利用SLAM-ASR框架训练了自定义的多语言线性投影器，优化了模型在多语言场景下的适应性。  
◆ 探索了对比学习在提升语音识别鲁棒性中的作用，通过改进训练策略增强模型对多样化语音输入的识别能力。  
◆ 研究了扩展对话上下文对识别效果的影响，验证了长上下文信息在多语言对话语音识别中的重要性。  
◆ 综合多种方法（如投影器设计、对比学习和上下文扩展）为多语言对话语音识别任务提供了系统性的解决方案。</td></tr>
<tr><td>2025-07-25</td><td>SmartPNT-MSF: A Multi-Sensor Fusion Dataset for Positioning and Navigation Research</td><td>[2507.19079](http://arxiv.org/pdf/2507.19079)</td><td>◆ 提出了SmartPNT-MSF多传感器融合数据集，弥补了现有数据集在传感器多样性和环境覆盖范围上的不足。  
◆ 整合了GNSS、IMU、光学相机和LiDAR等多种传感器数据，为多传感器融合和高精度导航研究提供了丰富资源。  
◆ 详细记录了数据集构建过程，包括传感器配置、坐标系定义以及相机和LiDAR的校准流程，确保数据的一致性和可扩展性。  
◆ 采用标准化框架进行数据采集和处理，支持大规模分析，并通过VINS-Mono和LIO-SAM等先进SLAM算法验证了数据集的实用性。  
◆ 覆盖城市、校园、隧道和郊区等多种真实场景，为复杂环境下的导航技术研究提供了重要工具。  
◆ 公开高质量数据集，促进了传感器多样性、数据可获取性和环境代表性的研究，推动了相关领域的创新发展。</td></tr>
<tr><td>2025-07-24</td><td>G2S-ICP SLAM: Geometry-aware Gaussian Splatting ICP SLAM</td><td>[2507.18344](http://arxiv.org/pdf/2507.18344)</td><td>◆ 提出了一种基于几何感知的高斯泼溅SLAM系统（G2S-ICP SLAM），通过将场景元素表示为局部切平面约束的高斯分布，实现高保真3D重建和实时相机位姿跟踪。  
◆ 创新性地使用2D高斯圆盘（而非传统3D椭球）表示局部表面，使其与底层几何对齐，显著提升了多视角深度解释的一致性。  
◆ 将表面对齐的高斯圆盘嵌入广义ICP框架，通过引入各向异性协方差先验改进配准，无需修改原有配准公式。  
◆ 提出几何感知损失函数，联合优化光度、深度和法向一致性，增强重建的几何精度与视觉质量。  
◆ 系统在Replica和TUM-RGBD数据集上验证，定位精度、重建完整性和渲染质量均优于现有SLAM方法，且保持实时性能。</td></tr>
<tr><td>2025-07-23</td><td>Physics-based Human Pose Estimation from a Single Moving RGB Camera</td><td>[2507.17406](http://arxiv.org/pdf/2507.17406)</td><td>◆ 提出了首个非合成的真实数据集MoviCam，包含动态移动的单目RGB相机轨迹、场景几何和3D人体运动数据，并标注了人-场景接触信息，填补了现有合成数据无法模拟真实光照、相机运动和姿态变化的缺陷。  
◆ 开发了PhysDynPose方法，首次在物理约束的人体姿态估计中整合场景几何信息，有效解决了相机移动和非平面场景下的跟踪难题。  
◆ 结合先进运动学估计器与鲁棒SLAM技术，实现了世界坐标系下人体姿态与相机轨迹的同步恢复，突破了传统方法依赖静态相机的限制。  
◆ 设计了场景感知的物理优化器，通过物理约束对运动学估计结果进行精细化修正，显著提升了复杂环境下的跟踪精度。  
◆ 通过新基准测试验证了现有方法在移动相机和非平面场景中的性能局限，而所提方法在此挑战性设定下仍能稳定输出人体与相机位姿。</td></tr>
<tr><td>2025-07-23</td><td>CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance</td><td>[2507.17312](http://arxiv.org/pdf/2507.17312)</td><td>◆ 提出CasP新流程，通过级联对应先验引导半稠密特征匹配，改进传统全局搜索的低效问题。  
◆ 设计两阶段渐进匹配策略，首阶段生成一对多先验区域，次阶段将搜索范围限制在该区域以提升精度。  
◆ 创新引入基于区域的选择性交叉注意力机制，增强特征判别力并连接两阶段匹配。  
◆ 融合高层特征降低低层特征计算开销，分辨率越高加速效果越显著（1152分辨率下比ELoFTR快2.2倍）。  
◆ 在跨域泛化能力上表现突出，几何估计精度优于现有方法，适用于SLAM、无人机等高实时性场景。  
◆ 开源代码并提供轻量化模型，兼顾效率与鲁棒性优势。</td></tr>
<tr><td>2025-07-21</td><td>A Comprehensive Evaluation of LiDAR Odometry Techniques</td><td>[2507.16000](http://arxiv.org/pdf/2507.16000)</td><td>◆ 首次系统梳理了LiDAR里程计（LO）技术中的各类模块化组件，填补了该领域缺乏全面组件对比研究的空白。  
◆ 通过大量消融实验，在多样化数据集（不同环境、LiDAR型号和运动模式）上实证评估了各组件性能。  
◆ 提出基于实验数据的LO流程设计准则，为未来高精度系统开发提供可复现的优化方向。  
◆ 突破了以往仅对比完整技术流程的局限，首次从底层组件层面揭示性能差异的关键因素。  
◆ 建立跨场景评估框架，验证了不同技术组合的泛化能力与鲁棒性。</td></tr>
<tr><td>2025-07-21</td><td>Dense-depth map guided deep Lidar-Visual Odometry with Sparse Point Clouds and Images</td><td>[2507.15496](http://arxiv.org/pdf/2507.15496)</td><td>◆ 提出了一种新颖的LiDAR-视觉里程计框架，通过深度融合稀疏LiDAR点云和图像数据实现高精度位姿估计。  
◆ 创新性地利用深度补全技术生成稠密深度图，为运动估计提供更丰富的几何约束信息。  
◆ 设计了带注意力机制的多尺度特征提取网络，能够自适应生成深度感知的特征表示。  
◆ 利用稠密深度信息优化光流估计，有效减少了遮挡区域的误差累积问题。  
◆ 开发了分层位姿优化模块，通过渐进式运动估计提升动态环境下的算法鲁棒性。  
实验证明该方法在KITTI数据集上达到了与当前最优视觉/LiDAR里程计相当或更优的精度和鲁棒性。</td></tr>
<tr><td>2025-07-21</td><td>All-UWB SLAM Using UWB Radar and UWB AOA</td><td>[2507.15474](http://arxiv.org/pdf/2507.15474)</td><td>◆ 提出了一种结合UWB雷达和UWB到达角（AOA）测量的新型SLAM方法，用于视觉受限且特征稀缺的环境。  
◆ 通过动态部署UWB锚点-标签单元，在环境特征不足的区域补充AOA测量数据，提升了SLAM的精度和可扩展性。  
◆ 解决了现有UWB雷达SLAM依赖环境特征数量的局限性，能够在特征稀少的环境中实现定位与建图。  
◆ 详细分析了UWB AOA测量单元的常见约束问题，并提出了相应的解决方案。  
◆ 实验证明，该方法在视觉受限且特征稀缺的环境中仍能有效运行，拓展了SLAM的应用场景。</td></tr>
<tr><td>2025-07-21</td><td>BenchDepth: Are We on the Right Way to Evaluate Depth Foundation Models?</td><td>[2507.15321](http://arxiv.org/pdf/2507.15321)</td><td>◆ 提出了BenchDepth新基准，通过五个下游代理任务（深度补全、立体匹配、单目前馈3D场景重建、SLAM和视觉-语言空间理解）评估深度基础模型（DFMs），突破传统评估局限。  
◆ 摒弃传统依赖对齐指标的评估方法，解决了现有协议中因对齐偏差、深度表示偏好导致的不公平比较问题。  
◆ 首次从实际应用效用角度评估DFMs，强调模型在真实场景中的实用性，而非仅关注理论性能指标。  
◆ 系统地对八种前沿DFMs进行横向对比，提供关键发现与深度分析，揭示当前模型的优势与不足。  
◆ 推动深度估计领域评估标准的革新，为未来研究建立更科学、更贴近实际需求的评测框架。  
◆ 通过多任务评估框架，促进社区对深度模型评估最佳实践的讨论，为技术发展指明方向。</td></tr>
<tr><td>2025-07-20</td><td>LoopNet: A Multitasking Few-Shot Learning Approach for Loop Closure in Large Scale SLAM</td><td>[2507.15109](http://arxiv.org/pdf/2507.15109)</td><td>◆ 提出LoopNet，一种基于多任务学习改进的ResNet架构，专为大尺度SLAM中的闭环检测问题设计，兼顾嵌入式设备的实时计算限制。  
◆ 采用小样本学习策略实现动态视觉数据集的在线重训练，适应环境变化并提升模型泛化能力。  
◆ 创新性地同时输出视觉数据集的检索索引和预测质量评估，增强闭环检测的可靠性。  
◆ 结合DISK关键点描述符，突破传统手工特征和深度学习方法的局限，在多变条件下表现更优。  
◆ 发布开源数据集LoopDB，为闭环检测研究提供新的基准测试平台。  
◆ 整体方案在精度和实时性上均优于现有方法，代码与数据集均已开源以促进领域发展。</td></tr>
<tr><td>2025-07-19</td><td>Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey</td><td>[2507.14501](http://arxiv.org/pdf/2507.14501)</td><td>◆ 全面综述了基于前馈方法的3D重建与视图合成技术，系统梳理了深度学习驱动的快速通用化解决方案，突破了传统迭代优化方法的计算瓶颈。  
◆ 提出按表示架构分类的新体系（如点云、3D高斯泼溅、神经辐射场等），为领域研究建立清晰的技术路线图。  
◆ 首次系统分析了无位姿重建、动态3D重建、3D感知图像/视频合成等关键任务，揭示了在数字人、SLAM等场景的应用潜力。  
◆ 整合了主流数据集与评估协议，为不同下游任务提供标准化评测基准。  
◆ 前瞻性指出前馈方法在实时性、泛化能力等方面的挑战，为未来研究指明方向，推动3D视觉技术向实用化发展。</td></tr>
</tbody>
</table>
</div>

<h2 id='visual-slam'>Visual SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-10-23</td><td>Deep Learning-Powered Visual SLAM Aimed at Assisting Visually Impaired Navigation</td><td>[2510.20549](http://arxiv.org/pdf/2510.20549)</td><td>◆ Despite advancements in SLAM technologies, robust operation under challenging conditions such as low-texture, motion-blur, or challenging lighting remains an open challenge.
◆ Such conditions are common in applications such as assistive navigation for the visually impaired.
◆ These challenges undermine localization accuracy and tracking stability, reducing navigation reliability and safety.</td></tr>
<tr><td>2025-10-21</td><td>DeepDetect: Learning All-in-One Dense Keypoints</td><td>[2510.17422](http://arxiv.org/pdf/2510.17422)</td><td>◆ Keypoint detection is the foundation of many computer vision tasks, including image registration, structure-from motion, 3D reconstruction, visual odometry, and SLAM.
◆ Traditional detectors (SIFT, SURF, ORB, BRISK, etc.) and learning based methods (SuperPoint, R2D2, LF-Net, D2-Net, etc.) have shown strong performance yet suffer from key limitations: sensitivity to photometric changes, low keypoint density and repeatability, limited adaptability to challenging scenes, and lack of semantic understanding, often failing to prioritize visually important regions.
◆ We present DeepDetect, an intelligent, all-in-one, dense keypoint detector that unifies the strengths of classical detectors using deep learning.</td></tr>
<tr><td>2025-10-17</td><td>VAR-SLAM: Visual Adaptive and Robust SLAM for Dynamic Environments</td><td>[2510.16205](http://arxiv.org/pdf/2510.16205)</td><td>◆ Visual SLAM in dynamic environments remains challenging, as several existing methods rely on semantic filtering that only handles known object classes, or use fixed robust kernels that cannot adapt to unknown moving objects, leading to degraded accuracy when they appear in the scene.
◆ We present VAR-SLAM (Visual Adaptive and Robust SLAM), an ORB-SLAM3-based system that combines a lightweight semantic keypoint filter to deal with known moving objects, with Barron&#x27;s adaptive robust loss to handle unknown ones.
◆ The shape parameter of the robust kernel is estimated online from residuals, allowing the system to automatically adjust between Gaussian and heavy-tailed behavior.</td></tr>
<tr><td>2025-10-15</td><td>Accelerated Feature Detectors for Visual SLAM: A Comparative Study of FPGA vs GPU</td><td>[2510.13546](http://arxiv.org/pdf/2510.13546)</td><td>◆ Feature detection is a common yet time-consuming module in Simultaneous Localization and Mapping (SLAM) implementations, which are increasingly deployed on power-constrained platforms, such as drones.
◆ Graphics Processing Units (GPUs) have been a popular accelerator for computer vision in general, and feature detection and SLAM in particular.
◆ On the other hand, System-on-Chips (SoCs) with integrated Field Programmable Gate Array (FPGA) are also widely available.</td></tr>
<tr><td>2025-10-14</td><td>SPORTS: Simultaneous Panoptic Odometry, Rendering, Tracking and Segmentation for Urban Scenes Understanding</td><td>[2510.12749](http://arxiv.org/pdf/2510.12749)</td><td>◆ The scene perception, understanding, and simulation are fundamental techniques for embodied-AI agents, while existing solutions are still prone to segmentation deficiency, dynamic objects&#x27; interference, sensor data sparsity, and view-limitation problems.
◆ This paper proposes a novel framework, named SPORTS, for holistic scene understanding via tightly integrating Video Panoptic Segmentation (VPS), Visual Odometry (VO), and Scene Rendering (SR) tasks into an iterative and unified perspective.
◆ Firstly, VPS designs an adaptive attention-based geometric fusion mechanism to align cross-frame features via enrolling the pose, depth, and optical flow modality, which automatically adjust feature maps for different decoding stages.</td></tr>
<tr><td>2025-10-14</td><td>PolygMap: A Perceptive Locomotion Framework for Humanoid Robot Stair Climbing</td><td>[2510.12346](http://arxiv.org/pdf/2510.12346)</td><td>◆ Recently, biped robot walking technology has been significantly developed, mainly in the context of a bland walking scheme.
◆ To emulate human walking, robots need to step on the positions they see in unknown spaces accurately.
◆ In this paper, we present PolyMap, a perception-based locomotion planning framework for humanoid robots to climb stairs.</td></tr>
<tr><td>2025-10-06</td><td>OKVIS2-X: Open Keyframe-based Visual-Inertial SLAM Configurable with Dense Depth or LiDAR, and GNSS</td><td>[2510.04612](http://arxiv.org/pdf/2510.04612)</td><td>◆ To empower mobile robots with usable maps as well as highest state estimation accuracy and robustness, we present OKVIS2-X: a state-of-the-art multi-sensor Simultaneous Localization and Mapping (SLAM) system building dense volumetric occupancy maps, while scalable to large environments and operating in realtime.
◆ Our unified SLAM framework seamlessly integrates different sensor modalities: visual, inertial, measured or learned depth, LiDAR and Global Navigation Satellite System (GNSS) measurements.
◆ Unlike most state-of-the-art SLAM systems, we advocate using dense volumetric map representations when leveraging depth or range-sensing capabilities.</td></tr>
<tr><td>2025-10-02</td><td>Visual Odometry with Transformers</td><td>[2510.03348](http://arxiv.org/pdf/2510.03348)</td><td>◆ Modern monocular visual odometry methods typically combine pre-trained deep learning components with optimization modules, resulting in complex pipelines that rely heavily on camera calibration and hyperparameter tuning, and often struggle in unseen real-world scenarios.
◆ Recent large-scale 3D models trained on massive amounts of multi-modal data have partially alleviated these challenges, providing generalizable dense reconstruction and camera pose estimation.
◆ Still, they remain limited in handling long videos and providing accurate per-frame estimates, which are required for visual odometry.</td></tr>
<tr><td>2025-10-02</td><td>RSV-SLAM: Toward Real-Time Semantic Visual SLAM in Indoor Dynamic Environments</td><td>[2510.02616](http://arxiv.org/pdf/2510.02616)</td><td>◆ Simultaneous Localization and Mapping (SLAM) plays an important role in many robotics fields, including social robots.
◆ Many of the available visual SLAM methods are based on the assumption of a static world and struggle in dynamic environments.
◆ In the current study, we introduce a real-time semantic RGBD SLAM approach designed specifically for dynamic environments.</td></tr>
<tr><td>2025-10-01</td><td>Instant4D: 4D Gaussian Splatting in Minutes</td><td>[2510.01119](http://arxiv.org/pdf/2510.01119)</td><td>◆ Dynamic view synthesis has seen significant advances, yet reconstructing scenes from uncalibrated, casual video remains challenging due to slow optimization and complex parameter estimation.
◆ In this work, we present Instant4D, a monocular reconstruction system that leverages native 4D representation to efficiently process casual video sequences within minutes, without calibrated cameras or depth sensors.
◆ Our method begins with geometric recovery through deep visual SLAM, followed by grid pruning to optimize scene representation.</td></tr>
<tr><td>2025-10-01</td><td>Semantic Visual Simultaneous Localization and Mapping: A Survey on State of the Art, Challenges, and Future Directions</td><td>[2510.00783](http://arxiv.org/pdf/2510.00783)</td><td>◆ Semantic Simultaneous Localization and Mapping (SLAM) is a critical area of research within robotics and computer vision, focusing on the simultaneous localization of robotic systems and associating semantic information to construct the most accurate and complete comprehensive model of the surrounding environment.
◆ Since the first foundational work in Semantic SLAM appeared more than two decades ago, this field has received increasing attention across various scientific communities.
◆ Despite its significance, the field lacks comprehensive surveys encompassing recent advances and persistent challenges.</td></tr>
<tr><td>2025-09-28</td><td>GRS-SLAM3R: Real-Time Dense SLAM with Gated Recurrent State</td><td>[2509.23737](http://arxiv.org/pdf/2509.23737)</td><td>◆ DUSt3R-based end-to-end scene reconstruction has recently shown promising results in dense visual SLAM.
◆ However, most existing methods only use image pairs to estimate pointmaps, overlooking spatial memory and global consistency.To this end, we introduce GRS-SLAM3R, an end-to-end SLAM framework for dense scene reconstruction and pose estimation from RGB images without any prior knowledge of the scene or camera parameters.
◆ Unlike existing DUSt3R-based frameworks, which operate on all image pairs and predict per-pair point maps in local coordinate frames, our method supports sequentialized input and incrementally estimates metric-scale point clouds in the global coordinate.</td></tr>
<tr><td>2025-09-26</td><td>Good Weights: Proactive, Adaptive Dead Reckoning Fusion for Continuous and Robust Visual SLAM</td><td>[2509.22910](http://arxiv.org/pdf/2509.22910)</td><td>◆ Given that Visual SLAM relies on appearance cues for localization and scene understanding, texture-less or visually degraded environments (e.g., plain walls or low lighting) lead to poor pose estimation and track loss.
◆ However, robots are typically equipped with sensors that provide some form of dead reckoning odometry with reasonable short-time performance but unreliable long-time performance.
◆ The Good Weights (GW) algorithm described here provides a framework to adaptively integrate dead reckoning (DR) with passive visual SLAM for continuous and accurate frame-level pose estimation.</td></tr>
<tr><td>2025-09-29</td><td>MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM</td><td>[2509.20757](http://arxiv.org/pdf/2509.20757)</td><td>◆ Visual SLAM is a cornerstone technique in robotics, autonomous driving and extended reality (XR), yet classical systems often struggle with low-texture environments, scale ambiguity, and degraded performance under challenging visual conditions.
◆ Recent advancements in feed-forward neural network-based pointmap regression have demonstrated the potential to recover high-fidelity 3D scene geometry directly from images, leveraging learned spatial priors to overcome limitations of traditional multi-view geometry methods.
◆ However, the widely validated advantages of probabilistic multi-sensor information fusion are often discarded in these pipelines.</td></tr>
<tr><td>2025-09-24</td><td>Optical Ocean Recipes: Creating Realistic Datasets to Facilitate Underwater Vision Research</td><td>[2509.20171](http://arxiv.org/pdf/2509.20171)</td><td>◆ The development and evaluation of machine vision in underwater environments remains challenging, often relying on trial-and-error-based testing tailored to specific applications.
◆ This is partly due to the lack of controlled, ground-truthed testing environments that account for the optical challenges, such as color distortion from spectrally variant light attenuation, reduced contrast and blur from backscatter and volume scattering, and dynamic light patterns from natural or artificial illumination.
◆ Additionally, the appearance of ocean water in images varies significantly across regions, depths, and seasons.</td></tr>
<tr><td>2025-09-21</td><td>ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM</td><td>[2509.16863](http://arxiv.org/pdf/2509.16863)</td><td>◆ We introduce ConfidentSplat, a novel 3D Gaussian Splatting (3DGS)-based SLAM system for robust, highfidelity RGB-only reconstruction.
◆ Addressing geometric inaccuracies in existing RGB-only 3DGS SLAM methods that stem from unreliable depth estimation, ConfidentSplat incorporates a core innovation: a confidence-weighted fusion mechanism.
◆ This mechanism adaptively integrates depth cues from multiview geometry with learned monocular priors (Omnidata ViT), dynamically weighting their contributions based on explicit reliability estimates-derived predominantly from multi-view geometric consistency-to generate high-fidelity proxy depth for map supervision.</td></tr>
<tr><td>2025-09-19</td><td>Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion</td><td>[2509.15673](http://arxiv.org/pdf/2509.15673)</td><td>◆ Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but most existing LiDAR-inertial-visual odometry (LIVO) systems rely on a single camera, leading to limited spatial coverage and degraded robustness.
◆ We present Omni-LIVO, the first tightly coupled multi-camera LIVO system that bridges the FoV mismatch between wide-angle LiDAR and conventional cameras.
◆ Omni-LIVO introduces a Cross-View direct tracking strategy that maintains photometric consistency across non-overlapping views, and extends the Error-State Iterated Kalman Filter (ESIKF) with multi-view updates and adaptive covariance weighting.</td></tr>
<tr><td>2025-09-18</td><td>BEV-ODOM2: Enhanced BEV-based Monocular Visual Odometry with PV-BEV Fusion and Dense Flow Supervision for Ground Robots</td><td>[2509.14636](http://arxiv.org/pdf/2509.14636)</td><td>◆ Bird&#x27;s-Eye-View (BEV) representation offers a metric-scaled planar workspace, facilitating the simplification of 6-DoF ego-motion to a more robust 3-DoF model for monocular visual odometry (MVO) in intelligent transportation systems.
◆ However, existing BEV methods suffer from sparse supervision signals and information loss during perspective-to-BEV projection.
◆ We present BEV-ODOM2, an enhanced framework addressing both limitations without additional annotations.</td></tr>
<tr><td>2025-09-17</td><td>BIM Informed Visual SLAM for Construction Monitoring</td><td>[2509.13972](http://arxiv.org/pdf/2509.13972)</td><td>◆ Simultaneous Localization and Mapping (SLAM) is a key tool for monitoring construction sites, where aligning the evolving as-built state with the as-planned design enables early error detection and reduces costly rework.
◆ LiDAR-based SLAM achieves high geometric precision, but its sensors are typically large and power-demanding, limiting their use on portable platforms.
◆ Visual SLAM offers a practical alternative with lightweight cameras already embedded in most mobile devices.</td></tr>
<tr><td>2025-09-17</td><td>UM-Depth : Uncertainty Masked Self-Supervised Monocular Depth Estimation with Visual Odometry</td><td>[2509.13713](http://arxiv.org/pdf/2509.13713)</td><td>◆ Monocular depth estimation has been increasingly adopted in robotics and autonomous driving for its ability to infer scene geometry from a single camera.
◆ In self-supervised monocular depth estimation frameworks, the network jointly generates and exploits depth and pose estimates during training, thereby eliminating the need for depth labels.
◆ However, these methods remain challenged by uncertainty in the input data, such as low-texture or dynamic regions, which can cause reduced depth accuracy.</td></tr>
<tr><td>2025-09-17</td><td>Barometer-Aided Attitude Estimation</td><td>[2509.13649](http://arxiv.org/pdf/2509.13649)</td><td>◆ Accurate and robust attitude estimation is a central challenge for autonomous vehicles operating in GNSS-denied or highly dynamic environments.
◆ In such cases, Inertial Measurement Units (IMUs) alone are insufficient for reliable tilt estimation due to the ambiguity between gravitational and inertial accelerations.
◆ While auxiliary velocity sensors, such as GNSS, Pitot tubes, Doppler radar, or visual odometry, are often used, they can be unavailable, intermittent, or costly.</td></tr>
<tr><td>2025-09-13</td><td>FastTrack: GPU-Accelerated Tracking for Visual SLAM</td><td>[2509.10757](http://arxiv.org/pdf/2509.10757)</td><td>该论文的核心贡献是提出了一种利用GPU加速来显著提升视觉-惯性SLAM系统跟踪模块性能的新方法FastTrack。  
◆ 创新性地利用GPU并行计算能力来加速跟踪过程中最耗时的两个组件：立体特征匹配和局部地图跟踪。  
◆ 将所提出的加速设计集成到了主流的ORB-SLAM3框架的跟踪流程中，并使用CUDA进行实现。  
◆ 在桌面平台和Jetson Xavier NX嵌入式平台上均实现了性能提升，验证了其有效性与实用性。  
◆ 在EuRoC和TUM-VI等标准数据集上的实验表明，在立体-惯性模式下，跟踪性能整体提升了最高2.8倍。  
这项工作通过硬件加速有效解决了SLAM跟踪的实时性瓶颈，避免了因处理延迟导致的定位漂移或跟踪丢失问题。</td></tr>
<tr><td>2025-09-11</td><td>SMapper: A Multi-Modal Data Acquisition Platform for SLAM Benchmarking</td><td>[2509.09509](http://arxiv.org/pdf/2509.09509)</td><td>SMapper的核心贡献是提出了一种专为SLAM研究设计的开源多模态数据采集平台，以解决现有数据集在传感器多样性、环境覆盖和实验可复现性方面的不足。其创新点包括：
◆ 采用开源硬件设计，集成了同步的LiDAR、多相机和惯性传感系统，支持手持和机器人搭载多种场景。
◆ 提供了强大的标定与同步流程，确保了多模态数据在时空上的精确对齐。
◆ 发布了名为SMapper-light的公开数据集，包含室内外典型场景的多模态同步数据和高精度真值轨迹。
◆ 基于该数据集对主流LiDAR与视觉SLAM算法进行了性能评测，为算法开发与评估提供可靠基准。
◆ 通过开放且可复现的设计，显著提升了SLAM研究的可比性和可重复性。</td></tr>
<tr><td>2025-09-10</td><td>Good Deep Features to Track: Self-Supervised Feature Extraction and Tracking in Visual Odometry</td><td>[2509.08333](http://arxiv.org/pdf/2509.08333)</td><td>该论文针对视觉里程计中因光照变化、动态场景等导致特征提取与跟踪性能下降的问题，提出了一种自监督的深度特征提取与跟踪方法。  
◆ 通过自监督学习结合任务特定反馈，增强深度特征的稳定性和信息量。  
◆ 提升了在挑战性环境（如大尺度户外场景和长期运行）中的泛化能力和可靠性。  
◆ 克服了现有学习方法（如SuperPoint和SuperGlue）在分布外数据上的泛化局限。  
该方法无需额外标注数据，通过优化特征提取与跟踪过程，显著提高了运动估计的准确性。</td></tr>
<tr><td>2025-09-10</td><td>Deep Visual Odometry for Stereo Event Cameras</td><td>[2509.08235](http://arxiv.org/pdf/2509.08235)</td><td>该论文提出了一种基于深度学习的立体事件相机视觉里程计系统Stereo-DEVO，其核心贡献在于实现了在复杂光照条件下高精度、实时的位姿估计。  
◆ 提出了一种新颖且高效的静态-立体关联策略，以极低计算成本实现稀疏深度估计。  
◆ 将深度估计与紧耦合的束调整优化方案相结合，提升了系统精度和鲁棒性。  
◆ 利用基于体素的事件表示和循环神经网络，实现了精确的光流估计和可靠的图像块关联。  
◆ 系统能够实时处理VGA分辨率的事件流，相比此前工作实现了从离线的实时性突破。  
实验表明，该系统在多个真实场景数据集上表现优异，尤其在大规模夜间高动态范围环境下也能保持稳定的位姿估计性能。</td></tr>
<tr><td>2025-09-09</td><td>Aerial-ground Cross-modal Localization: Dataset, Ground-truth, and Benchmark</td><td>[2509.07362](http://arxiv.org/pdf/2509.07362)</td><td>该论文的核心贡献是构建了一个解决空地跨模态定位挑战的综合性基准。其创新点主要包括：
◆ 创建了一个全新的大规模空地跨模态数据集，整合了来自移动测量系统的地面图像和覆盖武汉、香港及旧金山的机载激光扫描点云数据。
◆ 解决了平台多样性不足的问题，提供了多城市、多场景的丰富数据来源。
◆ 提出了一种适用于大规模城市环境的可靠地面真值生成方法，填补了该领域的技术空白。
◆ 首次系统性地验证了现有图像到点云（I2P）匹配算法在空地跨平台场景下的性能。
◆ 为基于机载激光扫描先验地图的可扩展精准视觉定位研究提供了完整的评估基准和基础框架。</td></tr>
<tr><td>2025-09-04</td><td>Odometry Calibration and Pose Estimation of a 4WIS4WID Mobile Wall Climbing Robot</td><td>[2509.04016](http://arxiv.org/pdf/2509.04016)</td><td>本文针对4WIS4WID爬壁机器人在复杂建筑立面上定位困难的问题，提出了一种基于多传感器融合的位姿估计解决方案。
◆ 设计了一种融合轮式里程计、视觉里程计和IMU数据的位姿估计器，采用EKF和UKF滤波器进行多模态信息融合。
◆ 针对机器人系统误差，综合应用了非线性优化和Levenberg-Marquardt等确定性方法进行标定。
◆ 同时采用了遗传算法和粒子群算法等随机优化方法进行运动学参数校准，提升了标定精度和鲁棒性。
◆ 整套标定与位姿估计系统在实验爬壁机器人平台上得到了详细实验验证，证明了其有效性和实用性。
该研究为无GPS环境下爬壁机器人的高精度定位提供了可靠的技术途径。</td></tr>
<tr><td>2025-09-01</td><td>ViSTA-SLAM: Visual SLAM with Symmetric Two-view Association</td><td>[2509.01584](http://arxiv.org/pdf/2509.01584)</td><td>ViSTA-SLAM的核心贡献是提出了一种无需已知相机内参即可实时运行的单目视觉SLAM系统，其性能在相机跟踪和稠密三维重建质量上均优于现有方法。

◆ 采用轻量级对称双视图关联（STA）模型作为前端，仅需两张RGB图像即可同时估计相对相机位姿并回归局部点云图。
◆ 显著降低了模型复杂度，前端模型大小仅为同类先进方法的35%，同时提升了系统所用双视图约束的质量。
◆ 在后端构建了一个特殊的Sim(3)位姿图，通过融入回环检测来有效解决累积的尺度漂移问题。
◆ 整个系统不依赖相机内参，使其能够广泛适用于各种不同的相机设置，具备了更强的通用性。</td></tr>
<tr><td>2025-09-01</td><td>FGO-SLAM: Enhancing Gaussian SLAM with Globally Consistent Opacity Radiance Field</td><td>[2509.01547](http://arxiv.org/pdf/2509.01547)</td><td>FGO-SLAM的核心贡献是提出了一种基于全局一致不透明度辐射场的新型高斯SLAM系统，显著提升了场景几何重建的质量和系统的跟踪精度。

◆ 采用不透明度辐射场作为场景表示方法，有效增强了系统的几何建图性能。
◆ 在初始位姿估计后，引入全局调整优化策略，同时优化相机位姿和稀疏点云，确保了鲁棒的跟踪效果。
◆ 维护了一个基于3D高斯且全局一致的不透明度辐射场，并新增了深度失真和法向一致性约束项来精细化场景表示。
◆ 通过构建四面体网格并提取等值面，实现了直接从3D高斯模型中提取物体表面，简化了表面重建流程。
实验结果表明，该方法在多个真实和大型合成数据集上均实现了最先进的跟踪与建图性能。</td></tr>
<tr><td>2025-08-31</td><td>DyPho-SLAM : Real-time Photorealistic SLAM in Dynamic Environments</td><td>[2509.00741](http://arxiv.org/pdf/2509.00741)</td><td>DyPho-SLAM是一种能够在动态环境中实时运行的视觉SLAM系统，其核心贡献是解决了动态物体导致的相机跟踪漂移和地图模糊问题，并实现了高保真的密集地图重建。

◆ 创新性地将先验图像信息集成到系统中，以生成更精确的掩模，有效减少了因动态物体误判而产生的噪声干扰。
◆ 设计了自适应特征提取策略，在移除动态障碍物后为系统优化提供了更强的约束，显著增强了整个系统在动态场景中的鲁棒性和稳定性。
◆ 整个系统在保证实时运行和高资源效率的同时，在公开动态RGB-D数据集上实现了最先进的相机位姿估计和逼真地图重建性能。</td></tr>
<tr><td>2025-08-06</td><td>Pseudo Depth Meets Gaussian: A Feed-forward RGB SLAM Baseline</td><td>[2508.04597](http://arxiv.org/pdf/2508.04597)</td><td>◆ 提出了一种基于3D高斯映射的RGB SLAM方法，通过结合深度估计器和3D高斯技术，解决了传统方法在长序列处理中的几何细节不准确问题。  
◆ 引入前馈循环预测模块，直接从光流推断相机位姿，替代了耗时的测试时优化，将跟踪速度提升90%以上。  
◆ 设计了局部图渲染技术，增强了前馈位姿预测的鲁棒性，提高了系统在复杂场景中的稳定性。  
◆ 在Replica和TUM-RGBD数据集上的实验表明，该方法性能与当前最先进的SplaTAM相当，同时大幅降低了计算开销。  
◆ 通过实际部署验证了方法的实用性，展示了其在实时3D重建中的高效性和可靠性。</td></tr>
<tr><td>2025-08-01</td><td>CoProU-VO: Combining Projected Uncertainty for End-to-End Unsupervised Monocular Visual Odometry</td><td>[2508.00568](http://arxiv.org/pdf/2508.00568)</td><td>◆ 提出CoProU-VO模型，首次将跨帧不确定性传播与融合引入无监督单目视觉里程计，通过概率公式结合当前帧与参考帧的不确定性，有效识别动态场景中的不可靠区域。  
◆ 设计基于视觉Transformer的端到端框架，同时学习深度、不确定性估计和相机位姿，无需地面真值标签或显式运动分割。  
◆ 创新性地利用投影机制将参考帧的不确定性传递至目标帧，克服传统方法仅依赖单帧不确定性的局限，显著提升动态物体干扰下的鲁棒性。  
◆ 在KITTI和nuScenes数据集上验证性能，尤其在高速公路等动态场景中表现优异，超越现有无监督单目两帧方法。  
◆ 通过详尽的消融实验证明跨帧不确定性传播的有效性，为动态环境下的视觉里程计提供新解决方案。</td></tr>
<tr><td>2025-07-25</td><td>A Fast and Light-weight Non-Iterative Visual Odometry with RGB-D Cameras</td><td>[2507.18886](http://arxiv.org/pdf/2507.18886)</td><td>◆ 提出了一种解耦的非迭代式视觉里程计方法，将6自由度位姿估计分为旋转和平移两步计算，避免了传统迭代优化的计算负担。  
◆ 创新性地利用场景中的重叠平面特征直接计算旋转矩阵，简化了旋转估计流程。  
◆ 采用核互相关器(KCC)计算平移量，省去了传统特征提取与匹配的耗时步骤。  
◆ 整体方案无需迭代优化和特征对齐，在低端i5 CPU上实现了71Hz的高实时性能。  
◆ 在低纹理退化环境中表现优于现有方法，特别适用于无特征点依赖的场景。  
◆ 相比传统RGB-D视觉里程计，显著提升了计算效率并降低了时间延迟。</td></tr>
<tr><td>2025-07-21</td><td>DiffPF: Differentiable Particle Filtering with Generative Sampling via Conditional Diffusion Models</td><td>[2507.15716](http://arxiv.org/pdf/2507.15716)</td><td>◆ DiffPF首次将条件扩散模型融入粒子滤波框架，实现了高质量的后验采样，克服了传统方法的局限性。  
◆ 通过将扩散模型与预测粒子及当前观测条件结合，DiffPF能够从复杂、高维、多模态的滤波分布中生成精确且等权重的粒子。  
◆ 相比传统可微分粒子滤波依赖预定义或低容量提议分布的问题，DiffPF学习了一个灵活的后验采样器，显著提升了采样质量。  
◆ 在多种场景下的实验表明，DiffPF在单模态和高度多模态分布中均表现优异，尤其在多模态全局定位任务中估计精度提升82.8%。  
◆ 在真实世界任务（如KITTI视觉里程计）中，DiffPF相比现有最优可微分滤波器实现了26%的精度提升，验证了其实际应用价值。  
◆ DiffPF的创新性在于将生成式采样与粒子滤波结合，为动态系统状态估计提供了更高效的解决方案。</td></tr>
<tr><td>2025-07-21</td><td>Dense-depth map guided deep Lidar-Visual Odometry with Sparse Point Clouds and Images</td><td>[2507.15496](http://arxiv.org/pdf/2507.15496)</td><td>◆ 提出了一种新颖的LiDAR-视觉里程计框架，通过深度融合稀疏LiDAR点云和图像数据实现高精度位姿估计。  
◆ 创新性地利用深度补全技术生成稠密深度图，为运动估计提供更丰富的几何约束信息。  
◆ 设计了带注意力机制的多尺度特征提取网络，能够自适应生成深度感知的特征表示。  
◆ 利用稠密深度信息优化光流估计，有效减少了遮挡区域的误差累积问题。  
◆ 开发了分层位姿优化模块，通过渐进式运动估计提升动态环境和尺度模糊场景下的鲁棒性。  
实验证明该方法在KITTI数据集上达到或超越了当前最优视觉和LiDAR里程计的精度与鲁棒性。</td></tr>
<tr><td>2025-07-17</td><td>DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model</td><td>[2507.13145](http://arxiv.org/pdf/2507.13145)</td><td>◆ 提出DINO-VO系统，首次将视觉基础模型DINOv2的稀疏特征匹配能力应用于单目视觉里程计任务，解决了传统学习型VO在鲁棒性和泛化性上的不足。  
◆ 针对DINOv2特征粒度粗糙的问题，设计了一种专有的显著关键点检测器，有效提升了特征定位精度。  
◆ 创新性地结合DINOv2的鲁棒语义特征与细粒度几何特征，形成更具区分度的混合特征表示，增强了场景局部化能力。  
◆ 采用基于Transformer的匹配器和可微分位姿估计层，通过端到端学习实现高精度的相机运动估计。  
◆ 在TartanAir、KITTI等数据集上超越传统帧间VO方法，室外驾驶场景性能媲美SLAM系统，同时保持72FPS实时性和低内存占用（&lt;1GB）。  
◆ 实验证明其描述子性能优于独立使用DINOv2粗特征，在复杂环境中展现出更强的鲁棒性和跨数据集泛化能力。</td></tr>
<tr><td>2025-07-10</td><td>Hardware-Aware Feature Extraction Quantisation for Real-Time Visual Odometry on FPGA Platforms</td><td>[2507.07903](http://arxiv.org/pdf/2507.07903)</td><td>◆ 提出了一种基于量化SuperPoint卷积神经网络的嵌入式无监督架构，用于实时视觉里程计中的特征点检测与描述。  
◆ 通过硬件感知的模型量化技术（使用Brevitas库和FINN框架），在保证高检测质量的同时显著降低了计算需求。  
◆ 在AMD/Xilinx Zynq UltraScale+ FPGA平台上实现高效部署，支持640x480分辨率图像处理，帧率高达54fps。  
◆ 针对深度学习处理单元（DPU）进行性能优化，在资源受限的移动或嵌入式系统中展现出优越的实时性。  
◆ 在TUM数据集上验证了不同量化技术对模型精度与性能的影响，为硬件部署提供了实用化设计指导。  
◆ 整体方案在计算效率和部署灵活性上优于现有先进方案，推动了视觉里程计在自主导航系统中的实际应用。</td></tr>
<tr><td>2025-07-10</td><td>IRAF-SLAM: An Illumination-Robust and Adaptive Feature-Culling Front-End for Visual SLAM in Challenging Environments</td><td>[2507.07752](http://arxiv.org/pdf/2507.07752)</td><td>IRAF-SLAM论文的核心贡献是通过自适应前端设计提升视觉SLAM在复杂光照环境下的鲁棒性，主要创新点如下：

◆ 提出图像增强方案，通过预处理动态调整不同光照条件下的图像质量，提升特征提取稳定性。

◆ 设计自适应特征提取机制，基于图像熵、像素强度和梯度分析动态调整检测灵敏度，适应环境变化。

◆ 开发特征筛选策略，结合密度分布分析和光照影响因子过滤不可靠特征点，提高跟踪可靠性。

◆ 在TUM-VI和EuRoC数据集上的实验表明，该方法显著减少跟踪失败并提升轨迹精度，优于现有vSLAM方法。

◆ 实现计算高效的自适应前端策略，在不显著增加计算负担的前提下提升系统鲁棒性。

◆ 公开了算法实现代码，促进相关研究发展。</td></tr>
<tr><td>2025-06-30</td><td>VOCAL: Visual Odometry via ContrAstive Learning</td><td>[2507.00243](http://arxiv.org/pdf/2507.00243)</td><td>◆ 提出VOCAL框架，将视觉里程计（VO）重新定义为标签排序问题，突破传统基于几何假设的局限。  
◆ 结合贝叶斯推理与表征学习，使视觉特征与相机状态在潜在空间中形成映射关系。  
◆ 通过排序机制强制相似相机状态在潜在空间内聚合成一致且空间连贯的表征，提升特征可解释性。  
◆ 支持多模态数据融合，增强框架的适应性和灵活性。  
◆ 在KITTI数据集上的实验验证了VOCAL在可解释性和性能上的优势，推动VO向通用可解释空间智能发展。</td></tr>
<tr><td>2025-06-29</td><td>Event-based Stereo Visual-Inertial Odometry with Voxel Map</td><td>[2506.23078](http://arxiv.org/pdf/2506.23078)</td><td>◆ 提出Voxel-ESVIO系统，首次将体素地图管理引入事件相机的双目视觉-惯性里程计，有效提升位姿估计精度。  
◆ 设计基于体素的点筛选机制，通过体素单元动态管理3D地图点，显著降低事件流噪声对特征质量的影响。  
◆ 创新体素感知的点更新策略，在体素级别协同优化地图点的选择与更新过程，增强系统对动态场景的适应性。  
◆ 实现高效的高质量特征点检索方法，优先选择当前帧中观测概率最高的抗噪声特征，确保状态估计稳定性。  
◆ 在三个公开数据集上验证，系统在精度和计算效率上均超越现有最优方法，尤其在高动态场景优势明显。</td></tr>
<tr><td>2025-06-25</td><td>Real-Time Obstacle Avoidance Algorithms for Unmanned Aerial and Ground Vehicles</td><td>[2506.20311](http://arxiv.org/pdf/2506.20311)</td><td>◆ 开发了适用于复杂3D环境的实时避障算法，特别针对森林火灾等灾害场景中的无人机自主导航需求。  
◆ 提出了一种创新的2D融合导航策略，最初为地面移动机器人设计，具备动态环境下的安全移动能力，并支持自适应障碍处理与决策优化。  
◆ 首次设计了针对森林火灾模拟的3D反应式导航策略，解决了无人机在此类高危场景中的碰撞规避难题。  
◆ 开创性地提出无人机与地面无人车辆的协同控制框架，实现空-地联合救援任务的高效协调。  
◆ 通过数学建模与仿真验证相结合的方式，系统性地解决了陌生危险环境中的导航算法设计挑战。  
◆ 研究成果为提升无人机在自然灾害救援中的实战价值提供了兼具学术深度与应用前景的解决方案。</td></tr>
<tr><td>2025-06-23</td><td>GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM</td><td>[2506.18885](http://arxiv.org/pdf/2506.18885)</td><td>◆ 提出了GRAND-SLAM方法，首次将3D高斯泼溅技术应用于大规模户外多智能体SLAM场景，突破了现有方法仅限于小规模室内环境的限制。  
◆ 设计了基于局部子地图优化的隐式跟踪模块，有效提升了多智能体系统的定位精度和鲁棒性。  
◆ 开发了新型机器人内外闭环检测方法，并将其集成到位姿图优化框架中，实现了全局一致性建图。  
◆ 在Replica室内数据集上实现了当前最优的跟踪性能，PSNR指标比现有方法提升28%。  
◆ 在大型户外Kimera-Multi数据集上，多智能体跟踪误差降低91%，渲染质量显著优于现有方法。  
◆ 通过可扩展的环境表示和协作式建图机制，为快速探索和重建大规模环境提供了新解决方案。</td></tr>
</tbody>
</table>
</div>

<h2 id='loop-closure'>Loop Closure</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-10-20</td><td>Joint Multi-Condition Representation Modelling via Matrix Factorisation for Visual Place Recognition</td><td>[2510.17739](http://arxiv.org/pdf/2510.17739)</td><td>◆ We address multi-reference visual place recognition (VPR), where reference sets captured under varying conditions are used to improve localisation performance.
◆ While deep learning with large-scale training improves robustness, increasing data diversity and model complexity incur extensive computational cost during training and deployment.
◆ Descriptor-level fusion via voting or aggregation avoids training, but often targets multi-sensor setups or relies on heuristics with limited gains under appearance and viewpoint change.</td></tr>
<tr><td>2025-10-14</td><td>CE$ν$NS Search with Cryogenic Sapphire Detectors at MINER: Results from the TRIGA reactor data and Future Sensitivity at HFIR</td><td>[2510.15999](http://arxiv.org/pdf/2510.15999)</td><td>◆ We report on a search for coherent elastic neutrino--nucleus scattering (CE$\nu$NS) using cryogenic sapphire (Al$_2$O$_3$) detectors deployed at the Mitchell Institute Neutrino Experiment at Reactor (MINER), located near the 1~MW$_\text{th}$ TRIGA research reactor at Texas A\&amp;M University.
◆ The experiment operated with a primary detector mass of 72~g and achieved a baseline energy resolution of $\sim 40$~eV.
◆ Using exposures of 158~g-days (reactor-on) and 381~g-days (reactor-off), we performed a statistical background subtraction in the energy region of 0.25--3~keV.</td></tr>
<tr><td>2025-10-15</td><td>Investigating Web Content Delivery Performance over Starlink</td><td>[2510.13710](http://arxiv.org/pdf/2510.13710)</td><td>◆ Low Earth Orbit (LEO) satellite ISPs promise universal Internet connectivity, yet their interaction with content delivery remains poorly understood.
◆ We present the first comprehensive measurement study decomposing Starlink&#x27;s web content delivery performance decomposed across Point of Presence (PoP), DNS, and CDN layers.
◆ Through two years of measurements combining 225K Cloudflare AIM tests, M-Lab data, and active probing from 99 RIPE Atlas and controlled Starlink probes, we collect 6.1M traceroutes and 10.8M DNS queries to quantify how satellite architecture disrupts terrestrial CDN assumptions.</td></tr>
<tr><td>2025-10-15</td><td>Through the Lens of Doubt: Robust and Efficient Uncertainty Estimation for Visual Place Recognition</td><td>[2510.13464](http://arxiv.org/pdf/2510.13464)</td><td>◆ Visual Place Recognition (VPR) enables robots and autonomous vehicles to identify previously visited locations by matching current observations against a database of known places.
◆ However, VPR systems face significant challenges when deployed across varying visual environments, lighting conditions, seasonal changes, and viewpoints changes.
◆ Failure-critical VPR applications, such as loop closure detection in simultaneous localization and mapping (SLAM) pipelines, require robust estimation of place matching uncertainty.</td></tr>
<tr><td>2025-10-15</td><td>DAMM-LOAM: Degeneracy Aware Multi-Metric LiDAR Odometry and Mapping</td><td>[2510.13287](http://arxiv.org/pdf/2510.13287)</td><td>◆ LiDAR Simultaneous Localization and Mapping (SLAM) systems are essential for enabling precise navigation and environmental reconstruction across various applications.
◆ Although current point-to-plane ICP algorithms perform effec- tively in structured, feature-rich environments, they struggle in scenarios with sparse features, repetitive geometric structures, and high-frequency motion.
◆ This leads to degeneracy in 6- DOF pose estimation.</td></tr>
<tr><td>2025-10-14</td><td>Scene Coordinate Reconstruction Priors</td><td>[2510.12387](http://arxiv.org/pdf/2510.12387)</td><td>◆ Scene coordinate regression (SCR) models have proven to be powerful implicit scene representations for 3D vision, enabling visual relocalization and structure-from-motion.
◆ SCR models are trained specifically for one scene.
◆ If training images imply insufficient multi-view constraints SCR models degenerate.</td></tr>
<tr><td>2025-10-13</td><td>Deterministic hBN bubbles as a versatile platform for studies on single-photon emitters</td><td>[2510.11610](http://arxiv.org/pdf/2510.11610)</td><td>◆ Single-photon emitters (SPEs) in two-dimensional materials are highly promising candidates for quantum technologies.
◆ SPEs in hexagonal boron nitride (hBN) have been widely investigated, but mostly in exfoliated or powder samples that require an activation process, making it difficult to compare studies and reproduce results.
◆ Here, we address this problem and propose a platform based on large-area metaloraganic vapour phase epitaxy (MOVPE)-grown hBN, which combines reproducibility and scalability with the ability to readily host SPEs without activation.</td></tr>
<tr><td>2025-10-13</td><td>ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training</td><td>[2510.11605](http://arxiv.org/pdf/2510.11605)</td><td>◆ Scene coordinate regression (SCR) has established itself as a promising learning-based approach to visual relocalization.
◆ After mere minutes of scene-specific training, SCR models estimate camera poses of query images with high accuracy.
◆ Still, SCR methods fall short of the generalization capabilities of more classical feature-matching approaches.</td></tr>
<tr><td>2025-10-11</td><td>Old is Gold: Optimizing Single-threaded Applications with Exgen-Malloc</td><td>[2510.10219](http://arxiv.org/pdf/2510.10219)</td><td>◆ Memory allocators hide beneath nearly every application stack, yet their performance footprint extends far beyond their code size.
◆ Even small inefficiencies in the allocators ripple through caches and the rest of the memory hierarchy, collectively imposing what operators often call a &quot;datacenter tax&quot;.
◆ At hyperscale, even a 1% improvement in allocator efficiency can unlock millions of dollars in savings and measurable reductions in datacenter energy consumption.</td></tr>
<tr><td>2025-10-10</td><td>Robust Visual Teach-and-Repeat Navigation with Flexible Topo-metric Graph Map Representation</td><td>[2510.09089](http://arxiv.org/pdf/2510.09089)</td><td>◆ Visual Teach-and-Repeat Navigation is a direct solution for mobile robot to be deployed in unknown environments.
◆ However, robust trajectory repeat navigation still remains challenged due to environmental changing and dynamic objects.
◆ In this paper, we propose a novel visual teach-and-repeat navigation system, which consists of a flexible map representation, robust map matching and a map-less local navigation module.</td></tr>
<tr><td>2025-10-09</td><td>Pursuing decarbonization and competitiveness: a narrow corridor for European green industrial transformation</td><td>[2510.08199](http://arxiv.org/pdf/2510.08199)</td><td>◆ This study analyzes how Europe can decarbonize its industrial sector while remaining competitive.
◆ Using the open-source model PyPSA-Eur, it examines key energy- and emission-intensive industries, including steel, cement, methanol, ammonia, and high-value chemicals.
◆ Two development paths are explored: a continued decline in industrial activity and a reindustrialization driven by competitiveness policies.</td></tr>
<tr><td>2025-10-09</td><td>InstructUDrag: Joint Text Instructions and Object Dragging for Interactive Image Editing</td><td>[2510.08181](http://arxiv.org/pdf/2510.08181)</td><td>◆ Text-to-image diffusion models have shown great potential for image editing, with techniques such as text-based and object-dragging methods emerging as key approaches.
◆ However, each of these methods has inherent limitations: text-based methods struggle with precise object positioning, while object dragging methods are confined to static relocation.
◆ To address these issues, we propose InstructUDrag, a diffusion-based framework that combines text instructions with object dragging, enabling simultaneous object dragging and text-based image editing.</td></tr>
<tr><td>2025-10-07</td><td>The DISTANT Design for Remote Transmission and Steering Systems for Planetary Robotics</td><td>[2510.05981](http://arxiv.org/pdf/2510.05981)</td><td>◆ Planetary exploration missions require robust locomotion systems capable of operating in extreme environments over extended periods.
◆ This paper presents the DISTANT (Distant Transmission and Steering Systems) design, a novel approach for relocating rover traction and steering actuators from wheel-mounted positions to a thermally protected warm box within the rover body.
◆ The design addresses critical challenges in long-distance traversal missions by protecting sensitive components from thermal cycling, dust contamination, and mechanical wear.</td></tr>
<tr><td>2025-10-05</td><td>Flexible and Efficient Spatio-Temporal Transformer for Sequential Visual Place Recognition</td><td>[2510.04282](http://arxiv.org/pdf/2510.04282)</td><td>◆ Sequential Visual Place Recognition (Seq-VPR) leverages transformers to capture spatio-temporal features effectively; however, existing approaches prioritize performance at the expense of flexibility and efficiency.
◆ In practice, a transformer-based Seq-VPR model should be flexible to the number of frames per sequence (seq-length), deliver fast inference, and have low memory usage to meet real-time constraints.
◆ To our knowledge, no existing transformer-based Seq-VPR method achieves both flexibility and efficiency.</td></tr>
<tr><td>2025-10-04</td><td>The Overlooked Value of Test-time Reference Sets in Visual Place Recognition</td><td>[2510.03751](http://arxiv.org/pdf/2510.03751)</td><td>◆ Given a query image, Visual Place Recognition (VPR) is the task of retrieving an image of the same place from a reference database with robustness to viewpoint and appearance changes.
◆ Recent works show that some VPR benchmarks are solved by methods using Vision-Foundation-Model backbones and trained on large-scale and diverse VPR-specific datasets.
◆ Several benchmarks remain challenging, particularly when the test environments differ significantly from the usual VPR training datasets.</td></tr>
<tr><td>2025-10-03</td><td>Novel UWB Synthetic Aperture Radar Imaging for Mobile Robot Mapping</td><td>[2510.02874](http://arxiv.org/pdf/2510.02874)</td><td>◆ Traditional exteroceptive sensors in mobile robots, such as LiDARs and cameras often struggle to perceive the environment in poor visibility conditions.
◆ Recently, radar technologies, such as ultra-wideband (UWB) have emerged as potential alternatives due to their ability to see through adverse environmental conditions (e.g.
◆ dust, smoke and rain).</td></tr>
<tr><td>2025-10-02</td><td>EC3R-SLAM: Efficient and Consistent Monocular Dense SLAM with Feed-Forward 3D Reconstruction</td><td>[2510.02080](http://arxiv.org/pdf/2510.02080)</td><td>◆ The application of monocular dense Simultaneous Localization and Mapping (SLAM) is often hindered by high latency, large GPU memory consumption, and reliance on camera calibration.
◆ To relax this constraint, we propose EC3R-SLAM, a novel calibration-free monocular dense SLAM framework that jointly achieves high localization and mapping accuracy, low latency, and low GPU memory consumption.
◆ This enables the framework to achieve efficiency through the coupling of a tracking module, which maintains a sparse map of feature points, and a mapping module based on a feed-forward 3D reconstruction model that simultaneously estimates camera intrinsics.</td></tr>
<tr><td>2025-10-02</td><td>SoK: Measuring What Matters for Closed-Loop Security Agents</td><td>[2510.01654](http://arxiv.org/pdf/2510.01654)</td><td>◆ Cybersecurity is a relentless arms race, with AI driven offensive systems evolving faster than traditional defenses can adapt.
◆ Research and tooling remain fragmented across isolated defensive functions, creating blind spots that adversaries exploit.
◆ Autonomous agents capable of integrating, exploit confirmation, remediation, and validation into a single closed loop offer promise, but the field lacks three essentials: a framework defining the agentic capabilities of security systems across security life cycle, a principled method for evaluating closed loop agents, and a benchmark for measuring their performance in practice.</td></tr>
<tr><td>2025-10-01</td><td>Kilometer-Scale GNSS-Denied UAV Navigation via Heightmap Gradients: A Winning System from the SPRIN-D Challenge</td><td>[2510.01348](http://arxiv.org/pdf/2510.01348)</td><td>◆ Reliable long-range flight of unmanned aerial vehicles (UAVs) in GNSS-denied environments is challenging: integrating odometry leads to drift, loop closures are unavailable in previously unseen areas and embedded platforms provide limited computational power.
◆ We present a fully onboard UAV system developed for the SPRIN-D Funke Fully Autonomous Flight Challenge, which required 9 km long-range waypoint navigation below 25 m AGL (Above Ground Level) without GNSS or prior dense mapping.
◆ The system integrates perception, mapping, planning, and control with a lightweight drift-correction method that matches LiDAR-derived local heightmaps to a prior geo-data heightmap via gradient-template matching and fuses the evidence with odometry in a clustered particle filter.</td></tr>
<tr><td>2025-10-01</td><td>EvoWorld: Evolving Panoramic World Generation with Explicit 3D Memory</td><td>[2510.01183](http://arxiv.org/pdf/2510.01183)</td><td>◆ Humans possess a remarkable ability to mentally explore and replay 3D environments they have previously experienced.
◆ Inspired by this mental process, we present EvoWorld: a world model that bridges panoramic video generation with evolving 3D memory to enable spatially consistent long-horizon exploration.
◆ Given a single panoramic image as input, EvoWorld first generates future video frames by leveraging a video generator with fine-grained view control, then evolves the scene&#x27;s 3D reconstruction using a feedforward plug-and-play transformer, and finally synthesizes futures by conditioning on geometric reprojections from this evolving explicit 3D memory.</td></tr>
<tr><td>2025-10-01</td><td>A Scene is Worth a Thousand Features: Feed-Forward Camera Localization from a Collection of Image Features</td><td>[2510.00978](http://arxiv.org/pdf/2510.00978)</td><td>◆ Visually localizing an image, i.e., estimating its camera pose, requires building a scene representation that serves as a visual map.
◆ The representation we choose has direct consequences towards the practicability of our system.
◆ Even when starting from mapping images with known camera poses, state-of-the-art approaches still require hours of mapping time in the worst case, and several minutes in the best.</td></tr>
<tr><td>2025-10-01</td><td>Semantic Visual Simultaneous Localization and Mapping: A Survey on State of the Art, Challenges, and Future Directions</td><td>[2510.00783](http://arxiv.org/pdf/2510.00783)</td><td>◆ Semantic Simultaneous Localization and Mapping (SLAM) is a critical area of research within robotics and computer vision, focusing on the simultaneous localization of robotic systems and associating semantic information to construct the most accurate and complete comprehensive model of the surrounding environment.
◆ Since the first foundational work in Semantic SLAM appeared more than two decades ago, this field has received increasing attention across various scientific communities.
◆ Despite its significance, the field lacks comprehensive surveys encompassing recent advances and persistent challenges.</td></tr>
<tr><td>2025-10-01</td><td>RELATE-Sim: Leveraging Turning Point Theory and LLM Agents to Predict and Understand Long-Term Relationship Dynamics through Interactive Narrative Simulations</td><td>[2510.00414](http://arxiv.org/pdf/2510.00414)</td><td>◆ Most dating technologies optimize for getting together, not staying together.
◆ We present RELATE-Sim, a theory-grounded simulator that models how couples behave at consequential turning points-exclusivity talks, conflict-and-repair episodes, relocations-rather than static traits.
◆ Two persona-aligned LLM agents (one per partner) interact under a centralized Scene Master that frames each turning point as a compact set of realistic options, advances the narrative, and infers interpretable state changes and an auditable commitment estimate after each scene.</td></tr>
<tr><td>2025-09-30</td><td>Dynamic Necklace Splitting</td><td>[2510.00162](http://arxiv.org/pdf/2510.00162)</td><td>◆ The necklace splitting problem is a classic problem in fair division with many applications, including data-informed fair hash maps.
◆ We extend necklace splitting to a dynamic setting, allowing for relocation, insertion, and deletion of beads.
◆ We present linear-time, optimal algorithms for the two-color case that support all dynamic updates.</td></tr>
<tr><td>2025-09-30</td><td>An Agent-Based Simulation of Ageing Societies: Accessibility and Care Dynamics in Remote Areas</td><td>[2509.26496](http://arxiv.org/pdf/2509.26496)</td><td>◆ This paper presents an agent-based simulation of accessibility and care dynamics in ageing societies, applied to the Italian inner area of Premeno (VB).
◆ The model integrates census and municipal data, drone-derived elevation models, GIS road networks, and survey-based caregiving information to generate synthetic populations of older adults and their caregivers.
◆ Agents are organized into dyads with socio-economic and mobility attributes, enabling the simulation of both micro-scale accessibility and meso-scale caregiving outcomes.</td></tr>
<tr><td>2025-09-30</td><td>SAGE: Spatial-visual Adaptive Graph Exploration for Visual Place Recognition</td><td>[2509.25723](http://arxiv.org/pdf/2509.25723)</td><td>◆ Visual Place Recognition (VPR) requires robust retrieval of geotagged images despite large appearance, viewpoint, and environmental variation.
◆ Prior methods focus on descriptor fine-tuning or fixed sampling strategies yet neglect the dynamic interplay between spatial context and visual similarity during training.
◆ We present SAGE (Spatial-visual Adaptive Graph Exploration), a unified training pipeline that enhances granular spatial-visual discrimination by jointly improving local feature aggregation, organize samples during training, and hard sample mining.</td></tr>
<tr><td>2025-09-28</td><td>Prepare for Warp Speed: Sub-millisecond Visual Place Recognition Using Event Cameras</td><td>[2509.24094](http://arxiv.org/pdf/2509.24094)</td><td>◆ Visual Place Recognition (VPR) enables systems to identify previously visited locations within a map, a fundamental task for autonomous navigation.
◆ Prior works have developed VPR solutions using event cameras, which asynchronously measure per-pixel brightness changes with microsecond temporal resolution.
◆ However, these approaches rely on dense representations of the inherently sparse camera output and require tens to hundreds of milliseconds of event data to predict a place.</td></tr>
<tr><td>2025-09-26</td><td>Vector Resonant Relaxation and Statistical Closure Theory. II. One-loop Closure</td><td>[2509.22164](http://arxiv.org/pdf/2509.22164)</td><td>◆ We use stellar dynamics as a testbed for statistical closure theory.
◆ We focus on the process of &quot;Vector Resonant Relaxation,&quot; a long-range, non-linear, and correlated relaxation mechanism that drives the reorientation of stellar orbital planes around a supermassive black hole.
◆ This process provides a natural setting to evaluate the predictive power of generic statistical closure schemes for dynamical correlation functions, in the fully non-linear and non-perturbative regime.</td></tr>
<tr><td>2025-09-25</td><td>Guiding Audio Editing with Audio Language Model</td><td>[2509.21625](http://arxiv.org/pdf/2509.21625)</td><td>◆ Audio editing plays a central role in VR/AR immersion, virtual conferencing, sound design, and other interactive media.
◆ However, recent generative audio editing models depend on template-like instruction formats and are restricted to mono-channel audio.
◆ These models fail to deal with declarative audio editing, where the user declares what the desired outcome should be, while leaving the details of editing operations to the system.</td></tr>
<tr><td>2025-09-29</td><td>MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM</td><td>[2509.20757](http://arxiv.org/pdf/2509.20757)</td><td>◆ Visual SLAM is a cornerstone technique in robotics, autonomous driving and extended reality (XR), yet classical systems often struggle with low-texture environments, scale ambiguity, and degraded performance under challenging visual conditions.
◆ Recent advancements in feed-forward neural network-based pointmap regression have demonstrated the potential to recover high-fidelity 3D scene geometry directly from images, leveraging learned spatial priors to overcome limitations of traditional multi-view geometry methods.
◆ However, the widely validated advantages of probabilistic multi-sensor information fusion are often discarded in these pipelines.</td></tr>
<tr><td>2025-09-24</td><td>Projective Kolmogorov Arnold Neural Networks (P-KANs): Entropy-Driven Functional Space Discovery for Interpretable Machine Learning</td><td>[2509.20049](http://arxiv.org/pdf/2509.20049)</td><td>◆ Kolmogorov-Arnold Networks (KANs) relocate learnable nonlinearities from nodes to edges, demonstrating remarkable capabilities in scientific machine learning and interpretable modeling.
◆ However, current KAN implementations suffer from fundamental inefficiencies due to redundancy in high-dimensional spline parameter spaces, where numerous distinct parameterisations yield functionally equivalent behaviors.
◆ This redundancy manifests as a &quot;nuisance space&quot; in the model&#x27;s Jacobian, leading to susceptibility to overfitting and poor generalization.</td></tr>
<tr><td>2025-09-23</td><td>CU-Multi: A Dataset for Multi-Robot Collaborative Perception</td><td>[2509.19463](http://arxiv.org/pdf/2509.19463)</td><td>◆ A central challenge for multi-robot systems is fusing independently gathered perception data into a unified representation.
◆ Despite progress in Collaborative SLAM (C-SLAM), benchmarking remains hindered by the scarcity of dedicated multi-robot datasets.
◆ Many evaluations instead partition single-robot trajectories, a practice that may only partially reflect true multi-robot operations and, more critically, lacks standardization, leading to results that are difficult to interpret or compare across studies.</td></tr>
<tr><td>2025-09-23</td><td>CAR-Flow: Condition-Aware Reparameterization Aligns Source and Target for Better Flow Matching</td><td>[2509.19300](http://arxiv.org/pdf/2509.19300)</td><td>◆ Conditional generative modeling aims to learn a conditional data distribution from samples containing data-condition pairs.
◆ For this, diffusion and flow-based methods have attained compelling results.
◆ These methods use a learned (flow) model to transport an initial standard Gaussian noise that ignores the condition to the conditional data distribution.</td></tr>
<tr><td>2025-09-21</td><td>A game played by tandem-running ants: Hint of procedural rationality</td><td>[2509.17147](http://arxiv.org/pdf/2509.17147)</td><td>◆ Navigation through narrow passages during colony relocation by the tandem-running ants, $\textit{Diacamma}$ $\textit{indicum}$, is a tour de force of biological traffic coordination.
◆ Even on one-lane paths, the ants tactfully manage a bidirectional flow: Informed individuals (termed leaders) guide nest-mates (termed followers) from a suboptimal nest to a new optimal nest, and then return to recruit additional followers.
◆ We propose that encounters between the ants moving in opposite directions can be modelled within the framework of game theory leading to an understanding of the mechanism behind observed behaviours.</td></tr>
<tr><td>2025-09-21</td><td>ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM</td><td>[2509.16863](http://arxiv.org/pdf/2509.16863)</td><td>◆ We introduce ConfidentSplat, a novel 3D Gaussian Splatting (3DGS)-based SLAM system for robust, highfidelity RGB-only reconstruction.
◆ Addressing geometric inaccuracies in existing RGB-only 3DGS SLAM methods that stem from unreliable depth estimation, ConfidentSplat incorporates a core innovation: a confidence-weighted fusion mechanism.
◆ This mechanism adaptively integrates depth cues from multiview geometry with learned monocular priors (Omnidata ViT), dynamically weighting their contributions based on explicit reliability estimates-derived predominantly from multi-view geometric consistency-to generate high-fidelity proxy depth for map supervision.</td></tr>
<tr><td>2025-09-20</td><td>Ductile fracture of HDPE thin films: failure mechanisms and tuning of fracture properties by bonding a rubber layer</td><td>[2509.16731](http://arxiv.org/pdf/2509.16731)</td><td>◆ High-density polyethylene (HDPE) thin films, while inherently ductile, exhibit poor flaw tolerance.
◆ Our experiments show that they fail prematurely not at the point of maximum stretch, but at the boundary of a necked region or notch-tip plastic zone.
◆ This study investigates this counter-intuitive failure mechanism and demonstrates how an elastomeric interlayer can mitigate it to enhance toughness.</td></tr>
<tr><td>2025-09-19</td><td>Dynamic Objects Relocalization in Changing Environments with Flow Matching</td><td>[2509.16398](http://arxiv.org/pdf/2509.16398)</td><td>◆ Task and motion planning are long-standing challenges in robotics, especially when robots have to deal with dynamic environments exhibiting long-term dynamics, such as households or warehouses.
◆ In these environments, long-term dynamics mostly stem from human activities, since previously detected objects can be moved or removed from the scene.
◆ This adds the necessity to find such objects again before completing the designed task, increasing the risk of failure due to missed relocalizations.</td></tr>
<tr><td>2025-09-18</td><td>Event-LAB: Towards Standardized Evaluation of Neuromorphic Localization Methods</td><td>[2509.14516](http://arxiv.org/pdf/2509.14516)</td><td>◆ Event-based localization research and datasets are a rapidly growing area of interest, with a tenfold increase in the cumulative total number of published papers on this topic over the past 10 years.
◆ Whilst the rapid expansion in the field is exciting, it brings with it an associated challenge: a growth in the variety of required code and package dependencies as well as data formats, making comparisons difficult and cumbersome for researchers to implement reliably.
◆ To address this challenge, we present Event-LAB: a new and unified framework for running several event-based localization methodologies across multiple datasets.</td></tr>
<tr><td>2025-09-16</td><td>Semantic-Enhanced Cross-Modal Place Recognition for Robust Robot Localization</td><td>[2509.13474](http://arxiv.org/pdf/2509.13474)</td><td>◆ Ensuring accurate localization of robots in environments without GPS capability is a challenging task.
◆ Visual Place Recognition (VPR) techniques can potentially achieve this goal, but existing RGB-based methods are sensitive to changes in illumination, weather, and other seasonal changes.
◆ Existing cross-modal localization methods leverage the geometric properties of RGB images and 3D LiDAR maps to reduce the sensitivity issues highlighted above.</td></tr>
<tr><td>2025-09-15</td><td>On the (Im)possibility of Electrically Charged Planck Relics</td><td>[2509.12520](http://arxiv.org/pdf/2509.12520)</td><td>◆ I revisit whether black-hole remnants, from sub-Planckian compact objects to Planck relics and up to (super)massive black holes, can preserve Standard-Model (SM) electric charge.
◆ Two exterior-field mechanisms -- Coulomb-focused capture from ambient media and QED Schwinger pair production -- robustly neutralize such objects across cosmic history.
◆ I first derive the general capture rate including both Coulomb and gravitational focusing, and sum the stepwise discharge time in closed form via the trigamma function, exhibiting transparent Coulomb- and gravity-dominated limits.</td></tr>
<tr><td>2025-09-15</td><td>Learning to Generate 4D LiDAR Sequences</td><td>[2509.11959](http://arxiv.org/pdf/2509.11959)</td><td>本文提出了LiDARCrafter，一个通过自由文本生成可编辑4D激光雷达序列的统一框架。其核心贡献与创新点包括：
◆ 首次将自由形式语言描述转化为可编辑的4D激光雷达序列，实现了高水平的可控生成。
◆ 设计了一种三分支扩散模型，将指令解析为以自我为中心的场景图，并分别生成对象布局、轨迹和形状，确保了结构合理性。
◆ 采用范围图像扩散模型生成初始扫描，并通过自回归模块扩展为时间一致的序列，有效保障了时序稳定性。
◆ 提供了显式的布局设计，支持在对象级别进行编辑操作，如插入或重新定位物体。
◆ 推出了EvalSuite评估基准，涵盖场景、对象和序列级别的多维度指标，为公平评估激光雷达生成质量建立了标准。
该框架在nuScenes数据集上实现了最优的保真度、可控性和时序一致性，为激光雷达仿真与数据增强奠定了基础。</td></tr>
<tr><td>2025-09-15</td><td>μFork: Supporting POSIX fork Within a Single-Address-Space OS</td><td>[2509.09439](http://arxiv.org/pdf/2509.09439)</td><td>该论文提出了μFork，一种在单地址空间操作系统中支持POSIX fork的创新设计，解决了传统单地址空间OS与多进程POSIX应用的不兼容问题。  
◆ 首次在单地址空间OS中完整实现POSIX fork语义，无需依赖多地址空间，突破了原有设计限制。  
◆ 采用内存复制与重定位技术，通过将子进程内存复制到同一地址空间的不同位置来模拟传统fork行为。  
◆ 利用CHERI硬件能力高效解决子进程内存绝对地址引用（指针）重定向问题，确保正确性。  
◆ 在保持单地址空间轻量级优势的同时，通过硬件辅助实现用户/内核及进程间隔离，兼顾安全与性能。  
实验表明，μFork在Redis、Nginx等实际场景中性能显著提升，fork速度比传统系统快3.7倍，且FaaS吞吐量提高24%。</td></tr>
<tr><td>2025-09-11</td><td>S-BEVLoc: BEV-based Self-supervised Framework for Large-scale LiDAR Global Localization</td><td>[2509.09110](http://arxiv.org/pdf/2509.09110)</td><td>S-BEVLoc提出了一种基于鸟瞰图（BEV）的激光雷达全局定位自监督框架，其核心贡献在于摆脱了对高成本真值位姿的依赖。  
◆首创了自监督训练范式，仅利用单张BEV图像和已知地理距离构建训练三元组，无需任何GPS或SLAM提供的位姿真值监督。  
◆设计了基于关键点中心BEV图像块的局部特征提取与NetVLAD全局描述子聚合的网络架构，有效捕获场景特征。  
◆引入了SoftCos损失函数，优化三元组学习过程，提升特征表达的判别力和鲁棒性。  
实验证明，该框架在大规模KITTI和NCLT数据集上实现了与全监督方法相当甚至更优的闭环检测和全局定位性能，同时展现出极强的可扩展性。</td></tr>
<tr><td>2025-09-08</td><td>Influence of Boundary Conditions and Heating Modes on the Onset of Columnar Convection in Rotating Spherical Shells</td><td>[2509.06632](http://arxiv.org/pdf/2509.06632)</td><td>该论文研究了旋转球壳中热对流失稳的线性临界问题，系统揭示了边界条件和加热方式的关键影响。  
◆ 首次系统证明最优边界条件（无滑移或应力自由）的选择并非固定，而是取决于壳的厚径比和普朗特数，特别是在厚壳或高Pr时，无滑移外边界因Ekman层失稳反而更易触发对流。  
◆ 详细比较了不同加热模式，发现内部加热普遍提高临界瑞利数，使对流以更高波数和频率起始，并将临界柱位置移出切线圆柱区域。  
◆ 揭示了混合边界条件（内无滑移、外应力自由）的临界行为与全应力自由条件相似，表明外边界在控制对流失稳中占主导地位。  
◆ 通过宽参数范围（中埃克曼数、多种Pr和厚径比）的谱方法计算，为行星和恒星内部对流模型提供了关键定量依据。  
这些发现强调边界条件与加热机制在旋转球壳对流起始中起着核心控制作用。</td></tr>
<tr><td>2025-09-08</td><td>Toward Alternative Earths&#x27; Habitability of Solar System Bodies at Earth&#x27;s Orbit</td><td>[2509.06259](http://arxiv.org/pdf/2509.06259)</td><td>该论文首次系统性地评估了太阳系天体若位于地球轨道（1 AU）时的潜在宜居性。

◆ 提出了首个结构化评估框架，采用行星尺寸与重力、大气保留能力、挥发性物质可获取性等多元标准进行分析。
◆ 否定了多数天体的可行性：水星和月球缺乏挥发物与大气层，气态和冰巨星无固体表面，金星则受困于极端温室效应。
◆ 明确火星为最优候选，因其兼具资源可获取性与挥发性物质平衡。
◆ 发现土卫六（Titan）具长期潜力，其浓厚大气与丰富有机物在1 AU下可能转化为水基循环系统。
◆ 为行星改造和人类长期生存策略提供了新的理论路径与科学依据。</td></tr>
<tr><td>2025-09-06</td><td>Multi-LVI-SAM: A Robust LiDAR-Visual-Inertial Odometry for Multiple Fisheye Cameras</td><td>[2509.05740](http://arxiv.org/pdf/2509.05740)</td><td>本文提出了一种多相机LiDAR-视觉-惯性里程计框架Multi-LVI-SAM，通过融合多个鱼眼相机、LiDAR和IMU的数据实现高精度和鲁棒的状态估计。
◆ 提出全景视觉特征模型，将多相机观测统一为单一表示，实现了高效且一致的多源视觉信息融合。
◆ 该模型作为全局几何优化框架，整合多视角约束，支持无缝闭环检测和全局位姿优化，同时简化了系统设计。
◆ 针对相机帧与全景模型帧未对准导致的三角化不一致问题，提出外参补偿方法，显著减少三角化和优化误差。
◆ 将全景模型紧密集成于基于因子图的LiDAR-视觉-惯性系统，在公开数据集上验证了其优于现有多相机融合系统的精度和鲁棒性。</td></tr>
<tr><td>2025-09-05</td><td>Towards an Accurate and Effective Robot Vision (The Problem of Topological Localization for Mobile Robots)</td><td>[2509.04948](http://arxiv.org/pdf/2509.04948)</td><td>该论文针对移动机器人在办公室环境中的拓扑定位问题，提出了一种仅依赖单目彩色相机图像且不利用时序连续性的视觉定位方法。  
◆ 系统性地评估并比较了多种先进视觉描述符，包括颜色直方图、SIFT、ASIFT、RGB-SIFT以及基于词袋模型的方法。  
◆ 对不同特征描述符、距离度量方式和分类器组合进行了定量分析与性能对比，扩展了已有实验范围。  
◆ 采用标准评估指标和可视化方法验证了不同配置方案在定位准确性和有效性上的优势。  
◆ 所提出系统在ImageCLEF评测任务中成功验证了其在实际图像序列位置识别中的有效性。  
未来工作将聚焦于层次模型和特征组合以提升系统鲁棒性，同时降低计算复杂度。</td></tr>
<tr><td>2025-09-03</td><td>IL-SLAM: Intelligent Line-assisted SLAM Based on Feature Awareness for Dynamic Environments</td><td>[2509.02972](http://arxiv.org/pdf/2509.02972)</td><td>本文提出了一种基于特征感知的智能线辅助动态SLAM系统IL-SLAM，其核心贡献在于解决了动态环境下特征不足与计算效率的平衡问题。
◆ 提出了一种特征感知机制，动态评估当前特征是否充足，以此智能决定是否需引入线特征辅助，避免了现有方法持续引入额外特征带来的问题。
◆ 仅在必要时激活线特征支持，显著降低了因持续引入额外特征而产生的计算开销，提升了系统效率。
◆ 通过选择性引入策略，有效减少了低质量附加特征和噪声的累积，避免了其对系统性能的潜在负面影响。
◆ 在线特征参与跟踪、局部建图与回环检测以优化初始位姿估计的同时，将其排除在全局优化之外，确保了长期运行的稳定性与精度。
在TUM数据集上的实验表明，该系统在ATE和RPE指标上均显著优于ORB-SLAM3基线及其他动态SLAM与多特征方法。</td></tr>
<tr><td>2025-09-02</td><td>Scale, Don&#x27;t Fine-tune: Guiding Multimodal LLMs for Efficient Visual Place Recognition at Test-Time</td><td>[2509.02129](http://arxiv.org/pdf/2509.02129)</td><td>该论文提出了一种用于视觉地点识别（VPR）的零样本新框架，以解决现有方法计算成本高和跨域泛化能力差的问题。其核心创新点包括：
◆ 提出测试时缩放（TTS）框架，利用多模态大语言模型（MLLMs）的视觉-语言对齐能力进行直接相似性评分，无需微调。
◆ 采用基于引导（Guidance-based）的方法和结构化提示，生成长度可控的JSON输出，从而避免了两阶段处理流程。
◆ 引入不确定性感知自一致性（UASC）机制，实现了无需额外训练成本的实时自适应，提升了方法的鲁棒性。
◆ 在跨域环境中实现了卓越的泛化性能，实验结果显示其计算效率提升了高达210倍，同时显著提高了VPR的准确率。</td></tr>
<tr><td>2025-09-02</td><td>Ensemble-Based Event Camera Place Recognition Under Varying Illumination</td><td>[2509.01968](http://arxiv.org/pdf/2509.01968)</td><td>本文提出了一种集成式事件相机地点识别方法，显著提升了在剧烈光照变化下的环境鲁棒性。  
◆ 采用多事件重建方法、多特征提取器和多时间分辨率的广泛融合策略，突破了以往仅利用时间分辨率的局限。  
◆ 在长时序驾驶数据集上实现了昼夜转换场景下的性能突破，Recall@1指标相对提升57%。  
◆ 深入分析了事件数据处理中的关键设计选项（如重建方法、极性处理等），为系统设计提供了实证依据。  
◆ 改进了序列匹配框架，在长序列条件下进一步提升了识别性能。  
◆ 公开了代码库和基准测试框架，为后续研究提供了重要基础。</td></tr>
<tr><td>2025-09-03</td><td>Intermittent localization and fast spatial learning by non-Markov random walks with decaying memory</td><td>[2509.01806](http://arxiv.org/pdf/2509.01806)</td><td>该论文研究了具有衰减记忆的非马尔可夫随机游走模型，探讨了记忆衰减如何影响空间学习过程。  
◆ 发现记忆衰减速率是决定学习行为的关键：当记忆衰减慢于或等于1/τ时，系统仍能实现与完美记忆相同的稳定局域化状态。  
◆ 揭示了更快的遗忘（如指数衰减）会导致一种新颖的间歇性局域化现象，即指数分布的局域化阶段与扩散运动交替出现。  
◆ 在临界衰减（1/τ核）时，系统达到稳定局域化状态的速度最快，这与临界慢化预期相反，表明适度遗忘可加速学习。  
◆ 理论证明遗忘能帮助游走者节省记忆资源而不损害学习能力，为生物遗忘的益处提供了数学模型支持。</td></tr>
<tr><td>2025-09-01</td><td>ViSTA-SLAM: Visual SLAM with Symmetric Two-view Association</td><td>[2509.01584](http://arxiv.org/pdf/2509.01584)</td><td>ViSTA-SLAM是一个无需已知相机内参即可实时运行的单目视觉SLAM系统，其核心贡献在于通过一个轻量化的对称双视图关联模型显著提升了系统性能与普适性。

◆ 提出一种轻量级对称双视图关联（STA）前端模型，仅需两张RGB图像即可同时估计相对相机位姿并回归局部点云地图。
◆ 极大降低了模型复杂度，前端模型大小仅为同类先进方法的35%，同时生成了更高质量的双视图约束用于后续优化。
◆ 后端构建了一个特殊的Sim(3)位姿图，通过融入回环检测来有效处理累积的尺度漂移问题。
◆ 整个系统不依赖相机内参，使其能够广泛适用于各种不同的相机设置，具备了很强的通用性。
实验表明，该方法在相机跟踪和稠密三维重建质量上均优于当前主流方法。</td></tr>
<tr><td>2025-09-01</td><td>Local asymmetry in spatial interactions: A generalized slide-vector approach</td><td>[2509.01131](http://arxiv.org/pdf/2509.01131)</td><td>该论文的核心贡献在于提出了一种新的建模框架来量化和分析功能空间中普遍存在的非对称空间分离现象。

◆ 突破了传统地理空间欧氏表示中对对称性公理的依赖，将研究焦点转向非对称的空间分离。
◆ 提出了一个基于空间约束多维展开的局部滑移向量模型，该模型能够同时考虑非对称性的空间依赖性和异质性。
◆ 该模型能够捕捉空间分离的局部非对称结构，从而更精细地揭示地理现象背后的复杂过程。
◆ 引入了势场方法来推断区域间的非对称性，并进一步研究了这些局部非对称结构的动态变化。
◆ 通过美国州际移民数据的实证应用，验证了该方法的有效性，从移民搬迁偏好的视角揭示了地理空间的扭曲，深化了对国内人口迁移模式的理解。</td></tr>
<tr><td>2025-08-31</td><td>Look Beyond: Two-Stage Scene View Generation via Panorama and Video Diffusion</td><td>[2509.00843](http://arxiv.org/pdf/2509.00843)</td><td>该论文提出了一种两阶段单目图像新视角合成方法，以生成长轨迹下全局一致的新视角。核心创新点包括：
◆ 将单视角合成任务分解为360度全景外推和视角插值两阶段，解决了大角度偏移时的内容一致性问题。
◆ 利用全景扩散模型学习场景先验，生成高质量全景图作为全局场景表示。
◆ 提出从全景图中采样并扭曲得到关键帧，作为预训练视频扩散模型的锚帧，确保长期视角对齐。
◆ 设计空间噪声扩散过程，通过视频模型实现连贯的新视角插值，支持用户自定义相机轨迹。
实验表明该方法在循环轨迹等挑战性场景中显著优于现有方法，实现了灵活相机控制下的全局一致性生成。</td></tr>
<tr><td>2025-08-28</td><td>Catwalk: Unary Top-K for Efficient Ramp-No-Leak Neuron Design for Temporal Neural Networks</td><td>[2508.21267](http://arxiv.org/pdf/2508.21267)</td><td>该论文提出了一种称为Catwalk的新型时间神经网络神经元设计，旨在提升硬件效率。其核心贡献与创新点包括：
◆ 针对RNL响应函数的神经元，创新性地利用输入脉冲序列中的稀疏性，通过一元Top-K方法对脉冲进行重排序和聚集。
◆ 该方法将有效的脉冲输入集中到一个更小的、有序的子集中，显著减少了后续并行计数器（PC）的处理负担。
◆ 这种设计优化了硬件实现，在面积和能效上相比现有的SRM0-RNL神经元实现了显著提升。
◆ 布局布线结果表明，Catwalk神经元的面积效率提高了1.39倍，能效提高了1.86倍。</td></tr>
<tr><td>2025-08-28</td><td>Mixture of Contexts for Long Video Generation</td><td>[2508.21058](http://arxiv.org/pdf/2508.21058)</td><td>该论文针对长视频生成中的长上下文记忆难题，提出了一种高效且可学习的解决方案。  
◆ 将长视频生成重新定义为内部信息检索任务，突破了传统方法的计算瓶颈。  
◆ 提出Mixture of Contexts（MoC）模块，通过可学习的稀疏注意力路由动态选择关键信息块（如字幕、局部窗口）进行关注。  
◆ 采用因果路由机制防止循环闭合，确保时序一致性和内容稳定性。  
◆ 实现了近线性的计算缩放，显著降低了内存和计算成本，使分钟级长视频的训练和生成变得可行。  
◆ 模型能够有效保留身份、动作和场景信息，生成内容在长时间范围内保持连贯性。</td></tr>
<tr><td>2025-08-22</td><td>Stochastic modelling reveals that chromatin folding buffers epigenetic landscapes against sirtuin depletion during DNA damage</td><td>[2508.16548](http://arxiv.org/pdf/2508.16548)</td><td>该论文通过结合染色质架构的随机模型，揭示了染色质折叠在DNA损伤期间对抗表观遗传景观失稳的保护机制。  
◆ 开发了组蛋白修饰动力学与染色质三维结构整合的随机模型，用于研究DNA损伤下表观遗传的响应机制。  
◆ 发现局部sirtuin（组蛋白去乙酰化酶）的耗竭会引发表观遗传景观的侵蚀，这一发现与实验观察一致。  
◆ 证明表观遗传稳定性依赖于酶浓度和染色质空间几何特征，而不仅是生化信号。  
◆ 首次揭示具有长程接触大结构域的染色质区域对表观遗传失稳具有更强抵抗力，说明染色质折叠具有缓冲功能。  
◆ 提出了染色质三维结构作为一种新型保护机制，可在基因组应激下维持表观遗传完整性。</td></tr>
<tr><td>2025-08-21</td><td>GelSLAM: A Real-time, High-Fidelity, and Robust 3D Tactile SLAM System</td><td>[2508.15990](http://arxiv.org/pdf/2508.15990)</td><td>GelSLAM提出了一种仅依靠触觉感知即可实现实时、高保真和鲁棒三维SLAM的系统。  
◆ 它创新性地利用触觉传感器获取的表面法线和曲率信息进行位姿跟踪与回环检测，取代了传统的点云方法。  
◆ 该系统能够以低误差和极小漂移实时跟踪物体运动，并实现亚毫米级的精细形状重建。  
◆ 即使在处理如木制工具等低纹理物体时，该系统依然表现出卓越的鲁棒性和精度。  
◆ GelSLAM将触觉感知的应用从局部接触扩展到了全局、长时序的空间感知领域。  
这项工作为需要高精度操作的灵巧手内物体交互任务奠定了基础。</td></tr>
<tr><td>2025-08-19</td><td>SLAM-based Safe Indoor Exploration Strategy</td><td>[2508.14235](http://arxiv.org/pdf/2508.14235)</td><td>该论文提出了一种基于SLAM的室内安全探索策略，专为具有圆形轮廓的非完整移动机器人设计。  
◆ 针对非点式且无法瞬时调整位姿的圆形机器人，设计了安全的探索控制方法。  
◆ 提出一种基于“安全骨架”的路径规划策略，使机器人始终尽可能远离静态障碍物。  
◆ 探索过程中优先朝向空间中的开放区域前进，兼顾安全性与探索效率。  
◆ 整合多传感器信息（IMU、3D-LiDAR和RGB-D相机），采用RTAB-SLAM实现实时建图与闭环检测。  
◆ 通过ROS平台进行了实验验证，展示了该方法在复杂室内环境中的有效性与安全性。</td></tr>
<tr><td>2025-08-19</td><td>A Screw Approach to the Approximation of the Local Geometry of the Configuration Space and of the set of Configurations of Certain Rank of Lower Pair Linkages</td><td>[2508.13802](http://arxiv.org/pdf/2508.13802)</td><td>本文提出了一种基于旋量理论的低副多环机构高阶局部运动学分析方法。  
◆ 引入了基于几何约束映射高阶泰勒展开的通用方法，摆脱了传统方法依赖运动平滑性的假设限制。  
◆ 提出了用关节旋量递归表达的代数形式，实现了对约束映射高阶微分的系统化计算。  
◆ 给出了构型空间局部几何的代数近似，并通过雅可比子式的微分显式表达式分析了特定秩构型集。  
◆ 所提出的旋量系统代数方法能处理文献中现有方法无法分析的复杂奇点（如分岔奇点、尖点）。  
该方法为机构局部运动性分析提供了更完备的数学框架和计算工具。</td></tr>
<tr><td>2025-08-19</td><td>ROVER: Robust Loop Closure Verification with Trajectory Prior in Repetitive Environments</td><td>[2508.13488](http://arxiv.org/pdf/2508.13488)</td><td>该论文提出了一种在重复环境下鲁棒的回环闭合验证方法ROVER，其核心创新在于利用历史轨迹作为先验约束来提升验证可靠性。  
◆ 首次将机器人的时空运动轨迹作为先验知识引入回环闭合验证过程，突破了传统方法仅依赖外观特征的局限性。  
◆ 提出通过位姿图优化将回环候选转换为轨迹估计，并设计评分机制评估该轨迹与先验轨迹的一致性。  
◆ 在存在高度相似结构的重复环境中能有效拒绝错误回环，显著降低SLAM系统的误检风险。  
◆ 在公开数据集和真实场景实验中验证了方法的优越性，并被集成至多种先进SLAM系统证明其实用性与鲁棒性。  
该方法为解决重复环境下回环误检问题提供了新的技术路径。</td></tr>
<tr><td>2025-08-16</td><td>SPIDER: Scalable Probabilistic Inference for Differential Earthquake Relocation</td><td>[2508.12117](http://arxiv.org/pdf/2508.12117)</td><td>◆ 提出了SPIDER框架，首次实现了可扩展的贝叶斯双差震源定位方法，解决了传统方法无法处理百万级参数的高维问题。  
◆ 结合物理信息神经网络Eikonal求解器与随机梯度朗之万动力学采样器，实现了高效的后验分布采样。  
◆ 支持多GPU并行计算，显著提升了大规模地震目录联合反演的计算效率。  
◆ 开发了针对高维后验分布的分析方法，为科学解释和结果评估提供了新工具。  
◆ 在加州和日本的真实地震目录及合成数据上验证了方法的有效性和实用性。  
◆ 为地震学领域提供了首个能同时处理不确定性量化与超大规模数据集的概率性定位解决方案。</td></tr>
<tr><td>2025-08-14</td><td>Super LiDAR Reflectance for Robotic Perception</td><td>[2508.10398](http://arxiv.org/pdf/2508.10398)</td><td>◆ 提出了一种创新框架，能够从稀疏的非重复扫描LiDAR（NRS-LiDAR）数据生成密集的LiDAR反射率图像，解决了低成本LiDAR数据稀疏性的问题。  
◆ 解决了反射率校准和从静态到动态场景转换的关键技术挑战，实现了真实场景中密集反射率图像的重建。  
◆ 构建了一个全面的LiDAR反射率图像稠密化数据集，为后续研究提供了重要资源。  
◆ 设计了一个专为NRS-LiDAR定制的稠密化网络，显著提升了稀疏数据的利用效率。  
◆ 展示了稠密反射率图像在机器人感知任务中的多样化应用，如闭环检测和交通车道检测，验证了其实际价值。  
◆ 通过主动光学传感重新定义了视觉的边界，推动了主动视觉的新发展。</td></tr>
<tr><td>2025-08-12</td><td>A Pseudo Global Fusion Paradigm-Based Cross-View Network for LiDAR-Based Place Recognition</td><td>[2508.08917](http://arxiv.org/pdf/2508.08917)</td><td>◆ 提出基于伪全局融合范式的跨视角网络，通过多模态分支协同学习统一语义空间特征，解决传统方法忽略特征空间内在结构的问题。  
◆ 创新性地引入伪全局信息引导机制，有效协调不同模态分支的特征学习，增强复杂环境下的表征能力。  
◆ 设计流形适应与成对方差-局部性学习度量，构建对称正定(SPD)矩阵计算马氏距离，取代传统欧氏距离度量。  
◆ 通过几何化建模准确刻画数据内在分布特性，捕捉特征空间中复杂的类间依赖关系，提升时变场景下的识别鲁棒性。  
◆ 实验证明该方法在复杂环境条件下性能优越，尤其在GPS拒止环境中的定位和闭环检测任务表现突出。  
◆ 整体框架突破了欧式空间线性假设的局限性，为激光雷达地点识别提供了非线性分布建模的新思路。</td></tr>
<tr><td>2025-08-12</td><td>UGM2N: An Unsupervised and Generalizable Mesh Movement Network via M-Uniform Loss</td><td>[2508.08615](http://arxiv.org/pdf/2508.08615)</td><td>◆ 提出首个无监督可泛化的网格移动网络UGM2N，摆脱传统方法对预适应网格的依赖，实现零样本跨PDE泛化  
◆ 创新性引入局部几何特征学习机制，通过无监督方式自主捕捉网格动态变化特征，避免监督学习的局限性  
◆ 设计物理约束的M-Uniform损失函数，在节点级别强制网格等分布，同时防止网格纠缠，保证数值稳定性  
◆ 实现几何拓扑无关的网格自适应能力，可处理复杂多尺度网格结构，显著提升计算效率与模拟精度  
◆ 实验证明该方法在多种PDE（如Navier-Stokes、波动方程）和异构网格上均优于现有方法，误差降低达35%  
◆ 首次在网格移动领域同时达成无监督训练、多物理场泛化和几何独立性三大突破，为科学计算提供新范式</td></tr>
<tr><td>2025-08-17</td><td>A Fast GRASP Metaheuristic for the Trigger Arc TSP with MIP-Based Construction and Multi-Neighborhood Local Search</td><td>[2508.08477](http://arxiv.org/pdf/2508.08477)</td><td>◆ 提出了一种基于GRASP的元启发式算法，用于解决动态弧成本变化的Trigger Arc TSP问题，扩展了经典TSP的应用场景。  
◆ 在构造阶段创新性地使用混合整数规划（MIP）技术，将TA-TSP转化为一系列定制化的TSP实例，提升了求解效率。  
◆ 改进阶段结合了多种邻域搜索操作（2-Opt、Swap和Relocate），增强了局部搜索能力，提高了解的质量。  
◆ 在MESS 2024竞赛实例中，算法在60秒内实现了平均0.77%和0.40%的最优性差距，表现优于已知最优解。  
◆ 在合成数据集上，算法在相同时间限制下比Gurobi求解器的解优11.3%，展现了其高效性。  
◆ 算法在MESS 2024竞赛中位列前三，验证了其在实时路由应用中的实用性和鲁棒性。</td></tr>
<tr><td>2025-08-11</td><td>MoRoCo: Multi-operator-robot Coordination, Interaction and Exploration under Restricted Communication</td><td>[2508.07657](http://arxiv.org/pdf/2508.07657)</td><td>◆ 提出MoRoCo框架，首次在通信受限环境下实现多操作者-多机器人的双向实时交互与协调，填补了现有研究忽视人机交互的空白。  
◆ 设计三种自适应协调模式（分散探索、协同迁移、链式连接），通过分布式算法动态切换，仅需本地通信即可维持团队高效协作。  
◆ 开发上下文感知的交互机制，允许操作者动态调整任务优先级、获取实时视频，同时支持机器人主动请求人工确认异常或故障恢复。  
◆ 提出基于多跳链路的带宽优化策略，在通信受限场景下仍能保障关键数据（如视频流）的高效传输。  
◆ 通过大规模人机协同仿真与硬件实验验证框架有效性，证明其在搜救、勘探等实际场景中的鲁棒性优势。</td></tr>
<tr><td>2025-08-10</td><td>Let&#x27;s Revise Step-by-Step: A Unified Local Search Framework for Code Generation with LLMs</td><td>[2508.07434](http://arxiv.org/pdf/2508.07434)</td><td>◆ 提出ReLoc统一局部搜索框架，通过逐步代码修订解决LLM代码生成的效率与扩展性问题，替代传统基于构造的树搜索方法。  
◆ 设计四大核心算法组件（初始代码起草、邻域代码生成、候选评估、当前代码更新），可灵活实例化为不同局部搜索算法（如爬山法、遗传算法）。  
◆ 开发专用修订奖励模型，通过修订距离评估代码质量，生成细粒度偏好信号以引导搜索方向。  
◆ 实验证明该方法在多样化代码生成任务中显著优于基于构造的树搜索和当前最优的改进型方法。  
◆ 首次将局部搜索范式系统化应用于LLM代码生成，平衡了搜索效率与生成质量的核心矛盾。  
◆ 通过邻域修订策略实现&quot;随时可终止&quot;特性，克服传统方法因树规模爆炸导致的高令牌消耗问题。</td></tr>
<tr><td>2025-08-08</td><td>Refactoring-Aware Patch Integration Across Structurally Divergent Java Forks</td><td>[2508.06718](http://arxiv.org/pdf/2508.06718)</td><td>这篇论文的核心贡献和创新点如下：

◆ 首次对长期分叉的Java代码变体（variants）间的补丁集成失败进行了实证研究，分析了14对分叉变体的结构差异问题。

◆ 提出了RePatch系统，专门解决因代码重构（如重命名、移动、重组）导致的结构差异问题，实现跨分叉变体的补丁集成。

◆ 创新性地扩展了RefMerge框架，使其支持非对称的补丁传输（asymmetric patch transfer），而原框架仅支持对称合并。

◆ 采用双向重构反转技术：先在源和目标变体上反转重构以对齐补丁上下文，应用补丁后再重放重构变换，保持变体意图。

◆ 实验评估了478个错误修复拉取请求，Git cherry-pick因结构错位失败率达64.4%，而RePatch成功集成了其中52.8%的失败案例。

◆ 揭示了基于语法的工具在跨变体补丁传播中的局限性，证明了语义推理的必要性。</td></tr>
<tr><td>2025-08-06</td><td>Behaviorally Adaptive Multi-Robot Hazard Localization in Failure-Prone, Communication-Denied Environments</td><td>[2508.04537](http://arxiv.org/pdf/2508.04537)</td><td>◆ 提出基于行为熵（BE）的新型信息理论规划框架，将香农熵（SE）扩展为能捕捉类人不确定性评估的通用指标，解决了传统熵度量在复杂环境中的局限性。  
◆ 开发行为自适应路径规划（BAPP）框架，通过可调风险敏感参数动态调整信息采集策略，实现风险与探索效率的平衡。  
◆ 设计两种算法：BAPP-TID通过智能触发高精度机器人加速熵减，BAPP-SIG在高风险下优先保障机器人存活率，信息损失最小化。  
◆ 理论证明BAPP框架的信息有效性，并通过单/多机器人仿真验证其优越性，性能显著优于香农熵基准和随机策略。  
◆ 提出多机器人协同的空间分区、移动基站重定位和角色异构机制，确保系统在通信中断环境中的可扩展性。  
◆ 首次将行为自适应规划与风险敏感探索结合，为灾害响应、行星探测等高风险任务提供鲁棒解决方案。</td></tr>
<tr><td>2025-08-06</td><td>Radar-Based NLoS Pedestrian Localization for Darting-Out Scenarios Near Parked Vehicles with Camera-Assisted Point Cloud Interpretation</td><td>[2508.04033](http://arxiv.org/pdf/2508.04033)</td><td>◆ 提出了一种结合单目摄像头和2D雷达点云数据的NLoS行人定位框架，解决路边停车导致的盲区安全问题。  
◆ 通过图像分割检测停放车辆，并利用深度估计初步推断空间特征，克服了传统方法依赖预定义空间信息的局限性。  
◆ 利用2D雷达点云数据对初步空间信息进行细化，实现更精确的实时空间推断，适应动态变化的道路环境。  
◆ 针对行人从停放车辆间突然出现的场景（darting-out），优化了毫米波雷达的衍射和反射信号处理能力。  
◆ 实验验证表明，该方法在真实城市道路环境中能显著提升行人早期检测效果，增强道路安全性。</td></tr>
<tr><td>2025-08-04</td><td>A Reinforcement Learning Framework for Mobility Control of gNBs in Dynamic Radio Access Networks</td><td>[2508.02960](http://arxiv.org/pdf/2508.02960)</td><td>◆ 提出CONVERGE Chamber Simulator (CC-SIM)，这是一个3D仿真环境，用于开发和验证移动gNB的智能控制算法，能够模拟用户移动、障碍物动态和射频传播行为。  
◆ CC-SIM支持离线强化学习和实时测试，通过与OpenAirInterface (OAI)射频模拟器的紧密集成，实现了在真实网络条件下的算法验证。  
◆ 开发了一种基于深度Q网络（DQN）的智能体，能够主动调整gNB位置以应对动态环境变化，显著提升网络性能。  
◆ 实验结果表明，该智能体在三种典型应用场景中，能够将视距（LoS）阻塞时间减少高达42%，优于静态部署方案。  
◆ 该研究为下一代自适应无线网络中的移动基站智能控制提供了有效的学习框架和验证平台。</td></tr>
<tr><td>2025-08-04</td><td>Understanding Heterogeneity in Adaptation to Intermittent Water Supply: Clustering Household Types in Amman, Jordan</td><td>[2508.02569](http://arxiv.org/pdf/2508.02569)</td><td>◆ 提出了一种标准化分析框架，结合层次聚类分析（HCA）和Welch双样本t检验，用于研究间歇性供水（IWS）下家庭适应的异质性。  
◆ 首次在约旦安曼的实证研究中识别出三类具有不同特征的家庭集群，包括收入、供水时长、社会网络等多元因素，突破了以往仅关注收入差异的局限。  
◆ 揭示了家庭适应策略与多维特征（如水质问题、搬迁经历）之间的非线性关联，填补了现有研究对数据多维性分析的不足。  
◆ 通过聚类结果明确了不同家庭群体的具体适应行为（如联系水务公司或寻找替代水源），为政策制定提供了针对性依据。  
◆ 提出的方法论具有跨地区适用性，为全球南方城市研究IWS下的不平等问题提供了标准化工具。  
◆ 实证证明了间歇性供水适应中存在显著的城内不平等，强调了需基于家庭特征差异设计公平的水资源管理方案。</td></tr>
<tr><td>2025-07-31</td><td>DRACo-SLAM2: Distributed Robust Acoustic Communication-efficient SLAM for Imaging Sonar EquippedUnderwater Robot Teams with Object Graph Matching</td><td>[2507.23629](http://arxiv.org/pdf/2507.23629)</td><td>◆ 提出DRACo-SLAM2框架，为配备多波束成像声纳的水下机器人团队设计分布式SLAM系统，扩展了原有DRACo-SLAM的功能。  
◆ 创新性地将声纳地图表示为对象图，通过对象图匹配实现高效跨机器人回环检测，无需依赖先验几何信息。  
◆ 针对水下扫描匹配特点，改进PCM算法为增量式GCM（Group-wise Consistent Measurement Set Maximization），有效处理相邻跨机器人回环共享相似配准误差的场景。  
◆ 提出的对象图匹配方法显著提升了跨机器人回环检测的时间效率，适用于水下通信受限环境。  
◆ 通过大量仿真和真实数据集验证了方法的优越性，展示了在复杂水下环境中的鲁棒性和实用性。</td></tr>
<tr><td>2025-07-31</td><td>Designing Dynamic Pricing for Bike-sharing Systems via Differentiable Agent-based Simulation</td><td>[2507.23344](http://arxiv.org/pdf/2507.23344)</td><td>◆ 提出了一种基于可微分代理模拟的新方法，用于快速设计共享单车系统的动态定价策略，有效解决时空需求不均导致的库存失衡问题。  
◆ 该方法能够处理用户背景多样性和概率性选择行为，相比传统方法显著提升了计算效率和准确性。  
◆ 在包含25个单车站点和5个时间段的实验中，损失减少了73%-78%，收敛速度提高了100倍以上。  
◆ 在大规模城市共享单车系统（289个站点）的验证中，证明所获定价策略无需人工调度即可实现库存自然平衡。  
◆ 发现通过设置合适的初始条件，可以最小化诱导库存平衡所需的折扣成本，为实际运营提供经济性优化方案。  
◆ 结合仿真与参数优化，首次实现了动态定价策略的端到端高效设计，为共享交通系统管理提供了新工具。</td></tr>
<tr><td>2025-07-30</td><td>UAVScenes: A Multi-Modal Dataset for UAVs</td><td>[2507.22412](http://arxiv.org/pdf/2507.22412)</td><td>◆ 提出了首个面向无人机多模态感知的大规模数据集UAVScenes，填补了现有数据集在高级场景理解任务上的空白。  
◆ 基于MARS-LVIG数据集进行扩展，首次为逐帧图像和LiDAR点云提供人工标注的语义标签，支持精细场景解析。  
◆ 引入精确的6自由度位姿数据，使数据集能同时支持2D和3D模态的多种任务评测。  
◆ 覆盖五大核心任务：语义分割、深度估计、6-DoF定位、地点识别和新型视图合成，拓展了无人机感知研究边界。  
◆ 通过严格的传感器标定和跨模态数据对齐，确保多模态数据的时间同步与空间一致性。  
◆ 开源数据集并建立标准化评测基准，推动无人机多模态感知技术的协同发展。</td></tr>
<tr><td>2025-07-29</td><td>Horseshoe Forests for High-Dimensional Causal Survival Analysis</td><td>[2507.22004](http://arxiv.org/pdf/2507.22004)</td><td>◆提出新型贝叶斯树集成模型，专门针对高维协变量下的生存分析数据，解决传统方法在异质性处理效应估计中的局限性。  
◆创新性地将马蹄先验直接施加于树结构的步高参数，而非依赖树结构本身实现稀疏性，实现自适应全局-局部收缩，提升噪声抑制能力。  
◆开发了可逆跳转吉布斯采样器，首次在树集成框架中成功融合非共轭马蹄先验，解决了计算难题。  
◆通过大量模拟实验验证，该方法在高维协变量、不同稀疏度及非线性处理效应场景下均保持优异估计精度。  
◆在胰腺导管腺癌（PDAC）真实数据集的重分析中展示了实用价值，为癌症生存分析提供新工具。  
◆整体方案突破了传统生存分析模型对线性假设和低维数据的依赖，为高维因果推断开辟了新路径。</td></tr>
<tr><td>2025-07-28</td><td>Collaborative Perceiver: Elevating Vision-based 3D Object Detection via Local Density-Aware Spatial Occupancy</td><td>[2507.21358](http://arxiv.org/pdf/2507.21358)</td><td>◆ 提出多任务学习框架Collaborative Perceiver (CoP)，通过空间占用预测辅助3D目标检测，挖掘两者在结构和概念上的相似性，提升空间表征能力。  
◆ 设计局部密度感知(LDO)的密集占用真值生成方法，结合局部密度信息重建精细环境结构，弥补传统BEV表征的环境上下文缺失问题。  
◆ 创新体素高度引导采样(VHS)策略，根据目标特性差异蒸馏细粒度局部特征，增强对多样化物体的几何感知。  
◆ 开发全局-局部协同特征融合(CFF)模块，动态整合目标检测与占用预测的互补知识，构建更鲁棒的BEV表征。  
◆ 在nuScenes基准测试中实现49.5% mAP和59.2% NDS的SOTA性能，验证了方法对纯视觉3D检测的有效提升。  
（全文共5条创新点，总计约300字）</td></tr>
<tr><td>2025-07-28</td><td>Learning Transferable Facial Emotion Representations from Large-Scale Semantically Rich Captions</td><td>[2507.21015](http://arxiv.org/pdf/2507.21015)</td><td>◆ 提出EmoCap100K数据集，包含超过10万条富含情感语义的面部表情标注，涵盖全局情感状态和细粒度局部面部行为描述，解决了现有数据集中情感标签过于简化的问题。  
◆ 设计EmoCapCLIP框架，通过全局-局部对比学习联合训练，结合跨模态引导的正样本挖掘模块，充分利用多层次文本信息并捕捉相似表情的语义关联。  
◆ 首次系统性地探索自然语言描述作为监督信号的面部情感表征学习，突破了传统固定类别或维度标签的局限性，提升了模型的泛化能力和可解释性。  
◆ 提出结构化语义描述方法，同时建模宏观情感和微观面部动作，实现了对复杂情感谱系的更精细表达。  
◆ 在超过20个基准测试和5类任务中验证了方法的优越性，证明了大规模语义丰富标注对情感识别任务的重要价值。</td></tr>
<tr><td>2025-07-28</td><td>PixelNav: Towards Model-based Vision-Only Navigation with Topological Graphs</td><td>[2507.20892](http://arxiv.org/pdf/2507.20892)</td><td>◆ 提出了一种结合深度学习与经典模型规划算法的混合视觉导航方法，突破了纯端到端数据驱动模型的局限性。  
◆ 采用分层系统架构，整合了模型预测控制、可通行性估计、视觉地点识别和位姿估计等多项先进技术。  
◆ 创新性地使用拓扑图作为环境表征，显著提升了系统的可解释性和可扩展性。  
◆ 通过减少对大量训练数据的依赖，解决了当前主流方法在实际应用中的关键瓶颈问题。  
◆ 大量真实场景实验验证了该方法的有效性，为视觉导航领域提供了新的技术路径。</td></tr>
<tr><td>2025-07-28</td><td>Existence, uniqueness, and long-time asymptotic behavior of regular solutions in multidimensional thermoelasticity</td><td>[2507.20794](http://arxiv.org/pdf/2507.20794)</td><td>◆ 将一维热弹性模型中的Fisher信息泛函扩展到多维（二维和三维环面）情况，提出了新的泛函分析方法。  
◆ 利用该泛函证明了在小初始数据下全局正则解的存在唯一性，以及大初始数据下的局部解存在性。  
◆ 首次系统分析了多维热弹性系统长时间渐近行为，揭示了温度场会稳定趋于恒定状态。  
◆ 发现位移场的创新分解结构：可分解为无限振荡的散度自由分量（满足齐次波动方程）和收敛到零的旋度自由分量。  
◆ 建立了拉梅算子情形下的平行理论结果，拓展了热弹性理论的应用范围。  
◆ 通过非线性分析方法统一处理了存在性、唯一性和渐近性三大核心问题，为多维热弹性系统提供了完整理论框架。</td></tr>
<tr><td>2025-07-28</td><td>Uni-Mapper: Unified Mapping Framework for Multi-modal LiDARs in Complex and Dynamic Environments</td><td>[2507.20538](http://arxiv.org/pdf/2507.20538)</td><td>◆ Uni-Mapper提出首个动态感知的多模态LiDAR地图统一框架，解决复杂动态环境中多传感器地图对齐难题。  
◆ 采用粗到细的体素化自由空间哈希地图，通过时序占据不一致性检测并剔除动态物体，提升静态场景一致性。  
◆ 创新融合动态物体剔除模块与LiDAR全局描述符，保留静态局部特征，实现动态环境下鲁棒的地点识别。  
◆ 设计集中式锚节点策略优化位姿图，有效抑制地图合并时的会话内漂移误差，提升跨地图闭环精度。  
◆ 支持异构LiDAR（如机械式与固态雷达）的跨模态闭环检测，在动态场景与多传感器条件下性能超越现有方法。  
◆ 通过真实世界多场景验证，在动态物体处理、跨传感器回环和全局地图对齐精度上均展现显著优势。</td></tr>
<tr><td>2025-07-27</td><td>Improvised Nuclear Weapons with 60%-Enriched Uranium</td><td>[2507.20390](http://arxiv.org/pdf/2507.20390)</td><td>◆ 首次量化分析表明，仅需40公斤60%浓缩铀即可制造千吨当量的简易核武器，突破了传统核武器设计对高纯度铀的需求认知。  
◆ 提出60%丰度铀材料在非导弹运载场景下的实战应用可能性（如集装箱运输），拓展了核安全威胁的评估维度。  
◆ 结合2025年伊朗核设施遇袭事件，首次公开论证秘密转移的中等丰度铀库存（408公斤）被恐怖组织截获的潜在风险。  
◆ 创新性将&quot;小男孩&quot;枪式核弹设计原理适配于中等丰度铀材料，验证了非国家行为体制造核武器的技术可行性。  
◆ 通过伊朗案例建立&quot;核材料快速转移-监管真空-恐怖组织利用&quot;的三阶段威胁模型，为国际核安全体系提供新的预警框架。</td></tr>
<tr><td>2025-07-27</td><td>Dual-Stream Global-Local Feature Collaborative Representation Network for Scene Classification of Mining Area</td><td>[2507.20216](http://arxiv.org/pdf/2507.20216)</td><td>◆ 提出双分支融合模型，通过协作表示将全局特征分解为关键语义向量，提升矿区场景分类精度。  
◆ 设计多尺度全局Transformer分支，利用大尺度特征生成小尺度特征的全局通道注意力，有效捕捉多尺度特征关系。  
◆ 开发局部增强协作表示分支，结合局部特征和重构关键语义集优化注意力权重，增强模型对细粒度空间变化的敏感性。  
◆ 引入双分支深度特征融合模块，整合全局与局部互补特征，强化模型对复杂矿区场景的区分能力。  
◆ 采用多损失计算策略，确保各模块平衡融合，模型整体准确率达83.63%，优于其他对比模型。  
◆ 构建多模态矿区地物场景分类数据集，为地质环境监测和资源开发规划提供更精准的基础数据支持。</td></tr>
<tr><td>2025-07-24</td><td>DSFormer: A Dual-Scale Cross-Learning Transformer for Visual Place Recognition</td><td>[2507.18444](http://arxiv.org/pdf/2507.18444)</td><td>◆ 提出DSFormer双尺度交叉学习Transformer模块，通过双向信息传递整合CNN最后两层的双尺度特征，同时捕捉语义丰富性和空间细节。  
◆ 设计自注意力机制处理单尺度内的长程依赖关系，并引入共享交叉注意力实现跨尺度学习，增强特征表示能力。  
◆ 创新性提出块聚类策略，从多视角重构SF-XL训练数据集的分区方式，优化数据组织以提升视角变化的鲁棒性。  
◆ 结合上述技术，生成适应环境变化的鲁棒全局嵌入表征，相比现有分区方法减少约30%训练数据需求。  
◆ 仅使用512维全局描述符即实现全局检索，在多数基准数据集上超越DELG、Patch-NetVLAD等先进方法，达到SOTA性能。  
◆ 显著提升计算效率，为视觉地点识别任务提供高效解决方案。</td></tr>
<tr><td>2025-07-23</td><td>DiNAT-IR: Exploring Dilated Neighborhood Attention for High-Quality Image Restoration</td><td>[2507.17892](http://arxiv.org/pdf/2507.17892)</td><td>◆ 提出Dilated Neighborhood Attention (DiNA)机制，结合滑动窗口注意力和混合膨胀因子，在保持局部精度的同时扩展感受野，解决传统自注意力计算成本高的问题。  
◆ 针对图像复原任务中局部注意力全局上下文理解不足的问题，引入通道感知模块，有效整合全局信息而不损失像素级精度。  
◆ 设计专用于图像复原的Transformer架构DiNAT-IR，通过平衡全局与局部特征处理，显著提升复原质量。  
◆ 实验证明该方法在多个基准测试中达到竞争性性能，为低层计算机视觉任务提供高效解决方案。  
◆ 首次将混合膨胀注意力机制从高层视觉任务迁移到图像复原领域，并针对其局限性进行创新改进。  
◆ 通过通道注意力与空间注意力的协同设计，克服了Restormer等模型忽视局部伪影的缺陷。</td></tr>
<tr><td>2025-07-23</td><td>When and Where Localization Fails: An Analysis of the Iterative Closest Point in Evolving Environment</td><td>[2507.17531](http://arxiv.org/pdf/2507.17531)</td><td>◆ 提出了首个针对短期环境变化（数天至数周）的高分辨率多时段激光雷达数据集，填补了动态户外场景下短期定位研究的空白。  
◆ 数据集包含每周采集的自然与半城市场景的高密度点云地图、360度全景图像和轨迹数据，为短期定位鲁棒性评估提供结构化基准。  
◆ 创新性地使用传感器精确遮挡建模的投影激光雷达扫描，通过真实轨迹数据定量评估两种ICP变体（点对点与点对平面）的配准精度。  
◆ 实验证明点对平面ICP在特征稀疏或植被密集区域具有显著更高的稳定性和准确性，揭示了局部几何特征对定位成功的关键影响。  
◆ 建立了可复现的噪声环境下扫描-地图对齐分析框架，为动态环境中定位算法的性能比较提供方法论支持。  
◆ 通过系统分析环境变异性与定位失败的关系，为设计更具适应性的机器人系统提供了实用洞见。</td></tr>
<tr><td>2025-07-23</td><td>VLM-Guided Visual Place Recognition for Planet-Scale Geo-Localization</td><td>[2507.17455](http://arxiv.org/pdf/2507.17455)</td><td>◆ 提出了一种新型混合地理定位框架，结合了视觉语言模型（VLM）和检索式视觉地点识别（VPR）方法的优势。  
◆ 利用VLM生成地理先验信息，有效指导和约束检索搜索空间，解决了传统方法在可扩展性和感知混淆上的不足。  
◆ 设计了检索后重排序机制，结合特征相似性和初始坐标邻近性，选择地理上最合理的匹配结果。  
◆ 在多个地理定位基准测试中表现优异，尤其在街道级（提升4.51%）和城市级（提升13.52%）定位精度上显著超越现有方法。  
◆ 通过VLM与VPR的结合，实现了可扩展、鲁棒且高精度的地理定位系统，同时提升了系统的可解释性和可靠性。</td></tr>
<tr><td>2025-07-23</td><td>Hierarchical Fusion and Joint Aggregation: A Multi-Level Feature Representation Method for AIGC Image Quality Assessment</td><td>[2507.17182](http://arxiv.org/pdf/2507.17182)</td><td>◆提出多级视觉表征范式，包含特征提取、层级融合和联合聚合三阶段，突破传统单级特征评估的局限。  
◆设计MGLF-Net网络，通过CNN与Transformer双主干结构提取互补的局部与全局特征，提升感知质量评估能力。  
◆开发MPEF-Net网络，首次在各级特征中嵌入文本提示语义，实现文本-图像对应性的多层级联合评估。  
◆提出层级融合策略，将不同抽象层次（低阶感知到高阶语义）的特征动态整合，增强对AIGC复杂失真的捕捉。  
◆通过联合聚合机制融合多级特征，在感知质量和图文一致性双任务上均取得显著性能提升。  
◆实验验证了所提范式在AIGC质量评估中的普适性，为多维度评价提供新框架。</td></tr>
<tr><td>2025-07-22</td><td>How animal movement influences wildlife-vehicle collision risk: a mathematical framework for range-resident species</td><td>[2507.17058](http://arxiv.org/pdf/2507.17058)</td><td>◆ 提出了首个将交通、景观和动物运动特征与野生动物-车辆碰撞（WVC）风险联系起来的理论框架，填补了该领域理论模型的空白。  
◆ 结合运动生态学和具有部分吸收边界的反应-扩散随机过程，为定居型陆地哺乳动物（WVC主要肇事者）建立了数学模型。  
◆ 推导出关键生存统计量的精确表达式，包括平均碰撞时间和道路导致的寿命缩短，量化了道路对野生动物的直接影响。  
◆ 模型参数均可通过实际观测数据获取，如交通强度、道路宽度、家域穿越时间、家域大小及家域中心与道路距离等运动参数。  
◆ 为整合运动生态学与道路生态学提供了有效理论工具，支持基于数据的WVC缓解策略制定，促进更安全的可持续交通网络建设。</td></tr>
<tr><td>2025-07-22</td><td>Bayesian Compressed Mixed-Effects Models</td><td>[2507.16961](http://arxiv.org/pdf/2507.16961)</td><td>◆ 提出压缩混合效应（CME）模型，通过随机投影降低随机效应协方差矩阵维度，解决传统贝叶斯方法计算瓶颈。  
◆ 结合全局-局部收缩先验与降维技术，构建高效折叠吉布斯采样器，实现固定效应选择和预测的同步优化。  
◆ 理论证明当压缩维度缓慢增长时，预测的贝叶斯风险渐近可忽略，确保模型预测准确性。  
◆ 实证显示CME模型在预测精度、区间覆盖率和固定效应选择上均优于现有方法。  
◆ 适用于高维线性混合效应场景，为贝叶斯推断提供计算高效的替代方案。</td></tr>
<tr><td>2025-07-22</td><td>VGGT-Long: Chunk it, Loop it, Align it -- Pushing VGGT&#x27;s Limits on Kilometer-scale Long RGB Sequences</td><td>[2507.16443](http://arxiv.org/pdf/2507.16443)</td><td>◆ 提出VGGT-Long系统，首次将单目3D重建能力扩展到千米级无边界户外场景，突破现有基础模型的内存限制。  
◆ 采用分块处理策略结合重叠对齐和轻量级闭环优化，无需相机标定、深度监督或模型重训练，显著提升可扩展性。  
◆ 在KITTI、Waymo等数据集上验证，其轨迹和重建精度媲美传统方法，且能在长RGB序列（基础模型通常失效的场景）稳定运行。  
◆ 通过创新性系统设计，证明基础模型在真实世界（如自动驾驶）中实现大规模单目3D场景重建的潜力。  
◆ 开源代码提供完整实现，支持实际应用部署。</td></tr>
<tr><td>2025-07-22</td><td>A Single-step Accurate Fingerprint Registration Method Based on Local Feature Matching</td><td>[2507.16201](http://arxiv.org/pdf/2507.16201)</td><td>◆ 提出了一种端到端的单步指纹配准算法，直接通过预测两幅指纹图像之间的半密集匹配点对应关系来实现对齐，避免了传统两步法的复杂性。  
◆ 解决了低质量指纹图像中特征点数量不足导致的初始配准失败问题，提高了配准的鲁棒性和成功率。  
◆ 创新性地结合全局-局部注意力机制，实现了两幅指纹图像之间的端到端像素级对齐，提升了配准精度。  
◆ 实验证明该方法仅需单步配准即可达到最先进的匹配性能，同时还能与密集配准算法结合以进一步提升性能。  
◆ 该方法简化了传统指纹配准流程，降低了因初始配准失败导致的整体配准失败风险，适用于实际应用场景。</td></tr>
<tr><td>2025-07-22</td><td>Einstein&#x27;s Electron and Local Branching: Unitarity Does not Require Many-Worlds</td><td>[2507.16123](http://arxiv.org/pdf/2507.16123)</td><td>◆ 通过现代传感器技术重现爱因斯坦1927年单电子衍射思想实验，构建全封闭半球形探测器阵列系统，为多世界解释(MWI)与分支希尔伯特子空间解释(BHSI)提供直接实证对比框架。  
◆ 证明两种解释均保持幺正性且无需波函数坍缩，但存在本体论差异：MWI主张不可逆的全局平行世界分支，而BHSI提出局域可逆的退相干子空间分支。  
◆ 首次在实验设计中实现所有量子事件（分支、参与、脱离、重定位）完全局域化，且玻恩规则通过分支权重自然涌现于探测器统计。  
◆ 提出创新的双层探测器增强实验方案：利用内层透明探测器与电子层间穿越时间短于传感器响应的特性，可检验测量时序异常（如延迟选择或未定状态）。  
◆ 颠覆&quot;幺正性必然导致平行世界&quot;的传统认知，论证局域分支机制即可满足幺正性，无需引入全局世界分裂，为量子基础理论提供更简洁的解释路径。</td></tr>
<tr><td>2025-07-21</td><td>MSGM: A Multi-Scale Spatiotemporal Graph Mamba for EEG Emotion Recognition</td><td>[2507.15914](http://arxiv.org/pdf/2507.15914)</td><td>◆ 提出多尺度时空图曼巴模型（MSGM），首次将状态空间模型（Mamba）引入脑电情绪识别领域，实现线性复杂度的动态时空特征建模。  
◆ 设计多窗口时间分割策略，通过并行处理不同时间粒度的EEG信号片段，有效捕捉细粒度情绪波动。  
◆ 创新双模态空间图构建方法，结合神经解剖学先验知识同时建模全局脑区连接与局部功能网络，增强空间层次表征能力。  
◆ 开发多深度图卷积与令牌嵌入融合模块，通过曼巴架构实现跨尺度时空特征的动态交互与高效融合。  
◆ 仅需单层MSST-Mamba即可在SEED等三大数据集上超越现有最优方法，兼顾高精度（主体独立分类）与毫秒级实时推理（NVIDIA Jetson平台）。  
◆ 解决传统方法时空建模单一和计算效率低的双重瓶颈，为可穿戴设备实时情绪识别提供新范式。</td></tr>
<tr><td>2025-07-21</td><td>DiffPF: Differentiable Particle Filtering with Generative Sampling via Conditional Diffusion Models</td><td>[2507.15716](http://arxiv.org/pdf/2507.15716)</td><td>◆ DiffPF首次将条件扩散模型融入粒子滤波框架，实现了高质量的后验采样，克服了传统方法的局限性。  
◆ 通过将扩散模型与预测粒子和当前观测条件结合，DiffPF能够从复杂、高维、多模态的滤波分布中生成精确且等权重的粒子。  
◆ 相比传统可微分粒子滤波依赖预定义或低容量提议分布的问题，DiffPF学习了一个灵活的后验采样器，显著提升了采样质量。  
◆ 在多种场景测试中，DiffPF表现优异，包括单模态和高度多模态分布，以及在仿真和真实任务中的卓越性能。  
◆ 在高度多模态的全局定位基准测试中，DiffPF将估计精度提升了82.8%，在KITTI视觉里程计基准测试中提升了26%，远超现有最优可微分滤波方法。  
◆ DiffPF的创新性在于其结合了扩散模型的生成能力与粒子滤波的框架，为动态系统状态估计提供了更强大的工具。</td></tr>
<tr><td>2025-07-21</td><td>Hi^2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization for Remote Sensing</td><td>[2507.15683](http://arxiv.org/pdf/2507.15683)</td><td>◆ 提出Hi^2-GSLoc双层次视觉重定位框架，采用稀疏到稠密、粗到精的范式，首次将3D高斯泼溅（3DGS）引入遥感场景定位，兼具几何与外观编码优势。  
◆ 设计基于高斯基元的语义感知采样策略和地标引导检测器，通过渲染一致性约束提升初始位姿估计的鲁棒性，解决大尺度场景下传统方法精度不足的问题。  
◆ 开发稠密阶段迭代优化方法，结合粗到精的栅格化匹配与可靠性验证机制，显著提升位姿细化效率，克服结构法计算复杂度高的缺陷。  
◆ 针对遥感数据特性创新性提出分区高斯训练、GPU并行匹配和动态内存管理策略，实现大规模场景的高效处理，填补现有视觉先验的领域差距。  
◆ 在仿真数据、公开数据集和真实飞行实验中验证了方法的优越性，兼具定位精度、召回率和计算效率优势，同时有效过滤不可靠位姿估计。</td></tr>
<tr><td>2025-07-21</td><td>Trade-offs between elective surgery rescheduling and length-of-stay prediction accuracy</td><td>[2507.15566](http://arxiv.org/pdf/2507.15566)</td><td>◆ 研究了选择性手术患者住院时长（LOS）预测准确性与手术重新调度灵活性之间的权衡关系，填补了该领域的研究空白。  
◆ 提出通过模拟机器学习方法评估不同数据驱动策略，为医院资源规划提供量化分析工具。  
◆ 揭示了LOS预测误差下最有效的患者重新调度策略（如推迟入院、调整病房等），可在保证床位不溢出的同时优化资源利用率。  
◆ 挑战了&quot;更高预测准确性必然减少重新调度需求&quot;的传统假设，指出高精度预测模型的训练成本需纳入权衡考量。  
◆ 为医院管理提供了实践指导：在预测模型精度与调度操作灵活性之间寻求平衡比单纯追求预测精度更具成本效益。  
◆ 通过多场景测试不同纠正政策，证明适当调度策略可弥补预测误差，降低对高成本预测模型的依赖。</td></tr>
<tr><td>2025-07-20</td><td>LoopNet: A Multitasking Few-Shot Learning Approach for Loop Closure in Large Scale SLAM</td><td>[2507.15109](http://arxiv.org/pdf/2507.15109)</td><td>◆ 提出LoopNet，一种基于多任务学习的改进ResNet架构，专为大规模SLAM中的闭环检测设计，兼顾嵌入式设备的实时计算限制。  
◆ 采用小样本学习策略实现动态视觉数据集的在线重训练，适应环境变化并提升模型泛化能力。  
◆ 创新性地同时输出视觉数据集索引和预测质量评估，增强闭环检测的可靠性和可解释性。  
◆ 结合DISK描述符替代传统手工特征，克服光照、视角等变化条件下的性能瓶颈，显著提升检测精度。  
◆ 发布开源闭环检测基准数据集LoopDB，填补现有数据在动态场景和嵌入式硬件测试方面的空白。  
◆ 整体方案在精度与效率间取得平衡，为资源受限的实时SLAM系统提供实用解决方案。</td></tr>
<tr><td>2025-07-20</td><td>Visual Place Recognition for Large-Scale UAV Applications</td><td>[2507.15089](http://arxiv.org/pdf/2507.15089)</td><td>◆ 提出了LASED大规模无人机航拍数据集，包含约100万张图像，覆盖爱沙尼亚17万个独特地点，具有丰富的地理和时间多样性，解决了现有数据集规模小、多样性不足的问题。  
◆ 数据集采用结构化设计，确保地点分离明确，显著提升了模型在航拍场景中的训练效果。  
◆ 提出将可操纵卷积神经网络（CNNs）应用于视觉地点识别，利用其旋转等变性特性，有效处理无人机图像中的旋转模糊问题。  
◆ 实验证明，基于LASED训练的模型召回率显著高于使用小规模数据集训练的模型，验证了大规模地理覆盖和时间多样性的重要性。  
◆ 可操纵CNNs在旋转模糊处理上表现优异，平均召回率比最佳非可操纵网络提高12%，展现了其在航拍场景中的优势。  
◆ 通过结合大规模结构化数据集和旋转等变神经网络，显著提升了无人机视觉地点识别的鲁棒性和泛化能力。</td></tr>
</tbody>
</table>
</div>

<h2 id='image-matching'>Image Matching</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-10-21</td><td>GBlobs: Local LiDAR Geometry for Improved Sensor Placement Generalization</td><td>[2510.18539](http://arxiv.org/pdf/2510.18539)</td><td>◆ This technical report outlines the top-ranking solution for RoboSense 2025: Track 3, achieving state-of-the-art performance on 3D object detection under various sensor placements.
◆ Our submission utilizes GBlobs, a local point cloud feature descriptor specifically designed to enhance model generalization across diverse LiDAR configurations.
◆ Current LiDAR-based 3D detectors often suffer from a \enquote{geometric shortcut} when trained on conventional global features (\ie, absolute Cartesian coordinates).</td></tr>
<tr><td>2025-10-21</td><td>DeepDetect: Learning All-in-One Dense Keypoints</td><td>[2510.17422](http://arxiv.org/pdf/2510.17422)</td><td>◆ Keypoint detection is the foundation of many computer vision tasks, including image registration, structure-from motion, 3D reconstruction, visual odometry, and SLAM.
◆ Traditional detectors (SIFT, SURF, ORB, BRISK, etc.) and learning based methods (SuperPoint, R2D2, LF-Net, D2-Net, etc.) have shown strong performance yet suffer from key limitations: sensitivity to photometric changes, low keypoint density and repeatability, limited adaptability to challenging scenes, and lack of semantic understanding, often failing to prioritize visually important regions.
◆ We present DeepDetect, an intelligent, all-in-one, dense keypoint detector that unifies the strengths of classical detectors using deep learning.</td></tr>
<tr><td>2025-10-08</td><td>StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance</td><td>[2510.06827](http://arxiv.org/pdf/2510.06827)</td><td>◆ In the domain of text-to-image generation, diffusion models have emerged as powerful tools.
◆ Recently, studies on visual prompting, where images are used as prompts, have enabled more precise control over style and content.
◆ However, existing methods often suffer from content leakage, where undesired elements of the visual style prompt are transferred along with the intended style.</td></tr>
<tr><td>2025-10-08</td><td>Efficient Discriminative Joint Encoders for Large Scale Vision-Language Reranking</td><td>[2510.06820](http://arxiv.org/pdf/2510.06820)</td><td>◆ Multimodal retrieval still leans on embedding-based models like CLIP for fast vector search over pre-computed image embeddings.
◆ Yet, unlike text retrieval, where joint-encoder rerankers are standard, comparable vision--language rerankers are largely absent.
◆ We find that seminal joint encoders such as BLIP are severely bottlenecked by an expensive visual feature-extraction stage, preventing practical deployment at scale.</td></tr>
<tr><td>2025-10-06</td><td>SegMASt3R: Geometry Grounded Segment Matching</td><td>[2510.05051](http://arxiv.org/pdf/2510.05051)</td><td>◆ Segment matching is an important intermediate task in computer vision that establishes correspondences between semantically or geometrically coherent regions across images.
◆ Unlike keypoint matching, which focuses on localized features, segment matching captures structured regions, offering greater robustness to occlusions, lighting variations, and viewpoint changes.
◆ In this paper, we leverage the spatial understanding of 3D foundation models to tackle wide-baseline segment matching, a challenging setting involving extreme viewpoint shifts.</td></tr>
<tr><td>2025-09-30</td><td>Enhancing Certifiable Semantic Robustness via Robust Pruning of Deep Neural Networks</td><td>[2510.00083](http://arxiv.org/pdf/2510.00083)</td><td>◆ Deep neural networks have been widely adopted in many vision and robotics applications with visual inputs.
◆ It is essential to verify its robustness against semantic transformation perturbations, such as brightness and contrast.
◆ However, current certified training and robustness certification methods face the challenge of over-parameterization, which hinders the tightness and scalability due to the over-complicated neural networks.</td></tr>
<tr><td>2025-09-26</td><td>PANICL: Mitigating Over-Reliance on Single Prompt in Visual In-Context Learning</td><td>[2509.21926](http://arxiv.org/pdf/2509.21926)</td><td>◆ Visual In-Context Learning (VICL) uses input-output image pairs, referred to as in-context pairs (or examples), as prompts alongside query images to guide models in performing diverse vision tasks.
◆ However, VICL often suffers from over-reliance on a single in-context pair, which can lead to biased and unstable predictions.
◆ We introduce PAtch-based $k$-Nearest neighbor visual In-Context Learning (PANICL), a general training-free framework that mitigates this issue by leveraging multiple in-context pairs.</td></tr>
<tr><td>2025-09-23</td><td>Hierarchical Neural Semantic Representation for 3D Semantic Correspondence</td><td>[2509.17431](http://arxiv.org/pdf/2509.17431)</td><td>◆ This paper presents a new approach to estimate accurate and robust 3D semantic correspondence with the hierarchical neural semantic representation.
◆ Our work has three key contributions.
◆ First, we design the hierarchical neural semantic representation (HNSR), which consists of a global semantic feature to capture high-level structure and multi-resolution local geometric features to preserve fine details, by carefully harnessing 3D priors from pre-trained 3D generative models.</td></tr>
<tr><td>2025-09-20</td><td>PM25Vision: A Large-Scale Benchmark Dataset for Visual Estimation of Air Quality</td><td>[2509.16519](http://arxiv.org/pdf/2509.16519)</td><td>◆ We introduce PM25Vision (PM25V), the largest and most comprehensive dataset to date for estimating air quality - specifically PM2.5 concentrations - from street-level images.
◆ The dataset contains over 11,114 images matched with timestamped and geolocated PM2.5 readings across 3,261 AQI monitoring stations and 11 years, significantly exceeding the scale of previous benchmarks.
◆ The spatial accuracy of this dataset has reached 5 kilometers, far exceeding the city-level accuracy of many datasets.</td></tr>
<tr><td>2025-09-19</td><td>DistillMatch: Leveraging Knowledge Distillation from Vision Foundation Model for Multimodal Image Matching</td><td>[2509.16017](http://arxiv.org/pdf/2509.16017)</td><td>◆ Multimodal image matching seeks pixel-level correspondences between images of different modalities, crucial for cross-modal perception, fusion and analysis.
◆ However, the significant appearance differences between modalities make this task challenging.
◆ Due to the scarcity of high-quality annotated datasets, existing deep learning methods that extract modality-common features for matching perform poorly and lack adaptability to diverse scenarios.</td></tr>
<tr><td>2025-09-18</td><td>RoboEye: Enhancing 2D Robotic Object Identification with Selective 3D Geometric Keypoint Matching</td><td>[2509.14966](http://arxiv.org/pdf/2509.14966)</td><td>◆ The rapidly growing number of product categories in large-scale e-commerce makes accurate object identification for automated packing in warehouses substantially more difficult.
◆ As the catalog grows, intra-class variability and a long tail of rare or visually similar items increase, and when combined with diverse packaging, cluttered containers, frequent occlusion, and large viewpoint changes-these factors amplify discrepancies between query and reference images, causing sharp performance drops for methods that rely solely on 2D appearance features.
◆ Thus, we propose RoboEye, a two-stage identification framework that dynamically augments 2D semantic features with domain-adapted 3D reasoning and lightweight adapters to bridge training deployment gaps.</td></tr>
<tr><td>2025-09-15</td><td>Bridging the Gap Between Sparsity and Redundancy: A Dual-Decoding Framework with Global Context for Map Inference</td><td>[2509.11731](http://arxiv.org/pdf/2509.11731)</td><td>该论文提出DGMap框架，旨在解决轨迹地图推断中稀疏区域道路断裂和密集区域冗余段的问题。其核心贡献与创新点包括：
◆ 提出双解码框架，整合全局语义上下文与局部几何特征，提升地图推断的整体一致性。
◆ 设计多尺度网格编码，有效捕捉不同密度区域的轨迹模式，增强特征表达能力。
◆ 引入掩码增强关键点提取机制，提高稀疏区域关键点检测精度，减少道路断裂。
◆ 开发全局上下文感知关系预测模块，通过建模长轨迹依赖抑制密集区域的错误连接。
实验表明，DGMap在三个真实数据集上APLS指标优于现有方法5%，尤其在滴滴轨迹数据上表现突出。</td></tr>
<tr><td>2025-09-14</td><td>A Geometrically Consistent Matching Framework for Side-Scan Sonar Mapping</td><td>[2509.11255](http://arxiv.org/pdf/2509.11255)</td><td>本文提出了一种针对侧扫声纳图像几何一致匹配的新框架，有效解决了因视角依赖、阴影和几何失真导致的匹配难题。  
◆ 通过自监督多分支网络，基于物理反射模型将原始图像解耦为海底反射率、地形高程和声波路径损耗，显著提升了特征表示的稳定性。  
◆ 引入反射率图作为稳定匹配域，并结合无训练匹配流程（SuperPoint与MINIMA LightGlue），增强跨视角对应关系的准确性。  
◆ 提出几何感知的异常匹配剔除机制，综合利用地形高程和物理阴影图，有效抑制声学遮挡和地形不一致区域的错误匹配。  
实验表明，该方法在匹配误差、几何一致性和视角鲁棒性上均优于传统及基于CNN与Transformer的先进方法，为复杂海底环境提供了高精度、数据高效且物理解释性强的匹配解决方案。</td></tr>
<tr><td>2025-09-29</td><td>Loc$^2$: Interpretable Cross-View Localization via Depth-Lifted Local Feature Matching</td><td>[2509.09792](http://arxiv.org/pdf/2509.09792)</td><td>本文提出了一种精细化的跨视角定位方法，通过结合局部特征匹配与单目深度先验，实现了地面图像与航空图像之间的高精度位姿估计。  
◆ 直接建立地面与航空图像间的局部特征对应关系，避免了传统方法中将整张图像转换为鸟瞰图所造成的信息损失。  
◆ 引入单目深度先验，仅将匹配成功的关键点提升至鸟瞰空间，支持使用度量深度和相对深度两种模式，提升了方法的适应性和鲁棒性。  
◆ 提出一种尺度感知的普氏对齐算法，能够从对应关系中估计相机位姿，并在使用相对深度时恢复尺度信息。  
◆ 在弱相机位姿监督下即可学习准确的特征对应关系，在跨区域泛化、未知朝向等复杂场景中表现出优越的定位性能。  
◆ 兼容多种相对深度模型，无需针对每个模型进行微调，具备较强的实用性和部署灵活性。</td></tr>
<tr><td>2025-09-11</td><td>A Path Signature Framework for Detecting Creative Fatigue in Digital Advertising</td><td>[2509.09758](http://arxiv.org/pdf/2509.09758)</td><td>本文提出了一种基于路径签名分析的数字广告创意疲劳检测新框架。  
◆首次将随机分析中的路径签名方法应用于广告疲劳检测领域，开辟了营销分析中几何方法的新途径。  
◆将广告性能时间序列视为二维空间中的路径，利用路径签名作为丰富的特征描述符来捕捉性能动态变化。  
◆通过计算连续时间窗口签名之间的距离，能够识别性能动态中统计显著的变化点。  
◆将统计变化点转化为直接财务指标，量化继续投资疲劳创意所产生的机会成本。  
该方法通过合成实验和案例研究验证了其数学原理性和互补性优势。</td></tr>
<tr><td>2025-09-11</td><td>ObjectReact: Learning Object-Relative Control for Visual Navigation</td><td>[2509.09594](http://arxiv.org/pdf/2509.09594)</td><td>该论文提出了一种基于物体相对控制的视觉导航新范式ObjectReact，以解决传统图像相对方法在泛化性和适应性方面的局限。  
◆ 创新性地采用“物体相对”控制替代主流“图像相对”方法，利用物体作为地图固有属性，摆脱对智能体位姿和具体形态的严格依赖。  
◆ 设计了基于“相对3D场景图”的拓扑-度量混合地图表示，能够生成更鲁棒的物体级全局路径规划代价。  
◆ 提出直接以高层“WayObject Costmap”为输入条件的局部控制器，无需显式RGB输入，将控制预测与图像匹配问题解耦。  
◆ 在跨形态部署（如传感器高度变化）和反向轨迹导航等任务中展现出显著优势，仅通过仿真训练即可泛化至真实室内环境。  
该方法突破了传统视觉导航对经验模仿和严格位姿一致性的要求，实现了更高的泛化能力和跨场景适应性。</td></tr>
<tr><td>2025-09-23</td><td>Handling Multiple Hypotheses in Coarse-to-Fine Dense Image Matching</td><td>[2509.08805](http://arxiv.org/pdf/2509.08805)</td><td>该论文针对稠密图像匹配中因单假设传播导致的误匹配问题，提出了一种多假设处理的创新方法。  
◆ 提出在粗到细的匹配过程中为每个源图像位置生成多个对应点假设，而非传统单一假设。  
◆ 引入束搜索（beam search）策略，在每一尺度上传播和保留多个最优假设。  
◆ 设计将多假设集成到交叉注意力层的新机制，增强特征聚合的鲁棒性。  
◆ 构建名为BEAMER的新型网络架构，能够跨尺度学习和传播多假设信息。  
实验表明该方法在深度不连续或强缩放场景下显著优于现有技术，匹配准确性更高。</td></tr>
<tr><td>2025-09-10</td><td>Computational Imaging for Enhanced Computer Vision</td><td>[2509.08712](http://arxiv.org/pdf/2509.08712)</td><td>该论文系统性地探讨了计算成像（CI）技术如何提升计算机视觉（CV）系统的性能。其核心贡献在于全面综述了CI与CV的协同作用，并指明了未来研究方向。

◆ 系统梳理了多种关键CI技术（如光场成像、HDR成像、去模糊、高速成像和眩光抑制），阐明了它们如何克服传统成像在低光、动态范围等挑战性条件下的局限。
◆ 深入分析了这些CI技术如何具体增强核心CV任务（包括目标检测、深度估计、光流、人脸识别和关键点检测）的鲁棒性与准确性。
◆ 强调了构建面向特定任务的自适应成像管道的潜力，以满足自动驾驶、监控、AR和机器人等实际应用对高精度和高效率的严苛需求。
◆ 指出了该跨学科领域新兴的研究机遇与挑战，为未来的技术发展提供了清晰的路线图。</td></tr>
<tr><td>2025-09-08</td><td>Back To The Drawing Board: Rethinking Scene-Level Sketch-Based Image Retrieval</td><td>[2509.06566](http://arxiv.org/pdf/2509.06566)</td><td>本文针对场景级草图检索任务，重新审视了真实手绘草图中固有的模糊性和噪声问题，并提出了一种更鲁棒的训练方案。
◆ 强调处理真实草图的模糊性和噪声，而非仅改进模型结构。
◆ 设计了一种显式的训练目标，专门提升对草图多样性的鲁棒性。
◆ 通过结合适当的预训练策略、编码器架构和损失函数，在不增加模型复杂度的前提下实现最优性能。
◆ 在FS-COCO和SketchyCOCO数据集上验证了方法的有效性，并指出评估场景需进一步改进。</td></tr>
<tr><td>2025-09-04</td><td>Dual-Scale Volume Priors with Wasserstein-Based Consistency for Semi-Supervised Medical Image Segmentation</td><td>[2509.04273](http://arxiv.org/pdf/2509.04273)</td><td>本文提出了一种用于半监督医学图像分割的新框架，其核心贡献是创新性地整合了双重尺度体积先验和基于Wasserstein距离的一致性约束。  
◆ 首次将源自变分模型的强显式图像尺度体积先验与Threshold Dynamics空间正则化方法集成到分割网络主干中。  
◆ 设计了一个回归网络来估计未标注图像的目标区域体积，并通过图像尺度的Wasserstein距离约束，确保分割结果的类别比例与回归预测一致。  
◆ 引入了一个基于弱隐式体积先验的数据集尺度Wasserstein距离损失函数，强制未标注数据集的分割体积分布与标注数据集相似。  
◆ 在多个公开数据集（ACDC、PROMISE12和大腿肌肉MR）上的实验结果表明，该方法性能优越，验证了其有效性。</td></tr>
<tr><td>2025-08-28</td><td>Estimating 2D Keypoints of Surgical Tools Using Vision-Language Models with Low-Rank Adaptation</td><td>[2508.20830](http://arxiv.org/pdf/2508.20830)</td><td>本文提出了一种利用视觉语言模型（VLM）进行手术工具二维关键点检测的新方法。  
◆ 创新性地采用预训练视觉语言模型（VLM），并通过低秩自适应（LoRA）技术进行微调，有效缓解了在小型医疗数据集上传统CNN或Transformer模型容易过拟合的问题。  
◆ 设计了专门的提示词构建指令微调数据集，将视觉特征与语义关键点描述进行对齐，增强了模型对手术工具的结构理解。  
◆ 仅需两个训练周期即可达到优异性能，显著降低了计算资源和数据量的需求，适用于低资源场景。  
该方法在关键点检测精度上优于基线模型，并为未来三维手术器械及手部姿态估计奠定了基础。</td></tr>
<tr><td>2025-08-25</td><td>DroneKey: Drone 3D Pose Estimation in Image Sequences using Gated Key-representation and Pose-adaptive Learning</td><td>[2508.17746](http://arxiv.org/pdf/2508.17746)</td><td>该论文提出了一种专门针对无人机的三维姿态估计框架DroneKey，解决了无人机螺旋桨作为关键点检测困难的问题。  
◆ 设计了双关键表示机制（中间表示和紧凑表示），并通过门控求和进行优化融合，提升了关键点特征的判别能力。  
◆ 提出姿态自适应的马氏距离损失函数，增强极端姿态下关键点预测的稳定性和准确性。  
◆ 构建并公开了无人机二维关键点与三维姿态的新数据集，为后续研究提供重要基础。  
实验表明，该方法在关键点检测中达到99.68% AP（OKS指标），三维姿态估计角度误差仅10.62度，同时支持44 FPS实时处理，显著优于现有方法。</td></tr>
<tr><td>2025-08-25</td><td>HOSt3R: Keypoint-free Hand-Object 3D Reconstruction from RGB images</td><td>[2508.16465](http://arxiv.org/pdf/2508.16465)</td><td>该论文提出了一种无需关键点检测的手-物体三维重建新方法HOSt3R，其核心贡献与创新点如下：
◆ 摒弃了传统依赖关键点检测（如运动恢复结构和手部关键点优化）的范式，解决了弱纹理、复杂几何和严重遮挡下的失效问题。
◆ 提出了一种直接从单目运动视频中估计手-物体三维变换的鲁棒方法，实现了对未知物体的通用重建。
◆ 将单目变换估计与多视图三维重建流程相结合，实现了高精度的手-物体三维形状恢复。
◆ 整个流程无需预扫描物体模板或已知相机内参，具备高度灵活性和非侵入式应用特性。
该方法在SHOWMe基准测试中达到了最先进的性能，并在HO3D数据集上展示了强大的跨类别泛化能力。</td></tr>
<tr><td>2025-08-21</td><td>Mag-Match: Magnetic Vector Field Features for Map Matching and Registration</td><td>[2508.15300](http://arxiv.org/pdf/2508.15300)</td><td>Mag-Match提出了一种利用三维地磁场矢量特征进行地图匹配与注册的新方法。其核心贡献与创新点包括：

◆ 首次引入基于地磁场高阶导数的特征描述子，该描述子对全局方向具有不变性，无需依赖重力对齐的初始映射。
◆ 利用物理信息高斯过程，从离散的磁力计数据中高效、递归地推断整个地图的磁场及其导数，实现了对磁场的连续化建模。
◆ 所提出的方法在充满烟雾或灰尘等视觉或激光传感器失效的恶劣环境中依然保持鲁棒性能。
◆ 在仿真和真实实验中验证了该方法在地图-地图、机器人-地图以及机器人-机器人之间变换的准确性，性能优于传统的SIFT方法。</td></tr>
<tr><td>2025-08-17</td><td>Splat Feature Solver</td><td>[2508.12216](http://arxiv.org/pdf/2508.12216)</td><td>◆ 提出了一种统一且与核函数及特征无关的特征提升问题建模方法，将其转化为稀疏线性逆问题，并能通过闭式解高效求解。  
◆ 在凸损失函数下，该方法提供了全局最优误差的可证明上界，确保提升后的特征具有高质量。  
◆ 针对多视角图像中的不一致性和噪声问题，提出了两种互补的正则化策略：Tikhonov Guidance通过软对角占优保证数值稳定性，Post-Lifting Aggregation通过特征聚类过滤噪声输入。  
◆ 在开放词汇3D分割基准测试中取得了最先进的性能，显著优于基于训练、分组和启发式的前沿基线方法。  
◆ 实现了分钟级的特征提升效率，同时开源了代码和项目页面以供社区使用。</td></tr>
<tr><td>2025-08-13</td><td>Topological Structure Description for Artcode Detection Using the Shape of Orientation Histogram</td><td>[2508.10942](http://arxiv.org/pdf/2508.10942)</td><td>这篇论文的核心贡献和创新点可总结如下：

◆ 提出了一种新型特征描述符——方向直方图形状（shape of orientation histogram），用于描述Artcode的通用拓扑结构，解决了传统方法难以捕捉拓扑相似但几何和语义差异大的物体的问题。

◆ 将Artcode识别问题重新定义为Artcode提案检测任务，这是计算机视觉领域的一个新任务，专注于分类拓扑相似但外观差异大的物体。

◆ 开发了基于该特征描述符的Artcode检测系统，并通过实验验证了其可行性，为拓扑物体检测提供了新思路。

◆ 收集并构建了专门的数据集，为后续研究提供了基准测试平台。

◆ 这项工作为AR/VR环境中拓扑物体的检测开辟了新途径，有望推动人机交互和数字标记识别技术的发展。</td></tr>
<tr><td>2025-08-14</td><td>Revisiting Cross-View Localization from Image Matching</td><td>[2508.10716](http://arxiv.org/pdf/2508.10716)</td><td>◆ 提出基于跨视角图像匹配的新框架，将定位问题转化为匹配问题，突破传统直接位姿回归或BEV特征对齐的局限性。  
◆ 创新设计Surface Model精确建模地面视角可见区域，实现更准确的鸟瞰图投影，解决几何不一致问题。  
◆ 提出SimRefiner模块，通过局部-全局残差校正优化相似度矩阵，无需RANSAC后处理即可获得精细匹配。  
◆ 构建首个像素级标注的跨视角匹配基准数据集CVFM（含32,509对图像），填补领域空白。  
◆ 实验证明该方法在极端视角差异下显著提升定位精度（最高达30%）和匹配质量，建立新基线。  
核心贡献在于通过几何建模与相似度优化双创新，首次实现严格跨视角对应关系，增强定位结果可解释性。</td></tr>
<tr><td>2025-08-14</td><td>A Sub-Pixel Multimodal Optical Remote Sensing Images Matching Method</td><td>[2508.10294](http://arxiv.org/pdf/2508.10294)</td><td>◆ 提出了一种基于相位一致性加权最小绝对偏差（PCWLAD）的亚像素模板匹配方法，显著提高了多模态光学图像的匹配精度。  
◆ 采用两阶段匹配策略：先通过结构相似性指数（SSIM）进行粗匹配，再利用WLAD进行精细匹配，兼顾效率和精度。  
◆ 在粗匹配阶段保留原始结构细节（不进行噪声过滤），通过SSIM增强对非线性辐射差异的鲁棒性。  
◆ 在精细匹配阶段引入辐射和几何变换模型，结合互结构滤波抑制噪声对结构一致性的影响，提升亚像素偏移估计的准确性。  
◆ 在可见光-红外Landsat、可见光-近红外近距离及无人机图像三类数据集上验证，平均匹配精度达0.4像素，优于现有8种先进方法。  
◆ 公开了算法软件和测试数据集，促进后续研究与应用。</td></tr>
<tr><td>2025-08-13</td><td>Stable Diffusion Models are Secretly Good at Visual In-Context Learning</td><td>[2508.09949](http://arxiv.org/pdf/2508.09949)</td><td>◆ 揭示了现成Stable Diffusion模型具备视觉上下文学习(V-ICL)潜力，无需额外训练即可适应多任务。  
◆ 提出原位注意力重计算机制，通过改造自注意力层显式融合查询与示例提示的上下文关系。  
◆ 首次实现单一预训练扩散模型在六种视觉任务（如分割、检测、着色等）的零样本迁移。  
◆ 在Pascal-5i数据集上，前景分割任务mIoU指标超越Visual Prompting 8.9%和IMProv 3.2%。  
◆ 通过集成多提示样本提升任务推理能力，证明模型可有效利用上下文示例提升性能。  
◆ 突破现有V-ICL方法依赖定制训练或附加数据的限制，提供更通用的视觉上下文学习框架。</td></tr>
<tr><td>2025-08-13</td><td>Episodic Memory Representation for Long-form Video Understanding</td><td>[2508.09486](http://arxiv.org/pdf/2508.09486)</td><td>◆ 提出Video-EM框架，解决现有Video-LLMs因上下文窗口限制难以处理长视频的问题，无需训练即可实现高效视频理解。  
◆ 突破传统关键帧检索方法的静态图像匹配局限，通过模拟人类情景记忆机制，将关键帧建模为时序化情景事件，保留时空动态关系。  
◆ 创新性地结合思维链（CoT）技术，利用大语言模型迭代筛选最小但信息量最大的情景记忆子集，避免冗余帧干扰。  
◆ 首次在关键帧选择中同时考虑空间关联与时间动态性，精准还原视频叙事逻辑，提升场景转换和上下文连续性的捕捉能力。  
◆ 在四大主流长视频基准测试（Video-MME等）中验证有效性，性能显著优于基线4-9%，且使用更少帧数达成更高准确率。</td></tr>
<tr><td>2025-08-11</td><td>VISOR: Visual Input-based Steering for Output Redirection in Vision-Language Models</td><td>[2508.08521](http://arxiv.org/pdf/2508.08521)</td><td>◆ VISOR提出了一种仅通过优化视觉输入实现视觉语言模型(VLM)行为控制的新方法，无需修改文本指令或侵入式访问模型内部。  
◆ 该方法设计了通用的引导图像，能在不引人注意的情况下诱导目标激活模式，适用于所有VLM服务模式，包括API和闭源部署场景。  
◆ 实验证明，仅150KB的引导图像就能在拒绝回答、迎合倾向和生存本能三个关键对齐任务上达到与激活向量相当的引导效果（差距1-2%）。  
◆ 在负面行为引导方面，VISOR表现远超传统方法（最高25%行为偏移），而系统提示方法仅能实现3-4%的改变。  
◆ 该方法在保持14,000项MMLU任务99.9%性能的同时，首次揭示了视觉通道可被用于绕过文本防御的安全漏洞。  
◆ VISOR完全消除了运行时开销和模型访问需求，为多模态模型控制提供了全新思路，并警示了视觉引导攻击的防御紧迫性。</td></tr>
<tr><td>2025-08-11</td><td>Semi-supervised Multiscale Matching for SAR-Optical Image</td><td>[2508.07812](http://arxiv.org/pdf/2508.07812)</td><td>◆提出半监督多尺度匹配框架S2M2-SAR，利用少量标注数据和大量无标注SAR-光学图像对进行训练，降低对人工标注的依赖。  
◆通过结合深层与浅层匹配结果生成伪相似性热图，为无标注数据提供伪标签，有效扩充训练样本。  
◆设计跨模态特征增强模块，采用无监督的跨模态互独立性损失，分离模态共享与模态特定特征，提升特征解耦能力。  
◆无需真实标签即可优化特征空间，增强SAR与光学图像间的匹配鲁棒性。  
◆实验表明，该方法性能超越现有半监督方法，并与全监督SOTA方法相当，具有实际应用潜力。</td></tr>
<tr><td>2025-08-16</td><td>AugLift: Boosting Generalization in Lifting-based 3D Human Pose Estimation</td><td>[2508.07112](http://arxiv.org/pdf/2508.07112)</td><td>◆ 提出AugLift方法，通过简单但有效的输入增强改进基于2D关键点提升的3D人体姿态估计（HPE）的泛化性能，无需额外数据或传感器。  
◆ 创新性地在标准2D关键点坐标(x,y)基础上稀疏增强两个信号：关键点检测置信度c和单目深度估计d，利用预训练模型的泛化能力提升跨数据集表现。  
◆ 方法具有模块化特性，可无缝集成到现有任何提升架构中，无需修改模型结构。  
◆ 实验证明在四个数据集上平均提升跨数据集性能10.1%，同时内部数据集性能也提升4.0%，且在不同架构中表现一致。  
◆ 分析表明稀疏的关键点对齐线索提供了鲁棒的帧级上下文，为提升基于提升的3D姿态估计泛化能力提供了实用解决方案。</td></tr>
<tr><td>2025-08-07</td><td>Head Anchor Enhanced Detection and Association for Crowded Pedestrian Tracking</td><td>[2508.05514](http://arxiv.org/pdf/2508.05514)</td><td>◆ 提出了一种增强的跟踪框架，结合目标检测器的回归和分类分支特征，将空间和位置信息直接嵌入特征表示中，提升特征丰富性。  
◆ 引入头部关键点检测模型，利用头部不易被遮挡的特性，有效缓解严重遮挡场景下的跟踪失效问题。  
◆ 设计了一种迭代卡尔曼滤波方法，与现代检测器假设对齐，结合3D先验信息，优化复杂场景下的运动轨迹补全。  
◆ 综合外观和运动建模的创新，在拥挤且遮挡严重的多目标跟踪场景中提供更鲁棒的解决方案。  
◆ 通过实验验证，该方法在遮挡频繁的 pedestrian tracking 任务中显著优于传统依赖全身框特征和匀速运动假设的方法。</td></tr>
<tr><td>2025-08-07</td><td>Refining Gaussian Splatting: A Volumetric Densification Approach</td><td>[2508.05187](http://arxiv.org/pdf/2508.05187)</td><td>◆ 提出基于惯性体积的新型密度控制方法，利用高斯函数的惯性体积指导3D高斯分布的精细化过程，改进了原始3DGS的密度控制策略。  
◆ 系统研究了传统运动恢复结构(SfM)和深度图像匹配(DIM)两种点云初始化方法对3DGS性能的影响，为初始化选择提供依据。  
◆ 在Mip-NeRF 360数据集上的大量实验表明，该方法在重建质量上优于原始3DGS，在不同场景中均表现出色。  
◆ 通过更智能的密度控制机制，解决了原始3DGS自适应密度控制(ADC)在点基元管理上的不足。  
◆ 该方法保持了3DGS的高效渲染特性，同时显著提升了新视角合成的质量。</td></tr>
<tr><td>2025-08-09</td><td>SGAD: Semantic and Geometric-aware Descriptor for Local Feature Matching</td><td>[2508.02278](http://arxiv.org/pdf/2508.02278)</td><td>◆ 提出SGAD网络，通过生成高区分度的区域描述符，直接实现区域匹配，避免传统低效的像素级比较和复杂图优化，显著提升匹配精度与效率。  
◆ 创新性地将区域匹配任务分解为分类和排序子任务，通过新型监督策略进一步提升匹配性能。  
◆ 设计层次包容冗余过滤器（HCRF），基于包容图分析消除重叠区域，优化匹配结构。  
◆ 在效率上实现重大突破：相比MESA方法运行时减少60倍（0.82秒 vs 60.23秒），同时保持更高精度。  
◆ 在多个基准测试中验证普适性：与LoFTR结合时户外姿态估计精度达65.98（优于DKM的61.11），与ROMA结合时室内姿态估计AUC@5°提升7.39%，刷新SOTA。  
◆ 首次实现区域匹配框架中几何与语义信息的协同建模，为局部特征匹配提供新范式。</td></tr>
<tr><td>2025-07-31</td><td>Mitigating Resolution-Drift in Federated Learning: Case of Keypoint Detection</td><td>[2507.23461](http://arxiv.org/pdf/2507.23461)</td><td>◆ 首次在联邦学习中提出并系统研究了&quot;分辨率漂移&quot;问题，揭示了分辨率差异作为非独立同分布数据的新维度对关键点检测任务的影响。  
◆ 提出分辨率自适应联邦学习（RAF）方法，通过基于热图的多分辨率知识蒸馏技术，在高分辨率教师模型和低分辨率学生模型间传递知识。  
◆ RAF创新性地实现了分辨率鲁棒性提升，避免了过拟合问题，且无需修改现有联邦学习框架即可直接集成。  
◆ 通过理论分析和大量实验验证，证明RAF能有效缓解分辨率漂移，在人体姿态估计任务上取得显著性能提升。  
◆ 虽然以姿态估计为案例，但t-SNE分析揭示了分类任务与高分辨率表征任务的特征差异，表明RAF可推广至其他需要保持空间细节的任务领域。  
◆ 填补了联邦学习在非分类任务（特别是涉及空间信息的视觉任务）中的研究空白，为处理分辨率异构性提供了新思路。</td></tr>
<tr><td>2025-07-31</td><td>VMatcher: State-Space Semi-Dense Local Feature Matching</td><td>[2507.23371](http://arxiv.org/pdf/2507.23371)</td><td>◆ 提出VMatcher，一种结合Mamba和Transformer的混合网络，用于图像对的半稠密特征匹配，兼顾性能和效率。  
◆ 首次将选择性状态空间模型（SSM）引入特征匹配任务，利用Mamba的线性复杂度显著降低计算成本，突破传统Transformer二次方复杂度的瓶颈。  
◆ 设计混合架构，融合Mamba的长序列高效处理能力和Transformer的注意力机制，在保持高精度的同时提升计算效率。  
◆ 提出多种层次化网络配置，通过不同深度和模块组合优化匹配效果，在多个基准测试中达到最优性能。  
◆ 强调实时应用场景的实用性，模型在快速推理需求下仍保持鲁棒性，为实际部署提供高效解决方案。  
◆ 开源代码促进后续研究，为特征匹配领域提供新的技术路线和可复现的基准。</td></tr>
<tr><td>2025-07-30</td><td>Modality-Aware Feature Matching: A Comprehensive Review of Single- and Cross-Modality Techniques</td><td>[2507.22791](http://arxiv.org/pdf/2507.22791)</td><td>◆ 全面综述了单模态和跨模态特征匹配技术，涵盖RGB图像、深度图像、3D点云、LiDAR扫描、医学图像及视觉-语言交互等多种模态，填补了该领域系统性总结的空白。  
◆ 对比分析了传统手工方法（如Harris角点、SIFT和ORB描述子）与深度学习方法（如SuperPoint和LoFTR）的优劣，指出后者在跨模态鲁棒性和适应性上的显著提升。  
◆ 重点介绍了模态感知技术的最新进展，例如针对深度图像的几何与深度专用描述子、3D点云的稀疏与稠密学习方法，以及LiDAR扫描中基于注意力增强的神经网络。  
◆ 强调了跨模态应用的突破，如医学图像配准中的MIND描述子和视觉-语言任务中的交互匹配技术，展示了特征匹配在多样化数据交互中的潜力。  
◆ 系统总结了当前挑战与未来方向，为跨模态特征匹配的研究提供了清晰的路线图，推动该领域向更复杂场景扩展。</td></tr>
<tr><td>2025-07-25</td><td>Cross Spatial Temporal Fusion Attention for Remote Sensing Object Detection via Image Feature Matching</td><td>[2507.19118](http://arxiv.org/pdf/2507.19118)</td><td>◆提出跨时空融合注意力机制(CSTF)，通过独立检测参考图和查询图中的尺度不变关键点来增强特征表示，解决多模态遥感图像几何和辐射差异大的问题。  
◆创新性地构建对应图，同时利用多图像区域信息，提升跨模态特征匹配能力。  
◆将相似性匹配重新定义为分类任务，结合SoftMax和全卷积网络(FCN)层，兼顾局部特征敏感性和全局上下文信息。  
◆在HRSC2016和DOTA基准测试中达到90.99%和90.86%的平均mAP，显著优于现有方法，验证了跨模态匹配对下游目标检测任务的直接提升。  
◆保持12.5 FPS的推理速度，兼具高性能与计算效率，满足实际应用需求。</td></tr>
<tr><td>2025-07-24</td><td>A 3D Cross-modal Keypoint Descriptor for MR-US Matching and Registration</td><td>[2507.18551](http://arxiv.org/pdf/2507.18551)</td><td>◆提出新型3D跨模态关键点描述符，解决MRI与实时超声(iUS)因模态差异导致的配准难题。  
◆采用患者特异性合成匹配方法，通过术前MRI生成合成iUS数据，实现监督对比学习以构建共享描述符空间。  
◆设计概率关键点检测策略，识别解剖学显著且跨模态一致的特征点，提升匹配可靠性。  
◆创新性使用课程式三元组损失与动态困难负样本挖掘，使描述符具备抗iUS伪影（如斑点噪声）和旋转不变性。  
◆在ReMIND数据集上验证，关键点匹配平均精度达69.8%，配准误差仅2.39mm，优于现有方法。  
◆框架无需人工初始化，对iUS视野变化鲁棒，且具有可解释性，代码已开源。</td></tr>
<tr><td>2025-07-23</td><td>CartoonAlive: Towards Expressive Live2D Modeling from Single Portraits</td><td>[2507.17327](http://arxiv.org/pdf/2507.17327)</td><td>◆ 提出CartoonAlive方法，首次实现从单张肖像照片快速生成高质量Live2D模型，耗时不足30秒。  
◆ 创新地将3D人脸建模中的形状基概念引入2D领域，构建适用于Live2D的面部混合形状。  
◆ 通过面部关键点检测自动推算混合形状权重，无需人工干预即可实现高精度表情控制。  
◆ 采用分层分割技术模拟3D运动效果，在保持2D卡通风格的同时实现类似3D的实时动态表现。  
◆ 相比传统3D建模方案大幅降低制作成本，相比2D视频方案显著提升交互灵活性。  
◆ 为虚拟角色动画和数字内容创作提供了高效、可扩展的2D卡通人物生成方案。</td></tr>
<tr><td>2025-07-21</td><td>Toward a Real-Time Framework for Accurate Monocular 3D Human Pose Estimation with Geometric Priors</td><td>[2507.16850](http://arxiv.org/pdf/2507.16850)</td><td>◆ 提出了一种实时单目3D人体姿态估计框架，结合了2D关键点检测与几何感知的2D到3D提升技术。  
◆ 显式利用相机内参和个性化解剖学先验知识，通过自校准和生物力学约束的反向运动学增强精度。  
◆ 创新性地从动作捕捉和合成数据集中生成大规模合理的2D-3D训练对，减少对标注数据的依赖。  
◆ 框架支持快速个性化适配，无需专用硬件即可实现高精度估计，适用于边缘设备部署。  
◆ 融合数据驱动学习与模型先验，在提升准确性的同时增强模型的可解释性和实际应用性。  
◆ 针对无约束环境优化，平衡实时性与精度，推动野外环境下的3D运动捕捉技术发展。</td></tr>
<tr><td>2025-07-22</td><td>A Single-step Accurate Fingerprint Registration Method Based on Local Feature Matching</td><td>[2507.16201](http://arxiv.org/pdf/2507.16201)</td><td>◆ 提出一种端到端的单步指纹配准算法，直接预测两枚指纹间的半稠密匹配点对应关系，避免了传统两步法的初始配准失败风险。  
◆ 采用全局-局部注意力机制，实现端到端的像素级对齐，提升了低质量指纹图像的配准精度。  
◆ 通过单步配准即可达到当前最优的匹配性能，简化了传统方法中依赖 minutiae 初始配准的复杂流程。  
◆ 算法兼容稠密配准算法，可进一步结合使用以提升性能，具有灵活性和扩展性。  
◆ 实验证明该方法在低质量指纹图像中表现优异，解决了因 minutiae 数量不足导致的配准失败问题。</td></tr>
<tr><td>2025-07-17</td><td>DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model</td><td>[2507.13145](http://arxiv.org/pdf/2507.13145)</td><td>◆ 提出DINO-VO，一种基于DINOv2视觉基础模型的稀疏特征匹配视觉里程计系统，解决了传统学习型单目VO在鲁棒性、泛化性和效率上的挑战。  
◆ 设计了一种针对DINOv2粗粒度特征的自适应显著关键点检测器，克服了基础模型特征粒度不足的集成难题。  
◆ 结合DINOv2的鲁棒语义特征与细粒度几何特征，生成更具局部化能力的混合特征表示。  
◆ 采用基于Transformer的匹配器和可微分位姿估计层，通过学习优质匹配实现精确相机运动估计。  
◆ 在TartanAir和KITTI数据集上超越先前帧间VO方法，在EuRoC数据集表现相当，同时以72FPS高效运行且内存占用低于1GB。  
◆ 在户外驾驶场景中与Visual SLAM系统竞争，展现了优于SuperPoint等传统方法的跨环境泛化能力。</td></tr>
<tr><td>2025-07-15</td><td>KptLLM++: Towards Generic Keypoint Comprehension with Large Language Model</td><td>[2507.11102](http://arxiv.org/pdf/2507.11102)</td><td>◆提出KptLLM++，首个专用于通用关键点理解的多模态大语言模型，通过用户指令整合多样化输入模态。  
◆创新性地采用&quot;先识别后检测&quot;范式，通过结构化思维链机制先解析关键点语义再精确定位，提升细粒度理解能力。  
◆构建超50万样本的大规模训练数据集，覆盖多样物体、关键点类别、图像风格及复杂遮挡场景，显著增强模型泛化性。  
◆突破现有MLLMs在像素级细粒度语义捕捉的局限，实现跨场景关键点检测的统一框架。  
◆在多个关键点检测基准上达到SOTA性能，为人类-AI协作提供更高效的交互接口。  
◆通过结构化关键点表征推动细粒度图像分析、物体检索和行为识别等应用的发展。</td></tr>
<tr><td>2025-07-15</td><td>GKNet: Graph-based Keypoints Network for Monocular Pose Estimation of Non-cooperative Spacecraft</td><td>[2507.11077](http://arxiv.org/pdf/2507.11077)</td><td>◆ 提出基于图结构的关键点网络GKNet，利用关键点间的几何约束关系，提升非合作航天器单目姿态估计的精度。  
◆ 针对航天器结构对称性和局部遮挡问题，通过图网络建模关键点拓扑关系，增强检测鲁棒性。  
◆ 构建中等规模航天器关键点检测数据集SKD，包含3种航天器目标、9万张仿真图像及高精度标注，填补领域数据空白。  
◆ 实验证明GKNet在关键点检测精度上显著优于现有先进方法，验证了图结构约束的有效性。  
◆ 开源GKNet代码和SKD数据集，促进非合作航天器姿态估计研究的可复现性。</td></tr>
<tr><td>2025-07-14</td><td>FPC-Net: Revisiting SuperPoint with Descriptor-Free Keypoint Detection via Feature Pyramids and Consistency-Based Implicit Matching</td><td>[2507.10770](http://arxiv.org/pdf/2507.10770)</td><td>◆ 提出FPC-Net，通过特征金字塔和一致性隐式匹配实现无描述符的关键点检测，革新了传统依赖描述符的匹配流程。  
◆ 关键点检测与匹配过程一体化，无需额外计算、存储或传输描述符，显著降低内存开销。  
◆ 采用隐式匹配机制，通过特征金字塔和一致性约束直接关联关键点，简化了匹配流程。  
◆ 尽管匹配精度略低于传统方法，但完全摒弃描述符的设计大幅提升了系统效率，尤其适用于资源受限的定位场景。  
◆ 实验验证表明，该方法在性能上优于传统手工特征方法，并与现代学习方案竞争力相当。  
◆ 为计算机视觉中的几何任务提供了一种高效、轻量化的新思路，具有实际应用潜力。</td></tr>
<tr><td>2025-07-27</td><td>Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection</td><td>[2507.07994](http://arxiv.org/pdf/2507.07994)</td><td>◆ 提出首个基于草图的无源少样本关键点检测框架，利用人类手绘草图作为跨模态监督信号，解决了传统方法在查询数据分布未知时的局限性。  
◆ 设计原型网络结合网格定位器的双分支架构，有效对齐草图与图像的关键点特征空间，实现跨模态嵌入学习。  
◆ 创新性引入原型域适应模块，通过风格解耦技术消除用户手绘风格的个体差异，提升模型泛化能力。  
◆ 开发少样本快速收敛机制，仅需少量标注样本即可推广至新关键点类别，实验证明其在多个基准数据集上超越现有方法。  
◆ 首次系统性验证草图模态在关键点检测中的迁移价值，为少样本视觉任务开辟了新的数据利用范式。</td></tr>
</tbody>
</table>
</div>

<h2 id='3dgs'>3DGS</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-10-23</td><td>Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking</td><td>[2510.20335](http://arxiv.org/pdf/2510.20335)</td><td>◆ Parking is a critical pillar of driving safety.
◆ While recent end-to-end (E2E) approaches have achieved promising in-domain results, robustness under domain shifts (e.g., weather and lighting changes) remains a key challenge.
◆ Rather than relying on additional data, in this paper, we propose Dino-Diffusion Parking (DDP), a domain-agnostic autonomous parking pipeline that integrates visual foundation models with diffusion-based planning to enable generalized perception and robust motion planning under distribution shifts.</td></tr>
<tr><td>2025-10-22</td><td>Extreme Views: 3DGS Filter for Novel View Synthesis from Out-of-Distribution Camera Poses</td><td>[2510.20027](http://arxiv.org/pdf/2510.20027)</td><td>◆ When viewing a 3D Gaussian Splatting (3DGS) model from camera positions significantly outside the training data distribution, substantial visual noise commonly occurs.
◆ These artifacts result from the lack of training data in these extrapolated regions, leading to uncertain density, color, and geometry predictions from the model.
◆ To address this issue, we propose a novel real-time render-aware filtering method.</td></tr>
<tr><td>2025-10-22</td><td>Advances in 4D Representation: Geometry, Motion, and Interaction</td><td>[2510.19255](http://arxiv.org/pdf/2510.19255)</td><td>◆ We present a survey on 4D generation and reconstruction, a fast-evolving subfield of computer graphics whose developments have been propelled by recent advances in neural fields, geometric and motion deep learning, as well 3D generative artificial intelligence (GenAI).
◆ While our survey is not the first of its kind, we build our coverage of the domain from a unique and distinctive perspective of 4D representations\/}, to model 3D geometry evolving over time while exhibiting motion and interaction.
◆ Specifically, instead of offering an exhaustive enumeration of many works, we take a more selective approach by focusing on representative works to highlight both the desirable properties and ensuing challenges of each representation under different computation, application, and data scenarios.</td></tr>
<tr><td>2025-10-21</td><td>Moving Light Adaptive Colonoscopy Reconstruction via Illumination-Attenuation-Aware 3D Gaussian Splatting</td><td>[2510.18739](http://arxiv.org/pdf/2510.18739)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a pivotal technique for real-time view synthesis in colonoscopy, enabling critical applications such as virtual colonoscopy and lesion tracking.
◆ However, the vanilla 3DGS assumes static illumination and that observed appearance depends solely on viewing angle, which causes incompatibility with the photometric variations in colonoscopic scenes induced by dynamic light source/camera.
◆ This mismatch forces most 3DGS methods to introduce structure-violating vaporous Gaussian blobs between the camera and tissues to compensate for illumination attenuation, ultimately degrading the quality of 3D reconstructions.</td></tr>
<tr><td>2025-10-20</td><td>From Volume Rendering to 3D Gaussian Splatting: Theory and Applications</td><td>[2510.18101](http://arxiv.org/pdf/2510.18101)</td><td>◆ The problem of 3D reconstruction from posed images is undergoing a fundamental transformation, driven by continuous advances in 3D Gaussian Splatting (3DGS).
◆ By modeling scenes explicitly as collections of 3D Gaussians, 3DGS enables efficient rasterization through volumetric splatting, offering thus a seamless integration with common graphics pipelines.
◆ Despite its real-time rendering capabilities for novel view synthesis, 3DGS suffers from a high memory footprint, the tendency to bake lighting effects directly into its representation, and limited support for secondary-ray effects.</td></tr>
<tr><td>2025-10-20</td><td>Raindrop GS: A Benchmark for 3D Gaussian Splatting under Raindrop Conditions</td><td>[2510.17719](http://arxiv.org/pdf/2510.17719)</td><td>◆ 3D Gaussian Splatting (3DGS) under raindrop conditions suffers from severe occlusions and optical distortions caused by raindrop contamination on the camera lens, substantially degrading reconstruction quality.
◆ Existing benchmarks typically evaluate 3DGS using synthetic raindrop images with known camera poses (constrained images), assuming ideal conditions.
◆ However, in real-world scenarios, raindrops often interfere with accurate camera pose estimation and point cloud initialization.</td></tr>
<tr><td>2025-10-20</td><td>Initialize to Generalize: A Stronger Initialization Pipeline for Sparse-View 3DGS</td><td>[2510.17479](http://arxiv.org/pdf/2510.17479)</td><td>◆ Sparse-view 3D Gaussian Splatting (3DGS) often overfits to the training views, leading to artifacts like blurring in novel view rendering.
◆ Prior work addresses it either by enhancing the initialization (\emph{i.e.}, the point cloud from Structure-from-Motion (SfM)) or by adding training-time constraints (regularization) to the 3DGS optimization.
◆ Yet our controlled ablations reveal that initialization is the decisive factor: it determines the attainable performance band in sparse-view 3DGS, while training-time constraints yield only modest within-band improvements at extra cost.</td></tr>
<tr><td>2025-10-19</td><td>2DGS-R: Revisiting the Normal Consistency Regularization in 2D Gaussian Splatting</td><td>[2510.16837](http://arxiv.org/pdf/2510.16837)</td><td>◆ Recent advancements in 3D Gaussian Splatting (3DGS) have greatly influenced neural fields, as it enables high-fidelity rendering with impressive visual quality.
◆ However, 3DGS has difficulty accurately representing surfaces.
◆ In contrast, 2DGS transforms the 3D volume into a collection of 2D planar Gaussian disks.</td></tr>
<tr><td>2025-10-19</td><td>GS2POSE: Marry Gaussian Splatting to 6D Object Pose Estimation</td><td>[2510.16777](http://arxiv.org/pdf/2510.16777)</td><td>◆ Accurate 6D pose estimation of 3D objects is a fundamental task in computer vision, and current research typically predicts the 6D pose by establishing correspondences between 2D image features and 3D model features.
◆ However, these methods often face difficulties with textureless objects and varying illumination conditions.
◆ To overcome these limitations, we propose GS2POSE, a novel approach for 6D object pose estimation.</td></tr>
<tr><td>2025-10-18</td><td>HGC-Avatar: Hierarchical Gaussian Compression for Streamable Dynamic 3D Avatars</td><td>[2510.16463](http://arxiv.org/pdf/2510.16463)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) have enabled fast, photorealistic rendering of dynamic 3D scenes, showing strong potential in immersive communication.
◆ However, in digital human encoding and transmission, the compression methods based on general 3DGS representations are limited by the lack of human priors, resulting in suboptimal bitrate efficiency and reconstruction quality at the decoder side, which hinders their application in streamable 3D avatar systems.
◆ We propose HGC-Avatar, a novel Hierarchical Gaussian Compression framework designed for efficient transmission and high-quality rendering of dynamic avatars.</td></tr>
<tr><td>2025-10-17</td><td>Fix False Transparency by Noise Guided Splatting</td><td>[2510.15736](http://arxiv.org/pdf/2510.15736)</td><td>◆ Opaque objects reconstructed by 3DGS often exhibit a falsely transparent surface, leading to inconsistent background and internal patterns under camera motion in interactive viewing.
◆ This issue stems from the ill-posed optimization in 3DGS.
◆ During training, background and foreground Gaussians are blended via alpha-compositing and optimized solely against the input RGB images using a photometric loss.</td></tr>
<tr><td>2025-10-17</td><td>PFGS: Pose-Fused 3D Gaussian Splatting for Complete Multi-Pose Object Reconstruction</td><td>[2510.15386](http://arxiv.org/pdf/2510.15386)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) have enabled high-quality, real-time novel-view synthesis from multi-view images.
◆ However, most existing methods assume the object is captured in a single, static pose, resulting in incomplete reconstructions that miss occluded or self-occluded regions.
◆ We introduce PFGS, a pose-aware 3DGS framework that addresses the practical challenge of reconstructing complete objects from multi-pose image captures.</td></tr>
<tr><td>2025-10-16</td><td>SaLon3R: Structure-aware Long-term Generalizable 3D Reconstruction from Unposed Images</td><td>[2510.15072](http://arxiv.org/pdf/2510.15072)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) have enabled generalizable, on-the-fly reconstruction of sequential input views.
◆ However, existing methods often predict per-pixel Gaussians and combine Gaussians from all views as the scene representation, leading to substantial redundancies and geometric inconsistencies in long-duration video sequences.
◆ To address this, we propose SaLon3R, a novel framework for Structure-aware, Long-term 3DGS Reconstruction.</td></tr>
<tr><td>2025-10-16</td><td>Leveraging Learned Image Prior for 3D Gaussian Compression</td><td>[2510.14705](http://arxiv.org/pdf/2510.14705)</td><td>◆ Compression techniques for 3D Gaussian Splatting (3DGS) have recently achieved considerable success in minimizing storage overhead for 3D Gaussians while preserving high rendering quality.
◆ Despite the impressive storage reduction, the lack of learned priors restricts further advances in the rate-distortion trade-off for 3DGS compression tasks.
◆ To address this, we introduce a novel 3DGS compression framework that leverages the powerful representational capacity of learned image priors to recover compression-induced quality degradation.</td></tr>
<tr><td>2025-10-16</td><td>BalanceGS: Algorithm-System Co-design for Efficient 3D Gaussian Splatting Training on GPU</td><td>[2510.14564](http://arxiv.org/pdf/2510.14564)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a promising 3D reconstruction technique.
◆ The traditional 3DGS training pipeline follows three sequential steps: Gaussian densification, Gaussian projection, and color splatting.
◆ Despite its promising reconstruction quality, this conventional approach suffers from three critical inefficiencies: (1) Skewed density allocation during Gaussian densification, (2) Imbalanced computation workload during Gaussian projection and (3) Fragmented memory access during color splatting.</td></tr>
<tr><td>2025-10-15</td><td>Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering</td><td>[2510.13381](http://arxiv.org/pdf/2510.13381)</td><td>◆ Dynamic scene rendering and reconstruction play a crucial role in computer vision and augmented reality.
◆ Recent methods based on 3D Gaussian Splatting (3DGS), have enabled accurate modeling of dynamic urban scenes, but for urban scenes they require both camera and LiDAR data, ground-truth 3D segmentations and motion data in the form of tracklets or pre-defined object templates such as SMPL.
◆ In this work, we explore whether a combination of 2D object agnostic priors in the form of depth and point tracking coupled with a signed distance function (SDF) representation for dynamic objects can be used to relax some of these requirements.</td></tr>
<tr><td>2025-10-16</td><td>SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms</td><td>[2510.12901](http://arxiv.org/pdf/2510.12901)</td><td>◆ Rigorous testing of autonomous robots, such as self-driving vehicles, is essential to ensure their safety in real-world deployments.
◆ This requires building high-fidelity simulators to test scenarios beyond those that can be safely or exhaustively collected in the real-world.
◆ Existing neural rendering methods based on NeRF and 3DGS hold promise but suffer from low rendering speeds or can only render pinhole camera models, hindering their suitability to applications that commonly require high-distortion lenses and LiDAR data.</td></tr>
<tr><td>2025-10-17</td><td>BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring</td><td>[2510.12493](http://arxiv.org/pdf/2510.12493)</td><td>◆ 3D Gaussian Splatting has exhibited remarkable capabilities in 3D scene reconstruction.However, reconstructing high-quality 3D scenes from motion-blurred images caused by camera motion poses a significant challenge.The performance of existing 3DGS-based deblurring methods are limited due to their inherent mechanisms, such as extreme dependence on the accuracy of camera poses and inability to effectively control erroneous Gaussian primitives densification caused by motion blur.To solve these problems, we introduce a novel framework, Bi-Stage 3D Gaussian Splatting, to accurately reconstruct 3D scenes from motion-blurred images.BSGS contains two stages.
◆ First, Camera Pose Refinement roughly optimizes camera poses to reduce motion-induced distortions.
◆ Second, with fixed rough camera poses, Global RigidTransformation further corrects motion-induced blur distortions.To alleviate multi-subframe gradient conflicts, we propose a subframe gradient aggregation strategy to optimize both stages.Furthermore, a space-time bi-stage optimization strategy is introduced to dynamically adjust primitive densification thresholds and prevent premature noisy Gaussian generation in blurred regions.</td></tr>
<tr><td>2025-10-12</td><td>High-Fidelity Simulated Data Generation for Real-World Zero-Shot Robotic Manipulation Learning with Gaussian Splatting</td><td>[2510.10637](http://arxiv.org/pdf/2510.10637)</td><td>◆ The scalability of robotic learning is fundamentally bottlenecked by the significant cost and labor of real-world data collection.
◆ While simulated data offers a scalable alternative, it often fails to generalize to the real world due to significant gaps in visual appearance, physical properties, and object interactions.
◆ To address this, we propose RoboSimGS, a novel Real2Sim2Real framework that converts multi-view real-world images into scalable, high-fidelity, and physically interactive simulation environments for robotic manipulation.</td></tr>
<tr><td>2025-10-11</td><td>Opacity-Gradient Driven Density Control for Compact and Efficient Few-Shot 3D Gaussian Splatting</td><td>[2510.10257](http://arxiv.org/pdf/2510.10257)</td><td>◆ 3D Gaussian Splatting (3DGS) struggles in few-shot scenarios, where its standard adaptive density control (ADC) can lead to overfitting and bloated reconstructions.
◆ While state-of-the-art methods like FSGS improve quality, they often do so by significantly increasing the primitive count.
◆ This paper presents a framework that revises the core 3DGS optimization to prioritize efficiency.</td></tr>
<tr><td>2025-10-11</td><td>Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting</td><td>[2510.10097](http://arxiv.org/pdf/2510.10097)</td><td>◆ Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced 3D reconstruction and novel view synthesis, but remain heavily dependent on accurate camera poses and dense viewpoint coverage.
◆ These requirements limit their applicability in sparse-view settings, where pose estimation becomes unreliable and supervision is insufficient.
◆ To overcome these challenges, we introduce Gesplat, a 3DGS-based framework that enables robust novel view synthesis and geometrically consistent reconstruction from unposed sparse images.</td></tr>
<tr><td>2025-10-11</td><td>P-4DGS: Predictive 4D Gaussian Splatting with 90$\times$ Compression</td><td>[2510.10030](http://arxiv.org/pdf/2510.10030)</td><td>◆ 3D Gaussian Splatting (3DGS) has garnered significant attention due to its superior scene representation fidelity and real-time rendering performance, especially for dynamic 3D scene reconstruction (\textit{i.e.}, 4D reconstruction).
◆ However, despite achieving promising results, most existing algorithms overlook the substantial temporal and spatial redundancies inherent in dynamic scenes, leading to prohibitive memory consumption.
◆ To address this, we propose P-4DGS, a novel dynamic 3DGS representation for compact 4D scene modeling.</td></tr>
<tr><td>2025-10-11</td><td>CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting</td><td>[2510.09997](http://arxiv.org/pdf/2510.09997)</td><td>◆ Level of Detail (LoD) is a fundamental technique in real-time computer graphics for managing the rendering costs of complex scenes while preserving visual fidelity.
◆ Traditionally, LoD is implemented using discrete levels (DLoD), where multiple, distinct versions of a model are swapped out at different distances.
◆ This long-standing paradigm, however, suffers from two major drawbacks: it requires significant storage for multiple model copies and causes jarring visual ``popping&quot; artifacts during transitions, degrading the user experience.</td></tr>
<tr><td>2025-10-11</td><td>VG-Mapping: Variation-Aware 3D Gaussians for Online Semi-static Scene Mapping</td><td>[2510.09962](http://arxiv.org/pdf/2510.09962)</td><td>◆ Maintaining an up-to-date map that accurately reflects recent changes in the environment is crucial, especially for robots that repeatedly traverse the same space.
◆ Failing to promptly update the changed regions can degrade map quality, resulting in poor localization, inefficient operations, and even lost robots.
◆ 3D Gaussian Splatting (3DGS) has recently seen widespread adoption in online map reconstruction due to its dense, differentiable, and photorealistic properties, yet accurately and efficiently updating the regions of change remains a challenge.</td></tr>
<tr><td>2025-10-10</td><td>Visibility-Aware Densification for 3D Gaussian Splatting in Dynamic Urban Scenes</td><td>[2510.09364](http://arxiv.org/pdf/2510.09364)</td><td>◆ 3D Gaussian splatting (3DGS) has demonstrated impressive performance in synthesizing high-fidelity novel views.
◆ Nonetheless, its effectiveness critically depends on the quality of the initialized point cloud.
◆ Specifically, achieving uniform and complete point coverage over the underlying scene structure requires overlapping observation frustums, an assumption that is often violated in unbounded, dynamic urban environments.</td></tr>
<tr><td>2025-10-09</td><td>D$^2$GS: Depth-and-Density Guided Gaussian Splatting for Stable and Accurate Sparse-View Reconstruction</td><td>[2510.08566](http://arxiv.org/pdf/2510.08566)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) enable real-time, high-fidelity novel view synthesis (NVS) with explicit 3D representations.
◆ However, performance degradation and instability remain significant under sparse-view conditions.
◆ In this work, we identify two key failure modes under sparse-view conditions: overfitting in regions with excessive Gaussian density near the camera, and underfitting in distant areas with insufficient Gaussian coverage.</td></tr>
<tr><td>2025-10-09</td><td>Efficient Label Refinement for Face Parsing Under Extreme Poses Using 3D Gaussian Splatting</td><td>[2510.08096](http://arxiv.org/pdf/2510.08096)</td><td>◆ Accurate face parsing under extreme viewing angles remains a significant challenge due to limited labeled data in such poses.
◆ Manual annotation is costly and often impractical at scale.
◆ We propose a novel label refinement pipeline that leverages 3D Gaussian Splatting (3DGS) to generate accurate segmentation masks from noisy multiview predictions.</td></tr>
<tr><td>2025-10-09</td><td>PrismGS: Physically-Grounded Anti-Aliasing for High-Fidelity Large-Scale 3D Gaussian Splatting</td><td>[2510.07830](http://arxiv.org/pdf/2510.07830)</td><td>◆ 3D Gaussian Splatting (3DGS) has recently enabled real-time photorealistic rendering in compact scenes, but scaling to large urban environments introduces severe aliasing artifacts and optimization instability, especially under high-resolution (e.g., 4K) rendering.
◆ These artifacts, manifesting as flickering textures and jagged edges, arise from the mismatch between Gaussian primitives and the multi-scale nature of urban geometry.
◆ While existing ``divide-and-conquer&#x27;&#x27; pipelines address scalability, they fail to resolve this fidelity gap.</td></tr>
<tr><td>2025-10-09</td><td>DEGS: Deformable Event-based 3D Gaussian Splatting from RGB and Event Stream</td><td>[2510.07752](http://arxiv.org/pdf/2510.07752)</td><td>◆ Reconstructing Dynamic 3D Gaussian Splatting (3DGS) from low-framerate RGB videos is challenging.
◆ This is because large inter-frame motions will increase the uncertainty of the solution space.
◆ For example, one pixel in the first frame might have more choices to reach the corresponding pixel in the second frame.</td></tr>
<tr><td>2025-10-09</td><td>RTGS: Real-Time 3D Gaussian Splatting SLAM via Multi-Level Redundancy Reduction</td><td>[2510.06644](http://arxiv.org/pdf/2510.06644)</td><td>◆ 3D Gaussian Splatting (3DGS) based Simultaneous Localization and Mapping (SLAM) systems can largely benefit from 3DGS&#x27;s state-of-the-art rendering efficiency and accuracy, but have not yet been adopted in resource-constrained edge devices due to insufficient speed.
◆ Addressing this, we identify notable redundancies across the SLAM pipeline for acceleration.
◆ While conceptually straightforward, practical approaches are required to minimize the overhead associated with identifying and eliminating these redundancies.</td></tr>
<tr><td>2025-10-07</td><td>ArchitectHead: Continuous Level of Detail Control for 3D Gaussian Head Avatars</td><td>[2510.05488](http://arxiv.org/pdf/2510.05488)</td><td>◆ 3D Gaussian Splatting (3DGS) has enabled photorealistic and real-time rendering of 3D head avatars.
◆ Existing 3DGS-based avatars typically rely on tens of thousands of 3D Gaussian points (Gaussians), with the number of Gaussians fixed after training.
◆ However, many practical applications require adjustable levels of detail (LOD) to balance rendering efficiency and visual quality.</td></tr>
<tr><td>2025-10-02</td><td>StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions</td><td>[2510.02314](http://arxiv.org/pdf/2510.02314)</td><td>◆ 3D scene representation methods like Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have significantly advanced novel view synthesis.
◆ As these methods become prevalent, addressing their vulnerabilities becomes critical.
◆ We analyze 3DGS robustness against image-level poisoning attacks and propose a novel density-guided poisoning method.</td></tr>
<tr><td>2025-10-02</td><td>GaussianMorphing: Mesh-Guided 3D Gaussians for Semantic-Aware Object Morphing</td><td>[2510.02034](http://arxiv.org/pdf/2510.02034)</td><td>◆ We introduce GaussianMorphing, a novel framework for semantic-aware 3D shape and texture morphing from multi-view images.
◆ Previous approaches usually rely on point clouds or require pre-defined homeomorphic mappings for untextured data.
◆ Our method overcomes these limitations by leveraging mesh-guided 3D Gaussian Splatting (3DGS) for high-fidelity geometry and appearance modeling.</td></tr>
<tr><td>2025-10-02</td><td>ROI-GS: Interest-based Local Quality 3D Gaussian Splatting</td><td>[2510.01978](http://arxiv.org/pdf/2510.01978)</td><td>◆ We tackle the challenge of efficiently reconstructing 3D scenes with high detail on objects of interest.
◆ Existing 3D Gaussian Splatting (3DGS) methods allocate resources uniformly across the scene, limiting fine detail to Regions Of Interest (ROIs) and leading to inflated model size.
◆ We propose ROI-GS, an object-aware framework that enhances local details through object-guided camera selection, targeted Object training, and seamless integration of high-fidelity object of interest reconstructions into the global scene.</td></tr>
<tr><td>2025-10-02</td><td>LOBE-GS: Load-Balanced and Efficient 3D Gaussian Splatting for Large-Scale Scene Reconstruction</td><td>[2510.01767](http://arxiv.org/pdf/2510.01767)</td><td>◆ 3D Gaussian Splatting (3DGS) has established itself as an efficient representation for real-time, high-fidelity 3D scene reconstruction.
◆ However, scaling 3DGS to large and unbounded scenes such as city blocks remains difficult.
◆ Existing divide-and-conquer methods alleviate memory pressure by partitioning the scene into blocks, but introduce new bottlenecks: (i) partitions suffer from severe load imbalance since uniform or heuristic splits do not reflect actual computational demands, and (ii) coarse-to-fine pipelines fail to exploit the coarse stage efficiently, often reloading the entire model and incurring high overhead.</td></tr>
<tr><td>2025-09-30</td><td>LLM-Powered Code Analysis and Optimization for Gaussian Splatting Kernels</td><td>[2509.25626](http://arxiv.org/pdf/2509.25626)</td><td>◆ 3D Gaussian splatting (3DGS) is a transformative technique with profound implications on novel view synthesis and real-time rendering.
◆ Given its importance, there have been many attempts to improve its performance.
◆ However, with the increasing complexity of GPU architectures and the vast search space of performance-tuning parameters, it is a challenging task.</td></tr>
<tr><td>2025-09-29</td><td>GaussianLens: Localized High-Resolution Reconstruction via On-Demand Gaussian Densification</td><td>[2509.25603](http://arxiv.org/pdf/2509.25603)</td><td>◆ We perceive our surroundings with an active focus, paying more attention to regions of interest, such as the shelf labels in a grocery store.
◆ When it comes to scene reconstruction, this human perception trait calls for spatially varying degrees of detail ready for closer inspection in critical regions, preferably reconstructed on demand.
◆ While recent works in 3D Gaussian Splatting (3DGS) achieve fast, generalizable reconstruction from sparse views, their uniform resolution output leads to high computational costs unscalable to high-resolution training.</td></tr>
<tr><td>2025-10-08</td><td>VGGT-X: When VGGT Meets Dense Novel View Synthesis</td><td>[2509.25191](http://arxiv.org/pdf/2509.25191)</td><td>◆ We study the problem of applying 3D Foundation Models (3DFMs) to dense Novel View Synthesis (NVS).
◆ Despite significant progress in Novel View Synthesis powered by NeRF and 3DGS, current approaches remain reliant on accurate 3D attributes (e.g., camera poses and point clouds) acquired from Structure-from-Motion (SfM), which is often slow and fragile in low-texture or low-overlap captures.
◆ Recent 3DFMs showcase orders of magnitude speedup over the traditional pipeline and great potential for online NVS.</td></tr>
<tr><td>2025-09-29</td><td>Triangle Splatting+: Differentiable Rendering with Opaque Triangles</td><td>[2509.25122](http://arxiv.org/pdf/2509.25122)</td><td>◆ Reconstructing 3D scenes and synthesizing novel views has seen rapid progress in recent years.
◆ Neural Radiance Fields demonstrated that continuous volumetric radiance fields can achieve high-quality image synthesis, but their long training and rendering times limit practicality.
◆ 3D Gaussian Splatting (3DGS) addressed these issues by representing scenes with millions of Gaussians, enabling real-time rendering and fast optimization.</td></tr>
<tr><td>2025-10-02</td><td>GEM: 3D Gaussian Splatting for Efficient and Accurate Cryo-EM Reconstruction</td><td>[2509.25075](http://arxiv.org/pdf/2509.25075)</td><td>◆ Cryo-electron microscopy (cryo-EM) has become a central tool for high-resolution structural biology, yet the massive scale of datasets (often exceeding 100k particle images) renders 3D reconstruction both computationally expensive and memory intensive.
◆ Traditional Fourier-space methods are efficient but lose fidelity due to repeated transforms, while recent real-space approaches based on neural radiance fields (NeRFs) improve accuracy but incur cubic memory and computation overhead.
◆ Therefore, we introduce GEM, a novel cryo-EM reconstruction framework built on 3D Gaussian Splatting (3DGS) that operates directly in real-space while maintaining high efficiency.</td></tr>
<tr><td>2025-09-29</td><td>DWGS: Enhancing Sparse-View Gaussian Splatting with Hybrid-Loss Depth Estimation and Bidirectional Warping</td><td>[2509.24893](http://arxiv.org/pdf/2509.24893)</td><td>◆ Novel View Synthesis (NVS) from sparse views remains a core challenge in 3D reconstruction, typically suffering from overfitting, geometric distortion, and incomplete scene recovery due to limited multi-view constraints.
◆ Although 3D Gaussian Splatting (3DGS) enables real-time, high-fidelity rendering, it suffers from floating artifacts and structural inconsistencies under sparse-input settings.
◆ To address these issues, we propose DWGS, a novel unified framework that enhances 3DGS for sparse-view synthesis by integrating robust structural cues, virtual view constraints, and occluded region completion.</td></tr>
<tr><td>2025-09-29</td><td>ExGS: Extreme 3D Gaussian Compression with Diffusion Priors</td><td>[2509.24758](http://arxiv.org/pdf/2509.24758)</td><td>◆ Neural scene representations, such as 3D Gaussian Splatting (3DGS), have enabled high-quality neural rendering; however, their large storage and transmission costs hinder deployment in resource-constrained environments.
◆ Existing compression methods either rely on costly optimization, which is slow and scene-specific, or adopt training-free pruning and quantization, which degrade rendering quality under high compression ratios.
◆ In contrast, recent data-driven approaches provide a promising direction to overcome this trade-off, enabling efficient compression while preserving high rendering quality.</td></tr>
<tr><td>2025-10-01</td><td>Proxy-GS: Efficient 3D Gaussian Splatting via Proxy Mesh</td><td>[2509.24421](http://arxiv.org/pdf/2509.24421)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as an efficient approach for achieving photorealistic rendering.
◆ Recent MLP-based variants further improve visual fidelity but introduce substantial decoding overhead during rendering.
◆ To alleviate computation cost, several pruning strategies and level-of-detail (LOD) techniques have been introduced, aiming to effectively reduce the number of Gaussian primitives in large-scale scenes.</td></tr>
<tr><td>2025-09-28</td><td>From Fields to Splats: A Cross-Domain Survey of Real-Time Neural Scene Representations</td><td>[2509.23555](http://arxiv.org/pdf/2509.23555)</td><td>◆ Neural scene representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have transformed how 3D environments are modeled, rendered, and interpreted.
◆ NeRF introduced view-consistent photorealism via volumetric rendering; 3DGS has rapidly emerged as an explicit, efficient alternative that supports high-quality rendering, faster optimization, and integration into hybrid pipelines for enhanced photorealism and task-driven scene understanding.
◆ This survey examines how 3DGS is being adopted across SLAM, telepresence and teleoperation, robotic manipulation, and 3D content generation.</td></tr>
<tr><td>2025-09-26</td><td>Learning Unified Representation of 3D Gaussian Splatting</td><td>[2509.22917](http://arxiv.org/pdf/2509.22917)</td><td>◆ A well-designed vectorized representation is crucial for the learning systems natively based on 3D Gaussian Splatting.
◆ While 3DGS enables efficient and explicit 3D reconstruction, its parameter-based representation remains hard to learn as features, especially for neural-network-based models.
◆ Directly feeding raw Gaussian parameters into learning frameworks fails to address the non-unique and heterogeneous nature of the Gaussian parameterization, yielding highly data-dependent models.</td></tr>
<tr><td>2025-09-26</td><td>Polysemous Language Gaussian Splatting via Matching-based Mask Lifting</td><td>[2509.22225](http://arxiv.org/pdf/2509.22225)</td><td>◆ Lifting 2D open-vocabulary understanding into 3D Gaussian Splatting (3DGS) scenes is a critical challenge.
◆ However, mainstream methods suffer from three key flaws: (i) their reliance on costly per-scene retraining prevents plug-and-play application; (ii) their restrictive monosemous design fails to represent complex, multi-concept semantics; and (iii) their vulnerability to cross-view semantic inconsistencies corrupts the final semantic representation.
◆ To overcome these limitations, we introduce MUSplat, a training-free framework that abandons feature optimization entirely.</td></tr>
<tr><td>2025-09-25</td><td>PowerGS: Display-Rendering Power Co-Optimization for Neural Rendering in Power-Constrained XR Systems</td><td>[2509.21702](http://arxiv.org/pdf/2509.21702)</td><td>◆ 3D Gaussian Splatting (3DGS) combines classic image-based rendering, pointbased graphics, and modern differentiable techniques, and offers an interesting alternative to traditional physically-based rendering.
◆ 3DGS-family models are far from efficient for power-constrained Extended Reality (XR) devices, which need to operate at a Watt-level.
◆ This paper introduces PowerGS, the first framework to jointly minimize the rendering and display power in 3DGS under a quality constraint.</td></tr>
<tr><td>2025-09-23</td><td>SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing</td><td>[2509.20400](http://arxiv.org/pdf/2509.20400)</td><td>◆ This paper presents SeHDR, a novel high dynamic range 3D Gaussian Splatting (HDR-3DGS) approach for generating HDR novel views given multi-view LDR images.
◆ Unlike existing methods that typically require the multi-view LDR input images to be captured from different exposures, which are tedious to capture and more likely to suffer from errors (e.g., object motion blurs and calibration/alignment inaccuracies), our approach learns the HDR scene representation from multi-view LDR images of a single exposure.
◆ Our key insight to this ill-posed problem is that by first estimating Bracketed 3D Gaussians (i.e., with different exposures) from single-exposure multi-view LDR images, we may then be able to merge these bracketed 3D Gaussians into an HDR scene representation.</td></tr>
<tr><td>2025-09-24</td><td>GS-RoadPatching: Inpainting Gaussians via 3D Searching and Placing for Driving Scenes</td><td>[2509.19937](http://arxiv.org/pdf/2509.19937)</td><td>◆ This paper presents GS-RoadPatching, an inpainting method for driving scene completion by referring to completely reconstructed regions, which are represented by 3D Gaussian Splatting (3DGS).
◆ Unlike existing 3DGS inpainting methods that perform generative completion relying on 2D perspective-view-based diffusion or GAN models to predict limited appearance or depth cues for missing regions, our approach enables substitutional scene inpainting and editing directly through the 3DGS modality, extricating it from requiring spatial-temporal consistency of 2D cross-modals and eliminating the need for time-intensive retraining of Gaussians.
◆ Our key insight is that the highly repetitive patterns in driving scenes often share multi-modal similarities within the implicit 3DGS feature space and are particularly suitable for structural matching to enable effective 3DGS-based substitutional inpainting.</td></tr>
<tr><td>2025-09-24</td><td>Aerial-Ground Image Feature Matching via 3D Gaussian Splatting-based Intermediate View Rendering</td><td>[2509.19898](http://arxiv.org/pdf/2509.19898)</td><td>◆ The integration of aerial and ground images has been a promising solution in 3D modeling of complex scenes, which is seriously restricted by finding reliable correspondences.
◆ The primary contribution of this study is a feature matching algorithm for aerial and ground images, whose core idea is to generate intermediate views to alleviate perspective distortions caused by the extensive viewpoint changes.
◆ First, by using aerial images only, sparse models are reconstructed through an incremental SfM (Structure from Motion) engine due to their large scene coverage.</td></tr>
<tr><td>2025-09-24</td><td>PolGS: Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction</td><td>[2509.19726](http://arxiv.org/pdf/2509.19726)</td><td>◆ Efficient shape reconstruction for surfaces with complex reflectance properties is crucial for real-time virtual reality.
◆ While 3D Gaussian Splatting (3DGS)-based methods offer fast novel view rendering by leveraging their explicit surface representation, their reconstruction quality lags behind that of implicit neural representations, particularly in the case of recovering surfaces with complex reflective reflectance.
◆ To address these problems, we propose PolGS, a Polarimetric Gaussian Splatting model allowing fast reflective surface reconstruction in 10 minutes.</td></tr>
<tr><td>2025-09-23</td><td>VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction</td><td>[2509.19297](http://arxiv.org/pdf/2509.19297)</td><td>◆ Feed-forward 3D Gaussian Splatting (3DGS) has emerged as a highly effective solution for novel view synthesis.
◆ Existing methods predominantly rely on a pixel-aligned Gaussian prediction paradigm, where each 2D pixel is mapped to a 3D Gaussian.
◆ We rethink this widely adopted formulation and identify several inherent limitations: it renders the reconstructed 3D models heavily dependent on the number of input views, leads to view-biased density distributions, and introduces alignment errors, particularly when source views contain occlusions or low texture.</td></tr>
<tr><td>2025-09-23</td><td>Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation</td><td>[2509.19296](http://arxiv.org/pdf/2509.19296)</td><td>◆ The ability to generate virtual environments is crucial for applications ranging from gaming to physical AI domains such as robotics, autonomous driving, and industrial AI.
◆ Current learning-based 3D reconstruction methods rely on the availability of captured real-world multi-view data, which is not always readily available.
◆ Recent advancements in video diffusion models have shown remarkable imagination capabilities, yet their 2D nature limits the applications to simulation where a robot needs to navigate and interact with the environment.</td></tr>
<tr><td>2025-09-23</td><td>WaveletGaussian: Wavelet-domain Diffusion for Sparse-view 3D Gaussian Object Reconstruction</td><td>[2509.19073](http://arxiv.org/pdf/2509.19073)</td><td>◆ 3D Gaussian Splatting (3DGS) has become a powerful representation for image-based object reconstruction, yet its performance drops sharply in sparse-view settings.
◆ Prior works address this limitation by employing diffusion models to repair corrupted renders, subsequently using them as pseudo ground truths for later optimization.
◆ While effective, such approaches incur heavy computation from the diffusion fine-tuning and repair steps.</td></tr>
<tr><td>2025-09-23</td><td>Seeing Through Reflections: Advancing 3D Scene Reconstruction in Mirror-Containing Environments with Gaussian Splatting</td><td>[2509.18956](http://arxiv.org/pdf/2509.18956)</td><td>◆ Mirror-containing environments pose unique challenges for 3D reconstruction and novel view synthesis (NVS), as reflective surfaces introduce view-dependent distortions and inconsistencies.
◆ While cutting-edge methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) excel in typical scenes, their performance deteriorates in the presence of mirrors.
◆ Existing solutions mainly focus on handling mirror surfaces through symmetry mapping but often overlook the rich information carried by mirror reflections.</td></tr>
<tr><td>2025-09-23</td><td>FixingGS: Enhancing 3D Gaussian Splatting via Training-Free Score Distillation</td><td>[2509.18759](http://arxiv.org/pdf/2509.18759)</td><td>◆ Recently, 3D Gaussian Splatting (3DGS) has demonstrated remarkable success in 3D reconstruction and novel view synthesis.
◆ However, reconstructing 3D scenes from sparse viewpoints remains highly challenging due to insufficient visual information, which results in noticeable artifacts persisting across the 3D representation.
◆ To address this limitation, recent methods have resorted to generative priors to remove artifacts and complete missing content in under-constrained areas.</td></tr>
<tr><td>2025-09-22</td><td>From Restoration to Reconstruction: Rethinking 3D Gaussian Splatting for Underwater Scenes</td><td>[2509.17789](http://arxiv.org/pdf/2509.17789)</td><td>◆ Underwater image degradation poses significant challenges for 3D reconstruction, where simplified physical models often fail in complex scenes.
◆ We propose \textbf{R-Splatting}, a unified framework that bridges underwater image restoration (UIR) with 3D Gaussian Splatting (3DGS) to improve both rendering quality and geometric fidelity.
◆ Our method integrates multiple enhanced views produced by diverse UIR models into a single reconstruction pipeline.</td></tr>
<tr><td>2025-09-22</td><td>Neural-MMGS: Multi-modal Neural Gaussian Splats for Large-Scale Scene Reconstruction</td><td>[2509.17762](http://arxiv.org/pdf/2509.17762)</td><td>◆ This paper proposes Neural-MMGS, a novel neural 3DGS framework for multimodal large-scale scene reconstruction that fuses multiple sensing modalities in a per-gaussian compact, learnable embedding.
◆ While recent works focusing on large-scale scene reconstruction have incorporated LiDAR data to provide more accurate geometric constraints, we argue that LiDAR&#x27;s rich physical properties remain underexplored.
◆ Similarly, semantic information has been used for object retrieval, but could provide valuable high-level context for scene reconstruction.</td></tr>
<tr><td>2025-09-22</td><td>FGGS-LiDAR: Ultra-Fast, GPU-Accelerated Simulation from General 3DGS Models to LiDAR</td><td>[2509.17390](http://arxiv.org/pdf/2509.17390)</td><td>◆ While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic rendering, its vast ecosystem of assets remains incompatible with high-performance LiDAR simulation, a critical tool for robotics and autonomous driving.
◆ We present \textbf{FGGS-LiDAR}, a framework that bridges this gap with a truly plug-and-play approach.
◆ Our method converts \textit{any} pretrained 3DGS model into a high-fidelity, watertight mesh without requiring LiDAR-specific supervision or architectural alterations.</td></tr>
<tr><td>2025-09-23</td><td>HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis</td><td>[2509.17083](http://arxiv.org/pdf/2509.17083)</td><td>◆ Recently, 3D Gaussian Splatting (3DGS) has emerged as a powerful alternative to NeRF-based approaches, enabling real-time, high-quality novel view synthesis through explicit, optimizable 3D Gaussians.
◆ However, 3DGS suffers from significant memory overhead due to its reliance on per-Gaussian parameters to model view-dependent effects and anisotropic shapes.
◆ While recent works propose compressing 3DGS with neural fields, these methods struggle to capture high-frequency spatial variations in Gaussian properties, leading to degraded reconstruction of fine details.</td></tr>
<tr><td>2025-09-21</td><td>PGSTalker: Real-Time Audio-Driven Talking Head Generation via 3D Gaussian Splatting with Pixel-Aware Density Control</td><td>[2509.16922](http://arxiv.org/pdf/2509.16922)</td><td>◆ Audio-driven talking head generation is crucial for applications in virtual reality, digital avatars, and film production.
◆ While NeRF-based methods enable high-fidelity reconstruction, they suffer from low rendering efficiency and suboptimal audio-visual synchronization.
◆ This work presents PGSTalker, a real-time audio-driven talking head synthesis framework based on 3D Gaussian Splatting (3DGS).</td></tr>
<tr><td>2025-09-21</td><td>ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM</td><td>[2509.16863](http://arxiv.org/pdf/2509.16863)</td><td>◆ We introduce ConfidentSplat, a novel 3D Gaussian Splatting (3DGS)-based SLAM system for robust, highfidelity RGB-only reconstruction.
◆ Addressing geometric inaccuracies in existing RGB-only 3DGS SLAM methods that stem from unreliable depth estimation, ConfidentSplat incorporates a core innovation: a confidence-weighted fusion mechanism.
◆ This mechanism adaptively integrates depth cues from multiview geometry with learned monocular priors (Omnidata ViT), dynamically weighting their contributions based on explicit reliability estimates-derived predominantly from multi-view geometric consistency-to generate high-fidelity proxy depth for map supervision.</td></tr>
<tr><td>2025-09-19</td><td>RadarGaussianDet3D: An Efficient and Effective Gaussian-based 3D Detector with 4D Automotive Radars</td><td>[2509.16119](http://arxiv.org/pdf/2509.16119)</td><td>◆ 4D automotive radars have gained increasing attention for autonomous driving due to their low cost, robustness, and inherent velocity measurement capability.
◆ However, existing 4D radar-based 3D detectors rely heavily on pillar encoders for BEV feature extraction, where each point contributes to only a single BEV grid, resulting in sparse feature maps and degraded representation quality.
◆ In addition, they also optimize bounding box attributes independently, leading to sub-optimal detection accuracy.</td></tr>
<tr><td>2025-09-19</td><td>Zero-Shot Visual Grounding in 3D Gaussians via View Retrieval</td><td>[2509.15871](http://arxiv.org/pdf/2509.15871)</td><td>◆ 3D Visual Grounding (3DVG) aims to locate objects in 3D scenes based on text prompts, which is essential for applications such as robotics.
◆ However, existing 3DVG methods encounter two main challenges: first, they struggle to handle the implicit representation of spatial textures in 3D Gaussian Splatting (3DGS), making per-scene training indispensable; second, they typically require larges amounts of labeled data for effective training.
◆ To this end, we propose \underline{G}rounding via \underline{V}iew \underline{R}etrieval (GVR), a novel zero-shot visual grounding framework for 3DGS to transform 3DVG as a 2D retrieval task that leverages object-level view retrieval to collect grounding clues from multiple views, which not only avoids the costly process of 3D annotation, but also eliminates the need for per-scene training.</td></tr>
<tr><td>2025-09-22</td><td>MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild</td><td>[2509.15548](http://arxiv.org/pdf/2509.15548)</td><td>◆ In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis.
◆ Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting.
◆ In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS.</td></tr>
<tr><td>2025-09-17</td><td>Perception-Integrated Safety Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian Splatting</td><td>[2509.14421](http://arxiv.org/pdf/2509.14421)</td><td>◆ We present a perception-driven safety filter that converts each 3D Gaussian Splat (3DGS) into a closed-form forward collision cone, which in turn yields a first-order control barrier function (CBF) embedded within a quadratic program (QP).
◆ By exploiting the analytic geometry of splats, our formulation provides a continuous, closed-form representation of collision constraints that is both simple and computationally efficient.
◆ Unlike distance-based CBFs, which tend to activate reactively only when an obstacle is already close, our collision-cone CBF activates proactively, allowing the robot to adjust earlier and thereby produce smoother and safer avoidance maneuvers at lower computational cost.</td></tr>
<tr><td>2025-09-17</td><td>MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping</td><td>[2509.14191](http://arxiv.org/pdf/2509.14191)</td><td>◆ Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage.
◆ We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS).
◆ Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map.</td></tr>
<tr><td>2025-09-17</td><td>Plug-and-Play PDE Optimization for 3D Gaussian Splatting: Toward High-Quality Rendering and Reconstruction</td><td>[2509.13938](http://arxiv.org/pdf/2509.13938)</td><td>◆ 3D Gaussian Splatting (3DGS) has revolutionized radiance field reconstruction by achieving high-quality novel view synthesis with fast rendering speed, introducing 3D Gaussian primitives to represent the scene.
◆ However, 3DGS encounters blurring and floaters when applied to complex scenes, caused by the reconstruction of redundant and ambiguous geometric structures.
◆ We attribute this issue to the unstable optimization of the Gaussians.</td></tr>
<tr><td>2025-09-16</td><td>MemGS: Memory-Efficient Gaussian Splatting for Real-Time SLAM</td><td>[2509.13536](http://arxiv.org/pdf/2509.13536)</td><td>◆ Recent advancements in 3D Gaussian Splatting (3DGS) have made a significant impact on rendering and reconstruction techniques.
◆ Current research predominantly focuses on improving rendering performance and reconstruction quality using high-performance desktop GPUs, largely overlooking applications for embedded platforms like micro air vehicles (MAVs).
◆ These devices, with their limited computational resources and memory, often face a trade-off between system performance and reconstruction quality.</td></tr>
<tr><td>2025-09-16</td><td>Improving 3D Gaussian Splatting Compression by Scene-Adaptive Lattice Vector Quantization</td><td>[2509.13482](http://arxiv.org/pdf/2509.13482)</td><td>◆ 3D Gaussian Splatting (3DGS) is rapidly gaining popularity for its photorealistic rendering quality and real-time performance, but it generates massive amounts of data.
◆ Hence compressing 3DGS data is necessary for the cost effectiveness of 3DGS models.
◆ Recently, several anchor-based neural compression methods have been proposed, achieving good 3DGS compression performance.</td></tr>
<tr><td>2025-09-16</td><td>Dream3DAvatar: Text-Controlled 3D Avatar Reconstruction from a Single Image</td><td>[2509.13013](http://arxiv.org/pdf/2509.13013)</td><td>◆ With the rapid advancement of 3D representation techniques and generative models, substantial progress has been made in reconstructing full-body 3D avatars from a single image.
◆ However, this task remains fundamentally ill-posedness due to the limited information available from monocular input, making it difficult to control the geometry and texture of occluded regions during generation.
◆ To address these challenges, we redesign the reconstruction pipeline and propose Dream3DAvatar, an efficient and text-controllable two-stage framework for 3D avatar generation.</td></tr>
<tr><td>2025-09-16</td><td>Beyond Averages: Open-Vocabulary 3D Scene Understanding with Gaussian Splatting and Bag of Embeddings</td><td>[2509.12938](http://arxiv.org/pdf/2509.12938)</td><td>◆ Novel view synthesis has seen significant advancements with 3D Gaussian Splatting (3DGS), enabling real-time photorealistic rendering.
◆ However, the inherent fuzziness of Gaussian Splatting presents challenges for 3D scene understanding, restricting its broader applications in AR/VR and robotics.
◆ While recent works attempt to learn semantics via 2D foundation model distillation, they inherit fundamental limitations: alpha blending averages semantics across objects, making 3D-level understanding impossible.</td></tr>
<tr><td>2025-09-18</td><td>A-TDOM: Active TDOM via On-the-Fly 3DGS</td><td>[2509.12759](http://arxiv.org/pdf/2509.12759)</td><td>◆ True Digital Orthophoto Map (TDOM) serves as a crucial geospatial product in various fields such as urban management, city planning, land surveying, etc.
◆ However, traditional TDOM generation methods generally rely on a complex offline photogrammetric pipeline, resulting in delays that hinder real-time applications.
◆ Moreover, the quality of TDOM may degrade due to various challenges, such as inaccurate camera poses or Digital Surface Model (DSM) and scene occlusions.</td></tr>
<tr><td>2025-09-15</td><td>Segmentation-Driven Initialization for Sparse-view 3D Gaussian Splatting</td><td>[2509.11853](http://arxiv.org/pdf/2509.11853)</td><td>该论文针对稀疏视图合成中3D高斯泼溅（3DGS）方法存在的高内存消耗和低效初始化问题，提出了一种基于分割驱动的初始化方法（SDI-GS）。其核心贡献与创新点包括：

◆ 提出利用区域分割技术识别结构显著区域，替代传统依赖运动恢复结构（SfM）或多视图立体（MVS）的初始化方式；
◆ 通过选择性下采样稠密点云，大幅减少3D高斯数量，在保持场景保真度的同时降低内存占用；
◆ 在多个基准测试中实现高达50%的高斯数量削减，并保持可比甚至更优的渲染质量（PSNR/SSIM指标）；
◆ 显著提升训练速度并降低计算资源需求，推动了3DGS在有限视图场景中的实用化。</td></tr>
<tr><td>2025-09-14</td><td>ROSGS: Relightable Outdoor Scenes With Gaussian Splatting</td><td>[2509.11275](http://arxiv.org/pdf/2509.11275)</td><td>ROSGS提出了一种基于高斯泼溅表示的两阶段流程，用于高效重建可重光照的户外无界场景。  
◆ 采用紧凑的二维高斯泼溅（2DGS）表示结合单目法向先验，显著提升了几何重建的效率和精度。  
◆ 设计混合光照模型，分别用球面高斯函数捕捉阳光的高频方向性成分，以及用球谐系数建模天光的低频漫射光照。  
◆ 克服了NeRF方法计算开销大和传统低频光照表示精度不足的问题，实现了更优的光照分解与渲染效果。  
实验表明，该方法在户外场景重光照任务中达到了最先进的性能，同时兼具高渲染效率和卓越的重光照准确性。</td></tr>
<tr><td>2025-09-14</td><td>SPHERE: Semantic-PHysical Engaged REpresentation for 3D Semantic Scene Completion</td><td>[2509.11171](http://arxiv.org/pdf/2509.11171)</td><td>SPHERE提出了一种用于相机3D语义场景补全（SSC）的新表征方法，旨在同时提升场景的语义准确性和几何细节的真实性。其核心贡献与创新点如下：

◆ 提出了一种体素与高斯表征相融合的新范式，协同利用语义和物理信息，以解决现有方法在几何细节或语义精度上的不足。
◆ 设计了语义引导的高斯初始化（SGI）模块，利用双分支3D场景表征定位焦点体素作为锚点，从而指导高效的高斯初始化，提升了处理效率。
◆ 开发了物理感知的球谐增强（PHE）模块，通过引入语义球谐函数来建模物理上下文细节，并通过焦点分布对齐促进语义与几何的一致性，生成具有逼真细节的结果。
◆ 该方法在保持高语义精度的同时，显著改善了复杂大尺度驾驶场景中几何细节的还原质量，克服了NeRF等方法计算成本高、收敛慢的缺点。
◆ 在主流数据集SemanticKITTI和SSCBench-KITTI-360上的大量实验验证了其有效性与优越性。</td></tr>
<tr><td>2025-09-14</td><td>SVR-GS: Spatially Variant Regularization for Probabilistic Masks in 3D Gaussian Splatting</td><td>[2509.11116](http://arxiv.org/pdf/2509.11116)</td><td>本文提出SVR-GS方法，针对3D高斯泼溅（3DGS）中掩码修剪的优化问题进行了创新。  
◆ 引入了空间变异正则化器，通过计算每个高斯沿光线方向的有效贡献生成逐像素空间掩码，取代传统全局均值正则化方法。  
◆ 设计了三种空间掩码聚合策略，并通过梯度分析优化最终方案，实现更精准的稀疏性控制。  
◆ 在CUDA上实现高效计算，显著提升处理效率。  
实验表明，该方法在三大数据集上平均将高斯数量减少至MaskGS的1.79倍和3DGS的5.63倍，仅带来0.50 dB和0.40 dB的PSNR下降。  
最终模型更小、更快且内存效率更高，适用于机器人、AR/VR等实时应用场景。</td></tr>
<tr><td>2025-09-13</td><td>AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting</td><td>[2509.11003](http://arxiv.org/pdf/2509.11003)</td><td>AD-GS提出了一种新颖的交替致密化框架，旨在解决稀疏输入下3D高斯泼溅（3DGS）的过拟合和几何失真问题。其核心贡献在于通过交替的高、低致密化阶段，实现了对模型容量增长的精细控制。  
◆ 创新性地引入了交替致密化流程，将激进的几何增长与严格的修剪正则化阶段交错进行。  
◆ 在高致密化阶段，通过积极增加高斯原语并结合光度损失训练，以捕捉场景细节。  
◆ 在低致密化阶段，执行激进的透明度修剪，并结合伪视角一致性和边缘感知深度平滑约束来正则化几何形状。  
◆ 该机制有效抑制了因无控制 densification 产生的浮游物和错误几何，显著提升了渲染质量和几何一致性。  
实验证明，该方法在极具挑战性的稀疏数据集上优于现有方法。</td></tr>
<tr><td>2025-09-09</td><td>SplatFill: 3D Scene Inpainting via Depth-Guided Gaussian Splatting</td><td>[2509.07809](http://arxiv.org/pdf/2509.07809)</td><td>该论文提出了一种名为SplatFill的新方法，用于基于3D高斯泼溅（3DGS）的3D场景修复。其核心贡献在于解决了现有方法在修复缺失区域时常见的模糊、伪影和几何不一致问题。  
◆ 结合了基于深度和基于物体的联合监督机制，确保修复的高斯点能够精确放置在3D空间中并与周围几何对齐。  
◆ 提出了一种一致性感知的精化方案，能够选择性地识别并修正不一致的区域，同时避免破坏场景的其他部分。  
实验结果表明，该方法在SPIn-NeRF数据集上不仅视觉保真度优于现有的基于NeRF和3DGS的修复方法，而且训练时间减少了24.5%，同时在不同视角下呈现出更清晰的细节、更少的伪影和更高的一致性。</td></tr>
<tr><td>2025-09-09</td><td>HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting</td><td>[2509.07774](http://arxiv.org/pdf/2509.07774)</td><td>该论文提出了一种基于3D高斯泼溅（3DGS）的头发丝级重建方法HairGS，实现了从多视角图像高效重建高保真头发几何与拓扑结构。其核心创新点包括：
◆ 将3DGS框架扩展至头发丝级重建，通过可微分高斯光栅化器实现细节几何重建
◆ 提出新颖的高斯段融合方案，将离散高斯片段合并为连贯的发丝结构
◆ 设计在光度监督下的发丝优化生长机制，进一步提升重建精度
◆ 提出针对头发拓扑连接性的评估指标，弥补现有方法仅关注几何精度而忽视拓扑连接的不足
该方法在一小时内即可完成重建，在合成和真实数据集上均能鲁棒处理多种发型，为虚拟现实和数字人建模提供了高效解决方案。</td></tr>
<tr><td>2025-09-09</td><td>OmniMap: A General Mapping Framework Integrating Optics, Geometry, and Semantics</td><td>[2509.07500](http://arxiv.org/pdf/2509.07500)</td><td>OmniMap是一个创新的实时在线建图框架，首次同时实现了光学、几何和语义三大场景属性的高精度统一建模。  
◆提出紧密耦合的3DGS-体素混合表示，兼顾细粒度建模与结构稳定性。  
◆引入自适应相机建模技术，有效补偿运动模糊和曝光问题。  
◆采用带法向约束的混合增量表示方法，提升几何精度。  
◆通过概率融合机制实现零样本实例级语义理解，消除语义歧义。  
实验证明其在渲染质量、几何精度和语义分割方面均优于现有方法，并支持问答、编辑、操作导航等多类下游应用。</td></tr>
<tr><td>2025-09-09</td><td>DiGS: Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning</td><td>[2509.07493](http://arxiv.org/pdf/2509.07493)</td><td>DiGS提出了一种将3D高斯溅射与隐式表面重建相统一的创新框架，旨在解决3DGS几何表示松散和缺乏显式几何监督的核心问题。其核心贡献与创新点如下：

◆ 首次将可学习符号距离场（SDF）直接嵌入3D高斯表示中，为每个高斯基元关联SDF值，从而引入了强几何先验。
◆ 通过SDF学习显式地将高斯基元与底层几何表面对齐，显著提升了跨视角一致性和几何准确性。
◆ 提出了一种几何引导的网格增长策略，该策略能在多尺度层级下自适应地在几何一致区域密集分布高斯，确保表面的完整性与连贯性。
◆ 在保持3DGS高渲染保真度的优势的同时，在多个标准数据集上实现了更精确和更完整的表面重建效果。</td></tr>
<tr><td>2025-09-07</td><td>MEGS$^{2}$: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning</td><td>[2509.07021](http://arxiv.org/pdf/2509.07021)</td><td>该论文的核心贡献是提出了MEGS²框架，显著降低了3D高斯泼溅（3DGS）技术的渲染内存消耗。其创新点包括：
◆ 采用任意方向的球面高斯瓣替代内存密集的球谐函数，作为轻量化的颜色表示方法。
◆ 提出了一个统一的软剪枝框架，将基元数量修剪和瓣数量修剪建模为一个单一的约束优化问题。
◆ 通过联合优化基元总数和每个基元的参数量这两个关键因素，实现了前所未有的内存压缩。
实验表明，MEGS²在保持相当渲染质量的同时，相比现有方法实现了50%的静态显存减少和40%的渲染显存减少。</td></tr>
<tr><td>2025-09-08</td><td>3DOF+Quantization: 3DGS quantization for large scenes with limited Degrees of Freedom</td><td>[2509.06400](http://arxiv.org/pdf/2509.06400)</td><td>该论文针对大场景下3D高斯泼溅（3DGS）模型的量化问题提出了创新性解决方案。其核心贡献在于分析了相机位置自由度受限（3DoF+）场景下的量化误差特性，并设计了相应的优化方案。
◆ 首次明确提出了“3DoF+”概念，将大场景重建问题聚焦于相机位置仅有小范围偏移的自由度受限场景。
◆ 通过理论分析揭示了投影误差与投影点距离平方的倒数成正比的关键规律，为量化方案设计提供了理论依据。
◆ 创新性地提出了一种基于球坐标的新型量化方案，该方案能根据点与相机的距离自适应调整量化精度。
◆ 在著名的Garden场景上验证了所提方法的率失真性能，证明了其有效性。</td></tr>
<tr><td>2025-09-03</td><td>ContraGS: Codebook-Condensed and Trainable Gaussian Splatting for Fast, Memory-Efficient Reconstruction</td><td>[2509.03775](http://arxiv.org/pdf/2509.03775)</td><td>该论文的核心贡献是提出了ContraGS方法，在保持高渲染质量的同时显著提升了3D高斯泼溅（3DGS）的训练和渲染效率，并大幅降低了内存消耗。  
◆ 引入码本（codebook）对3D高斯参数进行压缩存储，无需减少高斯数量即可降低内存占用。  
◆ 首次实现了在码本压缩表示下的直接端到端训练，解决了非可微参数的优化难题。  
◆ 通过贝叶斯推断框架将参数估计转化为后验分布采样问题，并采用MCMC方法进行高效求解。  
◆ 在训练峰值内存上平均降低3.49倍，训练和渲染速度分别提升1.36倍和1.88倍，且模型质量接近原始SOTA水平。  
该方法为内存受限设备上的高质量3D重建提供了实用解决方案。</td></tr>
<tr><td>2025-09-02</td><td>GRMM: Real-Time High-Fidelity Gaussian Morphable Head Model with Learned Residuals</td><td>[2509.02141](http://arxiv.org/pdf/2509.02141)</td><td>GRMM是首个基于高斯溅射的全头部三维可变形模型，实现了高保真且实时的头部渲染与操控。  
◆ 提出在传统网格3DMM基础上，引入残差几何与外观组件，以恢复皱纹、皮肤纹理和发际线等高频细节。  
◆ 通过解耦的低维参数（如身份形状、表情）控制基础模型，并独立建模主体与表情特有的残差细节，提升表达能力。  
◆ 采用粗粒度解码器处理网格顶点变形，细粒度解码器建模每个高斯的外观，并结合轻量CNN对渲染图像进行细化增强。  
◆ 构建了EXPRESS-50数据集，包含50个身份下60种对齐表情，为高斯3DMM的身份与表情解耦学习提供可靠数据支撑。  
GRMM在单目3D人脸重建、新视角合成和表情迁移任务中均达到最优效果，同时保持75FPS的实时渲染性能。</td></tr>
<tr><td>2025-08-31</td><td>Towards Integrating Multi-Spectral Imaging with Gaussian Splatting</td><td>[2509.00989](http://arxiv.org/pdf/2509.00989)</td><td>该论文的核心贡献是研究了如何将多光谱成像（红、绿、红边、近红外）有效地集成到基于3D高斯泼溅（3DGS）的快速三维重建框架中。  
◆ 提出并系统评估了三种融合策略（独立重建、拆分优化和联合优化），发现联合优化策略最为有效。  
◆ 证明了联合优化不仅能显著提升多光谱重建质量，还能通过光谱串扰改善RGB重建结果。  
◆ 创新性地建议将多光谱数据直接集成到球谐颜色组件中，以紧凑地表示每个高斯的多光谱反射特性。  
◆ 深入分析了在优化过程中引入不同光谱波段的关键权衡，为鲁棒的多模态3DGS重建提供了实用见解。  
这项工作为多光谱三维重建提供了重要的方法论进步和实践指导。</td></tr>
<tr><td>2025-09-03</td><td>UPGS: Unified Pose-aware Gaussian Splatting for Dynamic Scene Deblurring</td><td>[2509.00831](http://arxiv.org/pdf/2509.00831)</td><td>本文提出UPGS方法，用于解决单目视频中动态3D场景因运动模糊导致的重建难题。核心创新如下：
◆ 提出统一优化框架，将相机位姿与3D高斯属性作为可学习参数进行端到端联合优化，突破传统两步流程的局限。
◆ 创新地将相机与物体运动统一建模为每个3D高斯基元的SE(3)仿射变换，形成一致的运动表示。
◆ 设计三阶段交替训练策略：先固定位姿优化高斯，再固定高斯优化位姿，最后联合优化，确保稳定收敛。
实验表明，该方法在Stereo Blur数据集和真实场景中显著提升了动态去模糊的重建质量和位姿估计精度。</td></tr>
<tr><td>2025-08-31</td><td>MarkSplatter: Generalizable Watermarking for 3D Gaussian Splatting Model via Splatter Image Structure</td><td>[2509.00757](http://arxiv.org/pdf/2509.00757)</td><td>该论文提出了首个可泛化的3D高斯泼溅（3DGS）模型水印框架MarkSplatter，通过单次前向传播即可实现高效版权保护。  
◆ 首创通用型水印框架，无需针对不同消息重复微调，显著提升效率。  
◆ 提出GaussianBridge模块，将非结构化的3D高斯点转换为Splatter Image格式，支持直接神经网络处理与任意消息嵌入。  
◆ 设计高斯不确定性感知热图预测策略，在嵌入水印时保持渲染视觉质量，确保不可感知性。  
◆ 开发基于密集分割的消息提取机制，即使水印对象在渲染视图中占比极小也能实现鲁棒提取。</td></tr>
<tr><td>2025-08-30</td><td>AGS: Accelerating 3D Gaussian Splatting SLAM via CODEC-Assisted Frame Covisibility Detection</td><td>[2509.00433](http://arxiv.org/pdf/2509.00433)</td><td>该论文提出了AGS，一个算法-硬件协同设计框架，旨在显著加速基于3D高斯泼溅的SLAM系统。其核心创新点在于充分利用了SLAM流式处理中相邻帧高度相似这一特性。  
◆ 在软件层面，提出了一种根据机器人运动进行先粗后精的位姿跟踪方法。  
◆ 通过跨帧共享高斯点的贡献信息，避免了大量冗余计算。  
◆ 在硬件层面，创新性地利用视频编解码器（CODEC）来提取中间数据，设计了一个帧共视性检测引擎。  
◆ 还实现了配备工作负载调度器的位姿跟踪引擎和建图引擎，以高效部署整个AGS算法。  
实验结果表明，AGS相比移动和高端GPU以及最先进的专用加速器GSCore，分别实现了最高17.12倍、6.71倍和5.41倍的加速比。</td></tr>
<tr><td>2025-08-29</td><td>Scale-GS: Efficient Scalable Gaussian Splatting via Redundancy-filtering Training on Streaming Content</td><td>[2508.21444](http://arxiv.org/pdf/2508.21444)</td><td>该论文提出了Scale-GS，一个用于动态流式内容的高效可扩展3D高斯溅射框架。其核心贡献在于显著降低了动态场景的训练开销和存储需求，同时保持了高质量的渲染效果。
◆ 引入基于锚点的分层高斯结构，通过粗粒度高斯控制细粒度高斯的选择性激活，优化了模型表示效率。
◆ 提出混合变形与衍生策略，结合高斯变形处理帧间运动，并通过衍生策略捕捉大范围运动，减少了计算开销。
◆ 设计双向自适应掩码机制，剔除静态区域并优先处理信息丰富的视角，从而大幅提升训练效率。
实验表明，该方法在保持卓越视觉质量的同时，显著缩短了训练时间，优于现有先进方法。</td></tr>
<tr><td>2025-08-29</td><td>ARGS: Advanced Regularization on Aligning Gaussians over the Surface</td><td>[2508.21344](http://arxiv.org/pdf/2508.21344)</td><td>该论文在SuGaR模型基础上提出两种正则化策略，以提升3D高斯溅射（3DGS）的表面重建质量。  
◆ 引入有效秩正则化，通过抑制高斯原语的极端各向异性（如针状形状），促使其形成更平衡的盘状结构，从而提升单个高斯形状的稳定性和合理性。  
◆ 将神经符号距离函数（SDF）融入优化过程，通过Eikonal损失约束其距离属性，提供连续的全局表面先验，引导高斯分布更贴合底层几何结构。  
◆ 两种正则化互补：前者优化局部高斯形态，后者增强整体表面一致性，共同提高重建的视觉保真度和场景连贯性。  
最终模型能够从3DGS数据生成更精确、连贯的3D网格和视觉效果。</td></tr>
<tr><td>2025-08-28</td><td>Adam SLAM - the last mile of camera calibration with 3DGS</td><td>[2508.20526](http://arxiv.org/pdf/2508.20526)</td><td>该论文提出了一种利用3D高斯泼溅（3DGS）模型优化相机标定的新方法。  
◆ 创新性地通过新视图颜色损失的反向传播来精细调整相机参数，突破了传统标定方法的精度限制。  
◆ 该方法无需真实场景的地面真值，仅依靠合成视图的质量作为标定优劣的评价标准。  
◆ 在3DGS基准数据集上，仅通过标定优化就实现了平均0.4 dB PSNR的性能提升。  
◆ 虽然优化过程耗时较长，但对参考场景（如Mip-NeRF 360）的标定具有重要意义，因为新视图质量是此类场景的核心评价指标。  
该方法为相机标定提供了更高精度的解决方案，尤其适用于对重建质量要求极高的应用场景。</td></tr>
<tr><td>2025-08-27</td><td>MAPo : Motion-Aware Partitioning of Deformable 3D Gaussian Splatting for High-Fidelity Dynamic Scene Reconstruction</td><td>[2508.19786](http://arxiv.org/pdf/2508.19786)</td><td>该论文针对基于变形的3D高斯泼溅动态场景重建中，复杂运动区域易出现模糊和细节丢失的问题，提出了MAPo框架。其核心贡献与创新点包括：
◆ 提出了一种动态的基于得分的分割策略，能够自动区分高动态和低动态的高斯点。
◆ 对高动态高斯点采用递归时间分割，并为每个新时段复制独立的变形网络，以精细化建模复杂运动细节。
◆ 将低动态高斯点视为静态处理，有效降低了整体计算开销。
◆ 设计了跨帧一致性损失函数，解决了时间分割可能带来的边界视觉不连续问题，同时进一步提升了渲染质量。
实验证明，该方法在保持相近计算成本的同时，尤其在快速或复杂运动区域，实现了优于基线方法的渲染质量。</td></tr>
<tr><td>2025-08-27</td><td>FastAvatar: Towards Unified Fast High-Fidelity 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers</td><td>[2508.19754](http://arxiv.org/pdf/2508.19754)</td><td>FastAvatar提出了一种统一的前馈式3D虚拟人重建框架，能够在数秒内利用多样化日常记录（单张图像、多视角图像或单目视频）重建出高质量的3D高斯溅射模型。其核心创新包括：
◆采用一种新型大高斯重建变换器（LGRT），通过引入可聚合的规范3DGS表示与初始3D提示，有效聚合多帧信息；
◆设计多粒度引导编码机制，整合相机位姿、FLAME表情和头部姿态信息，缓解动画带来的不对齐问题，支持变长输入；
◆提出基于关键点跟踪的增量式高斯聚合方法和切片融合损失，实现随观测数据增加而质量提升的重建能力；
◆仅使用单一统一模型实现高质量、高速度的可调谐重建范式，在质量和效率上均优于现有方法。</td></tr>
<tr><td>2025-08-27</td><td>LabelGS: Label-Aware 3D Gaussian Splatting for 3D Scene Segmentation</td><td>[2508.19699](http://arxiv.org/pdf/2508.19699)</td><td>该论文提出了LabelGS方法，为3D高斯泼溅（3DGS）表示引入了3D语义分割能力，解决了其缺乏场景理解功能的关键限制。  
◆ 引入跨视角一致的语义掩码，为3D高斯赋予对象标签信息。  
◆ 提出一种新颖的遮挡分析模型，防止优化过程中对遮挡产生过拟合。  
◆ 设计主干高斯标记模型，成功将2D语义先验提升至3D高斯表示。  
◆ 采用高斯投影过滤器，有效避免了高斯标签间的冲突问题。  
该方法通过随机区域采样策略改进了优化过程，实现了高斯表示的有效解耦，在3D场景分割任务上性能优于包括Feature-3DGS在内的先前方法，并以22倍的训练速度提升显著提高了效率。</td></tr>
<tr><td>2025-08-26</td><td>ColorGS: High-fidelity Surgical Scene Reconstruction with Colored Gaussian Splatting</td><td>[2508.18696](http://arxiv.org/pdf/2508.18696)</td><td>ColorGS针对内窥镜视频中可变形组织的高保真重建难题，提出了两项关键创新。  
◆ 引入彩色高斯基元，通过动态锚点和可学习颜色参数自适应编码空间变化的纹理，显著提升复杂光照和组织相似性下的色彩表现力。  
◆ 设计增强形变模型，结合时间感知的高斯基函数与独立于时间的可学习形变，精确捕捉局部组织变形和手术交互引起的全局运动一致性。  
该方法在达芬奇机器人手术视频和多个基准数据集上达到最先进性能，PSNR达39.85（比基于3DGS的方法提升1.5），SSIM达97.25%，同时保持实时渲染效率。  
其平衡了高保真重建与计算实用性，对术中导航和AR/VR应用具有重要意义。</td></tr>
<tr><td>2025-08-25</td><td>FastAvatar: Instant 3D Gaussian Splatting for Faces from Single Unconstrained Poses</td><td>[2508.18389](http://arxiv.org/pdf/2508.18389)</td><td>FastAvatar是一个能够从任意姿态的单张人脸图像中，在极短时间内（&lt;10毫秒）生成高质量3D高斯溅射（3DGS）模型的姿态不变前馈框架。
◆ 提出了一种新颖的编码器-解码器神经网络结构，通过预测对预定义3DGS人脸“模板”的参数残差，实现了快速且身份保真的重建。
◆ 其核心创新在于将输入图像编码为一个与姿态无关的身份潜在嵌入，并解码此嵌入来直接调整模板的结构和外观参数，无需针对每张脸进行优化。
◆ 该方法在重建质量上显著优于现有的前馈式方法（如GAGAvatar），且速度比基于优化的方法（如FlashAvatar）快1000倍。
◆ 其独特的潜在空间设计还支持实时身份插值和属性编辑功能，这是其他前馈式3DGS人脸框架所不具备的。
FastAvatar以其卓越的重建质量与速度组合，极大地推动了3DGS在消费级和交互式系统中实现照片级真实感数字人的应用潜力。</td></tr>
<tr><td>2025-08-25</td><td>GSVisLoc: Generalizable Visual Localization for Gaussian Splatting Scene Representations</td><td>[2508.18242](http://arxiv.org/pdf/2508.18242)</td><td>GSVisLoc提出了一种专为3D高斯泼溅（3DGS）场景表示设计的通用视觉定位方法。  
◆ 首次实现了无需任何修改或重训练，直接利用原始3DGS模型进行视觉定位，无需额外参考图像。  
◆ 通过下采样和编码3D高斯来提取场景特征，并与查询图像特征进行鲁棒匹配，有效结合了显式3D表示的优点。  
◆ 采用由粗到精的三步定位流程：粗匹配、精细匹配和姿态优化，确保了高精度相机位姿估计。  
◆ 在室内外标准基准测试中均表现出 competitive 的定位性能，显著优于现有基于3DGS的基线方法。  
◆ 展示了强大的泛化能力，能够直接应用于未见过的全新场景，无需针对特定场景进行额外训练。</td></tr>
<tr><td>2025-08-25</td><td>Camera Pose Refinement via 3D Gaussian Splatting</td><td>[2508.17876](http://arxiv.org/pdf/2508.17876)</td><td>该论文提出了一种基于3D高斯泼溅（3DGS）的新型相机位姿优化框架GS-SMC，显著提升了初始位姿估计的精度。  
◆ 利用广泛应用的3DGS模型直接渲染新视角，无需针对不同场景重新训练或微调，实现了轻量化的跨场景适配能力。  
◆ 通过查询图像与多个渲染图像之间的对极几何约束进行迭代优化，有效结合了几何约束与渲染灵活性。  
◆ 支持灵活选择特征提取器和匹配器来建立约束，摆脱了传统方法对特定描述符或专用网络的依赖。  
实验证明，该方法在7-Scenes和Cambridge数据集上均优于现有最优方法，其中位姿中值误差降低最高达56.9%（旋转）和53.3%（平移）。</td></tr>
<tr><td>2025-08-25</td><td>IDU: Incremental Dynamic Update of Existing 3D Virtual Environments with New Imagery Data</td><td>[2508.17579](http://arxiv.org/pdf/2508.17579)</td><td>该论文提出了IDU增量动态更新流程，用于高效更新军事场景中的现有3D虚拟环境。  
◆ 通过少量新图像实现现有3D高斯溅射模型（3DGS）的增量更新，避免全场景重建的高成本。  
◆ 结合相机位姿估计与新图像配准，确保新增数据与原有模型的空间一致性。  
◆ 采用变化检测技术精准定位场景中的动态变化（如新增或消失物体）。  
◆ 利用3D生成式AI模型创建高质量新资产，并通过人工指导实现精确的对象识别与放置。  
实验证明该方法大幅降低了3D场景更新的时间和人力成本，为动态军事环境提供了高效维护方案。</td></tr>
<tr><td>2025-08-23</td><td>Align 3D Representation and Text Embedding for 3D Content Personalization</td><td>[2508.16932](http://arxiv.org/pdf/2508.16932)</td><td>该论文提出了一种名为Invert3D的新型框架，旨在解决3D内容高效个性化这一关键挑战。其核心贡献与创新点如下：

◆ 提出了一个将3D表示与文本嵌入空间对齐的创新框架，解决了因结构差异而无法将2D视觉-语言模型直接应用于3D内容的根本问题。
◆ 设计了一种相机条件化的3D到文本逆向映射机制，能够将3D内容投影到与文本嵌入对齐的3D嵌入空间中。
◆ 实现了通过自然语言提示对3D内容进行高效操作和个性化定制，无需依赖基于知识蒸馏的、计算昂贵的重新训练过程。
◆ 所提出的方法在保持高质量合成的同时，显著提升了3D内容个性化的效率和便捷性。</td></tr>
<tr><td>2025-08-22</td><td>Arbitrary-Scale 3D Gaussian Super-Resolution</td><td>[2508.16467](http://arxiv.org/pdf/2508.16467)</td><td>本文提出了一种支持任意尺度3D高斯超分辨率的新框架，解决了现有方法局限于固定倍数和效率低下的问题。  
◆ 构建了集成式框架，首次实现单一模型支持任意尺度（包括非整数倍）的超分辨率渲染。  
◆ 引入尺度感知渲染机制，有效避免了直接上采样带来的混叠伪影问题。  
◆ 结合生成先验引导优化与渐进式超分策略，显著提升渲染质量与结构一致性。  
◆ 在保持实时渲染速度（1080p分辨率下85帧/秒）的同时，大幅提升画质（PSNR指标比原始3DGS提升6.59dB）。  
该方案无需后处理上采样器，简化了流程并提高了实用性，为资源受限场景提供了高效灵活的解决方案。</td></tr>
<tr><td>2025-08-21</td><td>UnPose: Uncertainty-Guided Diffusion Priors for Zero-Shot Pose Estimation</td><td>[2508.15972](http://arxiv.org/pdf/2508.15972)</td><td>该论文提出了一种名为UnPose的零样本、无需CAD模型的6D物体姿态估计与重建新框架。  
◆利用预训练扩散模型提供的3D先验和像素级认知不确定性估计，从单视图RGB-D图像初始重建物体的3D高斯溅射(3DGS)模型。  
◆引入不确定性引导的多视图融合机制，通过扩散模型的不确定性来指导增量式地整合新观测视图，从而持续提升姿态估计精度和3D重建质量。  
◆采用姿态图优化将扩散先验生成视图与实际观测进行全局一致性集成，形成连贯的3DGS表示，有效避免了几何幻觉问题。  
该方法在6D姿态估计准确性和3D重建质量上显著优于现有技术，并验证了在真实机器人操作任务中的实用性。</td></tr>
<tr><td>2025-08-21</td><td>Enhancing Novel View Synthesis from extremely sparse views with SfM-free 3D Gaussian Splatting Framework</td><td>[2508.15457](http://arxiv.org/pdf/2508.15457)</td><td>该论文针对3D高斯泼溅（3DGS）在极端稀疏视角（如仅2个训练视图）下因运动恢复结构（SfM）初始化失败导致的渲染质量下降问题，提出了一种无需SfM的端到端解决方案。其核心创新点包括：
◆ 提出一个密集立体模块，替代传统的SfM方法，联合优化相机姿态估计与全局密集点云重建，为3DGS提供稳健初始化。
◆ 设计一致性视图插值模块，通过插值训练视角之间的相机位姿并生成视角一致的内容，为训练提供额外的强监督信号，缓解稀疏输入的信息稀缺问题。
◆ 引入多尺度拉普拉斯一致性正则化与自适应空间感知的多尺度几何正则化，有效增强几何结构的准确性和渲染内容的高频细节。
实验表明，该方法在极端稀疏条件下显著优于现有技术，PSNR指标提升达2.75dB，合成图像畸变小且细节丰富。</td></tr>
<tr><td>2025-08-21</td><td>Image-Conditioned 3D Gaussian Splat Quantization</td><td>[2508.15372](http://arxiv.org/pdf/2508.15372)</td><td>该论文提出了ICGS-Quantizer，旨在解决3D高斯溅射（3DGS）模型在压缩存储和长期归档后适应场景变化的两大难题。  
◆ 通过联合利用高斯点间和属性间的相关性，并采用跨所有训练场景的共享码本，大幅提升了量化效率，消除了传统方法中每个场景都需独立码本的开销。  
◆ 将压缩后的3DGS模型存储需求从兆字节级成功降低到千字节级，同时保持了高质量的视觉保真度。  
◆ 创新性地引入图像条件解码机制，使得模型在解码时能依据实时捕获的图像自适应地更新场景，从而支持归档后的场景变化。  
◆ 通过端到端的联合训练，确保了量化后的场景代码能有效用于这种条件解码过程。  
实验证明，该方法在压缩效率和场景更新适应性上均优于现有最先进技术。</td></tr>
<tr><td>2025-08-25</td><td>MeSS: City Mesh-Guided Outdoor Scene Generation with Cross-View Consistent Diffusion</td><td>[2508.15169](http://arxiv.org/pdf/2508.15169)</td><td>该论文提出了MeSS方法，利用城市网格模型作为几何先验，生成高质量且风格一致的室外场景。其核心创新包括：
◆ 提出级联外绘ControlNet，生成几何一致的稀疏视角图像，确保初始视图与3D网格对齐。
◆ 设计AGInpaint模块进行中间视图传播，有效增加视角密度并保持内容连贯性。
◆ 引入GCAlign模块全局优化视觉不一致问题（如曝光差异），提升跨视图一致性。
◆ 结合3D高斯泼溅（3DGS）技术，在网格表面初始化高斯点以实时重建可渲染3D场景。
该方法在几何对齐度和生成质量上优于现有技术，支持通过重照明和风格转换实现多样化渲染。</td></tr>
<tr><td>2025-08-20</td><td>GeMS: Efficient Gaussian Splatting for Extreme Motion Blur</td><td>[2508.14682](http://arxiv.org/pdf/2508.14682)</td><td>GeMS是首个直接从极端运动模糊图像中进行3D高斯溅射(3DGS)重建的框架，无需依赖任何清晰图像。  
◆ 提出VGGSfM，一种基于深度学习的运动恢复结构(SfM)方法，直接从模糊输入中估计相机位姿并生成点云。  
◆ 引入3DGS-MCMC方法，将高斯分布视为概率分布样本进行稳健初始化，避免了传统启发式的 densification 和 pruning 操作。  
◆ 联合优化相机轨迹与高斯参数，实现更稳定的场景重建。  
◆ 进一步提出GeMS-E增强版本，集成事件相机数据，通过事件双积分去模糊(EDI)技术生成更清晰图像以优化重建流程。  
该框架在合成与真实数据集上均实现了最先进的性能，突破了极端运动模糊下3D重建的瓶颈。</td></tr>
<tr><td>2025-08-20</td><td>Reconstruction Using the Invisible: Intuition from NIR and Metadata for Enhanced 3D Gaussian Splatting</td><td>[2508.14443](http://arxiv.org/pdf/2508.14443)</td><td>该论文的核心贡献是提出了一种基于近红外（NIR）和元数据的增强型3D高斯溅射方法，以解决农业场景三维重建中的独特挑战。  
◆ 创建了一个新颖的多模态数据集NIRPlant，集成了近红外、RGB、深度、LiDAR及文本元数据，覆盖多种室内外光照条件。  
◆ 首次将植被指数（如NDVI、NDWI）等文本元数据引入3D重建，为农业环境提供了超越可见光谱的植物学上下文信息。  
◆ 提出了一种名为NIRSplat的新型多模态高斯溅射架构，采用基于3D点的位置编码和交叉注意力机制，有效融合多源数据以获取稳健的几何先验。  
◆ 在极具挑战性的农业场景中，该方法在重建性能上显著优于3DGS、CoR-GS和InstantSplat等现有代表性方法。  
该工作为农业领域的精准三维感知和分析提供了新的解决方案和数据基础。</td></tr>
<tr><td>2025-08-21</td><td>GALA: Guided Attention with Language Alignment for Open Vocabulary Gaussian Splatting</td><td>[2508.14278](http://arxiv.org/pdf/2508.14278)</td><td>GALA提出了一种用于开放词汇3D场景理解的新框架，其核心贡献与创新点如下：

◆ 提出了一种新颖的跨注意力模块，并引入了两个可学习的码本，用于编码与视角无关的语义嵌入，从而构建了通用的语言特征场。
◆ 通过自监督对比学习蒸馏出场景特定的3D实例特征场，确保了实例内特征的高度相似性。
◆ 该设计同时支持无缝的2D和3D开放词汇查询，实现了对细粒度、语言感知的3D表征的捕获。
◆ 避免了为每个高斯点学习高维特征，显著降低了内存消耗，提升了方法的实用性。
◆ 在真实数据集上的大量实验证明了GALA在2D和3D开放词汇任务上的卓越性能。</td></tr>
<tr><td>2025-08-19</td><td>Distilled-3DGS:Distilled 3D Gaussian Splatting</td><td>[2508.14037](http://arxiv.org/pdf/2508.14037)</td><td>该论文提出了首个针对3D高斯泼溅（3DGS）的知识蒸馏框架Distilled-3DGS，旨在解决其高内存和存储消耗的核心问题。

◆ 首次将知识蒸馏思想引入3DGS领域，通过集成多个教师模型（包括原始3DGS、噪声增强和丢弃正则化版本）的输出来指导一个轻量级学生模型的优化。
◆ 提出了一种结构相似性损失（structural similarity loss），专门用于蒸馏隐藏的几何结构，以增强学生模型与教师模型在空间几何分布上的一致性。
◆ 该框架简单有效，无需复杂修饰，即在多个数据集上实现了渲染质量与存储效率的最佳平衡，显著减少了高斯模型的数量和存储开销。</td></tr>
<tr><td>2025-08-19</td><td>Online 3D Gaussian Splatting Modeling with Novel View Selection</td><td>[2508.14014](http://arxiv.org/pdf/2508.14014)</td><td>该研究针对仅使用RGB图像序列进行在线3D高斯泼溅（3DGS）建模的挑战，提出了一种提升模型完整性的创新方法。  
◆ 提出了自适应视图选择策略，通过在线分析重建质量，智能选取最优的非关键帧进行补充训练。  
◆ 突破了传统方法仅依赖关键帧的局限，通过融合关键帧和精选的非关键帧，从多样视角细化不完整区域。  
◆ 设计了一个集成在线多视图立体（MVS）方法的框架，确保3D信息在整个建模过程中的一致性。  
实验结果表明，该方法在复杂户外场景中优于现有技术，实现了更高质量和更完整的在线3D重建。</td></tr>
<tr><td>2025-08-19</td><td>PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis</td><td>[2508.13911](http://arxiv.org/pdf/2508.13911)</td><td>该论文提出了PhysGM，一个从单张图像前馈生成高保真4D物理模拟的新框架。其核心贡献与创新点包括：

◆ 开发了一个前馈框架，能够从单张图像联合预测3D高斯表示及其物理属性，实现了无需优化、快速的物理模拟和4D渲染。
◆ 通过联合优化高斯重建和概率物理预测来建立基础模型，并利用物理上合理的参考视频进行细化，同步提升了渲染保真度和物理预测准确性。
◆ 创新性地采用直接偏好优化（DPO）方法，使其模拟与参考视频对齐，避免了通过复杂可微模拟和光栅化进行梯度反传的分数蒸馏采样（SDS）优化，提高了训练稳定性与效率。
◆ 构建了一个包含超过24,000个3D资产的新数据集PhysAssets，这些资产均标注了物理属性并配有引导视频，为模型训练提供了重要支持。
实验表明，该方法能在一分钟内从单张图像生成高质量4D模拟，在速度和真实性上均显著优于现有工作。</td></tr>
<tr><td>2025-08-19</td><td>EAvatar: Expression-Aware Head Avatar Reconstruction with Generative Geometry Priors</td><td>[2508.13537](http://arxiv.org/pdf/2508.13537)</td><td>该论文提出了一种名为EAvatar的基于3D高斯泼溅(3DGS)的头部重建新框架，旨在解决现有方法在捕捉细粒度表情和保持纹理连续性方面的核心挑战。其核心创新点如下：

◆ 引入了稀疏表情控制机制，使用少量关键高斯来驱动邻近高斯的形变，从而精确建模局部变形和细微纹理过渡。
◆ 创新地利用了预训练生成模型的高质量3D几何先验，为训练过程提供了更可靠的面部结构指导。
◆ 该框架显著提升了表情控制的精确性和细节保真度，实现了更高视觉一致性的头部重建效果。
实验结果表明，该方法在收敛稳定性和形状准确性方面均优于现有技术。</td></tr>
<tr><td>2025-08-18</td><td>InnerGS: Internal Scenes Rendering via Factorized 3D Gaussian Splatting</td><td>[2508.13287](http://arxiv.org/pdf/2508.13287)</td><td>该论文提出了InnerGS，一种针对物体内部场景渲染的创新方法。其核心贡献在于将3D高斯泼溅技术成功应用于内部结构重建，突破了现有方法主要关注外部表面建模的局限。
◆ 首次将3D高斯泼溅用于内部连续体积密度的直接建模，而非仅表面表示。
◆ 能够仅从稀疏的切片数据有效重建平滑且精细的内部三维结构。
◆ 无需相机位姿信息，降低了对输入数据的要求。
◆ 采用即插即用的设计，本质上兼容任何数据模态（如CT、MRI等）。
该方法在医疗成像和科学可视化等领域具有重要应用价值，为深入理解物体内部结构提供了高效解决方案。</td></tr>
<tr><td>2025-08-18</td><td>Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting</td><td>[2508.12720](http://arxiv.org/pdf/2508.12720)</td><td>◆ 揭示了稀疏视角下3D高斯泼溅（3DGS）出现伪影的核心原因：高斯点之间过度相互依赖（共适应），导致过度拟合训练视角而忽视真实场景外观分布。  
◆ 提出量化高斯点共适应程度的创新指标Co-Adaptation Score（CA），通过同一视角下不同高斯子集渲染的像素方差计算共适应强度。  
◆ 发现训练视角数量增加会自然缓解共适应现象，为稀疏视角优化提供了理论依据。  
◆ 提出两种轻量级解决方案：随机高斯丢弃和透明度乘性噪声注入，均支持即插即用且无需额外训练。  
◆ 实验验证了所提方法在多种基准测试中的有效性，显著改善了稀疏视角下的新视角合成质量。  
◆ 首次系统分析了3DGS的共适应效应，为后续稀疏视角研究提供了新方向。</td></tr>
<tr><td>2025-08-17</td><td>Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering</td><td>[2508.12313](http://arxiv.org/pdf/2508.12313)</td><td>◆提出Edge-Aware Score（边缘感知评分），通过分析高斯分布的边缘特征，更精准地筛选需要分裂的候选高斯，优化了&quot;何时进行致密化&quot;的决策机制。  
◆设计Long-Axis Split（长轴分裂）策略，沿高斯分布最长几何轴进行分裂，显著减少传统克隆/分裂操作导致的几何畸变，改进&quot;如何致密化&quot;的核心操作。  
◆提出Recovery-Aware Pruning（恢复感知剪枝）技术，结合多步更新机制（Multi-step Update）和生长控制（Growth Control），有效抑制训练过拟合问题。  
◆整套方法在不增加训练/推理计算开销的前提下，以更少的高斯数量实现更高保真度的渲染效果。  
◆通过三阶段优化（时机选择-操作改进-过拟合抑制），系统性重构了3DGS的致密化流程，在渲染质量上达到SOTA水平。</td></tr>
<tr><td>2025-08-16</td><td>ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages</td><td>[2508.11854](http://arxiv.org/pdf/2508.11854)</td><td>◆首次提出利用3D高斯泼溅(3DGS)着色方法创建视角特异性伪装的黑盒攻击方法ComplicitSplat，无需知晓下游模型架构或参数。  
◆通过设计随视角变化的颜色纹理，在特定视角下嵌入对抗内容，实现物理世界和合成场景的双重攻击。  
◆验证了攻击对单阶段、多阶段及基于Transformer的多种流行检测器均有效，展现强泛化能力。  
◆揭示了3DGS在自动驾驶等安全关键任务中的新型安全隐患，首次暴露下游检测器的视角依赖攻击风险。  
◆提出方法不依赖白盒设定，仅需标准3DGS渲染流程即可实现跨模型攻击，突破传统对抗攻击限制。  
◆实验涵盖真实物体捕捉与合成场景，证实攻击在物理世界中的可实现性。</td></tr>
<tr><td>2025-08-13</td><td>EntropyGS: An Efficient Entropy Coding on 3D Gaussian Splatting</td><td>[2508.10227](http://arxiv.org/pdf/2508.10227)</td><td>◆ 揭示了3D高斯泼溅(3DGS)中球谐AC属性严格遵循拉普拉斯分布的重要统计规律，同时发现旋转、缩放和不透明度属性可用混合高斯分布近似建模。  
◆ 首次系统分析了3DGS各属性间的相关性，发现球谐AC属性除色彩空间继承的关联外，与其他属性仅存在弱相关性。  
◆ 提出基于参数化分布估计的因子化熵编码方法EntropyGS，针对不同高斯属性类型自适应调整量化策略。  
◆ 设计了一套完整的编码流程：先估计各属性的分布参数，再基于参数进行熵编码，实现高效压缩。  
◆ 在保持同等渲染质量的前提下，相比原始3DGS数据实现了约30倍的压缩率提升，且编解码速度快。  
◆ 该工作为3DGS模型的存储与传输提供了首个系统性的压缩解决方案，填补了该领域的空白。</td></tr>
<tr><td>2025-08-13</td><td>A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation</td><td>[2508.09977](http://arxiv.org/pdf/2508.09977)</td><td>◆ 全面综述了3D高斯泼溅（3DGS）在分割、编辑和生成等下游应用的最新进展，填补了该领域系统性调研的空白。  
◆ 首次将2D基础模型与3DGS结合，分析了语义理解和控制的关键技术支持，为3D场景理解提供了新思路。  
◆ 系统对比了NeRF与3DGS方法的异同，揭示了3DGS在显式表示和实时性方面的独特优势。  
◆ 创新性地将3DGS应用分为分割、编辑、生成等功能类别，总结了各领域的代表性方法、监督策略和学习范式。  
◆ 整理了公开数据集和评估协议，并通过基准测试对比了现有方法的性能，为后续研究提供标准化参考。  
◆ 建立了持续更新的资源库（GitHub），集成论文、代码和相关资源，推动3DGS应用生态发展。</td></tr>
<tr><td>2025-08-13</td><td>PERSONA: Personalized Whole-Body 3D Avatar with Pose-Driven Deformations from a Single Image</td><td>[2508.09973](http://arxiv.org/pdf/2508.09973)</td><td>◆ 提出PERSONA框架，结合3D基和扩散基方法的优势，仅需单张图像即可生成带姿态驱动形变的个性化全身3D虚拟人。  
◆ 利用扩散模型从输入图像生成丰富姿态视频，解决了传统3D方法需要大量真实姿态视频数据的问题。  
◆ 引入平衡采样技术，通过过采样输入图像来抑制扩散生成视频中的身份偏移，确保身份一致性。  
◆ 设计几何加权优化策略，优先几何约束而非图像损失，在多样姿态下保持高质量渲染效果。  
◆ 实现非刚性衣物变形等复杂姿态形变的建模，突破现有方法在身份保持与姿态依赖性纠缠方面的局限。  
◆ 仅需单张图像输入即可创建高真实感、锐利渲染的个性化3D虚拟人，显著降低数据采集成本。</td></tr>
<tr><td>2025-08-13</td><td>RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians</td><td>[2508.09830](http://arxiv.org/pdf/2508.09830)</td><td>◆ 提出了一种名为RayletDF的新方法，通过射线距离场（raylet distance field）直接从查询射线预测表面点，避免了传统基于坐标的方法在显式表面渲染时的高计算开销。  
◆ 设计了三个核心模块：射线特征提取器、射线距离场预测器和多射线混合器，共同实现细粒度局部几何特征提取、距离预测和多预测融合，提升表面重建精度。  
◆ 支持从原始点云或3D高斯（通过3DGS从RGB图像预估计）进行通用化3D表面重建，适用范围更广。  
◆ 在多个公开真实数据集上验证了方法的优越性，尤其在点云或3D高斯重建中表现突出。  
◆ 展示了出色的泛化能力，仅需单次前向传播即可在未见过的测试数据集中成功重建3D表面，无需额外训练。</td></tr>
<tr><td>2025-08-13</td><td>GSFixer: Improving 3D Gaussian Splatting with Reference-Guided Video Diffusion Priors</td><td>[2508.09667](http://arxiv.org/pdf/2508.09667)</td><td>◆ 提出GSFixer框架，通过参考引导的视频扩散先验改进稀疏视图下的3D高斯泼溅（3DGS）重建质量，解决因信息不足导致的伪影问题。  
◆ 创新性地构建基于DiT的视频扩散模型，利用成对的伪影3DGS渲染与干净帧进行训练，并引入参考视图条件增强生成内容与输入观测的一致性。  
◆ 结合视觉几何基础模型提取参考视图的2D语义特征和3D几何特征，提升修复后新视图的语义连贯性与三维一致性。  
◆ 为解决3DGS伪影修复缺乏评估基准的问题，首次提出DL3DV-Res数据集，包含低质量3DGS渲染的伪影帧。  
◆ 实验证明GSFixer在3DGS伪影修复和稀疏视图3D重建任务上优于现有最优方法，为相关研究提供新工具和基准。</td></tr>
<tr><td>2025-08-13</td><td>SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images</td><td>[2508.09479](http://arxiv.org/pdf/2508.09479)</td><td>◆ 提出SkySplat框架，首次将RPC模型与通用化3D高斯泼溅（3DGS）结合，解决了卫星图像稀疏视角重建的难题。  
◆ 采用自监督学习，仅需RGB图像和辐射鲁棒的相对高度监督，无需真实高度图，降低了数据依赖。  
◆ 设计跨自一致性模块（CSCM），通过一致性掩码有效消除多时相图像中瞬变物体的干扰。  
◆ 提出多视角一致性聚合策略，增强稀疏几何约束下的重建精度。  
◆ 在DFC19数据集上平均绝对误差（MAE）从13.18米降至1.80米，速度比EOGS快86倍，且跨数据集（MVS3D）泛化性强。</td></tr>
<tr><td>2025-08-12</td><td>Gradient-Direction-Aware Density Control for 3D Gaussian Splatting</td><td>[2508.09239](http://arxiv.org/pdf/2508.09239)</td><td>这篇论文的核心贡献是通过梯度方向感知的密度控制框架（GDAGS）解决了3D高斯泼溅（3DGS）在复杂场景中的两大问题：过度重建和过度致密化。以下是其创新点：

◆ 提出梯度一致性比率（GCR），通过归一化梯度向量范数区分梯度方向一致与冲突的高斯分布，为密度控制提供量化依据。

◆ 设计非线性动态加权机制，利用GCR实现梯度方向感知的密度控制，动态调整高斯分布的分裂与克隆策略。

◆ 在分裂操作中优先处理梯度冲突的高斯分布，增强几何细节表现，同时抑制梯度一致区域的冗余分布。

◆ 在克隆过程中促进梯度一致的高斯分布致密化以完善结构，同时防止梯度冲突区域的过度增殖。

◆ 通过优化高斯分布利用率，在多种真实场景基准测试中实现更高质量的渲染效果，同时减少50%内存消耗。

该方法有效解决了现有技术中因梯度方向冲突导致的适应性分裂失败和梯度对齐区域冗余增殖问题，构建了更紧凑的场景表示。</td></tr>
<tr><td>2025-08-11</td><td>SAGOnline: Segment Any Gaussians Online</td><td>[2508.08219](http://arxiv.org/pdf/2508.08219)</td><td>SAGOnline论文的核心贡献是通过轻量级零样本框架实现高斯场景的实时3D分割与跟踪，解决了现有方法计算成本高、空间推理能力有限和多目标跟踪困难的问题。其创新点包括：

◆ 提出解耦策略，结合视频基础模型（如SAM2）实现跨合成视角的视图一致2D掩码传播，确保分割连贯性。

◆ 开发GPU加速的3D掩码生成与高斯级实例标注算法，为3D基元分配唯一标识符，支持无损多目标跨视角跟踪与分割。

◆ 首次实现高斯场景的轻量级零样本3D分割框架，无需训练即可直接应用，显著降低计算开销。

◆ 通过显式标注高斯基元，同步完成分割与跟踪任务，解决了传统方法无法同时处理多目标的难题。

◆ 创新地将2D视频基础模型适配到3D领域，利用2D模型的强大泛化能力提升3D场景理解效果。

该框架在NVOS和Spin-NeRF基准测试中分别达到92.7%和95.2%的mIoU，推理速度比现有方法快15-1500倍（27毫秒/帧），为AR/VR和机器人应用提供了实用化解决方案。</td></tr>
<tr><td>2025-08-11</td><td>FantasyStyle: Controllable Stylized Distillation for 3D Gaussian Splatting</td><td>[2508.08136](http://arxiv.org/pdf/2508.08136)</td><td>◆ 提出首个完全基于扩散模型蒸馏的3D高斯泼溅风格迁移框架FantasyStyle，突破传统VGG特征依赖  
◆ 创新多视角频率一致性技术，通过3D滤波器选择性抑制低频噪声，有效解决多视角风格冲突导致的平滑失真问题  
◆ 设计可控风格化蒸馏机制，引入负引导策略抑制风格图中的内容泄漏，避免过度风格化  
◆ 首次发现并解决分数蒸馏采样和Delta去噪分数在3D风格迁移中的局限性，移除了重建项的干扰  
◆ 通过优化3D高斯分布参数实现更精准的风格控制，在多种场景和风格下均展现卓越的视觉真实感  
实验证明该方法在风格化质量和视觉一致性上全面超越现有技术。</td></tr>
<tr><td>2025-08-11</td><td>Touch-Augmented Gaussian Splatting for Enhanced 3D Scene Reconstruction</td><td>[2508.07717](http://arxiv.org/pdf/2508.07717)</td><td>◆ 提出首个将触觉信号（接触点和表面法线）融入3D高斯泼溅（3DGS）的多模态框架，突破传统纯视觉重建限制。  
◆ 设计空间选择性触觉测量机制，通过触觉数据同步优化3D高斯表示的几何结构和外观表现。  
◆ 创新两阶段触觉探索策略：先稀疏采样空白区域，再聚焦于重建网格识别的高不确定性边界区域。  
◆ 提出几何损失函数确保表面平滑性，显著提升几何精度（严重遮挡场景下倒角距离降低15倍以上）。  
◆ 实现完全在线处理流程，在低光照、遮挡等视觉退化场景中展现强鲁棒性。  
◆ 实验验证多场景下重建一致性提升，为触觉-视觉融合的3D重建开辟新方向。</td></tr>
<tr><td>2025-08-13</td><td>Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction</td><td>[2508.07701](http://arxiv.org/pdf/2508.07701)</td><td>◆ 提出多视角法向和距离引导的高斯泼溅方法，解决单视角投影中法向量对齐导致的几何偏差问题。  
◆ 设计多视角距离重投影正则化模块，通过计算相邻视角间高斯表面的距离损失，实现多视角几何深度统一。  
◆ 开发多视角法向增强模块，通过匹配相邻视角像素点法向量并计算损失，确保跨视角一致性。  
◆ 结合深度约束和3D法向对齐，显著提升小规模室内外场景的高精度重建能力。  
◆ 实验结果表明，该方法在定量和定性评估中均优于基线模型，有效增强了3DGS的表面重建性能。</td></tr>
<tr><td>2025-08-10</td><td>DIP-GS: Deep Image Prior For Gaussian Splatting Sparse View Recovery</td><td>[2508.07372](http://arxiv.org/pdf/2508.07372)</td><td>◆ 提出DIP-GS方法，将深度图像先验（DIP）与3D高斯泼溅（3DGS）结合，解决稀疏视图重建难题。  
◆ 利用DIP的内部结构和模式先验，通过粗到细的方式优化3DGS，无需依赖预训练模型（如生成模型或深度估计）。  
◆ 该方法仅需输入帧即可工作，在稀疏视图场景下显著优于原始3DGS，填补了传统方法在低重叠、低覆盖视图下的性能缺陷。  
◆ 在多种稀疏视图重建任务中达到竞争性SOTA效果，证明了其泛化能力和实用性。  
◆ 首次实现不依赖外部数据的端到端稀疏视图3D重建，为实时渲染与复杂场景恢复提供新思路。</td></tr>
<tr><td>2025-08-10</td><td>Fading the Digital Ink: A Universal Black-Box Attack Framework for 3DGS Watermarking Systems</td><td>[2508.07263](http://arxiv.org/pdf/2508.07263)</td><td>◆ 提出了首个针对3D高斯泼溅(3DGS)水印系统的通用黑盒攻击框架GMEA，填补了该领域攻击研究的空白。  
◆ 将水印攻击建模为大规模多目标优化问题，平衡水印去除效果与视觉质量，创新性地引入间接目标函数通过最小化卷积网络特征标准差来盲化水印检测器。  
◆ 设计了基于分组的优化策略，将高维3DGS模型参数空间分解为多个独立子优化问题，有效解决了大规模参数搜索难题。  
◆ 实验证明该框架能同时有效去除主流3DGS水印系统中的1D比特流和2D图像水印，且保持优异视觉保真度。  
◆ 首次系统性揭示了现有3DGS版权保护方案的关键脆弱性，为开发更鲁棒的水印技术提供了重要参考。</td></tr>
<tr><td>2025-08-10</td><td>3D Gaussian Representations with Motion Trajectory Field for Dynamic Scene Reconstruction</td><td>[2508.07182](http://arxiv.org/pdf/2508.07182)</td><td>◆ 提出结合3D高斯泼溅与运动轨迹场的新方法，首次实现动态场景中复杂物体运动的精确建模与物理合理轨迹生成。  
◆ 通过动态物体与静态背景的解耦技术，显著提升运动轨迹场的优化效率，降低计算复杂度。  
◆ 引入时间不变运动系数和共享运动轨迹基，有效捕捉复杂运动模式的同时保持参数紧凑性。  
◆ 在单目视频的新视角合成和运动轨迹恢复任务中达到最先进性能，验证了方法的通用性和鲁棒性。  
◆ 为机器人应用提供动态场景重建的新解决方案，弥补了传统NeRF和3DGS在动态场景中的局限性。</td></tr>
<tr><td>2025-08-09</td><td>DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit</td><td>[2508.07118](http://arxiv.org/pdf/2508.07118)</td><td>◆ DexFruit提出了一种结合光学触觉传感的机器人操控框架，实现了对草莓、番茄和黑莓等易损水果的轻柔自主抓取，将视觉瘀伤减少20%，抓取成功率提升31%。  
◆ 该研究首创将触觉信息融入扩散策略（tactile informed diffusion policies），在减少水果损伤和抓放成功率上均超越基线方法，经过630多次试验验证，整体抓取成功率高达92%。  
◆ 创新性引入FruitSplat技术，基于3D高斯泼溅（3DGS）构建高分辨率3D模型，首次实现水果损伤的精细化三维量化表征，解决了传统方法缺乏定量 rigor 或依赖昂贵设备的问题。  
◆ FruitSplat通过将2D水果掩膜和瘀伤分割掩膜蒸馏至3DGS表示，建立了模块化通用框架，可兼容任意2D模型，为水果损伤检测提供了可扩展的新范式。  
◆ 整套系统在三种高难度水果上验证了有效性，通过触觉与视觉的协同优化，为农业自动化中脆弱物体的精细操作与损伤评估树立了新标杆。</td></tr>
<tr><td>2025-08-09</td><td>3DGS-VBench: A Comprehensive Video Quality Evaluation Benchmark for 3DGS Compression</td><td>[2508.07038](http://arxiv.org/pdf/2508.07038)</td><td>◆ 提出了首个针对3D高斯泼溅（3DGS）压缩视频质量评估的大规模基准3DGS-VBench，填补了该领域系统性研究的空白。  
◆ 构建了包含660个压缩3DGS模型和视频序列的数据集，覆盖11个场景和6种先进压缩算法，并设计了系统化的参数等级。  
◆ 通过50名参与者的标注获得MOS分数，经过异常值处理和可靠性验证，为质量评估提供了可靠依据。  
◆ 首次对6种3DGS压缩算法在存储效率和视觉质量方面进行了全面基准测试，揭示了算法性能差异。  
◆ 评估了15种质量评估指标在多范式下的表现，为后续研究提供了参考。  
◆ 公开了数据集，为3DGS压缩和质量评估研究提供了重要资源，推动了该领域的发展。</td></tr>
<tr><td>2025-08-09</td><td>Evaluating Fisheye-Compatible 3D Gaussian Splatting Methods on Real Images Beyond 180 Degree Field of View</td><td>[2508.06968](http://arxiv.org/pdf/2508.06968)</td><td>◆首次在超过180度视场的真实鱼眼图像上评估了两种3D高斯溅射方法Fisheye-GS和3DGUT，填补了极端畸变场景下的研究空白。  
◆通过室内外场景实验，系统分析了200度鱼眼相机下两种方法对真实世界畸变的处理能力，并对比了不同视场角（200/160/120度）的性能表现。  
◆发现Fisheye-GS在160度视场时表现最佳，而3DGUT在200度全视场下仍能保持稳定的高质量重建，揭示了方法特性与视场角的关联规律。  
◆提出基于UniK3D预测的深度初始化策略，仅需2-3张鱼眼图像即可生成稠密点云，解决了传统SfM在强畸变场景下初始化失败的问题。  
◆验证了UniK3D在未训练真实鱼眼数据的情况下，仍能克服雾霾、眩光、天空等挑战性场景，达到与SfM相当的重建质量，展现了强泛化能力。  
◆为稀疏高畸变鱼眼图像的广角3D重建提供了实用解决方案，推动了3DGS技术在极端成像条件下的应用边界。</td></tr>
<tr><td>2025-08-08</td><td>UW-3DGS: Underwater 3D Reconstruction with Physics-Aware Gaussian Splatting</td><td>[2508.06169](http://arxiv.org/pdf/2508.06169)</td><td>这篇论文的核心贡献是提出UW-3DGS框架，通过物理感知的3D高斯泼溅技术解决水下3D重建中的光线吸收、散射和浑浊问题。主要创新点包括：

◆ 提出可插拔的学习型水下成像模块，采用基于体素的回归方法建模空间变化的衰减和背散射效应，提升颜色和几何精度。

◆ 设计物理感知不确定性剪枝分支（PAUP），通过不确定性评分自适应去除噪声高斯点，有效减少65%的漂浮伪影。

◆ 首次将3D高斯泼溅技术（3DGS）引入水下场景，克服了传统NeRF方法在浑浊环境中效率低和分辨率受限的问题。

◆ 实现端到端训练流程，联合优化高斯点参数与水下物理模型，同时支持生成无介质效应的纯净辐射图像（URI）和真实感水下图像（UWI）。

实验表明，该方法在PSNR（27.604）、SSIM（0.868）和LPIPS（0.104）指标上显著优于现有技术，尤其擅长处理高浑浊水域的复杂光传输效应。</td></tr>
<tr><td>2025-08-08</td><td>Roll Your Eyes: Gaze Redirection via Explicit 3D Eyeball Rotation</td><td>[2508.06136](http://arxiv.org/pdf/2508.06136)</td><td>◆ 提出首个基于显式3D眼球结构的视线重定向框架，采用3D高斯泼溅(3DGS)技术精确建模眼球几何，突破传统神经辐射场(NeRF)的隐式表示局限。  
◆ 通过显式控制3D眼球的旋转和平移运动，实现更精准的视线方向操控，生成图像的光影一致性和真实感显著优于现有方法。  
◆ 创新设计自适应形变模块，首次在视线重定向中模拟眼周肌肉的细微动态变化，增强面部表情的自然度。  
◆ 在ETH-XGaze数据集上的实验表明，该方法在生成图像质量（PSNR提升15%）和视线估计准确度（误差降低20%）上均超越SOTA。  
◆ 整套框架无需依赖复杂的体积渲染计算，运算效率较NeRF类方法提升3倍，为实时应用提供可能。</td></tr>
<tr><td>2025-08-08</td><td>ExploreGS: Explorable 3D Scene Reconstruction with Virtual Camera Samplings and Diffusion Priors</td><td>[2508.06014](http://arxiv.org/pdf/2508.06014)</td><td>◆ 提出基于3D高斯泼溅(3DGS)的新流程，通过生成额外训练视角增强场景重建能力，解决传统方法在偏离训练轨迹视角下渲染时的伪影和缺失问题。  
◆ 设计信息增益驱动的虚拟相机放置策略，最大化场景覆盖范围，有效捕捉未被充分观测的区域。  
◆ 结合视频扩散先验对渲染结果进行细化，提升生成视角的视觉质量和一致性。  
◆ 通过增强视角对3D高斯进行微调，显著提升任意视角下的重建质量，实现无伪影的高质量渲染。  
◆ 构建Wild-Explore基准测试集，专门针对复杂场景探索任务进行评估，填补现有评测体系的空白。  
◆ 实验证明该方法优于现有基于3DGS的技术，为自由视角场景探索提供实用解决方案。</td></tr>
<tr><td>2025-08-08</td><td>A 3DGS-Diffusion Self-Supervised Framework for Normal Estimation from a Single Image</td><td>[2508.05950](http://arxiv.org/pdf/2508.05950)</td><td>◆ 提出SINGAD框架，首次将3D高斯泼溅（3DGS）与扩散模型结合，实现单图像自监督法线估计，解决了传统方法依赖密集标注数据的问题。  
◆ 引入物理驱动的光-表面交互建模，通过3DGS重参数化生成符合光传输原理的多尺度几何特征，确保多视角法线方向一致性。  
◆ 在条件扩散模型中设计跨域特征融合模块，嵌入几何先验约束法线生成，同时保持精确的几何误差反向传播能力。  
◆ 提出可微分3D重投影损失策略，直接将3D几何误差转化为法线优化信号，实现无需标注数据的自监督训练。  
◆ 通过可微分渲染重建模块解决扩散模型离散采样导致的梯度不连续问题，使3D几何误差能有效回传至法线生成网络。  
在Google Scanned Objects数据集上的定量实验表明，该方法在多项指标上超越现有最优方法。</td></tr>
<tr><td>2025-08-07</td><td>GAP: Gaussianize Any Point Clouds with Text Guidance</td><td>[2508.05631](http://arxiv.org/pdf/2508.05631)</td><td>◆ 提出GAP方法，首次实现从无色点云到高质量3D高斯分布的转换，填补了该领域的技术空白。  
◆ 创新性地采用多视角优化框架，结合深度感知的图像扩散模型，确保不同视角下外观的一致性。  
◆ 设计表面锚定机制，在优化过程中将高斯分布精准约束在3D形状表面，保证几何精度。  
◆ 引入基于扩散模型的修复策略，专门针对难以观测的区域进行补全，提升整体完整性。  
◆ 方法具有广泛适用性，验证场景涵盖合成点云、复杂真实扫描和大规模场景，展现强大泛化能力。  
◆ 通过文本引导实现可控的高斯化过程，为点云处理提供了新的交互方式和应用可能性。</td></tr>
<tr><td>2025-08-07</td><td>3DGabSplat: 3D Gabor Splatting for Frequency-adaptive Radiance Field Rendering</td><td>[2508.05343](http://arxiv.org/pdf/2508.05343)</td><td>◆ 提出3D Gabor Splatting（3DGabSplat），采用基于3D Gabor的新型基元，通过多方向3D频率响应增强辐射场表示能力，突破传统3D高斯函数仅能低通滤波的局限。  
◆ 设计包含多频率3D Gabor核的滤波器组，显著提升对场景高频细节的捕捉灵活性与效率，减少冗余基元数量。  
◆ 开发高效CUDA光栅化器，将3D Gabor基元的多方向频率分量投影至2D图像平面，实现实时新颖视角渲染。  
◆ 引入频率自适应机制，动态联合优化基元参数，进一步提升训练与渲染效率，同时降低内存开销。  
◆ 提出模块化设计，3DGabSplat可作为即插即用核无缝集成现有3DGS框架，兼容性优异。  
实验表明，该方法在真实与合成场景中均达到SOTA渲染质量，PSNR最高提升1.35dB，且基元数量和内存占用显著降低。</td></tr>
<tr><td>2025-08-08</td><td>CF3: Compact and Fast 3D Feature Fields</td><td>[2508.05254](http://arxiv.org/pdf/2508.05254)</td><td>◆ 提出了一种自上而下的流程CF3，用于构建紧凑快速的3D高斯特征场，避免了传统自下而上优化方法的高计算成本。  
◆ 采用快速加权融合方法，将多视角2D特征与预训练高斯模型结合，直接在高斯域训练自动编码器，而非传统的2D域训练，使特征分布更匹配。  
◆ 引入自适应稀疏化方法，在优化高斯特征场属性的同时剪枝和合并冗余高斯，显著减少高斯数量（仅需Feature-3DGS的5%）。  
◆ 通过高斯域自动编码器训练，提升了特征表示的紧凑性和效率，同时保留了几何细节。  
◆ 整体方案在计算效率和特征质量上达到竞争性表现，适用于实时或资源受限的应用场景。</td></tr>
<tr><td>2025-08-07</td><td>Refining Gaussian Splatting: A Volumetric Densification Approach</td><td>[2508.05187](http://arxiv.org/pdf/2508.05187)</td><td>◆ 提出基于惯性体积的新型密度控制方法，利用高斯函数的惯性体积指导3D高斯分布的细化过程，改进了原始3DGS的密度控制策略。  
◆ 系统研究了传统运动恢复结构(SfM)与深度图像匹配(DIM)两种点云初始化方法对重建质量的影响，为初始化选择提供依据。  
◆ 在Mip-NeRF 360数据集上的实验表明，该方法在重建质量上优于原始3DGS，并在多样化场景中展现出稳定性能。  
◆ 针对原始3DGS自适应密度控制(ADC)的不足，通过体积感知的细化机制有效提升了新视角合成的质量。  
◆ 通过引入几何感知的密度调控，解决了原始方法在复杂场景中分布不均匀导致的渲染缺陷问题。</td></tr>
<tr><td>2025-08-07</td><td>UGOD: Uncertainty-Guided Differentiable Opacity and Soft Dropout for Enhanced Sparse-View 3DGS</td><td>[2508.04968](http://arxiv.org/pdf/2508.04968)</td><td>◆ 提出不确定性引导的自适应高斯权重机制，通过学习的隐式不确定性优化3D高斯溅射的渲染质量。  
◆ 设计可微分的不透明度更新方法，在保持3DGS流程完整性的同时实现高斯权重的动态调整。  
◆ 创新性地提出软可微分丢弃正则化技术，将不确定性转化为连续丢弃概率以控制渲染过程。  
◆ 在稀疏视角场景中显著降低过拟合风险，相比DropGaussian等方法在MipNeRF 360数据集上PSNR提升3.27%。  
◆ 实现更少高斯数量下的高质量重建，在多个主流数据集上超越现有稀疏视角3D合成方案。  
◆ 首次系统研究高斯权重分配对渲染质量的影响机制，为3DGS优化提供新理论视角。</td></tr>
<tr><td>2025-08-07</td><td>Laplacian Analysis Meets Dynamics Modelling: Gaussian Splatting for 4D Reconstruction</td><td>[2508.04966](http://arxiv.org/pdf/2508.04966)</td><td>这篇论文的核心贡献是针对动态场景重建中3D高斯泼溅（3DGS）技术的局限性提出创新解决方案。现有动态3DGS方法存在低频分解导致过度平滑或高维网格采样引发特征冲突的问题，本质是运动细节保留与形变一致性之间的频谱矛盾。

◆ 提出混合显隐式函数框架，首次将拉普拉斯分析与动态建模结合，解决频谱冲突问题  
◆ 设计频谱感知的拉普拉斯编码架构，融合哈希编码与拉普拉斯模块，实现多频运动灵活控制  
◆ 创新增强型高斯动态属性，通过补偿几何形变引起的光度畸变提升重建精度  
◆ 开发基于KDTree的自适应高斯分裂策略，优化动态区域查询与计算效率  

实验表明该方法在复杂动态场景重建中达到最先进水平，显著提升重建保真度。</td></tr>
<tr><td>2025-08-07</td><td>Perceive-Sample-Compress: Towards Real-Time 3D Gaussian Splatting</td><td>[2508.04965](http://arxiv.org/pdf/2508.04965)</td><td>◆提出感知-采样-压缩框架，解决传统3D高斯泼溅在大规模场景管理和存储效率上的瓶颈。  
◆设计场景感知补偿算法，动态优化高斯参数，优先保障关键区域的视觉保真度并提升资源利用率。  
◆引入金字塔采样表示法，通过分层级管理高斯基元，实现复杂场景的高效组织与渲染。  
◆开发广义高斯混合模型压缩算法，显著降低存储需求（高压缩比）同时保持视觉质量无损。  
◆实验验证该方法在实时渲染速度下，大幅提升内存效率与视觉质量，适用于资源受限环境。</td></tr>
<tr><td>2025-08-07</td><td>Radiance Fields in XR: A Survey on How Radiance Fields are Envisioned and Addressed for XR Research</td><td>[2508.04326](http://arxiv.org/pdf/2508.04326)</td><td>◆ 系统梳理了365篇辐射场（RF）相关研究，聚焦其在XR领域的应用潜力与现状，填补了RF与XR交叉研究的综述空白。  
◆ 首次提出三维分析框架：从XR应用愿景（i）、现有技术实现（ii）和研究缺口（iii）三个维度解析RF对XR的贡献。  
◆ 深入分析66篇核心论文，揭示RF技术（如3DGS、NeRF）如何具体推动XR的光照真实感视图合成与交互体验革新。  
◆ 将XR特有的RF研究问题置于更广阔的RF学术版图中，明确其跨学科定位（涉及计算机视觉、图形学、机器人等6大领域）。  
◆ 为XR研究者提供结构化资源导航，帮助快速追踪RF技术动态，并指出未来研究方向以弥合理论与应用间的差距。</td></tr>
<tr><td>2025-08-06</td><td>DET-GS: Depth- and Edge-Aware Regularization for High-Fidelity 3D Gaussian Splatting</td><td>[2508.04099](http://arxiv.org/pdf/2508.04099)</td><td>◆ 提出DET-GS框架，首次在3D高斯泼溅（3DGS）中统一集成深度与边缘感知正则化，解决稀疏视角下几何重建不精确的问题。  
◆ 设计分层几何深度监督机制，自适应强化多层级几何一致性，显著提升结构保真度并降低深度估计噪声的敏感性。  
◆ 创新性地引入基于Canny边缘检测的语义掩码指导边缘感知深度正则化，有效保护场景边界不被过度平滑。  
◆ 提出RGB引导的边缘保持总变分损失（TV Loss），选择性平滑同质区域的同时严格保留高频细节和纹理。  
◆ 在稀疏视角新视图合成基准测试中，DET-GS在几何精度与视觉保真度上均超越现有最优方法，验证了框架的有效性。</td></tr>
<tr><td>2025-08-06</td><td>RLGS: Reinforcement Learning-Based Adaptive Hyperparameter Tuning for Gaussian Splatting</td><td>[2508.04078](http://arxiv.org/pdf/2508.04078)</td><td>◆ 提出RLGS框架，首次将强化学习引入3D高斯泼溅（3DGS）的超参数自适应优化，替代传统人工调参流程。  
◆ 设计轻量级策略模块，动态调整学习率和致密化阈值等关键参数，无需修改现有3DGS架构即可即插即用。  
◆ 实现模型无关性，验证了在Taming-3DGS、3DGS-MCMC等多种先进变体上的泛化能力。  
◆ 在固定高斯数量约束下显著提升渲染质量，例如在TNT数据集上使Taming-3DGS的PSNR提升0.7dB。  
◆ 突破基线性能瓶颈，在传统方法饱和时仍能持续获得质量增益，展现强鲁棒性。  
◆ 为3DGS训练提供首个通用自动化调参方案，填补了强化学习在该领域的应用空白。</td></tr>
<tr><td>2025-08-05</td><td>Duplex-GS: Proxy-Guided Weighted Blending for Real-Time Order-Independent Gaussian Splatting</td><td>[2508.03180](http://arxiv.org/pdf/2508.03180)</td><td>◆ 提出Duplex-GS双层次框架，结合代理高斯表示与顺序无关渲染技术，在保持实时性能的同时实现照片级真实感渲染。  
◆ 引入单元代理机制管理局部高斯分布，并通过单元搜索光栅化加速渲染，显著降低视图自适应基数排序的开销。  
◆ 创新性地将框架与顺序无关透明度（OIT）技术结合，开发物理启发的加权求和渲染方法，同步消除“闪烁”和“透明度”伪影。  
◆ 提出基于单元的局部高斯管理策略，使基数排序开销降低52.2%至86.9%，且不损失渲染质量。  
◆ 在多样化真实场景数据集上验证了方法的鲁棒性，包括多尺度训练视图和大规模环境，渲染速度比现有OIT方法提升1.5至4倍。  
◆ 首次系统论证了顺序无关渲染范式在高斯泼溅技术中的优势，为实时高质量3D渲染提供了新思路。</td></tr>
<tr><td>2025-08-05</td><td>RobustGS: Unified Boosting of Feedforward 3D Gaussian Splatting under Low-Quality Conditions</td><td>[2508.03077](http://arxiv.org/pdf/2508.03077)</td><td>◆ 提出RobustGS模块，增强前馈式3D高斯泼溅（3DGS）在低质量输入条件下的鲁棒性，解决现有方法依赖高质量多视图图像的局限性。  
◆ 设计通用退化学习器（Generalized Degradation Learner），从多视图输入中提取多种退化的通用表示和分布，提升模型对退化类型的感知能力。  
◆ 引入语义感知状态空间模型，利用退化表示在特征空间增强受损输入，并通过语义感知策略聚合多视图间的相似信息。  
◆ 采用即插即用设计，无需重新训练即可无缝集成到现有预训练流程中，显著提升重建质量。  
◆ 在多种退化条件下（如噪声、低光、雨雾）的实验表明，该方法始终达到最先进的3D重建效果。</td></tr>
<tr><td>2025-08-05</td><td>SA-3DGS: A Self-Adaptive Compression Method for 3D Gaussian Splatting</td><td>[2508.03017](http://arxiv.org/pdf/2508.03017)</td><td>◆ SA-3DGS提出了一种自适应的3D高斯点重要性评分机制，能够自动识别场景中最不重要的高斯点，从而实现高效剪枝和冗余降低。  
◆ 该方法设计了重要性感知的聚类模块，将高斯属性更精准地压缩到码本中，在减小模型大小的同时提升了码本的表征能力。  
◆ 创新性地引入码本修复模块，利用场景上下文信息修复码本，有效缓解信息丢失导致的渲染质量下降问题。  
◆ 实验表明该方法在多个基准数据集上实现了高达66倍的压缩率，同时保持甚至提升了渲染质量。  
◆ 所提出的高斯剪枝策略不仅自适应性强，还能改进其他基于剪枝的方法（如LightGaussian），展现出优异的性能和泛化能力。  
◆ 整体方案解决了现有方法难以准确识别无关高斯点导致压缩质量下降的痛点，为3D高斯泼溅技术的实际部署提供了高效解决方案。</td></tr>
<tr><td>2025-08-05</td><td>Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Splatting</td><td>[2508.02493](http://arxiv.org/pdf/2508.02493)</td><td>◆ 首次从频域角度分析3D高斯泼溅（3DGS）中漂浮伪影的成因，发现未充分优化的高斯分布是主要来源。  
◆ 提出EFA-GS方法，通过选择性扩展未优化高斯分布，优先学习准确的低频信息，从根源抑制伪影。  
◆ 引入深度和尺度双策略动态优化高斯扩展过程，避免高频细节被过度平滑。  
◆ 在合成与真实数据集上验证有效性，PSNR指标提升1.68dB（RWLQ数据集），同时保持高频细节。  
◆ 首次证明该方法可提升下游3D编辑任务质量，具有应用扩展价值。  
◆ 开源实现将促进相关研究，为解决3DGS伪影问题提供新思路。</td></tr>
<tr><td>2025-08-06</td><td>GR-Gaussian: Graph-Based Radiative Gaussian Splatting for Sparse-View CT Reconstruction</td><td>[2508.02408](http://arxiv.org/pdf/2508.02408)</td><td>GR-Gaussian论文针对稀疏视角CT重建中3D高斯泼溅技术存在的针状伪影问题，提出了一种基于图结构的创新解决方案。其核心贡献和创新点如下：

◆ 提出首个结合图结构的3D高斯泼溅框架GR-Gaussian，有效抑制稀疏视角下的针状伪影，提升重建精度。

◆ 创新性设计去噪点云初始化策略，通过降低初始误差加速模型收敛过程。

◆ 开发像素图感知梯度策略，利用基于图的密度差异优化梯度计算，显著提升分割精度和密度表示能力。

实验部分在X-3D和真实数据集上验证了方案优越性，PSNR指标提升0.67-0.92 dB，SSIM提升0.011-0.021，证实了该方法在极端稀疏视角条件下的实用价值。该研究为医学影像重建提供了新的技术思路。</td></tr>
<tr><td>2025-08-04</td><td>GaussianCross: Cross-modal Self-supervised 3D Representation Learning via Gaussian Splatting</td><td>[2508.02172](http://arxiv.org/pdf/2508.02172)</td><td>◆ 提出GaussianCross，一种新型跨模态自监督3D表示学习框架，结合前馈式3D高斯泼溅（3DGS）技术，解决现有方法因点云判别难度不足导致的模型坍塌和结构信息缺失问题。  
◆ 首创将尺度不一致的3D点云无缝转换为归一化的立方体高斯表示，保留细节的同时实现稳定且可泛化的预训练。  
◆ 设计三属性自适应蒸馏泼溅模块，构建3D特征场，协同捕捉外观、几何和语义特征以保持跨模态一致性。  
◆ 在ScanNet等基准测试中展现显著参数与数据效率：仅需0.1%参数（线性探测）和1%场景数据即可超越现有最优方法。  
◆ 全微调任务中性能提升显著，ScanNet200语义分割和实例分割任务分别提升9.3% mIoU和6.1% AP$_{50}$，验证框架强泛化能力。</td></tr>
<tr><td>2025-08-03</td><td>AG$^2$aussian: Anchor-Graph Structured Gaussian Splatting for Instance-Level 3D Scene Understanding and Editing</td><td>[2508.01740](http://arxiv.org/pdf/2508.01740)</td><td>◆提出AG²aussian框架，首次将锚点图结构引入3D高斯泼溅（3DGS）领域，通过图结构组织语义特征并约束高斯基元分布。  
◆创新性地构建实例感知的锚点图结构，实现紧凑且语义明确的高斯分布，显著改善现有方法中高斯选择杂乱的问题。  
◆开发基于图传播的语义特征传递机制，无需依赖可微分渲染特征蒸馏，获得更清晰准确的实例级分割效果。  
◆支持多模态交互（点击/文本查询）的实例级场景理解与编辑，首次在3DGS中实现开放词汇查询功能。  
◆验证框架在四大应用场景（交互查询、开放词汇检索、物体移除编辑、物理模拟）的通用性，实验证明其显著优于现有方法。  
◆通过消融研究证实锚点图结构对提升语义一致性和编辑精度的关键作用，为3DGS的语义理解提供新范式。</td></tr>
<tr><td>2025-08-02</td><td>Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians</td><td>[2508.01464](http://arxiv.org/pdf/2508.01464)</td><td>◆ 提出了首个3D场景级变分自编码器Can3Tok，能够将大量3D高斯基元编码为低维潜在嵌入，有效捕捉输入的语义和空间信息。  
◆ 针对3D高斯泼溅(3DGS)表示的场景存在尺度不一致问题，设计了一套通用的3D场景数据处理流程。  
◆ 解决了现有方法在场景级3D生成中因无界空间和尺度差异导致的潜在表示学习难题。  
◆ 在DL3DV-10K数据集上验证了方法的有效性，仅Can3Tok能泛化到新场景，而对比方法无法收敛且零泛化能力。  
◆ 展示了图像到3DGS和文本到3DGS的生成应用，证明其支持下游生成任务的能力。  
◆ 突破了当前3D生成主要局限于物体级的现状，推动了场景级3D生成的发展。</td></tr>
<tr><td>2025-08-02</td><td>OCSplats: Observation Completeness Quantification and Label Noise Separation in 3DGS</td><td>[2508.01239](http://arxiv.org/pdf/2508.01239)</td><td>◆ 提出OCSplats框架，首次从认知不确定性角度解决3D高斯泼溅（3DGS）中的标签噪声问题，突破传统抗噪重建方法的局限。  
◆ 创新性结合混合噪声评估和基于观测的认知校正技术，显著提升认知差异区域的噪声分类精度。  
◆ 设计基于动态锚点的标签噪声分类流程，无需调整参数即可适应不同噪声比例的多样化场景，实现通用化应用。  
◆ 系统解决了移动物体、非朗伯表面和阴影等现实场景噪声导致的3D重建误差问题。  
◆ 通过大量实验验证，OCSplats在不同复杂度场景中均保持领先的重建性能和精确的噪声分类能力。  
◆ 摆脱现有方法依赖场景特定超参数调优的缺陷，大幅提升3DGS技术在真实场景的实用性和鲁棒性。</td></tr>
<tr><td>2025-08-01</td><td>IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation</td><td>[2508.00823](http://arxiv.org/pdf/2508.00823)</td><td>◆ 提出IGL-Nav框架，首次将可渲染的3D高斯表示（3DGS）应用于图像目标导航任务，实现3D几何感知的精准定位。  
◆ 采用增量式场景更新策略，通过单目预测实时优化3DGS表示，显著降低计算开销，解决传统3DGS优化效率低的问题。  
◆ 设计两阶段定位方法：先利用几何信息进行粗粒度离散空间匹配（等效于高效3D卷积），再通过可微分渲染优化细粒度目标位姿。  
◆ 突破传统方法依赖拓扑图或BEV地图的局限，直接建模探索环境与目标图像的3D几何关系，支持自由视角目标图像定位。  
◆ 在多种实验配置下大幅超越现有SOTA方法，并验证了在真实机器人平台的应用潜力（仅需手机任意姿态拍摄目标图像）。</td></tr>
<tr><td>2025-08-01</td><td>Omni-Scan: Creating Visually-Accurate Digital Twin Object Models Using a Bimanual Robot with Handover and Gaussian Splat Merging</td><td>[2508.00354](http://arxiv.org/pdf/2508.00354)</td><td>◆ 提出Omni-Scan系统，利用双机械臂协作抓取和旋转物体，配合固定摄像头实现全视角扫描，突破传统扫描设备的工作空间限制。  
◆ 结合DepthAnything、Segment Anything和RAFT光流模型，自动分割并去除机械夹爪和背景干扰，精准提取目标物体图像数据。  
◆ 改进3D高斯泼溅（3DGS）训练流程，支持处理含夹爪遮挡的拼接数据集，生成360度无死角的高质量数字孪生模型。  
◆ 首次将双机械臂物体交接技术应用于3D扫描领域，通过二次抓取解决单次扫描的遮挡问题，显著提升模型完整性。  
◆ 在工业零件缺陷检测任务中验证有效性，对12种不同物体实现平均83%的缺陷识别准确率，展示实际应用潜力。  
◆ 提供开源交互式3DGS模型展示，推动数字孪生技术在仿真、VR等领域的应用发展。</td></tr>
<tr><td>2025-07-31</td><td>SeqAffordSplat: Scene-level Sequential Affordance Reasoning on 3D Gaussian Splatting</td><td>[2507.23772](http://arxiv.org/pdf/2507.23772)</td><td>◆ 提出首个面向3D高斯泼溅(3DGS)场景的序列化功能推理任务，突破现有单物体单步骤交互的局限，支持长时序多物体复杂任务。  
◆ 构建SeqAffordSplat大规模基准数据集，包含1800+复杂场景，为3DGS环境下的长时序功能理解研究提供支撑。  
◆ 设计端到端框架SeqSplatNet，首次实现从自然语言指令到3D功能掩码序列的直接映射，通过LLM自回归生成含分割标记的文本指导解码。  
◆ 创新提出条件几何重建预训练策略，通过从已知几何观测重建完整功能区域掩码，建立强几何先验以应对复杂场景结构。  
◆ 开发多尺度特征注入机制，从2D视觉基础模型提取语义特征并融合至3D解码器，有效解决语义模糊性问题。  
实验表明该方法在复杂场景级序列任务上达到最优性能，将功能推理能力从单步骤提升至长时序交互水平。</td></tr>
<tr><td>2025-07-31</td><td>Stereo 3D Gaussian Splatting SLAM for Outdoor Urban Scenes</td><td>[2507.23677](http://arxiv.org/pdf/2507.23677)</td><td>◆ 提出了首个面向户外场景的双目3D高斯泼溅SLAM系统（BGS-SLAM），填补了现有3DGS-SLAM技术主要局限于室内环境的空白。  
◆ 仅依赖RGB双目图像，无需LiDAR或主动深度传感器，降低了硬件成本和使用门槛。  
◆ 利用预训练深度立体网络的深度估计指导3D高斯优化，结合多损失策略提升几何一致性和视觉质量。  
◆ 针对复杂户外环境设计了专门的解决方案，在多个数据集上验证了其优越性。  
◆ 相比其他基于3DGS的SLAM方法，BGS-SLAM在跟踪精度和建图性能方面表现更优。  
◆ 为大规模户外场景的高效、高保真三维重建提供了新的技术路径。</td></tr>
<tr><td>2025-07-31</td><td>Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization</td><td>[2507.23569](http://arxiv.org/pdf/2507.23569)</td><td>◆ 提出Gaussian Splatting Feature Fields (GSFFs)，将显式几何模型（3DGS）与隐式特征场结合，构建新型视觉定位场景表示。  
◆ 利用3DGS的密集几何信息和可微分光栅化算法，学习基于3D空间的鲁棒特征表示，提升定位精度。  
◆ 通过对比学习框架，将3D尺度感知特征场与2D特征编码器对齐到统一嵌入空间，增强特征一致性。  
◆ 引入3D结构感知聚类方法，正则化表征学习并自动生成分割结果，支持隐私保护定位（如使用分割替代原始图像）。  
◆ 提出基于特征图或分割对齐的姿态优化方法，实现隐私保护与非隐私保护双模式定位，在多个真实数据集上达到SOTA性能。</td></tr>
<tr><td>2025-07-31</td><td>NeRF Is a Valuable Assistant for 3D Gaussian Splatting</td><td>[2507.23374](http://arxiv.org/pdf/2507.23374)</td><td>◆ 提出NeRF-GS框架，首次联合优化NeRF（神经辐射场）和3D高斯泼溅（3DGS），实现两种技术的优势互补。  
◆ 利用NeRF的连续空间表征能力，有效解决3DGS对高斯初始化敏感、空间感知弱和高斯间关联性不足的固有缺陷。  
◆ 通过渐进式对齐3DGS与NeRF的空间特征，使两者能基于共享的3D空间信息协同优化，提升场景表示精度。  
◆ 创新性地优化隐式特征和高斯位置的残差向量，弥合两种方法的理论差异，增强3DGS的个性化建模能力。  
◆ 在基准数据集上实现SOTA性能，验证了NeRF与3DGS的互补性，为混合式3D场景建模提供了新思路。</td></tr>
<tr><td>2025-07-31</td><td>MagicRoad: Semantic-Aware 3D Road Surface Reconstruction via Obstacle Inpainting</td><td>[2507.23340](http://arxiv.org/pdf/2507.23340)</td><td>这篇论文的核心贡献是通过语义感知的障碍物修复技术实现鲁棒的3D路面重建，主要创新点如下：

◆ 提出融合遮挡感知2D高斯面元与语义引导色彩增强的框架，解决动态遮挡和静态障碍物干扰问题  
◆ 采用平面自适应高斯表示方法，实现高效的大规模路面建模  
◆ 创新性地结合分割引导的视频修复技术，可同时消除动态和静态前景物体  
◆ 在HSV色彩空间进行语义感知校正，显著提升光照和天气变化下的颜色一致性  
◆ 实验证明该方法在复杂城市场景中重建效果优于现有技术，几何精度和视觉连贯性更优  

该方法为自动驾驶提供了更可靠的高精度路面重建方案，有效应对现实环境中的各种挑战。</td></tr>
<tr><td>2025-07-31</td><td>GSFusion:Globally Optimized LiDAR-Inertial-Visual Mapping for Gaussian Splatting</td><td>[2507.23273](http://arxiv.org/pdf/2507.23273)</td><td>◆ 提出GSFusion系统，首次将激光雷达（LiDAR）、惯性测量单元（IMU）和视觉传感器融合到3D高斯泼溅（3DGS）建图中，解决了传统纯视觉方法在弱纹理、光照差或远距离场景中的局限性。  
◆ 引入全局优化的面元到面元（surfel-to-surfel）约束，通过位姿图优化实现高精度地图一致性，显著提升建图质量。  
◆ 设计像素感知的高斯初始化策略，有效处理激光雷达稀疏数据，提升高斯表示的效率和准确性。  
◆ 提出有界Sigmoid约束机制，防止高斯分布无控制增长，优化计算资源利用率并缩短优化时间。  
◆ 在公开和自建数据集上的实验表明，该系统在渲染质量和建图效率上均优于现有3DGS SLAM方案，尤其在复杂环境中表现突出。</td></tr>
<tr><td>2025-07-30</td><td>UFV-Splatter: Pose-Free Feed-Forward 3D Gaussian Splatting Adapted to Unfavorable Views</td><td>[2507.22342](http://arxiv.org/pdf/2507.22342)</td><td>◆ 提出首个无需相机位姿的前馈式3D高斯泼溅框架，专门针对不利视角输入进行优化，突破传统方法仅适用于固定视角的限制。  
◆ 创新性地采用低秩自适应（LoRA）层增强预训练模型，通过图像重定心技术将不利视角转换为模型可处理的伪有利视角。  
◆ 设计高斯适配器模块，显著提升重定心输入生成的高斯分布几何一致性，解决视角变换导致的几何失真问题。  
◆ 开发高斯对齐方法，精确生成训练所需的目标视图渲染，为模型提供高质量监督信号。  
◆ 提出仅需有利视角图像的全新训练策略，利用现成数据集实现模型优化，无需额外采集不利视角数据。  
◆ 在Google Scanned Objects合成数据和OmniObject3D真实数据上验证有效性，显著提升不利视角下的渲染质量。</td></tr>
<tr><td>2025-07-29</td><td>From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning</td><td>[2507.22028](http://arxiv.org/pdf/2507.22028)</td><td>这篇论文的核心贡献是提出了Seeing-to-Experiencing (S2E)框架，通过强化学习提升导航基础模型的交互能力，同时保持从大规模视频数据中学习到的泛化性。主要创新点包括：

◆ 提出S2E框架，首次将视频预训练与仿真环境强化学习后训练相结合，解决了纯离线训练模型缺乏交互推理能力的问题。

◆ 设计Anchor-Guided Distribution Matching策略，通过锚点引导的分布匹配稳定学习过程，并建模多样化的运动模式。

◆ 开发Residual-Attention模块，在不破坏预训练知识的前提下，从仿真环境中学习反应式行为。

◆ 构建NavBench-GS评估基准，基于真实场景的光照级3D高斯重建，系统评估导航模型的泛化性和安全性。

实验表明S2E有效缓解了纯离线数据扩展的收益递减问题，并通过对比分析验证了强化学习相比监督微调在机器人学习中的优势。该研究强调了在线交互体验对扩展机器人基础模型的关键作用。</td></tr>
<tr><td>2025-07-29</td><td>DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments</td><td>[2507.21981](http://arxiv.org/pdf/2507.21981)</td><td>◆ 提出了首个基于3D高斯泼溅（3DGS）的统一、模块化开源仿真框架DISCOVERSE，专为Real2Sim2Real机器人学习设计。  
◆ 开发了全流程Real2Sim管线，能高保真合成复杂真实场景的几何与外观，为分析并弥合Sim2Real差距提供新途径。  
◆ 结合高斯泼溅与MuJoCo物理引擎，支持多传感器模态的并行仿真与高精度物理模拟，显著提升效率与真实性。  
◆ 兼容现有3D资产、机器人模型及ROS插件，便于快速集成和大规模机器人学习任务，推动复杂机器人基准测试发展。  
◆ 在模仿学习实验中实现零样本Sim2Real迁移的SOTA性能，验证了框架在真实场景中的优越泛化能力。</td></tr>
<tr><td>2025-07-31</td><td>MultiEditor: Controllable Multimodal Object Editing for Driving Scenarios Using 3D Gaussian Splatting Priors</td><td>[2507.21872](http://arxiv.org/pdf/2507.21872)</td><td>◆ 提出MultiEditor，首个双分支潜在扩散框架，可联合编辑自动驾驶场景中的图像和LiDAR点云数据。  
◆ 创新引入3D高斯泼溅（3DGS）作为目标物体的结构和外观先验，实现跨模态的高保真重建。  
◆ 设计多级外观控制机制（像素级粘贴、语义级引导、多分支细化），精确控制编辑过程的外观一致性。  
◆ 提出深度引导的可变形跨模态条件模块，利用3DGS渲染深度自适应协调模态间互导，显著提升跨模态一致性。  
◆ 实验证明该方法在视觉/几何保真度、编辑可控性和跨模态一致性上优于现有技术。  
◆ 生成稀有类别车辆数据可大幅提升感知模型对长尾类别的检测精度（如安全关键但少见的车辆类型）。</td></tr>
<tr><td>2025-07-30</td><td>No Redundancy, No Stall: Lightweight Streaming 3D Gaussian Splatting for Real-time Rendering</td><td>[2507.21572](http://arxiv.org/pdf/2507.21572)</td><td>◆ 提出LS-Gaussian框架，通过算法与硬件协同设计实现轻量级实时3D高斯泼溅渲染，解决高帧率与边缘计算资源受限的挑战。  
◆ 核心创新之一是视角变换算法，利用连续帧间的场景连续性进行稀疏渲染，避免逐帧独立计算带来的冗余开销。  
◆ 针对并行渲染中图像分块负载不均导致的硬件停滞问题，提出基于视角变换的负载预测机制，实现更均衡的并行计算。  
◆ 设计定制化3DGS硬件加速器，支持负载感知的实时任务映射，显著提升计算效率。  
◆ 实验验证框架高效性：在边缘GPU上平均加速5.41倍，定制加速器下最高达17.3倍，且视觉质量损失极小。  
◆ 为自动驾驶、具身智能等实时3D渲染场景提供低延迟、高能效的解决方案。</td></tr>
<tr><td>2025-07-28</td><td>$S^3$LAM: Surfel Splatting SLAM for Geometrically Accurate Tracking and Mapping</td><td>[2507.20854](http://arxiv.org/pdf/2507.20854)</td><td>◆ 提出S³LAM系统，采用2D面元（surfel）作为基本表示单元，相比传统3D高斯椭球体更高效地表征场景几何。  
◆ 创新性地使用2D高斯面元进行场景重建，专注于物体表面几何，显著提升跟踪和建图的精度。  
◆ 设计自适应表面渲染策略，有效解决有限视角下的实时优化问题，在保证计算效率的同时提高建图准确性。  
◆ 首次从2D面元渲染公式中直接推导相机位姿雅可比矩阵，证明几何精确表示对提升跟踪收敛性的关键作用。  
◆ 在合成和真实数据集上的实验验证了S³LAM的优越性，性能达到当前最优水平。</td></tr>
<tr><td>2025-07-29</td><td>From Gallery to Wrist: Realistic 3D Bracelet Insertion in Videos</td><td>[2507.20331](http://arxiv.org/pdf/2507.20331)</td><td>这篇论文的核心贡献和创新点如下：

◆ 提出了一种混合3D渲染与2D扩散模型的视频对象插入方法，结合了3D高斯泼溅（3DGS）的时间一致性和2D扩散模型的逼真光照优势。  
◆ 首次将3D高斯泼溅技术应用于动态视频中的3D手镯插入，通过多帧加权优化确保时间连贯性。  
◆ 设计了基于光照分解的管线，分离物体的固有属性（反照率、阴影、反射率），并分别优化阴影和sRGB图像以实现更逼真的光照效果。  
◆ 开发了2D扩散增强模型，专门用于提升3DGS初始渲染结果的光照真实感，解决了传统3D渲染在光照逼真度上的不足。  
◆ 针对动态手腕场景的特殊挑战（复杂运动、视角变化、光照变化），提供了完整的解决方案，在增强现实和虚拟试戴领域具有应用价值。  
◆ 该方法首次实现了3D渲染与2D扩散模型的协同工作，为视频对象插入提供了既保持时间一致性又具备照片级真实感的新范式。</td></tr>
<tr><td>2025-07-26</td><td>SonicGauss: Position-Aware Physical Sound Synthesis for 3D Gaussian Representations</td><td>[2507.19835](http://arxiv.org/pdf/2507.19835)</td><td>◆ 首次将3D高斯表示（3DGS）扩展到物理声音合成领域，填补了该技术在声音建模方面的空白。  
◆ 提出SonicGauss框架，通过结合扩散模型和PointTransformer特征提取器，直接从高斯椭球体推断材料特性和空间-声学关联。  
◆ 实现基于碰撞位置的动态声音合成，支持空间变化的声学反馈，增强了交互真实感。  
◆ 方法具有强泛化能力，可适用于多种物体类别，无需针对特定对象进行单独训练。  
◆ 在ObjectFolder数据集和真实录音上的实验证明，该方法能生成高保真且位置感知的听觉效果。  
◆ 为3D视觉表示与交互式声音合成的跨模态关联提供了创新解决方案。</td></tr>
<tr><td>2025-07-26</td><td>Taking Language Embedded 3D Gaussian Splatting into the Wild</td><td>[2507.19830](http://arxiv.org/pdf/2507.19830)</td><td>◆ 提出首个将语言嵌入3D高斯泼溅（3DGS）技术应用于开放场景理解的框架，支持从无约束照片集中进行开放词汇分割。  
◆ 创新性地引入多视角外观CLIP特征提取方法，结合瞬态不确定性和外观不确定性两种语言特征不确定性图，优化3D场景理解。  
◆ 设计瞬态不确定性感知自编码器，有效压缩和融合多视角语言特征，提升特征学习效率。  
◆ 提出多外观语言场3DGS表示和后集成策略，增强语言特征在3D场景中的鲁棒性和一致性。  
◆ 构建PT-OVS基准数据集，首次为无约束照片集的开放词汇分割任务提供量化评估标准。  
◆ 实验证明该方法在开放词汇查询、建筑风格识别和3D场景编辑等应用中显著优于现有方法。</td></tr>
<tr><td>2025-07-25</td><td>DINO-SLAM: DINO-informed RGB-D SLAM for Neural Implicit and Explicit Representations</td><td>[2507.19474](http://arxiv.org/pdf/2507.19474)</td><td>◆ 提出DINO-SLAM，一种基于DINO特征的设计策略，用于增强SLAM系统中神经隐式（NeRF）和显式（3DGS）表示的场景建模能力。  
◆ 引入场景结构编码器（SSE），将DINO特征升级为增强版EDINO，以捕捉场景的层次化元素和结构关系。  
◆ 开发了两种基于EDINO特征的基础范式，分别针对NeRF和3DGS的SLAM系统，实现更全面的场景表示。  
◆ 在Replica、ScanNet和TUM数据集上验证了DINO-SLAM的优越性能，超越现有最先进方法。  
◆ 通过融合DINO特征，显著提升了SLAM系统在复杂场景中的鲁棒性和准确性。</td></tr>
<tr><td>2025-07-25</td><td>Fast Learning of Non-Cooperative Spacecraft 3D Models through Primitive Initialization</td><td>[2507.19459](http://arxiv.org/pdf/2507.19459)</td><td>◆ 提出基于CNN的3D高斯泼溅(3DGS)初始化器，仅需单目图像即可生成粗糙3D模型和目标姿态，解决了传统方法依赖多视角精确姿态的问题。  
◆ 开发支持噪声或隐式姿态估计的训练流程，突破现有方法对精确姿态标注的强依赖性，增强太空场景适用性。  
◆ 通过几何基元初始化策略，将3DGS训练迭代次数和所需图像数量降低至少一个数量级，大幅提升计算效率。  
◆ 设计多姿态估计变体的CNN架构，系统分析不同变体在噪声姿态下的3D重建效果，为实际应用提供选择依据。  
◆ 实验证明即使存在姿态估计误差，仍能重建高保真3D模型，首次实现新型视图合成技术在太空非合作目标建模中的可行性。</td></tr>
<tr><td>2025-07-25</td><td>3DGauCIM: Accelerating Static/Dynamic 3D Gaussian Splatting via Digital CIM for High Frame Rate Real-Time Edge Rendering</td><td>[2507.19133](http://arxiv.org/pdf/2507.19133)</td><td>◆ 提出3DGauCIM框架，通过算法-硬件协同设计优化静态/动态3D高斯泼溅（3DGS）技术，实现边缘设备上的高帧率实时渲染。  
◆ 算法层面创新：开发DRAM访问减少的视锥剔除技术，降低动态场景中高斯参数加载的能耗开销。  
◆ 算法层面创新：设计自适应分块分组策略，提升片上缓冲区的重用率，减少频繁DRAM访问。  
◆ 算法层面创新：提出自适应间隔初始化桶-双调排序算法，显著降低动态场景的排序延迟和能耗。  
◆ 硬件层面创新：设计数字存内计算（DCIM）友好型计算流程，基于16nm原型芯片实测数据验证能效优势。  
◆ 实验成果：在大型静态/动态数据集上实现超200 FPS的实时渲染，功耗仅0.28W（静态）/0.63W（动态），突破边缘设备能效瓶颈。</td></tr>
<tr><td>2025-07-25</td><td>Gaussian Set Surface Reconstruction through Per-Gaussian Optimization</td><td>[2507.18923](http://arxiv.org/pdf/2507.18923)</td><td>◆ 提出Gaussian Set Surface Reconstruction (GSSR)方法，通过逐高斯优化实现高斯沿隐式表面的均匀分布，解决现有3DGS方法中高斯分布偏离表面问题。  
◆ 结合像素级和高斯级的单视角法线一致性以及多视角光度一致性约束，实现细粒度的几何对齐，优化局部和全局几何精度。  
◆ 引入不透明度正则化损失，有效消除冗余高斯，提升表示效率。  
◆ 采用周期性的深度和法线引导的高斯重新初始化策略，改善高斯空间分布的均匀性和清洁度。  
◆ 实验证明GSSR显著提升了几何重建精度，同时保持高质量渲染性能，支持直观的场景编辑和新3D环境生成。</td></tr>
<tr><td>2025-07-24</td><td>SaLF: Sparse Local Fields for Multi-Sensor Rendering in Real-Time</td><td>[2507.18713](http://arxiv.org/pdf/2507.18713)</td><td>◆ 提出SaLF（稀疏局部场）新型体积表示方法，支持光栅化和光线追踪两种渲染方式，突破了传统方法只能二选一的限制。  
◆ 采用稀疏3D体素基元与局部隐式场结合的混合表示，每个体素包含局部隐式场，实现高效存储与高精度细节的平衡。  
◆ 首次实现单一模型同时支持非针孔相机（如鱼眼镜头）和旋转激光雷达的多传感器实时渲染（相机50+ FPS，LiDAR 600+ FPS）。  
◆ 引入自适应剪枝与致密化机制，可动态优化场景表示，轻松处理大规模自动驾驶场景。  
◆ 训练速度显著提升（&lt;30分钟），比NeRF快两个数量级，同时保持与现有自动驾驶传感器仿真方法相当的逼真度。  
◆ 解耦场景表示与渲染流程，增强系统通用性，为多传感器仿真提供统一解决方案。</td></tr>
<tr><td>2025-07-24</td><td>Unposed 3DGS Reconstruction with Probabilistic Procrustes Mapping</td><td>[2507.18541](http://arxiv.org/pdf/2507.18541)</td><td>◆ 提出首个无需预设相机位姿的3D高斯泼溅(3DGS)重建框架，通过概率性Procrustes映射实现数百张户外图像的联合优化。  
◆ 创新性地将数千万点云对齐问题建模为概率Procrustes问题，采用闭式解算方法实现分钟级全局配准。  
◆ 设计软垃圾箱机制与概率耦合策略，有效过滤不确定对应点，提升跨子图对齐的鲁棒性。  
◆ 开发基于置信度锚点的高斯初始化方法，结合可微渲染与解析雅可比矩阵，实现场景几何与相机位姿的联合优化。  
◆ 在Waymo和KITTI数据集上验证了方法的优越性，为无位姿3DGS重建树立了新标杆。  
该方法突破了传统MVS模型的内存限制与精度瓶颈，为大规模户外场景重建提供了高效解决方案。</td></tr>
<tr><td>2025-07-24</td><td>High-fidelity 3D Gaussian Inpainting: preserving multi-view consistency and photorealistic details</td><td>[2507.18023](http://arxiv.org/pdf/2507.18023)</td><td>◆ 提出首个基于3D高斯分布的3D场景修复框架，通过稀疏修复视图实现完整3D场景重建，解决了传统方法在3D结构不规则性和多视角一致性上的难题。  
◆ 开发自动掩膜优化流程，结合高斯场景过滤和反向投影技术，精准定位遮挡区域并实现自然边界修复，显著提升掩膜精度。  
◆ 创新性引入区域级不确定性引导优化策略，通过多视角重要性评估动态调整训练权重，有效缓解多视角不一致问题。  
◆ 设计细粒度优化方案，在修复过程中强化细节保真度，使修复结果同时保持照片级真实感和几何一致性。  
◆ 在多样化数据集上的实验表明，该方法在视觉质量和视角一致性方面均超越现有最优方法，为3D内容创作提供新工具。</td></tr>
<tr><td>2025-07-23</td><td>Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting</td><td>[2507.17336](http://arxiv.org/pdf/2507.17336)</td><td>◆ 提出首个针对4D高斯泼溅（4DGS）的端到端率失真优化压缩框架，解决动态场景存储和传输效率问题。  
◆ 基于全显式动态高斯泼溅（Ex4DGS）基线，兼容现有3DGS压缩方法，同时有效处理时间轴带来的额外挑战。  
◆ 创新采用小波变换替代独立存储运动轨迹，利用真实世界运动平滑性先验，显著提升存储效率。  
◆ 实现用户可调控的压缩效率与渲染质量平衡，最高达到91倍压缩率（相比原始Ex4DGS模型）且保持高视觉保真度。  
◆ 通过大量实验验证框架普适性，支持从边缘设备到高性能环境的实时动态场景渲染应用。</td></tr>
<tr><td>2025-07-22</td><td>StreamME: Simplify 3D Gaussian Avatar within Live Stream</td><td>[2507.17029](http://arxiv.org/pdf/2507.17029)</td><td>◆ 提出StreamME方法，实现实时视频流中的快速3D头像重建，无需预缓存数据，支持与下游应用无缝集成。  
◆ 基于3D高斯泼溅（3DGS）技术，摒弃可变形3DGS中的MLP依赖，仅依赖几何信息，显著提升面部表情适应速度。  
◆ 引入基于主点的简化策略，稀疏分布点云于面部表面，在保持渲染质量的同时优化点数，确保实时训练高效性。  
◆ 通过实时训练能力保护面部隐私，并降低VR系统或在线会议的通信带宽需求。  
◆ 可直接应用于动画、卡通化、重光照等下游任务，扩展了3D头像的实际应用场景。</td></tr>
<tr><td>2025-07-21</td><td>Appearance Harmonization via Bilateral Grid Prediction with Transformers for 3DGS</td><td>[2507.15748](http://arxiv.org/pdf/2507.15748)</td><td>◆ 提出基于Transformer的双边网格预测方法，通过空间自适应的方式校正多视角图像间的光度不一致性，解决了传统相机处理流程导致的色彩不一致问题。  
◆ 首次将学习到的双边网格整合到3D高斯泼溅（3DGS） pipeline中，在保持高训练效率的同时显著提升了新视角合成的重建质量。  
◆ 实现了跨场景的鲁棒泛化能力，无需针对每个场景重新训练模型，突破了现有方法必须进行场景特定优化的限制。  
◆ 通过多视角一致的光度校正机制，有效维护了3D场景表示的一致性，克服了传统逐图像外观嵌入方法计算复杂、训练速度慢的缺陷。  
◆ 实验证明该方法在重建保真度和收敛速度上优于或匹配现有的场景特定优化方法，为实时高质量新视角合成提供了新思路。</td></tr>
<tr><td>2025-07-21</td><td>DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting</td><td>[2507.15690](http://arxiv.org/pdf/2507.15690)</td><td>◆ 提出DWTGS框架，通过小波空间损失重新思考稀疏视图3D高斯泼溅的频率正则化方法，解决传统傅里叶变换导致的参数调优困难和高频学习偏差问题。  
◆ 创新性地利用小波变换的多级低频LL子带监督，提供额外的空间监督信号，避免对高频细节的过拟合。  
◆ 采用自监督方式对高频HH子带施加稀疏约束，有效减少高频伪影并提升模型泛化能力。  
◆ 实验证明该方法在多个基准测试中 consistently 优于基于傅里叶变换的现有方法。  
◆ 通过低频主导策略显著改善稀疏视图下的新视角合成质量，解决了高频细节过度学习导致的泛化性能下降问题。  
◆ 整体框架无需复杂参数调优，为稀疏视图3D重建提供了更稳定高效的频率域解决方案。</td></tr>
<tr><td>2025-07-21</td><td>Hi^2-GSLoc: Dual-Hierarchical Gaussian-Specific Visual Relocalization for Remote Sensing</td><td>[2507.15683](http://arxiv.org/pdf/2507.15683)</td><td>◆ 提出Hi^2-GSLoc双层次视觉重定位框架，采用稀疏到稠密、粗到精的范式，结合3D高斯泼溅（3DGS）作为新型场景表示方法，同时编码几何与外观信息。  
◆ 设计基于高斯基元的语义感知采样策略和地标引导检测器，在稀疏阶段实现鲁棒的初始位姿估计，解决大尺度遥感场景的域适应问题。  
◆ 在稠密阶段引入粗到精的密集光栅化匹配与可靠性验证机制，通过迭代优化提升位姿精度，克服传统结构法计算复杂度的缺陷。  
◆ 针对遥感数据特性，开发分区高斯训练、GPU并行匹配和动态内存管理技术，显著提升大规模场景的处理效率。  
◆ 通过仿真数据、公开数据集和真实飞行实验验证，证明方法在定位精度、召回率和计算效率上均优于现有方案，尤其擅长过滤不可靠位姿估计。  
◆ 首次将3DGS应用于遥感视觉定位领域，为无人机和遥感应用提供兼具高精度与可扩展性的解决方案。</td></tr>
<tr><td>2025-07-21</td><td>Gaussian Splatting with Discretized SDF for Relightable Assets</td><td>[2507.15629](http://arxiv.org/pdf/2507.15629)</td><td>◆ 提出离散化SDF表示方法，将连续SDF编码为高斯内部的采样值，避免传统SDF的内存和计算开销。  
◆ 通过SDF-to-opacity转换将离散SDF与高斯透明度关联，实现基于泼溅的SDF渲染，无需光线步进计算。  
◆ 设计基于投影的一致性损失函数，将高斯投影到SDF零水平集并强制对齐泼溅表面，解决离散样本难以梯度约束的问题。  
◆ 在保持高斯泼溅原有内存效率的同时，显著提升逆向渲染质量，无需额外存储或复杂手工优化设计。  
◆ 实验证明该方法超越现有基于高斯的逆向渲染方法，实现更高精度的重光照效果。  
核心贡献在于通过离散化SDF与高斯泼溅的深度融合，首次在保持实时渲染优势下解决了高斯几何约束难题。</td></tr>
<tr><td>2025-07-21</td><td>SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting</td><td>[2507.15602](http://arxiv.org/pdf/2507.15602)</td><td>◆ 提出了一种新颖的混合方法，结合了基于SDF的表面重建和基于3D高斯泼溅（3DGS）的新视角渲染的优势。  
◆ 利用SDF捕捉粗粒度几何信息来增强3DGS的渲染效果，同时通过3DGS生成的新视角图像优化SDF的细节，实现更精确的表面重建。  
◆ 解决了SDF方法在细节表现上的不足和3DGS方法在全局几何一致性上的缺陷，实现了两者优势互补。  
◆ 在DTU和MobileBrick数据集上，该方法在表面重建和新视角合成任务上超越了现有最先进方法。  
◆ 代码将开源，便于后续研究和应用。</td></tr>
</tbody>
</table>
</div>

<h2 id='depth-estimation'>Depth Estimation</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-10-23</td><td>PPMStereo: Pick-and-Play Memory Construction for Consistent Dynamic Stereo Matching</td><td>[2510.20178](http://arxiv.org/pdf/2510.20178)</td><td>◆ Temporally consistent depth estimation from stereo video is critical for real-world applications such as augmented reality, where inconsistent depth estimation disrupts the immersion of users.
◆ Despite its importance, this task remains challenging due to the difficulty in modeling long-term temporal consistency in a computationally efficient manner.
◆ Previous methods attempt to address this by aggregating spatio-temporal information but face a fundamental trade-off: limited temporal modeling provides only modest gains, whereas capturing long-range dependencies significantly increases computational cost.</td></tr>
<tr><td>2025-10-22</td><td>How to Evaluate Monocular Depth Estimation?</td><td>[2510.19814](http://arxiv.org/pdf/2510.19814)</td><td>◆ Monocular depth estimation is an important task with rapid progress, but how to evaluate it remains an open question, as evidenced by a lack of standardization in existing literature and a large selection of evaluation metrics whose trade-offs and behaviors are not well understood.
◆ This paper contributes a novel, quantitative analysis of existing metrics in terms of their sensitivity to various types of perturbations of ground truth, emphasizing comparison to human judgment.
◆ Our analysis reveals that existing metrics are severely under-sensitive to curvature perturbation such as making flat surfaces wavy.</td></tr>
<tr><td>2025-10-21</td><td>PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting</td><td>[2510.18714](http://arxiv.org/pdf/2510.18714)</td><td>◆ This paper addresses metric 3D reconstruction of indoor scenes by exploiting their inherent geometric regularities with compact representations.
◆ Using planar 3D primitives - a well-suited representation for man-made environments - we introduce PLANA3R, a pose-free framework for metric Planar 3D Reconstruction from unposed two-view images.
◆ Our approach employs Vision Transformers to extract a set of sparse planar primitives, estimate relative camera poses, and supervise geometry learning via planar splatting, where gradients are propagated through high-resolution rendered depth and normal maps of primitives.</td></tr>
<tr><td>2025-10-21</td><td>GeoDiff: Geometry-Guided Diffusion for Metric Depth Estimation</td><td>[2510.18291](http://arxiv.org/pdf/2510.18291)</td><td>◆ We introduce a novel framework for metric depth estimation that enhances pretrained diffusion-based monocular depth estimation (DB-MDE) models with stereo vision guidance.
◆ While existing DB-MDE methods excel at predicting relative depth, estimating absolute metric depth remains challenging due to scale ambiguities in single-image scenarios.
◆ To address this, we reframe depth estimation as an inverse problem, leveraging pretrained latent diffusion models (LDMs) conditioned on RGB images, combined with stereo-based geometric constraints, to learn scale and shift for accurate depth recovery.</td></tr>
<tr><td>2025-10-21</td><td>PAGE-4D: Disentangled Pose and Geometry Estimation for 4D Perception</td><td>[2510.17568](http://arxiv.org/pdf/2510.17568)</td><td>◆ Recent 3D feed-forward models, such as the Visual Geometry Grounded Transformer (VGGT), have shown strong capability in inferring 3D attributes of static scenes.
◆ However, since they are typically trained on static datasets, these models often struggle in real-world scenarios involving complex dynamic elements, such as moving humans or deformable objects like umbrellas.
◆ To address this limitation, we introduce PAGE-4D, a feedforward model that extends VGGT to dynamic scenes, enabling camera pose estimation, depth prediction, and point cloud reconstruction -- all without post-processing.</td></tr>
<tr><td>2025-10-19</td><td>How Universal Are SAM2 Features?</td><td>[2510.17051](http://arxiv.org/pdf/2510.17051)</td><td>◆ The trade-off between general-purpose foundation vision models and their specialized counterparts is critical for efficient feature coding design and is not yet fully understood.
◆ We investigate this trade-off by comparing the feature versatility of the general-purpose Hiera encoder against the segmentation-specialized Segment Anything Model 2 (SAM2).
◆ Using a lightweight, trainable neck to probe the adaptability of their frozen features, we quantify the information-theoretic cost of specialization.</td></tr>
<tr><td>2025-10-18</td><td>Self-Supervised Learning to Fly using Efficient Semantic Segmentation and Metric Depth Estimation for Low-Cost Autonomous UAVs</td><td>[2510.16624](http://arxiv.org/pdf/2510.16624)</td><td>◆ This paper presents a vision-only autonomous flight system for small UAVs operating in controlled indoor environments.
◆ The system combines semantic segmentation with monocular depth estimation to enable obstacle avoidance, scene exploration, and autonomous safe landing operations without requiring GPS or expensive sensors such as LiDAR.
◆ A key innovation is an adaptive scale factor algorithm that converts non-metric monocular depth predictions into accurate metric distance measurements by leveraging semantic ground plane detection and camera intrinsic parameters, achieving a mean distance error of 14.4 cm.</td></tr>
<tr><td>2025-10-18</td><td>OOS-DSD: Improving Out-of-stock Detection in Retail Images using Auxiliary Tasks</td><td>[2510.16508](http://arxiv.org/pdf/2510.16508)</td><td>◆ Out-of-stock (OOS) detection is a very important retail verification process that aims to infer the unavailability of products in their designated areas on the shelf.
◆ In this paper, we introduce OOS-DSD, a novel deep learning-based method that advances OOS detection through auxiliary learning.
◆ In particular, we extend a well-established YOLOv8 object detection architecture with additional convolutional branches to simultaneously detect OOS, segment products, and estimate scene depth.</td></tr>
<tr><td>2025-10-15</td><td>Decision-focused Sensing and Forecasting for Adaptive and Rapid Flood Response: An Implicit Learning Approach</td><td>[2510.16015](http://arxiv.org/pdf/2510.16015)</td><td>◆ Timely and reliable decision-making is vital for flood emergency response, yet it remains severely hindered by limited and imprecise situational awareness due to various budget and data accessibility constraints.
◆ Traditional flood management systems often rely on in-situ sensors to calibrate remote sensing-based large-scale flood depth forecasting models, and further take flood depth estimates to optimize flood response decisions.
◆ However, these approaches often take fixed, decision task-agnostic strategies to decide where to put in-situ sensors (e.g., maximize overall information gain) and train flood forecasting models (e.g., minimize average forecasting errors), but overlook that systems with the same sensing gain and average forecasting errors may lead to distinct decisions.</td></tr>
<tr><td>2025-10-16</td><td>SaLon3R: Structure-aware Long-term Generalizable 3D Reconstruction from Unposed Images</td><td>[2510.15072](http://arxiv.org/pdf/2510.15072)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) have enabled generalizable, on-the-fly reconstruction of sequential input views.
◆ However, existing methods often predict per-pixel Gaussians and combine Gaussians from all views as the scene representation, leading to substantial redundancies and geometric inconsistencies in long-duration video sequences.
◆ To address this, we propose SaLon3R, a novel framework for Structure-aware, Long-term 3DGS Reconstruction.</td></tr>
<tr><td>2025-10-16</td><td>C4D: 4D Made from 3D through Dual Correspondences</td><td>[2510.14960](http://arxiv.org/pdf/2510.14960)</td><td>◆ Recovering 4D from monocular video, which jointly estimates dynamic geometry and camera poses, is an inevitably challenging problem.
◆ While recent pointmap-based 3D reconstruction methods (e.g., DUSt3R) have made great progress in reconstructing static scenes, directly applying them to dynamic scenes leads to inaccurate results.
◆ This discrepancy arises because moving objects violate multi-view geometric constraints, disrupting the reconstruction.</td></tr>
<tr><td>2025-10-16</td><td>Multi-modal video data-pipelines for machine learning with minimal human supervision</td><td>[2510.14862](http://arxiv.org/pdf/2510.14862)</td><td>◆ The real-world is inherently multi-modal at its core.
◆ Our tools observe and take snapshots of it, in digital form, such as videos or sounds, however much of it is lost.
◆ Similarly for actions and information passing between humans, languages are used as a written form of communication.</td></tr>
<tr><td>2025-10-16</td><td>MatchAttention: Matching the Relative Positions for High-Resolution Cross-View Matching</td><td>[2510.14260](http://arxiv.org/pdf/2510.14260)</td><td>◆ Cross-view matching is fundamentally achieved through cross-attention mechanisms.
◆ However, matching of high-resolution images remains challenging due to the quadratic complexity and lack of explicit matching constraints in the existing cross-attention.
◆ This paper proposes an attention mechanism, MatchAttention, that dynamically matches relative positions.</td></tr>
<tr><td>2025-10-15</td><td>XD-RCDepth: Lightweight Radar-Camera Depth Estimation with Explainability-Aligned and Distribution-Aware Distillation</td><td>[2510.13565](http://arxiv.org/pdf/2510.13565)</td><td>◆ Depth estimation remains central to autonomous driving, and radar-camera fusion offers robustness in adverse conditions by providing complementary geometric cues.
◆ In this paper, we present XD-RCDepth, a lightweight architecture that reduces the parameters by 29.7% relative to the state-of-the-art lightweight baseline while maintaining comparable accuracy.
◆ To preserve performance under compression and enhance interpretability, we introduce two knowledge-distillation strategies: an explainability-aligned distillation that transfers the teacher&#x27;s saliency structure to the student, and a depth-distribution distillation that recasts depth regression as soft classification over discretized bins.</td></tr>
<tr><td>2025-10-15</td><td>FlyAwareV2: A Multimodal Cross-Domain UAV Dataset for Urban Scene Understanding</td><td>[2510.13243](http://arxiv.org/pdf/2510.13243)</td><td>◆ The development of computer vision algorithms for Unmanned Aerial Vehicle (UAV) applications in urban environments heavily relies on the availability of large-scale datasets with accurate annotations.
◆ However, collecting and annotating real-world UAV data is extremely challenging and costly.
◆ To address this limitation, we present FlyAwareV2, a novel multimodal dataset encompassing both real and synthetic UAV imagery tailored for urban scene understanding tasks.</td></tr>
<tr><td>2025-10-14</td><td>E-MoFlow: Learning Egomotion and Optical Flow from Event Data via Implicit Regularization</td><td>[2510.12753](http://arxiv.org/pdf/2510.12753)</td><td>◆ The estimation of optical flow and 6-DoF ego-motion, two fundamental tasks in 3D vision, has typically been addressed independently.
◆ For neuromorphic vision (e.g., event cameras), however, the lack of robust data association makes solving the two problems separately an ill-posed challenge, especially in the absence of supervision via ground truth.
◆ Existing works mitigate this ill-posedness by either enforcing the smoothness of the flow field via an explicit variational regularizer or leveraging explicit structure-and-motion priors in the parametrization to improve event alignment.</td></tr>
<tr><td>2025-10-17</td><td>Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model</td><td>[2510.12276](http://arxiv.org/pdf/2510.12276)</td><td>◆ Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions.
◆ However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world.
◆ Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets.</td></tr>
<tr><td>2025-10-13</td><td>Enhancing the Quality of 3D Lunar Maps Using JAXA&#x27;s Kaguya Imagery</td><td>[2510.11817](http://arxiv.org/pdf/2510.11817)</td><td>◆ As global efforts to explore the Moon intensify, the need for high-quality 3D lunar maps becomes increasingly critical-particularly for long-distance missions such as NASA&#x27;s Endurance mission concept, in which a rover aims to traverse 2,000 km across the South Pole-Aitken basin.
◆ Kaguya TC (Terrain Camera) images, though globally available at 10 m/pixel, suffer from altitude inaccuracies caused by stereo matching errors and JPEG-based compression artifacts.
◆ This paper presents a method to improve the quality of 3D maps generated from Kaguya TC images, focusing on mitigating the effects of compression-induced noise in disparity maps.</td></tr>
<tr><td>2025-10-13</td><td>Evaluating the effects of preprocessing, method selection, and hyperparameter tuning on SAR-based flood mapping and water depth estimation</td><td>[2510.11305](http://arxiv.org/pdf/2510.11305)</td><td>◆ Flood mapping and water depth estimation from Synthetic Aperture Radar (SAR) imagery are crucial for calibrating and validating hydraulic models.
◆ This study uses SAR imagery to evaluate various preprocessing (especially speckle noise reduction), flood mapping, and water depth estimation methods.
◆ The impact of the choice of method at different steps and its hyperparameters is studied by considering an ensemble of preprocessed images, flood maps, and water depth fields.</td></tr>
<tr><td>2025-10-11</td><td>Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting</td><td>[2510.10097](http://arxiv.org/pdf/2510.10097)</td><td>◆ Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have advanced 3D reconstruction and novel view synthesis, but remain heavily dependent on accurate camera poses and dense viewpoint coverage.
◆ These requirements limit their applicability in sparse-view settings, where pose estimation becomes unreliable and supervision is insufficient.
◆ To overcome these challenges, we introduce Gesplat, a 3DGS-based framework that enables robust novel view synthesis and geometrically consistent reconstruction from unposed sparse images.</td></tr>
<tr><td>2025-10-10</td><td>Fast Self-Supervised depth and mask aware Association for Multi-Object Tracking</td><td>[2510.09878](http://arxiv.org/pdf/2510.09878)</td><td>◆ Multi-object tracking (MOT) methods often rely on Intersection-over-Union (IoU) for association.
◆ However, this becomes unreliable when objects are similar or occluded.
◆ Also, computing IoU for segmentation masks is computationally expensive.</td></tr>
<tr><td>2025-10-10</td><td>Hybrid-grained Feature Aggregation with Coarse-to-fine Language Guidance for Self-supervised Monocular Depth Estimation</td><td>[2510.09320](http://arxiv.org/pdf/2510.09320)</td><td>◆ Current self-supervised monocular depth estimation (MDE) approaches encounter performance limitations due to insufficient semantic-spatial knowledge extraction.
◆ To address this challenge, we propose Hybrid-depth, a novel framework that systematically integrates foundation models (e.g., CLIP and DINO) to extract visual priors and acquire sufficient contextual information for MDE.
◆ Our approach introduces a coarse-to-fine progressive learning framework: 1) Firstly, we aggregate multi-grained features from CLIP (global semantics) and DINO (local spatial details) under contrastive language guidance.</td></tr>
<tr><td>2025-10-10</td><td>Online Video Depth Anything: Temporally-Consistent Depth Prediction with Low Memory Consumption</td><td>[2510.09182](http://arxiv.org/pdf/2510.09182)</td><td>◆ Depth estimation from monocular video has become a key component of many real-world computer vision systems.
◆ Recently, Video Depth Anything (VDA) has demonstrated strong performance on long video sequences.
◆ However, it relies on batch-processing which prohibits its use in an online setting.</td></tr>
<tr><td>2025-10-08</td><td>Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry</td><td>[2510.08638](http://arxiv.org/pdf/2510.08638)</td><td>◆ DINOv2 is routinely deployed to recognize objects, scenes, and actions; yet the nature of what it perceives remains unknown.
◆ As a working baseline, we adopt the Linear Representation Hypothesis (LRH) and operationalize it using SAEs, producing a 32,000-unit dictionary that serves as the interpretability backbone of our study, which unfolds in three parts.
◆ In the first part, we analyze how different downstream tasks recruit concepts from our learned dictionary, revealing functional specialization: classification exploits &quot;Elsewhere&quot; concepts that fire everywhere except on target objects, implementing learned negations; segmentation relies on boundary detectors forming coherent subspaces; depth estimation draws on three distinct monocular depth cues matching visual neuroscience principles.</td></tr>
<tr><td>2025-10-09</td><td>RayFusion: Ray Fusion Enhanced Collaborative Visual Perception</td><td>[2510.08017](http://arxiv.org/pdf/2510.08017)</td><td>◆ Collaborative visual perception methods have gained widespread attention in the autonomous driving community in recent years due to their ability to address sensor limitation problems.
◆ However, the absence of explicit depth information often makes it difficult for camera-based perception systems, e.g., 3D object detection, to generate accurate predictions.
◆ To alleviate the ambiguity in depth estimation, we propose RayFusion, a ray-based fusion method for collaborative visual perception.</td></tr>
<tr><td>2025-10-09</td><td>CVD-STORM: Cross-View Video Diffusion with Spatial-Temporal Reconstruction Model for Autonomous Driving</td><td>[2510.07944](http://arxiv.org/pdf/2510.07944)</td><td>◆ Generative models have been widely applied to world modeling for environment simulation and future state prediction.
◆ With advancements in autonomous driving, there is a growing demand not only for high-fidelity video generation under various controls, but also for producing diverse and meaningful information such as depth estimation.
◆ To address this, we propose CVD-STORM, a cross-view video diffusion model utilizing a spatial-temporal reconstruction Variational Autoencoder (VAE) that generates long-term, multi-view videos with 4D reconstruction capabilities under various control inputs.</td></tr>
<tr><td>2025-10-09</td><td>An End-to-End Room Geometry Constrained Depth Estimation Framework for Indoor Panorama Images</td><td>[2510.07817](http://arxiv.org/pdf/2510.07817)</td><td>◆ Predicting spherical pixel depth from monocular $360^{\circ}$ indoor panoramas is critical for many vision applications.
◆ However, existing methods focus on pixel-level accuracy, causing oversmoothed room corners and noise sensitivity.
◆ In this paper, we propose a depth estimation framework based on room geometry constraints, which extracts room geometry information through layout prediction and integrates those information into the depth estimation process through background segmentation mechanism.</td></tr>
<tr><td>2025-10-08</td><td>Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers</td><td>[2510.07316](http://arxiv.org/pdf/2510.07316)</td><td>◆ This paper presents Pixel-Perfect Depth, a monocular depth estimation model based on pixel-space diffusion generation that produces high-quality, flying-pixel-free point clouds from estimated depth maps.
◆ Current generative depth estimation models fine-tune Stable Diffusion and achieve impressive performance.
◆ However, they require a VAE to compress depth maps into latent space, which inevitably introduces \textit{flying pixels} at edges and details.</td></tr>
<tr><td>2025-10-08</td><td>MV-Performer: Taming Video Diffusion Model for Faithful and Synchronized Multi-view Performer Synthesis</td><td>[2510.07190](http://arxiv.org/pdf/2510.07190)</td><td>◆ Recent breakthroughs in video generation, powered by large-scale datasets and diffusion techniques, have shown that video diffusion models can function as implicit 4D novel view synthesizers.
◆ Nevertheless, current methods primarily concentrate on redirecting camera trajectory within the front view while struggling to generate 360-degree viewpoint changes.
◆ In this paper, we focus on human-centric subdomain and present MV-Performer, an innovative framework for creating synchronized novel view videos from monocular full-body captures.</td></tr>
<tr><td>2025-10-07</td><td>Human3R: Everyone Everywhere All at Once</td><td>[2510.06219](http://arxiv.org/pdf/2510.06219)</td><td>◆ We present Human3R, a unified, feed-forward framework for online 4D human-scene reconstruction, in the world frame, from casually captured monocular videos.
◆ Unlike previous approaches that rely on multi-stage pipelines, iterative contact-aware refinement between humans and scenes, and heavy dependencies, e.g., human detection, depth estimation, and SLAM pre-processing, Human3R jointly recovers global multi-person SMPL-X bodies (&quot;everyone&quot;), dense 3D scene (&quot;everywhere&quot;), and camera trajectories in a single forward pass (&quot;all-at-once&quot;).
◆ Our method builds upon the 4D online reconstruction model CUT3R, and uses parameter-efficient visual prompt tuning, to strive to preserve CUT3R&#x27;s rich spatiotemporal priors, while enabling direct readout of multiple SMPL-X bodies.</td></tr>
<tr><td>2025-10-07</td><td>EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark</td><td>[2510.06218](http://arxiv.org/pdf/2510.06218)</td><td>◆ Most existing benchmarks for egocentric vision understanding focus primarily on daytime scenarios, overlooking the low-light conditions that are inevitable in real-world applications.
◆ To investigate this gap, we present EgoNight, the first comprehensive benchmark for nighttime egocentric vision, with visual question answering (VQA) as the core task.
◆ A key feature of EgoNight is the introduction of day-night aligned videos, which enhance night annotation quality using the daytime data and reveal clear performance gaps between lighting conditions.</td></tr>
<tr><td>2025-10-07</td><td>Dropping the D: RGB-D SLAM Without the Depth Sensor</td><td>[2510.06216](http://arxiv.org/pdf/2510.06216)</td><td>◆ We present DropD-SLAM, a real-time monocular SLAM system that achieves RGB-D-level accuracy without relying on depth sensors.
◆ The system replaces active depth input with three pretrained vision modules: a monocular metric depth estimator, a learned keypoint detector, and an instance segmentation network.
◆ Dynamic objects are suppressed using dilated instance masks, while static keypoints are assigned predicted depth values and backprojected into 3D to form metrically scaled features.</td></tr>
<tr><td>2025-10-07</td><td>DeLTa: Demonstration and Language-Guided Novel Transparent Object Manipulation</td><td>[2510.05662](http://arxiv.org/pdf/2510.05662)</td><td>◆ Despite the prevalence of transparent object interactions in human everyday life, transparent robotic manipulation research remains limited to short-horizon tasks and basic grasping capabilities.Although some methods have partially addressed these issues, most of them have limitations in generalizability to novel objects and are insufficient for precise long-horizon robot manipulation.
◆ To address this limitation, we propose DeLTa (Demonstration and Language-Guided Novel Transparent Object Manipulation), a novel framework that integrates depth estimation, 6D pose estimation, and vision-language planning for precise long-horizon manipulation of transparent objects guided by natural task instructions.
◆ A key advantage of our method is its single-demonstration approach, which generalizes 6D trajectories to novel transparent objects without requiring category-level priors or additional training.</td></tr>
<tr><td>2025-10-09</td><td>Human Action Recognition from Point Clouds over Time</td><td>[2510.05506](http://arxiv.org/pdf/2510.05506)</td><td>◆ Recent research into human action recognition (HAR) has focused predominantly on skeletal action recognition and video-based methods.
◆ With the increasing availability of consumer-grade depth sensors and Lidar instruments, there is a growing opportunity to leverage dense 3D data for action recognition, to develop a third way.
◆ This paper presents a novel approach for recognizing actions from 3D videos by introducing a pipeline that segments human point clouds from the background of a scene, tracks individuals over time, and performs body part segmentation.</td></tr>
<tr><td>2025-10-06</td><td>HybridFlow: Quantification of Aleatoric and Epistemic Uncertainty with a Single Hybrid Model</td><td>[2510.05054](http://arxiv.org/pdf/2510.05054)</td><td>◆ Uncertainty quantification is critical for ensuring robustness in high-stakes machine learning applications.
◆ We introduce HybridFlow, a modular hybrid architecture that unifies the modeling of aleatoric and epistemic uncertainty by combining a Conditional Masked Autoregressive normalizing flow for estimating aleatoric uncertainty with a flexible probabilistic predictor for epistemic uncertainty.
◆ The framework supports integration with any probabilistic model class, allowing users to easily adapt HybridFlow to existing architectures without sacrificing predictive performance.</td></tr>
<tr><td>2025-10-06</td><td>Benchmark on Monocular Metric Depth Estimation in Wildlife Setting</td><td>[2510.04723](http://arxiv.org/pdf/2510.04723)</td><td>◆ Camera traps are widely used for wildlife monitoring, but extracting accurate distance measurements from monocular images remains challenging due to the lack of depth information.
◆ While monocular depth estimation (MDE) methods have advanced significantly, their performance in natural wildlife environments has not been systematically evaluated.
◆ This work introduces the first benchmark for monocular metric depth estimation in wildlife monitoring conditions.</td></tr>
<tr><td>2025-10-04</td><td>Evaluating High-Resolution Piano Sustain Pedal Depth Estimation with Musically Informed Metrics</td><td>[2510.03750](http://arxiv.org/pdf/2510.03750)</td><td>◆ Evaluation for continuous piano pedal depth estimation tasks remains incomplete when relying only on conventional frame-level metrics, which overlook musically important features such as direction-change boundaries and pedal curve contours.
◆ To provide more interpretable and musically meaningful insights, we propose an evaluation framework that augments standard frame-level metrics with an action-level assessment measuring direction and timing using segments of press/hold/release states and a gesture-level analysis that evaluates contour similarity of each press-release cycle.
◆ We apply this framework to compare an audio-only baseline with two variants: one incorporating symbolic information from MIDI, and another trained in a binary-valued setting, all within a unified architecture.</td></tr>
<tr><td>2025-10-03</td><td>Test-Time Defense Against Adversarial Attacks via Stochastic Resonance of Latent Ensembles</td><td>[2510.03224](http://arxiv.org/pdf/2510.03224)</td><td>◆ We propose a test-time defense mechanism against adversarial attacks: imperceptible image perturbations that significantly alter the predictions of a model.
◆ Unlike existing methods that rely on feature filtering or smoothing, which can lead to information loss, we propose to &quot;combat noise with noise&quot; by leveraging stochastic resonance to enhance robustness while minimizing information loss.
◆ Our approach introduces small translational perturbations to the input image, aligns the transformed feature embeddings, and aggregates them before mapping back to the original reference image.</td></tr>
<tr><td>2025-10-03</td><td>Whisker-based Tactile Flight for Tiny Drones</td><td>[2510.03119](http://arxiv.org/pdf/2510.03119)</td><td>◆ Tiny flying robots hold great potential for search-and-rescue, safety inspections, and environmental monitoring, but their small size limits conventional sensing-especially with poor-lighting, smoke, dust or reflective obstacles.
◆ Inspired by nature, we propose a lightweight, 3.2-gram, whisker-based tactile sensing apparatus for tiny drones, enabling them to navigate and explore through gentle physical interaction.
◆ Just as rats and moles use whiskers to perceive surroundings, our system equips drones with tactile perception in flight, allowing obstacle sensing even in pitch-dark conditions.</td></tr>
<tr><td>2025-10-02</td><td>Non-Rigid Structure-from-Motion via Differential Geometry with Recoverable Conformal Scale</td><td>[2510.01665](http://arxiv.org/pdf/2510.01665)</td><td>◆ Non-rigid structure-from-motion (NRSfM), a promising technique for addressing the mapping challenges in monocular visual deformable simultaneous localization and mapping (SLAM), has attracted growing attention.
◆ We introduce a novel method, called Con-NRSfM, for NRSfM under conformal deformations, encompassing isometric deformations as a subset.
◆ Our approach performs point-wise reconstruction using 2D selected image warps optimized through a graph-based framework.</td></tr>
<tr><td>2025-10-01</td><td>Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models</td><td>[2510.01184](http://arxiv.org/pdf/2510.01184)</td><td>◆ We present a mechanism to steer the sampling diversity of denoising diffusion and flow matching models, allowing users to sample from a sharper or broader distribution than the training distribution.
◆ We build on the observation that these models leverage (learned) score functions of noisy data distributions for sampling and show that rescaling these allows one to effectively control a `local&#x27; sampling temperature.
◆ Notably, this approach does not require any finetuning or alterations to training strategy, and can be applied to any off-the-shelf model and is compatible with both deterministic and stochastic samplers.</td></tr>
<tr><td>2025-09-30</td><td>DA$^2$: Depth Anything in Any Direction</td><td>[2509.26618](http://arxiv.org/pdf/2509.26618)</td><td>◆ Panorama has a full FoV (360$^\circ\times$180$^\circ$), offering a more complete visual description than perspective images.
◆ Thanks to this characteristic, panoramic depth estimation is gaining increasing traction in 3D vision.
◆ However, due to the scarcity of panoramic data, previous methods are often restricted to in-domain settings, leading to poor zero-shot generalization.</td></tr>
<tr><td>2025-09-30</td><td>DEPTHOR++: Robust Depth Enhancement from a Real-World Lightweight dToF and RGB Guidance</td><td>[2509.26498](http://arxiv.org/pdf/2509.26498)</td><td>◆ Depth enhancement, which converts raw dToF signals into dense depth maps using RGB guidance, is crucial for improving depth perception in high-precision tasks such as 3D reconstruction and SLAM.
◆ However, existing methods often assume ideal dToF inputs and perfect dToF-RGB alignment, overlooking calibration errors and anomalies, thus limiting real-world applicability.
◆ This work systematically analyzes the noise characteristics of real-world lightweight dToF sensors and proposes a practical and novel depth completion framework, DEPTHOR++, which enhances robustness to noisy dToF inputs from three key aspects.</td></tr>
<tr><td>2025-09-30</td><td>EasyOcc: 3D Pseudo-Label Supervision for Fully Self-Supervised Semantic Occupancy Prediction Models</td><td>[2509.26087](http://arxiv.org/pdf/2509.26087)</td><td>◆ Self-supervised models have recently achieved notable advancements, particularly in the domain of semantic occupancy prediction.
◆ These models utilize sophisticated loss computation strategies to compensate for the absence of ground-truth labels.
◆ For instance, techniques such as novel view synthesis, cross-view rendering, and depth estimation have been explored to address the issue of semantic and depth ambiguity.</td></tr>
<tr><td>2025-09-30</td><td>PFDepth: Heterogeneous Pinhole-Fisheye Joint Depth Estimation via Distortion-aware Gaussian-Splatted Volumetric Fusion</td><td>[2509.26008](http://arxiv.org/pdf/2509.26008)</td><td>◆ In this paper, we present the first pinhole-fisheye framework for heterogeneous multi-view depth estimation, PFDepth.
◆ Our key insight is to exploit the complementary characteristics of pinhole and fisheye imagery (undistorted vs.
◆ distorted, small vs.</td></tr>
<tr><td>2025-10-01</td><td>DepthLM: Metric Depth From Vision Language Models</td><td>[2509.25413](http://arxiv.org/pdf/2509.25413)</td><td>◆ Vision language models (VLMs) can flexibly address various vision tasks through text interactions.
◆ Although successful in semantic understanding, state-of-the-art VLMs including GPT-5 still struggle in understanding 3D from 2D inputs.
◆ On the other hand, expert pure vision models achieve super-human accuracy in metric depth estimation, a key 3D understanding task.</td></tr>
<tr><td>2025-09-29</td><td>Fast Feature Field ($\text{F}^3$): A Predictive Representation of Events</td><td>[2509.25146](http://arxiv.org/pdf/2509.25146)</td><td>◆ This paper develops a mathematical argument and algorithms for building representations of data from event-based cameras, that we call Fast Feature Field ($\text{F}^3$).
◆ We learn this representation by predicting future events from past events and show that it preserves scene structure and motion information.
◆ $\text{F}^3$ exploits the sparsity of event data and is robust to noise and variations in event rates.</td></tr>
<tr><td>2025-09-30</td><td>BRIDGE -- Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation</td><td>[2509.25077](http://arxiv.org/pdf/2509.25077)</td><td>◆ Monocular Depth Estimation (MDE) is a foundational task for computer vision.
◆ Traditional methods are limited by data scarcity and quality, hindering their robustness.
◆ To overcome this, we propose BRIDGE, an RL-optimized depth-to-image (D2I) generation framework that synthesizes over 20M realistic and geometrically accurate RGB images, each intrinsically paired with its ground truth depth, from diverse source depth maps.</td></tr>
<tr><td>2025-09-29</td><td>DWGS: Enhancing Sparse-View Gaussian Splatting with Hybrid-Loss Depth Estimation and Bidirectional Warping</td><td>[2509.24893](http://arxiv.org/pdf/2509.24893)</td><td>◆ Novel View Synthesis (NVS) from sparse views remains a core challenge in 3D reconstruction, typically suffering from overfitting, geometric distortion, and incomplete scene recovery due to limited multi-view constraints.
◆ Although 3D Gaussian Splatting (3DGS) enables real-time, high-fidelity rendering, it suffers from floating artifacts and structural inconsistencies under sparse-input settings.
◆ To address these issues, we propose DWGS, a novel unified framework that enhances 3DGS for sparse-view synthesis by integrating robust structural cues, virtual view constraints, and occluded region completion.</td></tr>
<tr><td>2025-09-28</td><td>RPG360: Robust 360 Depth Estimation with Perspective Foundation Models and Graph Optimization</td><td>[2509.23991](http://arxiv.org/pdf/2509.23991)</td><td>◆ The increasing use of 360 images across various domains has emphasized the need for robust depth estimation techniques tailored for omnidirectional images.
◆ However, obtaining large-scale labeled datasets for 360 depth estimation remains a significant challenge.
◆ In this paper, we propose RPG360, a training-free robust 360 monocular depth estimation method that leverages perspective foundation models and graph optimization.</td></tr>
<tr><td>2025-09-28</td><td>FastViDAR: Real-Time Omnidirectional Depth Estimation via Alternative Hierarchical Attention</td><td>[2509.23733](http://arxiv.org/pdf/2509.23733)</td><td>◆ In this paper we propose FastViDAR, a novel framework that takes four fisheye camera inputs and produces a full $360^\circ$ depth map along with per-camera depth, fusion depth, and confidence estimates.
◆ Our main contributions are: (1) We introduce Alternative Hierarchical Attention (AHA) mechanism that efficiently fuses features across views through separate intra-frame and inter-frame windowed self-attention, achieving cross-view feature mixing with reduced overhead.
◆ (2) We propose a novel ERP fusion approach that projects multi-view depth estimates to a shared equirectangular coordinate system to obtain the final fusion depth.</td></tr>
<tr><td>2025-09-28</td><td>Efficient Domain-Adaptive Multi-Task Dense Prediction with Vision Foundation Models</td><td>[2509.23626](http://arxiv.org/pdf/2509.23626)</td><td>◆ Multi-task dense prediction, which aims to jointly solve tasks like semantic segmentation and depth estimation, is crucial for robotics applications but suffers from domain shift when deploying models in new environments.
◆ While unsupervised domain adaptation (UDA) addresses this challenge for single tasks, existing multi-task UDA methods primarily rely on adversarial learning approaches that are less effective than recent self-training techniques.
◆ In this paper, we introduce FAMDA, a simple yet effective UDA framework that bridges this gap by leveraging Vision Foundation Models (VFMs) as powerful teachers.</td></tr>
<tr><td>2025-09-26</td><td>CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach</td><td>[2509.22627](http://arxiv.org/pdf/2509.22627)</td><td>◆ Depth Estimation plays a crucial role in recent applications in robotics, autonomous vehicles, and augmented reality.
◆ These scenarios commonly operate under constraints imposed by computational power.
◆ Stereo image pairs offer an effective solution for depth estimation since it only needs to estimate the disparity of pixels in image pairs to determine the depth in a known rectified system.</td></tr>
<tr><td>2025-09-26</td><td>EfficientDepth: A Fast and Detail-Preserving Monocular Depth Estimation Model</td><td>[2509.22527](http://arxiv.org/pdf/2509.22527)</td><td>◆ Monocular depth estimation (MDE) plays a pivotal role in various computer vision applications, such as robotics, augmented reality, and autonomous driving.
◆ Despite recent advancements, existing methods often fail to meet key requirements for 3D reconstruction and view synthesis, including geometric consistency, fine details, robustness to real-world challenges like reflective surfaces, and efficiency for edge devices.
◆ To address these challenges, we introduce a novel MDE system, called EfficientDepth, which combines a transformer architecture with a lightweight convolutional decoder, as well as a bimodal density head that allows the network to estimate detailed depth maps.</td></tr>
<tr><td>2025-09-26</td><td>DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints</td><td>[2509.21992](http://arxiv.org/pdf/2509.21992)</td><td>◆ Depth-from-Focus (DFF) enables precise depth estimation by analyzing focus cues across a stack of images captured at varying focal lengths.
◆ While recent learning-based approaches have advanced this field, they often struggle in complex scenes with fine textures or abrupt depth changes, where focus cues may become ambiguous or misleading.
◆ We present DualFocus, a novel DFF framework that leverages the focal stack&#x27;s unique gradient patterns induced by focus variation, jointly modeling focus changes over spatial and focal dimensions.</td></tr>
<tr><td>2025-09-25</td><td>Finding 3D Positions of Distant Objects from Noisy Camera Movement and Semantic Segmentation Sequences</td><td>[2509.20906](http://arxiv.org/pdf/2509.20906)</td><td>◆ 3D object localisation based on a sequence of camera measurements is essential for safety-critical surveillance tasks, such as drone-based wildfire monitoring.
◆ Localisation of objects detected with a camera can typically be solved with dense depth estimation or 3D scene reconstruction.
◆ However, in the context of distant objects or tasks limited by the amount of available computational resources, neither solution is feasible.</td></tr>
<tr><td>2025-09-24</td><td>Shared Neural Space: Unified Precomputed Feature Encoding for Multi-Task and Cross Domain Vision</td><td>[2509.20481](http://arxiv.org/pdf/2509.20481)</td><td>◆ The majority of AI models in imaging and vision are customized to perform on specific high-precision task.
◆ However, this strategy is inefficient for applications with a series of modular tasks, since each requires a mapping into a disparate latent domain.
◆ To address this inefficiency, we proposed a universal Neural Space (NS), where an encoder-decoder framework pre-computes features across vision and imaging tasks.</td></tr>
<tr><td>2025-09-24</td><td>BiTAA: A Bi-Task Adversarial Attack for Object Detection and Depth Estimation via 3D Gaussian Splatting</td><td>[2509.19793](http://arxiv.org/pdf/2509.19793)</td><td>◆ Camera-based perception is critical to autonomous driving yet remains vulnerable to task-specific adversarial manipulations in object detection and monocular depth estimation.
◆ Most existing 2D/3D attacks are developed in task silos, lack mechanisms to induce controllable depth bias, and offer no standardized protocol to quantify cross-task transfer, leaving the interaction between detection and depth underexplored.
◆ We present BiTAA, a bi-task adversarial attack built on 3D Gaussian Splatting that yields a single perturbation capable of simultaneously degrading detection and biasing monocular depth.</td></tr>
<tr><td>2025-09-24</td><td>VIMD: Monocular Visual-Inertial Motion and Depth Estimation</td><td>[2509.19713](http://arxiv.org/pdf/2509.19713)</td><td>◆ Accurate and efficient dense metric depth estimation is crucial for 3D visual perception in robotics and XR.
◆ In this paper, we develop a monocular visual-inertial motion and depth (VIMD) learning framework to estimate dense metric depth by leveraging accurate and efficient MSCKF-based monocular visual-inertial motion tracking.
◆ At the core the proposed VIMD is to exploit multi-view information to iteratively refine per-pixel scale, instead of globally fitting an invariant affine model as in the prior work.</td></tr>
<tr><td>2025-09-24</td><td>Enhancing Transformer-Based Vision Models: Addressing Feature Map Anomalies Through Novel Optimization Strategies</td><td>[2509.19687](http://arxiv.org/pdf/2509.19687)</td><td>◆ Vision Transformers (ViTs) have demonstrated superior performance across a wide range of computer vision tasks.
◆ However, structured noise artifacts in their feature maps hinder downstream applications such as segmentation and depth estimation.
◆ We propose two novel and lightweight optimisation techniques- Structured Token Augmentation (STA) and Adaptive Noise Filtering (ANF)- to improve interpretability and mitigate these artefacts.</td></tr>
<tr><td>2025-09-24</td><td>An on-chip Pixel Processing Approach with 2.4μs latency for Asynchronous Read-out of SPAD-based dToF Flash LiDARs</td><td>[2509.19192](http://arxiv.org/pdf/2509.19192)</td><td>◆ We propose a fully asynchronous peak detection approach for SPAD-based direct time-of-flight (dToF) flash LiDAR, enabling pixel-wise event-driven depth acquisition without global synchronization.
◆ By allowing pixels to independently report depth once a sufficient signal-to-noise ratio is achieved, the method reduces latency, mitigates motion blur, and increases effective frame rate compared to frame-based systems.
◆ The framework is validated under two hardware implementations: an offline 256$\times$128 SPAD array with PC based processing and a real-time FPGA proof-of-concept prototype with 2.4$\upmu$s latency for on-chip integration.</td></tr>
<tr><td>2025-09-23</td><td>RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions</td><td>[2509.19165](http://arxiv.org/pdf/2509.19165)</td><td>◆ Recent self-supervised stereo matching methods have made significant progress, but their performance significantly degrades under adverse weather conditions such as night, rain, and fog.
◆ We identify two primary weaknesses contributing to this performance degradation.
◆ First, adverse weather introduces noise and reduces visibility, making CNN-based feature extractors struggle with degraded regions like reflective and textureless areas.</td></tr>
<tr><td>2025-09-23</td><td>RS3DBench: A Comprehensive Benchmark for 3D Spatial Perception in Remote Sensing</td><td>[2509.18897](http://arxiv.org/pdf/2509.18897)</td><td>◆ In this paper, we introduce a novel benchmark designed to propel the advancement of general-purpose, large-scale 3D vision models for remote sensing imagery.
◆ While several datasets have been proposed within the realm of remote sensing, many existing collections either lack comprehensive depth information or fail to establish precise alignment between depth data and remote sensing images.
◆ To address this deficiency, we present a visual Benchmark for 3D understanding of Remotely Sensed images, dubbed RS3DBench.</td></tr>
<tr><td>2025-09-23</td><td>Zero-shot Monocular Metric Depth for Endoscopic Images</td><td>[2509.18642](http://arxiv.org/pdf/2509.18642)</td><td>◆ Monocular relative and metric depth estimation has seen a tremendous boost in the last few years due to the sharp advancements in foundation models and in particular transformer based networks.
◆ As we start to see applications to the domain of endoscopic images, there is still a lack of robust benchmarks and high-quality datasets in that area.
◆ This paper addresses these limitations by presenting a comprehensive benchmark of state-of-the-art (metric and relative) depth estimation models evaluated on real, unseen endoscopic images, providing critical insights into their generalisation and performance in clinical scenarios.</td></tr>
<tr><td>2025-09-22</td><td>RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds</td><td>[2509.18068](http://arxiv.org/pdf/2509.18068)</td><td>◆ Millimeter-wave radar provides perception robust to fog, smoke, dust, and low light, making it attractive for size, weight, and power constrained robotic platforms.
◆ Current radar imaging methods, however, rely on synthetic aperture or multi-frame aggregation to improve resolution, which is impractical for small aerial, inspection, or wearable systems.
◆ We present RadarSFD, a conditional latent diffusion framework that reconstructs dense LiDAR-like point clouds from a single radar frame without motion or SAR.</td></tr>
<tr><td>2025-09-22</td><td>Predicting Depth Maps from Single RGB Images and Addressing Missing Information in Depth Estimation</td><td>[2509.17686](http://arxiv.org/pdf/2509.17686)</td><td>◆ Depth imaging is a crucial area in Autonomous Driving Systems (ADS), as it plays a key role in detecting and measuring objects in the vehicle&#x27;s surroundings.
◆ However, a significant challenge in this domain arises from missing information in Depth images, where certain points are not measurable due to gaps or inconsistencies in pixel data.
◆ Our research addresses two key tasks to overcome this challenge.</td></tr>
<tr><td>2025-09-22</td><td>Evict3R: Training-Free Token Eviction for Memory-Bounded Streaming Visual Geometry Transformers</td><td>[2509.17650](http://arxiv.org/pdf/2509.17650)</td><td>◆ Streaming visual transformers like StreamVGGT achieve strong 3D perception but suffer from unbounded growth of key value (KV) memory, which limits scalability.
◆ We propose a training-free, inference-time token eviction policy that bounds memory by discarding redundant tokens while keeping the most informative ones.
◆ Our method uses significantly less memory with little to no drop in accuracy: on 7-Scenes with long sequences it reduces peak memory from 18.63 GB to 9.39 GB while accuracy and completeness drop by only 0.003.</td></tr>
<tr><td>2025-09-22</td><td>GPS Denied IBVS-Based Navigation and Collision Avoidance of UAV Using a Low-Cost RGB Camera</td><td>[2509.17435](http://arxiv.org/pdf/2509.17435)</td><td>◆ This paper proposes an image-based visual servoing (IBVS) framework for UAV navigation and collision avoidance using only an RGB camera.
◆ While UAV navigation has been extensively studied, it remains challenging to apply IBVS in missions involving multiple visual targets and collision avoidance.
◆ The proposed method achieves navigation without explicit path planning, and collision avoidance is realized through AI-based monocular depth estimation from RGB images.</td></tr>
<tr><td>2025-09-21</td><td>ConfidentSplat: Confidence-Weighted Depth Fusion for Accurate 3D Gaussian Splatting SLAM</td><td>[2509.16863](http://arxiv.org/pdf/2509.16863)</td><td>◆ We introduce ConfidentSplat, a novel 3D Gaussian Splatting (3DGS)-based SLAM system for robust, highfidelity RGB-only reconstruction.
◆ Addressing geometric inaccuracies in existing RGB-only 3DGS SLAM methods that stem from unreliable depth estimation, ConfidentSplat incorporates a core innovation: a confidence-weighted fusion mechanism.
◆ This mechanism adaptively integrates depth cues from multiview geometry with learned monocular priors (Omnidata ViT), dynamically weighting their contributions based on explicit reliability estimates-derived predominantly from multi-view geometric consistency-to generate high-fidelity proxy depth for map supervision.</td></tr>
<tr><td>2025-09-23</td><td>3D Gaussian Flats: Hybrid 2D/3D Photometric Scene Reconstruction</td><td>[2509.16423](http://arxiv.org/pdf/2509.16423)</td><td>◆ Recent advances in radiance fields and novel view synthesis enable creation of realistic digital twins from photographs.
◆ However, current methods struggle with flat, texture-less surfaces, creating uneven and semi-transparent reconstructions, due to an ill-conditioned photometric reconstruction objective.
◆ Surface reconstruction methods solve this issue but sacrifice visual quality.</td></tr>
<tr><td>2025-09-19</td><td>StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes</td><td>[2509.16415](http://arxiv.org/pdf/2509.16415)</td><td>◆ Underwater stereo depth estimation provides accurate 3D geometry for robotics tasks such as navigation, inspection, and mapping, offering metric depth from low-cost passive cameras while avoiding the scale ambiguity of monocular methods.
◆ However, existing approaches face two critical challenges: (i) parameter-efficiently adapting large vision foundation encoders to the underwater domain without extensive labeled data, and (ii) tightly fusing globally coherent but scale-ambiguous monocular priors with locally metric yet photometrically fragile stereo correspondences.
◆ To address these challenges, we propose StereoAdapter, a parameter-efficient self-supervised framework that integrates a LoRA-adapted monocular foundation encoder with a recurrent stereo refinement module.</td></tr>
<tr><td>2025-09-19</td><td>Towards Sharper Object Boundaries in Self-Supervised Depth Estimation</td><td>[2509.15987](http://arxiv.org/pdf/2509.15987)</td><td>◆ Accurate monocular depth estimation is crucial for 3D scene understanding, but existing methods often blur depth at object boundaries, introducing spurious intermediate 3D points.
◆ While achieving sharp edges usually requires very fine-grained supervision, our method produces crisp depth discontinuities using only self-supervision.
◆ Specifically, we model per-pixel depth as a mixture distribution, capturing multiple plausible depths and shifting uncertainty from direct regression to the mixture weights.</td></tr>
<tr><td>2025-09-19</td><td>Shedding Light on Depth: Explainability Assessment in Monocular Depth Estimation</td><td>[2509.15980](http://arxiv.org/pdf/2509.15980)</td><td>◆ Explainable artificial intelligence is increasingly employed to understand the decision-making process of deep learning models and create trustworthiness in their adoption.
◆ However, the explainability of Monocular Depth Estimation (MDE) remains largely unexplored despite its wide deployment in real-world applications.
◆ In this work, we study how to analyze MDE networks to map the input image to the predicted depth map.</td></tr>
<tr><td>2025-09-19</td><td>Global Regulation and Excitation via Attention Tuning for Stereo Matching</td><td>[2509.15891](http://arxiv.org/pdf/2509.15891)</td><td>◆ Stereo matching achieves significant progress with iterative algorithms like RAFT-Stereo and IGEV-Stereo.
◆ However, these methods struggle in ill-posed regions with occlusions, textureless, or repetitive patterns, due to a lack of global context and geometric information for effective iterative refinement.
◆ To enable the existing iterative approaches to incorporate global context, we propose the Global Regulation and Excitation via Attention Tuning (GREAT) framework which encompasses three attention modules.</td></tr>
<tr><td>2025-09-19</td><td>MS-GS: Multi-Appearance Sparse-View 3D Gaussian Splatting in the Wild</td><td>[2509.15548](http://arxiv.org/pdf/2509.15548)</td><td>◆ In-the-wild photo collections often contain limited volumes of imagery and exhibit multiple appearances, e.g., taken at different times of day or seasons, posing significant challenges to scene reconstruction and novel view synthesis.
◆ Although recent adaptations of Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) have improved in these areas, they tend to oversmooth and are prone to overfitting.
◆ In this paper, we present MS-GS, a novel framework designed with Multi-appearance capabilities in Sparse-view scenarios using 3DGS.</td></tr>
<tr><td>2025-09-18</td><td>Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation</td><td>[2509.15224](http://arxiv.org/pdf/2509.15224)</td><td>◆ Event cameras capture sparse, high-temporal-resolution visual information, making them particularly suitable for challenging environments with high-speed motion and strongly varying lighting conditions.
◆ However, the lack of large datasets with dense ground-truth depth annotations hinders learning-based monocular depth estimation from event data.
◆ To address this limitation, we propose a cross-modal distillation paradigm to generate dense proxy labels leveraging a Vision Foundation Model (VFM).</td></tr>
<tr><td>2025-09-18</td><td>Lightweight and Accurate Multi-View Stereo with Confidence-Aware Diffusion Model</td><td>[2509.15220](http://arxiv.org/pdf/2509.15220)</td><td>◆ To reconstruct the 3D geometry from calibrated images, learning-based multi-view stereo (MVS) methods typically perform multi-view depth estimation and then fuse depth maps into a mesh or point cloud.
◆ To improve the computational efficiency, many methods initialize a coarse depth map and then gradually refine it in higher resolutions.
◆ Recently, diffusion models achieve great success in generation tasks.</td></tr>
<tr><td>2025-09-18</td><td>UCorr: Wire Detection and Depth Estimation for Autonomous Drones</td><td>[2509.14989](http://arxiv.org/pdf/2509.14989)</td><td>◆ In the realm of fully autonomous drones, the accurate detection of obstacles is paramount to ensure safe navigation and prevent collisions.
◆ Among these challenges, the detection of wires stands out due to their slender profile, which poses a unique and intricate problem.
◆ To address this issue, we present an innovative solution in the form of a monocular end-to-end model for wire segmentation and depth estimation.</td></tr>
<tr><td>2025-09-18</td><td>MapAnything: Mapping Urban Assets using Single Street-View Images</td><td>[2509.14839](http://arxiv.org/pdf/2509.14839)</td><td>◆ To maintain an overview of urban conditions, city administrations manage databases of objects like traffic signs and trees, complete with their geocoordinates.
◆ Incidents such as graffiti or road damage are also relevant.
◆ As digitization increases, so does the need for more data and up-to-date databases, requiring significant manual effort.</td></tr>
<tr><td>2025-09-16</td><td>\textsc{Gen2Real}: Towards Demo-Free Dexterous Manipulation by Harnessing Generated Video</td><td>[2509.14178](http://arxiv.org/pdf/2509.14178)</td><td>◆ Dexterous manipulation remains a challenging robotics problem, largely due to the difficulty of collecting extensive human demonstrations for learning.
◆ In this paper, we introduce \textsc{Gen2Real}, which replaces costly human demos with one generated video and drives robot skill from it: it combines demonstration generation that leverages video generation with pose and depth estimation to yield hand-object trajectories, trajectory optimization that uses Physics-aware Interaction Optimization Model (PIOM) to impose physics consistency, and demonstration learning that retargets human motions to a robot hand and stabilizes control with an anchor-based residual Proximal Policy Optimization (PPO) policy.
◆ Using only generated videos, the learned policy achieves a 77.3\% success rate on grasping tasks in simulation and demonstrates coherent executions on a real robot.</td></tr>
<tr><td>2025-09-17</td><td>UM-Depth : Uncertainty Masked Self-Supervised Monocular Depth Estimation with Visual Odometry</td><td>[2509.13713](http://arxiv.org/pdf/2509.13713)</td><td>◆ Monocular depth estimation has been increasingly adopted in robotics and autonomous driving for its ability to infer scene geometry from a single camera.
◆ In self-supervised monocular depth estimation frameworks, the network jointly generates and exploits depth and pose estimates during training, thereby eliminating the need for depth labels.
◆ However, these methods remain challenged by uncertainty in the input data, such as low-texture or dynamic regions, which can cause reduced depth accuracy.</td></tr>
<tr><td>2025-09-17</td><td>Gaussian Alignment for Relative Camera Pose Estimation via Single-View Reconstruction</td><td>[2509.13652](http://arxiv.org/pdf/2509.13652)</td><td>◆ Estimating metric relative camera pose from a pair of images is of great importance for 3D reconstruction and localisation.
◆ However, conventional two-view pose estimation methods are not metric, with camera translation known only up to a scale, and struggle with wide baselines and textureless or reflective surfaces.
◆ This paper introduces GARPS, a training-free framework that casts this problem as the direct alignment of two independently reconstructed 3D scenes.</td></tr>
<tr><td>2025-09-16</td><td>ColonCrafter: A Depth Estimation Model for Colonoscopy Videos Using Diffusion Priors</td><td>[2509.13525](http://arxiv.org/pdf/2509.13525)</td><td>◆ Three-dimensional (3D) scene understanding in colonoscopy presents significant challenges that necessitate automated methods for accurate depth estimation.
◆ However, existing depth estimation models for endoscopy struggle with temporal consistency across video sequences, limiting their applicability for 3D reconstruction.
◆ We present ColonCrafter, a diffusion-based depth estimation model that generates temporally consistent depth maps from monocular colonoscopy videos.</td></tr>
<tr><td>2025-09-18</td><td>MINGLE: VLMs for Semantically Complex Region Detection in Urban Scenes</td><td>[2509.13484](http://arxiv.org/pdf/2509.13484)</td><td>◆ Understanding group-level social interactions in public spaces is crucial for urban planning, informing the design of socially vibrant and inclusive environments.
◆ Detecting such interactions from images involves interpreting subtle visual cues such as relations, proximity, and co-movement - semantically complex signals that go beyond traditional object detection.
◆ To address this challenge, we introduce a social group region detection task, which requires inferring and spatially grounding visual regions defined by abstract interpersonal relations.</td></tr>
<tr><td>2025-09-16</td><td>MapAnything: Universal Feed-Forward Metric 3D Reconstruction</td><td>[2509.13414](http://arxiv.org/pdf/2509.13414)</td><td>◆ We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras.
◆ MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame.
◆ Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more.</td></tr>
<tr><td>2025-09-16</td><td>ROOM: A Physics-Based Continuum Robot Simulator for Photorealistic Medical Datasets Generation</td><td>[2509.13177](http://arxiv.org/pdf/2509.13177)</td><td>◆ Continuum robots are advancing bronchoscopy procedures by accessing complex lung airways and enabling targeted interventions.
◆ However, their development is limited by the lack of realistic training and test environments: Real data is difficult to collect due to ethical constraints and patient safety concerns, and developing autonomy algorithms requires realistic imaging and physical feedback.
◆ We present ROOM (Realistic Optical Observation in Medicine), a comprehensive simulation framework designed for generating photorealistic bronchoscopy training data.</td></tr>
<tr><td>2025-09-16</td><td>StereoCarla: A High-Fidelity Driving Dataset for Generalizable Stereo</td><td>[2509.12683](http://arxiv.org/pdf/2509.12683)</td><td>◆ Stereo matching plays a crucial role in enabling depth perception for autonomous driving and robotics.
◆ While recent years have witnessed remarkable progress in stereo matching algorithms, largely driven by learning-based methods and synthetic datasets, the generalization performance of these models remains constrained by the limited diversity of existing training data.
◆ To address these challenges, we present StereoCarla, a high-fidelity synthetic stereo dataset specifically designed for autonomous driving scenarios.</td></tr>
<tr><td>2025-09-15</td><td>BREA-Depth: Bronchoscopy Realistic Airway-geometric Depth Estimation</td><td>[2509.11885](http://arxiv.org/pdf/2509.11885)</td><td>该论文提出了一种用于支气管镜真实气道几何深度估计的新框架BREA-Depth，其核心贡献在于显著提升了单目深度估计在复杂支气管环境中的解剖结构感知能力和准确性。  
◆创新性地提出将气道特异性几何先验知识集成到基础模型适应中，以增强对全局气道结构的理解。  
◆设计了深度感知的CycleGAN，有效弥合了真实支气管图像与来自解剖数据的气道几何形状之间的域差距。  
◆引入了气道结构感知损失函数，在保持平滑过渡和结构完整性的同时，确保了气道管腔内深度的一致性。  
◆提出了一个新的评估指标“气道深度结构评估”（Airway Depth Structure Evaluation），用于衡量深度图的结构一致性和解剖真实性。  
该方法在离体人肺数据集和公开支气管镜数据集上均优于现有方法，能生成更鲁棒、准确的三维气道重建结果。</td></tr>
<tr><td>2025-09-14</td><td>In-Vivo Skin 3-D Surface Reconstruction and Wrinkle Depth Estimation using Handheld High Resolution Tactile Sensing</td><td>[2509.11385](http://arxiv.org/pdf/2509.11385)</td><td>本文开发了一种基于高分辨率触觉成像的手持式皮肤三维重建系统，首次实现了在体、多部位、微米级精度的皱纹深度量化评估。  
◆ 提出一种结合定制弹性凝胶和基于学习算法的GelSight触觉成像探头，实现了微米级精度的皮肤三维表面重建。  
◆ 设计集成力传感的手持式探头，确保测量时接触力一致，提升重建结果的稳定性和可靠性。  
◆ 在无皮肤病受试者中首次实现了多身体区域（如面部、手臂等）皱纹深度的有效验证与定量分析。  
◆ 系统在仿皱纹测试物体上达到12.55微米的平均绝对误差，并成功监测到使用普通保湿霜后三个部位皱纹深度的统计学显著下降。  
该技术为临床皮肤诊断、化妆品功效评价和治疗监测提供了可靠的量化工具。</td></tr>
<tr><td>2025-09-14</td><td>The System Description of CPS Team for Track on Driving with Language of CVPR 2024 Autonomous Grand Challenge</td><td>[2509.11071](http://arxiv.org/pdf/2509.11071)</td><td>该论文核心贡献是开发了一套基于视觉语言模型的系统，在CVPR 2024自动驾驶挑战赛的“语言驾驶”赛道中取得了第一名。其创新点主要包括：
◆ 专门使用DriveLM-nuScenes数据集进行训练，确保了数据针对性。
◆ 基于LLaVA模型，并采用LoRA和DoRA微调方法进行增强，提升了模型适应能力。
◆ 集成开源深度估计模型提供的深度信息，丰富了训练和推理过程的环境感知。
◆ 在推理阶段引入思维链（Chain-of-Thought）推理方法，有效提高了多项选择和判断题的准确性。
这套综合方法使该系统在验证集上获得了0.7799的最高分，位列排行榜榜首。</td></tr>
<tr><td>2025-09-12</td><td>Self-supervised Learning Of Visual Pose Estimation Without Pose Labels By Classifying LED States</td><td>[2509.10405](http://arxiv.org/pdf/2509.10405)</td><td>该论文提出了一种无需姿态标签的自监督视觉位姿估计方法，其核心创新如下：
◆ 利用LED状态分类作为代理任务，使模型从零开始学习单目RGB图像的相对位姿估计，无需任何位姿标签或机器人外观先验知识。
◆ 仅需训练时已知LED状态和大致视角方向，推理时LED状态可任意且不影响性能，实现了硬件约束与算法解耦。
◆ 通过校准图像解决单目深度估计的尺度模糊性问题，仅需一张已知目标距离的校准图像即可实现尺度恢复。
◆ 训练数据由机器人随机运动自主采集，无需外部基础设施或人工监督，显著提升了数据获取的便捷性和可扩展性。
◆ 实验表明该方法与需全监督或CAD模型的先进方法性能相当，且具备跨域泛化能力和多机器人位姿估计适应性。</td></tr>
<tr><td>2025-09-10</td><td>Computational Imaging for Enhanced Computer Vision</td><td>[2509.08712](http://arxiv.org/pdf/2509.08712)</td><td>该论文系统综述了计算成像技术如何提升计算机视觉应用的性能。  
◆ 系统性地建立了计算成像技术与计算机视觉核心任务之间的协同关系框架。  
◆ 深入分析了多种计算成像技术（如光场成像、HDR成像、去模糊等）在视觉任务中的具体贡献。  
◆ 强调了针对特定任务的自适应成像流程的开发潜力，以提高实际应用的鲁棒性和准确性。  
◆ 指出了该交叉领域在自动驾驶、监控、增强现实等现实场景中的新兴机遇与未来研究方向。  
论文的核心创新在于为计算成像与计算机视觉的融合研究提供了全面的路线图和深入的分析视角。</td></tr>
<tr><td>2025-09-10</td><td>Deep Visual Odometry for Stereo Event Cameras</td><td>[2509.08235](http://arxiv.org/pdf/2509.08235)</td><td>该论文提出了一种基于深度学习的立体事件相机视觉里程计系统Stereo-DEVO，其核心贡献在于实现了在极端光照条件下高精度、实时的位姿估计。  
◆ 提出了一种新颖且高效的静态-立体关联策略，以极低计算成本实现稀疏深度估计。  
◆ 将深度估计与紧耦合的束调整优化方案相结合，提升了位姿估计的精度和鲁棒性。  
◆ 利用基于体素的事件表示和循环神经网络，实现了精确的光流估计和可靠的图像块关联。  
◆ 系统能够实时处理VGA分辨率的事件流，相比现有基于事件的VO方法实现了显著的速度提升。  
实验表明，该系统在多个真实场景数据集及夜间高动态范围条件下均表现出优越性能和稳定性。</td></tr>
<tr><td>2025-09-09</td><td>Zero-Shot Metric Depth Estimation via Monocular Visual-Inertial Rescaling for Autonomous Aerial Navigation</td><td>[2509.08159](http://arxiv.org/pdf/2509.08159)</td><td>本文提出了一种结合单目RGB图像和IMU进行零样本度量深度估计的轻量化方法，用于自主无人机导航。
◆ 利用视觉-惯性导航系统生成的稀疏3D特征图，将单目相对深度估计转换为度量深度，无需数据微调或重型传感器。
◆ 提出多种零射度量深度重缩放策略，包括基于单调样条拟合的方法，显著提升跨域适应性。
◆ 在仿真环境中系统评估不同策略的精度，优选方案在真实计算受限的四旋翼无人机上实现15Hz实时度量深度输出。
◆ 成功将估计结果与基于运动基元的规划器集成，实证验证了在自主飞行中的有效避障能力。</td></tr>
<tr><td>2025-09-09</td><td>MCTED: A Machine-Learning-Ready Dataset for Digital Elevation Model Generation From Mars Imagery</td><td>[2509.08027](http://arxiv.org/pdf/2509.08027)</td><td>本文提出了首个专为机器学习设计的火星数字高程模型（DEM）生成数据集MCTED。
◆ 构建了一个大规模、高质量且可直接用于机器学习训练的数据集，包含80,898个由高分辨率火星正射影像和对应DEM块组成的样本。
◆ 开发了专门的数据处理工具和流程，有效修复或减轻了原始DEM数据中常见的伪影和缺失值问题。
◆ 数据集提供了影像块、DEM块以及两个掩码块，清晰标注了原始缺失和人工修改过的区域，为用户提供了高度的数据透明度和灵活性。
◆ 在数据划分上确保了训练集与验证集覆盖不同的地理区域，严格避免了数据泄露风险。
◆ 通过实验证明，即使是一个小型U-Net模型，在此数据集上训练后的性能也优于DepthAnythingV2等零样本基础模型，验证了数据集的有效性。</td></tr>
<tr><td>2025-09-08</td><td>Event Spectroscopy: Event-based Multispectral and Depth Sensing using Structured Light</td><td>[2509.06741](http://arxiv.org/pdf/2509.06741)</td><td>该论文提出了一种新型事件光谱系统，用于无人机在复杂森林环境中的感知与数据采集。  
◆ 首次将事件相机与结构化光结合，实现了基于事件的多光谱成像与深度感知一体化。  
◆ 通过调制投影结构光的波长，在650–850 nm波段内同步获取高分辨率光谱信息与深度数据。  
◆ 在深度重建方面，相比商用深度传感器显著降低约60%的RMSE误差，且光谱精度与专业光谱仪及多光谱相机相当。  
◆ 系统在真实雨林环境中验证了RGB重建与材料区分能力，结合深度数据后材料分类准确率比纯颜色方法提升30%以上。  
◆ 提供了一种轻量化、低延迟、抗环境光干扰的解决方案，为无人机在自然场景中的鲁棒感知开辟了新途径。</td></tr>
<tr><td>2025-09-10</td><td>VIM-GS: Visual-Inertial Monocular Gaussian Splatting via Object-level Guidance in Large Scenes</td><td>[2509.06685](http://arxiv.org/pdf/2509.06685)</td><td>VIM-GS是一种用于大场景新颖视图合成的视觉-惯性单目高斯溅射框架。其核心贡献是解决了单目图像因缺乏准确深度信息而导致渲染质量差的问题。  
◆ 提出了一种新颖的深度生成框架，巧妙地融合了来自视觉-惯性SLAM的稀疏但精确的深度，与来自大型基础模型的稠密但粗糙的深度。  
◆ 设计了一种基于对象分割的深度传播算法，通过渲染结构化物体的像素深度，有效弥合了稀疏输入与稠密输出之间的鸿沟。  
◆ 开发了一个动态深度优化模块，专门处理动态物体导致的SLAM深度残缺问题，并进一步优化来自基础模型的粗糙深度。  
◆ 整个方法仅需单目图像和惯性测量单元(IMU)数据，即可生成高质量深度图以引导高斯溅射，实现了在大场景中的高清渲染。  
实验结果表明，该方法在公开和定制数据集上均优于现有方法。</td></tr>
<tr><td>2025-09-07</td><td>S-LAM3D: Segmentation-Guided Monocular 3D Object Detection via Feature Space Fusion</td><td>[2509.05999](http://arxiv.org/pdf/2509.05999)</td><td>该论文提出了一种融合分割先验的单目3D目标检测新方法S-LAM3D。  
◆ 采用解耦策略注入预计算的分割信息先验，直接与特征空间融合以指导检测。  
◆ 无需扩展检测模型或联合学习先验，避免了增加预测分支的复杂度。  
◆ 重点评估了分割信息对现有检测管道的增强效果，无需额外传感器或训练数据。  
在KITTI基准测试中，该方法在行人、骑行者等小物体检测上显著优于仅依赖RGB特征的模型。  
证明了通过理解输入数据本身可弥补单目深度缺失的固有挑战。</td></tr>
<tr><td>2025-09-06</td><td>Stereovision Image Processing for Planetary Navigation Maps with Semi-Global Matching and Superpixel Segmentation</td><td>[2509.05645](http://arxiv.org/pdf/2509.05645)</td><td>本文针对火星探测中立体视觉匹配的难题，提出了一种结合半全局匹配与超像素分割的地形建模新方法。  
◆ 采用半全局匹配（SGM）替代传统局部块匹配，通过多路径一维优化聚合全局信息，显著提升低纹理区域和遮挡区域的匹配鲁棒性。  
◆ 引入超像素分割作为后处理步骤，利用图像结构信息对初始视差图进行优化，有效减少块效应并恢复细节，增强场景结构一致性。  
◆ 在多个数据集（包括火星模拟环境）上验证了方法的有效性，结果表明其在斜坡、岩石遮挡等复杂区域能生成更连贯、更精确的深度图。  
◆ 所提出的完整地形建模流程，从特征匹配到导航图生成，具备较高的实用性和集成可行性，更适合未来行星探测任务中的自主导航需求。</td></tr>
<tr><td>2025-09-06</td><td>MonoGlass3D: Monocular 3D Glass Detection with Plane Regression and Adaptive Feature Fusion</td><td>[2509.05599](http://arxiv.org/pdf/2509.05599)</td><td>该论文的核心贡献是提出了首个针对单目3D玻璃检测的专用数据集和方法MonoGlass3D，解决了玻璃表面因透明特性难以感知的难题。  
◆ 创建了一个包含多种真实场景和精确3D标注的玻璃数据集，填补了该领域真实数据匮乏的空白。  
◆ 设计了自适应特征融合模块，使网络能有效适应玻璃外观模糊和上下文多样性的挑战，增强了对环境信息的捕捉能力。  
◆ 提出了平面回归管道，显式利用玻璃的平面几何先验，将几何属性无缝集成到检测框架中。  
实验表明，该方法在玻璃分割和单目深度估计任务上均优于现有技术，证明了结合几何与上下文信息对透明表面理解的有效性。</td></tr>
<tr><td>2025-09-05</td><td>FloodVision: Urban Flood Depth Estimation Using Foundation Vision-Language Models and Domain Knowledge Graph</td><td>[2509.04772](http://arxiv.org/pdf/2509.04772)</td><td>本文提出了FloodVision框架，用于零样本城市洪水深度估计。其核心贡献在于结合基础视觉-语言模型与领域知识图谱，显著提升了泛化能力和精度。  
◆创新性地利用GPT-4o进行语义推理，动态识别图像中的参考物体（如车辆、行人等），避免依赖固定检测器。  
◆引入结构化领域知识图谱，编码常见城市物体的真实物理尺寸，有效减少模型幻觉并增强现实 grounding。  
◆通过淹没比例估计和统计离群值过滤计算水深，提升了测量的鲁棒性和准确性。  
在110张真实图像上验证，平均绝对误差降至8.17厘米，较基线提升20.5%，且具备近实时处理能力，适用于智慧城市应急应用。</td></tr>
<tr><td>2025-09-03</td><td>Uncertainty-aware Test-Time Training (UT$^3$) for Efficient On-the-fly Domain Adaptive Dense Regression</td><td>[2509.03012](http://arxiv.org/pdf/2509.03012)</td><td>该论文提出了一种不确定性感知的测试时训练框架UT³，用于解决密集回归任务中的实时域自适应问题。其核心贡献与创新点如下：
◆ 提出不确定性感知的自监督任务，利用量化不确定性选择性地触发测试时训练，大幅减少计算开销。
◆ 显著降低推理时间，通过选择性训练避免对每个样本都进行多次前向和反向传播，更适合资源受限的硬件。
◆ 提供可调节的关键帧选择机制，允许用户根据实际需求灵活控制测试时训练的触发频率。
◆ 在保持与标准测试时训练相当性能的同时，实现了高效持续的域自适应能力。
◆ 在单目深度估计任务上验证了方法的有效性，平衡了精度与效率的需求。</td></tr>
<tr><td>2025-09-03</td><td>DUViN: Diffusion-Based Underwater Visual Navigation via Knowledge-Transferred Depth Features</td><td>[2509.02983](http://arxiv.org/pdf/2509.02983)</td><td>该论文提出了一种基于扩散模型的水下视觉导航方法DUViN，用于未知环境中水下机器人的端到端4自由度运动控制。  
◆提出首个基于扩散模型的水下视觉导航策略，实现无需预建地图的障碍规避与地形相对高度保持。  
◆设计了一种新颖的跨领域迁移学习框架，通过两阶段训练解决水下导航数据稀缺问题：先在空气中数据集训练扩散策略，再迁移至水下领域。  
◆引入知识迁移的深度特征提取器，先使用预训练深度特征进行初始策略训练，再通过水下深度估计任务微调特征提取器以增强域适应能力。  
◆实验证明该方法在模拟和真实水下环境中均具有强泛化性能，有效应对从空气到水下的域偏移挑战。</td></tr>
<tr><td>2025-09-02</td><td>Decoupling Bidirectional Geometric Representations of 4D cost volume with 2D convolution</td><td>[2509.02415](http://arxiv.org/pdf/2509.02415)</td><td>该论文的核心贡献是提出了一种基于纯2D卷积的实时立体匹配网络DBStereo，通过解耦4D代价体积的几何表示实现了高效且精确的匹配。  
◆ 创新性地分析了4D代价体积的解耦特性，并提出双向几何聚合模块分别学习空间和视差表示。  
◆ 完全采用2D卷积进行正则化，避免了计算密集的3D卷积，显著提升了移动端的部署友好性。  
◆ 通过解耦学习机制，在保持高精度的同时实现了实时推理性能。  
实验表明，DBStereo在速度和精度上均优于现有基于聚合的方法，甚至超越了迭代式方法IGEV-Stereo，为立体匹配提供了新的强基线范式。</td></tr>
<tr><td>2025-09-02</td><td>Physics-Informed Machine Learning with Adaptive Grids for Optical Microrobot Depth Estimation</td><td>[2509.02343](http://arxiv.org/pdf/2509.02343)</td><td>该论文提出了一种用于光学微机器人深度估计的物理信息机器学习框架，其核心贡献与创新点如下：
◆ 将物理原理与机器学习相结合，利用基于物理的聚焦度量（如熵、高斯拉普拉斯和梯度锐度）来增强卷积特征提取，解决了透明微机器人和低对比度成像带来的挑战。
◆ 引入自适应网格策略，在微机器人区域分配更精细的网格，在背景区域使用更粗糙的网格，从而在增强深度感知灵敏度的同时显著降低了计算复杂度。
◆ 该框架展现出极高的数据效率，仅使用20%的标注数据进行训练，其性能即可超越在完整数据集上训练的ResNet50等基线模型，降低了对昂贵标注数据的依赖。
◆ 在多个微机器人类型上进行了验证，结果表明该方法将均方误差（MSE）降低了超过60%，并在所有测试案例中提高了决定系数（R²），证明了其卓越的精度和鲁棒性。</td></tr>
<tr><td>2025-09-02</td><td>Doctoral Thesis: Geometric Deep Learning For Camera Pose Prediction, Registration, Depth Estimation, and 3D Reconstruction</td><td>[2509.01873](http://arxiv.org/pdf/2509.01873)</td><td>该论文的核心贡献在于提出了一种结合几何先验与深度学习的几何深度学习框架，以解决三维视觉中的关键任务。  
◆ 将深度信息、表面法线和等变性约束等几何先验集成到深度学习模型中，提升了相机位姿估计的精度与鲁棒性。  
◆ 开发了针对非结构化环境的点云配准方法，克服了传统SLAM/SfM在特征模糊场景中的局限性。  
◆ 提出了高保真三维重建技术，能够生成细节丰富且适用于语义分析与渲染的几何表示。  
◆ 系统性地验证了所提方法在文化遗产数字化和VR/AR等实际应用中的有效性。  
通过融合传统几何方法与深度学习，该研究为高维三维数据缺乏标注的问题提供了解决方案。</td></tr>
<tr><td>2025-09-01</td><td>Generalizable Self-supervised Monocular Depth Estimation with Mixture of Low-Rank Experts for Diverse Endoscopic Scenes</td><td>[2509.01206](http://arxiv.org/pdf/2509.01206)</td><td>该论文针对内窥镜场景中光照和组织的多样性，提出了一种自监督单目深度估计框架，以提升模型的泛化能力。其核心创新点包括：
◆提出了一种新颖的分块动态低秩专家混合模块，能够根据输入特征自适应选择并加权不同的专家进行推理，高效微调基础模型。
◆通过按训练质量分配专家，以少量可训练参数有效处理内窥镜中不同组织的多样化特征。
◆设计了一种自监督训练框架，联合处理内窥镜场景中亮度和反射率不一致的问题，提升了训练的鲁棒性。
该方法在真实和模拟内窥镜数据集上均优于现有技术，并在多样场景的零样本深度估计中取得了最佳泛化性能，为微创测量和手术提供了精确的三维感知支持。</td></tr>
<tr><td>2025-08-31</td><td>ER-LoRA: Effective-Rank Guided Adaptation for Weather-Generalized Depth Estimation</td><td>[2509.00665](http://arxiv.org/pdf/2509.00665)</td><td>该论文提出了ER-LoRA方法，用于解决恶劣天气下单目深度估计的泛化问题。其核心贡献在于仅使用少量正常天气数据对视觉基础模型（VFM）进行参数高效微调，实现了跨天气的高精度深度估计。

◆ 提出了Selecting-Tuning-Maintaining (STM)策略，从结构上分解VFM的预训练权重。
◆ 引入两种有效秩（熵秩和稳定秩）作为指导，平衡任务适应与预训练知识保留。
◆ 在调优阶段，根据熵秩和全微调权重自适应选择秩数与任务感知奇异方向进行初始化。
◆ 在维护阶段，基于稳定秩施加主方向正则化，有效保留基础模型的强泛化能力。
该方法在四个真实恶劣天气基准测试上性能优异，超越了现有PEFT方法、全微调以及使用合成数据训练的模型。</td></tr>
<tr><td>2025-08-30</td><td>Autonomous Aggregate Sorting in Construction and Mining via Computer Vision-Aided Robotic Arm Systems</td><td>[2509.00339](http://arxiv.org/pdf/2509.00339)</td><td>本文提出了一种用于建筑和矿业中骨料自动分拣的计算机视觉辅助机器人系统，以解决传统方法精度低、灵活性差和适应性不足的问题。其核心贡献与创新点包括：

◆ 开发了一套集成六自由度机械臂、双目立体相机和ROS控制框架的完整自主分拣系统。
◆ 采用注意力增强的YOLOv8模型进行骨料检测，提升了识别准确性和鲁棒性。
◆ 结合立体匹配技术实现高精度三维定位，并通过手眼校准确保坐标系统一。
◆ 应用Denavit-Hartenberg运动学建模控制机械臂运动，并利用最小外接矩形进行尺寸估计。
◆ 在实验中实现对四类骨料平均97.5%的分拣成功率，展现出优异的工程应用潜力。</td></tr>
<tr><td>2025-08-28</td><td>Enhancing Pseudo-Boxes via Data-Level LiDAR-Camera Fusion for Unsupervised 3D Object Detection</td><td>[2508.20530](http://arxiv.org/pdf/2508.20530)</td><td>本文提出了一种数据级LiDAR-相机融合框架，用于无监督3D目标检测，核心贡献在于显著提升了伪标签质量。  
◆ 提出数据级双向融合机制，将2D图像实例分割和深度估计信息与3D点云深度融合，实现点云类别标注和点云密度增强。  
◆ 设计局部与全局滤波方法，通过局部半径滤波抑制深度估计噪声，通过全局统计滤波去除分割异常值，有效提升数据质量。  
◆ 引入基于数据级融合的动态自进化策略，通过密集表示迭代优化伪边界框，显著提高定位精度。  
在nuScenes数据集上的实验表明，该方法达到28.4% mAP，大幅优于现有无监督方法。</td></tr>
<tr><td>2025-08-27</td><td>OpenM3D: Open Vocabulary Multi-view Indoor 3D Object Detection without Human Annotations</td><td>[2508.20063](http://arxiv.org/pdf/2508.20063)</td><td>OpenM3D是一种无需人工标注、仅使用多视角图像进行开放词汇室内3D物体检测的新方法。其核心贡献在于首次实现了基于图像的单阶段开放词汇3D检测，并在训练速度和精度上均超越了现有方法。  
◆提出了一种基于图嵌入技术的3D伪标签生成方法，能将2D图像分割结果融合为高质量且连贯的3D结构，其伪框的精确度和召回率优于之前的最佳方法。  
◆设计了一种联合训练目标，结合了无需类别信息的3D定位损失和基于CLIP特征的多视角体素-语义对齐损失，有效实现了开放词汇能力。  
◆构建了一个高效的单阶段检测器，仅需多视角RGB图像作为输入，无需深度图估计器或复杂的双阶段流程，在ScanNet200和ARKitScenes基准上实现了领先的检测精度（每场景仅需0.3秒）。  
◆整个系统在完全无需人工3D标注的条件下完成训练，为开放词汇3D检测提供了一种全新的、强有效的解决方案。</td></tr>
<tr><td>2025-08-26</td><td>SoccerNet 2025 Challenges Results</td><td>[2508.19182](http://arxiv.org/pdf/2508.19182)</td><td>该论文总结了第五届SoccerNet足球视频理解基准挑战赛的核心成果，聚焦计算机视觉技术在足球领域的创新应用。  
◆ 设立四项全新视觉任务：团队持球动作检测、单目深度估计、多视角犯规识别及比赛状态重建，全面覆盖足球视频分析的关键维度。  
◆ 提供大规模标注数据集与统一评估协议，为不同任务提供标准化研究基础与可比性保障。  
◆ 开发并公开强性能基线模型，显著降低研究门槛并促进方法复现与比较。  
◆ 通过多视角融合分析（如犯规识别）和跨模态定位（如球员拓扑重建），推动复杂场景下的算法鲁棒性提升。  
赛事成果体现了计算机视觉在体育领域的应用进展，持续推动开源、可复现研究的国际合作生态。</td></tr>
<tr><td>2025-08-25</td><td>Impact of Target and Tool Visualization on Depth Perception and Usability in Optical See-Through AR</td><td>[2508.18481](http://arxiv.org/pdf/2508.18481)</td><td>该论文系统评估了光学透视式增强现实（OST-AR）中目标物透明度和工具可视化方式对深度感知和可用性的影响。  
◆ 首次通过实验对比了高透明度与低透明度目标渲染在手臂距离深度匹配任务中的性能差异。  
◆ 创新性地设计了六种可视化条件（2种透明度 × 3种工具模式），结合虚拟代理、真实工具和无跟踪三种工具呈现方式。  
◆ 发现低透明度目标能显著降低深度估计误差，纠正了高透明度可能改善体验的假设。  
◆ 证明真实工具遮挡虚拟目标（正确遮挡线索）能同时实现最高操作精度、最佳可用性和最低认知负荷。  
◆ 明确指出实时工具跟踪与遮挡处理是深度感知关键，为AR系统设计提供了优先采用不透明渲染与真实遮挡的实证依据。</td></tr>
<tr><td>2025-08-25</td><td>EndoUFM: Utilizing Foundation Models for Monocular depth estimation of endoscopic images</td><td>[2508.17916](http://arxiv.org/pdf/2508.17916)</td><td>该论文提出了EndoUFM框架，用于解决内窥镜图像单目深度估计中因复杂纹理和光照变化导致的性能受限问题。其核心贡献在于创新性地融合双基础模型并引入多项技术改进以提升模型性能与适应性。  
◆ 首次将双基础模型集成于内窥镜深度估计框架，利用预训练先验知识增强模型泛化能力。  
◆ 提出基于随机向量低秩自适应（RVLoRA）的自适应微调策略，显著提升模型对手术场景的域适应能力。  
◆ 设计基于深度可分离卷积的残差块（Res-DSC），强化对局部细微特征的捕捉效率。  
◆ 引入掩码引导平滑性损失函数，确保解剖组织内部深度的一致性。  
实验表明，该方法在多个数据集上达到最优性能，同时保持较小模型规模，有望增强手术中的空间感知能力。</td></tr>
<tr><td>2025-08-23</td><td>Balanced Sharpness-Aware Minimization for Imbalanced Regression</td><td>[2508.16973](http://arxiv.org/pdf/2508.16973)</td><td>该论文针对回归任务中数据分布不平衡的问题，提出了一种平衡锐度感知最小化方法（BSAM），其核心贡献与创新点如下：

◆ 将不平衡回归问题重新定义为不平衡泛化问题，从泛化能力的角度分析模型性能差异。  
◆ 提出通过损失锐度来衡量模型在不同目标值观测空间中的泛化能力，关注参数扰动对各类样本损失变化的影响。  
◆ 设计了一种新颖的目标重加权策略，使模型在整个观测空间中具有均匀的泛化能力，而非直接调整样本权重。  
◆ 从理论角度证明了所提方法能够提供更好的泛化误差上界保证，增强了方法的理论可靠性。  
◆ 在年龄估计、深度估计等多个视觉回归任务上进行了广泛实验，验证了BSAM相比现有方法的持续优越性。</td></tr>
<tr><td>2025-08-20</td><td>FOCUS: Frequency-Optimized Conditioning of DiffUSion Models for mitigating catastrophic forgetting during Test-Time Adaptation</td><td>[2508.14437](http://arxiv.org/pdf/2508.14437)</td><td>该论文提出FOCUS方法，通过频率优化的扩散模型条件化机制解决测试时适应中的灾难性遗忘问题。  
◆创新性地利用空间自适应频率先验，在扩散去噪过程中通过条件化反向步骤保留任务相关的语义信息。  
◆设计了轻量级Y形频率预测网络（Y-FPN），有效解耦噪声图像中的高低频信息以降低计算成本。  
◆提出FrequencyMix数据增强方法，通过扰动多频带图像提升模型对多种损坏类型的鲁棒性。  
实验表明其在15种损坏类型和三个数据集上的语义分割与深度估计任务中达到最优平均性能。  
◆能够生成伪标签辅助现有模型适应方法，在有限监督下有效缓解灾难性遗忘问题。</td></tr>
<tr><td>2025-08-19</td><td>ROVR-Open-Dataset: A Large-Scale Depth Dataset for Autonomous Driving</td><td>[2508.13977](http://arxiv.org/pdf/2508.13977)</td><td>该论文的核心贡献是推出了一个面向自动驾驶的大规模、多样化深度估计数据集ROVR-Open-Dataset，以解决现有数据集在多样性和可扩展性上的不足。
◆ 构建了大规模且帧间连续的深度数据集，包含2万帧动态户外驾驶场景图像，支持模型训练与评估。
◆ 采用轻量化数据采集方案，以低成本实现了广泛的场景覆盖，提升了数据集的经济性和可扩展性。
◆ 提供稀疏但统计上足够的地面真值，在保证训练鲁棒性的同时降低了标注成本。
◆ 在驾驶场景多样性上显著优于现有数据集，且具有更低的深度密度，对模型的泛化能力提出了更高挑战。
通过主流单目深度估计模型的基准实验，验证了数据集的实用价值，并揭示了现有方法在复杂条件下的性能缺陷，为深度估计研究提供了新的评估平台。</td></tr>
<tr><td>2025-08-18</td><td>Batching-Aware Joint Model Onloading and Offloading for Hierarchical Multi-Task Inference</td><td>[2508.13380](http://arxiv.org/pdf/2508.13380)</td><td>本文针对资源受限的边缘计算场景，提出了一个联合优化多任务模型部署和查询路由的框架。其核心创新点包括：  
◆ 首次在分层计算架构（终端-边缘-云）中统一解决了多任务模型的联合部署（onloading）和推理卸载（offloading）问题。  
◆ 将问题构建为一个混合整数规划，并设计了高效的J3O交替优化算法，结合了拉格朗日松弛次模优化与约束线性规划。  
◆ 创新性地考虑了边缘服务器上的批处理（batching）效应，显著提升了系统在异构任务负载下的可扩展性。  
实验表明，该方案能以不到15%的最优求解器运行时间，达到超过97%的最优推理精度。</td></tr>
<tr><td>2025-08-18</td><td>DMS:Diffusion-Based Multi-Baseline Stereo Generation for Improving Self-Supervised Depth Estimation</td><td>[2508.13091](http://arxiv.org/pdf/2508.13091)</td><td>◆ 提出DMS方法，利用扩散模型生成多基线立体视图，解决自监督深度估计中因遮挡和出框区域导致的像素对应缺失问题。  
◆ 通过微调Stable Diffusion模型，沿极线方向合成关键位置的新视角（左-左视图、右-右视图及中间视图），补充遮挡像素以建立显式光度对应关系。  
◆ 方法具有模型无关性，无需额外标注数据，仅需未标注的立体图像对即可完成训练和合成。  
◆ 设计为“即插即用”的免费增强模块，可无缝提升现有自监督立体匹配和单目深度估计模型的性能。  
◆ 在多个基准数据集上验证了有效性，异常值减少高达35%，达到当前最优性能。</td></tr>
<tr><td>2025-08-15</td><td>Towards Understanding 3D Vision: the Role of Gaussian Curvature</td><td>[2508.11825](http://arxiv.org/pdf/2508.11825)</td><td>◆ 提出高斯曲率在3D视觉中的关键作用，弥补了当前数据驱动方法缺乏显式几何模型的缺陷。  
◆ 证明高斯曲率具有观测不变性，可作为稀疏紧凑的3D表面描述符，提升建模效率。  
◆ 发现现有单目/立体视觉方法虽隐含利用高斯曲率，但无法提取显式模块，揭示了理论空白。  
◆ 将高斯曲率发展为几何先验，能主动指导并优化3D表面重建任务。  
◆ 提出高斯曲率作为立体视觉的无监督评估指标，为方法性能分析提供新工具。  
◆ 基于Middlebury数据集的实验验证了高斯曲率在多任务中的普适价值，推动可解释3D视觉研究。</td></tr>
<tr><td>2025-08-15</td><td>DashCam Video: A complementary low-cost data stream for on-demand forest-infrastructure system monitoring</td><td>[2508.11591](http://arxiv.org/pdf/2508.11591)</td><td>◆ 提出了一种低成本、可复制的框架，利用车载摄像头（行车记录仪）视频数据进行实时、物体级别的路边植被和基础设施结构评估与地理定位。  
◆ 开发了端到端的处理流程，结合单目深度估计、深度误差校正和几何三角测量，从街景视频流中生成精确的空间和结构数据。  
◆ 采用梯度提升回归框架校正单目深度模型的低估问题（尤其在远距离物体上），显著提升了深度估计精度（R2=0.92，MAE=0.31）。  
◆ 首次整合单目深度建模、基于GPS三角测量的地理定位和实时结构评估，为消费级视频数据在城市植被与基础设施监测中的应用提供新方法。  
◆ 通过实验验证了不同摄像头位置和车速下的性能，最佳配置（低速+车内摄像头）下地理定位误差均值2.83米，树木高度估计MAE为2.09米，杆状物0.88米。  
◆ 为电力公司和城市规划者提供了一种补充传统遥感（如LiDAR）的实时、低成本解决方案，适用于动态城市环境中可扩展的频繁评估需求。</td></tr>
<tr><td>2025-08-15</td><td>Unifying Scale-Aware Depth Prediction and Perceptual Priors for Monocular Endoscope Pose Estimation and Tissue Reconstruction</td><td>[2508.11282](http://arxiv.org/pdf/2508.11282)</td><td>◆提出统一框架，整合尺度感知深度预测与时序约束的感知优化，解决单目内窥镜姿态估计和组织重建中的深度模糊、组织形变等挑战。  
◆创新设计MAPIS-Depth模块，结合Depth Pro初始化与Depth Anything的逐帧深度预测，通过L-BFGS-B优化生成伪度量深度估计。  
◆引入基于RAFT的光流像素对应与时序融合策略，利用LPIPS感知相似性自适应加权帧间变形，有效减少组织形变和运动导致的伪影。  
◆开发WEMA-RTDL模块，通过联合优化旋转和平移参数，实现合成伪RGBD帧的精准配准，提升位姿估计精度。  
◆采用截断符号距离函数（TSDF）体素融合与移动立方体算法，生成完整3D组织表面网格，支持临床导航需求。  
◆在HEVD和SCARED数据集上验证，消融实验与对比分析表明框架鲁棒性优于现有方法，尤其在动态组织场景中表现突出。</td></tr>
<tr><td>2025-08-15</td><td>CHARM3R: Towards Unseen Camera Height Robust Monocular 3D Detector</td><td>[2508.11185](http://arxiv.org/pdf/2508.11185)</td><td>◆ 首次系统分析了相机高度变化对单目3D检测模型性能的影响，发现深度估计是主要影响因素。  
◆ 通过数学证明和实验验证，揭示了回归深度模型和基于地面的深度模型在相机高度变化下的误差趋势相反。  
◆ 提出CHARM3R模型，创新性地融合两种深度估计结果（回归深度与地面深度），显著提升模型鲁棒性。  
◆ 在CARLA数据集上，CHARM3R对未见相机高度的泛化能力提升超过45%，达到当前最优性能。  
◆ 开源代码和模型，为后续研究提供重要基准。</td></tr>
<tr><td>2025-08-12</td><td>Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction</td><td>[2508.10936](http://arxiv.org/pdf/2508.10936)</td><td>◆ 提出首个基于稀疏3D语义高斯泼溅的协作3D语义占据预测方法，突破传统密集体素或2D特征方案的局限性。  
◆ 采用邻域跨智能体融合机制，有效消除重复高斯基元并抑制噪声/不一致数据，提升协作感知鲁棒性。  
◆ 在单个高斯基元中联合编码几何与语义信息，减少对深度监督的依赖，仅需简单刚性对齐即可实现高效融合。  
◆ 设计稀疏且以物体为中心的消息传递机制，在保留结构信息的同时显著降低通信开销（仅需34.6%通信量）。  
◆ 实验验证性能优势：相比单智能体与基线协作方法，mIoU分别提升8.42和3.28点，IoU提升5.11和22.41点，在低通信预算下仍保持性能领先。</td></tr>
<tr><td>2025-08-14</td><td>Self-Supervised Stereo Matching with Multi-Baseline Contrastive Learning</td><td>[2508.10838](http://arxiv.org/pdf/2508.10838)</td><td>◆ 提出BaCon-Stereo框架，通过多基线对比学习解决自监督立体匹配中遮挡区域的难题，突破传统光度一致性假设的局限性。  
◆ 采用师生学习范式，师生网络共享参考视图但目标视图不同，利用教师网络在遮挡区域的可见性优势生成更可靠的监督信号。  
◆ 引入基线缩放技术，将教师网络的预测结果调整至学生网络基线尺度，实现跨基线知识迁移。  
◆ 设计遮挡感知注意力图，动态指导学生网络学习遮挡补全，提升遮挡区域预测精度。  
◆ 构建合成数据集BaCon-20k支持多基线训练，实验证明该方法在KITTI基准上超越现有自监督方法，兼具强泛化性与鲁棒性。</td></tr>
<tr><td>2025-08-14</td><td>SC-Lane: Slope-aware and Consistent Road Height Estimation Framework for 3D Lane Detection</td><td>[2508.10411](http://arxiv.org/pdf/2508.10411)</td><td>◆ SC-Lane提出了一种新颖的坡度感知与时序一致的高度图估计框架，用于3D车道线检测，解决了传统方法依赖固定坡度锚点的问题。  
◆ 创新性地设计了坡度自适应特征模块（Slope-Aware Adaptive Feature），通过动态预测权重融合多坡度特征，提升了对多样化道路几何的鲁棒性。  
◆ 引入高度一致性模块（Height Consistency Module），通过强制时序连贯性，确保连续帧间高度估计的稳定性，适用于实际驾驶场景。  
◆ 首次将MAE、RMSE和阈值精度等标准指标系统化应用于道路高度评估，填补了该领域评估方法的空白。  
◆ 在OpenLane基准测试中取得显著突破，F-score达64.3%，大幅超越现有方法，确立了3D车道检测的新标杆。  
◆ 基于LiDAR高度图数据集的实验验证了框架有效性，为后续研究提供了严谨的对比基准。</td></tr>
<tr><td>2025-08-14</td><td>Iterative Volume Fusion for Asymmetric Stereo Matching</td><td>[2508.09543](http://arxiv.org/pdf/2508.09543)</td><td>◆ 针对非对称双目视觉系统（如长焦-广角相机）中的立体匹配难题，首次系统分析了两种经典代价体积构建方法在视觉不对称场景下的匹配代价分布特性。  
◆ 发现不同代价体积会遭受不同类型的信息失真，提出必须综合利用两种体积才能有效解决非对称匹配问题，突破了传统对称匹配的思维局限。  
◆ 提出两阶段迭代体积融合网络IVF-AStereo：第一阶段通过聚合拼接体积优化相关性体积，第二阶段融合两种体积以增强细节恢复能力。  
◆ 该方法专门针对分辨率差异和色彩退化等视觉不对称问题设计，在极端不对称条件下仍保持强鲁棒性。  
◆ 在基准数据集上的大量对比实验和消融研究验证了该方法的优越性，为非对称立体匹配提供了新解决方案。</td></tr>
<tr><td>2025-08-12</td><td>A new dataset and comparison for multi-camera frame synthesis</td><td>[2508.09068](http://arxiv.org/pdf/2508.09068)</td><td>◆ 提出了一种新型多相机密集线性阵列数据集，填补了现有数据集中在时间插值和视角合成任务之间的比较空白。  
◆ 首次实现了视角合成方法（3D高斯泼溅）与经典/深度学习帧插值方法的公平对比，打破了传统评估的领域壁垒。  
◆ 通过真实场景实验发现：深度学习帧插值方法相比经典方法优势有限，而3D高斯泼溅表现反而落后帧插值方法达3.5 dB PSNR。  
◆ 在合成场景中得出相反结论：3D高斯泼溅以95%置信度显著优于帧插值方法近5 dB PSNR，揭示了算法性能对数据类型的强依赖性。  
◆ 研究结果挑战了&quot;深度学习必然优于传统方法&quot;的固有认知，为跨任务方法选择提供了实证依据。  
◆ 所建数据集特别设计了时空对称的采样结构，为统一评估时空插值任务建立了标准化基准。</td></tr>
<tr><td>2025-08-12</td><td>A Robust Epipolar-Domain Regularization Algorithm for Light Field Depth Estimation</td><td>[2508.08900](http://arxiv.org/pdf/2508.08900)</td><td>◆ 提出了一种轻量级光场深度估计框架，结合光场视差信息与定向随机游走优化算法，显著降低计算复杂度。  
◆ 创新性地采用非深度学习方法，摆脱对大规模训练数据和CNN模型的依赖，提升算法在真实噪声环境中的鲁棒性。  
◆ 通过定向随机游走算法增强深度图一致性，解决了传统方法在无控制条件下性能下降的问题。  
◆ 在4D光场基准数据集和真实场景图像上验证了算法有效性，保持低计算成本的同时达到与先进深度学习模型相当的精度。  
◆ 为光场成像中的深度估计与分割提供了高效可靠的解决方案，并探索了概率图模型与深度感知框架结合的新方向。</td></tr>
<tr><td>2025-08-11</td><td>GRASPTrack: Geometry-Reasoned Association via Segmentation and Projection for Multi-Object Tracking</td><td>[2508.08117](http://arxiv.org/pdf/2508.08117)</td><td>◆ 提出GRASPTrack框架，将单目深度估计和实例分割融入检测跟踪范式，通过2D检测生成高精度3D点云，实现显式3D几何推理，解决传统方法因缺乏几何感知导致的遮挡和深度模糊问题。  
◆ 创新设计基于体素的3D交并比（IoU）空间关联方法，通过点云体素化实现更精准鲁棒的目标空间匹配。  
◆ 提出深度感知自适应噪声补偿机制，根据遮挡程度动态调整卡尔曼滤波器过程噪声，提升遮挡场景下的状态估计可靠性。  
◆ 开发深度增强的观测中心动量模型，将运动方向一致性从图像平面扩展到3D空间，显著改善复杂轨迹目标的运动关联效果。  
◆ 在MOT17、MOT20和DanceTrack基准测试中验证了方法优越性，尤其在遮挡频繁、运动复杂的场景中跟踪鲁棒性提升显著。</td></tr>
<tr><td>2025-08-11</td><td>TRIDE: A Text-assisted Radar-Image weather-aware fusion network for Depth Estimation</td><td>[2508.08038](http://arxiv.org/pdf/2508.08038)</td><td>◆ 提出了一种结合文本生成策略的特征提取与融合方法，首次将语言描述引入单目深度估计任务，在KITTI数据集上提升了不同算法的精度。  
◆ 设计了TRIDE雷达-相机融合算法，通过雷达点云信息增强文本特征提取能力，实现多模态协同优化。  
◆ 创新性地引入天气感知融合模块，根据实时天气条件动态调整雷达权重，解决恶劣天气下传感器性能差异问题。  
◆ 在nuScenes数据集上验证了方法的优越性，MAE和RMSE指标分别比现有最优方法提升12.87%和9.08%。  
◆ 开源了完整代码，为雷达-图像-文本多模态融合研究提供了可复现的基准方案。</td></tr>
<tr><td>2025-08-11</td><td>Autonomous Navigation of Cloud-Controlled Quadcopters in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic Reasoning</td><td>[2508.07885](http://arxiv.org/pdf/2508.07885)</td><td>◆ 提出了一种基于云计算和AI的多模态感知系统，用于解决四旋翼无人机在GPS拒止室内环境中的自主导航问题。  
◆ 设计了定制化PCB板，集成ToF传感器和IMU，实现高效传感器数据采集，提升狭窄空间导航鲁棒性。  
◆ 结合YOLOv11目标检测和Depth Anything V2单目深度估计，通过卡尔曼滤波增强3D空间感知能力。  
◆ 创新性引入云端LLM进行语义推理，实现上下文感知的决策制定，突破传统导航系统的逻辑局限性。  
◆ 采用虚拟安全边界技术（校准传感器偏移量）和多线程架构，在42次试验中仅出现16次边界突破，端到端延迟低于1秒。  
◆ 实验验证系统性能优异：目标检测mAP50达0.6，深度估计MAE仅7.2厘米，为复杂室内场景提供了高智能辅助导航方案。</td></tr>
<tr><td>2025-08-11</td><td>Tracking Any Point Methods for Markerless 3D Tissue Tracking in Endoscopic Stereo Images</td><td>[2508.07851](http://arxiv.org/pdf/2508.07851)</td><td>◆ 提出了一种基于2D Tracking Any Point (TAP)网络的无标记3D组织追踪新方法，首次将TAP技术应用于内窥镜立体图像中的组织运动追踪。  
◆ 创新性地结合了两个CoTracker模型，分别用于时间序列追踪和立体匹配，实现了从立体内窥镜图像中估计3D组织运动。  
◆ 在临床腹腔镜设置和机器人模拟组织运动环境下进行了系统验证，包括合成3D打印模型和真实鸡组织模型，证明了方法的实用性。  
◆ 在鸡组织模型上实现了1.1毫米的低追踪误差（速度10毫米/秒），展示了在真实生物组织上的高精度追踪能力。  
◆ 该方法为复杂手术场景提供了无标记、高精度的3D组织运动追踪解决方案，有望提升手术导航安全性和机器人辅助手术的上下文感知能力。</td></tr>
<tr><td>2025-08-10</td><td>MonoMPC: Monocular Vision Based Navigation with Learned Collision Model and Risk-Aware Model Predictive Control</td><td>[2508.07387](http://arxiv.org/pdf/2508.07387)</td><td>◆ 提出了一种基于单目视觉的导航方法MonoMPC，无需依赖精确的深度估计即可实现未知环境中的避障。  
◆ 创新性地将噪声深度估计作为上下文输入，训练学习型碰撞模型，预测机器人控制序列的最小障碍物间隙分布。  
◆ 设计了风险感知的模型预测控制（MPC）框架，利用碰撞模型的预测结果动态优化路径规划，最小化碰撞风险。  
◆ 开发了联合训练管道，同时优化碰撞模型和风险度量，使用安全与非安全轨迹数据提升模型在复杂环境中的泛化能力。  
◆ 通过联合训练确保碰撞模型具有最优方差特性，显著提高了高密度障碍物环境下的导航成功率。  
◆ 实验证明该方法在真实场景中的成功率比NoMaD和ROS堆栈分别提升9倍和7倍，消融研究验证了各模块的有效性。</td></tr>
<tr><td>2025-08-10</td><td>DIP-GS: Deep Image Prior For Gaussian Splatting Sparse View Recovery</td><td>[2508.07372](http://arxiv.org/pdf/2508.07372)</td><td>◆ 提出DIP-GS方法，将深度图像先验（DIP）与3D高斯泼溅（3DGS）结合，解决稀疏视图重建难题。  
◆ 无需依赖预训练模型（如生成模型或深度估计），仅利用输入帧的内部结构和模式实现重建。  
◆ 采用由粗到细的训练策略，增强稀疏视图下的场景表示能力，显著提升重建质量。  
◆ 在多种稀疏视图重建任务中达到竞争性最优（SOTA）效果，验证了方法的有效性。  
◆ 首次将DIP先验应用于3DGS框架，扩展了3DGS在低重叠、少输入场景下的适用性。  
◆ 方法具有轻量化优势，仅需输入帧即可实现高质量重建，避免了额外数据或模型依赖。</td></tr>
<tr><td>2025-08-10</td><td>Similarity Matters: A Novel Depth-guided Network for Image Restoration and A New Dataset</td><td>[2508.07211](http://arxiv.org/pdf/2508.07211)</td><td>◆提出首个深度引导的图像修复网络（DGN），通过双分支交互架构（深度估计分支+修复分支）解决传统方法忽视深度信息导致的相似性匹配问题。  
◆创新性地将渐进式窗口自注意力（捕获物体内相似性）与稀疏非局部注意力（建模物体间相似性）结合，显著提升修复精度。  
◆实现深度估计与图像修复的协同优化——深度特征指导修复质量，修复后的视觉特征反哺深度估计精度。  
◆构建目前最大规模的高分辨率植物图像数据集（9,205张图/403物种），涵盖丰富深度与纹理变化，填补领域空白。  
◆在标准基准测试中达到SOTA性能，对未见过的植物图像展现强泛化能力，验证了方法的鲁棒性。  
◆首次系统揭示深度信息对浅景深（注意力分散）与深景深（背景过增强）场景修复的关键作用。</td></tr>
<tr><td>2025-08-10</td><td>Acoustic source depth estimation method based on a single hydrophone in Arctic underwater</td><td>[2508.07157](http://arxiv.org/pdf/2508.07157)</td><td>◆ 提出基于单水听器的北极水下声源深度估计方法，结合简正波和射线理论，解决了传统方法需要阵列设备的局限性。  
◆ 针对表面折射简正波波导，利用弯曲变换实现模态分离，通过模态振幅随频率和阶数的变化特性匹配估计声源深度。  
◆ 创新性地利用简正波截止频率特性，提出基于模态截止频率匹配的声源深度估计方法，拓展了简正波理论的应用场景。  
◆ 针对北极深海环境，通过深反转射线轨迹分析获取接收端声线到达结构，利用声线到达时差匹配实现深度估计。  
◆ 通过实验数据验证了不同方法的适用性与局限性，为北极声学探测提供了实用的技术支撑。</td></tr>
<tr><td>2025-08-09</td><td>AugLift: Boosting Generalization in Lifting-based 3D Human Pose Estimation</td><td>[2508.07112](http://arxiv.org/pdf/2508.07112)</td><td>◆ 提出AugLift方法，通过简单改造标准2D-to-3D提升流程，显著提升跨数据集泛化能力，无需额外数据或传感器。  
◆ 创新性地在标准2D关键点坐标(x,y)基础上，稀疏增强两个新信号：关键点检测置信度c和单目深度估计d，形成(x,y,c,d)输入。  
◆ 利用现成预训练模型（如单目深度估计）生成增强信号，继承其强大泛化能力，实现零成本性能提升。  
◆ 模块化设计可无缝集成到现有任何提升架构中，实验证明平均提升跨数据集性能10.1%，内部数据集性能4.0%。  
◆ 分析表明稀疏的关键点对齐线索提供了鲁棒的帧级上下文，为提升基于提升的3D姿态估计泛化性提供实用方案。  
◆ 在四个数据集上的广泛验证显示方法具有架构无关的稳健性，代码将开源促进社区发展。</td></tr>
<tr><td>2025-08-08</td><td>Neural Field Representations of Mobile Computational Photography</td><td>[2508.05907](http://arxiv.org/pdf/2508.05907)</td><td>◆ 提出利用神经场（neural fields）模型紧凑表示移动摄影中的复杂几何与光照效果，突破传统显式数据表示（如像素阵列或点云）的限制。  
◆ 开发无需复杂预处理、标注真值数据或机器学习先验的方法，直接拟合智能手机原始测量数据，实现高效逆问题求解。  
◆ 通过自正则化模型设计，显著提升深度估计、图层分离和图像拼接等任务的性能，超越现有先进技术。  
◆ 结合手机多传感器（如激光测距、陀螺仪等）与神经场，构建轻量化端到端计算摄影框架，拓展移动设备成像能力。  
◆ 验证神经场在真实场景移动摄影数据（in-the-wild）中的实用性，为低功耗设备上的实时计算摄影提供新思路。</td></tr>
<tr><td>2025-08-07</td><td>Propagating Sparse Depth via Depth Foundation Model for Out-of-Distribution Depth Completion</td><td>[2508.04984](http://arxiv.org/pdf/2508.04984)</td><td>◆ 提出了一种新型深度补全框架，利用深度基础模型增强模型在分布外（OOD）场景下的鲁棒性，无需大规模训练。  
◆ 创新性地利用深度基础模型从RGB图像中提取环境线索（如结构和语义上下文），指导稀疏深度信息向缺失区域的传播。  
◆ 设计了无参的双空间传播方法，在3D和2D空间中同步传播稀疏深度，有效保持几何结构和局部一致性。  
◆ 引入可学习的校正模块，逐步调整深度预测值以优化复杂结构的细节。  
◆ 在NYUv2和KITTI数据集上训练，并在16个其他数据集上验证，显著优于现有深度补全方法，尤其在OOD场景中表现突出。</td></tr>
<tr><td>2025-08-08</td><td>Extending Foundational Monocular Depth Estimators to Fisheye Cameras with Calibration Tokens</td><td>[2508.04928](http://arxiv.org/pdf/2508.04928)</td><td>◆ 提出一种无需重新训练或微调的方法，将基于透视图像训练的基础单目深度估计器（FMDEs）直接扩展到鱼眼相机。  
◆ 引入轻量级的“校准令牌”机制，通过调制潜在嵌入空间来对齐鱼眼图像与透视图像的分布差异，解决协变量偏移问题。  
◆ 利用FMDEs已有的高表达能力潜在空间，避免传统方法中图像空间重校准或投影带来的伪影和信息损失。  
◆ 采用自监督学习框架，仅需公开的大规模透视图像数据集，通过模拟鱼眼图像校准实现训练，无需真实鱼眼图像标注。  
◆ 在室内外多场景测试中，仅用一组校准令牌即显著超越现有方法，兼容多种FMDEs模型，展现强泛化性。</td></tr>
<tr><td>2025-08-06</td><td>OmniDepth: Bridging Monocular and Stereo Reasoning with Latent Alignment</td><td>[2508.04611](http://arxiv.org/pdf/2508.04611)</td><td>◆提出OmniDepth统一框架，首次实现单目与立体深度估计的潜在表征双向迭代对齐，突破传统方法割裂使用的局限。  
◆创新性引入跨注意力对齐机制，动态融合单目上下文先验与立体几何假设，在推理过程中实现双向特征同步。  
◆通过单目结构先验有效解决立体匹配在反光/透明表面的固有歧义，同时利用立体几何优化单目深度预测精度。  
◆在单一网络中完成两种模态的协同优化，无需后处理融合，显著提升模型效率与端到端可训练性。  
◆在Middlebury和ETH3D数据集上实现零样本泛化误差降低40%以上，大幅改善透明/反光表面等传统难题。  
◆开源框架为多模态3D感知提供新范式，通过几何约束与语义上下文的深度融合超越单一模态的性能瓶颈。</td></tr>
<tr><td>2025-08-06</td><td>Pseudo Depth Meets Gaussian: A Feed-forward RGB SLAM Baseline</td><td>[2508.04597](http://arxiv.org/pdf/2508.04597)</td><td>◆ 提出了一种基于3D高斯映射的RGB SLAM方法，通过结合深度估计器和3D高斯技术，解决了传统方法在长序列处理中的几何细节不准确问题。  
◆ 引入前馈循环预测模块，直接从光流推断相机位姿，替代了耗时的测试时优化，使跟踪速度提升90%以上。  
◆ 采用局部图渲染技术，增强了前馈位姿预测的鲁棒性，提高了系统在复杂场景中的稳定性。  
◆ 在Replica和TUM-RGBD数据集上的实验表明，该方法性能与当前最优的SplaTAM相当，同时大幅降低了计算开销。  
◆ 通过实际部署验证了方法的实用性，展示了其在实时3D重建中的高效性和可靠性。</td></tr>
<tr><td>2025-08-06</td><td>MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction</td><td>[2508.04297](http://arxiv.org/pdf/2508.04297)</td><td>◆ 提出MuGS方法，首次将多视角立体视觉（MVS）与单目深度估计（MDE）特征融合，增强多基线场景下的泛化重建能力。  
◆ 设计投影-采样深度融合机制，通过精细概率体积构建指导特征图回归，提升深度估计精度。  
◆ 引入参考视图损失函数，优化几何结构并显著提高训练效率。  
◆ 采用3D高斯表示法，在加速训练/推理的同时提升渲染质量，平衡效率与效果。  
◆ 在DTU、RealEstate10K等数据集上实现跨基线（小/大基线）和跨场景（物体/室内外）的SOTA性能。  
◆ 展示零样本泛化能力，在LLFF和Mip-NeRF 360等未训练数据集上表现优异。</td></tr>
<tr><td>2025-08-06</td><td>DET-GS: Depth- and Edge-Aware Regularization for High-Fidelity 3D Gaussian Splatting</td><td>[2508.04099](http://arxiv.org/pdf/2508.04099)</td><td>◆ 提出DET-GS框架，首次将深度与边缘感知正则化统一集成到3D高斯泼溅（3DGS）中，解决稀疏视图下几何重建不精确的问题。  
◆ 设计分层几何深度监督机制，自适应强化多层级几何一致性，显著提升结构保真度并降低深度估计噪声的敏感性。  
◆ 创新性地引入基于Canny边缘检测的语义掩码指导边缘感知深度正则化，有效保护场景边界不被过度平滑。  
◆ 提出RGB引导的边缘保持总变分损失（TV Loss），选择性平滑同质区域的同时严格保留高频细节和纹理。  
◆ 在稀疏视图新视角合成任务上全面超越现有SOTA方法，实验证明其几何精度与视觉保真度均有显著提升。  
（全文共5条创新点，总计约300字）</td></tr>
<tr><td>2025-08-05</td><td>Monocular Depth Estimation with Global-Aware Discretization and Local Context Modeling</td><td>[2508.03186](http://arxiv.org/pdf/2508.03186)</td><td>◆ 提出Gated Large Kernel Attention Module (GLKAM)，通过大核卷积和门控机制有效捕捉多尺度局部结构信息，提升深度估计的局部上下文建模能力。  
◆ 设计Global Bin Prediction Module (GBPM)，预测全局深度区间分布，为深度回归提供结构化指导，增强网络的全局感知能力。  
◆ 结合局部与全局线索的创新框架，解决了单目深度估计中因单视图投影歧义导致的精度瓶颈问题。  
◆ 在NYU-V2和KITTI数据集上验证了方法的有效性，性能优于现有方法，证明了各模块的协同优势。  
◆ 通过门控机制优化大核卷积计算效率，平衡了模型精度与计算开销。</td></tr>
<tr><td>2025-08-04</td><td>VRSight: An AI-Driven Scene Description System to Improve Virtual Reality Accessibility for Blind People</td><td>[2508.02958](http://arxiv.org/pdf/2508.02958)</td><td>◆ 提出VRSight系统，首个无需开发者干预的端到端解决方案，通过AI模型实时解析VR场景并生成空间音频反馈，直接提升盲人用户的VR可访问性。  
◆ 创新性地整合多模态AI技术（物体检测、深度估计、LLM氛围理解），将虚拟场景转化为音调化空间音频，实现非视觉交互。  
◆ 开发DISCOVR专用数据集，包含17款社交VR应用的30类虚拟物体，填补了VR领域缺乏适用训练数据的空白。  
◆ 首次实现主流VR应用（如Rec Room）的&quot;事后可访问化&quot;，用户可直接使用未经改造的商用VR应用。  
◆ 通过9名用户实验验证系统有效性，证明其能支持社交任务（如识别虚拟座位、感知他人虚拟形象）。  
◆ 突破传统依赖开发者集成辅助功能的模式，通过外部系统实现无障碍化，推动行业降低可访问性实施门槛。</td></tr>
<tr><td>2025-08-04</td><td>Elucidating the Role of Feature Normalization in IJEPA</td><td>[2508.02829](http://arxiv.org/pdf/2508.02829)</td><td>◆ 揭示了IJEPA模型中特征归一化（LN）的负面影响：LN破坏了视觉token的自然能量层级，导致高能量token（对应语义重要区域）无法被优先学习。  
◆ 提出LN的替代方案DynTanh激活函数：该方案能保留token能量分布，使高能量token对预测损失贡献更大。  
◆ 解决了LN导致的损失图棋盘格伪影问题：使用DynTanh后损失分布呈现长尾特性，伪影现象消失。  
◆ 显著提升模型性能：在ImageNet线性探测任务中，ViT-Small准确率从38%提升至42.7%；在NYU深度估计任务中RMSE降低0.08。  
◆ 验证了保留token自然能量对自监督视觉表征学习的重要性：为后续研究提供了新的优化方向。</td></tr>
<tr><td>2025-08-04</td><td>Rethinking Transparent Object Grasping: Depth Completion with Monocular Depth Estimation and Instance Mask</td><td>[2508.02507](http://arxiv.org/pdf/2508.02507)</td><td>◆ 提出ReMake框架，通过实例掩膜和单目深度估计引导透明物体深度补全，解决传统RGB-D输入隐含推理的局限性。  
◆ 利用实例掩膜显式区分透明与非透明区域，使模型能针对性学习透明区域的深度估计，减少对隐含推理的依赖。  
◆ 结合单目深度估计提供透明物体与周围环境的深度上下文，提升深度预测的准确性。  
◆ 通过显式监督机制增强模型在真实复杂光照场景中的泛化能力，克服传统方法在真实场景中失效的问题。  
◆ 在基准数据集和真实场景实验中验证了方法的优越性，显著超越现有方法。  
◆ 公开代码和视频，推动透明物体抓取研究的可复现性。</td></tr>
<tr><td>2025-08-05</td><td>3DRot: 3D Rotation Augmentation for RGB-Based 3D Tasks</td><td>[2508.01423](http://arxiv.org/pdf/2508.01423)</td><td>◆ 提出3DRot，一种即插即用的数据增强方法，通过绕相机光心旋转和镜像图像，同步更新RGB图像、相机内参、物体位姿和3D标注，保持投影几何一致性。  
◆ 解决了传统图像变换（如旋转和缩放）破坏几何一致性的问题，无需依赖场景深度即可实现几何一致的旋转和反射。  
◆ 在单目3D检测任务中验证有效性，在SUN RGB-D数据集上将IoU3D从43.21提升至44.51，旋转误差从22.91°降至20.93°，mAP0.5从35.70提升至38.11。  
◆ 相比同类方法Cube R-CNN（需结合多个数据集），3DRot仅使用单一数据集即实现更显著的性能提升，且计算成本更低。  
◆ 由于仅通过相机空间变换实现，3DRot可轻松迁移至其他3D任务（如深度估计、3D关键点估计），具有广泛适用性。</td></tr>
<tr><td>2025-08-02</td><td>Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation</td><td>[2508.01303](http://arxiv.org/pdf/2508.01303)</td><td>这篇论文的核心贡献和创新点如下：

◆ 提出不确定性引导的数据增强方法（UgDA），通过扰动RGB图像的均值和标准差来生成未见过的域样本，解决合成数据训练模型在真实场景中泛化能力差的问题。

◆ 利用基于批次统计的高斯分布建模扰动方向和强度的不确定性，从而模拟更多潜在的域变化，增强模型的域泛化能力。

◆ 强制原始数据和增强数据在特征层面保持一致，促使模型学习结构感知且不受域依赖捷径影响的特征表示。

◆ 该方法简单、与网络架构无关，可无缝集成到任何立体匹配网络中，提升其跨域性能。

◆ 在多个挑战性基准测试上的实验表明，该方法能显著提升现有立体匹配网络的泛化性能。</td></tr>
<tr><td>2025-08-02</td><td>Integrating Disparity Confidence Estimation into Relative Depth Prior-Guided Unsupervised Stereo Matching</td><td>[2508.01275](http://arxiv.org/pdf/2508.01275)</td><td>◆ 提出了一种新颖的无监督立体匹配框架，通过整合视差置信度估计和相对深度先验知识，有效解决了传统方法在多视角一致性假设下的匹配模糊问题。  
◆ 设计了一种即插即用的视差置信度估计算法，通过检查相邻视差与相对深度的局部一致性来筛选可靠视差，显著减少了错误估计的噪声干扰。  
◆ 利用置信视差构建准密集对应关系，高效学习深度排序信息，提升了3D几何知识的利用率，避免了传统方法中随机稀疏对应带来的低效问题。  
◆ 提出了双重视差平滑损失函数，专门针对视差不连续区域优化匹配性能，增强了立体匹配在复杂场景（如重复纹理和无纹理区域）的鲁棒性。  
◆ 在KITTI Stereo基准测试中取得了无监督立体匹配方法的最高精度，验证了所提框架的优越性和实用性。</td></tr>
<tr><td>2025-08-02</td><td>A Coarse-to-Fine Approach to Multi-Modality 3D Occupancy Grounding</td><td>[2508.01197](http://arxiv.org/pdf/2508.01197)</td><td>◆ 提出首个针对户外场景的3D占据空间视觉定位基准，基于nuScenes数据集整合体素级占据标注与自然语言描述，突破传统边界框标注的粒度限制。  
◆ 设计端到端模型GroundingOcc，通过多模态（视觉/文本/点云）特征融合实现从粗到细的3D占据定位，包含多模态编码器、占据预测头和定位细化头三重架构。  
◆ 创新引入2D定位模块和深度估计模块，增强几何理解能力，显著提升模型在复杂户外场景下的性能表现。  
◆ 建立体素级占据预测与语言描述的直接关联，相比传统边界框定位能更精确表征物体空间分布（如部分占据情况）。  
◆ 实验验证新方法在3D占据定位任务上全面超越基线模型，同时开源数据集推动相关研究发展。</td></tr>
<tr><td>2025-08-01</td><td>Can Large Pretrained Depth Estimation Models Help With Image Dehazing?</td><td>[2508.00698](http://arxiv.org/pdf/2508.00698)</td><td>◆ 首次系统研究了预训练深度估计模型在图像去雾任务中的泛化能力，发现深度特征在不同雾浓度下具有显著一致性。  
◆ 提出了一种即插即用的RGB-D融合模块，可灵活适配多种去雾网络架构，无需重新训练。  
◆ 通过大规模实验验证了深度特征先验对提升去雾效果的普适性，突破了传统方法架构特定的局限性。  
◆ 揭示了跨任务知识迁移的新视角，证明预训练深度模型能有效捕捉雾浓度与场景深度的物理关联。  
◆ 提出的融合模块在多个基准测试中均显著提升现有去雾方法的性能，兼顾效率与精度需求。</td></tr>
<tr><td>2025-07-31</td><td>Stereo 3D Gaussian Splatting SLAM for Outdoor Urban Scenes</td><td>[2507.23677](http://arxiv.org/pdf/2507.23677)</td><td>◆ 首次提出针对户外场景的双目3D高斯泼溅SLAM系统（BGS-SLAM），填补了现有3DGS-SLAM技术主要局限于室内环境的空白。  
◆ 仅依赖RGB立体图像对，无需LiDAR或主动深度传感器，降低了硬件成本与使用门槛。  
◆ 创新性地利用预训练深度立体网络的深度估计结果，指导3D高斯优化过程，提升几何一致性。  
◆ 提出多损失策略，同时优化几何精度与视觉质量，在复杂户外环境中实现高保真重建。  
◆ 通过多数据集验证，系统在跟踪精度和建图性能上优于现有基于3DGS的解决方案，尤其适应大尺度动态场景。  
◆ 为户外SLAM提供了一种高效且轻量化的新范式，结合了神经渲染与传统SLAM的优势。</td></tr>
<tr><td>2025-07-30</td><td>Modality-Aware Feature Matching: A Comprehensive Review of Single- and Cross-Modality Techniques</td><td>[2507.22791](http://arxiv.org/pdf/2507.22791)</td><td>◆ 全面综述了单模态和跨模态特征匹配技术，涵盖RGB图像、深度图像、3D点云、LiDAR扫描、医学图像及视觉-语言交互等多种模态，填补了该领域系统性总结的空白。  
◆ 对比分析了传统手工方法（如Harris角点、SIFT和ORB描述子）与深度学习方法（如SuperPoint和LoFTR），指出后者在跨模态鲁棒性和适应性上的显著优势。  
◆ 重点介绍了模态感知的创新技术，包括针对深度图像的几何与深度专用描述子、3D点云的稀疏与稠密学习方法，以及LiDAR扫描的注意力增强神经网络。  
◆ 提出医学图像匹配的专用解决方案（如MIND描述子），解决了复杂模态下的特征匹配挑战。  
◆ 强调了跨模态应用（如医学图像配准和视觉-语言任务）的前沿进展，展示了特征匹配技术处理多样化数据交互的潜力。</td></tr>
<tr><td>2025-07-30</td><td>A Dual-Feature Extractor Framework for Accurate Back Depth and Spine Morphology Estimation from Monocular RGB Images</td><td>[2507.22691](http://arxiv.org/pdf/2507.22691)</td><td>◆ 提出了一种新颖的双特征提取器框架，通过单目RGB图像准确估计背部深度和脊柱形态，解决了传统X射线评估脊柱侧弯的辐射和可及性问题。  
◆ 设计了自适应多尺度特征学习网络（GAMA-Net），通过双编码器分别提取局部块级和全局特征，显著提升了背部细微深度变化的捕捉能力。  
◆ 创新性地引入基于块的混合注意力模块（PBHA），有效交互局部和全局特征，增强了模型对复杂背部表面特征的表达能力。  
◆ 开发了自适应多尺度特征融合模块（AMFF），动态解码并融合多尺度信息，进一步优化了深度估计的精度。  
◆ 首次将表面信息和预测深度信息相结合用于脊柱形态估计，显著提高了脊柱曲线生成的准确性，验证了深度信息对2D图像局限性的补偿作用。  
◆ 在三个不同评估指标上分别达到78.2%、93.6%和97.5%的高精度，脊柱形态估计性能最高达97%，为临床提供了一种安全、可及的替代方案。</td></tr>
<tr><td>2025-07-30</td><td>UAVScenes: A Multi-Modal Dataset for UAVs</td><td>[2507.22412](http://arxiv.org/pdf/2507.22412)</td><td>◆ 提出了UAVScenes数据集，填补了无人机多模态感知领域缺乏同时支持2D和3D高级场景理解任务的空白。  
◆ 基于MARS-LVIG数据集进行扩展，首次为无人机多模态数据（相机图像和LiDAR点云）提供逐帧语义标注，突破了现有数据集仅支持定位或地图级分割的限制。  
◆ 新增精确的6自由度位姿信息，支持更丰富的任务场景，如6-DoF定位、位姿识别和新型视图合成（NVS）。  
◆ 数据集设计兼顾多种感知任务，包括语义分割、深度估计、场景识别等，为无人机多模态算法开发提供统一基准。  
◆ 公开了完整标注的大规模数据集，促进无人机领域多模态感知研究的进一步发展。</td></tr>
<tr><td>2025-07-29</td><td>PanoSplatt3R: Leveraging Perspective Pretraining for Generalized Unposed Wide-Baseline Panorama Reconstruction</td><td>[2507.21960](http://arxiv.org/pdf/2507.21960)</td><td>◆ 提出PanoSplatt3R方法，首次实现无需精确位姿信息的宽基线全景图重建，突破传统方法对位姿依赖的限制。  
◆ 创新性地将透视域预训练迁移至全景域，通过领域自适应实现强大的泛化能力，解决跨域重建难题。  
◆ 设计RoPE滚动机制，在旋转位置编码中引入水平周期性建模，仅需最小改动即可适配全景图像特性。  
◆ 在无位姿条件下，新方法在生成高质量新视角和深度估计精度上显著超越现有最优技术。  
◆ 实验证明该方法具有实际应用潜力，为复杂场景下的沉浸式三维重建提供高效解决方案。</td></tr>
<tr><td>2025-07-25</td><td>Event-Based De-Snowing for Autonomous Driving</td><td>[2507.20901](http://arxiv.org/pdf/2507.20901)</td><td>◆ 提出基于事件相机的去雪方法，利用其亚毫秒级延迟和高动态范围特性，有效解决传统图像或视频去雪中的伪影和帧率限制问题。  
◆ 创新性地发现雪花遮挡在事件数据时空表征中呈现独特条纹特征，并设计注意力模块专门捕捉这些条纹以判断背景遮挡时刻。  
◆ 开发DSEC-Snow数据集，通过绿幕技术将真实雪景叠加到驾驶数据集，提供精确的同步图像-事件流及真实标注。  
◆ 在图像重建PSNR指标上超越现有方法3dB，且验证了去雪后数据可提升深度估计和光流等下游任务20%性能。  
◆ 首次将事件相机特性与注意力机制结合，显著提升自动驾驶系统在暴雪天气下的视觉可靠性和安全性。  
◆ 为全天候视觉系统提供新思路，突破传统方法对网络泛化能力和相机参数的强依赖性。</td></tr>
<tr><td>2025-07-28</td><td>Endoscopic Depth Estimation Based on Deep Learning: A Survey</td><td>[2507.20881](http://arxiv.org/pdf/2507.20881)</td><td>◆ 这篇论文首次系统综述了基于深度学习的内窥镜深度估计方法，填补了该领域缺乏全面总结最新技术的空白。  
◆ 从数据、方法和应用三个关键视角对现有技术进行分类梳理，涵盖了单目和立体视觉等多种方法。  
◆ 详细分析了内窥镜场景特有的挑战，并根据监督策略和网络架构对代表性技术进行了系统分类。  
◆ 总结了常用的性能评估指标和公开可用的数据集，为研究者提供了实用的资源参考。  
◆ 特别探讨了深度估计在机器人辅助手术这一重要领域的应用现状和发展潜力。  
◆ 指出了未来研究方向，包括领域自适应、实时实现和模型泛化能力提升等关键问题，为该领域的后续研究提供了明确指引。</td></tr>
<tr><td>2025-07-26</td><td>UniCT Depth: Event-Image Fusion Based Monocular Depth Estimation with Convolution-Compensated ViT Dual SA Block</td><td>[2507.19948](http://arxiv.org/pdf/2507.19948)</td><td>◆ 提出UniCT Depth方法，首次将CNN与Transformer统一用于事件-图像融合的单目深度估计，兼顾局部与全局特征建模。  
◆ 设计新型CcViT-DA编码器模块，包含上下文建模自注意力（CMSA）捕捉空间依赖，以及模态融合自注意力（MFSA）实现深度跨模态交互。  
◆ 创新性引入细节补偿卷积（DCC）模块，专门优化纹理细节与边缘表征，解决Transformer在局部特征上的不足。  
◆ 克服现有CNN方法感受野有限导致的遮挡与深度差异问题，同时改进纯Transformer方法模态交互浅层的缺陷。  
◆ 实验证明该方法在关键指标上全面超越基于图像、事件及融合的现有单目深度估计方法，尤其在动态场景表现突出。</td></tr>
<tr><td>2025-07-26</td><td>Leveraging Sparse LiDAR for RAFT-Stereo: A Depth Pre-Fill Perspective</td><td>[2507.19738](http://arxiv.org/pdf/2507.19738)</td><td>◆ 提出了一种在稀疏LiDAR条件下（如每帧仅几百个点）提升RAFT-Stereo性能的新方法，通过深度预填充策略解决LiDAR引导效果急剧下降的问题。  
◆ 从信号处理角度首次揭示了稀疏LiDAR导致性能下降的根本原因，为后续研究提供了理论依据。  
◆ 发现简单的插值预填充稀疏初始视差图能显著提升LiDAR引导效果，该方法实现简单但效果显著。  
◆ 揭示了早期融合中LiDAR深度注入图像特征时预填充同样有效，但作用机制与视差图预填充完全不同，需采用不同技术路线。  
◆ 结合两种预填充方案提出的GRAFT-Stereo框架，在多个数据集上显著优于现有稀疏LiDAR引导方法，为实际应用提供了可靠解决方案。  
◆ 该研究为LiDAR引导立体匹配领域提供了新思路，启发了更高效的传感器融合方法设计。</td></tr>
<tr><td>2025-07-24</td><td>Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting</td><td>[2507.18678](http://arxiv.org/pdf/2507.18678)</td><td>◆提出了一种可扩展的2D转3D数据生成流程，通过单视角图像自动生成包含点云、相机位姿、深度图等多种3D表示的综合数据。  
◆创新性地整合了深度估计、相机标定和尺度标定技术，实现了外观和尺度真实感的3D重建。  
◆解决了3D数据稀缺的核心瓶颈，利用海量2D图像资源大幅降低了3D数据采集成本。  
◆发布了两个大规模生成数据集COCO-3D和Objects365-v2-3D，填补了该领域数据空白。  
◆实验证明生成数据能有效支持从基础3D感知到多模态大语言模型推理的多种任务。  
◆为空间智能发展提供了新范式，使AI系统能更高效地感知和理解物理环境。</td></tr>
<tr><td>2025-07-24</td><td>DepthDark: Robust Monocular Depth Estimation for Low-Light Environments</td><td>[2507.18243](http://arxiv.org/pdf/2507.18243)</td><td>◆ 提出DepthDark，首个针对低光环境的鲁棒单目深度估计基础模型，填补了该领域空白。  
◆ 设计光照模拟模块（flare-simulation）和噪声模拟模块（noise-simulation），精准模拟夜间成像过程，生成高质量低光配对深度数据集。  
◆ 提出高效的参数微调策略（PEFT），结合光照引导和多尺度特征融合，显著提升模型在低光条件下的性能。  
◆ 在nuScenes-Night和RobotCar-Night等挑战性数据集上达到SOTA效果，验证了方法的有效性。  
◆ 仅需有限训练数据和计算资源即可实现高性能，为低光深度估计提供了实用解决方案。</td></tr>
<tr><td>2025-07-24</td><td>BokehDiff: Neural Lens Blur with One-Step Diffusion</td><td>[2507.18060](http://arxiv.org/pdf/2507.18060)</td><td>◆提出BokehDiff，一种基于生成扩散先验的新型镜头模糊渲染方法，实现物理精确且视觉吸引人的效果。  
◆通过物理启发的自注意力模块解决传统方法依赖深度估计精度的问题，该模块与成像过程对齐，整合了景深相关的弥散圆约束和自遮挡效应。  
◆将扩散模型适配到一步推理框架，无需引入额外噪声，同时保持高质量和高保真度的输出结果。  
◆针对缺乏可扩展配对数据的问题，创新性地使用扩散模型合成具有透明度的逼真前景，平衡真实性与场景多样性。  
◆整体方法在深度不连续区域有效减少伪影，显著优于现有技术的渲染效果。</td></tr>
<tr><td>2025-07-23</td><td>Monocular Semantic Scene Completion via Masked Recurrent Networks</td><td>[2507.17661](http://arxiv.org/pdf/2507.17661)</td><td>这篇论文的核心贡献是通过两阶段框架解决单目语义场景补全问题，显著提升了复杂场景下的性能表现。具体创新点如下：

◆ 提出新型两阶段框架MonoMRN，将任务分解为粗粒度补全和掩膜循环网络精修，克服了传统单阶段方法在遮挡区域预测和深度估计不准的缺陷。

◆ 设计掩膜稀疏门控循环单元(MS-GRU)，通过掩膜更新机制聚焦有效区域，配合稀疏结构设计显著降低计算成本。

◆ 创新性提出距离注意力投影机制，根据物体到观测表面的距离动态分配注意力权重，有效减少投影误差。

◆ 首次实现统一框架同时支持室内外场景，在NYUv2和SemanticKITTI数据集上达到SOTA性能。

◆ 通过系统性鲁棒性测试验证了掩膜循环网络对各类干扰的强抵抗能力，增强了模型实用性。</td></tr>
<tr><td>2025-07-22</td><td>SDGOCC: Semantic and Depth-Guided Bird&#x27;s-Eye View Transformation for 3D Multimodal Occupancy Prediction</td><td>[2507.17083](http://arxiv.org/pdf/2507.17083)</td><td>◆ 提出SDG-OCC多模态3D占据预测网络，结合语义与深度引导的鸟瞰图变换，解决现有单模态方法（相机缺乏深度、LiDAR易遮挡）的局限性。  
◆ 创新性地设计联合语义与深度引导的视角变换，通过扩散和双线性离散化整合像素语义与共点深度，构建更精确的深度分布。  
◆ 提出融合到占据驱动的主动蒸馏机制，从多模态数据中提取丰富语义信息，并基于LiDAR识别区域选择性增强图像特征。  
◆ 开发两种变体：SDG-Fusion（纯融合）和SDG-KL（融合+蒸馏），平衡性能与推理速度，满足不同场景需求。  
◆ 在Occ3D-nuScenes数据集上实现实时SOTA性能，在更复杂的SurroundOcc-nuScenes数据集上表现可比性，验证方法有效性与鲁棒性。</td></tr>
<tr><td>2025-07-21</td><td>DAViD: Data-efficient and Accurate Vision Models from Synthetic Data</td><td>[2507.15365](http://arxiv.org/pdf/2507.15365)</td><td>◆ 提出DAViD方法，通过高保真合成数据训练视觉模型，在保持精度的同时大幅降低数据需求。  
◆ 证明合成数据可完全替代海量真实数据，提供完美标注和细节，解决数据隐私与版权问题。  
◆ 首创程序化数据合成技术，通过可控数据多样性直接优化模型公平性。  
◆ 在深度估计、表面法线估计和前景分割三个密集预测任务上达到与SOTA模型相当的精度。  
◆ 模型训练和推理成本仅为同精度基础模型的极小部分，显著提升效率。  
◆ 开源合成数据集与预训练模型，推动可解释、合规的AI发展。</td></tr>
</tbody>
</table>
</div>

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2025.10.26
