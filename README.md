# SLAM领域最新论文 (2026.03.18)

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
<tr><td>2026-03-17</td><td>M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM</td><td>[2603.16844](http://arxiv.org/pdf/2603.16844)</td><td>◆ Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments.
◆ While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization.
◆ To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM.</td></tr>
<tr><td>2026-03-17</td><td>GenZ-LIO: Generalizable LiDAR-Inertial Odometry Beyond Indoor--Outdoor Boundaries</td><td>[2603.16273](http://arxiv.org/pdf/2603.16273)</td><td>◆ Light detection and ranging (LiDAR)-inertial odometry (LIO) enables accurate localization and mapping for autonomous navigation in various scenes.
◆ However, its performance remains sensitive to variations in spatial scale, which refers to the spatial extent of the scene reflected in the distribution of point ranges in a LiDAR scan.
◆ Transitions between confined indoor and expansive outdoor spaces induce substantial variations in point density, which may reduce robustness and computational efficiency.</td></tr>
<tr><td>2026-03-17</td><td>Industrial cuVSLAM Benchmark &amp; Integration</td><td>[2603.16240](http://arxiv.org/pdf/2603.16240)</td><td>◆ This work presents a comprehensive benchmark evaluation of visual odometry (VO) and visual SLAM (VSLAM) systems for mobile robot navigation in real-world logistical environments.
◆ We compare multiple visual odometry approaches across controlled trajectories covering translational, rotational, and mixed motion patterns, as well as a large-scale production facility dataset spanning approximately 1.7 km.
◆ Performance is evaluated using Absolute Pose Error (APE) against ground truth from a Vicon motion capture system and a LiDAR-based SLAM reference.</td></tr>
<tr><td>2026-03-17</td><td>SE(3)-LIO: Smooth IMU Propagation With Jointly Distributed Poses on SE(3) Manifold for Accurate and Robust LiDAR-Inertial Odometry</td><td>[2603.16118](http://arxiv.org/pdf/2603.16118)</td><td>◆ In estimating odometry accurately, an inertial measurement unit (IMU) is widely used owing to its high-rate measurements, which can be utilized to obtain motion information through IMU propagation.
◆ In this paper, we address the limitations of existing IMU propagation methods in terms of motion prediction and motion compensation.
◆ In motion prediction, the existing methods typically represent a 6-DoF pose by separating rotation and translation and propagate them on their respective manifold, so that the rotational variation is not effectively incorporated into translation propagation.</td></tr>
<tr><td>2026-03-16</td><td>On the Derivation of Tightly-Coupled LiDAR-Inertial Odometry with VoxelMap</td><td>[2603.15471](http://arxiv.org/pdf/2603.15471)</td><td>◆ This note presents a concise mathematical formulation of tightly-coupled LiDAR-Inertial Odometry within an iterated error-state Kalman filter framework using a VoxelMap representation.
◆ Rather than proposing a new algorithm, it provides a clear and self-contained derivation that unifies the geometric modeling and probabilistic state estimation through consistent notation and explicit formulations.
◆ The document is intended to serve both as a technical reference and as an accessible entry point for a foundational understanding of the system architecture and estimation principles.</td></tr>
<tr><td>2026-03-16</td><td>Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3</td><td>[2603.14998](http://arxiv.org/pdf/2603.14998)</td><td>◆ Autonomous navigation in GPS-denied and visually degraded environments remains challenging for unmanned aerial vehicles (UAVs).
◆ To this end, we investigate the use of a monocular thermal camera as a standalone sensor on a UAV platform for real-time depth estimation and simultaneous localization and mapping (SLAM).
◆ To extract depth information from thermal images, we propose a novel pipeline employing a lightweight supervised network with recurrent blocks (RBs) integrated to capture temporal dependencies, enabling more robust predictions.</td></tr>
<tr><td>2026-03-15</td><td>eNavi: Event-based Imitation Policies for Low-Light Indoor Mobile Robot Navigation</td><td>[2603.14397](http://arxiv.org/pdf/2603.14397)</td><td>◆ Event cameras provide high dynamic range and microsecond-level temporal resolution, making them well-suited for indoor robot navigation, where conventional RGB cameras degrade under fast motion or low-light conditions.
◆ Despite advances in event-based perception spanning detection, SLAM, and pose estimation, there remains limited research on end-to-end control policies that exploit the asynchronous nature of event streams.
◆ To address this gap, we introduce a real-world indoor person-following dataset collected using a TurtleBot 2 robot, featuring synchronized raw event streams, RGB frames, and expert control actions across multiple indoor maps, trajectories under both normal and low-light conditions.</td></tr>
<tr><td>2026-03-14</td><td>Evaluation of Visual Place Recognition Methods for Image Pair Retrieval in 3D Vision and Robotics</td><td>[2603.13917](http://arxiv.org/pdf/2603.13917)</td><td>◆ Visual Place Recognition (VPR) is a core component in computer vision, typically formulated as an image retrieval task for localization, mapping, and navigation.
◆ In this work, we instead study VPR as an image pair retrieval front-end for registration pipelines, where the goal is to find top-matching image pairs between two disjoint image sets for downstream tasks such as scene registration, SLAM, and Structure-from-Motion.
◆ We comparatively evaluate state-of-the-art VPR families - NetVLAD-style baselines, classification-based global descriptors (CosPlace, EigenPlaces), feature-mixing (MixVPR), and foundation-model-driven methods (AnyLoc, SALAD, MegaLoc) - on three challenging datasets: object-centric outdoor scenes (Tanks and Temples), indoor RGB-D scans (ScanNet-GS), and autonomous-driving sequences (KITTI).</td></tr>
<tr><td>2026-03-13</td><td>Semantic Aware Feature Extraction for Enhanced 3D Reconstruction</td><td>[2603.13556](http://arxiv.org/pdf/2603.13556)</td><td>◆ Feature matching is a fundamental problem in computer vision with wide-ranging applications, including simultaneous localization and mapping (SLAM), image stitching, and 3D reconstruction.
◆ While recent advances in deep learning have improved keypoint detection and description, most approaches focus primarily on geometric attributes and often neglect higher-level semantic information.
◆ This work proposes a semantic-aware feature extraction framework that employs multi-task learning to jointly train keypoint detection, keypoint description, and semantic segmentation.</td></tr>
<tr><td>2026-03-13</td><td>Consistent and Efficient MSCKF-based LiDAR-Inertial Odometry with Inferred Cluster-to-Plane Constraints for UAVs</td><td>[2603.12904](http://arxiv.org/pdf/2603.12904)</td><td>◆ Robust and accurate navigation is critical for Unmanned Aerial Vehicles (UAVs) especially for those with stringent Size, Weight, and Power (SWaP) constraints.
◆ However, most state-of-the-art (SOTA) LiDAR-Inertial Odometry (LIO) systems still suffer from estimation inconsistency and computational bottlenecks when deployed on such platforms.
◆ To address these issues, this paper proposes a consistent and efficient tightly-coupled LIO framework tailored for UAVs.</td></tr>
<tr><td>2026-03-13</td><td>CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration</td><td>[2603.12721](http://arxiv.org/pdf/2603.12721)</td><td>◆ Robust point cloud registration is a fundamental task in 3D computer vision and geometric deep learning, essential for applications such as large-scale 3D reconstruction, augmented reality, and scene understanding.
◆ However, the performance of established learning-based methods often degrades in complex, real world scenarios characterized by incomplete data, sensor noise, and low overlap regions.
◆ To address these limitations, we propose CMHANet, a novel Cross-Modal Hybrid Attention Network.</td></tr>
<tr><td>2026-03-14</td><td>Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos</td><td>[2603.12064](http://arxiv.org/pdf/2603.12064)</td><td>该论文针对多台自由移动相机拍摄的动态场景，提出了一个创新的两阶段优化框架，用于实现稠密三维重建与相机姿态估计。其核心贡献与创新点如下：

◆ 提出了一个两阶段优化框架，将任务解耦为鲁棒的相机跟踪和稠密深度优化，解决了多自由移动相机场景下的重建难题。

◆ 在第一阶段，通过构建一个同时利用相机内时间连续性和相机间空间重叠的时空连接图，将单相机视觉SLAM扩展至多相机系统，实现了尺度一致且鲁棒的跟踪。

◆ 引入了一种基于前馈重建模型的宽基线初始化策略，确保了在相机间视野重叠有限情况下的系统鲁棒性。

◆ 在第二阶段，提出利用宽基线光流来优化稠密的相机间与相机内一致性，从而联合精细化深度估计与相机姿态。

◆ 创建并公开了一个名为MultiCamRobolab的新真实世界数据集，该数据集使用运动捕捉系统提供了真实的地面姿态真值，用于方法评估。

◆ 实验证明，该方法在合成与真实基准测试上均显著优于当前最先进的前馈模型，同时所需内存更少。</td></tr>
<tr><td>2026-03-11</td><td>D-SLAMSpoof: An Environment-Agnostic LiDAR Spoofing Attack using Dynamic Point Cloud Injection</td><td>[2603.11365](http://arxiv.org/pdf/2603.11365)</td><td>该论文的核心贡献在于提出了一种新型激光雷达欺骗攻击方法及其防御方案。  
◆ 创新性地提出了D-SLAMSpoof攻击，通过外部激光干扰向激光雷达扫描中动态注入虚假点云，能够有效攻击特征丰富的真实环境（如城市和室内场景），突破了传统欺骗方法在复杂环境中成功率低的限制。  
◆ 该攻击设计了基于扫描匹配原理的空间注入形状与时间协同的动态注入模式，实现了对环境不敏感（环境无关）的欺骗，显著提高了攻击的隐蔽性和成功率。  
◆ 论文同时提出了首个仅利用自动驾驶系统常见惯性航位推算信号的实用防御方法ISD-SLAM。  
◆ ISD-SLAM能够准确检测包括D-SLAMSpoof在内的激光雷达欺骗攻击，并有效缓解攻击导致的位置漂移，无需额外专用传感器。  
◆ 这项工作揭示了基于激光雷达的SLAM系统固有的安全漏洞，并为提升自主系统的安全性与可靠性提供了关键见解。</td></tr>
<tr><td>2026-03-11</td><td>MirrorDrift: Actuated Mirror-Based Attacks on LiDAR SLAM</td><td>[2603.11364](http://arxiv.org/pdf/2603.11364)</td><td>该论文提出了一种针对LiDAR SLAM系统的创新物理攻击方法MirrorDrift。其核心贡献与创新点如下：

◆ 首次利用镜面反射原理，提出了一种无需信号注入的攻击方式。该方法通过一个受控的平面镜产生虚假点云，避开了传统攻击依赖的激光雷达信号欺骗技术。

◆ 攻击无需传感器特定的时序知识，因此能够有效绕过现代激光雷达的防御机制，如时序混淆和注入拒绝等先进干扰缓解技术。

◆ 系统性地优化了镜子的放置位置、对准方式和驱动控制，从而能够有针对性地干扰扫描匹配中的点云对应关系，导致SLAM系统定位发生系统性偏差。

◆ 在仿真和真实世界实验中均验证了有效性。在仿真中，其攻击效果比随机放置镜子使平均位姿误差提升6.1倍；在配备了先进抗干扰技术的现代激光雷达实车上，成功引发了高达6.03米的定位误差。

◆ 据作者所知，这是首个成功针对生产级安全激光雷达的、以SLAM为目标的攻击，揭示了基于几何假设的LiDAR SLAM在面对物理反射攻击时存在新的脆弱性。</td></tr>
<tr><td>2026-03-11</td><td>Edge-Assisted Multi-Robot Visual-Inertial SLAM with Efficient Communication</td><td>[2603.11085](http://arxiv.org/pdf/2603.11085)</td><td>该论文的核心贡献是提出了一种基于机器人-边缘-云三层架构的集中式多机器人视觉惯性SLAM系统，旨在实现全局一致且实时的协同定位与建图。其创新点主要体现在以下方面：

◆ 提出了一种轻量化的单机器人SLAM方法，采用基于金字塔IMU预测的光流跟踪，降低了特征跟踪的计算成本，提升了机器人端的实时性。

◆ 设计了机器人-边缘-云分层协同架构，将计算任务合理分配，克服了单机板载计算资源有限和执行效率低的问题，同时利用云端实现全局一致性。

◆ 实现了高效通信机制，仅传输特征点和关键帧描述子，并对其进行无损编码与压缩，从而在有限带宽下实现实时远程传输。

◆ 该通信策略显著降低了数据传输的实际带宽占用，并且通过无损压缩避免了因数据压缩导致的SLAM精度损失。

实验基于EuRoC数据集验证，该方法相比先进的局部特征压缩方法能以更少的数据量传输特征，相比先进的集中式多机器人SLAM方案能在低计算负载下达到相同或更高的定位精度。</td></tr>
<tr><td>2026-03-11</td><td>Semantic Landmark Particle Filter for Robot Localisation in Vineyards</td><td>[2603.10847](http://arxiv.org/pdf/2603.10847)</td><td>该论文针对葡萄园等高度重复的农业环境中机器人定位的难题，提出了创新解决方案。其核心贡献在于通过语义信息显著提升了在感知混淆场景下的定位鲁棒性和精度。

◆ 提出了语义地标粒子滤波器（SLPF），将树干和杆柱等语义地标检测与2D激光雷达数据融合到一个概率定位框架中。
◆ 创新地将检测到的树干转换为语义墙，并将其作为结构化的行边界嵌入测量模型，从而有效区分外观相似的相邻作物行。
◆ 设计了一种轻量级方法，将GNSS数据作为先验信息融入系统，在语义观测稀疏时稳定定位，增强了系统的整体可靠性。
◆ 通过实地实验验证，该方法在绝对姿态误差、行正确率和横向跟踪误差等多个指标上，均显著优于仅依赖几何、视觉或GNSS的基线方法，证明了语义信息在解决行级感知混淆问题上的关键作用。</td></tr>
<tr><td>2026-03-11</td><td>Adaptive Manipulation Potential and Haptic Estimation for Tool-Mediated Interaction</td><td>[2603.10352](http://arxiv.org/pdf/2603.10352)</td><td>该论文针对视觉遮挡和触觉感知不确定的挑战，提出了一个用于工具化操作的新型闭环框架。其核心贡献与创新点如下：

◆ 提出了一种参数化的平衡流形作为工具交互的统一表征，将复杂的物理接触互动封装为该流形上的连续操作。
◆ 建立了物理-几何对偶性，通过融合可微接触模型的自适应操作势能，诱导出流形的几何结构。
◆ 将触觉估计重新定义为流形参数估计问题，并创新性地采用混合推理策略，即结合粒子滤波进行离散物体形状分类，同时利用解析梯度高效优化估计连续物体位姿。
◆ 开发了一个集成触觉估计、在线规划和自适应刚度控制的闭环框架，通过实时更新操作势能参数来动态重塑平衡流形，从而指导在线轨迹重规划并实施不确定性感知的阻抗控制。
◆ 通过大量实物实验验证了框架的有效性，证明其在标准场景中能实现鲁棒的识别与操作，且触觉SLAM与刚度调制方法优于固定阻抗基线，能有效防止卡死。</td></tr>
<tr><td>2026-03-10</td><td>VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM</td><td>[2603.09673](http://arxiv.org/pdf/2603.09673)</td><td>该论文的核心贡献是提出了VarSplat，一个将显式不确定性建模融入3D高斯溅射（3DGS）的RGB-D SLAM系统，以提升其在挑战性场景中的鲁棒性。

其核心创新点如下：
◆ 首次在3DGS-SLAM中显式地学习每个高斯椭球体（splat）的外观方差，从而量化其不确定性。
◆ 提出基于全方差定律和alpha合成的渲染方法，能够通过高效的单次光栅化，直接渲染出可微的逐像素不确定性图。
◆ 利用生成的不确定性图，在SLAM的多个关键环节（如跟踪、子地图配准和回环检测）中引导系统关注可靠区域，抑制噪声干扰，从而实现了更稳定的优化。
◆ 在合成与多个真实世界数据集上的实验表明，该系统在跟踪鲁棒性、建图和新视图合成质量上达到或超越了现有先进方法。</td></tr>
<tr><td>2026-03-12</td><td>X-GS: An Extensible Open Framework for Perceiving and Thinking via 3D Gaussian Splatting</td><td>[2603.09632](http://arxiv.org/pdf/2603.09632)</td><td>X-GS的核心贡献是提出了一个统一且可扩展的开放框架，旨在解决现有3D高斯溅射（3DGS）技术各自为政、难以与下游多模态模型集成的问题。

◆ 框架创新：首次构建了一个可扩展的开放框架，将在线SLAM、语义增强、无位姿图像处理等多种3DGS架构统一起来，并桥接至下游多模态模型。
◆ 核心管线：设计了高效的X-GS-Perceiver管线，能够从无位姿的RGB（或RGB-D）视频流中，实时协同优化场景几何与相机位姿。
◆ 语义蒸馏：能够从大规模视觉基础模型中提取高维语义特征，并将其蒸馏到3D高斯表示中，从而创建出富含语义信息的3D场景。
◆ 实时性能：通过创新的在线向量量化模块、GPU加速的网格采样方案以及高度并行化的管线设计，实现了整个系统的实时运行。
◆ 下游应用：通过X-GS-Thinker组件，使语义化的3D高斯场能够被视觉语言模型直接利用，解锁了物体检测、零样本描述生成等新颖的多模态下游任务。</td></tr>
<tr><td>2026-03-10</td><td>Cutting the Cord: System Architecture for Low-Cost, GPU-Accelerated Bimanual Mobile Manipulation</td><td>[2603.09051](http://arxiv.org/pdf/2603.09051)</td><td>这篇论文的核心贡献是构建了一个低成本、高性能的双臂移动操作机器人平台。其创新点主要体现在系统架构和集成设计上。

◆ 首先，提出了一个优化的机械设计，旨在最大化结构的刚度与重量比，从而在保证低成本的同时提升机器人的运动性能。
◆ 其次，创新性地设计了一种Tri-Bus电源拓扑结构，该设计将计算单元与电机驱动电路隔离，有效避免了电机工作引起的电压瞬变对精密计算模块的干扰。
◆ 最后，实现了高度集成的嵌入式自主系统，通过搭载NVIDIA Jetson Orin Nano作为核心计算单元，使机器人能够不依赖外部设备（即“剪断线缆”）独立运行。
◆ 综合以上创新，该平台以低于1300美元的总成本，完整支持远程遥操作、自主SLAM导航以及基于视觉的操控任务。
◆ 因此，该工作为机器人学和机器人学习领域的研究与教育提供了一个极具成本效益且功能齐全的开放式软硬件平台。</td></tr>
<tr><td>2026-03-09</td><td>Overlapping Schwarz Preconditioners for Pose-Graph SLAM in Robotics</td><td>[2603.08975](http://arxiv.org/pdf/2603.08975)</td><td>本文核心贡献是将求解偏微分方程的领域分解预条件器创新性地应用于机器人SLAM中的位姿图优化问题。其主要创新点如下：

◆ 首次将加性重叠Schwarz领域分解方法用作预条件器，以求解SLAM后端非线性最小二乘问题线性化后产生的大规模稀疏线性系统。

◆ 通过数值实验证明，采用该预条件器后，预处理共轭梯度法的迭代次数不随问题规模增大而增加，展现了方法的数值可扩展性。

◆ 建立了一个新颖的理论类比，将简化的SLAM问题阐释为使用线性弹性杆的有限元问题，从而揭示了其与偏微分方程离散化在数学结构上的相似性。

◆ 这一类比为将成熟的偏微分方程数值解法（特别是可扩展的领域分解预条件器）引入SLAM领域提供了理论动机和坚实基础。

◆ 论文的阐述方式兼顾了SLAM和领域分解两个领域的背景知识，使得缺乏任一领域先验知识的读者也能理解其交叉研究内容。</td></tr>
<tr><td>2026-03-09</td><td>Improving Continual Learning for Gaussian Splatting based Environments Reconstruction on Commercial Off-the-Shelf Edge Devices</td><td>[2603.08499](http://arxiv.org/pdf/2603.08499)</td><td>该论文的核心贡献是提出了一种精度自适应的优化框架，使得基于变分贝叶斯高斯泼溅（VBGS）的持续学习三维重建方法能够在资源受限的边缘设备上高效运行。其创新点主要包括：

◆ 首次系统剖析了VBGS在内存和延迟方面的性能瓶颈，为针对性优化提供了依据。
◆ 设计了内存主导内核的融合策略，显著减少了中间张量的显式存储，从而大幅降低了峰值内存占用。
◆ 提出了一种基于有界相对误差的混合精度搜索方法，可自动为不同操作分配合适的计算精度，在保证质量的同时提升效率。
◆ 通过上述优化，在多个数据集上验证了其框架能大幅降低内存消耗和训练时间，同时保持甚至提升了重建质量。
◆ 首次在Jetson Orin Nano等商用嵌入式平台上实现了高质量的神经渲染训练，为边缘机器人的即时场景建模与更新提供了实用方案。</td></tr>
<tr><td>2026-03-09</td><td>FoMo: A Multi-Season Dataset for Robot Navigation in Forêt Montmorency</td><td>[2603.08433](http://arxiv.org/pdf/2603.08433)</td><td>该论文的核心贡献是发布了FoMo数据集，这是一个用于机器人导航研究的多季节、多传感器数据集。其创新点主要体现在以下几个方面：

◆ 提供了在北方森林中跨越一整年收集的、涵盖显著季节性变化（如超1米积雪、植被生长）的全面数据，这挑战了现有的里程计与SLAM算法。
◆ 数据集包含超过64公里的六条多样化轨迹，并在全年12次部署中重复采集，确保了环境变化的连贯性与可比性。
◆ 集成了丰富的传感器套件，包括两种激光雷达（旋转式与混合固态）、FMCW雷达、立体与广角单目相机以及两个IMU，支持多模态研究。
◆ 通过后处理UGV上的三个GNSS接收机与静态基站数据提供了高精度地面真值，并附带了气象站数据、相机标定参数等元数据。
◆ 通过初步评估证明了季节变化对当前先进的激光雷达-惯性、雷达-陀螺及视觉-惯性定位与建图方法的重新定位能力有严重影响，凸显了该数据集对推动鲁棒导航算法发展的价值。</td></tr>
<tr><td>2026-03-09</td><td>Perception-Aware Communication-Free Multi-UAV Coordination in the Wild</td><td>[2603.08379](http://arxiv.org/pdf/2603.08379)</td><td>该论文的核心贡献是提出了一种适用于野外复杂环境（如GNSS拒止的茂密森林）的无通信多无人机协同导航方法。

◆ 提出了一种创新的“无通信”协同机制，仅依赖机载感知实现多机协调，解决了通信受限或不可靠环境下的协同难题。
◆ 开发了一种新颖的“感知感知”三维导航框架，该框架能主动考虑传感器有限视场的约束，规划出既安全又有效的路径。
◆ 实现了传感器（3D LiDAR）的“一感多用”，即同时用于SLAM建图、障碍物检测以及邻近机器人识别，提升了系统集成度与效率。
◆ 通过广泛的仿真与真实的野外实验，验证了该方法在复杂场景下的可扩展性、鲁棒性和可靠性。</td></tr>
<tr><td>2026-03-09</td><td>Edged USLAM: Edge-Aware Event-Based SLAM with Learning-Based Depth Priors</td><td>[2603.08150](http://arxiv.org/pdf/2603.08150)</td><td>本文提出Edged USLAM，一个融合事件相机、惯性测量单元与标准相机的混合视觉惯性SLAM系统，旨在解决传统视觉SLAM在高速运动、弱光及光照突变下的失效问题。其核心贡献与创新点如下：

◆ 提出一种边缘感知的前端处理模块，通过增强事件帧以实现鲁棒的特征跟踪和非线性运动补偿，有效应对事件数据稀疏异步的挑战。

◆ 引入一个轻量化的深度估计模块，提供基于感兴趣区域的粗略场景深度，以此提升运动补偿的精度和系统的尺度一致性。

◆ 构建了一个完整的混合视觉惯性系统，将事件相机的高动态范围与高时间分辨率优势，与传统相机及惯性数据进行有效融合，增强了系统在不同场景下的适应性。

◆ 通过公开数据集和真实无人机飞行测试验证，系统在缓慢或结构化轨迹中表现出卓越的稳定性和低漂移特性，为多样化的空中导航任务提供了一个鲁棒的解决方案。

◆ 研究结果揭示了纯事件方法、基于学习的方法与混合方法各自的互补优势，明确了Edged USLAM在特定性能平衡下的适用场景。</td></tr>
<tr><td>2026-03-09</td><td>RLPR: Radar-to-LiDAR Place Recognition via Two-Stage Asymmetric Cross-Modal Alignment for Autonomous Driving</td><td>[2603.07920](http://arxiv.org/pdf/2603.07920)</td><td>该论文提出了一种名为RLPR的鲁棒雷达到激光雷达地点识别框架，旨在解决全天候自动驾驶中跨模态定位的难题。其核心贡献与创新点如下：

◆ 提出了一种兼容多种雷达（单芯片、扫描和4D雷达）的雷达到激光雷达地点识别框架，增强了系统在实际部署中的适用性和鲁棒性。

◆ 设计了一个双流网络，用于提取抽象于传感器特定信号属性（如多普勒或RCS）的结构特征，从而专注于跨模态共享的判别性信息。

◆ 基于对雷达与激光雷达之间任务特定不对称性的观察，创新性地引入了一种两阶段非对称跨模态对齐策略。

◆ 该对齐策略利用预训练的雷达分支作为判别性锚点，来指导整个对齐过程，有效缓解了配对训练数据稀缺和跨雷达类型信号异质性带来的挑战。

◆ 在四个数据集上的实验表明，该方法实现了最先进的识别精度，并展现出强大的零样本泛化能力。</td></tr>
<tr><td>2026-03-07</td><td>MipSLAM: Alias-Free Gaussian Splatting SLAM</td><td>[2603.06989](http://arxiv.org/pdf/2603.06989)</td><td>MipSLAM是一个基于3D高斯泼溅的频率感知SLAM框架，其核心贡献在于通过频域方法显著提升了渲染质量与定位鲁棒性。具体创新点如下：

◆ 提出了椭圆自适应抗锯齿算法，通过几何感知的数值积分来近似高斯贡献，避免了昂贵解析计算，有效抑制了混叠伪影。
◆ 设计了频谱感知的位姿图优化模块，将轨迹估计重新在频域中建模，利用图拉普拉斯分析抑制高频噪声和轨迹漂移。
◆ 引入了一种新颖的局部频域感知损失函数，以增强对细粒度几何细节的恢复能力。
◆ 整个系统在多种相机配置下，能够同时实现高保真抗锯齿的新视图合成与鲁棒的实时定位。
实验证明，该系统在Replica和TUM数据集上实现了领先的渲染质量和定位精度。</td></tr>
<tr><td>2026-03-05</td><td>Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM</td><td>[2603.05397](http://arxiv.org/pdf/2603.05397)</td><td>该论文针对3D激光SLAM中闭环检测的可靠性问题，提出了一种名为CliReg的新型确定性算法。其核心贡献与创新点如下：

◆ 提出了一种全新的闭环验证方法，用最大团搜索替代了传统的RANSAC随机采样验证。该方法通过构建特征匹配对的兼容性图，并在图中寻找最大团，从而避免了RANSAC的随机性。

◆ 该算法显著提升了在噪声、外点以及环境稀疏或模糊条件下的鲁棒性。其确定性本质使得闭环检测结果更加稳定可靠，有效减少了地图不一致的风险。

◆ 将所提算法集成进了一个实时处理流程，该流程结合了二进制3D描述子和基于汉明距离嵌入的二叉搜索树匹配方法，保证了系统的实用效率。

◆ 在多个配备不同激光雷达的真实世界数据集上进行了评估，实验结果表明，与RANSAC方法相比，该方法能持续获得更低的位姿误差和更可靠的闭环结果。

◆ 通过在2D投影地图上的附加实验，验证了该方法在不同空间领域（2D与3D）的通用性，表明其是一个跨领域的鲁棒且高效的闭环检测替代方案。</td></tr>
<tr><td>2026-03-06</td><td>AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization with Foundation Model</td><td>[2603.05097](http://arxiv.org/pdf/2603.05097)</td><td>该论文提出了AIM-SLAM，一个用于单目视觉SLAM的密集重建框架，其核心贡献与创新点如下：

◆ 提出了自适应且信息丰富的多视图关键帧优先选择机制，突破了以往方法局限于两视图对或固定长度输入的局限。
◆ 设计了SIGMA模块，该模块结合体素重叠率和信息增益来检索关键帧候选集，并能自适应地确定其规模，从而更充分地利用几何上下文。
◆ 构建了一个联合多视图Sim(3)优化方法，强制所选视图间保持一致的配准，显著提升了位姿估计的精度。
◆ 整个框架有效利用了视觉几何基础模型（VGGT）预测的密集点云图，实现了在真实世界数据集上领先的位姿估计与密集重建性能。
◆ 系统支持ROS集成，提供了实用的开源代码。</td></tr>
<tr><td>2026-03-04</td><td>Efficient Autonomous Navigation of a Quadruped Robot in Underground Mines on Edge Hardware</td><td>[2603.04470](http://arxiv.org/pdf/2603.04470)</td><td>该论文针对地下矿井环境（狭窄、不平、黑暗、无GPS、通信差）的自主导航难题，提出了一套完全运行在低功耗边缘计算设备上的四足机器人全自主导航系统。

◆ 提出了一套完全脱离GPU和网络连接、仅依靠低功耗Intel NUC边缘计算机的完整自主导航解决方案，显著提升了系统在严苛环境下的独立性和部署便利性。
◆ 集成激光雷达惯性里程计、先验地图扫描匹配定位、地形分割、可见性图全局规划与速度调节局部路径跟随，实现了在有限算力下稳定控制频率的实时感知-动作闭环。
◆ 系统仅需对环境进行一次建图，即可在已知地图内处理任意目标点导航，无需任何针对特定环境的训练或学习组件，降低了部署门槛与数据依赖。
◆ 在真实地下矿井环境中进行了系统验证，累计超过700米的全自主行走，在20次不同难度任务中成功率达到100%，平均SPL（路径长度加权成功率）达到0.73，证明了其可靠性与实用性。</td></tr>
<tr><td>2026-03-04</td><td>HBRB-BoW: A Retrained Bag-of-Words Vocabulary for ORB-SLAM via Hierarchical BRB-KMeans</td><td>[2603.04144](http://arxiv.org/pdf/2603.04144)</td><td>本文针对ORB-SLAM中传统二进制视觉词袋词汇表因精度损失导致性能下降的问题，提出了一种创新的词汇表训练方法。其核心贡献与创新点如下：

◆ 提出了一种名为HBRB-BoW的分层词汇训练新算法，旨在解决传统基于k-majority的二进制聚类方法固有的结构缺陷。
◆ 创新地在分层聚类过程中引入了全局实值流，从而在最终叶节点二值化之前，能够长时间保持描述子的高保真度信息。
◆ 该方法有效缓解了传统二进制聚类无法表征细微特征分布的问题，减少了因误差在树结构中累积和传播所导致的视觉单词退化。
◆ 实验证明，所生成的词汇表比传统方法更具区分度且结构更优，显著提升了复杂环境下视觉词典的表征完整性。
◆ 所生成的词汇文件可直接替换ORB-SLAM的默认文件，有望在回环检测和重定位等关键任务中提升系统整体性能。</td></tr>
<tr><td>2026-03-04</td><td>TreeLoc++: Robust 6-DoF LiDAR Localization in Forests with a Compact Digital Forest Inventory</td><td>[2603.03695](http://arxiv.org/pdf/2603.03695)</td><td>该论文提出了TreeLoc++，一种用于森林环境的6-DoF激光雷达全局定位框架。其核心贡献在于首次直接利用紧凑的数字森林清单作为地图进行定位，无需依赖传统庞大、昂贵的原始点云数据。主要创新点如下：
◆ 首次将数字森林清单这一林业管理中的标准结构化数据（仅用树干几何属性表示）作为定位的判别性表征，实现了从依赖密集点云到使用紧凑语义地图的范式转变。
◆ 提出了一种结合成对距离直方图的检索方法，编码局部树木布局上下文以增强区分度，并通过基于胸径的过滤和偏航角一致性内点选择来细化候选匹配，有效减少了森林结构相似环境下的误匹配。
◆ 设计了一种利用树干几何约束的优化方法，联合估计滚转、俯仰和高度，提升了全6自由度位姿估计的稳定性和准确性，摆脱了对密集3D点云的依赖。
该系统在跨多国的27个序列上实现了厘米级精确定位，仅用250KB地图数据即可覆盖近8公里轨迹，并展示了跨两年时间间隔的长期鲁棒性，其性能优于依赖点云地图的传统及学习基线方法。</td></tr>
<tr><td>2026-03-03</td><td>Probabilistic Occupancy Grid for Radio-Based SLAM</td><td>[2603.03559](http://arxiv.org/pdf/2603.03559)</td><td>本文针对6G及未来无线感知中依赖简化几何模型（如点散射体）导致无法准确捕捉复杂物体的问题，提出了一种用于无线电SLAM的概率占据栅格框架。其核心创新如下：

◆ 提出了一种概率占据栅格地图表示法，并将其集成到基于多径的SLAM框架中，能同时进行移动代理定位和环境地图构建。
◆ 该框架通过栅格单元状态联合重建环境的几何结构及其无线电相关属性（如反射系数），突破了传统简化几何模型的限制。
◆ 采用表面模型将射频测量与栅格地图关联起来，从而捕捉测量不确定性和细粒度几何细节。
◆ 提供了一种原理性的、可物理解释的无线电地图构建方法，仿真验证了其在几何、材料属性重建及高精度定位方面的有效性。
◆ 展示了利用从其他无线电设备或互补传感器获得的先验占据地图进行后续地图扩展与优化的潜力。</td></tr>
<tr><td>2026-03-03</td><td>The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes</td><td>[2603.02985](http://arxiv.org/pdf/2603.02985)</td><td>该论文的核心贡献是创建了一个名为D4D的大规模、高质量数据集，专门用于在接近真实手术的条件下，评估腹部非刚性软组织变形场景的4D重建算法。

其核心创新点包括：
◆ 提供了首个在逼真手术条件下（猪尸体实验）配对的腹腔镜视频与高精度结构光几何数据，实现了对变形组织的定量几何评估。
◆ 设计了三种序列类型（整体变形、增量变形和相机移动），系统性地用于测试算法对非刚性运动、大变形和视野外更新的鲁棒性。
◆ 数据集内容丰富且经过精细处理，不仅提供立体图像，还包括仪器掩码、立体深度、起始与结束的结构光点云、精确的相机位姿与内参。
◆ 通过光学追踪与半自动配准技术，确保了多模态数据（视频、点云、位姿）在时空上的高精度对齐。
◆ 数据集规模庞大且用途明确，包含超过30万帧图像和369个点云，可作为非刚性SLAM、4D重建和深度估计方法的综合基准。</td></tr>
<tr><td>2026-03-03</td><td>Exploiting Double-Bounce Paths in Snapshot Radio SLAM: Bounds, Algorithms and Experiments</td><td>[2603.02832](http://arxiv.org/pdf/2603.02832)</td><td>该论文的核心贡献在于首次系统性地利用双跳非视距路径提升双基地快照无线电SLAM性能。传统方法仅利用视距和单跳非视距路径，而将高阶反射视为干扰，本文则挖掘双跳路径的潜在价值。

◆ 理论层面，首次推导了在存在双跳非视距路径情况下，联合估计用户设备状态与路标位置的克拉美-罗下界，为性能评估提供了理论基准。

◆ 算法层面，提出了一种能够识别双跳非视距路径，并将其有效整合到用户设备与路标联合估计过程中的新算法。

◆ 性能层面，通过仿真与真实的毫米波5G波束成形测量实验验证，证明与单跳路径共享至少一个入射点的双跳路径，能显著提升用户设备状态及现有入射点的估计精度。

◆ 应用层面，利用双跳路径能够揭示仅靠单跳路径无法观测到的环境路标，从而显著增强了系统的环境建图能力。</td></tr>
<tr><td>2026-03-03</td><td>PathSpace: Rapid continuous map approximation for efficient SLAM using B-Splines in constrained environments</td><td>[2603.02538](http://arxiv.org/pdf/2603.02538)</td><td>该论文提出了一种名为PathSpace的新型语义SLAM框架，其核心贡献在于利用连续B样条进行高效的环境建模。其创新点可总结如下：

◆ 提出了一种使用连续B样条紧凑表示环境的新框架，替代了传统依赖密集几何地图的方法。
◆ 该框架能够维护并推理连续概率密度函数，从而支持完整的概率推理过程。
◆ 巧妙利用了B样条在插值与拟合方面的优势，将原本离散稀疏的环境数据转化为连续表达。
◆ 在自动驾驶赛车场景中验证了框架有效性，通过利用预知的赛道先验结构知识，极大压缩了环境表征的数据量。
◆ 实验表明，该方法在精度与基于传统路标的方法持平时，能显著减少系统所需的计算资源，实现了精度与效率的良好平衡。</td></tr>
<tr><td>2026-03-02</td><td>&quot;Game, Set, Match&quot;: Double Delight Watching a Grand Slam Tennis Match</td><td>[2603.02360](http://arxiv.org/pdf/2603.02360)</td><td>该论文在经典独立同分布概率模型下，系统分析了网球独特计分体系的概率特性。其核心贡献与创新点如下：

◆ 系统推导并比较了网球比赛（包括局、抢七、盘和整场比赛）与标准“K局胜”赛制的获胜概率计算公式，揭示了网球计分结构的数学本质。

◆ 提出将网球比赛这一复杂概率问题分解为局、盘等子问题的递推求解方法，展示了如何运用全概率定理等工具解决多层嵌套的随机过程问题。

◆ 将网球计分体系确立为概率论教学的优秀案例，特别适用于讲解全概率定理以及均值、方差和协方差的迭代计算规则。

◆ 探讨了网球比赛中的关键理论问题，例如在模型假设下比赛是否必然结束，以及发球方是否在实力相当时具有优势，并给出了解答。

◆ 将网球赛制视为一种统计决策系统，评估其通过有限分数判别“更强球员”的可靠性，并比较不同赛制在正确决策概率与比赛时长（分数数量）之间的权衡。</td></tr>
<tr><td>2026-03-02</td><td>LEAR: Learning Edge-Aware Representations for Event-to-LiDAR Localization</td><td>[2603.01839](http://arxiv.org/pdf/2603.01839)</td><td>该论文提出LEAR框架，解决事件相机与LiDAR点云在GPS失效和视觉退化环境中的定位难题。其核心贡献是通过双任务协同学习，实现跨模态的鲁棒定位。

◆ 提出双任务学习框架，联合估计边缘结构和稠密事件-深度光流场，直接桥接事件与LiDAR的模态差异。
◆ 创新性地将边缘信息作为核心线索而非后处理工具，通过跨模态融合机制将模态不变的几何线索注入运动表示中。
◆ 设计迭代优化策略，在多个更新步骤中强制边缘估计与光流估计任务相互一致，实现两者的协同增强。
◆ 最终生成具有边缘感知和深度对齐的光流场，通过PnP求解实现更鲁棒、准确的姿态估计。
◆ 在多个挑战性数据集上性能超越现有最佳方法，并公开了代码、模型与演示视频。</td></tr>
<tr><td>2026-03-01</td><td>riMESA: Consensus ADMM for Real-World Collaborative SLAM</td><td>[2603.01178](http://arxiv.org/pdf/2603.01178)</td><td>该论文提出了riMESA算法，旨在解决现实世界中多机器人协同SLAM的实际部署难题。其核心贡献在于构建了一个鲁棒、增量且分布式的协同SLAM后端优化框架。具体创新点如下：

◆ 提出了基于共识ADMM的分布式优化框架，为机器人领域的分布式任务提供了灵活、精确且快速收敛的理论基础。
◆ 设计了鲁棒的增量式求解方法，能够实时处理在线操作，适应动态环境与连续数据流。
◆ 系统具备强抗干扰能力，能有效处理异常观测值，确保在存在错误数据时的估计可靠性。
◆ 算法对通信限制具有高容错性，即使在通信受限或不稳定的网络条件下也能可靠运行。
◆ 在真实数据集上的实验表明，其估计精度显著优于先前方法，提升超过7倍，且能泛化于多种场景与网络条件。</td></tr>
<tr><td>2026-03-01</td><td>AI-enhanced Direct SLAM: A Principled Approach to Unsupervised Learning in Bayesian Inference</td><td>[2603.01071](http://arxiv.org/pdf/2603.01071)</td><td>本文提出了一种人工智能增强的混合SLAM方法，其核心贡献在于实现了在贝叶斯推理框架下，直接利用原始射频信号进行无监督学习与同步定位与建图。其创新点可总结如下：

◆ 提出了一种新颖的混合建模方法，将具有物理可解释性的视距信号模型与捕获多径分量统计特性的人工智能模型相结合。

◆ 在此基础上，构建了一种基于因子图的粒子和积算法，能够联合估计移动终端状态、可见性、多径参数和噪声方差。

◆ 设计了一个变分推理框架，通过最大化证据下界，直接从测量数据中以无监督方式学习神经网络的参数化表示，从而学习生成性的、与环境相关的信号模型。

◆ 提出了一种高效的基于GPU的实现方案，支持跨粒子和基站的并行似然评估，显著提升了计算效率。

◆ 仿真结果表明，该方法能在多径环境中无监督地学习信号模型，并在非视距场景下精确定位移动终端，有效利用所学环境地图。</td></tr>
<tr><td>2026-02-26</td><td>FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time</td><td>[2602.23115](http://arxiv.org/pdf/2602.23115)</td><td>该论文针对已知旋转下的相机航向估计问题，提出了一种新颖高效的实时求解方法。其核心贡献与创新点如下：

◆ 提出一种在单位球面（S²）上广义霍夫变换的新方法，用于从单目视频中稳健估计相机航向。
◆ 创新地采用斐波那契晶格对单位球面进行离散化，以此作为投票箱中心，实现了高效且均匀的球面采样。
◆ 通过将每对特征匹配点生成的兼容大圆，对一定范围内的方向进行投票，使正确运动方向能获得噪声和动态物体影响较小的特征的一致性投票。
◆ 该方法在噪声和异常值增加的情况下，仍能保持较高的估计精度，同时满足实时计算效率要求。
◆ 实验证明，该方法在精度与效率的权衡中处于帕累托前沿，并且通过在校准位姿初始化时修正航向，有效降低了SLAM系统的轨迹误差。</td></tr>
<tr><td>2026-02-26</td><td>Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring</td><td>[2602.22731](http://arxiv.org/pdf/2602.22731)</td><td>该论文的核心贡献是提出了一套名为Sapling-NeRF的集成系统，用于对森林中的幼树进行高精度、可重复且地理定位的三维重建与生态监测。

◆ 提出了一种创新的三级表征与重建流程，融合了NeRF、激光雷达SLAM和GNSS技术，解决了单一技术各自的局限性。
◆ 实现了场景的真实尺度恢复与精确地理定位，使长期、可重复的定量生态监测成为可能，这是现有隐式神经重建方法（如NeRF）未能实现的。
◆ 能够精细重建幼树的细枝、茂密叶片等细微结构，并量化茎干高度、分枝模式和叶木比等性状，其精度超越了传统的陆地激光扫描等方法。
◆ 通过以对象为中心的NeRF密集重建，为生态学家提供了更丰富的幼树结构与定量数据，用于分析森林动态。</td></tr>
<tr><td>2026-02-27</td><td>Parallel Continuous-Time Relative Localization with Augmented Clamped Non-Uniform B-Splines</td><td>[2602.22006](http://arxiv.org/pdf/2602.22006)</td><td>本文提出了一种名为CT-RIO的新型连续时间相对惯性里程计框架，旨在解决多机器人系统中高精度、低延迟的相对定位问题。其核心贡献与创新点如下：

◆ 首次将钳位非均匀B样条（C-NUBS）应用于机器人状态表示，从根本上消除了传统非钳位样条固有的查询时间延迟问题。
◆ 为C-NUBS设计了具有闭式解的样条扩展与收缩操作，该操作能保持样条形状，从而实现了适用于在线估计的灵活节点管理。
◆ 基于上述灵活性，提出了“节点-关键节点”策略，能够支持高频下的样条扩展，同时保留稀疏的关键节点以用于自适应的相对运动建模。
◆ 构建了一个纯基于相对运动学与机器人间约束的滑动窗口相对定位问题，避免了全局状态的依赖。
◆ 为满足大规模集群的计算需求，将紧耦合优化问题分解为按机器人划分的子问题，并采用增量式异步块坐标下降法进行并行求解，显著提升了计算效率。</td></tr>
<tr><td>2026-02-25</td><td>Dream-SLAM: Dreaming the Unseen for Active SLAM in Dynamic Environments</td><td>[2602.21967](http://arxiv.org/pdf/2602.21967)</td><td>Dream-SLAM提出了一种新颖的单目主动SLAM方法，旨在解决动态环境中传统方法存在的三大局限。其核心贡献在于通过“梦境”生成与融合机制，提升了系统的感知与规划能力。

◆ 创新性地引入“梦境”机制，生成动态环境中未观测区域的跨时空图像和语义结构，以补全不完整信息。
◆ 将梦境生成的数据与真实观测相融合，有效降低了噪声影响，从而提高了相机位姿估计的精度和三维场景重建的连贯性。
◆ 综合利用梦境与观测得到的场景结构进行长视距规划，生成具有远见的机器人运动轨迹，实现了更高效、更彻底的自主探索。
◆ 整体框架显著提升了在动态场景下的性能，在定位精度、建图质量和探索效率上均优于现有先进方法。</td></tr>
<tr><td>2026-02-27</td><td>DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling</td><td>[2602.21644](http://arxiv.org/pdf/2602.21644)</td><td>DAGS-SLAM的核心贡献是提出了一种面向移动部署的动态感知3D高斯溅射SLAM系统，在动态环境中实现了鲁棒的实时定位与重建。其创新点如下：

◆ 引入了时空运动概率（MP）状态，为每个3D高斯点维护一个动态概率估计，替代了传统上依赖逐帧分割或计算量大的光流方法。
◆ 设计了一个不确定性感知的调度器，根据系统跟踪的不确定性按需触发语义分割（如YOLO），显著减少了计算开销，提升了移动端的实用性。
◆ 提出了一种融合策略，将轻量级实例语义先验与几何线索相结合，以估计并随时间更新MP状态，增强了动态判断的鲁棒性，特别是在光照挑战下。
◆ 将MP状态传播至系统前端，用于动态感知的特征对应点选择，从而提升了跟踪的鲁棒性。
◆ 在后端优化中，利用MP引导的优化来抑制动态伪影，改善了静态场景的重建质量。实验表明，该系统在公开动态RGB-D数据集上实现了更好的重建与跟踪，同时在消费级GPU上保持了实时吞吐。</td></tr>
<tr><td>2026-02-24</td><td>LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments</td><td>[2602.20925](http://arxiv.org/pdf/2602.20925)</td><td>该论文提出了LST-SLAM，一个面向公里级动态环境的大规模立体热成像SLAM系统，旨在解决热成像SLAM在特征提取、运动跟踪和全局建图方面的核心难题。其核心贡献与创新点如下：

◆ 提出了一种自监督的热成像特征学习方法，提升了在复杂光照和天气条件下特征提取的可靠性。
◆ 设计了立体双级运动跟踪策略，结合了直接法与特征法，增强了运动估计的稳定性。
◆ 引入了语义-几何混合约束，通过抑制帧间几何一致性弱的潜在动态特征，有效应对动态环境的干扰。
◆ 开发了在线增量词袋模型用于回环检测，并结合全局位姿优化，显著减少了长距离运行中的累积漂移。
◆ 在公里级动态热成像数据集上的实验表明，该系统在鲁棒性和精度上均显著优于现有的代表性SLAM方案。</td></tr>
<tr><td>2026-02-24</td><td>RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction</td><td>[2602.20807](http://arxiv.org/pdf/2602.20807)</td><td>该论文提出了RU4D-SLAM框架，旨在解决动态环境中SLAM与场景重建的难题，其核心贡献是将3D高斯泼溅与SLAM结合，并扩展至4D（时空）动态场景重建。主要创新点如下：

◆ 首次将4D高斯泼溅引入SLAM系统，在空间3D表示中融入时间维度，实现了动态场景的连续重建。
◆ 集成了运动模糊渲染机制，增强了动态场景的表征能力，并改进了对模糊图像的感知。
◆ 扩展了逐像素不确定性建模方法，使其从静态场景适配到模糊图像处理，提升了不确定性感知的跟踪鲁棒性。
◆ 提出了一种语义引导的重新加权机制，用于动态场景中逐像素不确定性估计，优化了对场景变化的感知。
◆ 引入了可学习的不透明度权重，支持自适应的4D地图构建，提高了重建的灵活性与准确性。

实验表明，该方法在轨迹精度和4D场景重建质量上显著优于现有技术，尤其在包含运动物体和低质量输入的动态环境中表现突出。</td></tr>
<tr><td>2026-02-25</td><td>From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection</td><td>[2602.20630](http://arxiv.org/pdf/2602.20630)</td><td>该论文的核心贡献是提出了一种基于强化学习的全新关键点检测框架，旨在直接优化关键点在图像序列中的长期跟踪能力。

◆ 将关键点检测重新定义为序列决策问题，突破了传统基于图像对训练的范式。
◆ 提出了名为TraqPoint的端到端强化学习框架，可直接在图像序列上训练。
◆ 设计了创新的轨迹感知奖励机制，该机制联合优化关键点在多视角下的一致性和独特性。
◆ 通过策略梯度方法进行优化，使关键点能在严峻的视角和光照变化下保持长期可跟踪性。
◆ 在相对姿态估计和三维重建等稀疏匹配任务上的实验表明，其性能显著优于多种先进方法。</td></tr>
<tr><td>2026-02-21</td><td>Enhancing 3D LiDAR Segmentation by Shaping Dense and Accurate 2D Semantic Predictions</td><td>[2602.18869](http://arxiv.org/pdf/2602.18869)</td><td>该论文的核心贡献是提出了一种名为MM2D3D的多模态模型，通过生成密集且准确的二维语义预测，来增强三维激光雷达点云的分割性能。其创新点主要在于解决了投影过程中因数据稀疏性导致的二维预测不准和稀疏问题，从而提升了最终三维分割的精度。

◆ 提出多模态分割模型MM2D3D，利用相机图像作为辅助数据来增强仅基于激光雷达的二维预测。
◆ 引入跨模态引导滤波技术，利用从相机图像中提取的密集语义关系来约束中间二维预测，从而克服三维标签投影到二维时产生的标签稀疏性问题。
◆ 提出动态交叉伪监督方法，鼓励激光雷达的二维预测模仿相机图像预测的密集分布特性，以克服激光雷达点云投影本身的稀疏性。
◆ 通过上述技术，模型能够获得分布更密集、精度更高的中间二维语义预测，并最终有效提升了三维点云语义分割的准确性。
◆ 实验证明，该方法在二维和三维分割任务上均优于先前方法，实现了更优的性能。</td></tr>
<tr><td>2026-02-21</td><td>IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping</td><td>[2602.18709](http://arxiv.org/pdf/2602.18709)</td><td>IRIS-SLAM的核心贡献在于提出了一种利用统一几何-实例表征的鲁棒语义SLAM系统，有效融合了几何重建与语义理解。其创新点具体如下：

◆ 提出了一种新颖的统一几何-实例表征，通过扩展一个几何基础模型，使其能同时预测稠密几何信息和具有跨视角一致性的实例嵌入。
◆ 设计了一种语义协同关联机制，利用上述统一表征，解决了传统语义SLAM中数据关联脆弱的问题。
◆ 引入了实例引导的回环检测方法，利用对视角不敏感的语义锚点，显著提升了宽基线场景下的回环检测可靠性。
◆ 构建了一个紧密耦合的系统架构，弥合了几何重建与开放词汇语义地图生成之间的鸿沟，避免了传统方案中两者解耦的弊端。
◆ 通过实验验证，该系统在场景地图一致性和鲁棒性方面显著优于现有先进方法。</td></tr>
<tr><td>2026-02-20</td><td>Have We Mastered Scale in Deep Monocular Visual SLAM? The ScaleMaster Dataset and Benchmark</td><td>[2602.18174](http://arxiv.org/pdf/2602.18174)</td><td>该论文的核心贡献在于首次系统性地揭示并评估了深度单目视觉SLAM系统在大规模室内场景中面临的尺度一致性问题，并为此建立了专门的基准。

◆ 提出了首个专注于评估大规模室内环境下尺度一致性的数据集ScaleMaster，其特点包含多层结构、长轨迹、重复视角和低纹理区域等挑战性场景。
◆ 系统分析了当前先进的深度单目视觉SLAM系统对尺度不一致性的脆弱性，提供了定量与定性评估。
◆ 将评估维度从传统的轨迹精度，拓展到直接的地图间质量评估，引入了如基于高精度3D真值的Chamfer距离等度量标准。
◆ 研究发现，尽管现有系统在传统基准上表现良好，但在真实大规模室内环境中会出现严重的尺度相关故障，指出了当前技术的局限。
◆ 通过公开发布数据集和基线结果，为未来开发具有尺度一致性的可靠视觉SLAM系统奠定了研究基础。</td></tr>
<tr><td>2026-02-20</td><td>GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation</td><td>[2602.18164](http://arxiv.org/pdf/2602.18164)</td><td>该论文的核心贡献是发布了首个大规模、多模态、真实复杂环境下的足式机器人开源数据集GrandTour，旨在解决该领域长期缺乏可用于算法开发与评测的综合性数据的问题。

◆ 首创性：推出了当前最大规模的开源足式机器人数据集，填补了该领域缺乏覆盖真实复杂场景的公开数据集的空白。
◆ 环境多样性：数据采集跨越高山、森林、废墟、城市等多种极具挑战性的室内外环境，并涵盖了不同尺度、复杂度、光照及天气条件。
◆ 传感器配置丰富：提供了ANYmal-D四足机器人搭载的多种时间同步传感器数据，包括旋转激光雷达、多个特性互补的RGB相机、本体感知传感器和立体深度相机。
◆ 高精度真值：通过卫星RTK-GNSS和徕卡全站仪提供了高精度的地面真实轨迹，为算法评估提供了可靠基准。
◆ 应用支持广泛：该数据集旨在支持同步定位与建图、高精度状态估计、多模态学习及传感器融合等多个关键研究方向的发展与严格评测。</td></tr>
<tr><td>2026-02-19</td><td>Multi-session Localization and Mapping Exploiting Topological Information</td><td>[2602.17226](http://arxiv.org/pdf/2602.17226)</td><td>该论文提出了一种新颖的多会话SLAM框架，其核心贡献在于通过利用拓扑信息来提升在重复访问环境中的建图与定位的效率和一致性。

◆ 提出了一个基于地图定位的多会话框架，区别于常见的贪婪式运行完整SLAM会话再匹配地图的做法，实现了更高效的会话间数据融合。
◆ 设计了一种结合拓扑信息和不确定性感知的决策机制，通过分析位姿图结构来智能识别低连通性区域。
◆ 基于上述决策，系统能够选择性地触发局部建图和闭环检测模块，避免了不必要的全局计算，提升了系统效率。
◆ 实现了新会话数据与现有地图模型的无缝集成，有效减少了累积误差，显著增强了全局地图的一致性。
论文在数据集重叠序列和真实矿井类环境中验证了该方法的有效性。</td></tr>
<tr><td>2026-02-19</td><td>NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting</td><td>[2602.17182](http://arxiv.org/pdf/2602.17182)</td><td>该论文提出了一种用于内窥镜场景的单目非刚性SLAM系统NRGS-SLAM，其核心贡献在于利用可变形感知的3D高斯溅射技术，有效解决了组织形变带来的运动估计模糊问题，显著提升了相机跟踪与场景重建的质量。

◆ 提出了一种可变形感知的3D高斯地图表示，为每个高斯基元引入可学习的形变概率，无需外部标注，通过贝叶斯自监督策略进行优化，从而有效解耦相机自身运动与场景形变。
◆ 设计了一个可变形跟踪模块，采用由粗到细的位姿估计策略，优先利用低形变区域进行鲁棒跟踪，随后高效更新每帧的形变场。
◆ 开发了一个可变形建图模块，能够渐进式扩展和优化地图，在保证高保真度重建的同时，平衡了表示能力与计算效率。
◆ 引入了一个统一的鲁棒几何损失函数，融合外部几何先验知识，以缓解单目非刚性SLAM固有的病态性问题。
◆ 在多个公开内窥镜数据集上的实验表明，该系统在相机位姿估计精度（均方根误差降低高达50%）和照片级真实感重建质量上均优于现有先进方法。</td></tr>
<tr><td>2026-02-19</td><td>Cholec80-port: A Geometrically Consistent Trocar Port Segmentation Dataset for Robust Surgical Scene Understanding</td><td>[2602.17060](http://arxiv.org/pdf/2602.17060)</td><td>该论文的核心贡献是创建了一个用于提升手术场景几何理解鲁棒性的、几何一致性的套管端口分割数据集。

◆ 首次提出了“几何一致性”的套管端口分割标注标准，明确要求标注时排除中央管腔（开口）区域，即使能看到内部解剖结构也予以保留，这纠正了现有标注中的常见错误。
◆ 基于广泛使用的Cholec80数据集，构建了高质量、高保真度的专用套管端口分割数据集Cholec80-port，并制定了详细的标注标准操作程序。
◆ 按照同一标准，对现有多个公共手术数据集中的端口标注进行了清理和统一，形成了一个更一致、可比的基准。
◆ 通过实验证明，采用这种几何一致的标注方法，能显著提升模型在不同数据集间的泛化能力和鲁棒性，其效果超越了单纯增加数据集规模所带来的提升。这直接有益于依赖几何一致性的下游任务，如图像拼接和三维重建。</td></tr>
<tr><td>2026-02-18</td><td>Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM</td><td>[2602.16308](http://arxiv.org/pdf/2602.16308)</td><td>本文针对多智能体SLAM中数据关联困难的问题，提出了一种创新的解决方案。其核心贡献在于利用基于深度学习的6D姿态估计技术，实现了无需人工标记的机器人相互检测与定位，从而提升多机器人系统的协同建图与定位能力。

◆ 首次将无标记（markerless）的6D姿态估计方法集成到去中心化多机器人SLAM系统中，替代了传统的依赖标定标记阵列（如AprilTag）的方法。
◆ 通过直接相互观测来连接各机器人的局部SLAM图，有效解决了因视角差异大或环境相似性导致的回环检测失败问题。
◆ 克服了传统标记方法在观测距离、强光反射或曝光过度等恶劣光照条件下的局限性，增强了系统的鲁棒性和适用场景。
◆ 在行星类比环境的外场测试数据上进行了实验验证，证明了该方法能有效提高机器人团队间的相对定位精度。
◆ 整体上为多智能体SLAM提供了一种更灵活、更可靠的数据关联与系统关联途径。</td></tr>
<tr><td>2026-02-13</td><td>Adaptive Illumination Control for Robot Perception</td><td>[2602.15900](http://arxiv.org/pdf/2602.15900)</td><td>该论文的核心贡献是提出了一个名为Lightning的闭环照明控制框架，旨在通过主动调控机器人自带的程序化光源来直接改善视觉SLAM在弱光或高动态范围场景下的感知鲁棒性。

◆ 提出了一个共置照明分解模型，能够将观测图像分解为环境光成分和可控光源贡献场，从而无需重复实地采集就能物理一致地合成不同光照强度的训练数据。
◆ 基于合成数据，构建了一个离线最优照明强度调度问题，该问题能在SLAM图像效用、功耗与光照时序平滑性之间进行权衡优化。
◆ 通过行为克隆将离线优化方案蒸馏为一个可实时运行的照明控制策略，该策略能在线运行于移动机器人，并泛化至训练分布之外的场景。
◆ 整个框架实现了从感知建模、离线优化到在线策略的完整闭环，首次系统性地将主动照明控制与视觉SLAM相结合。
◆ 实验表明，该框架显著提升了SLAM轨迹的鲁棒性，同时避免了不必要的照明功耗。</td></tr>
<tr><td>2026-02-19</td><td>SceneVGGT: VGGT-based online 3D semantic SLAM for indoor scene understanding and navigation</td><td>[2602.15899](http://arxiv.org/pdf/2602.15899)</td><td>本文提出SceneVGGT，一个用于室内场景理解与导航的在线3D语义SLAM系统。其核心贡献在于构建了一个高效、鲁棒的时空语义理解框架，将SLAM与语义建图紧密结合以支持自主与辅助导航。

◆ 提出基于VGGT的滑动窗口处理流程，有效处理长视频序列，解决了内存与计算效率随序列增长而飙升的难题。
◆ 通过相机位姿变换对齐局部子地图，在保证几何一致性的同时，实现了内存占用低且速度高效的全局地图构建。
◆ 利用VGGT的跟踪头将2D实例掩码提升至3D物体，并维持其跨帧的时序一致性身份ID，从而支持场景变化检测等高级任务。
◆ 系统作为一个集成化概念验证，可将语义物体位置投影至估计的地面平面，直接为辅助导航（如音频反馈）提供支持。
◆ 整个流程GPU内存占用稳定在17GB以下，不受输入序列长度影响，并在ScanNet++基准测试中取得了有竞争力的点云性能。</td></tr>
<tr><td>2026-02-14</td><td>High-fidelity 3D reconstruction for planetary exploration</td><td>[2602.13909](http://arxiv.org/pdf/2602.13909)</td><td>该论文的核心贡献是提出了一种面向行星探测的高保真三维重建新流程，旨在解决传统方法在行星环境下遇到的挑战。

◆ 创新性地将辐射场方法（NeRF与高斯泼溅）集成到行星机器人自动化重建流程中，以生成兼具几何一致性与辐射度量细节的密集三维模型。
◆ 开发了一个统一系统，结合了Nerfstudio与COLMAP框架，并构建了与ROS2兼容的工作流，可直接处理来自rosbag记录的原始探测车数据。
◆ 该系统能够在典型行星环境（如非结构化、低纹理地形）中，仅依靠最小化的视觉输入，高效生成高保真、照片级真实感且度量一致的三维场景表达。
◆ 该流程为基于辐射场的测绘研究奠定了基础，弥合了行星探测中传统几何方法与神经表征之间的鸿沟，从而提升自主系统的环境感知与任务规划能力。</td></tr>
<tr><td>2026-02-13</td><td>HoRAMA: Holistic Reconstruction with Automated Material Assignment for Ray Tracing using NYURay</td><td>[2602.12942](http://arxiv.org/pdf/2602.12942)</td><td>该论文提出了HoRAMA方法，旨在为无线射线追踪（RT）信道预测自动生成高保真且兼容的三维环境模型。其核心贡献与创新点如下：

◆ 提出了一种从普通RGB视频（如智能手机拍摄）自动生成射线追踪兼容三维模型的全流程方法（HoRAMA），解决了传统手动建模耗时过长（数月级）的瓶颈。
◆ 创新性地将稠密点云重建（MASt3R-SLAM）与视觉语言模型辅助的材料属性自动赋值相结合，在生成精确几何的同时，补全了传统视觉重建方法缺失的关键材料电磁特性。
◆ 通过在实际工厂场景中，将基于HoRAMA生成模型的射线追踪预测结果与现场测量数据及基于人工模型的结果对比，验证了其有效性。其预测误差（2.28 dB RMSE）与人工模型基线（2.18 dB）相当。
◆ 该方法将大规模环境的三维重建时间从两个月大幅缩短至约16小时，为实现可扩展的无线数字孪生（用于5G/6G网络规划、部署与管理）提供了关键技术支持。</td></tr>
<tr><td>2026-02-13</td><td>Unbiased Gradient Estimation for Event Binning via Functional Backpropagation</td><td>[2602.12590](http://arxiv.org/pdf/2602.12590)</td><td>该论文针对事件相机数据处理的梯度估计难题提出了创新解决方案。事件数据通常需分箱为离散帧以适配传统图像算法，但分箱函数的不连续性会导致梯度截断或估计偏差，限制学习效率。

◆ 提出了一种新颖的无偏梯度估计框架，通过在前向传播保持输出不变的同时，在反向传播中合成弱导数来克服分箱函数的不连续性。
◆ 核心创新是利用分部积分思想，将目标函数提升为泛函，使得反向传播中分箱函数的导数呈现积分形式，并自然引出余切函数。
◆ 通过从采样的余切向量重构余切函数，计算出的弱导数被证明能够匹配平滑与非平滑目标的长程有限差分，确保了理论可靠性。
◆ 实验验证了方法的广泛有效性：在优化型自运动估计中降低了误差并加速收敛；在自监督光流和SLAM等复杂任务中显著提升了性能，展示了其在事件视觉感知中的普遍优势。</td></tr>
<tr><td>2026-02-12</td><td>GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry</td><td>[2602.11714](http://arxiv.org/pdf/2602.11714)</td><td>GSO-SLAM的核心贡献在于提出了一种新颖的双向耦合单目密集SLAM系统，它通过创新的方式整合了视觉里程计与高斯泼溅场景表示，实现了实时、高精度的建图与跟踪。

◆ 核心创新是提出了视觉里程计与高斯泼溅表示之间的双向耦合机制，克服了现有方法在统一场景表示下计算成本高，或松散集成导致冗余的缺点。
◆ 该方法在期望最大化框架内制定了联合优化，能够同步优化视觉里程计导出的半稠密深度估计与高斯场景表示，且不引入额外计算开销。
◆ 提出了高斯泼溅初始化技术，直接利用视觉里程计提供的图像信息、关键帧位姿和像素关联来生成接近最终效果的高斯场景初始近似，从而避免了传统启发式方法的依赖。
◆ 整个系统能够在实时运行的同时，在场景重建的几何/光度保真度以及跟踪精度方面达到先进水平，并通过大量实验验证了其有效性。</td></tr>
<tr><td>2026-02-09</td><td>Thegra: Graph-based SLAM for Thermal Imagery</td><td>[2602.08531](http://arxiv.org/pdf/2602.08531)</td><td>该论文的核心贡献是提出了一种适用于热成像的、基于图优化的单目SLAM系统，旨在解决热图像纹理少、对比度低和噪声高导致的特征提取与匹配难题。

◆ 创新性地将在大规模可见光谱数据上训练的通用学习特征（SuperPoint检测器与LightGlue匹配器）应用于热成像SLAM，提升了系统的跨域泛化能力。
◆ 设计了一个针对热图像的前处理流程，以增强图像质量，使其更适合后续特征提取。
◆ 改进了核心SLAM模块，使其能够有效处理热图像中稀疏且易产生异常值的特征匹配。
◆ 将SuperPoint生成的关键点置信度分数融入一个置信度加权的因子图优化框架，提高了位姿估计的鲁棒性。
◆ 整个系统在公开热数据集上验证有效，且无需针对特定数据集进行训练或微调特征检测器，缓解了高质量热数据稀缺的制约。</td></tr>
<tr><td>2026-02-09</td><td>Chamelion: Reliable Change Detection for Long-Term LiDAR Mapping in Transient Environments</td><td>[2602.08189](http://arxiv.org/pdf/2602.08189)</td><td>该论文针对动态变化环境（如建筑工地、室内空间）中的长期激光雷达建图，提出了一种可靠的在线变化检测方法Chamelion。其核心贡献与创新点如下：

◆ 提出了一种双分支网络架构，专门用于在线变化检测与长期地图维护，解决了现有方法在瞬变环境中难以检测变化并更新地图的难题。
◆ 针对真实世界数据难以采集和对齐的瓶颈，创新性地开发了一种数据增强策略，通过从不同场景导入元素来合成结构变化，从而无需大量人工标注即可有效训练模型。
◆ 所提出的方法在真实建筑工地和室内办公环境等多种场景中进行了验证，表现出良好的泛化能力，能够实现高效且准确的地图更新。</td></tr>
<tr><td>2026-02-11</td><td>Thermal odometry and dense mapping using learned odometry and Gaussian splatting</td><td>[2602.07493](http://arxiv.org/pdf/2602.07493)</td><td>本文提出了一种名为TOM-GS的新型热成像里程计与稠密建图方法，其核心贡献在于首次将基于学习的方法与高斯泼溅技术相结合，以解决热成像在恶劣环境下的鲁棒感知与稠密重建问题。

◆ 提出了首个专为热成像相机设计的高斯泼溅SLAM系统（TOM-GS），将基于学习的里程计与基于高斯泼溅的稠密建图相集成。
◆ 设计了专门的热图像增强模块，以优化热成像的输入质量，并集成了单目深度估计，以提升几何感知能力。
◆ 在运动估计和新视角渲染任务上进行了广泛实验，证明该方法优于现有的基于学习的方法，验证了学习框架在热成像里程计和稠密重建中的优势。
◆ 解决了现有几何方法在多样化数据集上易失效且无法生成稠密地图的局限性，利用高斯泼溅的高效性与高质量重建能力，实现了鲁棒的稠密环境建模。</td></tr>
<tr><td>2026-02-06</td><td>A Consistency-Improved LiDAR-Inertial Bundle Adjustment</td><td>[2602.06380](http://arxiv.org/pdf/2602.06380)</td><td>◆ Simultaneous Localization and Mapping (SLAM) using 3D LiDAR has emerged as a cornerstone for autonomous navigation in robotics.
◆ While feature-based SLAM systems have achieved impressive results by leveraging edge and planar structures, they often suffer from the inconsistent estimator associated with feature parameterization and estimated covariance.
◆ In this work, we present a consistency-improved LiDAR-inertial bundle adjustment (BA) with tailored parameterization and estimator.</td></tr>
<tr><td>2026-02-05</td><td>Geometric Observability Index: An Operator-Theoretic Framework for Per-Feature Sensitivity, Weak Observability, and Dynamic Effects in SE(3) Pose Estimation</td><td>[2602.05582](http://arxiv.org/pdf/2602.05582)</td><td>◆ We present a unified operator-theoretic framework for analyzing per-feature sensitivity in camera pose estimation on the Lie group SE(3).
◆ Classical sensitivity tools - conditioning analyses, Euclidean perturbation arguments, and Fisher information bounds - do not explain how individual image features influence the pose estimate, nor why dynamic or inconsistent observations can disproportionately distort modern SLAM and structure-from-motion systems.
◆ To address this gap, we extend influence function theory to matrix Lie groups and derive an intrinsic perturbation operator for left-trivialized M-estimators on SE(3).</td></tr>
<tr><td>2026-02-05</td><td>VGGT-Motion: Motion-Aware Calibration-Free Monocular SLAM for Long-Range Consistency</td><td>[2602.05508](http://arxiv.org/pdf/2602.05508)</td><td>◆ Despite recent progress in calibration-free monocular SLAM via 3D vision foundation models, scale drift remains severe on long sequences.
◆ Motion-agnostic partitioning breaks contextual coherence and causes zero-motion drift, while conventional geometric alignment is computationally expensive.
◆ To address these issues, we propose VGGT-Motion, a calibration-free SLAM system for efficient and robust global consistency over kilometer-scale trajectories.</td></tr>
<tr><td>2026-02-04</td><td>Towards Next-Generation SLAM: A Survey on 3DGS-SLAM Focusing on Performance, Robustness, and Future Directions</td><td>[2602.04251](http://arxiv.org/pdf/2602.04251)</td><td>◆ Traditional Simultaneous Localization and Mapping (SLAM) systems often face limitations including coarse rendering quality, insufficient recovery of scene details, and poor robustness in dynamic environments.
◆ 3D Gaussian Splatting (3DGS), with its efficient explicit representation and high-quality rendering capabilities, offers a new reconstruction paradigm for SLAM.
◆ This survey comprehensively reviews key technical approaches for integrating 3DGS with SLAM.</td></tr>
<tr><td>2026-02-03</td><td>Beyond the Vehicle: Cooperative Localization by Fusing Point Clouds for GPS-Challenged Urban Scenarios</td><td>[2602.03908](http://arxiv.org/pdf/2602.03908)</td><td>◆ Accurate vehicle localization is a critical challenge in urban environments where GPS signals are often unreliable.
◆ This paper presents a cooperative multi-sensor and multi-modal localization approach to address this issue by fusing data from vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) systems.
◆ Our approach integrates cooperative data with a point cloud registration-based simultaneous localization and mapping (SLAM) algorithm.</td></tr>
<tr><td>2026-02-02</td><td>Relationship-Aware Hierarchical 3D Scene Graph for Task Reasoning</td><td>[2602.02456](http://arxiv.org/pdf/2602.02456)</td><td>◆ Representing and understanding 3D environments in a structured manner is crucial for autonomous agents to navigate and reason about their surroundings.
◆ While traditional Simultaneous Localization and Mapping (SLAM) methods generate metric reconstructions and can be extended to metric-semantic mapping, they lack a higher level of abstraction and relational reasoning.
◆ To address this gap, 3D scene graphs have emerged as a powerful representation for capturing hierarchical structures and object relationships.</td></tr>
<tr><td>2026-02-02</td><td>3D Foundation Model-Based Loop Closing for Decentralized Collaborative SLAM</td><td>[2602.02430](http://arxiv.org/pdf/2602.02430)</td><td>◆ Decentralized Collaborative Simultaneous Localization And Mapping (C-SLAM) techniques often struggle to identify map overlaps due to significant viewpoint variations among robots.
◆ Motivated by recent advancements in 3D foundation models, which can register images despite large viewpoint differences, we propose a robust loop closing approach that leverages these models to establish inter-robot measurements.
◆ In contrast to resource-intensive methods requiring full 3D reconstruction within a centralized map, our approach integrates foundation models into existing SLAM pipelines, yielding scalable and robust multi-robot mapping.</td></tr>
<tr><td>2026-02-02</td><td>Mapping-Guided Task Discovery and Allocation for Robotic Inspection of Underwater Structures</td><td>[2602.02389](http://arxiv.org/pdf/2602.02389)</td><td>◆ Task generation for underwater multi-robot inspections without prior knowledge of existing geometry can be achieved and optimized through examination of simultaneous localization and mapping (SLAM) data.
◆ By considering hardware parameters and environmental conditions, a set of tasks is generated from SLAM meshes and optimized through expected keypoint scores and distance-based pruning.
◆ In-water tests are used to demonstrate the effectiveness of the algorithm and determine the appropriate parameters.</td></tr>
<tr><td>2026-02-02</td><td>Real-Time Loop Closure Detection in Visual SLAM via NetVLAD and Faiss</td><td>[2602.01673](http://arxiv.org/pdf/2602.01673)</td><td>◆ Loop closure detection (LCD) is a core component of simultaneous localization and mapping (SLAM): it identifies revisited places and enables pose-graph constraints that correct accumulated drift.
◆ Classic bag-of-words approaches such as DBoW are efficient but often degrade under appearance change and perceptual aliasing.
◆ In parallel, deep learning-based visual place recognition (VPR) descriptors (e.g., NetVLAD and Transformer-based models) offer stronger robustness, but their computational cost is often viewed as a barrier to real-time SLAM.</td></tr>
<tr><td>2026-01-29</td><td>IROS: A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation</td><td>[2601.21506](http://arxiv.org/pdf/2601.21506)</td><td>◆ Indoor mobile robot navigation requires fast responsiveness and robust semantic understanding, yet existing methods struggle to provide both.
◆ Classical geometric approaches such as SLAM offer reliable localization but depend on detailed maps and cannot interpret human-targeted cues (e.g., signs, room numbers) essential for indoor reasoning.
◆ Vision-Language-Action (VLA) models introduce semantic grounding but remain strictly reactive, basing decisions only on visible frames and failing to anticipate unseen intersections or reason about distant textual cues.</td></tr>
<tr><td>2026-01-28</td><td>Multi-Robot Decentralized Collaborative SLAM in Planetary Analogue Environments: Dataset, Challenges, and Lessons Learned</td><td>[2601.21063](http://arxiv.org/pdf/2601.21063)</td><td>◆ Decentralized collaborative simultaneous localization and mapping (C-SLAM) is essential to enable multirobot missions in unknown environments without relying on preexisting localization and communication infrastructure.
◆ This technology is anticipated to play a key role in the exploration of the Moon, Mars, and other planets.
◆ In this article, we share insights and lessons learned from C-SLAM experiments involving three robots operating on a Mars analogue terrain and communicating over an ad hoc network.</td></tr>
<tr><td>2026-01-30</td><td>VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction</td><td>[2601.19887](http://arxiv.org/pdf/2601.19887)</td><td>◆ We present VGGT-SLAM 2.0, a real time RGB feed-forward SLAM system which substantially improves upon VGGT-SLAM for incrementally aligning submaps created from VGGT.
◆ Firstly, we remove high-dimensional 15-degree-of-freedom drift and planar degeneracy from VGGT-SLAM by creating a new factor graph design while still addressing the reconstruction ambiguity of VGGT given unknown camera intrinsics.
◆ Secondly, by studying the attention layers of VGGT, we show that one of the layers is well suited to assist in image retrieval verification for free without additional training, which enables both rejecting false positive matches and allows for completing more loop closures.</td></tr>
<tr><td>2026-01-27</td><td>The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments</td><td>[2601.19557](http://arxiv.org/pdf/2601.19557)</td><td>◆ We release the S3LI Vulcano dataset, a multi-modal dataset towards development and benchmarking of Simultaneous Localization and Mapping (SLAM) and place recognition algorithms that rely on visual and LiDAR modalities.
◆ Several sequences are recorded on the volcanic island of Vulcano, from the Aeolian Islands in Sicily, Italy.
◆ The sequences provide users with data from a variety of environments, textures and terrains, including basaltic or iron-rich rocks, geological formations from old lava channels, as well as dry vegetation and water.</td></tr>
<tr><td>2026-01-28</td><td>Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction</td><td>[2601.19489](http://arxiv.org/pdf/2601.19489)</td><td>◆ We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge.
◆ The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate).
◆ To robustly handle these heterogeneous settings, we develop a two-stage solution.</td></tr>
<tr><td>2026-01-26</td><td>Co-PLNet: A Collaborative Point-Line Network for Prompt-Guided Wireframe Parsing</td><td>[2601.18252](http://arxiv.org/pdf/2601.18252)</td><td>◆ Wireframe parsing aims to recover line segments and their junctions to form a structured geometric representation useful for downstream tasks such as Simultaneous Localization and Mapping (SLAM).
◆ Existing methods predict lines and junctions separately and reconcile them post-hoc, causing mismatches and reduced robustness.
◆ We present Co-PLNet, a point-line collaborative framework that exchanges spatial cues between the two tasks, where early detections are converted into spatial prompts via a Point-Line Prompt Encoder (PLP-Encoder), which encodes geometric attributes into compact and spatially aligned maps.</td></tr>
<tr><td>2026-01-22</td><td>Keyframe-Based Feed-Forward Visual Odometry</td><td>[2601.16020](http://arxiv.org/pdf/2601.16020)</td><td>◆ The emergence of visual foundation models has revolutionized visual odometry~(VO) and SLAM, enabling pose estimation and dense reconstruction within a single feed-forward network.
◆ However, unlike traditional pipelines that leverage keyframe methods to enhance efficiency and accuracy, current foundation model based methods, such as VGGT-Long, typically process raw image sequences indiscriminately.
◆ This leads to computational redundancy and degraded performance caused by low inter-frame parallax, which provides limited contextual stereo information.</td></tr>
<tr><td>2026-01-24</td><td>Accurate Calibration and Robust LiDAR-Inertial Odometry for Spinning Actuated LiDAR Systems</td><td>[2601.15946](http://arxiv.org/pdf/2601.15946)</td><td>◆ Accurate calibration and robust localization are fundamental for downstream tasks in spinning actuated LiDAR applications.
◆ Existing methods, however, require parameterizing extrinsic parameters based on different mounting configurations, limiting their generalizability.
◆ Additionally, spinning actuated LiDAR inevitably scans featureless regions, which complicates the balance between scanning coverage and localization robustness.</td></tr>
<tr><td>2026-01-22</td><td>Parallelizable Riemannian Alternating Direction Method of Multipliers for Non-convex Pose Graph Optimization</td><td>[2601.15684](http://arxiv.org/pdf/2601.15684)</td><td>◆ Pose graph optimization (PGO) is fundamental to robot perception and navigation systems, serving as the mathematical backbone for solving simultaneous localization and mapping (SLAM).
◆ Existing solvers suffer from polynomial growth in computational complexity with graph size, hindering real-time deployment in large-scale scenarios.
◆ In this paper, by duplicating variables and introducing equality constraints, we reformulate the problem and propose a Parallelizable Riemannian Alternating Direction Method of Multipliers (PRADMM) to solve it efficiently.</td></tr>
<tr><td>2026-01-19</td><td>Autonomous Navigation at the Nano-Scale: Algorithms, Architectures, and Constraints</td><td>[2601.13252](http://arxiv.org/pdf/2601.13252)</td><td>◆ Autonomous navigation for nano-scale unmanned aerial vehicles (nano-UAVs) is governed by extreme Size, Weight, and Power (SWaP) constraints (with the weight &lt; 50 g and sub-100 mW onboard processor), distinguishing it fundamentally from standard robotic paradigms.
◆ This review synthesizes the state-of-the-art in sensing, computing, and control architectures designed specifically for these sub- 100mW computational envelopes.
◆ We critically analyse the transition from classical geometry-based methods to emerging &quot;Edge AI&quot; paradigms, including quantized deep neural networks deployed on ultra-low-power System-on-Chips (SoCs) and neuromorphic event-based control.</td></tr>
<tr><td>2026-01-18</td><td>R-VoxelMap: Accurate Voxel Mapping with Recursive Plane Fitting for Online LiDAR Odometry</td><td>[2601.12377](http://arxiv.org/pdf/2601.12377)</td><td>◆ This paper proposes R-VoxelMap, a novel voxel mapping method that constructs accurate voxel maps using a geometry-driven recursive plane fitting strategy to enhance the localization accuracy of online LiDAR odometry.
◆ VoxelMap and its variants typically fit and check planes using all points in a voxel, which may lead to plane parameter deviation caused by outliers, over segmentation of large planes, and incorrect merging across different physical planes.
◆ To address these issues, R-VoxelMap utilizes a geometry-driven recursive construction strategy based on an outlier detect-and-reuse pipeline.</td></tr>
<tr><td>2026-01-10</td><td>PointSLAM++: Robust Dense Neural Gaussian Point Cloud-based SLAM</td><td>[2601.11617](http://arxiv.org/pdf/2601.11617)</td><td>◆ Real-time 3D reconstruction is crucial for robotics and augmented reality, yet current simultaneous localization and mapping(SLAM) approaches often struggle to maintain structural consistency and robust pose estimation in the presence of depth noise.
◆ This work introduces PointSLAM++, a novel RGB-D SLAM system that leverages a hierarchically constrained neural Gaussian representation to preserve structural relationships while generating Gaussian primitives for scene mapping.
◆ It also employs progressive pose optimization to mitigate depth sensor noise, significantly enhancing localization accuracy.</td></tr>
<tr><td>2026-01-16</td><td>ShapeR: Robust Conditional 3D Shape Generation from Casual Captures</td><td>[2601.11514](http://arxiv.org/pdf/2601.11514)</td><td>◆ Recent advances in 3D shape generation have achieved impressive results, but most existing methods rely on clean, unoccluded, and well-segmented inputs.
◆ Such conditions are rarely met in real-world scenarios.
◆ We present ShapeR, a novel approach for conditional 3D object shape generation from casually captured sequences.</td></tr>
<tr><td>2026-01-20</td><td>SurfSLAM: Sim-to-Real Underwater Stereo Reconstruction For Real-Time SLAM</td><td>[2601.10814](http://arxiv.org/pdf/2601.10814)</td><td>◆ Localization and mapping are core perceptual capabilities for underwater robots.
◆ Stereo cameras provide a low-cost means of directly estimating metric depth to support these tasks.
◆ However, despite recent advances in stereo depth estimation on land, computing depth from image pairs in underwater scenes remains challenging.</td></tr>
<tr><td>2026-01-14</td><td>SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings</td><td>[2601.09665](http://arxiv.org/pdf/2601.09665)</td><td>◆ Monocular visual SLAM enables 3D reconstruction from internet video and autonomous navigation on resource-constrained platforms, yet suffers from scale drift, i.e., the gradual divergence of estimated scale over long sequences.
◆ Existing frame-to-frame methods achieve real-time performance through local optimization but accumulate scale drift due to the lack of global constraints among independent windows.
◆ To address this, we propose SCE-SLAM, an end-to-end SLAM system that maintains scale consistency through scene coordinate embeddings, which are learned patch-level representations encoding 3D geometric relationships under a canonical scale reference.</td></tr>
<tr><td>2026-01-14</td><td>Multimodal Signal Processing For Thermo-Visible-Lidar Fusion In Real-time 3D Semantic Mapping</td><td>[2601.09578](http://arxiv.org/pdf/2601.09578)</td><td>◆ In complex environments, autonomous robot navigation and environmental perception pose higher requirements for SLAM technology.
◆ This paper presents a novel method for semantically enhancing 3D point cloud maps with thermal information.
◆ By first performing pixel-level fusion of visible and infrared images, the system projects real-time LiDAR point clouds onto this fused image stream.</td></tr>
<tr><td>2026-01-14</td><td>SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing</td><td>[2601.09385](http://arxiv.org/pdf/2601.09385)</td><td>◆ The recent surge in open-source Multimodal Large Language Models (MLLM) frameworks, such as LLaVA, provides a convenient kickoff for artificial intelligence developers and researchers.
◆ However, most of the MLLM frameworks take vision as the main input modality, and provide limited in-depth support for the modality of speech, audio, and music.
◆ This situation hinders the development of audio-language models, and forces researchers to spend a lot of effort on code writing and hyperparameter tuning.</td></tr>
<tr><td>2026-01-13</td><td>Thermo-LIO: A Novel Multi-Sensor Integrated System for Structural Health Monitoring</td><td>[2601.08977](http://arxiv.org/pdf/2601.08977)</td><td>◆ Traditional two-dimensional thermography, despite being non-invasive and useful for defect detection in the construction field, is limited in effectively assessing complex geometries, inaccessible areas, and subsurface defects.
◆ This paper introduces Thermo-LIO, a novel multi-sensor system that can enhance Structural Health Monitoring (SHM) by fusing thermal imaging with high-resolution LiDAR.
◆ To achieve this, the study first develops a multimodal fusion method combining thermal imaging and LiDAR, enabling precise calibration and synchronization of multimodal data streams to create accurate representations of temperature distributions in buildings.</td></tr>
<tr><td>2026-01-13</td><td>Efficient Incremental SLAM via Information-Guided and Selective Optimization</td><td>[2601.08110](http://arxiv.org/pdf/2601.08110)</td><td>◆ We present an efficient incremental SLAM back-end that achieves the accuracy of full batch optimization while substantially reducing computational cost.
◆ The proposed approach combines two complementary ideas: information-guided gating (IGG) and selective partial optimization (SPO).
◆ IGG employs an information-theoretic criterion based on the log-determinant of the information matrix to quantify the contribution of new measurements, triggering global optimization only when a significant information gain is observed.</td></tr>
<tr><td>2026-01-09</td><td>InsSo3D: Inertial Navigation System and 3D Sonar SLAM for turbid environment inspection</td><td>[2601.05805](http://arxiv.org/pdf/2601.05805)</td><td>◆ This paper presents InsSo3D, an accurate and efficient method for large-scale 3D Simultaneous Localisation and Mapping (SLAM) using a 3D Sonar and an Inertial Navigation System (INS).
◆ Unlike traditional sonar, which produces 2D images containing range and azimuth information but lacks elevation information, 3D Sonar produces a 3D point cloud, which therefore does not suffer from elevation ambiguity.
◆ We introduce a robust and modern SLAM framework adapted to the 3D Sonar data using INS as prior, detecting loop closure and performing pose graph optimisation.</td></tr>
<tr><td>2026-01-09</td><td>FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time</td><td>[2601.05738](http://arxiv.org/pdf/2601.05738)</td><td>◆ We present a real-time tracking SLAM system that unifies efficient camera tracking with photorealistic feature-enriched mapping using 3D Gaussian Splatting (3DGS).
◆ Our main contribution is integrating dense feature rasterization into the novel-view synthesis, aligned with a visual foundation model.
◆ This yields strong semantics, going beyond basic RGB-D input, aiding both tracking and mapping accuracy.</td></tr>
<tr><td>2026-01-08</td><td>UniLiPs: Unified LiDAR Pseudo-Labeling with Geometry-Grounded Dynamic Scene Decomposition</td><td>[2601.05105](http://arxiv.org/pdf/2601.05105)</td><td>◆ Unlabeled LiDAR logs, in autonomous driving applications, are inherently a gold mine of dense 3D geometry hiding in plain sight - yet they are almost useless without human labels, highlighting a dominant cost barrier for autonomous-perception research.
◆ In this work we tackle this bottleneck by leveraging temporal-geometric consistency across LiDAR sweeps to lift and fuse cues from text and 2D vision foundation models directly into 3D, without any manual input.
◆ We introduce an unsupervised multi-modal pseudo-labeling method relying on strong geometric priors learned from temporally accumulated LiDAR maps, alongside with a novel iterative update rule that enforces joint geometric-semantic consistency, and vice-versa detecting moving objects from inconsistencies.</td></tr>
<tr><td>2026-01-08</td><td>Discrete Fourier Transform-based Point Cloud Compression for Efficient SLAM in Featureless Terrain</td><td>[2601.04551](http://arxiv.org/pdf/2601.04551)</td><td>◆ Simultaneous Localization and Mapping (SLAM) is an essential technology for the efficiency and reliability of unmanned robotic exploration missions.
◆ While the onboard computational capability and communication bandwidth are critically limited, the point cloud data handled by SLAM is large in size, attracting attention to data compression methods.
◆ To address such a problem, in this paper, we propose a new method for compressing point cloud maps by exploiting the Discrete Fourier Transform (DFT).</td></tr>
<tr><td>2026-01-08</td><td>Fast Continuum Robot Shape and External Load State Estimation on SE(3)</td><td>[2601.04493](http://arxiv.org/pdf/2601.04493)</td><td>◆ Previous on-manifold approaches to continuum robot state estimation have typically adopted simplified Cosserat rod models, which cannot directly account for actuation inputs or external loads.
◆ We introduce a general framework that incorporates uncertainty models for actuation (e.g., tendon tensions), applied forces and moments, process noise, boundary conditions, and arbitrary backbone measurements.
◆ By adding temporal priors across time steps, our method additionally performs joint estimation in both the spatial (arclength) and temporal domains, enabling full \textit{spacetime} state estimation.</td></tr>
<tr><td>2026-01-06</td><td>Loop Closure using AnyLoc Visual Place Recognition in DPV-SLAM</td><td>[2601.02723](http://arxiv.org/pdf/2601.02723)</td><td>◆ Loop closure is crucial for maintaining the accuracy and consistency of visual SLAM.
◆ We propose a method to improve loop closure performance in DPV-SLAM.
◆ Our approach integrates AnyLoc, a learning-based visual place recognition technique, as a replacement for the classical Bag of Visual Words (BoVW) loop detection method.</td></tr>
<tr><td>2026-01-05</td><td>Differential Barometric Altimetry for Submeter Vertical Localization and Floor Recognition Indoors</td><td>[2601.02184](http://arxiv.org/pdf/2601.02184)</td><td>◆ Accurate altitude estimation and reliable floor recognition are critical for mobile robot localization and navigation within complex multi-storey environments.
◆ In this paper, we present a robust, low-cost vertical estimation framework leveraging differential barometric sensing integrated within a fully ROS-compliant software package.
◆ Our system simultaneously publishes real-time altitude data from both a stationary base station and a mobile sensor, enabling precise and drift-free vertical localization.</td></tr>
<tr><td>2026-01-03</td><td>VISO: Robust Underwater Visual-Inertial-Sonar SLAM with Photometric Rendering for Dense 3D Reconstruction</td><td>[2601.01144](http://arxiv.org/pdf/2601.01144)</td><td>◆ Visual challenges in underwater environments significantly hinder the accuracy of vision-based localisation and the high-fidelity dense reconstruction.
◆ In this paper, we propose VISO, a robust underwater SLAM system that fuses a stereo camera, an inertial measurement unit (IMU), and a 3D sonar to achieve accurate 6-DoF localisation and enable efficient dense 3D reconstruction with high photometric fidelity.
◆ We introduce a coarse-to-fine online calibration approach for extrinsic parameters estimation between the 3D sonar and the camera.</td></tr>
<tr><td>2025-12-28</td><td>RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization</td><td>[2601.00705](http://arxiv.org/pdf/2601.00705)</td><td>◆ We introduce RGS-SLAM, a robust Gaussian-splatting SLAM framework that replaces the residual-driven densification stage of GS-SLAM with a training-free correspondence-to-Gaussian initialization.
◆ Instead of progressively adding Gaussians as residuals reveal missing geometry, RGS-SLAM performs a one-shot triangulation of dense multi-view correspondences derived from DINOv3 descriptors refined through a confidence-aware inlier classifier, generating a well-distributed and structure-aware Gaussian seed prior to optimization.
◆ This initialization stabilizes early mapping and accelerates convergence by roughly 20\%, yielding higher rendering fidelity in texture-rich and cluttered scenes while remaining fully compatible with existing GS-SLAM pipelines.</td></tr>
<tr><td>2026-01-13</td><td>Variable Elimination in Hybrid Factor Graphs for Discrete-Continuous Inference &amp; Estimation</td><td>[2601.00545](http://arxiv.org/pdf/2601.00545)</td><td>◆ Many hybrid problems in robotics involve both continuous and discrete components, and modeling them together for estimation tasks has been a long standing and difficult problem.
◆ Hybrid Factor Graphs give us a mathematical framework to model these types of problems, however existing approaches for solving them are based on approximations.
◆ In this work, we propose an efficient Hybrid Factor Graph framework alongwith a variable elimination algorithm to produce a hybrid Bayes network, which can then be used for exact Maximum A Posteriori estimation and marginalization over both sets of variables.</td></tr>
<tr><td>2026-01-01</td><td>FoundationSLAM: Unleashing the Power of Depth Foundation Models for End-to-End Dense Visual SLAM</td><td>[2512.25008](http://arxiv.org/pdf/2512.25008)</td><td>◆ We present FoundationSLAM, a learning-based monocular dense SLAM system that addresses the absence of geometric consistency in previous flow-based approaches for accurate and robust tracking and mapping.
◆ Our core idea is to bridge flow estimation with geometric reasoning by leveraging the guidance from foundation depth models.
◆ To this end, we first develop a Hybrid Flow Network that produces geometry-aware correspondences, enabling consistent depth and pose inference across diverse keyframes.</td></tr>
<tr><td>2025-12-27</td><td>Mesquite MoCap: Democratizing Real-Time Motion Capture with Affordable, Bodyworn IoT Sensors and WebXR SLAM</td><td>[2512.22690](http://arxiv.org/pdf/2512.22690)</td><td>◆ Motion capture remains costly and complex to deploy, limiting use outside specialized laboratories.
◆ We present Mesquite, an open-source, low-cost inertial motion-capture system that combines a body-worn network of 15 IMU sensor nodes with a hip-worn Android smartphone for position tracking.
◆ A low-power wireless link streams quaternion orientations to a central USB dongle and a browser-based application for real-time visualization and recording.</td></tr>
<tr><td>2025-12-30</td><td>Simultaneous Source Separation, Synchronization, Localization and Mapping for 6G Systems</td><td>[2512.22393](http://arxiv.org/pdf/2512.22393)</td><td>◆ Multipath-based simultaneous localization and mapping (MP-SLAM) is a promising approach for future 6G networks to jointly estimate the positions of transmitters and receivers together with the propagation environment.
◆ In cooperative MP-SLAM, information collected by multiple mobile terminals (MTs) is fused to enhance accuracy and robustness.
◆ Existing methods, however, typically assume perfectly synchronized base stations (BSs) and orthogonal transmission sequences, rendering inter-BS interference at the MTs negligible.</td></tr>
<tr><td>2025-12-25</td><td>World-Coordinate Human Motion Retargeting via SAM 3D Body</td><td>[2512.21573](http://arxiv.org/pdf/2512.21573)</td><td>◆ Recovering world-coordinate human motion from monocular videos with humanoid robot retargeting is significant for embodied intelligence and robotics.
◆ To avoid complex SLAM pipelines or heavy temporal models, we propose a lightweight, engineering-oriented framework that leverages SAM 3D Body (3DB) as a frozen perception backbone and uses the Momentum HumanRig (MHR) representation as a robot-friendly intermediate.
◆ Our method (i) locks the identity and skeleton-scale parameters of per tracked subject to enforce temporally consistent bone lengths, (ii) smooths per-frame predictions via efficient sliding-window optimization in the low-dimensional MHR latent space, and (iii) recovers physically plausible global root trajectories with a differentiable soft foot-ground contact model and contact-aware global optimization.</td></tr>
<tr><td>2025-12-25</td><td>FAR-AVIO: Fast and Robust Schur-Complement Based Acoustic-Visual-Inertial Fusion Odometry with Sensor Calibration</td><td>[2512.20355](http://arxiv.org/pdf/2512.20355)</td><td>◆ Underwater environments impose severe challenges to visual-inertial odometry systems, as strong light attenuation, marine snow and turbidity, together with weakly exciting motions, degrade inertial observability and cause frequent tracking failures over long-term operation.
◆ While tightly coupled acoustic-visual-inertial fusion, typically implemented through an acoustic Doppler Velocity Log (DVL) integrated with visual-inertial measurements, can provide accurate state estimation, the associated graph-based optimization is often computationally prohibitive for real-time deployment on resource-constrained platforms.
◆ Here we present FAR-AVIO, a Schur-Complement based, tightly coupled acoustic-visual-inertial odometry framework tailored for underwater robots.</td></tr>
<tr><td>2026-01-05</td><td>LIMOncello: Iterated Error-State Kalman Filter on the SGal(3) Manifold for Fast LiDAR-Inertial Odometry</td><td>[2512.19567](http://arxiv.org/pdf/2512.19567)</td><td>◆ This work introduces LIMOncello, a tightly coupled LiDAR-Inertial Odometry system that models 6-DoF motion on the $\mathrm{SGal}(3)$ manifold within an iterated error-state Kalman filter backend.
◆ Compared to state representations defined on $\mathrm{SO}(3)\times\mathbb{R}^6$, the use of $\mathrm{SGal}(3)$ provides a coherent and numerically stable discrete-time propagation model that helps limit drift in low-observability conditions.
◆ LIMOncello also includes a lightweight incremental i-Octree mapping backend that enables faster updates and substantially lower memory usage than incremental kd-tree style map structures, without relying on locality-restricted search heuristics.</td></tr>
<tr><td>2025-12-18</td><td>SNOW: Spatio-Temporal Scene Understanding with World Knowledge for Open-World Embodied Reasoning</td><td>[2512.16461](http://arxiv.org/pdf/2512.16461)</td><td>◆ Autonomous robotic systems require spatio-temporal understanding of dynamic environments to ensure reliable navigation and interaction.
◆ While Vision-Language Models (VLMs) provide open-world semantic priors, they lack grounding in 3D geometry and temporal dynamics.
◆ Conversely, geometric perception captures structure and motion but remains semantically sparse.</td></tr>
<tr><td>2025-12-17</td><td>Spatia: Video Generation with Updatable Spatial Memory</td><td>[2512.15716](http://arxiv.org/pdf/2512.15716)</td><td>◆ Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals.
◆ To overcome this limitation, we propose Spatia, a spatial memory-aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory.
◆ Spatia iteratively generates video clips conditioned on this spatial memory and continuously updates it through visual SLAM.</td></tr>
<tr><td>2025-12-17</td><td>NAP3D: NeRF Assisted 3D-3D Pose Alignment for Autonomous Vehicles</td><td>[2512.15080](http://arxiv.org/pdf/2512.15080)</td><td>◆ Accurate localization is essential for autonomous vehicles, yet sensor noise and drift over time can lead to significant pose estimation errors, particularly in long-horizon environments.
◆ A common strategy for correcting accumulated error is visual loop closure in SLAM, which adjusts the pose graph when the agent revisits previously mapped locations.
◆ These techniques typically rely on identifying visual mappings between the current view and previously observed scenes and often require fusing data from multiple sensors.</td></tr>
<tr><td>2025-12-17</td><td>A Parameter-Free Stochastic LineseArch Method (SLAM) for Minimizing Expectation Residuals</td><td>[2512.14979](http://arxiv.org/pdf/2512.14979)</td><td>◆ Most existing rate and complexity guarantees for stochastic gradient methods in $L$-smooth settings mandates that such sequences be non-adaptive, non-increasing, and upper bounded by $\tfrac{a}{L}$ for $a &gt; 0$.
◆ This requires knowledge of $L$ and may preclude larger steps.
◆ Motivated by these shortcomings, we present an Armijo-enabled stochastic linesearch framework with standard stochastic zeroth- and first-order oracles.</td></tr>
<tr><td>2025-12-16</td><td>Odyssey: An Automotive Lidar-Inertial Odometry Dataset for GNSS-denied situations</td><td>[2512.14428](http://arxiv.org/pdf/2512.14428)</td><td>◆ The development and evaluation of Lidar-Inertial Odometry (LIO) and Simultaneous Localization and Mapping (SLAM) systems requires a precise ground truth.
◆ The Global Navigation Satellite System (GNSS) is often used as a foundation for this, but its signals can be unreliable in obstructed environments due to multi-path effects or loss-of-signal.
◆ While existing datasets compensate for the sporadic loss of GNSS signals by incorporating Inertial Measurement Unit (IMU) measurements, the commonly used Micro-Electro-Mechanical Systems (MEMS) or Fiber Optic Gyroscope (FOG)-based systems do not permit the prolonged study of GNSS-denied environments.</td></tr>
<tr><td>2025-12-16</td><td>Field evaluation and optimization of a lightweight lidar-based UAV navigation system for dense boreal forest environments</td><td>[2512.14340](http://arxiv.org/pdf/2512.14340)</td><td>◆ The interest in the usage of uncrewed aerial vehicles (UAVs) for forest applications has increased in recent years.
◆ While above-canopy flight has reached a high level of autonomy, navigating under-canopy remains a significant challenge.
◆ The use of autonomous UAVs could reduce the burden of data collection, which has motivated the development of numerous solutions for under-canopy autonomous flight.</td></tr>
<tr><td>2025-12-16</td><td>SUPER -- A Framework for Sensitivity-based Uncertainty-aware Performance and Risk Assessment in Visual Inertial Odometry</td><td>[2512.14189](http://arxiv.org/pdf/2512.14189)</td><td>◆ While many visual odometry (VO), visual-inertial odometry (VIO), and SLAM systems achieve high accuracy, the majority of existing methods miss to assess risks at runtime.
◆ This paper presents SUPER (Sensitivity-based Uncertainty-aware PErformance and Risk assessment) that is a generic and explainable framework that propagates uncertainties via sensitivities for real-time risk assessment in VIO.
◆ The scientific novelty lies in the derivation of a real-time risk indicator that is backend-agnostic and exploits the Schur complement blocks of the Gauss-Newton normal matrix to propagate uncertainties.</td></tr>
<tr><td>2025-12-16</td><td>ACE-SLAM: Scene Coordinate Regression for Neural Implicit Real-Time SLAM</td><td>[2512.14032](http://arxiv.org/pdf/2512.14032)</td><td>◆ We present a novel neural RGB-D Simultaneous Localization And Mapping (SLAM) system that learns an implicit map of the scene in real time.
◆ For the first time, we explore the use of Scene Coordinate Regression (SCR) as the core implicit map representation in a neural SLAM pipeline, a paradigm that trains a lightweight network to directly map 2D image features to 3D global coordinates.
◆ SCR networks provide efficient, low-memory 3D map representations, enable extremely fast relocalization, and inherently preserve privacy, making them particularly suitable for neural implicit SLAM.</td></tr>
<tr><td>2025-12-16</td><td>Deep Learning Perspective of Scene Understanding in Autonomous Robots</td><td>[2512.14020](http://arxiv.org/pdf/2512.14020)</td><td>◆ This paper provides a review of deep learning applications in scene understanding in autonomous robots, including innovations in object detection, semantic and instance segmentation, depth estimation, 3D reconstruction, and visual SLAM.
◆ It emphasizes how these techniques address limitations of traditional geometric models, improve depth perception in real time despite occlusions and textureless surfaces, and enhance semantic reasoning to understand the environment better.
◆ When these perception modules are integrated into dynamic and unstructured environments, they become more effective in decisionmaking, navigation and interaction.</td></tr>
<tr><td>2025-12-16</td><td>Autonomous Construction-Site Safety Inspection Using Mobile Robots: A Multilayer VLM-LLM Pipeline</td><td>[2512.13974](http://arxiv.org/pdf/2512.13974)</td><td>◆ Construction safety inspection remains mostly manual, and automated approaches still rely on task-specific datasets that are hard to maintain in fast-changing construction environments due to frequent retraining.
◆ Meanwhile, field inspection with robots still depends on human teleoperation and manual reporting, which are labor-intensive.
◆ This paper aims to connect what a robot sees during autonomous navigation to the safety rules that are common in construction sites, automatically generating a safety inspection report.</td></tr>
<tr><td>2025-12-13</td><td>INDOOR-LiDAR: Bridging Simulation and Reality for Robot-Centric 360 degree Indoor LiDAR Perception -- A Robot-Centric Hybrid Dataset</td><td>[2512.12377](http://arxiv.org/pdf/2512.12377)</td><td>◆ We present INDOOR-LIDAR, a comprehensive hybrid dataset of indoor 3D LiDAR point clouds designed to advance research in robot perception.
◆ Existing indoor LiDAR datasets often suffer from limited scale, inconsistent annotation formats, and human-induced variability during data collection.
◆ INDOOR-LIDAR addresses these limitations by integrating simulated environments with real-world scans acquired using autonomous ground robots, providing consistent coverage and realistic sensor behavior under controlled variations.</td></tr>
<tr><td>2025-12-13</td><td>Semantic Zone based 3D Map Management for Mobile Robot</td><td>[2512.12228](http://arxiv.org/pdf/2512.12228)</td><td>◆ Mobile robots in large-scale indoor environments, such as hospitals and logistics centers, require accurate 3D spatial representations.
◆ However, 3D maps consume substantial memory, making it difficult to maintain complete map data within limited computational resources.
◆ Existing SLAM frameworks typically rely on geometric distance or temporal metrics for memory management, often resulting in inefficient data retrieval in spatially compartmentalized environments.</td></tr>
<tr><td>2025-12-13</td><td>Navigation Around Unknown Space Objects Using Visible-Thermal Image Fusion</td><td>[2512.12203](http://arxiv.org/pdf/2512.12203)</td><td>◆ As the popularity of on-orbit operations grows, so does the need for precise navigation around unknown resident space objects (RSOs) such as other spacecraft, orbital debris, and asteroids.
◆ The use of Simultaneous Localization and Mapping (SLAM) algorithms is often studied as a method to map out the surface of an RSO and find the inspector&#x27;s relative pose using a lidar or conventional camera.
◆ However, conventional cameras struggle during eclipse or shadowed periods, and lidar, though robust to lighting conditions, tends to be heavier, bulkier, and more power-intensive.</td></tr>
<tr><td>2025-12-11</td><td>Contact SLAM: An Active Tactile Exploration Policy Based on Physical Reasoning Utilized in Robotic Fine Blind Manipulation Tasks</td><td>[2512.10481](http://arxiv.org/pdf/2512.10481)</td><td>◆ Contact-rich manipulation is difficult for robots to execute and requires accurate perception of the environment.
◆ In some scenarios, vision is occluded.
◆ The robot can then no longer obtain real-time scene state information through visual feedback.</td></tr>
<tr><td>2025-12-11</td><td>CLASH: Collaborative Large-Small Hierarchical Framework for Continuous Vision-and-Language Navigation</td><td>[2512.10360](http://arxiv.org/pdf/2512.10360)</td><td>◆ Vision-and-Language Navigation (VLN) requires robots to follow natural language instructions and navigate complex environments without prior maps.
◆ While recent vision-language large models demonstrate strong reasoning abilities, they often underperform task-specific panoramic small models in VLN tasks.
◆ To address this, we propose CLASH (Collaborative Large-Small Hierarchy), a VLN-CE framework that integrates a reactive small-model planner (RSMP) with a reflective large-model reasoner (RLMR).</td></tr>
<tr><td>2025-12-10</td><td>Inertial Magnetic SLAM Systems Using Low-Cost Sensors</td><td>[2512.10128](http://arxiv.org/pdf/2512.10128)</td><td>◆ Spatially inhomogeneous magnetic fields offer a valuable, non-visual information source for positioning.
◆ Among systems leveraging this, magnetic field-based simultaneous localization and mapping (SLAM) systems are particularly attractive because they can provide positioning information and build a magnetic field map on the fly.
◆ Moreover, they have bounded error within mapped regions.</td></tr>
<tr><td>2025-12-10</td><td>Super4DR: 4D Radar-centric Self-supervised Odometry and Gaussian-based Map Optimization</td><td>[2512.09608](http://arxiv.org/pdf/2512.09608)</td><td>◆ Conventional SLAM systems using visual or LiDAR data often struggle in poor lighting and severe weather.
◆ Although 4D radar is suited for such environments, its sparse and noisy point clouds hinder accurate odometry estimation, while the radar maps suffer from obscure and incomplete structures.
◆ Thus, we propose Super4DR, a 4D radar-centric framework for learning-based odometry estimation and gaussian-based map optimization.</td></tr>
<tr><td>2025-12-10</td><td>D$^2$GSLAM: 4D Dynamic Gaussian Splatting SLAM</td><td>[2512.09411](http://arxiv.org/pdf/2512.09411)</td><td>◆ Recent advances in Dense Simultaneous Localization and Mapping (SLAM) have demonstrated remarkable performance in static environments.
◆ However, dense SLAM in dynamic environments remains challenging.
◆ Most methods directly remove dynamic objects and focus solely on static scene reconstruction, which ignores the motion information contained in these dynamic objects.</td></tr>
<tr><td>2025-12-09</td><td>A Sensor-Aware Phenomenological Framework for Lidar Degradation Simulation and SLAM Robustness Evaluation</td><td>[2512.08653](http://arxiv.org/pdf/2512.08653)</td><td>◆ Lidar-based SLAM systems are highly sensitive to adverse conditions such as occlusion, noise, and field-of-view (FoV) degradation, yet existing robustness evaluation methods either lack physical grounding or do not capture sensor-specific behavior.
◆ This paper presents a sensor-aware, phenomenological framework for simulating interpretable lidar degradations directly on real point clouds, enabling controlled and reproducible SLAM stress testing.
◆ Unlike image-derived corruption benchmarks (e.g., SemanticKITTI-C) or simulation-only approaches (e.g., lidarsim), the proposed system preserves per-point geometry, intensity, and temporal structure while applying structured dropout, FoV reduction, Gaussian noise, occlusion masking, sparsification, and motion distortion.</td></tr>
<tr><td>2025-12-09</td><td>OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics</td><td>[2512.08625](http://arxiv.org/pdf/2512.08625)</td><td>◆ Simultaneous Localization and Mapping (SLAM) is a foundational component in robotics, AR/VR, and autonomous systems.
◆ With the rising focus on spatial AI in recent years, combining SLAM with semantic understanding has become increasingly important for enabling intelligent perception and interaction.
◆ Recent efforts have explored this integration, but they often rely on depth sensors or closed-set semantic models, limiting their scalability and adaptability in open-world environments.</td></tr>
<tr><td>2025-12-08</td><td>Sparse Variable Projection in Robotic Perception: Exploiting Separable Structure for Efficient Nonlinear Optimization</td><td>[2512.07969](http://arxiv.org/pdf/2512.07969)</td><td>◆ Robotic perception often requires solving large nonlinear least-squares (NLS) problems.
◆ While sparsity has been well-exploited to scale solvers, a complementary and underexploited structure is \emph{separability} -- where some variables (e.g., visual landmarks) appear linearly in the residuals and, for any estimate of the remaining variables (e.g., poses), have a closed-form solution.
◆ Variable projection (VarPro) methods are a family of techniques that exploit this structure by analytically eliminating the linear variables and presenting a reduced problem in the remaining variables that has favorable properties.</td></tr>
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
<tr><td>2026-03-17</td><td>Industrial cuVSLAM Benchmark &amp; Integration</td><td>[2603.16240](http://arxiv.org/pdf/2603.16240)</td><td>◆ This work presents a comprehensive benchmark evaluation of visual odometry (VO) and visual SLAM (VSLAM) systems for mobile robot navigation in real-world logistical environments.
◆ We compare multiple visual odometry approaches across controlled trajectories covering translational, rotational, and mixed motion patterns, as well as a large-scale production facility dataset spanning approximately 1.7 km.
◆ Performance is evaluated using Absolute Pose Error (APE) against ground truth from a Vicon motion capture system and a LiDAR-based SLAM reference.</td></tr>
<tr><td>2026-03-17</td><td>PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment</td><td>[2603.16228](http://arxiv.org/pdf/2603.16228)</td><td>◆ Real-time LiDAR-visual-inertial odometry and mapping is crucial for navigation and planning tasks in intelligent transportation systems.
◆ This study presents a pose-only bundle adjustment (PA) LiDAR-visual-inertial odometry (LVIO), named PA-LVIO, to meet the urgent need for real-time navigation and mapping.
◆ The proposed PA framework for LiDAR and visual measurements is highly accurate and efficient, and it can derive reliable frame-to-frame constraints within multiple frames.</td></tr>
<tr><td>2026-03-16</td><td>Perception-Aware Autonomous Exploration in Feature-Limited Environments</td><td>[2603.15605](http://arxiv.org/pdf/2603.15605)</td><td>◆ Autonomous exploration in unknown environments typically relies on onboard state estimation for localisation and mapping.
◆ Existing exploration methods primarily maximise coverage efficiency, but often overlook that visual-inertial odometry (VIO) performance strongly depends on the availability of robust visual features.
◆ As a result, exploration policies can drive a robot into feature-sparse regions where tracking degrades, leading to odometry drift, corrupted maps, and mission failure.</td></tr>
<tr><td>2026-03-14</td><td>Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos</td><td>[2603.12064](http://arxiv.org/pdf/2603.12064)</td><td>本文针对多自由移动相机下的稠密动态场景重建与相机姿态估计难题，提出了一种两阶段优化框架。其核心贡献与创新点如下：

◆ 提出了一种两阶段优化框架，将任务解耦为鲁棒的相机跟踪与稠密深度优化，解决了多自由移动相机（非固定标定阵列）在此类任务中的适用性问题。

◆ 在第一阶段，通过构建一个同时利用相机内时间连续性和相机间空间重叠的时空连接图，将单相机视觉SLAM扩展至多相机设置，实现了尺度一致且鲁棒的跟踪。

◆ 为应对相机间重叠有限的情况，引入了一种基于前馈重建模型的宽基线初始化策略，增强了系统在复杂场景下的启动鲁棒性。

◆ 在第二阶段，利用宽基线光流优化稠密的相机间与相机内一致性，从而联合精细化深度估计与相机姿态。

◆ 引入了一个名为MultiCamRobolab的新真实世界数据集，该数据集通过动作捕捉系统提供了真实姿态作为基准，促进了相关研究。

实验表明，该方法在合成与真实基准测试上均显著优于先进的前馈模型，且内存需求更低。</td></tr>
<tr><td>2026-03-10</td><td>OTPL-VIO: Robust Visual-Inertial Odometry with Optimal Transport Line Association and Adaptive Uncertainty</td><td>[2603.09653](http://arxiv.org/pdf/2603.09653)</td><td>该论文提出了一种鲁棒的立体视觉惯性里程计系统OTPL-VIO，其核心贡献在于通过创新的线特征关联与自适应优化策略，显著提升了在弱纹理和光照突变场景下的估计精度与鲁棒性。

◆ 提出了基于熵正则化最优传输的线特征全局匹配方法，替代传统依赖点特征的引导关联，能在特征稀疏、存在外点和部分观测的模糊场景下实现更一致、更可靠的线段对应关系。
◆ 设计了一种免训练的深度线段描述子，通过采样和池化网络特征图计算得到，为最优传输匹配提供了高质量的特征表达，且无需专门的数据训练。
◆ 深入分析了线测量噪声的影响，并在此基础上引入了可靠性自适应的加权机制，在优化过程中动态调节线约束的权重，从而提高了状态估计的稳定性。
◆ 构建了一个完整的立体点线VIO系统，在公开数据集和真实世界的低纹理、光照挑战性环境中验证了其优于现有代表性基线的精度与鲁棒性，同时保持了实时性能。</td></tr>
<tr><td>2026-03-09</td><td>Edged USLAM: Edge-Aware Event-Based SLAM with Learning-Based Depth Priors</td><td>[2603.08150](http://arxiv.org/pdf/2603.08150)</td><td>本文提出Edged USLAM，一个针对事件相机的混合视觉-惯性SLAM系统，旨在解决传统视觉SLAM在高速运动、弱光及光照突变下的失效问题。其核心贡献与创新点如下：

◆ 提出一个边缘感知的前端，通过增强事件帧来实现鲁棒的特征跟踪和非线性运动补偿，克服了事件数据稀疏异步带来的处理难题。
◆ 引入一个轻量级的深度模块，提供基于感兴趣区域的粗略场景深度，以此提升运动补偿的效果并增强系统的尺度一致性。
◆ 构建了一个完整的混合视觉-惯性系统，将事件相机、惯性测量单元和标准相机优势结合，扩展了Ultimate SLAM框架。
◆ 通过公开数据集和真实无人机飞行验证，系统在缓慢或结构化轨迹中表现出卓越的稳定性和低漂移，在挑战性光照下能提供持续精准的定位。
研究发现，纯事件方法、基于学习的方法与混合方法各具优势，而Edged USLAM作为一种稳健的混合方案，为多样化的空中导航任务提供了有效解决方案。</td></tr>
<tr><td>2026-03-03</td><td>PathSpace: Rapid continuous map approximation for efficient SLAM using B-Splines in constrained environments</td><td>[2603.02538](http://arxiv.org/pdf/2603.02538)</td><td>该论文提出了一种名为PathSpace的新型语义SLAM框架，其核心贡献在于利用连续B样条进行环境建模，以解决在约束环境中高效运行SLAM的问题。

◆ 提出了一种创新的语义SLAM框架（PathSpace），采用连续的B样条来表示环境，实现了高度紧凑的地图表达。
◆ 该方法能够维护并推理连续的概率密度函数，从而支持完整的概率推理，这是传统依赖离散密集几何表示的方法所局限的。
◆ 巧妙地将B样条的多重优势应用于SLAM上下文，特别是对离散稀疏的环境进行插值与拟合，生成连续的地图近似。
◆ 在自动驾驶赛车等具有预知结构特征（如已知赛道）的约束环境中验证了框架有效性，能以极少的资源消耗实现与基于传统路标方法相当的精度。
◆ 整体上，该工作展示了在保证精度的前提下，显著减少系统所需计算与存储资源的潜力，推动了语义SLAM在资源受限场景下的应用。</td></tr>
<tr><td>2026-03-03</td><td>D-GVIO: A Buffer-Driven and Efficient Decentralized GNSS-Visual-Inertial State Estimator for Multi-Agent Systems</td><td>[2603.01404](http://arxiv.org/pdf/2603.01404)</td><td>本文提出了一种名为D-GVIO的完全去中心化GNSS-视觉-惯性状态估计框架，旨在解决多智能体系统在资源受限下实现实时、鲁棒且高效协同定位的挑战。其核心创新点如下：

◆ 提出一种新颖的缓冲驱动策略，结合协方差分割与协方差交集技术，将分布式状态估计的传播与更新步骤模块化，显著降低了计算与通信负担。

◆ 采用左不变扩展卡尔曼滤波器进行信息融合，因其状态转移矩阵与系统状态无关，相比传统EKF具有更优越的状态估计性能。

◆ 设计了一种基于缓冲区的重传播策略，利用L-IEKF的特性高效且准确地处理延迟测量，避免了代价高昂的重新计算。

◆ 提出了一种自适应的缓冲驱动异常值检测方法，能够动态剔除GNSS异常值，增强了在GNSS信号受挑战环境中的系统鲁棒性。</td></tr>
<tr><td>2026-02-27</td><td>Motion-aware Event Suppression for Event Cameras</td><td>[2602.23204](http://arxiv.org/pdf/2602.23204)</td><td>本文提出了首个运动感知事件抑制框架，用于实时过滤事件相机中由独立运动物体和自身运动触发的事件。
◆ 首次实现了运动感知的事件抑制框架，能实时学习并过滤来自独立运动物体和自运动的干扰事件。
◆ 提出联合处理机制，在分割当前事件流中独立运动物体的同时，预测其未来运动，从而实现对动态事件的预见性抑制。
◆ 设计了轻量级架构，在消费级GPU上推理速度达173 Hz且内存占用低于1 GB，兼具高效与实用性。
◆ 在EVIMO基准测试中，分割精度较之前最优方法提升67%，同时推理速度提高53%，性能优势显著。
◆ 在下游应用中展现出重要价值：通过事件令牌修剪，将视觉Transformer推理速度提升83%；同时提升视觉里程计精度，使绝对轨迹误差降低13%。</td></tr>
<tr><td>2026-02-26</td><td>FLIGHT: Fibonacci Lattice-based Inference for Geometric Heading in real-Time</td><td>[2602.23115](http://arxiv.org/pdf/2602.23115)</td><td>本文提出了一种名为FLIGHT的新方法，用于从单目视频中实时估计相机运动方向（航向）。其核心贡献与创新点如下：

◆ 提出了一种新颖的广义霍夫变换方法，将其应用于单位球面S(2)上，以鲁棒地估计相机航向。
◆ 创新性地采用斐波那契晶格来离散化单位球面，并将其作为投票箱的中心，这种采样方式在球面上分布均匀且高效。
◆ 方法通过特征对应关系生成大圆，每个大圆为一个范围内的方向投票，使得不受噪声或动态物体影响的特征能一致地为正确运动方向投票，从而增强了抗噪声和异常值的能力。
◆ 该方法在精度与效率之间取得了优异平衡，实验证明其在三个数据集上的性能位于帕累托前沿。
◆ 在SLAM系统中应用的实验表明，该方法可通过在相机位姿初始化阶段校正航向，有效降低整体运动的均方根误差。</td></tr>
<tr><td>2026-02-26</td><td>Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones</td><td>[2602.21101](http://arxiv.org/pdf/2602.21101)</td><td>本论文针对高速飞行无人机因图像运动模糊和位姿估计噪声导致神经辐射场重建质量下降的问题，提出了一种新颖的解决方案。其核心贡献在于构建了一个融合事件相机与RGB图像的统一框架，以实现从敏捷飞行中重建高保真锐利辐射场。

◆ 提出了一种统一的异步事件流与运动模糊帧融合框架，用于从高速无人机飞行中重建高质量辐射场。
◆ 将事件-图像融合机制嵌入神经辐射场的优化过程中，直接利用事件数据辅助恢复清晰场景。
◆ 联合优化基于事件的视觉惯性里程计先验，利用事件和帧两种模态信息共同细化相机轨迹，无需地面真值监督。
◆ 在合成数据与真实世界高速无人机采集的序列上验证了方法有效性，即使在图像严重模糊、位姿先验不可靠的情况下，仍能重建高保真辐射场并保留细节。
◆ 在真实数据上相比现有先进方法取得了超过50%的性能提升，显著优于现有技术。</td></tr>
<tr><td>2026-02-22</td><td>Distributed and Consistent Multi-Robot Visual-Inertial-Ranging Odometry on Lie Groups</td><td>[2602.19173](http://arxiv.org/pdf/2602.19173)</td><td>本文提出了一种分布式协同视觉-惯性-测距里程计框架，用于解决多机器人在无GPS环境中的定位问题。其核心贡献与创新点如下：

◆ 提出了一种分布式多机器人协同定位框架，能够紧密融合视觉惯性里程计与超宽带测距观测，有效抑制单机器人系统的累积漂移。
◆ 将超宽带锚点位置显式纳入系统状态进行在线估计，无需依赖预先精确校准的固定锚点，增强了系统在实际部署中的鲁棒性。
◆ 通过机器人间的通信共享对锚点的观测，利用这些额外的几何约束来共同提升整个机器人团队的定位精度。
◆ 在李群上采用右不变误差公式进行状态估计，这一数学框架保持了标准视觉惯性里程计的正确可观测性结构，从而确保了估计器的一致性。
◆ 整个系统以分布式方式运行，在提升定位精度与鲁棒性的同时，实现了锚点的自校准，适用于去中心化的多机器人协同场景。</td></tr>
<tr><td>2026-02-22</td><td>OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness</td><td>[2602.19035](http://arxiv.org/pdf/2602.19035)</td><td>OpenVO是一个面向开放世界的视觉里程计框架，其核心贡献在于解决了单目行车记录仪视频在观测频率多变和相机未标定条件下的鲁棒运动估计问题。  
◆ 首次在视觉里程计中显式编码时序动态信息，克服了传统方法固定观测频率（如10Hz）的局限，使其能适应真实世界中变化的帧率。  
◆ 提出利用基础模型提供的3D几何先验，无需依赖已知相机内参，从而直接处理未标定相机拍摄的视频。  
◆ 构建了一个能够从罕见驾驶事件的行车记录仪数据中构建轨迹数据集的系统，显著提升了在多变观测频率下的鲁棒性，误差降低46%至92%。  
◆ 在KITTI、nuScenes和Argoverse 2三大自动驾驶基准测试中性能超越现有方法20%以上，展示了其强大的泛化能力和实际应用潜力。</td></tr>
<tr><td>2026-02-20</td><td>Have We Mastered Scale in Deep Monocular Visual SLAM? The ScaleMaster Dataset and Benchmark</td><td>[2602.18174](http://arxiv.org/pdf/2602.18174)</td><td>该论文的核心贡献在于首次系统性地揭示并评估了当前深度单目视觉SLAM系统在大规模室内场景中面临的尺度一致性问题，并为此建立了专门的基准。

◆ 提出了首个专注于评估视觉SLAM尺度一致性的数据集ScaleMaster，其场景包含多层结构、长轨迹、重复视角和低纹理区域等挑战。
◆ 系统性地分析了先进深度单目视觉SLAM的脆弱性，发现它们在现有基准上表现良好，但在真实大规模环境中会出现严重的尺度相关故障。
◆ 将评估维度从传统的轨迹精度，拓展到直接的图到图质量评估，引入了基于高精度3D真值的倒角距离等度量方法。
◆ 公开了数据集与基线结果，为未来开发尺度一致、可靠的视觉SLAM系统研究奠定了基础。</td></tr>
<tr><td>2026-02-19</td><td>Cholec80-port: A Geometrically Consistent Trocar Port Segmentation Dataset for Robust Surgical Scene Understanding</td><td>[2602.17060](http://arxiv.org/pdf/2602.17060)</td><td>该论文的核心贡献是创建了一个用于提升手术场景几何理解鲁棒性的、几何一致性的套管端口分割数据集。

◆ 首次在公开手术数据集Cholec80上提供了高保真度的套管端口分割标注，填补了该关键结构标注稀缺的空白。
◆ 明确指出了现有标注将端口中央通孔（管腔）也掩蔽掉的做法违反了几何一致性，因为通过通孔可见的解剖组织本应是场景的一部分。
◆ 为此制定了一套严格的标准操作程序，定义了一种排除中央通孔的“端口套管”掩膜标注方式，确保了标注的几何正确性。
◆ 不仅创建了新数据集，还按照同一标准清理和统一了现有公共数据集，增强了数据集的协调性与可用性。
◆ 通过实验证明，采用这种几何一致的标注能显著提升模型在不同数据集间的泛化鲁棒性，其效果超越了单纯增加数据集规模所带来的提升。</td></tr>
<tr><td>2026-02-13</td><td>Adaptive Illumination Control for Robot Perception</td><td>[2602.15900](http://arxiv.org/pdf/2602.15900)</td><td>该论文的核心贡献是提出了一个名为Lightning的闭环照明控制框架，旨在通过主动调控机器人自带的程序化光源来直接改善视觉SLAM在弱光或高动态范围场景下的感知鲁棒性。

◆ 提出了一个共置照明分解模型，能够将观测图像分解为环境光成分和可控光源贡献场，从而无需重复实地采集即可物理一致地合成不同光照强度下的场景图像，生成了密集的多强度训练数据。

◆ 基于合成数据，构建了一个离线最优照明强度调度问题，该问题能在图像序列上权衡SLAM所需的图像效用、功耗以及光照的时间平滑性，以优化选择光照强度。

◆ 通过行为克隆将离线优化方案蒸馏为一个可实时运行的照明控制策略，该策略能泛化至初始训练分布之外，并能在移动机器人上在线运行以指令离散的光照强度等级。

◆ 整个框架实现了从感知建模、离线优化到在线策略的完整闭环，首次系统性地将主动照明控制与视觉SLAM相结合，直接改善了图像捕获质量这一上游环节。

◆ 实验评估表明，该框架显著提升了SLAM轨迹的鲁棒性，同时减少了不必要的照明功耗。</td></tr>
<tr><td>2026-02-16</td><td>Understanding Sensor Vulnerabilities in Industrial XR Tracking</td><td>[2602.14413](http://arxiv.org/pdf/2602.14413)</td><td>该论文的核心贡献在于通过受控实验，首次系统性地揭示了工业XR视觉-惯性里程计在持续传感器退化条件下的脆弱性，并指出了现有评估体系的不足。

◆ 研究视角创新：将评估重点从理想的传感器标称性能，转向实际工业环境中持续性的传感器性能退化，填补了该领域的研究空白。
◆ 方法创新：采用系统化的故障注入方法，在一系列操作状态下，分别对视觉和惯性传感器的退化故障进行了可控的实证研究。
◆ 发现关键不对称性：定量评估揭示了一个重要的不对称影响——视觉传感退化通常仅导致厘米级的有限位姿误差，而惯性传感退化则可能引发巨大轨迹偏差，可达数百至数千米。
◆ 提出重要设计导向：基于上述发现，研究强调在工业XR系统的评估与设计中，必须更加重视惯性测量单元的可靠性，这为提升系统在实际环境中的鲁棒性提供了明确方向。</td></tr>
<tr><td>2026-02-14</td><td>UAV-SEAD: State Estimation Anomaly Dataset for UAVs</td><td>[2602.13900](http://arxiv.org/pdf/2602.13900)</td><td>该论文的核心贡献是创建并发布了一个用于无人机状态估计异常检测的大规模真实世界数据集，旨在解决该研究领域缺乏真实数据的问题。

◆ 首创性地提供了一个大规模、真实世界的无人机异常数据集，包含1396条真实飞行日志，总时长超过52小时，而非使用常见的模拟或注入故障数据。
◆ 数据采集环境多样，涵盖室内外多种场景，并使用多架基于PX4飞控、配备不同传感器套件的无人机，确保了数据的广泛代表性。
◆ 数据集包含了未经合成处理的正常与异常飞行数据，使其特别适用于开发贴近现实场景的异常检测算法。
◆ 提出了一种针对无人机状态估计异常的结构化分类体系，将异常分为机械电气、外部位置、全局位置和高度异常四类，为研究提供了清晰框架。
◆ 数据集基于PX4日志机制，提供了多变量传感器数据流，包括IMU、GPS、视觉里程计等，可用于研究上下文异常和集体异常等多种类型。

该数据集预期将为无人机异常检测与隔离系统的开发、训练和评估提供关键资源，弥补可靠性研究领域的空白。</td></tr>
<tr><td>2026-02-12</td><td>GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry</td><td>[2602.11714](http://arxiv.org/pdf/2602.11714)</td><td>GSO-SLAM的核心贡献在于提出了一种新颖的双向耦合单目密集SLAM系统，它通过创新的方式整合了视觉里程计与高斯泼溅场景表示，实现了实时、高精度的同步定位与建图。

◆ 核心创新是提出了视觉里程计与高斯泼溅之间的双向耦合机制，克服了现有方法在统一场景表示下计算成本高，或松散集成导致冗余的问题。
◆ 该方法在期望最大化框架内制定了联合优化，能够同步优化VO产生的半稠密深度估计与GS场景表示，且不引入额外计算开销。
◆ 提出了高斯泼溅初始化技术，直接利用VO提供的图像信息、关键帧位姿和像素关联来生成接近最终效果的高斯场景初始状态，从而避免了传统启发式方法的需要。
◆ 整个系统能够在实时运行的前提下，在场景重建的几何/光度保真度以及跟踪精度方面达到先进水平，并通过大量实验验证了其有效性。</td></tr>
<tr><td>2026-02-11</td><td>MDE-VIO: Enhancing Visual-Inertial Odometry Using Learned Depth Priors</td><td>[2602.11323](http://arxiv.org/pdf/2602.11323)</td><td>该论文提出了一种将学习式深度先验融入视觉惯性里程计（VIO）的新方法MDE-VIO，旨在解决传统单目VIO在低纹理环境中因特征稀疏导致的性能下降问题。

其核心贡献与创新点如下：
◆ 提出了一种新颖的融合框架，将学习到的稠密深度先验直接集成到经典的VINS-Mono优化后端中，而非前端，在提升性能的同时严格遵循边缘设备的计算限制。
◆ 设计了仿射不变深度一致性约束与成对顺序约束，有效利用学习深度图中的几何一致性信息，增强了系统的鲁棒性。
◆ 引入了一种基于方差的门控机制，能够显式地过滤学习深度图中不稳定的伪影，提高了先验信息的可靠性。
◆ 该方法能够在边缘设备上实时运行，并鲁棒地恢复出公制尺度，解决了复杂基础模型计算量大、难以部署的瓶颈。
◆ 在公开数据集上的实验表明，该方法能有效防止在挑战性场景中的轨迹发散，并将绝对轨迹误差（ATE）显著降低高达28.3%。</td></tr>
<tr><td>2026-02-09</td><td>Thegra: Graph-based SLAM for Thermal Imagery</td><td>[2602.08531](http://arxiv.org/pdf/2602.08531)</td><td>该论文提出了一种基于图优化的稀疏单目热成像SLAM系统，旨在解决热图像纹理低、对比度差和噪声高导致的特征提取与匹配难题。

◆ 首次将通用学习特征SuperPoint检测器和LightGlue匹配器应用于热成像SLAM，利用可见光光谱大数据训练以获得跨域泛化能力，无需针对热数据重新训练或微调。
◆ 设计了一套针对热图像的预处理流程，以增强输入图像对学习特征的适配性，提升特征提取质量。
◆ 改进了核心SLAM模块，使其能够有效处理热图像中稀疏且易含异常值的特征匹配，增强了系统的稳定性。
◆ 创新地将SuperPoint生成的关键点置信度分数融入置信度加权因子图优化中，提高了位姿估计的鲁棒性和准确性。
实验表明，该系统在公开热数据集上实现了可靠性能，为视觉退化环境下的SLAM提供了实用解决方案。</td></tr>
<tr><td>2026-02-09</td><td>Aerial Manipulation with Contact-Aware Onboard Perception and Hybrid Control</td><td>[2602.08251](http://arxiv.org/pdf/2602.08251)</td><td>该论文提出了一套完全基于机载感知与控制的空中机器人操作方案，旨在实现无需外部动捕的、接触丰富的精确操作。

◆ 设计了增强型视觉惯性里程计（VIO），其创新在于引入了仅在机械臂与环境接触时才激活的接触一致性约束。这有效收紧了接触坐标系的不确定性，显著减少了估计漂移。

◆ 提出了一种结合图像视觉伺服（IBVS）与混合力-运动控制的策略。IBVS用于减轻感知与控制间的耦合干扰，而混合控制器则能同时调节接触力/力矩和横向运动。

◆ 实现了完全基于机载传感器的“感知-力控”闭环系统。实验表明，该系统在接触时的速度估计精度提升了66.01%，能可靠接近目标并保持稳定的接触力，推动了空中操作在真实野外环境的部署应用。</td></tr>
<tr><td>2026-01-28</td><td>When Simultaneous Localization and Mapping Meets Wireless Communications: A Survey</td><td>[2602.06995](http://arxiv.org/pdf/2602.06995)</td><td>本文综述了同步定位与建图（SLAM）与无线通信融合领域的最新进展，核心贡献在于系统阐述了两者间的双向赋能关系与集成路径。其创新点可总结如下：

◆ 首次系统性地综述了SLAM（尤其是视觉SLAM）与无线通信之间的交叉领域，阐明两者存在双向互惠的深刻联系。

◆ 提出无线通信信息（如射频信号）可辅助解决单目视觉SLAM中的尺度模糊问题，从而增强SLAM的鲁棒性与精度。

◆ 指出SLAM中的视觉里程计等技术可为5G及后续移动通信网络提供环境感知与路径预测能力，优化无线信道资源配置。

◆ 分析了利用概率模型、空间信号处理等数学方法，以及多天线等技术，实现机器人状态高效估计的多种技术路径。

◆ 揭示当前通信与SLAM的联合解决方案仍处于起步阶段，未来需在理论与实践中融入更高层次的定位和语义感知能力。</td></tr>
<tr><td>2026-02-06</td><td>POPL-KF: A Pose-Only Geometric Representation-Based Kalman Filter for Point-Line-Based Visual-Inertial Odometry</td><td>[2602.06425](http://arxiv.org/pdf/2602.06425)</td><td>◆ Mainstream Visual-inertial odometry  (VIO) systems rely on point features for motion estimation and localization.
◆ However, their performance degrades in challenging scenarios.
◆ Moreover, the localization accuracy of multi-state constraint Kalman filter (MSCKF)-based VIO systems suffers from linearization errors associated with feature 3D coordinates and delayed measurement updates.</td></tr>
<tr><td>2026-02-05</td><td>Feature points evaluation on omnidirectional vision with a photorealistic fisheye sequence -- A report on experiments done in 2014</td><td>[2602.05487](http://arxiv.org/pdf/2602.05487)</td><td>◆ What is this report: This is a scientific report, contributing with a detailed bibliography, a dataset which we will call now PFSeq for &#x27;&#x27;Photorealistic Fisheye Sequence&#x27;&#x27; and make available at https://doi.org/10.
◆ 57745/DYIVVU, and comprehensive experiments.
◆ This work should be considered as a draft, and has been done during my PhD thesis &#x27;&#x27;Construction of 3D models from fisheye video data-Application to the localisation in urban area&#x27;&#x27; in 2014 [Mor16].</td></tr>
<tr><td>2026-02-03</td><td>LEVIO: Lightweight Embedded Visual Inertial Odometry for Resource-Constrained Devices</td><td>[2602.03294](http://arxiv.org/pdf/2602.03294)</td><td>◆ Accurate, infrastructure-less sensor systems for motion tracking are essential for mobile robotics and augmented reality (AR) applications.
◆ The most popular state-of-the-art visual-inertial odometry (VIO) systems, however, are too computationally demanding for resource-constrained hardware, such as micro-drones and smart glasses.
◆ This work presents LEVIO, a fully featured VIO pipeline optimized for ultra-low-power compute platforms, allowing six-degrees-of-freedom (DoF) real-time sensing.</td></tr>
<tr><td>2026-02-02</td><td>Vision-only UAV State Estimation for Fast Flights Without External Localization Systems: A2RL Drone Racing Finalist Approach</td><td>[2602.01860](http://arxiv.org/pdf/2602.01860)</td><td>◆ Fast flights with aggressive maneuvers in cluttered GNSS-denied environments require fast, reliable, and accurate UAV state estimation.
◆ In this paper, we present an approach for onboard state estimation of a high-speed UAV using a monocular RGB camera and an IMU.
◆ Our approach fuses data from Visual-Inertial Odometry (VIO), an onboard landmark-based camera measurement system, and an IMU to produce an accurate state estimate.</td></tr>
<tr><td>2026-02-02</td><td>Real-Time Loop Closure Detection in Visual SLAM via NetVLAD and Faiss</td><td>[2602.01673](http://arxiv.org/pdf/2602.01673)</td><td>◆ Loop closure detection (LCD) is a core component of simultaneous localization and mapping (SLAM): it identifies revisited places and enables pose-graph constraints that correct accumulated drift.
◆ Classic bag-of-words approaches such as DBoW are efficient but often degrade under appearance change and perceptual aliasing.
◆ In parallel, deep learning-based visual place recognition (VPR) descriptors (e.g., NetVLAD and Transformer-based models) offer stronger robustness, but their computational cost is often viewed as a barrier to real-time SLAM.</td></tr>
<tr><td>2026-01-22</td><td>Keyframe-Based Feed-Forward Visual Odometry</td><td>[2601.16020](http://arxiv.org/pdf/2601.16020)</td><td>◆ The emergence of visual foundation models has revolutionized visual odometry~(VO) and SLAM, enabling pose estimation and dense reconstruction within a single feed-forward network.
◆ However, unlike traditional pipelines that leverage keyframe methods to enhance efficiency and accuracy, current foundation model based methods, such as VGGT-Long, typically process raw image sequences indiscriminately.
◆ This leads to computational redundancy and degraded performance caused by low inter-frame parallax, which provides limited contextual stereo information.</td></tr>
<tr><td>2026-01-14</td><td>SCE-SLAM: Scale-Consistent Monocular SLAM via Scene Coordinate Embeddings</td><td>[2601.09665](http://arxiv.org/pdf/2601.09665)</td><td>◆ Monocular visual SLAM enables 3D reconstruction from internet video and autonomous navigation on resource-constrained platforms, yet suffers from scale drift, i.e., the gradual divergence of estimated scale over long sequences.
◆ Existing frame-to-frame methods achieve real-time performance through local optimization but accumulate scale drift due to the lack of global constraints among independent windows.
◆ To address this, we propose SCE-SLAM, an end-to-end SLAM system that maintains scale consistency through scene coordinate embeddings, which are learned patch-level representations encoding 3D geometric relationships under a canonical scale reference.</td></tr>
<tr><td>2026-01-12</td><td>Nonlinear Observer Design for Visual-Inertial Odometry</td><td>[2601.07156](http://arxiv.org/pdf/2601.07156)</td><td>◆ This paper addresses the problem of Visual-Inertial Odometry (VIO) for rigid body systems evolving in three-dimensional space.
◆ We introduce a novel matrix Lie group structure, denoted SE_{3+n}(3), that unifies the pose, gravity, linear velocity, and landmark positions within a consistent geometric framework tailored to the VIO problem.
◆ Building upon this formulation, we design an almost globally asymptotically stable nonlinear geometric observer that tightly integrates data from an Inertial Measurement Unit (IMU) and visual sensors.</td></tr>
<tr><td>2026-01-06</td><td>Loop Closure using AnyLoc Visual Place Recognition in DPV-SLAM</td><td>[2601.02723](http://arxiv.org/pdf/2601.02723)</td><td>◆ Loop closure is crucial for maintaining the accuracy and consistency of visual SLAM.
◆ We propose a method to improve loop closure performance in DPV-SLAM.
◆ Our approach integrates AnyLoc, a learning-based visual place recognition technique, as a replacement for the classical Bag of Visual Words (BoVW) loop detection method.</td></tr>
<tr><td>2026-01-09</td><td>360DVO: Deep Visual Odometry for Monocular 360-Degree Camera</td><td>[2601.02309](http://arxiv.org/pdf/2601.02309)</td><td>◆ Monocular omnidirectional visual odometry (OVO) systems leverage 360-degree cameras to overcome field-of-view limitations of perspective VO systems.
◆ However, existing methods, reliant on handcrafted features or photometric objectives, often lack robustness in challenging scenarios, such as aggressive motion and varying illumination.
◆ To address this, we present 360DVO, the first deep learning-based OVO framework.</td></tr>
<tr><td>2026-01-02</td><td>DefVINS: Visual-Inertial Odometry for Deformable Scenes</td><td>[2601.00702](http://arxiv.org/pdf/2601.00702)</td><td>◆ Deformable scenes violate the rigidity assumptions underpinning classical visual-inertial odometry (VIO), often leading to over-fitting to local non-rigid motion or severe drift when deformation dominates visual parallax.
◆ We introduce DefVINS, a visual-inertial odometry framework that explicitly separates a rigid, IMU-anchored state from a non--rigid warp represented by an embedded deformation graph.
◆ The system is initialized using a standard VIO procedure that fixes gravity, velocity, and IMU biases, after which non-rigid degrees of freedom are activated progressively as the estimation becomes well conditioned.</td></tr>
<tr><td>2026-01-01</td><td>FoundationSLAM: Unleashing the Power of Depth Foundation Models for End-to-End Dense Visual SLAM</td><td>[2512.25008](http://arxiv.org/pdf/2512.25008)</td><td>◆ We present FoundationSLAM, a learning-based monocular dense SLAM system that addresses the absence of geometric consistency in previous flow-based approaches for accurate and robust tracking and mapping.
◆ Our core idea is to bridge flow estimation with geometric reasoning by leveraging the guidance from foundation depth models.
◆ To this end, we first develop a Hybrid Flow Network that produces geometry-aware correspondences, enabling consistent depth and pose inference across diverse keyframes.</td></tr>
<tr><td>2025-12-23</td><td>Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing</td><td>[2512.20475](http://arxiv.org/pdf/2512.20475)</td><td>◆ The Abu Dhabi Autonomous Racing League(A2RL) x Drone Champions League competition(DCL) requires teams to perform high-speed autonomous drone racing using only a single camera and a low-quality inertial measurement unit -- a minimal sensor set that mirrors expert human drone racing pilots.
◆ This sensor limitation makes the system susceptible to drift from Visual-Inertial Odometry (VIO), particularly during long and fast flights with aggressive maneuvers.
◆ This paper presents the system developed for the championship, which achieved a competitive performance.</td></tr>
<tr><td>2025-12-25</td><td>FAR-AVIO: Fast and Robust Schur-Complement Based Acoustic-Visual-Inertial Fusion Odometry with Sensor Calibration</td><td>[2512.20355](http://arxiv.org/pdf/2512.20355)</td><td>◆ Underwater environments impose severe challenges to visual-inertial odometry systems, as strong light attenuation, marine snow and turbidity, together with weakly exciting motions, degrade inertial observability and cause frequent tracking failures over long-term operation.
◆ While tightly coupled acoustic-visual-inertial fusion, typically implemented through an acoustic Doppler Velocity Log (DVL) integrated with visual-inertial measurements, can provide accurate state estimation, the associated graph-based optimization is often computationally prohibitive for real-time deployment on resource-constrained platforms.
◆ Here we present FAR-AVIO, a Schur-Complement based, tightly coupled acoustic-visual-inertial odometry framework tailored for underwater robots.</td></tr>
<tr><td>2025-12-22</td><td>Trifocal Tensor and Relative Pose Estimation with Known Vertical Direction</td><td>[2512.19110](http://arxiv.org/pdf/2512.19110)</td><td>◆ This work presents two novel solvers for estimating the relative poses among views with known vertical directions.
◆ The vertical directions of camera views can be easily obtained using inertial measurement units (IMUs) which have been widely used in autonomous vehicles, mobile phones, and unmanned aerial vehicles (UAVs).
◆ Given the known vertical directions, our lgorithms only need to solve for two rotation angles and two translation vectors.</td></tr>
<tr><td>2025-12-19</td><td>Deep Learning-based Robust Autonomous Navigation of Aerial Robots in Dense Forests</td><td>[2512.17553](http://arxiv.org/pdf/2512.17553)</td><td>◆ Autonomous aerial navigation in dense natural environments remains challenging due to limited visibility, thin and irregular obstacles, GNSS-denied operation, and frequent perceptual degradation.
◆ This work presents an improved deep learning-based navigation framework that integrates semantically enhanced depth encoding with neural motion-primitive evaluation for robust flight in cluttered forests.
◆ Several modules are incorporated on top of the original sevae-ORACLE algorithm to address limitations observed during real-world deployment, including lateral control for sharper maneuvering, a temporal consistency mechanism to suppress oscillatory planning decisions, a stereo-based visual-inertial odometry solution for drift-resilient state estimation, and a supervisory safety layer that filters unsafe actions in real time.</td></tr>
<tr><td>2025-12-19</td><td>Adaptive Covariance and Quaternion-Focused Hybrid Error-State EKF/UKF for Visual-Inertial Odometry</td><td>[2512.17505](http://arxiv.org/pdf/2512.17505)</td><td>◆ This study presents an innovative hybrid Visual-Inertial Odometry (VIO) method for Unmanned Aerial Vehicles (UAVs) that is resilient to environmental challenges and capable of dynamically assessing sensor reliability.
◆ Built upon a loosely coupled sensor fusion architecture, the system utilizes a novel hybrid Quaternion-focused Error-State EKF/UKF (Qf-ES-EKF/UKF) architecture to process inertial measurement unit (IMU) data.
◆ This architecture first propagates the entire state using an Error-State Extended Kalman Filter (ESKF) and then applies a targeted Scaled Unscented Kalman Filter (SUKF) step to refine only the orientation.</td></tr>
<tr><td>2025-12-17</td><td>Spatia: Video Generation with Updatable Spatial Memory</td><td>[2512.15716](http://arxiv.org/pdf/2512.15716)</td><td>◆ Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals.
◆ To overcome this limitation, we propose Spatia, a spatial memory-aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory.
◆ Spatia iteratively generates video clips conditioned on this spatial memory and continuously updates it through visual SLAM.</td></tr>
<tr><td>2025-12-16</td><td>SUPER -- A Framework for Sensitivity-based Uncertainty-aware Performance and Risk Assessment in Visual Inertial Odometry</td><td>[2512.14189](http://arxiv.org/pdf/2512.14189)</td><td>◆ While many visual odometry (VO), visual-inertial odometry (VIO), and SLAM systems achieve high accuracy, the majority of existing methods miss to assess risks at runtime.
◆ This paper presents SUPER (Sensitivity-based Uncertainty-aware PErformance and Risk assessment) that is a generic and explainable framework that propagates uncertainties via sensitivities for real-time risk assessment in VIO.
◆ The scientific novelty lies in the derivation of a real-time risk indicator that is backend-agnostic and exploits the Schur complement blocks of the Gauss-Newton normal matrix to propagate uncertainties.</td></tr>
<tr><td>2025-12-16</td><td>Deep Learning Perspective of Scene Understanding in Autonomous Robots</td><td>[2512.14020](http://arxiv.org/pdf/2512.14020)</td><td>◆ This paper provides a review of deep learning applications in scene understanding in autonomous robots, including innovations in object detection, semantic and instance segmentation, depth estimation, 3D reconstruction, and visual SLAM.
◆ It emphasizes how these techniques address limitations of traditional geometric models, improve depth perception in real time despite occlusions and textureless surfaces, and enhance semantic reasoning to understand the environment better.
◆ When these perception modules are integrated into dynamic and unstructured environments, they become more effective in decisionmaking, navigation and interaction.</td></tr>
<tr><td>2025-12-09</td><td>Enabling Autonomous Navigation in a Snake Robot through Visual-Inertial Odometry and Closed-Loop Trajectory Tracking Control</td><td>[2512.11886](http://arxiv.org/pdf/2512.11886)</td><td>◆ Snake robots offer exceptional mobility across extreme terrain inaccessible to conventional rovers, yet their highly articulated bodies present fundamental challenges for autonomous navigation in environments lacking external tracking infrastructure.
◆ This thesis develops a complete autonomy pipeline for COBRA, an 11 degree-of-freedom modular snake robot designed for planetary exploration.
◆ While the robot&#x27;s biologically inspired serpentine gaits achieve impressive mobility, prior work has relied entirely on open-loop teleoperation.</td></tr>
<tr><td>2025-12-10</td><td>Inertial Magnetic SLAM Systems Using Low-Cost Sensors</td><td>[2512.10128](http://arxiv.org/pdf/2512.10128)</td><td>◆ Spatially inhomogeneous magnetic fields offer a valuable, non-visual information source for positioning.
◆ Among systems leveraging this, magnetic field-based simultaneous localization and mapping (SLAM) systems are particularly attractive because they can provide positioning information and build a magnetic field map on the fly.
◆ Moreover, they have bounded error within mapped regions.</td></tr>
<tr><td>2025-12-11</td><td>Development and Testing for Perception Based Autonomous Landing of a Long-Range QuadPlane</td><td>[2512.09343](http://arxiv.org/pdf/2512.09343)</td><td>◆ QuadPlanes combine the range efficiency of fixed-wing aircraft with the maneuverability of multi-rotor platforms for long-range autonomous missions.
◆ In GPS-denied or cluttered urban environments, perception-based landing is vital for reliable operation.
◆ Unlike structured landing zones, real-world sites are unstructured and highly variable, requiring strong generalization capabilities from the perception system.</td></tr>
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
<tr><td>2026-03-17</td><td>Search2Motion: Training-Free Object-Level Motion Control via Attention-Consensus Search</td><td>[2603.16711](http://arxiv.org/pdf/2603.16711)</td><td>◆ We present Search2Motion, a training-free framework for object-level motion editing in image-to-video generation.
◆ Unlike prior methods requiring trajectories, bounding boxes, masks, or motion fields, Search2Motion adopts target-frame-based control, leveraging first-last-frame motion priors to realize object relocation while preserving scene stability without fine-tuning.
◆ Reliable target-frame construction is achieved through semantic-guided object insertion and robust background inpainting.</td></tr>
<tr><td>2026-03-17</td><td>Reconciling distributed compliance with high-performance control in continuum soft robotics</td><td>[2603.16630](http://arxiv.org/pdf/2603.16630)</td><td>◆ High-performance closed-loop control of truly soft continuum manipulators has remained elusive.
◆ Experimental demonstrations have largely relied on sufficiently stiff, piecewise architectures in which each actuated segment behaves as a distributed yet effectively rigid element, while deformation modes beyond simple bending are suppressed.
◆ This strategy simplifies modeling and control, but sidesteps the intrinsic complexity of a fully compliant body and makes the system behave as a serial kinematic chain, much like a conventional articulated robot.</td></tr>
<tr><td>2026-03-17</td><td>Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty</td><td>[2603.16538](http://arxiv.org/pdf/2603.16538)</td><td>◆ 3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement.
◆ However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry.
◆ In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers.</td></tr>
<tr><td>2026-03-17</td><td>Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies</td><td>[2603.16536](http://arxiv.org/pdf/2603.16536)</td><td>◆ We present Kamino, a GPU-based physics solver for massively parallel simulations of heterogeneous highly-coupled mechanical systems.
◆ Implemented in Python using NVIDIA Warp and integrated into the Newton framework, it enables the application of data-driven methods, such as large-scale reinforcement learning, to complex robotic systems that exhibit strongly coupled kinematic and dynamic constraints such as kinematic loops.
◆ The latter are often circumvented by practitioners; approximating the system topology as a kinematic tree and incorporating explicit loop-closure constraints or so-called mimic joints.</td></tr>
<tr><td>2026-03-16</td><td>Optimizing Hospital Capacity During Pandemics: A Dual-Component Framework for Strategic Patient Relocation</td><td>[2603.15960](http://arxiv.org/pdf/2603.15960)</td><td>◆ The COVID-19 pandemic has placed immense strain on hospital systems worldwide, leading to critical capacity challenges.
◆ This research proposes a two-part framework to optimize hospital capacity through patient relocation strategies.
◆ The first component involves developing a time series prediction model to forecast patient arrival rates.</td></tr>
<tr><td>2026-03-16</td><td>Multi-Year Spectral Structure of 6G Candidate Bands at 2.7 GHz and 4.4 GHz</td><td>[2603.15837](http://arxiv.org/pdf/2603.15837)</td><td>◆ Mid-band spectrum between 2 and 8 GHz is a critical resource for sixth-generation (6G) systems as it uniquely balances favorable propagation characteristics with scalable bandwidth.
◆ Recent U.S.
◆ policy highlights candidate bands near 2.7, 4.4, and 7.1 GHz, all of which host substantial federal and non-federal incumbency, including high-power radiolocation and aeronautical telemetry systems.</td></tr>
<tr><td>2026-03-16</td><td>CLRNet: Targetless Extrinsic Calibration for Camera, Lidar and 4D Radar Using Deep Learning</td><td>[2603.15767](http://arxiv.org/pdf/2603.15767)</td><td>◆ In this paper, we address extrinsic calibration for camera, lidar, and 4D radar sensors.
◆ Accurate extrinsic calibration of radar remains a challenge due to the sparsity of its data.
◆ We propose CLRNet, a novel, multi-modal end-to-end deep learning (DL) calibration network capable of addressing joint camera-lidar-radar calibration, or pairwise calibration between any two of these sensors.</td></tr>
<tr><td>2026-03-16</td><td>Voronoi-based Second-order Descriptor with Whitened Metric in LiDAR Place Recognition</td><td>[2603.14974](http://arxiv.org/pdf/2603.14974)</td><td>◆ The pooling layer plays a vital role in aggregating local descriptors into the metrizable global descriptor in the LiDAR Place Recognition (LPR).
◆ In particular, the second-order pooling is capable of capturing higher-order interactions among local descriptors.
◆ However, its existing methods in the LPR adhere to conventional implementations and post-normalization, and incur the descriptor unsuitable for Euclidean distancing.</td></tr>
<tr><td>2026-03-16</td><td>The impact of machine learning forecasting on strategic decision-making for Bike Sharing Systems</td><td>[2603.14901](http://arxiv.org/pdf/2603.14901)</td><td>◆ In this paper, machine learning techniques are used to forecast the difference between bike returns and withdrawals at each station of a bike sharing system.
◆ The forecasts are integrated into a simulation framework that is used to support long-term decisions and model the daily dynamics, including the relocation of bikes.
◆ We assess the quality of the machine learning-based forecasts in two ways.</td></tr>
<tr><td>2026-03-14</td><td>H-RINS: Hierarchical Tightly-coupled Radar-Inertial Navigation via Smoothing and Mapping</td><td>[2603.14109](http://arxiv.org/pdf/2603.14109)</td><td>◆ Millimeter-wave radar provides robust perception in visually degraded environments.
◆ However, radar-inertial state estimation is inherently susceptible to drift.
◆ Because radar yields only sparse, body-frame velocity measurements, it provides weak constraints on absolute orientation.</td></tr>
<tr><td>2026-03-14</td><td>Evaluation of Visual Place Recognition Methods for Image Pair Retrieval in 3D Vision and Robotics</td><td>[2603.13917](http://arxiv.org/pdf/2603.13917)</td><td>◆ Visual Place Recognition (VPR) is a core component in computer vision, typically formulated as an image retrieval task for localization, mapping, and navigation.
◆ In this work, we instead study VPR as an image pair retrieval front-end for registration pipelines, where the goal is to find top-matching image pairs between two disjoint image sets for downstream tasks such as scene registration, SLAM, and Structure-from-Motion.
◆ We comparatively evaluate state-of-the-art VPR families - NetVLAD-style baselines, classification-based global descriptors (CosPlace, EigenPlaces), feature-mixing (MixVPR), and foundation-model-driven methods (AnyLoc, SALAD, MegaLoc) - on three challenging datasets: object-centric outdoor scenes (Tanks and Temples), indoor RGB-D scans (ScanNet-GS), and autonomous-driving sequences (KITTI).</td></tr>
<tr><td>2026-03-14</td><td>Exploring the Dimensions of a Variational Neuron</td><td>[2603.13849](http://arxiv.org/pdf/2603.13849)</td><td>◆ We introduce EVE (Elemental Variational Expanse), a variational distributional neuron formulated as a local probabilistic computational unit with an explicit prior, an amortized posterior, and unit-level variational regularization.
◆ In most modern architectures, uncertainty is modeled through global latent variables or parameter uncertainty, while the computational unit itself remains scalar.
◆ EVE instead relocates probabilistic structure to the neuron level, making it locally observable and controllable.</td></tr>
<tr><td>2026-03-13</td><td>Ridged Lagrangian Perturbation Theory (RLPT)</td><td>[2603.13106](http://arxiv.org/pdf/2603.13106)</td><td>◆ Galaxy surveys demand fast large-scale structure forward models that preserve large-scale phases while providing realistic nonlinear morphology at fixed force resolution.
◆ Single-step Lagrangian Perturbation Theory (LPT) solvers are efficient, but they typically yield overly diffuse filaments and knots and underpredict small-scale clustering.
◆ We introduce Ridged Lagrangian Perturbation Theory (RLPT), a modular two-step scheme: a standard long-range LPT/ALPT transport is followed by a single post-processing Eulerian {ridging} update that reconstructs a short-range, curl-free displacement from the realised density field through a smooth scale separation and a Poisson inversion.</td></tr>
<tr><td>2026-03-13</td><td>Improving critical buildings energy resilience via shared autonomous electric vehicles -- A sequential optimization framework</td><td>[2603.12771](http://arxiv.org/pdf/2603.12771)</td><td>◆ The interdependence between electric power systems and transportation systems is rapidly increasing due to the high adoption of Electric Vehicles (EVs) and their charging infrastructures.
◆ Electric vehicles can represent additional load for the power system, but can also bring new opportunities for contributing to the efficient and resilient operations of the power grid.
◆ This is mainly because of their ability to provide back power to the system when it is not used for transportation, essentially serving as a moving battery source for the power grid.</td></tr>
<tr><td>2026-03-12</td><td>ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection</td><td>[2603.12409](http://arxiv.org/pdf/2603.12409)</td><td>◆ Although recent Open-Vocabulary Object Detection architectures, such as Grounding DINO, demonstrate strong zero-shot capabilities, their performance degrades significantly under domain shifts.
◆ Moreover, many domains of practical interest, such as nighttime or foggy scenes, lack large annotated datasets, preventing direct fine-tuning.
◆ In this paper, we introduce Aligned Basis Relocation for Adaptation(ABRA), a method that transfers class-specific detection knowledge from a labeled source domain to a target domain where no training images containing these classes are accessible.</td></tr>
<tr><td>2026-03-12</td><td>TopoBench: Benchmarking LLMs on Hard Topological Reasoning</td><td>[2603.12133](http://arxiv.org/pdf/2603.12133)</td><td>该论文的核心贡献是系统性地评估和诊断大语言模型在复杂拓扑推理任务上的能力瓶颈。

◆ 提出了TopoBench基准测试，包含六个拓扑网格谜题家族和三个难度等级，为评估大语言模型的拓扑推理能力提供了受控环境。
◆ 通过评估发现，即使是最先进的大语言模型，在困难实例上的解决率也低于四分之一，揭示了模型在此类任务上存在显著缺陷。
◆ 设计了一套错误分类法，并对750条思维链进行了人工标注，从而归纳出四种候选的因果失败模式，为深入分析模型失败原因提供了框架。
◆ 通过针对性的干预实验，验证了某些错误模式（如过早承诺和约束遗忘）会直接影响解题能力，而重复推理则是搜索过程中的良性行为。
◆ 研究了一系列缓解策略，最终定位出核心瓶颈在于模型难以从空间表征中提取约束，而非在已知约束上进行推理。</td></tr>
<tr><td>2026-03-12</td><td>Anomaly detection in time-series via inductive biases in the latent space of conditional normalizing flows</td><td>[2603.11756](http://arxiv.org/pdf/2603.11756)</td><td>该论文的核心贡献是提出了一种基于条件归一化流和状态空间模型的时序异常检测新框架，将异常定义从观测空间转移到了具有明确动态约束的潜空间。

◆ 创新性地将异常检测问题从观测空间的重建似然判断，转移到具有明确结构约束的潜空间中进行。
◆ 提出在条件归一化流中引入归纳偏置，通过离散时间状态空间框架来建模，强制潜表示遵循预设的时间动态演化规律。
◆ 在此框架下，正常行为被定义为符合预设的潜轨迹分布，而异常则被定义为对这些动态规律的违反。
◆ 将异常检测转化为一个基于统计的合规性检验问题，即通过拟合优度检验来评估观测数据映射到潜空间后是否服从预设的动态演化。
◆ 由此产生了一个有理论依据的决策规则，该规则即使在观测似然值很高的区域也能有效检测异常，并能提供模型合规性的可解释诊断。</td></tr>
<tr><td>2026-03-11</td><td>Crustal Structure Imaging of Ghana from Single-Station Ambient Noise Autocorrelations and Earthquake Arrival Time Inversion</td><td>[2603.10574](http://arxiv.org/pdf/2603.10574)</td><td>本论文针对加纳南部地壳结构成像问题，提出了创新的方法组合并取得了新的地质认识。其核心贡献与创新点如下：

◆ 创新性地将单台站环境噪声自相关技术应用于加纳地区，利用连续波形数据获得了高分辨率的零偏移距P波反射响应，从而清晰揭示了层状地壳结构。
◆ 发展了一套包含数据预处理、相位互相关和相位加权叠加的完整处理流程，有效增强了从环境噪声中提取可靠反射信号的稳健性。
◆ 通过联合反演本地地震的P波和S波走时数据，构建了该地区增强型的一维地壳速度模型，为自相关结果的深度转换提供了关键依据。
◆ 首次明确约束了沃尔坦盆地下方古生代基底的深度和形态，为理解该区域的构造演化和资源潜力提供了新的直接证据。
◆ 利用新速度模型重新定位地震事件并更新了地震目录，分析了地震活动的空间丛集特征，展示了被动源地震方法在仪器稀疏地区进行地壳成像和资源评估的有效性。</td></tr>
<tr><td>2026-03-09</td><td>The Struggle Between Continuation and Refusal: A Mechanistic Analysis of the Continuation-Triggered Jailbreak in LLMs</td><td>[2603.08234](http://arxiv.org/pdf/2603.08234)</td><td>该论文的核心贡献在于从机制可解释性角度，深入剖析了一种特定越狱攻击的内在原理。其创新点可总结如下：

◆ 聚焦并系统研究了一种“续写触发式越狱”现象，即仅通过调整指令后缀的位置就能显著提高越狱成功率，揭示了现有安全防御的一个脆弱环节。

◆ 首次在注意力头层面，运用因果干预和激活缩放等机制可解释性方法，对该越狱行为进行了全面的机理分析。

◆ 发现并论证了该越狱的根本机制源于模型内在的“续写驱动力”与对齐训练获得的“安全防御力”之间的固有竞争，为理解越狱提供了新颖的机制视角。

◆ 对识别出的安全关键注意力头进行了详细的行为分析，揭示了不同模型架构中安全头在功能和行为上的显著差异。

◆ 这些发现不仅深化了对大模型越狱漏洞根源的理论认识，也为从模型内部机制出发改进安全性提供了实践方向。</td></tr>
<tr><td>2026-03-13</td><td>ZK-ACE: Identity-Centric Zero-Knowledge Authorization for Post-Quantum Blockchain Systems</td><td>[2603.07974](http://arxiv.org/pdf/2603.07974)</td><td>ZK-ACE的核心贡献是提出了一种面向后量子区块链的身份中心零知识授权层，从根本上改变了传统的签名验证范式。其创新点主要包括：

◆ 彻底摒弃了交易携带大型后量子签名的模式，代之以身份绑定的零知识授权声明，将授权核心从验证具体签名转移到证明身份一致性上。

◆ 设计了基于确定性身份派生原语（DIDP）和链上紧凑身份承诺的架构，辅以防重放状态，作为新的链上身份锚点。

◆ 首次为这种新型授权模型形式化了严格的安全定义，包括授权健全性、抗重放、抗替换和跨域分离性，并基于标准假设给出了规约安全证明。

◆ 在协议层面实现了共识可见授权数据量的大幅削减（数量级减少），同时支持批量聚合和递归证明组合，兼容账户抽象和Rollup部署架构。</td></tr>
<tr><td>2026-03-09</td><td>RLPR: Radar-to-LiDAR Place Recognition via Two-Stage Asymmetric Cross-Modal Alignment for Autonomous Driving</td><td>[2603.07920](http://arxiv.org/pdf/2603.07920)</td><td>该论文提出了一种名为RLPR的鲁棒雷达到激光雷达地点识别框架，旨在解决全天候自动驾驶中跨模态定位的难题。其核心贡献与创新点如下：

◆ 提出了一种兼容多种雷达（单芯片、扫描和4D雷达）的雷达到激光雷达地点识别框架，实现了在现有激光雷达地图中对雷达扫描的精准定位。

◆ 设计了一个双流网络，用于提取抽象于传感器特定信号属性（如多普勒或RCS）的结构特征，增强了特征的判别性和跨模态泛化能力。

◆ 基于对雷达与激光雷达之间任务特定不对称性的观察，创新性地引入了一种两阶段非对称跨模态对齐策略。

◆ 该对齐策略利用预训练的雷达分支作为判别性锚点来指导对齐过程，有效克服了大规模配对训练数据稀缺和雷达信号异构的挑战。

实验表明，RLPR在四个数据集上取得了领先的识别精度，并展现出强大的零样本泛化能力。</td></tr>
<tr><td>2026-03-08</td><td>QdaVPR: A novel query-based domain-agnostic model for visual place recognition</td><td>[2603.07414](http://arxiv.org/pdf/2603.07414)</td><td>该论文提出了一种新颖的基于查询的、领域无关的视觉地点识别模型QdaVPR，其核心贡献与创新点如下：

◆ 提出了一种新颖的双层级对抗学习框架，该框架不仅促使构成全局描述符的查询特征具有领域不变性，还使生成这些查询的底层图像特征也具备领域不变性，从而显式地应对领域变化挑战。
◆ 设计了一种基于查询组合的三元组监督方法，通过组合不同的查询来增强全局描述符的判别力，以更好地区分不同的地点。
◆ 创新性地采用风格迁移方法对大规模VPR数据集进行增强，生成具有明确领域标签的多种合成领域数据，为对抗学习提供了关键的辅助监督信号。
◆ 通过上述方法的结合，QdaVPR实现了真正的领域无关性，无需针对特定目标域进行适配，便能泛化至未知的领域偏移。
◆ 在多个具有显著领域变化（如季节、昼夜、天气）的公开基准测试中，该模型取得了最先进的性能，特别是在Nordland、Tokyo24/7和SVOX数据集上获得了最高的Recall@1和Recall@10指标。</td></tr>
<tr><td>2026-03-07</td><td>pqRPKI: A Practical RPKI Architecture for the Post-Quantum Era</td><td>[2603.06968](http://arxiv.org/pdf/2603.06968)</td><td>该论文提出了pqRPKI，一种面向后量子时代的实用RPKI架构，旨在解决现有RPKI依赖的RSA密码体系易受量子计算攻击的问题。其核心贡献与创新点如下：

◆ 设计了多层Merkle树梯（MTL）与RPKI对象相结合的框架，将每个对象的验证材料从证书重新定位到清单（Manifest）中，从而适配后量子签名。
◆ 重新设计了RPKI清单和授权链，引入了基于梯子的同步与批量验证工作流，使验证者能够自上而下定位差异并自下而上重建树，优化了验证效率。
◆ 保持了现有RPKI对象和编码的兼容性，同时支持托管和委托操作模式，确保了与当前系统的平滑集成。
◆ 提供了一种增量式迁移路径，在与现有信任锚共存的双栈部署中，仅带来很小的尺寸开销（相比当前RPKI仓库仅增加3.4%）。
◆ 实际实现并评估表明，pqRPKI显著减少了仓库存储占用（平均546.8 MB），大幅缩短了完整验证周期时间（102.7秒），实现了端到端118.3秒的发布点到路由器的时效，支持每周期进行全仓库验证的低于2分钟的操作节奏。</td></tr>
<tr><td>2026-03-06</td><td>T2Nav Algebraic Topology Aware Temporal Graph Memory and Loop Detection for ZeroShot Visual Navigation</td><td>[2603.06918](http://arxiv.org/pdf/2603.06918)</td><td>该论文提出了一种名为T2Nav的零样本视觉导航系统，其核心贡献在于通过创新的图记忆与推理机制，解决了未知环境中高效、灵活导航的难题。

◆ 提出了一种代数拓扑感知的时序图记忆模型，能够整合异构的视觉与环境数据，构建并利用环境的结构化表示。
◆ 通过将视觉信息直接嵌入图节点并与环境实时匹配，实现了可靠的闭环检测，有效避免了冗余探索。
◆ 系统在探索与目标达成之间取得了良好平衡，支持基于目标物体实例参考图像的导航任务，能处理视觉相似但空间位置不同的实例。
◆ 整个系统无需针对新任务进行训练或调优，展现了强大的零样本适应能力，实验证明其在未知环境中具有高效和稳健的性能。</td></tr>
<tr><td>2026-03-06</td><td>Long-time behaviour of a nonlocal stochastic fractional reaction--diffusion equation arising in tumour dynamics</td><td>[2603.06414](http://arxiv.org/pdf/2603.06414)</td><td>该论文的核心贡献在于建立并分析了一个用于描述肿瘤动力学的随机非局部反应-扩散方程模型，并深入研究了其长时间行为。

◆ 提出了一个新颖的随机模型，该模型结合了分数阶拉普拉斯算子描述空间异常扩散和非局部反应项，并首次引入了乘法分数布朗运动作为具有时间相关性的随机扰动源。

◆ 在一般乘法分数噪声下，系统性地建立了模型的适定性，并完整刻画了导致解全局存在或在有限时间内爆破的参数区域。

◆ 针对线性乘法噪声情形，通过应用Doss-Sussmann变换，获得了更精确的解析结果，包括给出了爆破时间的显式上下界估计。

◆ 定量分析了噪声强度对肿瘤动力学的影响，揭示了噪声如何通过不同路径既能加速肿瘤进展（导致爆破），也可能增强抑制效果（导致灭绝），并提供了爆破概率的定量估计。

◆ 通过一维数值模拟，直观展示了异常扩散、分数噪声与非局部反应机制之间的复杂相互作用如何共同塑造系统的长时间动力学行为。</td></tr>
<tr><td>2026-03-06</td><td>PROBE: Probabilistic Occupancy BEV Encoding with Analytical Translation Robustness for 3D Place Recognition</td><td>[2603.05965](http://arxiv.org/pdf/2603.05965)</td><td>PROBE提出了一种免学习的激光雷达地点识别描述符，其核心贡献与创新点如下：

◆ 提出了一种概率化BEV占用编码方法，将每个鸟瞰图单元的占用状态建模为伯努利随机变量，从而更稳健地表示场景。

◆ 通过引入极坐标雅可比行列式，在连续笛卡尔平移空间上进行解析边缘化，避免了传统的离散点云扰动方法，计算效率高（O(R×S)时间）。

◆ 推导出具有距离自适应特性的角度不确定性σθ = σt / r，其中核心参数σt表示以米为单位的预期平移不确定性。该参数是传感器无关的物理量，无需针对不同数据集进行调整即可实现跨传感器泛化。

◆ 设计了一种综合的相似性度量方法，结合了伯努利KL杰卡德距离、指数不确定性门控以及基于FFT的高度余弦相似度，能够有效处理旋转对齐问题。

◆ 在涵盖四种不同类型激光雷达的四个数据集上验证，PROBE在多会话评估中取得了手工描述符中最高的准确率，在单会话评估中与手工及有监督基线方法相比也具有竞争力。</td></tr>
<tr><td>2026-03-06</td><td>Systematic Evaluation of Novel View Synthesis for Video Place Recognition</td><td>[2603.05876](http://arxiv.org/pdf/2603.05876)</td><td>该论文的核心贡献在于首次系统性地评估了合成新视角生成技术在视频地点识别任务中的影响与潜力。

◆ 首次在视频地点识别领域，系统性地评估了合成新视角（如地空视角转换）对识别性能的影响，填补了该研究空白。
◆ 构建了一个全面的评估框架，使用了五个公开的VPR图像数据库和七种典型的图像相似性计算方法，确保了评估的广泛性和可靠性。
◆ 通过实验得出了关键结论：当少量添加合成视图时，能够有效提升VPR的识别统计指标。
◆ 进一步发现，当添加大量视图时，视角变化的幅度大小并非最关键因素，而所添加视图的数量以及数据集中图像本身的类型（如地面或空中影像）更为重要。这一发现为实际应用提供了重要指导。</td></tr>
<tr><td>2026-03-06</td><td>EventGeM: Global-to-Local Feature Matching for Event-Based Visual Place Recognition</td><td>[2603.05807](http://arxiv.org/pdf/2603.05807)</td><td>该论文提出了EventGeM，一种用于基于事件的视觉位置识别（VPR）的先进方法。其核心贡献在于构建了一个高效且鲁棒的全局到局部特征匹配流程，能够在多种挑战性条件下实现实时、高精度的定位。

◆ 提出了一种新颖的全局到局部特征融合流水线。首先利用预训练的视觉变换器（ViT）从事件直方图图像中提取全局特征进行初始匹配，然后使用预训练的MaxViT检测局部关键点进行精细重排序。
◆ 创新性地引入基于2D单应性变换和RANSAC的局部特征重排序机制。这通过几何验证显著提升了匹配的准确性，尤其是在视角变化剧烈的场景中。
◆ 额外集成了预训练的视觉基础模型进行深度估计，用于比较查询图像与参考图像之间的结构相似性。这一步骤作为进一步的重排序优化，增强了算法在光照和外观变化下的鲁棒性。
◆ 整个系统在多个基准数据集上达到了最先进的性能，并且设计上支持在各种计算架构上实时运行，最终在真实机器人平台上使用事件流成功验证了在线定位能力。</td></tr>
<tr><td>2026-03-05</td><td>Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM</td><td>[2603.05397](http://arxiv.org/pdf/2603.05397)</td><td>该论文针对3D激光SLAM中闭环检测的可靠性问题，提出了一种名为CliReg的新型确定性算法。其核心贡献与创新点如下：

◆ 提出了一种全新的闭环验证算法CliReg，用确定性最大团搜索替代了传统的随机采样一致性（RANSAC）验证方法。
◆ 该方法通过构建特征匹配对的兼容性图，并在图中搜索最大团来求解，避免了RANSAC的随机采样过程，从而提高了算法的鲁棒性和确定性。
◆ 该算法能更有效地处理噪声和异常值，尤其在点云稀疏或环境特征模糊的条件下，能实现比RANSAC更低的位姿误差和更可靠的闭环。
◆ 研究将CliReg集成到了一个实时处理流程中，该流程结合了二进制3D描述子和基于汉明距离嵌入的二叉搜索树匹配方法。
◆ 通过在多种真实世界数据集和不同激光雷达上的测试，以及额外的2D投影地图实验，验证了该方法在不同传感器和空间域中的有效性与泛化能力。</td></tr>
<tr><td>2026-03-04</td><td>SSR: A Generic Framework for Text-Aided Map Compression for Localization</td><td>[2603.04272](http://arxiv.org/pdf/2603.04272)</td><td>该论文提出了一种用于机器人定位任务的文本辅助地图压缩通用框架，旨在解决大规模地图存储与传输成本高昂的问题。其核心贡献与创新点如下：

◆ 提出了一种新颖的文本增强压缩框架，将文本作为一种可被无损压缩的模态，与极小的图像特征向量结合，形成紧凑的地图表征。
◆ 引入了“互补信息”的关键思想，让文本描述与图像特征分别捕捉不同类型的信息，从而在压缩后仍能保持高保真的定位能力。
◆ 提出了名为相似性空间复制（SSR）的创新技术，该技术能够一次性学习一个自适应的图像嵌入，该嵌入专门捕获文本描述之外的“互补”视觉信息。
◆ 该框架是通用的，在包括视觉地点识别和以物体为中心的蒙特卡洛定位等多个下游任务上得到验证，适用于室内外多种场景。
◆ 实验表明，该方法在多个前沿数据集上实现了比现有基线方法高2倍的压缩性能，显著降低了内存和带宽占用。</td></tr>
<tr><td>2026-03-04</td><td>HBRB-BoW: A Retrained Bag-of-Words Vocabulary for ORB-SLAM via Hierarchical BRB-KMeans</td><td>[2603.04144](http://arxiv.org/pdf/2603.04144)</td><td>本文针对ORB-SLAM中传统二进制视觉词袋词汇表因精度损失导致性能下降的问题，提出了一种创新的词汇表训练方法。其核心贡献与创新点如下：

◆ 提出了一种名为HBRB-BoW的层次化词汇训练新算法，旨在解决传统基于k-majority的二进制词袋方法固有的结构缺陷。
◆ 创新地在层次化聚类过程中引入了全局实值流，从而在最终叶节点二值化之前，能够保留描述子的高保真度信息。
◆ 该方法有效缓解了传统二进制聚类无法表征细微特征分布的问题，减少了因误差在树结构中累积和传播所导致的视觉单词退化。
◆ 实验证明，所生成的词汇表比传统方法更具区分度且结构更优，显著提升了在复杂环境中视觉词典的表征完整性。
◆ 所生成的词汇文件可直接替换ORB-SLAM的默认文件，有望在不改动系统框架的情况下，直接提升其回环检测和重定位任务的性能。</td></tr>
<tr><td>2026-03-04</td><td>Long-Term Visual Localization in Dynamic Benthic Environments: A Dataset, Footprint-Based Ground Truth, and Visual Place Recognition Benchmark</td><td>[2603.04056](http://arxiv.org/pdf/2603.04056)</td><td>该论文针对动态海底环境中长期视觉定位研究不足的问题，提出了三项核心贡献。

◆ 创建了首个专为长期视觉定位定制的多地点、多年份海底数据集，包含长达六年的重复观测地理参考图像与高精度位姿数据。

◆ 提出了一种基于三维海底图像足迹的新地面真值生成方法，通过重叠足迹关联图像，确保真值反映实际的视觉内容重叠，而非单纯的地理位置接近。

◆ 利用上述数据集与真值方法，对八种先进视觉地点识别方法进行了基准测试，揭示了其在复杂海底场景中性能显著下降，并证明传统基于距离的真值方法在崎岖地形下会高估性能。

◆ 整体工作为动态海底环境的长期视觉定位研究提供了关键的数据集、更可靠的评估工具和性能基准，推动了该领域的发展。</td></tr>
<tr><td>2026-03-04</td><td>HE-VPR: Height Estimation Enabled Aerial Visual Place Recognition Against Scale Variance</td><td>[2603.04050](http://arxiv.org/pdf/2603.04050)</td><td>该论文提出了HE-VPR框架，其核心贡献是通过结合高度估计来解决航空视觉位置识别中的尺度变化难题。

◆ 主要创新在于将高度估计与位置识别解耦，两者共享一个冻结的DINOv2骨干网络，大幅降低了训练成本。
◆ 设计了两个轻量级旁路适配器分支：一个通过检索紧凑高度数据库来估计查询图像的高度分区，另一个则在对应高度子数据库中进行位置识别。
◆ 这种适配设计不仅减少了数据库的搜索空间，还结合了中心加权掩码策略，增强了对尺度差异的鲁棒性。
◆ 在自建的多高度数据集上验证，该框架在Recall@1指标上比基于ViT的先进基线提升了最高6.1%，同时内存使用减少了高达90%。
◆ 整体上，该工作为GNSS拒止环境提供了一个可扩展且高效的、具有高度感知能力的航空视觉位置识别解决方案，并开源了所有代码与数据集。</td></tr>
<tr><td>2026-03-03</td><td>Stochastic modeling of long-legged ant A. gracilipes locomotion in laboratory experiments</td><td>[2603.02665](http://arxiv.org/pdf/2603.02665)</td><td>该论文的核心贡献在于为长腿蚁（A. gracilipes）在实验室环境下的个体运动建立了随机模型，并通过实验数据验证了其有效性。其创新点可总结如下：

◆ 首次将主动布朗运动模型和奔跑-停顿模型相结合，用于描述长腿蚁的个体运动轨迹，该混合模型在定性和定量上均能复现实验观测到的轨迹统计特征。

◆ 通过大量实验室追踪实验，识别并量化了多个关键运动参数（如转向角、奔跑时间、等待时间）具有可重复性的概率分布，为运动行为提供了具体的统计基础。

◆ 成功实现了分析预测与从实验轨迹中经验测量的数据之间的良好吻合，验证了模型在解析和预测运动生态学方面的可靠性。

◆ 所建立的模型不仅可用于模拟和预测运动生态，还能为理解蚂蚁运动的内在生成机制及其感觉系统的工作原理提供新的见解。</td></tr>
<tr><td>2026-03-02</td><td>WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments</td><td>[2603.01475](http://arxiv.org/pdf/2603.01475)</td><td>该论文的核心贡献是创建了一个名为WildCross的大规模跨模态基准数据集，旨在解决自然非结构化环境中机器人感知的挑战。

◆ 首创性地针对复杂非结构化自然环境，构建了一个大规模跨模态基准，弥补了现有数据集主要集中于结构化城市环境的不足。
◆ 提供了规模庞大的数据，包含超过47.6万帧序列RGB图像，且每帧都配有半稠密深度和表面法线标注。
◆ 确保了多模态数据的高质量对齐，每帧图像均与精确的6自由度位姿及同步的稠密激光雷达子图严格对应。
◆ 通过全面的实验验证了数据集的价值，涵盖了视觉、激光雷达、跨模态地点识别以及度量深度估计等多个关键机器人感知任务。
◆ 公开了数据集和代码，旨在推动自然环境中多模态场景理解与机器人技术的研究发展。</td></tr>
<tr><td>2026-02-28</td><td>The On-Chain and Off-Chain Mechanisms of DAO-to-DAO Voting</td><td>[2603.00708](http://arxiv.org/pdf/2603.00708)</td><td>该论文的核心贡献是提出了一种识别和分析以太坊区块链上DAO-to-DAO元治理关系的方法，揭示了这种隐蔽治理层对去中心化决策透明度的挑战。

◆ 创新性地提出了一种基于签名匹配的算法，能够灵活识别多种DAO框架与投票方案下的链上交互，从而系统性地发现DAO之间的治理关联。
◆ 构建了一个包含61个DAO和72条元治理关系的网络图谱，首次以数据驱动的方式实证揭示了以太坊上DAO间元治理的规模与结构。
◆ 通过三个案例研究，归纳了元治理的多种形式，如战略性、决定性和中心化枢纽型，具体展示了元治理如何实际运作并影响决策。
◆ 实证论证了元治理会模糊投票背景、引入利己主义实体，并可能显著操控治理结果，从而指出了当前治理工具在揭示此类动态关系上的不足。
◆ 研究最终强调了开发增强型治理工具的迫切性，以应对元治理带来的透明度风险，维护DAO的核心精神。</td></tr>
<tr><td>2026-02-27</td><td>Altitude-Aware Visual Place Recognition in Top-Down View</td><td>[2602.23872](http://arxiv.org/pdf/2602.23872)</td><td>本文针对航空视觉位置识别在高度变化下的挑战，提出了一种创新的纯视觉解决方案。其核心贡献与创新点如下：

◆ 提出了一种高度自适应的视觉位置识别方法，通过分析图像中地面特征的密度来估计平台的相对高度，无需依赖气压计或飞行时间传感器等额外硬件。
◆ 设计了基于相对高度的图像裁剪策略，生成标准化的查询图像，以应对不同拍摄高度带来的图像尺度变化问题。
◆ 构建了一个分类式视觉位置识别框架，将处理后的图像用于精准定位，实现了仅凭视觉的三维场景识别。
◆ 该方法在多种地形和高度条件下表现出高精度与强鲁棒性。实验表明，其高度估计模块使检索准确率（R@1和R@5）大幅提升，且高度估计误差较传统单目深度估计方法显著降低。
◆ 整体方案为中小型空中平台提供了一个即插即用的实用解决方案，尤其适用于传感器有限且高度变化大的复杂环境。</td></tr>
<tr><td>2026-02-26</td><td>Fermi-LAT 16-year Source List</td><td>[2602.22148](http://arxiv.org/pdf/2602.22148)</td><td>本文基于费米大面积望远镜累积16年的观测数据，发布了早期16年源表（FL16Y），其核心贡献在于利用翻倍的数据量实现了对伽马射线天空更精确的普查。主要创新点如下：

◆ 利用长达16年的观测数据，显著提升了源定位精度，平均改善幅度达24%。

◆ 生成了一个全新的独立星表，而非在旧版基础上增量更新，这涉及对所有现有源进行重新定位与命名。

◆ 源表包含7220个源，在增加新发现源的同时，系统性地审查并更新了已有源的坐标与天体物理关联。

◆ 在星表分析流程中改进了若干环节，提升了数据处理的可靠性。

◆ 本次发布为早期版本，仍沿用上一版本（4FGL-DR4）的星际弥漫辐射模型，为未来最终完整星表的建立奠定了基础。</td></tr>
<tr><td>2026-02-25</td><td>Automatic Map Density Selection for Locally-Performant Visual Place Recognition</td><td>[2602.21473](http://arxiv.org/pdf/2602.21473)</td><td>该论文的核心贡献是提出了一种动态视觉位置识别地图构建方法，旨在根据用户定义的局部性能要求自动选择最合适的地图参考点密度，从而实现对局部性能的先验控制。

◆ 首次将满足用户定义的局部性能要求（而非全局平均性能）作为地图密度选择的明确目标，引入了“召回达成率”这一新指标来量化在多大比例的操作环境中能达到目标局部召回率。

◆ 提出了一种基于多参考轨迹匹配模式的预测方法，通过分析不同地图密度下的匹配模式，能够预测在未见过的部署数据上达到特定性能目标所需的地图密度。

◆ 所提出的系统能够自动选择满足局部性能要求的最小适宜地图密度，有效避免了不必要的过度密集化建图，从而节省了存储和计算资源。

◆ 通过大量实验验证了该方法的有效性，证明其能在用户指定的环境比例内稳定达到或超过目标局部召回率，并且所选密度操作点优于其他基线方法。

◆ 研究还揭示了传统的全局平均召回率指标并不能很好地预测对实际运营更有意义的局部召回达成率，凸显了所提新评估维度的重要性。</td></tr>
<tr><td>2026-02-24</td><td>LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments</td><td>[2602.20925](http://arxiv.org/pdf/2602.20925)</td><td>该论文提出了LST-SLAM，一个面向公里级动态环境的大规模立体热成像SLAM系统，旨在解决热成像SLAM在特征提取、运动跟踪和全局建图方面的核心难题。其核心贡献与创新点如下：

◆ 提出了一种自监督的热成像特征学习方法，提升了在复杂环境下特征提取的可靠性。
◆ 设计了立体双级运动跟踪策略，结合了直接法与特征法，增强了运动估计的稳定性。
◆ 引入了语义-几何混合约束，通过抑制帧间几何一致性弱的潜在动态特征，提高了系统在动态场景中的鲁棒性。
◆ 开发了在线增量词袋模型用于回环检测，并结合全局位姿优化，有效消除了累积漂移。
◆ 在公里级动态热成像数据集上的实验表明，该系统在鲁棒性与精度上显著优于AirSLAM、DROID-SLAM等现有代表性方案。</td></tr>
<tr><td>2026-02-24</td><td>Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change</td><td>[2602.20584](http://arxiv.org/pdf/2602.20584)</td><td>这篇论文针对长期环境监测中因外观剧烈变化导致三维重建失败的问题，提出了创新的解决方案。其核心贡献在于突破了现有方法在长期跨时段场景下的重建瓶颈。

◆ 指出了现有SfM流程在长期监测中的根本缺陷：它们依赖事后独立重建再对齐的策略，在跨时段外观剧烈变化时完全失效。
◆ 提出了新颖的联合SfM重建框架，将跨时段对应关系约束直接嵌入到重建过程中，而非事后处理，从而构建出统一连贯的三维模型。
◆ 设计了一种混合特征匹配策略，结合了手工特征与学习特征的互补优势，以鲁棒地建立跨越长时间间隔的图像对应关系。
◆ 引入了基于视觉位置识别的预筛选机制，仅对少量可能匹配的跨时段图像对使用计算昂贵的学习特征匹配，大幅提升了方法的可扩展性和鲁棒性。
◆ 在珊瑚礁等经历显著真实变化的长期数据集上验证了方法的有效性，在现有标准独立或联合SfM流程均失败的情况下，仍能实现跨时段的一致重建。</td></tr>
<tr><td>2026-02-24</td><td>Generative AI and Machine Learning Collaboration for Container Dwell Time Prediction via Data Standardization</td><td>[2602.20540](http://arxiv.org/pdf/2602.20540)</td><td>该论文的核心贡献是提出了一种生成式人工智能与机器学习协作的框架，以提升进口集装箱滞留时间的预测精度，从而优化码头堆场作业。

◆ 创新性地提出一个生成式人工智能与机器学习协同工作的框架，用于解决港口物流中的预测问题。
◆ 利用生成式人工智能将决定滞留时间的关键非结构化文本信息标准化为国际标准代码，使数据能被机器学习模型有效利用。
◆ 设计动态重新预测机制，当电子数据交换状态更新时触发，确保预测模型能基于最新信息进行更新。
◆ 通过真实数据实验验证，该方法使平均绝对误差相比传统模型提升13.88%，并将堆场翻箱率降低最高达14.68%。
◆ 从技术和方法论层面，实证了生成式人工智能在提升港口运营生产率方面的潜力与有效性。</td></tr>
<tr><td>2026-02-23</td><td>VGGT-MPR: VGGT-Enhanced Multimodal Place Recognition in Autonomous Driving Environments</td><td>[2602.19735](http://arxiv.org/pdf/2602.19735)</td><td>该论文提出VGGT-MPR，一种用于自动驾驶场景的多模态地点识别框架，其核心贡献与创新点如下：

◆ 首次采用视觉几何基础变换器（VGGT）作为统一的几何引擎，同时服务于全局检索和重排序两个阶段，实现了多模态特征的高效融合与利用。

◆ 在全局检索阶段，通过深度感知与点云图监督提取富含几何信息的视觉嵌入，并利用预测深度图增强稀疏激光雷达点云的结构表示，从而提升多模态特征的判别力。

◆ 设计了一种无需训练的重排序机制，利用VGGT的跨视图关键点跟踪能力，结合掩码引导的关键点提取与置信度感知的匹配评分，有效优化检索结果，无需额外参数优化。

◆ 整个框架避免了传统方法依赖手工融合策略与庞大参数主干网络的问题，降低了重新训练的成本。

◆ 在多个大规模自动驾驶数据集及自采数据上验证了方法的先进性，对环境变化、视角偏移与遮挡表现出强鲁棒性，性能达到当前最优水平。</td></tr>
<tr><td>2026-02-21</td><td>Marginalized Bundle Adjustment: Multi-View Camera Pose from Monocular Depth Estimates</td><td>[2602.18906](http://arxiv.org/pdf/2602.18906)</td><td>该论文的核心贡献是提出了一种名为边缘化光束法平差（MBA）的新方法，成功将单目深度估计（MDE）集成到传统运动恢复结构（SfM）流程中，解决了MDE深度图误差方差大、难以直接用于多视图几何的挑战。

◆ 核心创新在于提出了边缘化光束法平差（MBA）方法。该方法受现代RANSAC估计器启发，利用MDE深度图密集的特性，通过概率边缘化来有效抑制其高误差方差的影响。
◆ 首次系统性地论证了MDE深度图在精确多视图相机姿态估计方面的潜力。研究表明，经过MBA处理后，MDE的精度足以支持高质量的SfM，而无需依赖传统的稀疏三角化点云。
◆ 该方法在性能上实现了突破，在SfM和相机重定位任务上达到了业界领先或具有竞争力的水平。
◆ 展现了卓越的鲁棒性和可扩展性。经过广泛评估，该方法在不同规模的数据集上均表现稳定，既能处理少量图像，也能处理包含数千张图像的大型多视图系统。</td></tr>
<tr><td>2026-02-21</td><td>IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping</td><td>[2602.18709](http://arxiv.org/pdf/2602.18709)</td><td>该论文提出了IRIS-SLAM系统，其核心贡献在于通过统一的几何-实例表征，显著提升了语义SLAM的鲁棒性和地图一致性。

◆ 提出了一种新颖的RGB语义SLAM系统，它利用一个经过扩展的基础模型来生成统一的几何-实例表征。
◆ 扩展了一个几何基础模型，使其能够同时预测稠密几何信息和具有跨视角一致性的实例嵌入向量。
◆ 设计了一种语义协同关联机制，利用上述统一表征来增强数据关联的鲁棒性。
◆ 实现了实例引导的回环检测方法，有效提升了系统在宽基线场景下的回环闭合可靠性。
◆ 通过利用与视角无关的语义锚点，有效弥合了几何重建与开放词汇语义建图之间的鸿沟。
实验表明，该系统在地图一致性和宽基线回环可靠性方面显著优于现有先进方法。</td></tr>
<tr><td>2026-02-18</td><td>System Identification under Constraints and Disturbance: A Bayesian Estimation Approach</td><td>[2602.16358](http://arxiv.org/pdf/2602.16358)</td><td>该论文提出了一种新颖的贝叶斯系统辨识框架，旨在高精度联合估计机器人的状态轨迹和物理参数。其核心贡献与创新点如下：

◆ 提出一个集成多种物理约束的贝叶斯估计框架，将逆动力学、接触约束、闭环约束及关节摩擦模型作为严格的逐阶段等式约束嵌入，确保了物理一致性。

◆ 采用基于能量的回归器来增强参数的可观测性，并同时支持对惯性及驱动参数的等式与不等式先验，提高了估计的准确性和可靠性。

◆ 引入了动态一致的扰动投影方法，并通过在本体感知测量中增加能量观测，有效消除了非线性摩擦效应的歧义。

◆ 为保障算法可扩展性，推导出一种参数化的等式约束Riccati递归方法，该方法保留了问题的带状结构，实现了时间跨度上的线性计算复杂度，并开发了高效的计算导数。

◆ 在仿真和Unitree B1机器人硬件实验上验证了框架的优越性，相比基线方法，在收敛速度、惯性及摩擦估计误差、接触一致性方面均有提升，且将所得模型用于模型预测控制时，能显著改善在复杂环境中的运动跟踪性能。</td></tr>
<tr><td>2026-02-18</td><td>Markerless Robot Detection and 6D Pose Estimation for Multi-Agent SLAM</td><td>[2602.16308](http://arxiv.org/pdf/2602.16308)</td><td>该论文针对多智能体SLAM中数据关联困难的问题，提出了一种基于无标记物的机器人检测与6D位姿估计新方法，以提升团队间的相对定位精度。其核心贡献与创新点如下：

◆ 首次将基于深度学习的无标记6D位姿估计技术集成到去中心化多机器人SLAM系统中，替代了传统的标定标记阵列。
◆ 解决了传统基于AprilTag等标记物方法在观测距离、剧烈光照变化（如反光、过曝）及视角差异大时易失效的局限性。
◆ 通过机器人间的直接相互观测与位姿估计，为连接各机器人的局部SLAM图提供了更鲁棒的数据关联，有效应对环境感知歧义问题。
◆ 整个方案在行星类比环境的实地测试数据中进行了实验验证，证明了其对提升团队相对定位精度的有效性。</td></tr>
<tr><td>2026-02-17</td><td>Criteria-first, semantics-later: reproducible structure discovery in image-based sciences</td><td>[2602.15712](http://arxiv.org/pdf/2602.15712)</td><td>该论文的核心贡献是批判并颠覆了当前图像分析中“语义优先”的主流范式，提出了一种“标准优先、语义后置”的新方法论，以解决科学发现中的可复现性和长期可比性问题。

◆ 提出“标准优先、语义后置”的颠覆性分析范式，将结构发现与语义解释分离。先根据明确的数学或统计标准（而非领域标签）从图像中提取无语义的结构。

◆ 建立了一个统一框架，其上游进行由标准驱动的、无语义的结构发现，生成稳定的分割、结构场或层次；下游则将发现的结构明确映射到领域本体或词汇表。

◆ 该方法将语义重新定位为下游的显式映射步骤，从而支持对同一结构进行多种解释，并能在不更改上游分析的情况下，实现不同语义体系间的明确转换。

◆ 该范式根植于控制论、信息论等基础理论，强调“观察即区分”，并将信息与意义分离，为跨传感器、跨站点、长期监测的研究提供了可复现的分析支架。

◆ 推动了验证标准超越分类精度，并倡导将发现的结构产品本身作为符合FAIR原则、AI就绪的数字对象进行管理和重用，服务于长期监测和数字孪生等应用。</td></tr>
<tr><td>2026-02-18</td><td>AI-Paging: Lease-Based Execution Anchoring for Network-Exposed AI-as-a-Service</td><td>[2602.15286](http://arxiv.org/pdf/2602.15286)</td><td>该论文的核心贡献是提出了一种名为“AI-Paging”的网络架构，旨在解决6G网络中AI即服务（AIaaS）的动态匹配与可靠执行问题。其核心创新点如下：

◆ 提出了“AI寻呼”机制，将网络类比蜂窝寻呼，由网络侧根据用户意图、策略、信任和QoS约束，自动完成AI模型实例的发现、匹配与执行锚点选择，使用户无需感知底层复杂选择。

◆ 设计了基于租约的执行锚定控制平面事务，将用户意图解析为三个关键要素：AI服务身份、会话令牌和具有时效性的准入租约，只有持有有效租约才授权用户平面流量转向指定AI执行锚点。

◆ 强制执行两大关键不变性：一是租约门控的流量转向，确保无有效租约则不安装转发状态，增强了安全与控制；二是“先建后断”的锚点切换，保障动态网络条件下AIaaS服务的连续性与可靠性。

◆ 实现了与现有3GPP架构的兼容性，原型仅利用现有服务化控制、QoS流和基于策略的转向机制实现，无需新增报文头，便于部署。

◆ 通过原型评估验证了该架构在事务延迟、重定位中断、租约过期后的执行正确性以及移动与故障场景下的审计开销等方面的有效性。</td></tr>
<tr><td>2026-02-13</td><td>EPRBench: A High-Quality Benchmark Dataset for Event Stream Based Visual Place Recognition</td><td>[2602.12919](http://arxiv.org/pdf/2602.12919)</td><td>该论文的核心贡献是构建了一个高质量、大规模的事件流视觉地点识别基准数据集EPRBench，并提出了一个新颖的多模态融合框架。

◆ 针对事件流视觉地点识别领域专用数据集稀缺的问题，首次构建了一个大规模、高质量的基准数据集EPRBench，包含手持和车载设备采集的1万条事件序列和6.5万个事件帧，覆盖了多样化的视角、天气和光照挑战。
◆ 为支持语义感知研究，为数据集提供了由大语言模型生成并经人工精修的详细场景描述，为将大语言模型融入事件感知流程奠定了基础。
◆ 在数据集上系统性地实现并评估了15种前沿视觉地点识别算法，为未来的算法比较提供了坚实的基准。
◆ 创新性地提出了一种新颖的多模态融合范式：利用大语言模型从原始事件流生成文本描述，进而指导空间注意力令牌选择、跨模态特征融合与多尺度表征学习。
◆ 该框架不仅实现了高精度的地点识别，还能提供可解释的推理过程，显著增强了模型的透明度和可解释性。</td></tr>
<tr><td>2026-02-13</td><td>Topology and edge modes surviving criticality in non-Hermitian Floquet systems</td><td>[2602.12588](http://arxiv.org/pdf/2602.12588)</td><td>该论文的核心贡献在于将对称保护拓扑相的研究拓展到非厄米弗洛凯系统的无能隙临界区域，并建立了相应的拓扑理论。

◆ 将无能隙对称保护拓扑相的概念推广到非平衡系统，具体研究了周期驱动和非厄米耦合共同作用的体系。
◆ 针对具有子晶格对称性的一维模型，提出通过应用柯西辐角原理到广义布里渊区来定义拓扑绕数。
◆ 该拓扑绕数实现了统一的拓扑刻画，能在有能隙相和无能隙临界点同时成立，并建立了体边对应关系。
◆ 理论在一大类弗洛凯二分格点模型中得到验证，揭示了非厄米弗洛凯系统所独有的拓扑临界性。
◆ 研究发现了在驱动开放系统中存在的无能隙拓扑相，并揭示了超越平衡态相变点的鲁棒拓扑边缘模。</td></tr>
<tr><td>2026-02-12</td><td>DiffPlace: Street View Generation via Place-Controllable Diffusion Model Enhancing Place Recognition</td><td>[2602.11875](http://arxiv.org/pdf/2602.11875)</td><td>该论文提出DiffPlace框架，旨在解决现有多视角扩散模型在生成具有地点感知和背景一致性的街景图像方面的不足，以增强视觉地点识别的性能。

◆ 核心创新是引入了一个“地点ID控制器”，通过线性投影、感知器变换器和对比学习，将地点ID嵌入映射到固定的CLIP空间，从而实现对生成图像背景地点特征的精确控制。
◆ 该模型能够合成背景建筑保持一致的多视角街景图像，同时能灵活地修改前景物体和天气条件，实现了地点可控的生成。
◆ 通过将地点标识信息融入生成过程，DiffPlace生成了更真实、更具地点一致性的样本，有效支持了视觉地点识别任务的增强训练。
◆ 大量实验证明，该框架在生成质量和对地点识别模型的训练支持方面均优于现有方法，展示了生成模型在提升场景级、地点感知合成方面的潜力。</td></tr>
<tr><td>2026-02-11</td><td>Ctrl&amp;Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation</td><td>[2602.11440](http://arxiv.org/pdf/2602.11440)</td><td>该论文提出Ctrl&amp;Shift，一个用于视觉生成中高质量几何感知物体操纵的端到端扩散框架。其核心贡献在于首次在不依赖显式三维建模的情况下，统一了细粒度几何控制与现实世界泛化能力。具体创新点如下：

◆ 提出一种两阶段分解的操纵方法，将过程拆分为物体移除和显式相机姿态控制下的参考引导修复，并统一编码于单个扩散过程中。

◆ 设计了一种多任务、多阶段的训练策略，通过在不同任务中分离背景、物体身份和姿态信号，实现了精确且解耦的控制。

◆ 引入一个可扩展的真实世界数据集构建流程，能生成带有估计相对相机姿态的配对图像与视频样本，显著提升了方法的泛化能力。

◆ 该框架在背景保持、视角变化的几何一致性以及用户可控变换这三个核心目标上实现了最佳平衡，在保真度、视角一致性和可控性方面达到了先进水平。</td></tr>
<tr><td>2026-02-17</td><td>Geometry-Aware Rotary Position Embedding for Consistent Video World Model</td><td>[2602.07854](http://arxiv.org/pdf/2602.07854)</td><td>该论文的核心贡献是提出了一种几何感知的旋转位置编码方法，以解决视频世界模型中长期空间一致性的难题。现有模型因依赖屏幕空间位置编码而产生几何漂移，导致在长轨迹中场景结构不稳定。本文的创新点如下：

◆ 提出了ViewRope，一种几何感知编码机制，将相机光线方向直接注入视频Transformer的自注意力层，通过相对射线几何而非像素局部性来参数化注意力，从而为跨时间间隙检索三维一致内容提供了模型固有的归纳偏置。

◆ 设计了Geometry-Aware Frame-Sparse Attention，利用几何线索选择性地关注相关的历史帧，在保持记忆一致性的同时显著提升了计算效率。

◆ 构建了ViewBench诊断基准，专门用于评估模型在闭环轨迹中的保真度和几何漂移程度，为相关研究提供了量化工具。

实验证明，该方法能大幅提升长期一致性并降低计算成本。</td></tr>
<tr><td>2026-02-05</td><td>AnyThermal: Towards Learning Universal Representations for Thermal Perception</td><td>[2602.06203](http://arxiv.org/pdf/2602.06203)</td><td>◆ We present AnyThermal, a thermal backbone that captures robust task-agnostic thermal features suitable for a variety of tasks such as cross-modal place recognition, thermal segmentation, and monocular depth estimation using thermal images.
◆ Existing thermal backbones that follow task-specific training from small-scale data result in utility limited to a specific environment and task.
◆ Unlike prior methods, AnyThermal can be used for a wide range of environments (indoor, aerial, off-road, urban) and tasks, all without task-specific training.</td></tr>
<tr><td>2026-02-05</td><td>Location-Aware Dispersion on Anonymous Graphs</td><td>[2602.05948](http://arxiv.org/pdf/2602.05948)</td><td>◆ The well-studied DISPERSION problem is a fundamental coordination problem in distributed robotics, where a set of mobile robots must relocate so that each occupies a distinct node of a network.
◆ DISPERSION assumes that a robot can settle at any node as long as no other robot settles on that node.
◆ In this work, we introduce LOCATION-AWARE DISPERSION, a novel generalization of DISPERSION that incorporates location awareness: Let $G = (V, E)$ be an anonymous, connected, undirected graph with $n =</td></tr>
<tr><td>2026-02-05</td><td>VGGT-Motion: Motion-Aware Calibration-Free Monocular SLAM for Long-Range Consistency</td><td>[2602.05508](http://arxiv.org/pdf/2602.05508)</td><td>◆ Despite recent progress in calibration-free monocular SLAM via 3D vision foundation models, scale drift remains severe on long sequences.
◆ Motion-agnostic partitioning breaks contextual coherence and causes zero-motion drift, while conventional geometric alignment is computationally expensive.
◆ To address these issues, we propose VGGT-Motion, a calibration-free SLAM system for efficient and robust global consistency over kilometer-scale trajectories.</td></tr>
<tr><td>2026-02-04</td><td>S-MUSt3R: Sliding Multi-view 3D Reconstruction</td><td>[2602.04517](http://arxiv.org/pdf/2602.04517)</td><td>◆ The recent paradigm shift in 3D vision led to the rise of foundation models with remarkable capabilities in 3D perception from uncalibrated images.
◆ However, extending these models to large-scale RGB stream 3D reconstruction remains challenging due to memory limitations.
◆ This work proposes S-MUSt3R, a simple and efficient pipeline that extends the limits of foundation models for monocular 3D reconstruction.</td></tr>
<tr><td>2026-02-04</td><td>Quantile Transfer for Reliable Operating Point Selection in Visual Place Recognition</td><td>[2602.04401](http://arxiv.org/pdf/2602.04401)</td><td>◆ Visual Place Recognition (VPR) is a key component for localisation in GNSS-denied environments, but its performance critically depends on selecting an image matching threshold (operating point) that balances precision and recall.
◆ Thresholds are typically hand-tuned offline for a specific environment and fixed during deployment, leading to degraded performance under environmental change.
◆ We propose a method that, given a user-defined precision requirement, automatically selects the operating point of a VPR system to maximise recall.</td></tr>
<tr><td>2026-02-03</td><td>LaVPR: Benchmarking Language and Vision for Place Recognition</td><td>[2602.03253](http://arxiv.org/pdf/2602.03253)</td><td>◆ Visual Place Recognition (VPR) often fails under extreme environmental changes and perceptual aliasing.
◆ Furthermore, standard systems cannot perform &quot;blind&quot; localization from verbal descriptions alone, a capability needed for applications such as emergency response.
◆ To address these challenges, we introduce LaVPR, a large-scale benchmark that extends existing VPR datasets with over 650,000 rich natural-language descriptions.</td></tr>
<tr><td>2026-02-03</td><td>From Single Scan to Sequential Consistency: A New Paradigm for LIDAR Relocalization</td><td>[2602.03198](http://arxiv.org/pdf/2602.03198)</td><td>◆ LiDAR relocalization aims to estimate the global 6-DoF pose of a sensor in the environment.
◆ However, existing regression-based approaches are prone to dynamic or ambiguous scenarios, as they either solely rely on single-frame inference or neglect the spatio-temporal consistency across scans.
◆ In this paper, we propose TempLoc, a new LiDAR relocalization framework that enhances the robustness of localization by effectively modeling sequential consistency.</td></tr>
<tr><td>2026-02-02</td><td>Multi-Agent Monte Carlo Tree Search for Makespan-Efficient Object Rearrangement in Cluttered Spaces</td><td>[2602.02411](http://arxiv.org/pdf/2602.02411)</td><td>◆ Object rearrangement planning in complex, cluttered environments is a common challenge in warehouses, households, and rescue sites.
◆ Prior studies largely address monotone instances, whereas real-world tasks are often non-monotone-objects block one another and must be temporarily relocated to intermediate positions before reaching their final goals.
◆ In such settings, effective multi-agent collaboration can substantially reduce the time required to complete tasks.</td></tr>
<tr><td>2026-02-03</td><td>Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory</td><td>[2602.02393](http://arxiv.org/pdf/2602.02393)</td><td>◆ We propose Infinite-World, a robust interactive world model capable of maintaining coherent visual memory over 1000+ frames in complex real-world environments.
◆ While existing world models can be efficiently optimized on synthetic data with perfect ground-truth, they lack an effective training paradigm for real-world videos due to noisy pose estimations and the scarcity of viewpoint revisits.
◆ To bridge this gap, we first introduce a Hierarchical Pose-free Memory Compressor (HPMC) that recursively distills historical latents into a fixed-budget representation.</td></tr>
<tr><td>2026-02-02</td><td>A Two-Stage Stochastic Optimization Model for the Equitable Deployment of Fixed and Mobile Electric Vehicle Charging Stations</td><td>[2602.02333](http://arxiv.org/pdf/2602.02333)</td><td>◆ A major barrier to wide adoption of Electric Vehicles (EVs) is the absence of reliable and equitable charging infrastructure.
◆ Poorly located charging stations create coverage gaps and slow down EV adoption, especially in underserved communities.
◆ This paper proposes a two-stage stochastic mixed-integer programming model for the optimal deployment of Fixed and Mobile Charging Stations (FCSs and MCSs) across multiple zones and periods.</td></tr>
<tr><td>2026-02-03</td><td>Tidehunter: Large-Value Storage With Minimal Data Relocation</td><td>[2602.01873](http://arxiv.org/pdf/2602.01873)</td><td>◆ Log-Structured Merge-Trees (LSM-trees) dominate persistent key-value storage but suffer from high write amplification from 10x to 30x under random workloads due to repeated compaction.
◆ This overhead becomes prohibitive for large values with uniformly distributed keys, a workload common in content-addressable storage, deduplication systems, and blockchain validators.
◆ We present Tidehunter, a storage engine that eliminates value compaction by treating the Write-Ahead Log (WAL) as permanent storage rather than a temporary recovery buffer.</td></tr>
<tr><td>2026-02-02</td><td>Real-Time Loop Closure Detection in Visual SLAM via NetVLAD and Faiss</td><td>[2602.01673](http://arxiv.org/pdf/2602.01673)</td><td>◆ Loop closure detection (LCD) is a core component of simultaneous localization and mapping (SLAM): it identifies revisited places and enables pose-graph constraints that correct accumulated drift.
◆ Classic bag-of-words approaches such as DBoW are efficient but often degrade under appearance change and perceptual aliasing.
◆ In parallel, deep learning-based visual place recognition (VPR) descriptors (e.g., NetVLAD and Transformer-based models) offer stronger robustness, but their computational cost is often viewed as a barrier to real-time SLAM.</td></tr>
<tr><td>2026-02-03</td><td>TreeLoc: 6-DoF LiDAR Global Localization in Forests via Inter-Tree Geometric Matching</td><td>[2602.01501](http://arxiv.org/pdf/2602.01501)</td><td>◆ Reliable localization is crucial for navigation in forests, where GPS is often degraded and LiDAR measurements are repetitive, occluded, and structurally complex.
◆ These conditions weaken the assumptions of traditional urban-centric localization methods, which assume that consistent features arise from unique structural patterns, necessitating forest-centric solutions to achieve robustness in these environments.
◆ To address these challenges, we propose TreeLoc, a LiDAR-based global localization framework for forests that handles place recognition and 6-DoF pose estimation.</td></tr>
<tr><td>2026-01-31</td><td>Invariance on Manifolds: Understanding Robust Visual Representations for Place Recognition</td><td>[2602.00841](http://arxiv.org/pdf/2602.00841)</td><td>◆ Visual Place Recognition (VPR) demands representations robust to drastic environmental and viewpoint shifts.
◆ Current aggregation paradigms, however, either rely on data-hungry supervision or simplistic first-order statistics, often neglecting intrinsic structural correlations.
◆ In this work, we propose a Second-Order Geometric Statistics framework that inherently captures geometric stability without training.</td></tr>
<tr><td>2026-01-31</td><td>Refining Strokes by Learning Offset Attributes between Strokes for Flexible Sketch Edit at Stroke-Level</td><td>[2602.00489](http://arxiv.org/pdf/2602.00489)</td><td>◆ Sketch edit at stroke-level aims to transplant source strokes onto a target sketch via stroke expansion or replacement, while preserving semantic consistency and visual fidelity with the target sketch.
◆ Recent studies addressed it by relocating source strokes at appropriate canvas positions.
◆ However, as source strokes could exhibit significant variations in both size and orientation, we may fail to produce plausible sketch editing results by merely repositioning them without further adjustments.</td></tr>
<tr><td>2026-01-30</td><td>Fragmentation of a longitudinal population-scale social network: Decreasing network closure in the Netherlands</td><td>[2602.00234](http://arxiv.org/pdf/2602.00234)</td><td>◆ Population-level dynamics of social cohesion and its underlying mechanisms remain difficult to study.
◆ In this paper, we propose a network approach to measure the evolution of social cohesion at the population scale and identify mechanisms driving the change.
◆ We use twelve annual snapshots (2010-2021) of a population-scale social network from the Netherlands linking all residents through family, household, work, school, and neighbor relations.</td></tr>
<tr><td>2026-01-29</td><td>Advanced techniques and applications of LiDAR Place Recognition in Agricultural Environments: A Comprehensive Survey</td><td>[2601.22198](http://arxiv.org/pdf/2601.22198)</td><td>◆ An optimal solution to the localization problem is essential for developing autonomous robotic systems.
◆ Apart from autonomous vehicles, precision agriculture is one of the elds that can bene t most from these systems.
◆ Although LiDAR place recognition is a widely used technique in recent years to achieve accurate localization, it is mostly used in urban settings.</td></tr>
<tr><td>2026-01-27</td><td>VGGT-SLAM 2.0: Real time Dense Feed-forward Scene Reconstruction</td><td>[2601.19887](http://arxiv.org/pdf/2601.19887)</td><td>◆ We present VGGT-SLAM 2.0, a real time RGB feed-forward SLAM system which substantially improves upon VGGT-SLAM for incrementally aligning submaps created from VGGT.
◆ Firstly, we remove high-dimensional 15-degree-of-freedom drift and planar degeneracy from VGGT-SLAM by creating a new factor graph design while still addressing the reconstruction ambiguity of VGGT given unknown camera intrinsics.
◆ Secondly, by studying the attention layers of VGGT, we show that one of the layers is well suited to assist in image retrieval verification for free without additional training, which enables both rejecting false positive matches and allows for completing more loop closures.</td></tr>
<tr><td>2026-01-27</td><td>The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments</td><td>[2601.19557](http://arxiv.org/pdf/2601.19557)</td><td>◆ We release the S3LI Vulcano dataset, a multi-modal dataset towards development and benchmarking of Simultaneous Localization and Mapping (SLAM) and place recognition algorithms that rely on visual and LiDAR modalities.
◆ Several sequences are recorded on the volcanic island of Vulcano, from the Aeolian Islands in Sicily, Italy.
◆ The sequences provide users with data from a variety of environments, textures and terrains, including basaltic or iron-rich rocks, geological formations from old lava channels, as well as dry vegetation and water.</td></tr>
<tr><td>2026-01-27</td><td>Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity</td><td>[2601.19144](http://arxiv.org/pdf/2601.19144)</td><td>◆ This paper proposes a framework for improving the operational efficiency of automated storage systems under uncertainty.
◆ It considers a 2D grid-based storage for uniform-sized loads (e.g., containers, pallets, or totes), which are moved by a robot (or other manipulator) along a collision-free path in the grid.
◆ The loads are labeled (i.e., unique) and must be stored in a given sequence, and later be retrieved in a different sequence -- an operational pattern that arises in logistics applications, such as last-mile distribution centers and shipyards.</td></tr>
<tr><td>2026-01-26</td><td>Operationally induced preferred basis in unitary quantum mechanics</td><td>[2601.18856](http://arxiv.org/pdf/2601.18856)</td><td>◆ The preferred-basis problem and the definite-outcome aspect of the measurement problem persist even if the detector is modeled unitarily, because experimental data are necessarily represented in a Boolean event algebra of mutually exclusive records whereas the theoretical description is naturally formulated in a noncommutative operator algebra with continuous unitary symmetry.
◆ This change of mathematical type constitutes the core of the &#x27;cut&#x27;: a structurally necessary interface from group-based kinematics to set-based counting.
◆ In the presented view the basis relevant for recorded outcomes is not determined by the system Hamiltonian alone; it is induced by the measurement mapping, i.e., by the detector channel together with the coarse-grained readout that defines an instrument.</td></tr>
<tr><td>2026-01-26</td><td>Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning</td><td>[2601.18714](http://arxiv.org/pdf/2601.18714)</td><td>◆ Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks.
◆ Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art.
◆ In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach.</td></tr>
<tr><td>2026-01-26</td><td>MarioChart: Autonomous Tangibles as Active Proxy Interfaces for Embodied Casual Data Exploration</td><td>[2601.18328](http://arxiv.org/pdf/2601.18328)</td><td>◆ We introduce the notion of an Active Proxy interface, i.e.
◆ tangible models as proxies for physical data referents, supporting interactive exploration of data through active manipulation.
◆ We realise an active proxy data visualisation system, &quot;MarioChart&quot;, using robot carts relocating themselves on a tabletop, e.g., to align with their data referents in a map or other visual layout.</td></tr>
<tr><td>2026-01-21</td><td>Variable Stepsize Distributed Forward-Backward Splitting Methods as Relocated Fixed-Point Iterations</td><td>[2601.15531](http://arxiv.org/pdf/2601.15531)</td><td>◆ We present a family of distributed forward-backward methods with variable stepsizes to find a solution of structured monotone inclusion problems.
◆ The framework is constructed by means of relocated fixed-point iterations, extending the approach introduced in arXiv:2507.07428 to conically averaged operators, thus including iteration operators for methods of forward-backward type devised by graphs.
◆ The family of methods we construct preserve the per-iteration computational cost and the convergence properties of their constant stepsize counterparts.</td></tr>
<tr><td>2026-01-21</td><td>Multi-Input Ciphertext Multiplication for Homomorphic Encryption</td><td>[2601.15401](http://arxiv.org/pdf/2601.15401)</td><td>◆ Homomorphic encryption (HE) enables arithmetic operations to be performed directly on encrypted data.
◆ It is essential for privacy-preserving applications such as machine learning, medical diagnosis, and financial data analysis.
◆ In popular HE schemes, ciphertext multiplication is only defined for two inputs.</td></tr>
<tr><td>2026-01-21</td><td>Designing DNA nanostar hydrogels with programmable degradation and antibody release</td><td>[2601.14934](http://arxiv.org/pdf/2601.14934)</td><td>◆ DNA nanostar (DNAns) hydrogels are promising materials for in vivo applications, including tissue regeneration and drug and antibody delivery.
◆ However, a systematic and quantitative understanding of the design principles controlling their degradation is lacking.
◆ Here, we investigate hydrogels made of three-armed DNAns with varying flexible joints, arm lengths, and mesh sizes and use restriction enzymes to cut the DNAns structures while monitoring the gel&#x27;s degradation.</td></tr>
<tr><td>2026-01-21</td><td>DroneVLA: VLA based Aerial Manipulation</td><td>[2601.13809](http://arxiv.org/pdf/2601.13809)</td><td>◆ As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally.
◆ This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user.
◆ The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera.</td></tr>
<tr><td>2026-01-20</td><td>Why Does the LLM Stop Computing: An Empirical Study of User-Reported Failures in Open-Source LLMs</td><td>[2601.13655](http://arxiv.org/pdf/2601.13655)</td><td>◆ The democratization of open-source Large Language Models (LLMs) allows users to fine-tune and deploy models on local infrastructure but exposes them to a First Mile deployment landscape.
◆ Unlike black-box API consumption, the reliability of user-managed orchestration remains a critical blind spot.
◆ To bridge this gap, we conduct the first large-scale empirical study of 705 real-world failures from the open-source DeepSeek, Llama, and Qwen ecosystems.</td></tr>
<tr><td>2026-01-19</td><td>DC-VLAQ: Query-Residual Aggregation for Robust Visual Place Recognition</td><td>[2601.12729](http://arxiv.org/pdf/2601.12729)</td><td>◆ One of the central challenges in visual place recognition (VPR) is learning a robust global representation that remains discriminative under large viewpoint changes, illumination variations, and severe domain shifts.
◆ While visual foundation models (VFMs) provide strong local features, most existing methods rely on a single model, overlooking the complementary cues offered by different VFMs.
◆ However, exploiting such complementary information inevitably alters token distributions, which challenges the stability of existing query-based global aggregation schemes.</td></tr>
<tr><td>2026-01-16</td><td>Modular and Mobile Capacity Planning for Hyperconnected Supply Chain Networks</td><td>[2601.11107](http://arxiv.org/pdf/2601.11107)</td><td>◆ The increased volatility of markets and the pressing need for resource sustainability are driving supply chains towards more agile, distributed, and dynamic designs.
◆ Motivated by the Physical Internet initiative, we introduce the Dynamic Stochastic Modular and Mobile Capacity Planning (DSMMCP) problem, which fosters hyperconnectivity through a network-of-networks architecture with modular and mobile capacities.
◆ The problem addresses both demand and supply uncertainties by incorporating short-term leasing of modular facilities and dynamic relocation of resources.</td></tr>
<tr><td>2026-01-15</td><td>A Unified Framework for Kinematic Simulation of Rigid Foldable Structures</td><td>[2601.10225](http://arxiv.org/pdf/2601.10225)</td><td>◆ Origami-inspired structures with rigid panels now span thick, kirigami, and multi-sheet realizations, making unified kinematic analysis essential.
◆ Yet a general method that consolidates their loop constraints has been lacking.
◆ We present an automated approach that generates the Pfaffian constraint matrix for arbitrary rigid foldable structures (RFS).</td></tr>
<tr><td>2026-01-14</td><td>Hybrid guided variational autoencoder for visual place recognition</td><td>[2601.09248](http://arxiv.org/pdf/2601.09248)</td><td>◆ Autonomous agents such as cars, robots and drones need to precisely localize themselves in diverse environments, including in GPS-denied indoor environments.
◆ One approach for precise localization is visual place recognition (VPR), which estimates the place of an image based on previously seen places.
◆ State-of-the-art VPR models require high amounts of memory, making them unwieldy for mobile deployment, while more compact models lack robustness and generalization capabilities.</td></tr>
<tr><td>2026-01-13</td><td>Keyframe-based Dense Mapping with the Graph of View-Dependent Local Maps</td><td>[2601.08520](http://arxiv.org/pdf/2601.08520)</td><td>◆ In this article, we propose a new keyframe-based mapping system.
◆ The proposed method updates local Normal Distribution Transform maps (NDT) using data from an RGB-D sensor.
◆ The cells of the NDT are stored in 2D view-dependent structures to better utilize the properties and uncertainty model of RGB-D cameras.</td></tr>
<tr><td>2026-01-13</td><td>CogniMap3D: Cognitive 3D Mapping and Rapid Retrieval</td><td>[2601.08175](http://arxiv.org/pdf/2601.08175)</td><td>◆ We present CogniMap3D, a bioinspired framework for dynamic 3D scene understanding and reconstruction that emulates human cognitive processes.
◆ Our approach maintains a persistent memory bank of static scenes, enabling efficient spatial knowledge storage and rapid retrieval.
◆ CogniMap3D integrates three core capabilities: a multi-stage motion cue framework for identifying dynamic objects, a cognitive mapping system for storing, recalling, and updating static scenes across multiple visits, and a factor graph optimization strategy for refining camera poses.</td></tr>
<tr><td>2026-01-12</td><td>Anisotropic anomalous Hall effect in distorted kagome GdTi3Bi4</td><td>[2601.07578](http://arxiv.org/pdf/2601.07578)</td><td>◆ Topological kagome magnets offer a rich landscape for exploring the intricate interplay of quantum interactions among geometry, topology, spin, and correlation.
◆ GdTi3Bi4 crystallizes in layered Ti based kagome nets intertwined with zigzag Gd chains along the a axis and orders antiferromagnetically below 15 K.
◆ Here, we present the temperature and field dependent electrical transport of GdTi3Bi4 in different directions.</td></tr>
<tr><td>2026-01-11</td><td>SARA: Scene-Aware Reconstruction Accelerator</td><td>[2601.06831](http://arxiv.org/pdf/2601.06831)</td><td>◆ We present SARA (Scene-Aware Reconstruction Accelerator), a geometry-driven pair selection module for Structure-from-Motion (SfM).
◆ Unlike conventional pipelines that select pairs based on visual similarity alone, SARA introduces geometry-first pair selection by scoring reconstruction informativeness - the product of overlap and parallax - before expensive matching.
◆ A lightweight pre-matching stage uses mutual nearest neighbors and RANSAC to estimate these cues, then constructs an Information-Weighted Spanning Tree (IWST) augmented with targeted edges for loop closure, long-baseline anchors, and weak-view reinforcement.</td></tr>
<tr><td>2026-01-11</td><td>Reconfiguration of Hamiltonian Paths and Cycles in Rectangular Grid Graphs</td><td>[2601.06749](http://arxiv.org/pdf/2601.06749)</td><td>◆ \noindent An \textit{\(m \times n\) grid graph} is the induced subgraph of the square lattice whose vertex set consists of all integer grid points \(\{(i,j) : 0 \leq i &lt; m,\ 0 \leq j &lt; n\}\).
◆ Let $H$ and $K$ be Hamiltonian cycles in an $m \times n$ grid graph $G$.
◆ We study the problem of reconfiguring $H$ into $K$ using a sequence of local transformations called \textit{moves}.</td></tr>
<tr><td>2026-01-11</td><td>Imaginary Gauge-steerable Edge Modes In Non-Hermitian Aubry-André-Harper Model</td><td>[2601.06746](http://arxiv.org/pdf/2601.06746)</td><td>◆ We investigate a non-Hermitian Aubry-André-Harper lattice exhibiting quasiperiodicity, featuring an imaginary gauge field that varies spatially but averages to zero.
◆ In the presence of open boundary conditions, this system is precisely mapped, through a nonunitary gauge transformation, to the Hermitian AAH model with balanced hopping terms.
◆ The mapping leaves the spectrum unchanged but reshapes each eigenfunction by a realization-dependent random-walk envelope.</td></tr>
<tr><td>2026-01-10</td><td>WHU-PCPR: A cross-platform heterogeneous point cloud dataset for place recognition in complex urban scenes</td><td>[2601.06442](http://arxiv.org/pdf/2601.06442)</td><td>◆ Point Cloud-based Place Recognition (PCPR) demonstrates considerable potential in applications such as autonomous driving, robot localization and navigation, and map update.
◆ In practical applications, point clouds used for place recognition are often acquired from different platforms and LiDARs across varying scene.
◆ However, existing PCPR datasets lack diversity in scenes, platforms, and sensors, which limits the effective development of related research.</td></tr>
<tr><td>2026-01-09</td><td>InsSo3D: Inertial Navigation System and 3D Sonar SLAM for turbid environment inspection</td><td>[2601.05805](http://arxiv.org/pdf/2601.05805)</td><td>◆ This paper presents InsSo3D, an accurate and efficient method for large-scale 3D Simultaneous Localisation and Mapping (SLAM) using a 3D Sonar and an Inertial Navigation System (INS).
◆ Unlike traditional sonar, which produces 2D images containing range and azimuth information but lacks elevation information, 3D Sonar produces a 3D point cloud, which therefore does not suffer from elevation ambiguity.
◆ We introduce a robust and modern SLAM framework adapted to the 3D Sonar data using INS as prior, detecting loop closure and performing pose graph optimisation.</td></tr>
<tr><td>2026-01-08</td><td>Earthquakes and cluster dynamics during Interseismic phases between the Northern and Central Apennines (Italy)</td><td>[2601.04829](http://arxiv.org/pdf/2601.04829)</td><td>◆ In the last thirty years, the Northern and Central Apennines (Italy) have been affected by three main destructive seismic sequences: the 1997 Colfiorito (three events $M_L &gt; 5.5$), the 2009 L&#x27;Aquila (one event $M_L &gt; 5.5$), and the 2016--2017 Amatrice--Visso--Norcia (three events $M_L &gt; 5.5$).
◆ Several studies have analysed the spatio-temporal evolution and processes driving each sequence, focusing mainly on the foreshock--mainshock--aftershock periods.
◆ Here, we focus on the 2018--2024 interseismic phase, aiming to unravel the long-term seismogenic behaviour of this region.</td></tr>
<tr><td>2026-01-07</td><td>Multi-agent Optimization of Non-cooperative Multimodal Mobility Systems</td><td>[2601.03777](http://arxiv.org/pdf/2601.03777)</td><td>◆ While multimodal mobility systems have the potential to bring many benefits to travelers, drivers, the environment, and traffic congestion, such systems typically involve multiple non-cooperative decision-makers who may selfishly optimize their own objectives without considering the overall system benefits.
◆ This paper aims to investigate market-based interactions of travelers and ride-sourcing drivers in the context of multimodal mobility systems.
◆ We propose a unified mathematical modeling framework to capture the decentralized travelers and drivers&#x27; decision-making process and balance the network&#x27;s demand and supply by equilibrium pricing.</td></tr>
<tr><td>2026-01-06</td><td>Loop Closure using AnyLoc Visual Place Recognition in DPV-SLAM</td><td>[2601.02723](http://arxiv.org/pdf/2601.02723)</td><td>◆ Loop closure is crucial for maintaining the accuracy and consistency of visual SLAM.
◆ We propose a method to improve loop closure performance in DPV-SLAM.
◆ Our approach integrates AnyLoc, a learning-based visual place recognition technique, as a replacement for the classical Bag of Visual Words (BoVW) loop detection method.</td></tr>
<tr><td>2026-01-03</td><td>From Random Walks to Thermal Rides: Universal Anomalous Transport in Soaring Flights</td><td>[2601.01293](http://arxiv.org/pdf/2601.01293)</td><td>◆ Cross-country soaring flights rely on intermittent atmospheric updrafts to cover long distances, producing trajectories that alternate between rapid relocation and local exploration.
◆ From a large dataset of paraglider, hang glider, and sailplane flights, we uncover a universal transport law: beyond short ballistic times, horizontal motion is persistently sub-ballistic, with a Hurst exponent $\approx 0.88$ largely independent of aircraft type.
◆ Phase-resolved analysis using a probabilistic segmentation method shows that this scaling arises from the fundamentally intermittent, two-dimensional, and directionally correlated nature of soaring transport, in which successive ballistic segments do not add coherently.</td></tr>
<tr><td>2025-12-31</td><td>Vibe Coding, Interface Flattening</td><td>[2512.24939](http://arxiv.org/pdf/2512.24939)</td><td>◆ Large language models are reshaping programming by enabling &#x27;vibe coding&#x27;: the development of softwares through natural-language interaction with model-driven toolchains.
◆ This article argues that vibe coding is best understood as interface flattening, a reconfiguration in which previously distinct modalities (GUI, CLI, and API) appear to converge into a single conversational surface, even as the underlying chain of translation from intention to machinic effect lengthens and thickens.
◆ Drawing on Friedrich Kittler&#x27;s materialist media theory and Alexander Galloway&#x27;s account of interfaces as sites of protocol control, the paper situates programming as a historically localised interface arrangement rather than an essential relation to computation.</td></tr>
<tr><td>2025-12-31</td><td>Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand: Toward Dexterous Manipulation for Payload Constrained Humanoids</td><td>[2512.24657](http://arxiv.org/pdf/2512.24657)</td><td>◆ Humanoid robots toward human-level dexterity require robotic hands capable of simultaneously providing high grasping force, rapid actuation speeds, multiple degrees of freedom, and lightweight structures within human-like size constraints.
◆ Meeting these conflicting requirements remains challenging, as satisfying this combination typically necessitates heavier actuators and bulkier transmission systems, significantly restricting the payload capacity of robot arms.
◆ In this letter, we present a lightweight anthropomorphic hand actuated by Bowden cables, which uniquely combines rolling-contact joint optimization with antagonistic cable actuation, enabling single-motor-per-joint control with negligible cable-length deviation.</td></tr>
<tr><td>2025-12-30</td><td>Geometric Multi-Session Map Merging with Learned Local Descriptors</td><td>[2512.24384](http://arxiv.org/pdf/2512.24384)</td><td>◆ Multi-session map merging is crucial for extended autonomous operations in large-scale environments.
◆ In this paper, we present GMLD, a learning-based local descriptor framework for large-scale multi-session point cloud map merging that systematically aligns maps collected across different sessions with overlapping regions.
◆ The proposed framework employs a keypoint-aware encoder and a plane-based geometric transformer to extract discriminative features for loop closure detection and relative pose estimation.</td></tr>
<tr><td>2025-12-26</td><td>Reloc-VGGT: Visual Re-localization with Geometry Grounded Transformer</td><td>[2512.21883](http://arxiv.org/pdf/2512.21883)</td><td>◆ Visual localization has traditionally been formulated as a pair-wise pose regression problem.
◆ Existing approaches mainly estimate relative poses between two images and employ a late-fusion strategy to obtain absolute pose estimates.
◆ However, the late motion average is often insufficient for effectively integrating spatial information, and its accuracy degrades in complex environments.</td></tr>
<tr><td>2025-12-28</td><td>UniPR-3D: Towards Universal Visual Place Recognition with Visual Geometry Grounded Transformer</td><td>[2512.21078](http://arxiv.org/pdf/2512.21078)</td><td>◆ Visual Place Recognition (VPR) has been traditionally formulated as a single-image retrieval task.
◆ Using multiple views offers clear advantages, yet this setting remains relatively underexplored and existing methods often struggle to generalize across diverse environments.
◆ In this work we introduce UniPR-3D, the first VPR architecture that effectively integrates information from multiple views.</td></tr>
<tr><td>2025-12-21</td><td>Bridging the divide: Economic exchange and segregation in dual-income cities</td><td>[2512.18680](http://arxiv.org/pdf/2512.18680)</td><td>◆ Segregation is a growing concern around the world.
◆ One of its main manifestations is the creation of ghettos, whose inhabitants have difficult access to well-paid jobs, which are often located far from their homes.
◆ In order to study this phenomenon, we propose an extension of Schelling&#x27;s model of segregation to take into account the existence of economic exchanges.</td></tr>
<tr><td>2025-12-21</td><td>Text2Graph VPR: A Text-to-Graph Expert System for Explainable Place Recognition in Changing Environments</td><td>[2512.18613](http://arxiv.org/pdf/2512.18613)</td><td>◆ Visual Place Recognition (VPR) in long-term deployment requires reasoning beyond pixel similarity: systems must make transparent, interpretable decisions that remain robust under lighting, weather and seasonal change.
◆ We present Text2Graph VPR, an explainable semantic localization system that converts image sequences into textual scene descriptions, parses those descriptions into structured scene graphs, and reasons over the resulting graphs to identify places.
◆ Scene graphs capture objects, attributes and pairwise relations; we aggregate per-frame graphs into a compact place representation and perform retrieval with a dual-similarity mechanism that fuses learned Graph Attention Network (GAT) embeddings and a Shortest-Path (SP) kernel for structural matching.</td></tr>
<tr><td>2025-12-23</td><td>UniMPR: A Unified Framework for Multimodal Place Recognition with Heterogeneous Sensor Configurations</td><td>[2512.18279](http://arxiv.org/pdf/2512.18279)</td><td>◆ Place recognition is a critical component of autonomous vehicles and robotics, enabling global localization in GPS-denied environments.
◆ Recent advances have spurred significant interest in multimodal place recognition (MPR), which leverages complementary strengths of multiple modalities.
◆ Despite its potential, most existing MPR methods still face three key challenges: (1) dynamically adapting to arbitrary modality inputs within a unified framework, (2) maintaining robustness with missing or degraded modalities, and (3) generalizing across diverse sensor configurations and setups.</td></tr>
<tr><td>2025-12-19</td><td>Exploring the Effect of Basis Rotation on NQS Performance</td><td>[2512.17893](http://arxiv.org/pdf/2512.17893)</td><td>◆ Neural Quantum States (NQS) use neural networks to represent wavefunctions of quantum many-body systems, but their performance depends on the choice of basis, yet the underlying mechanism remains poorly understood.
◆ We use a fully solvable one-dimensional Ising model to show that local basis rotations leave the loss landscape unchanged while relocating the exact wavefunction in parameter space, effectively increasing its geometric distance from typical initializations.
◆ By sweeping a rotation angle, we compute quantum Fisher information and Fubini-Study distances to quantify how the rotated wavefunction moves within the loss landscape.</td></tr>
<tr><td>2025-12-19</td><td>Autonomous Picosecond-Precision Synchronization in Measurement-Device-Independent Quantum Key Distribution</td><td>[2512.17510](http://arxiv.org/pdf/2512.17510)</td><td>◆ Measurement-device-independent quantum key distribution (MDI-QKD) eliminates detector side-channel attacks by relocating all measurements to an untrusted intermediate node.
◆ However, its practical implementation critically relies on picosecond-level temporal synchronization between spatially separated users.
◆ In this work, we present a physically motivated autonomous synchronization algorithm for fiber-based MDI-QKD networks that does not require auxiliary optical channels or shared clock references.</td></tr>
<tr><td>2025-12-17</td><td>NAP3D: NeRF Assisted 3D-3D Pose Alignment for Autonomous Vehicles</td><td>[2512.15080](http://arxiv.org/pdf/2512.15080)</td><td>◆ Accurate localization is essential for autonomous vehicles, yet sensor noise and drift over time can lead to significant pose estimation errors, particularly in long-horizon environments.
◆ A common strategy for correcting accumulated error is visual loop closure in SLAM, which adjusts the pose graph when the agent revisits previously mapped locations.
◆ These techniques typically rely on identifying visual mappings between the current view and previously observed scenes and often require fusing data from multiple sensors.</td></tr>
<tr><td>2025-12-16</td><td>Odyssey: An Automotive Lidar-Inertial Odometry Dataset for GNSS-denied situations</td><td>[2512.14428](http://arxiv.org/pdf/2512.14428)</td><td>◆ The development and evaluation of Lidar-Inertial Odometry (LIO) and Simultaneous Localization and Mapping (SLAM) systems requires a precise ground truth.
◆ The Global Navigation Satellite System (GNSS) is often used as a foundation for this, but its signals can be unreliable in obstructed environments due to multi-path effects or loss-of-signal.
◆ While existing datasets compensate for the sporadic loss of GNSS signals by incorporating Inertial Measurement Unit (IMU) measurements, the commonly used Micro-Electro-Mechanical Systems (MEMS) or Fiber Optic Gyroscope (FOG)-based systems do not permit the prolonged study of GNSS-denied environments.</td></tr>
<tr><td>2025-12-16</td><td>SUPER -- A Framework for Sensitivity-based Uncertainty-aware Performance and Risk Assessment in Visual Inertial Odometry</td><td>[2512.14189](http://arxiv.org/pdf/2512.14189)</td><td>◆ While many visual odometry (VO), visual-inertial odometry (VIO), and SLAM systems achieve high accuracy, the majority of existing methods miss to assess risks at runtime.
◆ This paper presents SUPER (Sensitivity-based Uncertainty-aware PErformance and Risk assessment) that is a generic and explainable framework that propagates uncertainties via sensitivities for real-time risk assessment in VIO.
◆ The scientific novelty lies in the derivation of a real-time risk indicator that is backend-agnostic and exploits the Schur complement blocks of the Gauss-Newton normal matrix to propagate uncertainties.</td></tr>
<tr><td>2025-12-16</td><td>ACE-SLAM: Scene Coordinate Regression for Neural Implicit Real-Time SLAM</td><td>[2512.14032](http://arxiv.org/pdf/2512.14032)</td><td>◆ We present a novel neural RGB-D Simultaneous Localization And Mapping (SLAM) system that learns an implicit map of the scene in real time.
◆ For the first time, we explore the use of Scene Coordinate Regression (SCR) as the core implicit map representation in a neural SLAM pipeline, a paradigm that trains a lightweight network to directly map 2D image features to 3D global coordinates.
◆ SCR networks provide efficient, low-memory 3D map representations, enable extremely fast relocalization, and inherently preserve privacy, making them particularly suitable for neural implicit SLAM.</td></tr>
<tr><td>2025-12-15</td><td>CogniEdit: Dense Gradient Flow Optimization for Fine-Grained Image Editing</td><td>[2512.13276](http://arxiv.org/pdf/2512.13276)</td><td>◆ Instruction-based image editing with diffusion models has achieved impressive results, yet existing methods struggle with fine-grained instructions specifying precise attributes such as colors, positions, and quantities.
◆ While recent approaches employ Group Relative Policy Optimization (GRPO) for alignment, they optimize only at individual sampling steps, providing sparse feedback that limits trajectory-level control.
◆ We propose a unified framework CogniEdit, combining multi-modal reasoning with dense reward optimization that propagates gradients across consecutive denoising steps, enabling trajectory-level gradient flow through the sampling process.</td></tr>
<tr><td>2025-12-15</td><td>Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing</td><td>[2512.13055](http://arxiv.org/pdf/2512.13055)</td><td>◆ Visual Place Recognition (VPR) has advanced significantly with high-capacity foundation models like DINOv2, achieving remarkable performance.
◆ Nonetheless, their substantial computational cost makes deployment on resource-constrained devices impractical.
◆ In this paper, we introduce an efficient asymmetric VPR framework that incorporates a high-capacity gallery model for offline feature extraction with a lightweight query network for online processing.</td></tr>
<tr><td>2025-12-15</td><td>Linear convergence of relocated fixed-point iterations</td><td>[2512.12954](http://arxiv.org/pdf/2512.12954)</td><td>◆ We establish linear convergence of relocated fixed-point iterations as introduced by Atenas et al.
◆ (2025) assuming the algorithmic operator satisfies a linear error bound.
◆ In particular, this framework applies to the setting where the algorithmic operator is a contraction.</td></tr>
<tr><td>2025-12-11</td><td>Quantifying displacement: a gentrification&#x27;s consequence via persistent homology</td><td>[2512.10753](http://arxiv.org/pdf/2512.10753)</td><td>◆ Gentrification is the process by which wealthier individuals move into a previously lower-income neighbourhood.
◆ Among the effects of this multi-faceted phenomenon are rising living costs, cultural and social changes-where local traditions, businesses, and community networks are replaced or diluted by new, more affluent lifestyles-and population displacement, where long-term, lower-income residents are priced out by rising rents and property taxes.
◆ Despite its relevance, quantifying displacement presents difficulties stemming from lack of information on motives for relocation and from the fact that a long time-span must be analysed: displacement is a gradual process (leases end or conditions change at different times), impossible to capture in one data snapshot.</td></tr>
<tr><td>2025-12-11</td><td>HypeR Adaptivity: Joint $hr$-Adaptive Meshing via Hypergraph Multi-Agent Deep Reinforcement Learning</td><td>[2512.10439](http://arxiv.org/pdf/2512.10439)</td><td>◆ Adaptive mesh refinement is central to the efficient solution of partial differential equations (PDEs) via the finite element method (FEM).
◆ Classical $r$-adaptivity optimizes vertex positions but requires solving expensive auxiliary PDEs such as the Monge-Ampère equation, while classical $h$-adaptivity modifies topology through element subdivision but suffers from expensive error indicator computation and is constrained by isotropic refinement patterns that impose accuracy ceilings.
◆ Combined $hr$-adaptive techniques naturally outperform single-modality approaches, yet inherit both computational bottlenecks and the restricted cost-accuracy trade-off.</td></tr>
<tr><td>2025-12-10</td><td>YOPO-Nav: Visual Navigation using 3DGS Graphs from One-Pass Videos</td><td>[2512.09903](http://arxiv.org/pdf/2512.09903)</td><td>◆ Visual navigation has emerged as a practical alternative to traditional robotic navigation pipelines that rely on detailed mapping and path planning.
◆ However, constructing and maintaining 3D maps is often computationally expensive and memory-intensive.
◆ We address the problem of visual navigation when exploration videos of a large environment are available.</td></tr>
<tr><td>2025-12-10</td><td>Sequential Testing for Descriptor-Agnostic LiDAR Loop Closure in Repetitive Environments</td><td>[2512.09447](http://arxiv.org/pdf/2512.09447)</td><td>◆ We propose a descriptor-agnostic, multi-frame loop closure verification method that formulates LiDAR loop closure as a truncated Sequential Probability Ratio Test (SPRT).
◆ Instead of deciding from a single descriptor comparison or using fixed thresholds with late-stage Iterative Closest Point (ICP) vetting, the verifier accumulates a short temporal stream of descriptor similarities between a query and each candidate.
◆ It then issues an accept/reject decision adaptively once sufficient multi-frame evidence has been observed, according to user-specified Type-I/II error design targets.</td></tr>
<tr><td>2025-12-09</td><td>Adaptive Thresholding for Visual Place Recognition using Negative Gaussian Mixture Statistics</td><td>[2512.09071](http://arxiv.org/pdf/2512.09071)</td><td>◆ Visual place recognition (VPR) is an important component technology for camera-based mapping and navigation applications.
◆ This is a challenging problem because images of the same place may appear quite different for reasons including seasonal changes, weather illumination, structural changes to the environment, as well as transient pedestrian or vehicle traffic.
◆ Papers focusing on generating image descriptors for VPR report their results using metrics such as recall@K and ROC curves.</td></tr>
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
<tr><td>2026-03-13</td><td>Semantic Aware Feature Extraction for Enhanced 3D Reconstruction</td><td>[2603.13556](http://arxiv.org/pdf/2603.13556)</td><td>◆ Feature matching is a fundamental problem in computer vision with wide-ranging applications, including simultaneous localization and mapping (SLAM), image stitching, and 3D reconstruction.
◆ While recent advances in deep learning have improved keypoint detection and description, most approaches focus primarily on geometric attributes and often neglect higher-level semantic information.
◆ This work proposes a semantic-aware feature extraction framework that employs multi-task learning to jointly train keypoint detection, keypoint description, and semantic segmentation.</td></tr>
<tr><td>2026-03-13</td><td>CM-Bench: A Comprehensive Cross-Modal Feature Matching Benchmark Bridging Visible and Infrared Images</td><td>[2603.12690](http://arxiv.org/pdf/2603.12690)</td><td>◆ Infrared-visible (IR-VIS) feature matching plays an essential role in cross-modality visual localization, navigation and perception.
◆ Along with the rapid development of deep learning techniques, a number of representative image matching methods have been proposed.
◆ However, crossmodal feature matching is still a challenging task due to the significant appearance difference.</td></tr>
<tr><td>2026-03-09</td><td>Enhancing Cross-View UAV Geolocalization via LVLM-Driven Relational Modeling</td><td>[2603.08063](http://arxiv.org/pdf/2603.08063)</td><td>该论文的核心贡献是提出了一种新颖的、即插即用的排序架构，以解决跨视角无人机地理定位中不同视图间交互关系建模不足的问题，从而显著提升了无人机图像与卫星图像的匹配精度。

◆ 提出了一种新颖的即插即用排序架构，能够显式地对无人机和卫星视图进行联合关系建模，突破了现有方法独立提取特征和依赖简单启发式相似度计算的局限。
◆ 创新性地利用大型视觉-语言模型的能力，通过学习深层的视觉-语义关联，来有效捕捉连接无人机图像与卫星图像之间的复杂相关性。
◆ 设计了一种新颖的关系感知损失函数，该函数通过使用软标签提供细粒度监督，避免对近似正样本匹配进行过度惩罚，从而同时增强了模型的判别力和训练稳定性。
◆ 所提方法作为一个通用增强模块，在多种基线架构和标准基准测试上进行了全面评估，证明其能大幅提升现有模型的检索精度，即使在极具挑战性的条件下也能实现优越性能。</td></tr>
<tr><td>2026-03-09</td><td>Speed3R: Sparse Feed-forward 3D Reconstruction Models</td><td>[2603.08055](http://arxiv.org/pdf/2603.08055)</td><td>该论文针对现有前馈式三维重建模型因密集注意力机制导致计算量过大、推理速度慢的问题，提出了Speed3R模型，其核心贡献与创新点如下：

◆ 核心思想借鉴运动恢复结构，提出仅需稀疏关键点即可进行稳健姿态估计，从而大幅降低计算复杂度。
◆ 设计了双分支注意力机制，其中压缩分支生成粗略上下文先验，以指导选择分支进行细粒度处理。
◆ 选择分支仅对信息量最大的图像令牌执行注意力计算，模仿了传统关键点匹配的高效性。
◆ 该模型在千视图序列上实现了12.4倍的推理加速，同时仅以极小的、可控的几何精度损失作为代价。
◆ 在多个标准基准测试中，使用不同骨干网络均验证了其有效性，能以极低计算成本获得高质量重建，为高效大规模场景建模开辟了新途径。</td></tr>
<tr><td>2026-03-06</td><td>EventGeM: Global-to-Local Feature Matching for Event-Based Visual Place Recognition</td><td>[2603.05807](http://arxiv.org/pdf/2603.05807)</td><td>该论文提出了EventGeM，一种用于基于事件的视觉位置识别（VPR）的先进方法。其核心贡献在于构建了一个高效且鲁棒的全局到局部特征匹配流程，能够在多种挑战性条件下实现实时、高精度的定位。

◆ 创新性地提出了一个全局到局部的特征融合流水线。首先利用预训练的视觉变换器（ViT）从事件直方图图像中提取全局特征进行初始匹配，再使用预训练的MaxViT检测局部关键点进行精细重排序。
◆ 引入基于2D单应性变换和RANSAC的局部几何验证，对初始全局匹配结果进行重排序，提升了匹配的几何一致性。
◆ 进一步利用预训练的视觉基础模型进行深度估计，通过比较查询图像与参考图像之间的结构相似性，实现了额外的重排序优化，增强了算法在视角和外观变化下的鲁棒性。
◆ 该方法在多个基准数据集和不同光照条件下，性能超越了当前最佳的基于事件的位置识别方法，同时证明了其在不同计算架构上实时运行的可行性。
◆ 通过在实际机器人平台上使用直接从事件相机获取的事件流进行在线定位部署，验证了该系统的实用性和有效性。</td></tr>
<tr><td>2026-03-05</td><td>From Decoupled to Coupled: Robustness Verification for Learning-based Keypoint Detection with Joint Specifications</td><td>[2603.05604](http://arxiv.org/pdf/2603.05604)</td><td>该论文首次为基于热图的关键点检测器提出了耦合的鲁棒性验证框架，解决了该领域此前缺乏形式化验证方法的难题。

◆ 核心创新在于从“解耦”转向“耦合”的验证范式，首次对多个关键点的联合偏差进行有界验证，而非独立验证单个点，从而捕捉了关键点间的相互依赖关系。

◆ 该方法将验证问题构建为一个混合整数线性规划问题，通过结合热图的可达集与编码联合约束的多面体，来严格验证模型在扰动下的集体行为。

◆ 该框架被证明是可靠的：若方法认证模型鲁棒，则模型在给定联合规范下保证鲁棒；若不可行，则能提供反例。

◆ 实验表明，在严格的误差阈值下，该耦合方法能实现较高的可验证鲁棒率，而传统的解耦方法在此情况下会失效，凸显了其对于下游任务要求（如姿态估计）的实际价值。</td></tr>
<tr><td>2026-03-04</td><td>Yolo-Key-6D: Single Stage Monocular 6D Pose Estimation with Keypoint Enhancements</td><td>[2603.03879](http://arxiv.org/pdf/2603.03879)</td><td>本文提出Yolo-Key-6D，一种用于单目6D姿态估计的新型单阶段端到端框架，旨在兼顾速度与精度。其核心贡献与创新点如下：

◆ 提出新颖的单阶段端到端框架，直接基于YOLO架构进行增强，解决了现有先进多阶段方法延迟高、难以实时应用的问题。
◆ 通过集成一个辅助预测头来回归物体3D边界框角点的2D投影，这一关键点检测任务显著提升了网络对3D几何结构的理解能力。
◆ 采用连续的9D旋转表示，并通过奇异值分解投影到SO(3)流形上，实现了稳定的端到端旋转回归训练。
◆ 在LINEMOD和LINEMOD-Occluded基准测试中取得了有竞争力的精度（ADD(-S) 0.1d指标下分别为96.24%和69.41%），同时保持了实时运行速度。
◆ 整体工作表明，经过精心设计的单阶段方法能够在现实世界部署中，提供性能与效率之间的实用且有效的平衡。</td></tr>
<tr><td>2026-02-27</td><td>No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency</td><td>[2602.23559](http://arxiv.org/pdf/2602.23559)</td><td>该论文首次系统研究了跨不同传感器的多模态视图合成问题，针对RGB-X数据对齐需繁琐校准这一被忽视的实践难题提出了创新解决方案。

◆ 首次系统探索跨传感器多模态视图合成，聚焦于解决实际中RGB-X数据配对校准困难这一基础瓶颈。
◆ 提出“匹配-稠密化-巩固”方法，通过RGB-X图像匹配、引导式点云稠密化及自匹配滤波，无需X传感器的3D先验信息。
◆ 引入置信感知的稠密化与自匹配过滤技术，有效提升跨模态视图合成质量。
◆ 将合成结果整合至3D高斯溅射框架中，确保三维一致性，仅需对RGB侧使用近乎零成本的COLMAP工具。
◆ 旨在免除多种RGB-X传感器间的复杂校准流程，为大规模真实世界跨传感器数据采集与学习提供可扩展方案。</td></tr>
<tr><td>2026-02-25</td><td>UNet-Based Keypoint Regression for 3D Cone Localization in Autonomous Racing</td><td>[2602.21904](http://arxiv.org/pdf/2602.21904)</td><td>该论文的核心贡献是提出了一种基于UNet神经网络的新方法，用于在自动驾驶赛车场景中实现高精度的三维锥桶定位。

◆ 提出了一种基于UNet架构的关键点回归网络，用于检测锥桶上的关键点，从而估计其精确的三维位置。
◆ 构建并利用了目前该领域最大规模的自定义标注数据集进行模型训练，增强了模型的鲁棒性。
◆ 该方法在关键点检测精度上相比传统计算机视觉方法有显著提升，并且克服了传统方法对环境变化敏感、以及以往神经网络数据有限或难以实时运行的问题。
◆ 所提出的模型不仅实现了位置估计，还具备预测锥桶颜色的潜力，丰富了感知信息。
◆ 将预测的关键点集成到完整的感知系统中，并进行了端到端的自动驾驶系统评估，在实际赛道场景中展现了全面的高性能，证明了其在竞技性自动驾驶系统中的实用潜力。</td></tr>
<tr><td>2026-02-26</td><td>FlowFixer: Towards Detail-Preserving Subject-Driven Generation</td><td>[2602.21402](http://arxiv.org/pdf/2602.21402)</td><td>FlowFixer是一个用于主体驱动生成的精细化框架，其核心贡献在于有效恢复生成过程中因主体尺度与视角变化而丢失的细节。其创新点主要包括：

◆ 提出了一种直接的图像到图像转换方法，利用视觉参考进行细节修复，避免了语言提示的模糊性。

◆ 引入了一步去噪方案来生成自监督训练数据，该方案能自动去除高频细节同时保留全局结构，从而有效模拟真实的主体驱动生成错误。

◆ 提出了一种基于关键点匹配的评估指标，该指标能够超越CLIP或DINO通常测量的语义相似性，更准确地评估细节层面的保真度。

实验结果表明，FlowFixer在定性和定量评估上均优于现有先进方法，为高保真度的主体驱动生成设立了新基准。</td></tr>
<tr><td>2026-03-03</td><td>From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection</td><td>[2602.20630](http://arxiv.org/pdf/2602.20630)</td><td>该论文的核心贡献是将关键点检测重新构建为序列决策问题，并提出了一种基于强化学习的端到端框架TraqPoint，以直接在图像序列上优化关键点的长期跟踪质量。

◆ 创新性地将关键点检测从传统的图像对训练范式转变为序列决策问题，强调跨多视角的长期可跟踪性。
◆ 提出了TraqPoint这一端到端强化学习框架，通过策略梯度方法直接优化关键点的轨迹质量（Traq）。
◆ 设计了轨迹感知的奖励机制，联合促进关键点在多视角下的一致性和独特性，从而提升在挑战性视角与光照变化下的稳定性。
◆ 在相对姿态估计和三维重建等稀疏匹配任务上验证了其有效性，性能显著优于多种先进的关键点检测与描述方法。</td></tr>
<tr><td>2026-02-23</td><td>Generative 6D Pose Estimation via Conditional Flow Matching</td><td>[2602.19719](http://arxiv.org/pdf/2602.19719)</td><td>该论文提出了一种名为Flose的生成式6D位姿估计新方法，其核心贡献与创新点如下：

◆ 将6D位姿估计问题重新定义为三维空间中的条件流匹配问题，提出了一种生成式推理框架，通过去噪过程预测物体姿态。

◆ 创新性地融合了基于外观的语义特征作为条件，而不仅仅依赖几何引导。这有效缓解了因物体对称性导致的姿态歧义问题，克服了传统回归方法在此方面的不足。

◆ 方法整合了基于局部特征的匹配思想，使其在物体缺乏显著局部纹理时仍能保持鲁棒性，解决了纯特征匹配方法在此类情况下的失效问题。

◆ 引入了基于RANSAC的配准步骤来处理异常值，进一步提升了系统的鲁棒性和精度。

◆ 在权威的BOP基准测试的五个数据集上进行了验证，结果表明Flose平均召回率平均提升了4.5个百分点，显著优于现有方法。</td></tr>
<tr><td>2026-02-20</td><td>Morphological Addressing of Identity Basins in Text-to-Image Diffusion Models</td><td>[2602.18533](http://arxiv.org/pdf/2602.18533)</td><td>该论文的核心贡献在于揭示了形态学结构能够为文本到图像扩散模型的潜在空间提供可导航的梯度，从而实现对特定视觉身份的定位与塑造。

◆ 提出了一种无需目标姓名或真实照片、仅使用形态特征描述符（如“铂金色头发”）来导航并收敛到特定人物身份的方法，通过自蒸馏循环训练LoRA实现。
◆ 发现训练所得的LoRA不仅塑造目标身份，还能定义其“逆身份”，在极端负向引导下产生“恐怖谷”式的连贯但错误的输出，揭示了身份空间的局部坐标系。
◆ 将形态学分析扩展到提示词层面，首次将语音象征理论应用于扩散模型，证明无意义的语音象征词素能生成高度一致的视觉概念。
◆ 通过实验发现如“snudgeoid”等无意义词汇能实现完美的视觉一致性，这证明了亚词汇的语音模式本身可承载并生成新颖、连贯的视觉身份。
◆ 整体上，论文从特征描述和语音形式两个层面，系统论证了形态结构如何创造可导航梯度，并记录了身份盆地中的相变、CFG不变的稳定性等新现象。</td></tr>
<tr><td>2026-02-17</td><td>SAM 3D Body: Robust Full-Body Human Mesh Recovery</td><td>[2602.15989](http://arxiv.org/pdf/2602.15989)</td><td>该论文的核心贡献是提出了一个名为SAM 3D Body（3DB）的、可提示的单图像全身三维人体网格恢复模型，其在复杂真实场景下实现了最先进的性能。

◆ 提出了首个可提示的全身3D人体网格恢复模型3DB，支持用户通过2D关键点或掩码等辅助提示进行引导推理，增强了可控性。
◆ 引入了一种全新的参数化网格表示方法Momentum Human Rig（MHR），其创新性地将骨骼结构与表面形状解耦，提供了更灵活的建模方式。
◆ 设计了一个高效的数据引擎与多阶段标注流程，通过结合人工标注、可微分优化等技术生成高质量数据，并主动收集罕见姿态和成像条件的数据以确保多样性。
◆ 建立了一个按姿态和外观类别组织的新评估数据集，使得对模型行为的细致分析成为可能，超越了传统单一指标的评价方式。</td></tr>
<tr><td>2026-02-15</td><td>Differential pose optimization in descriptor space -- Combining Geometric and Photometric Methods for Motion Estimation</td><td>[2602.14297](http://arxiv.org/pdf/2602.14297)</td><td>该论文的核心贡献是提出并深入分析了一种结合几何与光度方法优点的相对位姿估计新框架。

◆ 创新性地提出了一种统一的运动估计方法，将光度法和几何法的优势相结合。
◆ 具体方案是使用密集采样的几何特征描述子，构建了一种新的“描述子残差”来替代传统的光度误差。
◆ 该方法使得在微分光度法中可以同时利用描述子的强表达能力和亚像素级精度。
◆ 通过实验验证了新方法的可行性，能够实现精确的跟踪，但其性能最终并未超越基于重投影误差的优化方法。
◆ 论文进一步分析了性能未达预期的原因，提出了关键假设：描述子相似性度量变化过于平缓，且与关键点定位精度并非严格对应，这揭示了该融合路径的内在局限性。</td></tr>
<tr><td>2026-02-13</td><td>Matching of SAR and optical images based on transformation to shared modality</td><td>[2602.12515](http://arxiv.org/pdf/2602.12515)</td><td>该论文的核心贡献是提出了一种新颖的光学与合成孔径雷达图像匹配方法，通过将两种异源图像转换到一个共享的模态来解决其因成像原理不同而难以配准的问题。

◆ 创新性地提出了将光学与SAR图像转换至一个共享中间模态的思路，而非直接在原始模态间进行转换或匹配。
◆ 为该共享模态定义了明确约束：需具有相同预设通道数、转换后已配准的图像应尽可能相似，且必须保留原始图像的重要特征，避免退化。
◆ 成功利用为普通照片设计的先进匹配模型（如RoMa），直接对转换至共享模态的图像进行匹配，无需针对新模态重新训练模型。
◆ 在公开数据集上的实验表明，该方法在匹配质量上优于基于原始模态间图像翻译和各种特征匹配的传统方法，且通用性更强。</td></tr>
<tr><td>2026-02-09</td><td>Understanding and Optimizing Attention-Based Sparse Matching for Diverse Local Features</td><td>[2602.08430](http://arxiv.org/pdf/2602.08430)</td><td>该论文的核心贡献在于对基于注意力的稀疏图像匹配模型进行了深入分析与优化，并提出了一个通用的匹配模型。其创新点可总结如下：

◆ 首次指出了一个先前被忽视、但对LightGlue模型性能有关键影响的设计选择，为模型理解提供了新视角。

◆ 系统研究了在基于Transformer的匹配框架中，检测器与描述子的作用，发现性能差异的主要来源通常是检测器而非描述子。

◆ 提出了一种新颖的微调方法，能够利用来自多种不同检测器的关键点数据来训练现有图像匹配模型。

◆ 最终训练出一个通用的、与检测器无关的模型，该模型在作为新检测器的零样本匹配器时，其精度能达到甚至超过为特定特征专门训练的模型。

这些发现为基于Transformer的匹配模型的部署以及未来局部特征的设计提供了宝贵的见解。</td></tr>
<tr><td>2026-01-31</td><td>Gaussian-Constrained LeJEPA Representations for Unsupervised Scene Discovery and Pose Consistency</td><td>[2602.07016](http://arxiv.org/pdf/2602.07016)</td><td>本文的核心贡献在于探索并实证了高斯约束表征在无监督三维场景重建中的应用，特别是在多场景发现与相机姿态估计任务中。其创新点可总结如下：

◆ 首次将受LeJEPA启发的各向同性高斯约束应用于无监督场景发现的图像嵌入学习，旨在提升表征的区分度。
◆ 提出了三种逐步优化的处理流程，最终形成一种约束嵌入以服从高斯分布的方法，增强聚类一致性。
◆ 不追求理论证明，而是通过实证评估，验证了高斯约束在实际任务中对场景分离和姿态估计鲁棒性的积极影响。
◆ 在IMC2025挑战赛的复杂真实数据（包含异常值和视觉模糊内容）上验证了方法的有效性，表明其优于启发式基线方法。
◆ 为连接自监督学习原理与实际运动恢复结构流程提供了一个有前景的研究方向，即通过理论驱动的表征约束来提升实际系统性能。</td></tr>
<tr><td>2026-02-25</td><td>Perception-Control Coupled Visual Servoing for Textureless Objects Using Keypoint-Based EKF</td><td>[2602.06834](http://arxiv.org/pdf/2602.06834)</td><td>◆ Visual servoing is fundamental to robotic applications, enabling precise positioning and control.
◆ However, applying it to textureless objects remains a challenge due to the absence of reliable visual features.
◆ Moreover, adverse visual conditions, such as occlusions, often corrupt visual feedback, leading to reduced accuracy and instability in visual servoing.</td></tr>
<tr><td>2026-02-05</td><td>DroneKey++: A Size Prior-free Method and New Benchmark for Drone 3D Pose Estimation from Sequential Images</td><td>[2602.06211](http://arxiv.org/pdf/2602.06211)</td><td>◆ Accurate 3D pose estimation of drones is essential for security and surveillance systems.
◆ However, existing methods often rely on prior drone information such as physical sizes or 3D meshes.
◆ At the same time, current datasets are small-scale, limited to single models, and collected under constrained environments, which makes reliable validation of generalization difficult.</td></tr>
<tr><td>2026-02-05</td><td>SOMA-1M: A Large-Scale SAR-Optical Multi-resolution Alignment Dataset for Multi-Task Remote Sensing</td><td>[2602.05480](http://arxiv.org/pdf/2602.05480)</td><td>◆ Synthetic Aperture Radar (SAR) and optical imagery provide complementary strengths that constitute the critical foundation for transcending single-modality constraints and facilitating cross-modal collaborative processing and intelligent interpretation.
◆ However, existing benchmark datasets often suffer from limitations such as single spatial resolution, insufficient data scale, and low alignment accuracy, making them inadequate for supporting the training and generalization of multi-scale foundation models.
◆ To address these challenges, we introduce SOMA-1M (SAR-Optical Multi-resolution Alignment), a pixel-level precisely aligned dataset containing over 1.3 million pairs of georeferenced images with a specification of 512 x 512 pixels.</td></tr>
<tr><td>2026-02-04</td><td>Quantile Transfer for Reliable Operating Point Selection in Visual Place Recognition</td><td>[2602.04401](http://arxiv.org/pdf/2602.04401)</td><td>◆ Visual Place Recognition (VPR) is a key component for localisation in GNSS-denied environments, but its performance critically depends on selecting an image matching threshold (operating point) that balances precision and recall.
◆ Thresholds are typically hand-tuned offline for a specific environment and fixed during deployment, leading to degraded performance under environmental change.
◆ We propose a method that, given a user-defined precision requirement, automatically selects the operating point of a VPR system to maximise recall.</td></tr>
<tr><td>2026-01-22</td><td>Coarse-to-Fine Non-rigid Multi-modal Image Registration for Historical Panel Paintings based on Crack Structures</td><td>[2601.16348](http://arxiv.org/pdf/2601.16348)</td><td>◆ Art technological investigations of historical panel paintings rely on acquiring multi-modal image data, including visual light photography, infrared reflectography, ultraviolet fluorescence photography, x-radiography, and macro photography.
◆ For a comprehensive analysis, the multi-modal images require pixel-wise alignment, which is still often performed manually.
◆ Multi-modal image registration can reduce this laborious manual work, is substantially faster, and enables higher precision.</td></tr>
<tr><td>2026-01-21</td><td>ZENITH: Automated Gradient Norm Informed Stochastic Optimization</td><td>[2601.15212](http://arxiv.org/pdf/2601.15212)</td><td>◆ Training deep computer vision models requires manual oversight or hyperparameter tuning of the learning rate (LR) schedule.
◆ While existing adaptive optimizers schedule the LR automatically, they suffer from computational and memory overhead, incompatibility with regularization, and suboptimal LR choices.
◆ In this work, we introduce the ZENITH (Zero-overhead Evolution using Norm-Informed Training History) optimizer, which adapts the LR using the temporal evolution of the gradient norm.</td></tr>
<tr><td>2026-01-19</td><td>A Streamlined Attention-Based Network for Descriptor Extraction</td><td>[2601.13126](http://arxiv.org/pdf/2601.13126)</td><td>◆ We introduce SANDesc, a Streamlined Attention-Based Network for Descriptor extraction that aims to improve on existing architectures for keypoint description.
◆ Our descriptor network learns to compute descriptors that improve matching without modifying the underlying keypoint detector.
◆ We employ a revised U-Net-like architecture enhanced with Convolutional Block Attention Modules and residual paths, enabling effective local representation while maintaining computational efficiency.</td></tr>
<tr><td>2026-01-18</td><td>XRefine: Attention-Guided Keypoint Match Refinement</td><td>[2601.12530](http://arxiv.org/pdf/2601.12530)</td><td>◆ Sparse keypoint matching is crucial for 3D vision tasks, yet current keypoint detectors often produce spatially inaccurate matches.
◆ Existing refinement methods mitigate this issue through alignment of matched keypoint locations, but they are typically detector-specific, requiring retraining for each keypoint detector.
◆ We introduce XRefine, a novel, detector-agnostic approach for sub-pixel keypoint refinement that operates solely on image patches centered at matched keypoints.</td></tr>
<tr><td>2026-01-17</td><td>SupScene: Learning Overlap-Aware Global Descriptor for Unconstrained SfM</td><td>[2601.11930](http://arxiv.org/pdf/2601.11930)</td><td>◆ Image retrieval is a critical step for alleviating the quadratic complexity of image matching in unconstrained Structure-from-Motion (SfM).
◆ However, in this context, image retrieval typically focuses more on the image pairs of geometric matchability than on those of semantic similarity, a nuance that most existing deep learning-based methods guided by batched binaries (overlapping vs.
◆ non-overlapping pairs) fail to capture.</td></tr>
<tr><td>2026-01-14</td><td>CLIDD: Cross-Layer Independent Deformable Description for Efficient and Discriminative Local Feature Representation</td><td>[2601.09230](http://arxiv.org/pdf/2601.09230)</td><td>◆ Robust local feature representations are essential for spatial intelligence tasks such as robot navigation and augmented reality.
◆ Establishing reliable correspondences requires descriptors that provide both high discriminative power and computational efficiency.
◆ To address this, we introduce Cross-Layer Independent Deformable Description (CLIDD), a method that achieves superior distinctiveness by sampling directly from independent feature hierarchies.</td></tr>
<tr><td>2026-01-13</td><td>Near-perfect photo-ID of the Hula painted frog with zero-shot deep local-feature matching</td><td>[2601.08798](http://arxiv.org/pdf/2601.08798)</td><td>◆ Accurate individual identification is essential for monitoring rare amphibians, yet invasive marking is often unsuitable for critically endangered species.
◆ We evaluate state-of-the-art computer-vision methods for photographic re-identification of the Hula painted frog (Latonia nigriventer) using 1,233 ventral images from 191 individuals collected during 2013-2020 capture-recapture surveys.
◆ We compare deep local-feature matching in a zero-shot setting with deep global-feature embedding models.</td></tr>
<tr><td>2026-01-13</td><td>Second-order Gaussian directional derivative representations for image high-resolution corner detection</td><td>[2601.08182](http://arxiv.org/pdf/2601.08182)</td><td>◆ Corner detection is widely used in various computer vision tasks, such as image matching and 3D reconstruction.
◆ Our research indicates that there are theoretical flaws in Zhang et al.&#x27;s use of a simple corner model to obtain a series of corner characteristics, as the grayscale information of two adjacent corners can affect each other.
◆ In order to address the above issues, a second-order Gaussian directional derivative (SOGDD) filter is used in this work to smooth two typical high-resolution angle models (i.e.</td></tr>
<tr><td>2026-01-09</td><td>Stationaere Kurven auf endlichdimensionalen Mannigfaltigkeiten</td><td>[2601.05695](http://arxiv.org/pdf/2601.05695)</td><td>◆ In this work we discuss the notion of stationary curves of the length functional, the so-called (weak) geodesics, on a Riemannian manifold.
◆ The motivation behind this work is to give a detailed description of many key concepts from differential geometry that one needs in order to understand the important notion of a (weak) geodesic.
◆ For this, we mainly focus on finite-dimensional smooth manifolds, so that we can develop an intuitive and geometric understanding of the concepts that we want to discuss.</td></tr>
<tr><td>2026-01-05</td><td>Exact Clique Number Manipulation via Edge Interdiction</td><td>[2601.01869](http://arxiv.org/pdf/2601.01869)</td><td>◆ The Edge Interdiction Clique Problem (EICP) aims to remove at most $k$ edges from a graph so as to minimize the size of the largest clique in the remaining graph.
◆ This problem captures a fundamental question in graph manipulation: which edges are structurally critical for preserving large cliques?
◆ Such a problem is also motivated by practical applications including protein function maintenance and image matching.</td></tr>
<tr><td>2026-01-02</td><td>UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data</td><td>[2601.00991](http://arxiv.org/pdf/2601.00991)</td><td>◆ Diverse, accurately labeled 3D human pose data is expensive and studio-bound, while in-the-wild datasets lack known ground truth.
◆ We introduce UnrealPose-Gen, an Unreal Engine 5 pipeline built on Movie Render Queue for high-quality offline rendering.
◆ Our generated frames include: (i) 3D joints in world and camera coordinates, (ii) 2D projections and COCO-style keypoints with occlusion and joint-visibility flags, (iii) person bounding boxes, and (iv) camera intrinsics and extrinsics.</td></tr>
<tr><td>2025-12-31</td><td>GenZ: Foundational models as latent variable generators within traditional statistical models</td><td>[2512.24834](http://arxiv.org/pdf/2512.24834)</td><td>◆ We present GenZ, a hybrid model that bridges foundational models and statistical modeling through interpretable semantic features.
◆ While large language models possess broad domain knowledge, they often fail to capture dataset-specific patterns critical for prediction tasks.
◆ Our approach addresses this by discovering semantic feature descriptions through an iterative process that contrasts groups of items identified via statistical modeling errors, rather than relying solely on the foundational model&#x27;s domain understanding.</td></tr>
<tr><td>2025-12-31</td><td>Quantum Visual Word Sense Disambiguation: Unraveling Ambiguities Through Quantum Inference Model</td><td>[2512.24687](http://arxiv.org/pdf/2512.24687)</td><td>◆ Visual word sense disambiguation focuses on polysemous words, where candidate images can be easily confused.
◆ Traditional methods use classical probability to calculate the likelihood of an image matching each gloss of the target word, summing these to form a posterior probability.
◆ However, due to the challenge of semantic uncertainty, glosses from different sources inevitably carry semantic biases, which can lead to biased disambiguation results.</td></tr>
<tr><td>2025-12-24</td><td>VisRes Bench: On Evaluating the Visual Reasoning Capabilities of VLMs</td><td>[2512.21194](http://arxiv.org/pdf/2512.21194)</td><td>◆ Vision-Language Models (VLMs) have achieved remarkable progress across tasks such as visual question answering and image captioning.
◆ Yet, the extent to which these models perform visual reasoning as opposed to relying on linguistic priors remains unclear.
◆ To address this, we introduce VisRes Bench, a benchmark designed to study visual reasoning in naturalistic settings without contextual language supervision.</td></tr>
<tr><td>2025-12-20</td><td>Analog Quantum Image Representation with Qubit-Frugal Encoding</td><td>[2512.18451](http://arxiv.org/pdf/2512.18451)</td><td>◆ In this work, we introduce a fundamentally new paradigm for quantum image representation tailored for neutral-atom quantum devices.
◆ The proposed method constructs a qubit-efficient image representation by first applying a cartographic generalization algorithm to a classical edge-extracted input image, yielding a highly optimized sparse-dot based geometric description.
◆ While ensuring the structural integrity of the image, this sparse representation is then embedded into the atomic configuration of Aquila (QuEra Computing Inc.), modeled through the Bloqade simulation software stack.</td></tr>
<tr><td>2025-12-17</td><td>The Perceptual Observatory Characterizing Robustness and Grounding in MLLMs</td><td>[2512.15949](http://arxiv.org/pdf/2512.15949)</td><td>◆ Recent advances in multimodal large language models (MLLMs) have yielded increasingly powerful models, yet their perceptual capacities remain poorly characterized.
◆ In practice, most model families scale language component while reusing nearly identical vision encoders (e.g., Qwen2.5-VL 3B/7B/72B), which raises pivotal concerns about whether progress reflects genuine visual grounding or reliance on internet-scale textual world knowledge.
◆ Existing evaluation methods emphasize end-task accuracy, overlooking robustness, attribution fidelity, and reasoning under controlled perturbations.</td></tr>
<tr><td>2026-01-11</td><td>BLANKET: Anonymizing Faces in Infant Video Recordings</td><td>[2512.15542](http://arxiv.org/pdf/2512.15542)</td><td>◆ Ensuring the ethical use of video data involving human subjects, particularly infants, requires robust anonymization methods.
◆ We propose BLANKET (Baby-face Landmark-preserving ANonymization with Keypoint dEtection consisTency), a novel approach designed to anonymize infant faces in video recordings while preserving essential facial attributes.
◆ Our method comprises two stages.</td></tr>
<tr><td>2025-12-17</td><td>Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting</td><td>[2512.15508](http://arxiv.org/pdf/2512.15508)</td><td>◆ Feed-forward 3D Gaussian Splatting (3DGS) models enable real-time scene generation but are hindered by suboptimal pixel-aligned primitive placement, which relies on a dense, rigid grid and limits both quality and efficiency.
◆ We introduce a new feed-forward architecture that detects 3D Gaussian primitives at a sub-pixel level, replacing the pixel grid with an adaptive, &quot;Off The Grid&quot; distribution.
◆ Inspired by keypoint detection, our multi-resolution decoder learns to distribute primitives across image patches.</td></tr>
<tr><td>2025-12-15</td><td>JoVA: Unified Multimodal Learning for Joint Video-Audio Generation</td><td>[2512.13677](http://arxiv.org/pdf/2512.13677)</td><td>◆ In this paper, we present JoVA, a unified framework for joint video-audio generation.
◆ Despite recent encouraging advances, existing methods face two critical limitations.
◆ First, most existing approaches can only generate ambient sounds and lack the capability to produce human speech synchronized with lip movements.</td></tr>
<tr><td>2025-12-11</td><td>Self-Supervised Contrastive Embedding Adaptation for Endoscopic Image Matching</td><td>[2512.10379](http://arxiv.org/pdf/2512.10379)</td><td>◆ Accurate spatial understanding is essential for image-guided surgery, augmented reality integration and context awareness.
◆ In minimally invasive procedures, where visual input is the sole intraoperative modality, establishing precise pixel-level correspondences between endoscopic frames is critical for 3D reconstruction, camera tracking, and scene interpretation.
◆ However, the surgical domain presents distinct challenges: weak perspective cues, non-Lambertian tissue reflections, and complex, deformable anatomy degrade the performance of conventional computer vision techniques.</td></tr>
<tr><td>2025-12-14</td><td>MotionEdit: Benchmarking and Learning Motion-Centric Image Editing</td><td>[2512.10284](http://arxiv.org/pdf/2512.10284)</td><td>◆ We introduce MotionEdit, a novel dataset for motion-centric image editing-the task of modifying subject actions and interactions while preserving identity, structure, and physical plausibility.
◆ Unlike existing image editing datasets that focus on static appearance changes or contain only sparse, low-quality motion edits, MotionEdit provides high-fidelity image pairs depicting realistic motion transformations extracted and verified from continuous videos.
◆ This new task is not only scientifically challenging but also practically significant, powering downstream applications such as frame-controlled video synthesis and animation.</td></tr>
<tr><td>2025-12-16</td><td>UnCageNet: Tracking and Pose Estimation of Caged Animal</td><td>[2512.07712](http://arxiv.org/pdf/2512.07712)</td><td>◆ Animal tracking and pose estimation systems, such as STEP (Simultaneous Tracking and Pose Estimation) and ViTPose, experience substantial performance drops when processing images and videos with cage structures and systematic occlusions.
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
<tr><td>2026-03-17</td><td>M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM</td><td>[2603.16844](http://arxiv.org/pdf/2603.16844)</td><td>◆ Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments.
◆ While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization.
◆ To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM.</td></tr>
<tr><td>2026-03-17</td><td>Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty</td><td>[2603.16538](http://arxiv.org/pdf/2603.16538)</td><td>◆ 3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement.
◆ However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry.
◆ In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers.</td></tr>
<tr><td>2026-03-17</td><td>Leveling3D: Leveling Up 3D Reconstruction with Feed-Forward 3D Gaussian Splatting and Geometry-Aware Generation</td><td>[2603.16211](http://arxiv.org/pdf/2603.16211)</td><td>◆ Feed-forward 3D reconstruction has revolutionized 3D vision, providing a powerful baseline for downstream tasks such as novel-view synthesis with 3D Gaussian Splatting.
◆ Previous works explore fixing the corrupted rendering results with a diffusion model.
◆ However, they lack geometric concern and fail at filling the missing area on the extrapolated view.</td></tr>
<tr><td>2026-03-17</td><td>NanoGS: Training-Free Gaussian Splat Simplification</td><td>[2603.16103](http://arxiv.org/pdf/2603.16103)</td><td>◆ 3D Gaussian Splat (3DGS) enables high-fidelity, real-time novel view synthesis by representing scenes with large sets of anisotropic primitives, but often requires millions of Splats, incurring significant storage and transmission costs.
◆ Most existing compression methods rely on GPU-intensive post-training optimization with calibrated images, limiting practical deployment.
◆ We introduce NanoGS, a training-free and lightweight framework for Gaussian Splat simplification.</td></tr>
<tr><td>2026-03-16</td><td>Feed-forward Gaussian Registration for Head Avatar Creation and Editing</td><td>[2603.15811](http://arxiv.org/pdf/2603.15811)</td><td>◆ We present MATCH (Multi-view Avatars from Topologically Corresponding Heads), a multi-view Gaussian registration method for high-quality head avatar creation and editing.
◆ State-of-the-art multi-view head avatar methods require time-consuming head tracking followed by expensive avatar optimization, often resulting in a total creation time of more than one day.
◆ MATCH, in contrast, directly predicts Gaussian splat textures in correspondence from calibrated multi-view images in just 0.5 seconds per frame, without requiring data preprocessing.</td></tr>
<tr><td>2026-03-16</td><td>IRIS: Intersection-aware Ray-based Implicit Editable Scenes</td><td>[2603.15368](http://arxiv.org/pdf/2603.15368)</td><td>◆ Neural Radiance Fields achieve high-fidelity scene representation but suffer from costly training and rendering, while 3D Gaussian splatting offers real-time performance with strong empirical results.
◆ Recently, solutions that harness the best of both worlds by using Gaussians as proxies to guide neural field evaluations, still suffer from significant computational inefficiencies.
◆ They typically rely on stochastic volumetric sampling to aggregate features, which severely limits rendering performance.</td></tr>
<tr><td>2026-03-16</td><td>NavGSim: High-Fidelity Gaussian Splatting Simulator for Large-Scale Navigation</td><td>[2603.15186](http://arxiv.org/pdf/2603.15186)</td><td>◆ Simulating realistic environments for robots is widely recognized as a critical challenge in robot learning, particularly in terms of rendering and physical simulation.
◆ This challenge becomes even more pronounced in navigation tasks, where trajectories often extend across multiple rooms or entire floors.
◆ In this work, we present NavGSim, a Gaussian Splatting-based simulator designed to generate high-fidelity, large-scale navigation environments.</td></tr>
<tr><td>2026-03-16</td><td>GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis</td><td>[2603.14965](http://arxiv.org/pdf/2603.14965)</td><td>◆ Novel view synthesis requires strong 3D geometric consistency and the ability to generate visually coherent images across diverse viewpoints.
◆ While recent camera-controlled video diffusion models show promising results, they often suffer from geometric distortions and limited camera controllability.
◆ To overcome these challenges, we introduce GeoNVS, a geometry-grounded novel-view synthesizer that enhances both geometric fidelity and camera controllability through explicit 3D geometric guidance.</td></tr>
<tr><td>2026-03-16</td><td>LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision</td><td>[2603.14763](http://arxiv.org/pdf/2603.14763)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time LiDAR and camera synthesis in autonomous driving simulation.
◆ However, simulating LiDAR with 3DGS remains challenging for extrapolated views beyond the training trajectory, as existing methods are typically trained on single-traversal sensor scans, suffer from severe overfitting and poor generalization to novel ego-vehicle paths.
◆ To enable reliable simulation of LiDAR along unseen driving trajectories without external multi-pass data, we present LiDAR-EVS, a lightweight framework for robust extrapolated-view LiDAR simulation in autonomous driving.</td></tr>
<tr><td>2026-03-16</td><td>E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction</td><td>[2603.14684](http://arxiv.org/pdf/2603.14684)</td><td>◆ The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS).
◆ These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting.
◆ Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution.</td></tr>
<tr><td>2026-03-15</td><td>Direct Object-Level Reconstruction via Probabilistic Gaussian Splatting</td><td>[2603.14316](http://arxiv.org/pdf/2603.14316)</td><td>◆ Object-level 3D reconstruction play important roles across domains such as cultural heritage digitization, industrial manufacturing, and virtual reality.
◆ However, existing Gaussian Splatting-based approaches generally rely on full-scene reconstruction, in which substantial redundant background information is introduced, leading to increased computational and storage overhead.
◆ To address this limitation, we propose an efficient single-object 3D reconstruction method based on 2D Gaussian Splatting.</td></tr>
<tr><td>2026-03-15</td><td>In-Field 3D Wheat Head Instance Segmentation From TLS Point Clouds Using Deep Learning Without Manual Labels</td><td>[2603.14309](http://arxiv.org/pdf/2603.14309)</td><td>◆ 3D instance segmentation for laser scanning (LiDAR) point clouds remains a challenge in many remote sensing-related domains.
◆ Successful solutions typically rely on supervised deep learning and manual annotations, and consequently focus on objects that can be well delineated through visual inspection and manual labeling of point clouds.
◆ However, for tasks with more complex and cluttered scenes, such as in-field plant phenotyping in agriculture, such approaches are often infeasible.</td></tr>
<tr><td>2026-03-15</td><td>4D Synchronized Fields: Motion-Language Gaussian Splatting for Temporal Scene Understanding</td><td>[2603.14301](http://arxiv.org/pdf/2603.14301)</td><td>◆ Current 4D representations decouple geometry, motion, and semantics: reconstruction methods discard interpretable motion structure; language-grounded methods attach semantics after motion is learned, blind to how objects move; and motion-aware methods encode dynamics as opaque per-point residuals without object-level organization.
◆ We propose 4D Synchronized Fields, a 4D Gaussian representation that learns object-factored motion in-loop during reconstruction and synchronizes language to the resulting kinematics through a per-object conditioned field.
◆ Each Gaussian trajectory is decomposed into shared object motion plus an implicit residual, and a kinematic-conditioned ridge map predicts temporal semantic variation, yielding a single representation in which reconstruction, motion, and semantics are structurally coupled and enabling open-vocabulary temporal queries that retrieve both objects and moments.</td></tr>
<tr><td>2026-03-15</td><td>S2GS: Streaming Semantic Gaussian Splatting for Online Scene Understanding and Reconstruction</td><td>[2603.14232](http://arxiv.org/pdf/2603.14232)</td><td>◆ Existing offline feed-forward methods for joint scene understanding and reconstruction on long image streams often repeatedly perform global computation over an ever-growing set of past observations, causing runtime and GPU memory to increase rapidly with sequence length and limiting scalability.
◆ We propose Streaming Semantic Gaussian Splatting (S2GS), a strictly causal, incremental 3D Gaussian semantic field framework: it does not leverage future frames and continuously updates scene geometry, appearance, and instance-level semantics without reprocessing historical frames, enabling scalable online joint reconstruction and understanding.
◆ S2GS adopts a geometry-semantic decoupled dual-backbone design: the geometry branch performs causal modeling to drive incremental Gaussian updates, while the semantic branch leverages a 2D foundation vision model and a query-driven decoder to predict segmentation masks and identity embeddings, further stabilized by query-level contrastive alignment and lightweight online association with an instance memory.</td></tr>
<tr><td>2026-03-14</td><td>PhyGaP: Physically-Grounded Gaussians with Polarization Cues</td><td>[2603.14001](http://arxiv.org/pdf/2603.14001)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated great success in modeling reflective 3D objects and their interaction with the environment via deferred rendering (DR).
◆ However, existing methods often struggle with correctly reconstructing physical attributes such as albedo and reflectance, and therefore they do not support high-fidelity relighting.
◆ Observing that this limitation stems from the lack of shape and material information in RGB images, we present PhyGaP, a physically-grounded 3DGS method that leverages polarization cues to facilitate precise reflection decomposition and visually consistent relighting of reconstructed objects.</td></tr>
<tr><td>2026-03-13</td><td>Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting</td><td>[2603.12796](http://arxiv.org/pdf/2603.12796)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) deliver high-quality rendering, yet the Gaussian representation exposes a new attack surface, the resource-targeting attack.
◆ This attack poisons training images, excessively inducing Gaussian growth to cause resource exhaustion.
◆ Although efficiency-oriented methods such as smoothing, thresholding, and pruning have been explored, these spatial-domain strategies operate on visible structures but overlook how stealthy perturbations distort the underlying spectral behaviors of training data.</td></tr>
<tr><td>2026-03-13</td><td>Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation</td><td>[2603.12766](http://arxiv.org/pdf/2603.12766)</td><td>◆ Recent advances in 3D scene editing using NeRF and 3DGS enable high-quality static scene editing.
◆ In contrast, dynamic scene editing remains challenging, as methods that directly extend 2D diffusion models to 4D often produce motion artifacts, temporal flickering, and inconsistent style propagation.
◆ We introduce Catalyst4D, a framework that transfers high-quality 3D edits to dynamic 4D Gaussian scenes while maintaining spatial and temporal coherence.</td></tr>
<tr><td>2026-03-13</td><td>LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction</td><td>[2603.12647](http://arxiv.org/pdf/2603.12647)</td><td>◆ Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis.
◆ However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting.
◆ To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures.</td></tr>
<tr><td>2026-03-12</td><td>AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies</td><td>[2603.11969](http://arxiv.org/pdf/2603.11969)</td><td>该论文的核心贡献是提出了AstroSplat，一个用于小天体渲染与重建的、基于物理的神经渲染框架。其核心创新点在于将神经辐射场技术与天体物理模型相结合，解决了传统方法在空间任务应用中的关键局限。

◆ 核心创新是首次将基于物理的行星反射率模型引入高斯泼溅框架，取代了传统仅基于外观的球谐函数参数化。
◆ 该方法能够显式地建模光与表面的物理相互作用，从而不仅能渲染图像，还能进行表面物理特性（如材质）的光度学表征。
◆ 此框架提升了从小天体原位图像进行自主三维重建的精度和物理真实性，直接服务于任务规划与科学分析。
◆ 研究在NASA黎明号的真实任务图像上验证了其有效性，证明了其在渲染质量和表面重建准确性上优于主流神经渲染方法。</td></tr>
<tr><td>2026-03-12</td><td>Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting</td><td>[2603.11543](http://arxiv.org/pdf/2603.11543)</td><td>该论文提出Mango-GS，一个用于动态4D场景重建的新框架，旨在提升时空一致性并实现实时渲染。其核心贡献与创新点如下：

◆ 提出多帧节点引导的4D高斯溅射框架，通过引入稀疏控制节点来高效建模动态场景，避免传统逐帧优化导致的过拟合与时间不一致问题。

◆ 设计基于时序Transformer的运动建模模块，在一个短时帧窗口内捕捉运动依赖关系，从而生成时间一致的形变场，增强了动态序列的连贯性。

◆ 采用解耦的节点表示方法，将每个控制节点表示为规范位置与潜在编码的组合，这为运动传播提供了稳定的语义锚点，有效防止了大运动下的对应关系漂移。

◆ 提出输入掩码策略与两种多帧损失函数，增强了训练过程的鲁棒性，并驱动模型学习更准确的动态先验，进一步提升了重建质量。

◆ 通过端到端训练，该框架在保持实时渲染速度的同时，实现了高质量的动态场景重建，在多项实验中达到了先进的性能水平。</td></tr>
<tr><td>2026-03-12</td><td>Mobile-GS: Real-time Gaussian Splatting for Mobile Devices</td><td>[2603.11531](http://arxiv.org/pdf/2603.11531)</td><td>该论文的核心贡献是提出了一种专为移动设备设计的实时3D高斯泼溅渲染方法Mobile-GS，旨在解决原方法计算量大、存储成本高而难以在移动端部署的问题。其创新点主要包括：

◆ 提出一种深度感知的顺序无关渲染方案，通过消除耗时的高斯深度排序过程，解决了alpha混合这一主要计算瓶颈，大幅加速了渲染。

◆ 设计了一种神经视角依赖增强策略，通过结合视角方向、3D高斯几何与外观属性来更精确地建模视角依赖效果，从而弥补了顺序无关渲染可能带来的透明区域伪影。

◆ 引入了一阶球谐蒸馏、神经向量量化技术以及基于贡献度的剪枝策略，在神经网络辅助下减少了高斯图元数量并压缩了表示，从而降低了存储需求，使其更适合内存受限的移动平台。

最终，Mobile-GS在保持高视觉质量的同时，实现了实时渲染与紧凑的模型尺寸，为移动应用提供了可行的解决方案。</td></tr>
<tr><td>2026-03-11</td><td>InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction</td><td>[2603.11298](http://arxiv.org/pdf/2603.11298)</td><td>该论文提出InstantHDR，一个前馈网络，用于从多曝光低动态范围图像进行高动态范围三维重建。其核心贡献与创新点如下：

◆ 提出首个前馈网络，能够从未标定的多曝光图像集合中，通过单次前向传播直接重建三维HDR场景，无需已知相机位姿或耗时逐场景优化。

◆ 设计了几何引导的外观建模模块，专门用于多曝光图像融合，有效处理不同曝光下的外观变化。

◆ 引入一个元网络，实现可泛化的场景特定色调映射，提升了模型对不同场景的适应能力。

◆ 针对HDR数据缺乏的问题，构建了名为HDR-Pretrain的预训练数据集，包含168个Blender渲染场景、多样光照类型和多种相机响应函数，以支持可泛化前馈HDR模型的训练。

◆ 该方法在合成质量上媲美基于优化的先进方法，同时重建速度在单次前向模式下提升约700倍，在后优化设置下也提升约20倍，实现了效率与质量的平衡。</td></tr>
<tr><td>2026-03-11</td><td>Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning</td><td>[2603.11219](http://arxiv.org/pdf/2603.11219)</td><td>该论文的核心贡献是提出了Senna-2模型，旨在解决视觉语言模型与端到端驾驶策略之间决策与规划不一致的关键问题，从而提升自动驾驶系统的安全性与可靠性。

◆ 核心创新在于明确提出了“双系统一致性”问题，即VLM的高层语义决策与E2E模型的底层轨迹规划需保持一致，这是对现有研究盲点的直接回应。
◆ 提出了一种全新的、面向一致性的三阶段训练范式，系统性地从预训练、开环对齐到闭环强化学习逐步优化模型。
◆ 设计了一个决策适配器，能够将VLM的决策以隐式嵌入的形式有效传递给E2E策略，作为连接两个系统的关键桥梁。
◆ 在闭环对齐阶段，创新性地采用了自底向上的分层强化学习方法，并在3D高斯泼溅环境中进行训练，以强化策略的安全与效率。
◆ 实验验证充分，结果表明该方法在双系统一致性指标上显著提升，并有效降低了轨迹误差与事故率，证明了其优越性。</td></tr>
<tr><td>2026-03-11</td><td>S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs</td><td>[2603.10893](http://arxiv.org/pdf/2603.10893)</td><td>该论文提出了一种名为S2D的新流程，旨在以极少的输入图像实现高质量的三维重建。其核心贡献与创新点如下：

◆ 提出了一种新颖的S2D流程，桥接了稀疏点云与3D高斯泼溅两种表示，解决了它们在稀疏输入下渲染不真实和质量严重下降的问题。
◆ 设计了一个高效的一步扩散模型，用于提升稀疏点云的质量，能够修复图像伪影，生成高保真的新视角引导图像。
◆ 为保障三维场景的一致性，提出了一种包含随机样本丢弃和加权梯度的重建策略，使模型能够从稀疏输入视图鲁棒地拟合出密集的新视图。
◆ 通过上述方法，显著降低了对输入数据量的需求，在现有方法中实现了用最少拍摄数量重建稳定场景，拓展了3D高斯泼溅技术的应用边界。
◆ 大量实验验证了该方法的有效性，其在生成新视角的一致性和稀疏视图重建质量方面均达到了领先水平。</td></tr>
<tr><td>2026-03-11</td><td>PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction</td><td>[2603.10801](http://arxiv.org/pdf/2603.10801)</td><td>该论文的核心贡献是提出了PolGS++框架，用于快速且高质量地重建反射表面。其创新点主要在于将物理模型与高效的3D高斯泼溅技术相结合，具体如下：

◆ 首次将偏振双向反射分布函数模型集成到3D高斯泼溅中，显式解耦了表面的漫反射和镜面反射分量，从而提供了物理上更准确的反射建模。
◆ 提出了一种深度引导的可见性掩码获取机制，无需昂贵的射线追踪计算，即可在高斯泼溅中实现基于偏振角的切空间一致性约束。
◆ 整个物理引导的设计显著提升了反射表面重建的质量与效率，仅需约10分钟的训练时间，在几何细节与法线恢复上优于原3D高斯泼溅方法。
◆ 通过合成与真实数据集的广泛实验，验证了该方法在反射表面重建任务上的有效性和优越性。</td></tr>
<tr><td>2026-03-11</td><td>Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting</td><td>[2603.10638](http://arxiv.org/pdf/2603.10638)</td><td>该论文针对物理AI中训练与部署视角差异的问题，提出了一种基于3D高斯溅射的新视角扩展方法Splat2Real，以提升单目RGB到3D感知的视角鲁棒性。  
◆ 核心思想是将真实到渲染再到真实的单目深度预训练构建为一种模仿学习框架，学生深度网络模仿从场景网格数字孪生专家渲染的度量深度和可见性。  
◆ 提出“新视角扩展”概念，强调性能提升关键不在于原始视角数量，而在于所添加视角的质量与新颖性。  
◆ 设计了CN-Coverage课程策略，通过几何增益和外推惩罚贪婪选择新视角，并引入质量感知的防护回退机制以应对低可靠性教师模型。  
◆ 在TUM RGB-D数据集上的实验表明，该方法相比朴素扩展或Robot/Coverage等策略，能有效缓解最差情况下的性能衰退，并在中高预算下实现更稳定的性能。  
◆ 通过下游控制代理任务验证了该方法在视角变化下对安全性/进度权衡的改善，体现了其在具身AI中的实际应用价值。</td></tr>
<tr><td>2026-03-11</td><td>P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video</td><td>[2603.10551](http://arxiv.org/pdf/2603.10551)</td><td>该论文提出了P-GSVC，首个用于图像和视频的统一、可扩展分层渐进式2D高斯泼溅框架。

◆ 首次提出了分层渐进式的2D高斯泼溅框架，为图像和视频的高斯表示提供了一个统一的、可扩展的解决方案。
◆ 将2D高斯泼溅组织为基础层和多个增强层，实现了从粗到细的重建过程。
◆ 提出了一种联合训练策略，同时优化所有层的高斯泼溅，而非传统的逐层顺序训练。
◆ 该联合训练策略确保了层间兼容性和稳定的渐进重建，显著提升了性能。
◆ 实验证明，该框架在质量和分辨率上均支持可扩展性，其联合训练相比逐层训练在视频上PSNR提升最高达1.9 dB，在图像上提升最高达2.6 dB。</td></tr>
<tr><td>2026-03-12</td><td>SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning</td><td>[2603.10446](http://arxiv.org/pdf/2603.10446)</td><td>该论文的核心贡献是提出了SignSparK框架，通过稀疏关键帧学习实现了高效、高质量的多语种手语生成，解决了现有方法在自然度与流畅性上的权衡难题。

◆ 提出了一种新颖的稀疏关键帧训练范式，通过预测离散关键帧之间的密集运动，有效缓解了直接回归方法中的“回归到均值”问题，从而生成更自然、符合人体运动学的手语动作。
◆ 引入了FAST模型，这是一个高效的自动手语分割模型，能够自动挖掘精确的时间边界，为大规模关键帧提取提供了基础。
◆ 构建了基于条件流匹配（CFM）的大规模框架SignSparK，利用提取的关键帧合成SMPL-X和MANO模型下的3D手语序列，并首次实现了关键帧到姿态（KF2P）的生成，使得对序列进行精确的时空编辑成为可能。
◆ 采用基于重建的CFM目标，实现了在少于十次采样步骤下的高保真合成，这使得框架能够高效扩展到四种手语，建立了迄今为止最大的多语种手语生产框架。
◆ 通过集成3D高斯泼溅技术进行照片级真实感渲染，并在多项任务和基准测试中验证了SignSparK达到了最新的最优性能。</td></tr>
<tr><td>2026-03-10</td><td>ReCoSplat: Autoregressive Feed-Forward Gaussian Splatting Using Render-and-Compare</td><td>[2603.09968](http://arxiv.org/pdf/2603.09968)</td><td>ReCoSplat提出了一种用于在线新视角合成的自回归前馈高斯泼溅模型。其核心贡献与创新点如下：

◆ 模型支持多种输入模式，包括有无相机位姿以及有无相机内参，增强了实用性。
◆ 采用基于相机位姿组装局部高斯的方法，相比在规范空间预测，具有更好的扩展性。
◆ 针对训练时使用真值位姿而推理时使用预测位姿导致的分布不匹配问题，创新性地引入了“渲染-比较”模块。该模块通过从预测视角渲染当前重建结果并与新观测进行比较，提供了一个稳定的条件信号，有效补偿了位姿误差。
◆ 为支持长序列处理，提出了一种混合KV缓存压缩策略，结合了早期层截断和块级选择性保留技术，在处理超过100帧时将KV缓存大小减少了90%以上。
◆ 该模型在分布内和分布外的多个基准测试中，于不同输入设置下均实现了最先进的性能。</td></tr>
<tr><td>2026-03-10</td><td>GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System</td><td>[2603.09718](http://arxiv.org/pdf/2603.09718)</td><td>本文提出GSStream，一个基于3D高斯泼溅的实时体积场景流式传输系统。其核心贡献与创新点如下：

◆ 设计了一个新颖的体积场景流式传输系统，专门支持高数据量的3D高斯泼溅格式，旨在解决其带宽消耗巨大的挑战。
◆ 集成了一个协同视口预测模块，通过融合多用户协同先验和用户历史视口序列先验，更准确地预测用户未来行为。
◆ 提出了一个基于深度强化学习的码率自适应模块，以应对码率自适应问题中状态和动作空间多变性的挑战，从而实现高效的体积场景数据传输。
◆ 首次构建了用于体积场景的用户视口轨迹数据集，以支持相关模型的训练和流式传输模拟。
◆ 通过大量实验验证，该系统在视觉质量和网络使用效率方面均优于现有代表性的体积场景流式传输系统。</td></tr>
<tr><td>2026-03-10</td><td>ProGS: Towards Progressive Coding for 3D Gaussian Splatting</td><td>[2603.09703](http://arxiv.org/pdf/2603.09703)</td><td>该论文针对3D高斯泼溅（3DGS）数据量庞大、不利于存储与传输的问题，提出了一种支持渐进式编码的新方法ProGS，以适配流式传输等动态带宽场景。其核心贡献与创新点如下：

◆ 首次为3DGS数据引入了八叉树组织结构，实现了高效的渐进式编码，使数据能够分层传输与重建。
◆ 提出了一种流式友好的编解码器ProGS，它通过动态调整八叉树中的锚节点，在保证渲染质量的同时实现可扩展的数据压缩。
◆ 设计了互信息增强机制，利用八叉树层级节点间的相关性，有效减少了结构冗余。
◆ 该方法在压缩性能上取得显著突破，相比原始3DGS格式实现了45倍的存储缩减，同时视觉质量还提升了超过10%。</td></tr>
<tr><td>2026-03-10</td><td>VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM</td><td>[2603.09673](http://arxiv.org/pdf/2603.09673)</td><td>该论文的核心贡献是提出了VarSplat，一个将显式不确定性建模融入3D高斯溅射（3DGS）的RGB-D SLAM系统，以提升其在挑战性场景中的鲁棒性。

◆ 核心创新在于为每个高斯图元（splat）显式地学习外观方差，从而量化其不确定性。
◆ 提出利用全方差定律与alpha合成相结合的方法，通过高效的单次光栅化，即可渲染出可微的逐像素不确定性图。
◆ 该不确定性图被系统地用于指导SLAM的关键环节：在跟踪、子地图配准和回环检测中，系统能自动聚焦于可靠性高的图像区域，抑制不可靠区域（如低纹理、透明或高反光表面）的干扰。
◆ 这使得位姿估计和全局对齐的优化过程更加稳定，有效减少了漂移。
◆ 在合成与真实世界多个数据集上的实验表明，VarSplat在跟踪鲁棒性、建图以及新视图合成渲染质量上，达到或超越了现有先进的稠密RGB-D SLAM方法。</td></tr>
<tr><td>2026-03-10</td><td>DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics</td><td>[2603.09668](http://arxiv.org/pdf/2603.09668)</td><td>DiffWind的核心贡献是提出了一个统一的、物理可微的框架，用于从视频中重建并模拟风与物体的复杂交互动力学。

◆ 提出了一种新颖的统一框架，将风-物体交互建模、基于视频的动态重建与正向物理仿真融合在一个可微分系统中。
◆ 创新性地采用基于网格的风力场与基于3D高斯溅射的物体粒子系统表示，并使用物质点法来模拟两者间的相互作用。
◆ 设计了一个联合优化框架，通过可微分渲染与仿真，从视频中共同推断出时空变化的风力场和物体的运动。
◆ 引入了基于格子玻尔兹曼方法的物理约束，确保推断出的风力场符合流体动力学规律，增强了结果的物理真实性。
◆ 该框架不仅支持高保真重建，还能直接进行新风速条件下的正向仿真，并实现了如风场重定向等新应用。
◆ 为此研究领域贡献了一个包含合成与真实场景的风驱物体数据集WD-Objects。</td></tr>
<tr><td>2026-03-10</td><td>X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models</td><td>[2603.09632](http://arxiv.org/pdf/2603.09632)</td><td>X-GS的核心贡献是提出了一个统一且可扩展的开放框架，旨在解决现有3D高斯溅射（3DGS）技术各自为政、难以与下游多模态模型集成的问题。其创新点可总结如下：

◆ 首创了一个统一的开放框架，将在线SLAM、语义增强、无位姿图像处理等多种孤立的3DGS技术整合到一个系统中。
◆ 设计了核心高效管线X-GS-Perceiver，能够从无位姿的RGB（或RGB-D）视频流中，实时协同优化场景几何与相机位姿。
◆ 创新性地将视觉基础模型的高维语义特征提炼并注入3D高斯表示中，创建了富含语义信息的3D高斯场景。
◆ 通过新颖的在线向量量化模块、GPU加速的网格采样方案及高度并行化的管线设计，实现了整个系统的实时运行性能。
◆ 开发了X-GS-Thinker组件，使语义化的3D高斯能够被视觉-语言模型直接利用，从而解锁了物体检测、零样本描述生成等一系列下游多模态任务。</td></tr>
<tr><td>2026-03-10</td><td>DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction</td><td>[2603.09291](http://arxiv.org/pdf/2603.09291)</td><td>该论文的核心贡献是提出了一种名为DenoiseSplat的前馈式方法，用于从有噪声的多视角图像中直接重建高质量的3D场景。

◆ 提出DenoiseSplat，这是首个面向噪声输入的前馈式3D高斯溅射重建方法，实现了端到端的噪声鲁棒性重建与视图合成。
◆ 构建了一个大规模、场景一致的噪声-干净配对基准数据集（基于RE10K），系统性地注入了高斯、泊松、散斑和椒盐等多种类型与强度的噪声。
◆ 采用轻量级的前馈网络骨干（类似MVSplat），无需任何3D真值监督，仅使用干净的2D渲染图像作为训练目标，简化了流程并提升了实用性。
◆ 在多种噪声类型和强度下，其性能均优于原始MVSplat以及“图像去噪后重建”的两阶段基线，在PSNR、SSIM和LPIPS指标上全面领先。</td></tr>
<tr><td>2026-03-10</td><td>Learning Convex Decomposition via Feature Fields</td><td>[2603.09285](http://arxiv.org/pdf/2603.09285)</td><td>该论文的核心贡献是提出了一种通过特征场学习进行凸分解的新方法，实现了首个面向开放世界的、前馈式的凸分解模型。

◆ 提出了一种创新的学习框架，通过预测连续特征场而非直接分解，将凸分解问题转化为特征学习与聚类问题。
◆ 设计了一个纯粹基于几何、自监督的训练目标，该目标源自凸性的经典定义，无需人工标注数据。
◆ 该方法不仅能针对单个形状进行优化，更重要的是，其特征预测能力支持在大规模数据集上进行可扩展的自监督学习。
◆ 由此训练出了首个能够处理开放世界对象的凸分解学习模型，其分解质量优于现有方法。
◆ 该模型展现出强大的泛化能力，能够适用于多种3D表示形式，包括网格、CAD模型甚至高斯泼溅表示。</td></tr>
<tr><td>2026-03-10</td><td>Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists</td><td>[2603.09277](http://arxiv.org/pdf/2603.09277)</td><td>该论文的核心目标是提升3D高斯溅射（3DGS）模型的学习效率，其核心贡献在于通过缩短渲染每个像素所需的高斯列表来加速训练过程，且不牺牲渲染质量。具体创新点如下：

◆ 提出通过定期重置高斯尺度来缩小每个高斯函数的尺寸，使其覆盖更少的邻近像素，从而直接缩短了像素对应的高斯列表。

◆ 引入针对alpha混合过程的熵约束，以锐化每条射线上高斯函数的权重分布，增大主导权重并减小次要权重，使每个高斯函数更专注于其主导像素，减少对邻近像素的影响。

◆ 将上述方法集成到一个渐进式分辨率调度器中，通过逐步增加渲染分辨率来进一步提升整体训练效率。

◆ 在广泛使用的基准测试中验证了该方法，结果表明其在保持渲染质量的同时，显著提升了训练效率，优于现有先进方法。</td></tr>
<tr><td>2026-03-09</td><td>SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training</td><td>[2603.08997](http://arxiv.org/pdf/2603.08997)</td><td>该论文针对3D高斯泼溅训练效率问题，提出了一种名为SkipGS的高效训练方法。其核心贡献与创新点如下：

◆ 首次观察到并利用了3DGS后致密化训练阶段存在显著的更新冗余，即许多采样视图的损失已趋于稳定，其反向传播带来的梯度收益递减。
◆ 提出了一种新颖的视图自适应反向门控机制，该机制基于每个视图的损失统计信息，动态决定是否跳过当前视图的反向传播。
◆ 方法设计上，始终保持前向传播以更新损失统计，但仅在采样视图的损失与其近期基线一致时选择性跳过反向传播，同时强制执行最小反向预算以确保优化稳定性。
◆ 实现了显著的训练加速，在Mip-NeRF 360数据集上，相比原始3DGS，端到端训练时间减少23.1%，其中后致密化阶段训练时间大幅减少42.0%，且重建质量相当。
◆ 该方法具有即插即用和良好兼容性，因其仅改变反向传播的时机，而未修改渲染器、表征或损失函数，故可与其他效率优化策略叠加使用以获得进一步的加速效果。</td></tr>
<tr><td>2026-03-09</td><td>ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting</td><td>[2603.08661](http://arxiv.org/pdf/2603.08661)</td><td>本论文提出了ImprovedGS+，一种针对3D高斯泼溅（3DGS）的高性能重新实现策略，其核心贡献在于通过底层系统优化，在速度、质量和资源效率上实现了新的帕累托最优。

◆ 实现了从高层Python到硬件优化的C++/CUDA内核的彻底转变，显著减少了主机-设备同步开销和训练延迟。
◆ 设计了创新的长轴分割（LAS）CUDA内核，以提升计算效率。
◆ 引入了基于拉普拉斯算子的自定义重要性评估内核，并结合非极大值抑制（NMS）处理边缘得分，优化了高斯分布的管理。
◆ 采用了自适应的指数尺度调度器，以改进训练过程的控制。
◆ 实验证明，该方法在Mip-NeRF360数据集上建立了新的性能权衡前沿，能以更少的参数和更短的训练时间，实现优于或媲美先进基线的重建质量。</td></tr>
<tr><td>2026-03-09</td><td>Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction</td><td>[2603.08503](http://arxiv.org/pdf/2603.08503)</td><td>该论文的核心贡献是提出了Spherical-GOF，一个用于全景图像的高质量三维场景重建框架。其核心创新点在于解决了将3D高斯泼溅技术直接应用于全景相机模型时产生的失真和几何不一致问题。

◆ 提出了一个创新的全景高斯渲染框架，将高斯不透明度场的射线采样直接定义在球面射线空间的单位球体上，确保了射线与高斯体交互在全景渲染中的一致性。
◆ 推导了一个保守的球面包围规则，用于快速进行射线-高斯体剔除，从而显著提升了球面射线投射的效率和鲁棒性。
◆ 引入了一种球面滤波方案，该方案能根据全景图像像素采样中变化着的失真情况，自适应地调整高斯足迹，有效缓解了失真问题。
◆ 在标准全景基准数据集上的实验表明，该方法在保持优异光度质量的同时，大幅提升了几何一致性，深度重投影误差降低了57%，循环匹配内点率提高了21%。
◆ 此外，论文还贡献了一个名为OmniRob的真实世界机器人全景数据集，并验证了所提方法在该数据集上的良好泛化能力。</td></tr>
<tr><td>2026-03-09</td><td>Improving Continual Learning for Gaussian Splatting based Environments Reconstruction on Commercial Off-the-Shelf Edge Devices</td><td>[2603.08499](http://arxiv.org/pdf/2603.08499)</td><td>该论文的核心贡献是提出了一种精度自适应的优化框架，使得基于变分贝叶斯高斯泼溅（VBGS）的持续学习三维重建方法能够在资源受限的边缘设备上高效运行。其创新点具体如下：

◆ 首次实现了VBGS在商用现成边缘设备（如Jetson Orin Nano）上的训练，将新颖视图合成（NVS）的持续学习能力引入边缘机器人等内存和延迟预算严格的应用场景。
◆ 提出一个系统性的优化框架，通过剖析VBGS，精准定位了其在内存和延迟方面的性能瓶颈。
◆ 设计了内存主导内核的融合技术，有效减少了计算过程中实例化的中间张量数量，从而大幅降低了内存占用。
◆ 开发了一种基于有界相对误差的混合精度搜索方法，能够自动为不同操作分配合适的计算精度，在保证质量的同时优化计算效率。
◆ 该框架在保持甚至提升原有VBGS重建质量的前提下，在多个数据集上验证了其卓越性能，例如将峰值内存从9.44 GB降至1.11 GB，训练时间从约234分钟缩短至约61分钟，并在嵌入式平台上实现了相比3DGS高达19倍的每帧延迟降低。</td></tr>
<tr><td>2026-03-09</td><td>HDR-NSFF: High Dynamic Range Neural Scene Flow Fields</td><td>[2603.08313](http://arxiv.org/pdf/2603.08313)</td><td>本文提出HDR-NSFF，实现了从传统2D图像融合到4D时空建模的范式转变，用于从交替曝光单目视频中重建动态高动态范围场景。其核心贡献与创新点如下：

◆ 提出首个将动态HDR重建与4D时空建模统一的端到端框架，能联合建模HDR辐射场、三维场景流、几何与色调映射，确保物理合理性与全局一致性。
◆ 实现了方法兼容性，可支持基于神经辐射场或4D高斯泼溅的动态场景表示，提升了框架的灵活性。
◆ 引入了曝光不变的运动估计方法，通过结合DINO特征的语义光流来增强动态场景中运动估计的鲁棒性。
◆ 利用生成式先验作为正则化器，以补偿单目捕获中有限的观测信息以及过曝/欠曝导致的数据丢失，提高了重建质量。
◆ 创建并发布了首个真实世界动态HDR场景数据集HDR-GoPro，专门用于评估HDR时空视图合成任务。实验表明，该方法在挑战性曝光变化下能恢复精细的辐射细节与连贯的动态，取得了先进的性能。</td></tr>
<tr><td>2026-03-09</td><td>DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving</td><td>[2603.08254](http://arxiv.org/pdf/2603.08254)</td><td>该论文提出DynamicVGGT，一个用于自动驾驶中动态4D场景重建的统一前馈框架。其核心贡献是将静态3D感知模型VGGT扩展至动态4D重建，以解决动态物体和复杂场景变化的挑战。

主要创新点包括：
◆ 提出一种联合预测当前与未来点地图的方法，在共享参考坐标系中通过时序对应隐式学习动态点表示。
◆ 设计了运动感知时序注意力模块，以高效捕捉时序依赖并学习运动连续性。
◆ 引入动态3D高斯泼溅头，利用可学习运动令牌在场景流监督下预测高斯速度，显式建模点运动。
◆ 通过连续的3D高斯优化细化动态几何，实现了复杂驾驶场景下鲁棒且高精度的前馈式4D动态重建。</td></tr>
<tr><td>2026-03-09</td><td>Fast Low-light Enhancement and Deblurring for 3D Dark Scenes</td><td>[2603.08133](http://arxiv.org/pdf/2603.08133)</td><td>该论文提出FLED-GS框架，用于解决低光照、噪声和运动模糊图像的新视角合成难题。其核心贡献在于通过交替优化实现三维场景的快速恢复与增强。

◆ 提出交替循环框架，将三维场景恢复重新定义为增强与重建的迭代过程，避免传统顺序处理导致的伪影问题。
◆ 引入渐进式亮度锚点机制，逐步恢复场景亮度，有效防止噪声放大干扰去模糊或几何重建。
◆ 设计噪声感知的三维高斯溅射重建方法，在重建过程中同步估计并抑制噪声，同时为下一阶段生成清洁先验。
◆ 结合现成二维去模糊工具与三维重建的协同优化，在保证质量的同时大幅提升效率。
实验表明该方法在性能上超越现有最佳技术，训练速度提升21倍，渲染速度提升11倍，实现了高质量实时恢复。</td></tr>
<tr><td>2026-03-08</td><td>SGI: Structured 2D Gaussians for Efficient and Compact Large Image Representation</td><td>[2603.07789](http://arxiv.org/pdf/2603.07789)</td><td>该论文提出了一种名为结构化高斯图像（SGI）的新框架，旨在高效、紧凑地表示高分辨率图像。其核心贡献与创新点如下：

◆ 提出了“种子”结构，将图像分解为多尺度局部空间，每个种子对应一个空间连贯区域，从而为原本无组织的2D高斯图元引入了结构正则性。

◆ 通过种子结合轻量级多层感知机（MLP）来生成结构化的隐式2D神经高斯，取代了独立存储数百万个高斯图元的传统方式，显著减少了参数冗余。

◆ 设计了基于种子层级的熵压缩方法，利用其结构特性进一步降低了整体存储开销。

◆ 针对高分辨率图像优化种子参数困难的问题，创新性地采用了一种从粗到精的多尺度拟合策略，大幅加速了优化收敛过程。

实验表明，该方法在保持甚至提升图像质量的同时，相比先前的非量化2D高斯方法实现了最高7.5倍的压缩，优化速度也提升了1.6至6.5倍。</td></tr>
<tr><td>2026-03-08</td><td>Ref-DGS: Reflective Dual Gaussian Splatting</td><td>[2603.07664](http://arxiv.org/pdf/2603.07664)</td><td>该论文针对具有强镜面反射（尤其是近场镜面反射）场景的重建与新视图合成难题，提出了Ref-DGS框架。其核心贡献与创新点如下：

◆ 提出了一种反射双高斯溅射框架，通过将表面重建与镜面反射解耦，在基于高效光栅化的流程中解决了质量与速度的权衡问题。
◆ 引入了双高斯场景表示，包含负责几何的几何高斯和负责捕捉近场镜面相互作用的局部反射高斯，无需显式光线追踪即可建模近场镜面反射。
◆ 同时结合了一个全局环境反射场，用于建模远场镜面反射，从而完整地处理了不同范围的反射现象。
◆ 设计了一个轻量级、物理感知的自适应混合着色器，用于融合全局与局部反射特征以预测镜面辐射度，增强了渲染的真实感与物理准确性。
实验表明，该方法在反射场景上取得了领先性能，且训练速度显著快于基于光线追踪的高斯方法。</td></tr>
<tr><td>2026-03-08</td><td>Holi-Spatial: Evolving Video Streams into Holistic 3D Spatial Intelligence</td><td>[2603.07660](http://arxiv.org/pdf/2603.07660)</td><td>本文的核心贡献是构建了首个全自动、大规模、空间感知的多模态3D数据集Holi-Spatial，并基于此创建了高质量基准数据集Holi-Spatial-4M，旨在推动空间智能的发展。
◆ 首创了从原始视频流全自动构建大规模3D空间数据集的完整流程，无需人工标注，突破了传统方法依赖有限人工数据的可扩展性瓶颈。
◆ 提供了多层次、密集的空间监督信号，包括几何精确的3D高斯溅射重建与深度图、物体级语义标注、关系语义以及对应的空间问答对。
◆ 构建了首个大规模高质量3D语义数据集Holi-Spatial-4M，包含数万个优化3D场景、数百万个2D掩码、3D边界框、实例描述、3D grounding实例和空间QA对，覆盖多样化的几何、关系和语义推理任务。
◆ 在数据质量上表现出色，在多个公开数据集上显著优于现有的前馈和逐场景优化方法。
◆ 利用该数据集对视觉语言模型进行空间推理任务的微调，有效提升了模型在相关任务上的性能。</td></tr>
<tr><td>2026-03-08</td><td>EmbedTalk: Triplane-Free Talking Head Synthesis using Embedding-Driven Gaussian Deformation</td><td>[2603.07604](http://arxiv.org/pdf/2603.07604)</td><td>该论文提出了一种名为EmbedTalk的实时说话头合成方法，其核心贡献在于用学习嵌入替代传统的三平面编码，以驱动3D高斯形变。具体创新点如下：

◆ 首次将学习嵌入机制引入说话头合成领域，用于建模语音驱动的面部形变，取代了以往依赖三平面编码的标准方案。

◆ 克服了三平面表示因网格分辨率和3D到2D投影近似误差带来的局限性，提升了表示的连续性和精确性。

◆ 所提出的方法在渲染质量、唇部同步和运动一致性方面均优于现有的基于3D高斯泼溅的方法，并可媲美先进的生成模型。

◆ 嵌入表示实现了显著的模型压缩，模型尺寸更小，计算效率更高。

◆ 在移动GPU（RTX 2060 6 GB）上实现了超过60 FPS的实时推理速度，为轻量级部署提供了可能。</td></tr>
<tr><td>2026-03-06</td><td>EntON: Eigenentropy-Optimized Neighborhood Densification in 3D Gaussian Splatting</td><td>[2603.06216](http://arxiv.org/pdf/2603.06216)</td><td>本文提出了一种名为EntON的新型3D高斯泼溅（3DGS）邻域致密化策略，旨在实现几何精确且高质量的三维重建。其核心贡献与创新点如下：

◆ 提出了一种几何感知的自适应致密化策略，替代了传统仅依赖视图空间位置梯度幅度的致密化方法。
◆ 引入了基于局部协方差矩阵特征值的“特征熵”概念，用以量化每个高斯中心k近邻区域的局部结构有序性。
◆ 设计了一个交替优化框架，将标准的基于梯度的致密化与新颖的特征熵感知致密化相结合。
◆ 在优化过程中，优先在低特征熵（有序、平坦）的邻域分裂高斯以捕捉精细表面几何，并在高特征熵（无序、球形）区域进行修剪。
◆ 该方法在多个数据集上验证有效，显著提升了重建的几何精度与渲染质量，同时大幅减少了所需的高斯数量及训练时间，实现了精度、质量与效率的更好平衡。</td></tr>
<tr><td>2026-03-06</td><td>VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction</td><td>[2603.06210](http://arxiv.org/pdf/2603.06210)</td><td>该论文提出VG3S框架，旨在解决自动驾驶场景理解中语义占据预测任务的关键问题。其核心贡献在于将视觉基础模型的几何先验知识融入基于高斯泼溅的占据预测中，显著提升了预测精度。

◆ 核心创新是提出了视觉几何接地的三维高斯泼溅框架，首次将视觉基础模型强大的三维几何理解能力与高斯泼溅的轻量化占据建模相结合。
◆ 设计了一种即插即用的分层几何特征适配器，能够有效提取并转换冻结的视觉基础模型中的通用三维几何先验。
◆ 该适配器通过特征聚合、任务特定对齐和多尺度重组三个关键步骤，实现了跨视图三维几何信息的有效注入与利用。
◆ 方法在nuScenes基准测试上取得显著性能提升，IoU和mIoU分别比基线大幅提高12.6%和7.5%。
◆ 框架具有良好的通用性和可移植性，可无缝适配于不同的视觉基础模型，并一致性地提升占据预测精度，验证了引入预训练几何先验的巨大价值。</td></tr>
<tr><td>2026-03-06</td><td>Transforming Omnidirectional RGB-LiDAR data into 3D Gaussian Splatting</td><td>[2603.06061](http://arxiv.org/pdf/2603.06061)</td><td>该论文的核心贡献是提出了一套将已存档的全向RGB图像和LiDAR点云数据转化为可用于3D高斯泼溅（3DGS）建模的初始化资源的流程，从而利用大量被弃置的常规传感器数据来高效构建大规模数字孪生场景。

其创新点主要包括：
◆ 提出一个完整的全向RGB-LiDAR数据重用流程，将原本因传输限制和缺乏处理流程而被丢弃或未充分利用的存档传感器数据，转化为3DGS可用的高质量初始化资源。
◆ 设计了ERP到立方体贴图的转换模块，以确定性的空间锚定方法，解决了全向图像固有非线性失真导致的运动恢复结构（SfM）跟踪不可靠的难题。
◆ 引入了PRISM策略，这是一种基于颜色分层的点云下采样方法，有效处理了密集、无组织的LiDAR点云带来的计算开销问题。
◆ 通过基于快速点特征直方图（FPFH）的全局配准和迭代最近点（ICP）算法，可靠地桥接了RGB与LiDAR多模态数据，成功将大量废弃数据转化为可用的SfM几何结构。
◆ 该流程提供的LiDAR增强型初始化，在结构复杂的场景中，相比纯视觉基线方法，能持续提升最终3DGS模型的渲染保真度。</td></tr>
<tr><td>2026-03-06</td><td>FTSplat: Feed-forward Triangle Splatting Network</td><td>[2603.05932](http://arxiv.org/pdf/2603.05932)</td><td>该论文的核心贡献是提出了一种前馈式三角形基元生成框架，能够从多视角图像直接预测连续的三角形表面，以快速生成可用于仿真和机器人的高保真三维模型。

◆ 提出前馈式三角形生成框架，无需针对每个场景进行耗时优化或后处理，单次前向传播即可生成仿真就绪的模型。
◆ 设计了像素对齐的三角形生成模块，直接从校准的多视角图像预测连续的三角形表面，实现了显式的流形几何表达。
◆ 引入了相对三维点云监督，增强了几何学习的稳定性和一致性，提升了重建质量。
◆ 所生成的模型与标准图形及机器人仿真器无缝兼容，克服了现有方法（如NeRF、3DGS）在直接仿真应用上的局限性。
◆ 在保持高效重建的同时，兼顾了渲染质量与几何的显式性，为实时部署提供了新方案。</td></tr>
<tr><td>2026-03-06</td><td>CylinderSplat: 3D Gaussian Splatting with Cylindrical Triplanes for Panoramic Novel View Synthesis</td><td>[2603.05882](http://arxiv.org/pdf/2603.05882)</td><td>本文提出CylinderSplat，一个用于全景新颖视图合成的前馈式3D高斯溅射框架。其核心贡献与创新点如下：
◆ 提出了一种全新的圆柱形三平面表示，它比传统的笛卡尔三平面更贴合全景数据的几何特性，并符合曼哈顿世界假设，从而有效减少了全景场景重建中的失真和混叠问题。
◆ 设计了一个双分支架构，结合了基于像素的分支和基于体素的分支，前者能精细重建观测良好的区域，后者则利用圆柱形三平面来补全被遮挡或稀疏观测的区域。
◆ 该框架能够灵活处理从单张到多张不固定数量的全景图输入，提升了在稀疏视图场景下的应用能力。
◆ 通过广泛的实验验证，该方法在单视图和多视图全景新颖视图合成任务上均取得了最先进的性能，在重建质量和几何精度上超越了先前方法。</td></tr>
<tr><td>2026-03-05</td><td>SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction</td><td>[2603.05152](http://arxiv.org/pdf/2603.05152)</td><td>该论文SSR-GS的核心贡献是提出了一种在3D高斯泼溅框架中分离镜面反射的方法，以提升复杂光照下光泽表面的重建质量。其创新点主要包括：

◆ 引入预滤波Mip-Cubemap来高效建模直接的镜面反射。
◆ 提出IndiASG模块，专门用于捕捉间接的镜面反射（如多表面相互反射）。
◆ 设计视觉几何先验（VGP），通过反射分数（RS）耦合反射感知的视觉先验，降低以反射主导区域的光度损失权重。
◆ 在VGP中整合来自VGGT的几何先验，包括渐进衰减的深度监督和变换法向约束，以提升几何准确性。
通过合成与真实场景的实验验证，该方法在光泽表面重建任务上达到了最先进的性能。</td></tr>
<tr><td>2026-03-05</td><td>GaussTwin: Unified Simulation and Correction with Gaussian Splatting for Robotic Digital Twins</td><td>[2603.05108](http://arxiv.org/pdf/2603.05108)</td><td>该论文提出了GaussTwin，一个用于机器人操作的实时数字孪生系统，其核心贡献与创新点如下：

◆ 提出了一个统一模型，将基于位置的动力学与离散Cosserat杆模型相结合，实现了物理基础扎实的仿真，能有效处理复杂动态交互。
◆ 创新性地使用高斯泼溅技术进行高效渲染和视觉校正，并通过将高斯图元锚定到物理基元上，实现了仿真与感知的紧密耦合。
◆ 设计了一种由光度误差和分割掩码驱动的SE(3)一致性更新机制，在保持物理保真度的同时，实现了稳定的预测-校正循环，有效弥合了真实与仿真的差距。
◆ 整个系统支持实时运行，并通过在仿真和真实Franka机器人平台上的实验验证，其跟踪精度和鲁棒性均优于基于形状匹配和纯刚性模型的基线方法。
◆ 该系统展示了在下游任务（如基于推动的规划）中的应用潜力，推动了能够支持闭环机器人交互与学习的、具有统一物理意义的数字孪生发展。</td></tr>
<tr><td>2026-03-05</td><td>GloSplat: Joint Pose-Appearance Optimization for Faster and More Accurate 3D Reconstruction</td><td>[2603.04847](http://arxiv.org/pdf/2603.04847)</td><td>本文核心贡献是提出了GloSplat框架，通过联合优化相机位姿与外观来实现更快、更准的3D高斯溅射重建。其创新点如下：

◆ 提出了联合位姿-外观优化框架，将特征匹配、运动恢复结构和新视图合成这些传统上分离的任务统一到3D高斯溅射训练过程中进行联合优化。

◆ 在优化中保留了显式的运动恢复结构特征轨迹作为一等实体，使三维轨迹点作为独立于高斯基元的可优化参数持续存在，为几何约束提供了锚点。

◆ 设计了结合重投影损失与光度监督的混合监督机制，既利用光度梯度进行外观细化，又通过几何约束防止早期位姿漂移，实现了更精细的优化。

◆ 提出了两种实用流程变体：GloSplat-F无需传统运动恢复结构系统，依靠检索式图像对选择实现高效重建；GloSplat-A则采用详尽匹配以追求最高质量。

◆ 实验证明该框架性能领先：GloSplat-F在无需运动恢复结构的方法中达到最优，而GloSplat-A甚至超越了所有依赖传统运动恢复结构的基线方法。</td></tr>
<tr><td>2026-03-05</td><td>DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction</td><td>[2603.04770](http://arxiv.org/pdf/2603.04770)</td><td>本文提出了首个用于动态稀疏视角DSA重建的超分辨率高斯泼溅框架DSA-SRGS，其核心贡献与创新点如下：

◆ 首次将超分辨率能力引入动态稀疏视角DSA的4D重建中，克服了现有方法因输入投影分辨率限制而无法恢复细微血管细节的问题。
◆ 设计了多保真度纹理学习模块，通过整合一个微调后的DSA专用超分辨率模型所提供的高质量先验，来优化4D重建过程。
◆ 在该模块中采用了置信度感知策略，能自适应地权衡原始低分辨率投影与生成的高分辨率伪标签之间的监督信号，有效缓解了伪标签可能带来的伪影。
◆ 提出了辐射亚像素致密化策略，这是一种自适应方法，利用高分辨率亚像素采样带来的梯度累积来精细化4D辐射高斯核。
◆ 在两个临床DSA数据集上的实验表明，该方法在定量指标和定性视觉保真度上均显著优于现有先进技术。</td></tr>
<tr><td>2026-03-04</td><td>LLM-supported 3D Modeling Tool for Radio Radiance Field Reconstruction</td><td>[2603.04368](http://arxiv.org/pdf/2603.04368)</td><td>本文核心贡献是开发了一个本地可部署的、基于大语言模型的工具，旨在极大简化用于无线电信道建模的无线电辐射场重建所需的3D环境创建过程。

◆ 提出并实现了一个集成化工具，将微调的语言模型、生成式3D建模框架与Blender软件相结合，通过直观的聊天对话方式即可设计3D场景，显著降低了专业3D建模的技术门槛。
◆ 针对系统不同环节优化了模型选型与集成：使用微调的T5-mini解析用户指令，采用all-MiniLM-L6-v2进行本地物体库的语义检索，利用LLaMA-Mesh和Shap-E分别实现快速网格创建和高质量模型生成。
◆ 开发了定制的Blender导出插件，确保了由该工具生成的3D模型能够与前沿的RF-3DGS无线电辐射场重建流程完全兼容，形成了从场景描述到信道参数渲染的完整工作流。
◆ 通过为NIST大厅和UW-Madison无线实验室构建3D模型并进行RRF重建的实例，验证了该工具的有效性，证明了其能提升无线电辐射场技术在无线研究和频谱规划中的实用性和可访问性。</td></tr>
<tr><td>2026-03-04</td><td>EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding</td><td>[2603.04254](http://arxiv.org/pdf/2603.04254)</td><td>EmbodiedSplat的核心贡献是提出了一种在线、前馈式的开放词汇语义3D高斯泼溅方法，用于在智能体探索过程中近乎实时地完成三维场景的重建与理解。其创新点主要包括：

◆ 实现了从超过300张流式图像中进行在线、前馈的语义三维重建与理解，突破了现有开放词汇3DGS方法通常局限于离线或逐场景优化的限制。

◆ 提出了结合CLIP全局码本的在线稀疏系数场，将2D CLIP语义嵌入绑定到每个3D高斯上，在最小化内存消耗的同时，保持了CLIP模型完整的语义泛化能力。

◆ 通过3D U-Net聚合3D高斯点云来生成具有三维几何感知的CLIP特征，为面向2D的语言嵌入补偿了重要的三维几何先验信息。

◆ 该方法高度泛化于新场景，当与实时2D模型结合时，支持近乎实时的三维语义重建，在多个室内数据集上验证了其有效性与效率。</td></tr>
<tr><td>2026-03-03</td><td>VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats</td><td>[2603.02986](http://arxiv.org/pdf/2603.02986)</td><td>该论文的核心贡献是提出了VIRGi方法，实现了对3D高斯泼溅（3DGS）建模场景的快速、逼真且保留视角依赖效果的颜色编辑。

◆ 提出了一种新颖的架构，将颜色分解为漫反射和视角依赖（如高光）两个独立组件，从而在编辑颜色时能有效保留场景原有的逼真镜面反射等效果。
◆ 设计了一种多视图训练策略，整合来自多个视角的图像块进行训练，相比传统的单视图批次训练，能获得更准确的3DGS重建结果，为重新着色任务奠定了更优的表示基础。
◆ 引入了一套高效的交互式编辑方案，用户仅需提供一张手动编辑的参考图像，系统便能通过一个单次前向传播的模块分割可编辑区域，并仅微调一个多层感知机（MLP）的权重。
◆ 该方案能在短短两秒内将颜色编辑无缝传播至整个3D场景，支持实时交互，并允许用户控制视角依赖效果的强度，在效率与实用性上取得显著进步。</td></tr>
<tr><td>2026-03-03</td><td>Articulation in Motion: Prior-free Part Mobility Analysis for Articulated Objects By Dynamic-Static Disentanglement</td><td>[2603.02910](http://arxiv.org/pdf/2603.02910)</td><td>该论文提出了一种名为“运动中的铰接”的新框架，用于从单段交互视频和初始静态扫描中，无需先验知识地分析铰接物体的部件分割与运动学。

◆ 核心创新在于提出了一个双高斯场景表示，它从初始3D高斯溅射扫描和部件运动视频中学习，利用运动线索实现部件分割与关节分配，无需已知部件数量等先验假设。

◆ 方法采用了一种鲁棒的顺序RANSAC算法，无需任何部件级结构先验，即可将运动的基元聚类为刚性部件并估计运动学，同时自动确定部件的数量。

◆ 该框架成功实现了从动态交互观测中解耦静态与动态信息，能够同时输出高质量的部件分割、关节运动学分析以及可交互的3D数字模型重建。

◆ 与需要多状态清晰观测和已知部件数量的先前方法相比，本方法在复杂性和适用性上具有显著优势，在简单和复杂物体上的实验均验证了其有效性与强大的泛化能力。</td></tr>
<tr><td>2026-03-03</td><td>Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting</td><td>[2603.02893](http://arxiv.org/pdf/2603.02893)</td><td>该论文的核心贡献是提出MVD-HuGaS方法，实现了从单张图像进行高质量自由视角三维人体渲染。其创新点主要包括：

◆ 提出一种增强的多视角扩散模型，该模型在高质量三维人体数据集上精调，能够从单张参考图像生成具有三维几何和人体结构先验的多视角图像。

◆ 引入一个对齐模块，对稀疏生成的多视角图像进行相机姿态估计，并联合优化三维高斯模型与相机姿态，解决了姿态估计不准导致的伪影问题。

◆ 设计了一个基于深度的面部失真缓解模块，专门对生成图像的面部区域进行细化，显著提升了重建结果中关键面部区域的真实感和保真度。

◆ 整体框架利用优化后的多视角图像及其精确相机姿态，高效优化目标人体的三维高斯表征，最终实现高保真的自由视角渲染。实验证明该方法在多个数据集上达到了领先性能。</td></tr>
<tr><td>2026-03-04</td><td>Generalized non-exponential Gaussian splatting</td><td>[2603.02887](http://arxiv.org/pdf/2603.02887)</td><td>本文提出了一种广义非指数高斯泼溅方法，对3D高斯泼溅技术进行了重要扩展。其核心贡献在于将传统的渲染模型推广到更广泛的物理基础混合算子家族中。

◆ 将3DGS的图像形成模型从经典的指数透射率推广到非指数体系，突破了原有理论框架。
◆ 基于二次透射率函数，定义了亚线性、线性和超线性三种非指数3DGS变体，实现了比指数衰减更快的衰减特性。
◆ 新方法在保持与原始3DGS相近渲染质量的同时，显著减少了渲染过程中的过度绘制现象。
◆ 在基于光线追踪的渲染器中，该方法在复杂的真实场景捕获数据上实现了高达4倍的渲染速度提升，大幅提高了效率。</td></tr>
<tr><td>2026-03-03</td><td>Multimodal-Prior-Guided Importance Sampling for Hierarchical Gaussian Splatting in Sparse-View Novel View Synthesis</td><td>[2603.02866](http://arxiv.org/pdf/2603.02866)</td><td>本文提出了一种用于稀疏视角新视图合成的多模态先验引导重要性采样方法，其核心贡献与创新点如下：

◆ 提出了多模态先验引导的重要性采样机制，作为分层3D高斯泼溅的核心。它融合了渲染残差、语义先验和几何先验三种互补线索，生成鲁棒的局部可恢复性估计，以精准指导细节增强。

◆ 构建了一个由粗到精的高斯表示框架。该框架包含一个稳定的粗糙层用于编码全局形状，并依据多模态度量指示，仅在可恢复细节的区域选择性地添加精细高斯基元。

◆ 设计了一种几何感知的采样与保留策略。该策略将优化资源集中于几何关键和复杂区域，同时保护欠约束区域中新添加的基元免遭过早剪枝，从而提升了重建的稳定性与完整性。

◆ 通过优先考虑多模态证据一致支持的区域，而非单纯依赖渲染残差，该方法有效缓解了由过拟合纹理引起的误差，并抑制了位姿与外观不一致所产生的噪声。

◆ 在多个稀疏视角基准测试上的实验表明，该方法实现了最先进的重建质量，例如在DTU数据集上PSNR指标提升最高达0.3 dB。</td></tr>
<tr><td>2026-03-03</td><td>R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild</td><td>[2603.02801](http://arxiv.org/pdf/2603.02801)</td><td>该论文提出了R3GW方法，旨在解决野外捕获的户外场景的逼真重光照问题。其核心贡献与创新点如下：

◆ 首创了适用于野外无约束照片集的、可重光照的3D高斯溅射（3DGS）表示方法，将场景解耦为可重光照的前景和非反射的背景（天空）。
◆ 采用两组独立的高斯集合分别建模前景与天空，有效处理了复杂户外场景中动态光照的挑战。
◆ 创新地将基于物理的渲染（PBR）与3DGS表示相结合，在变化光照条件下建模了前景中视角依赖的照明效果（如反射）。
◆ 提出的天空表示方法缓解了深度重建伪影，显著提升了天空与前景边界的渲染质量。
◆ 在NeRF-OSR数据集上的定量与定性评估表明，该方法在实现最先进性能的同时，支持对无约束场景进行基于物理的逼真重光照，并能合成任意光照条件下的新视图。</td></tr>
<tr><td>2026-03-03</td><td>SemGS: Feed-Forward Semantic 3D Gaussian Splatting from Sparse Views for Generalizable Scene Understanding</td><td>[2603.02548](http://arxiv.org/pdf/2603.02548)</td><td>该论文提出SemGS，一种从稀疏图像进行通用语义场景理解的创新框架。其核心贡献与创新点如下：

◆ 提出前馈式框架，仅需稀疏视角图像即可一次性完成三维语义场重建，无需针对每个场景进行优化，提升了实用性与扩展性。
◆ 设计双分支架构提取颜色与语义特征，并共享浅层CNN，使语义推理能利用颜色外观中的纹理与结构线索。
◆ 在特征提取器中引入相机感知注意力机制，显式建模不同相机视角间的几何关系，增强了多视图一致性。
◆ 采用共享几何一致性的双高斯表示，分别解码颜色与语义属性，并通过光栅化实现新视角下的语义地图合成。
◆ 提出区域平滑损失函数，以增强语义预测的空间连贯性，提升语义场的整体一致性。

该框架在多个数据集上实现了先进的性能，并展现出快速的推理速度与强大的跨场景泛化能力。</td></tr>
<tr><td>2026-03-03</td><td>OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution</td><td>[2603.02134](http://arxiv.org/pdf/2603.02134)</td><td>该论文提出了OnlineX，一个用于在线场景的3D重建与理解的前馈框架。其核心贡献在于解决了在线连续重建中的累积漂移难题，并实现了外观与语义的联合建模。

◆ 创新性地提出了在线3D重建与理解的统一前馈框架，仅需流式图像输入即可实时重建3D外观和语言语义场，突破了现有方法局限于离线重建的范式。
◆ 针对在线重建中记忆状态既要主动更新局部细节又要保持全局稳定的根本矛盾，提出了解耦的“主动-稳定”状态演化范式。该范式将记忆状态分离为专司高频几何捕获的主动状态和负责长期结构保持的稳定状态。
◆ 设计了将主动状态信息协同融合至稳定状态的机制，从而在保证重建细节逼真度的同时，有效抑制了累积漂移，实现了稳定性与高保真的统一。
◆ 通过联合建模视觉外观与语言场，并引入隐式高斯融合模块，显著提升了重建质量与语义理解能力。实验证明，该方法在新视图合成和语义理解任务上均优于现有方法，且能适应不同长度的输入序列并保持实时推理速度。</td></tr>
<tr><td>2026-03-02</td><td>LiftAvatar: Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation</td><td>[2603.02129](http://arxiv.org/pdf/2603.02129)</td><td>本文提出LiftAvatar，一种通过运动学空间补全来驱动高保真三维虚拟人动画的新范式。其核心贡献在于将单目视频中稀疏的表情与姿态信号提升为丰富的运动学表示，从而显著增强下游三维虚拟人的重建与动画效果。

◆ 提出运动学空间补全新范式，将稀疏的单目观测数据（如表情、头部姿态）补全为完整、连贯的序列，以驱动高质量三维高斯溅射虚拟人。
◆ 设计了一个细粒度、可表情控制的大规模视频扩散Transformer模型，能够基于单张或多张参考图像，合成高质量、时序一致的表情序列。
◆ 引入多粒度表情控制方案，结合着色图与表情系数，实现精准稳定的虚拟人驱动。
◆ 开发多参考帧条件机制，聚合多帧图像的互补信息，增强三维一致性与可控性。
◆ 作为即插即用增强模块，有效解决了日常单目视频因运动学线索稀疏导致的虚拟人表现力不足与重建瑕疵问题，并能将大规模视频生成模型的先验知识蒸馏到三维流程中，大幅提升现有方法的动画质量与量化指标。</td></tr>
<tr><td>2026-03-02</td><td>Sparse View Distractor-Free Gaussian Splatting</td><td>[2603.01603](http://arxiv.org/pdf/2603.01603)</td><td>该论文针对稀疏视角下动态干扰物去除的3D高斯溅射（3DGS）性能下降问题，提出一个融合多源先验信息的增强框架。其核心贡献与创新点如下：

◆ 首次系统解决了稀疏输入条件下动态干扰物去除3DGS训练性能显著下降的难题，填补了该领域的研究空白。

◆ 创新性地引入几何基础模型VGGT，利用其估计相机参数并生成密集初始3D点云，为稀疏条件下的重建提供了稳健的几何初始化。

◆ 提出利用VGGT的注意力图进行高效准确的语义实体匹配，有效关联不同视角下的相同静态场景元素，克服了传统颜色残差启发式方法在稀疏观测下不可靠的缺陷。

◆ 集成视觉语言模型（VLM）来识别并保护场景中的大范围静态区域，增强了模型对场景静态结构的理解与保持能力。

◆ 设计了一个模块化框架，能够将上述几何、语义及语言先验无缝集成到现有动态干扰物去除3DGS方法中，提升了方法的通用性与实用性。</td></tr>
<tr><td>2026-03-02</td><td>Radiometrically Consistent Gaussian Surfels for Inverse Rendering</td><td>[2603.01491](http://arxiv.org/pdf/2603.01491)</td><td>该论文的核心贡献是提出了一个名为RadioGS的逆向渲染框架，旨在解决高斯溅射方法在复杂全局光照下难以准确解耦材质属性的难题。

◆ 创新性地提出了辐射度量一致性这一物理约束，通过最小化高斯基元学习到的辐射度与其基于物理渲染结果之间的残差，为未观测视角提供了关键监督。
◆ 该约束建立了一个自校正反馈循环，结合了基于物理的渲染和新视角合成的监督，从而实现了对间接光照（如互反射）的精确建模。
◆ 提出了Radiometrically Consistent Gaussian Surfels框架，通过使用高斯面元与二维高斯光线追踪技术，高效地集成了辐射度量一致性原则。
◆ 提出了一种基于微调的快速重光照策略，能在数分钟内使高斯面元辐射度适应新的光照条件，同时保持极低的渲染成本（&lt;10毫秒）。
实验表明，RadioGS在逆向渲染任务上超越了现有基于高斯的方法，并保持了计算高效性。</td></tr>
<tr><td>2026-03-01</td><td>FLICKER: A Fine-Grained Contribution-Aware Accelerator for Real-Time 3D Gaussian Splatting</td><td>[2603.01158](http://arxiv.org/pdf/2603.01158)</td><td>该论文的核心贡献是提出了FLICKER，一个基于软硬件协同设计的贡献感知3D高斯溅射加速器，旨在解决3DGS在边缘设备上因处理大量无效高斯点而导致的效率低下问题。其创新点主要包括：

◆ 采用硬件-软件协同设计框架，实现了近像素级的贡献驱动渲染，显著降低了计算开销。
◆ 引入了自适应引导像素技术，优化了渲染流程的起点选择，提升了处理效率。
◆ 提出了像素矩形分组方法，有效组织像素数据，减少了测试和渲染的复杂度。
◆ 设计了分层高斯测试机制，通过层次化筛选避免了大量非贡献高斯点的冗余计算。
◆ 构建了混合精度架构，在保证渲染质量的同时，进一步提高了计算速度和能效。
实验结果表明，该加速器在速度、能效和面积上均优于现有先进方案，为3DGS在AR/VR边缘设备上的实时部署提供了高效解决方案。</td></tr>
<tr><td>2026-03-01</td><td>D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping</td><td>[2603.01151](http://arxiv.org/pdf/2603.01151)</td><td>该论文的核心贡献是提出了一个名为D-REX的可微分“真实-仿真-真实”引擎，用于缩小机器人灵巧抓取的仿真与现实差距。其核心创新点如下：

◆ 提出了一个创新的可微分“真实-仿真-真实”流程框架，利用高斯溅射表示法构建可微分引擎，实现了从真实世界视觉观察和机器人控制信号中直接识别物体质量。
◆ 通过优化被操作物体的质量参数，能够自动构建高保真且物理合理的数字孪生，为仿真提供更准确的物理模型。
◆ 引入了一种新颖的策略学习方法，能够将有限的人类可行演示转化为模拟机器人演示，从而利用少量数据训练出具有力感知的抓取策略。
◆ 该引擎实现了质量识别与抓取策略学习的同步进行，二者相互促进：优化的质量参数提升了策略学习的仿真真实性，而策略交互数据又辅助了质量识别。
◆ 实验证明，该方法在各种物体几何形状和质量值下都能实现准确、稳健的质量识别，并最终学习到高性能的抓取策略，有效缩小了仿真到现实的差距。</td></tr>
<tr><td>2026-03-01</td><td>HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views</td><td>[2603.01099](http://arxiv.org/pdf/2603.01099)</td><td>该论文针对稀疏视角下3D高斯泼溅（3DGS）重建质量下降的问题，提出了HeroGS框架，其核心贡献在于构建了一个分层引导的优化体系，显著提升了稀疏视角下的渲染保真度与几何一致性。

◆ 提出了一个统一的分层引导框架，在图像、特征和参数三个层级上系统性地约束和优化高斯分布。
◆ 在图像层级，将稀疏监督转化为伪密集引导，全局规整高斯分布，为后续优化奠定一致基础。
◆ 在特征层级，设计了特征自适应致密化与剪枝（FADP）方法，利用低级特征修复高频细节，并自适应地在背景区域致密化高斯。
◆ 在参数层级，提出了协同剪枝的几何一致性（CPG）策略，通过参数冻结与协同剪枝来引导几何一致性，有效移除不一致的泼溅。
◆ 通过上述分层协同优化，整体框架有效解决了稀疏视角下高斯分布不规则、背景模糊和细节扭曲等问题，在多个实验中超越了现有先进方法。</td></tr>
<tr><td>2026-03-01</td><td>Decoupling Motion and Geometry in 4D Gaussian Splatting</td><td>[2603.00952](http://arxiv.org/pdf/2603.00952)</td><td>该论文的核心贡献是提出了一种名为VeGaS的新型4D高斯溅射框架，旨在解决动态场景重建中运动与几何耦合导致的表达受限和伪影问题。其核心创新点如下：

◆ 提出了速度驱动的4D高斯溅射框架VeGaS，首次将高斯点的运动属性与几何属性进行解耦，突破了原有方法中两者在单一协方差公式中耦合的限制。

◆ 引入了伽利略剪切矩阵，该矩阵显式地结合了时变速度，能够灵活且精确地建模复杂的非线性运动，同时严格确保运动效应不影响几何相关的高斯条件协方差。

◆ 设计了一个几何形变网络，该网络利用时空上下文信息和速度线索来优化高斯点的形状与方向，从而增强了动态场景中几何结构的时序建模能力。

◆ 通过在公开数据集上进行大量实验，验证了该方法的有效性，并取得了当前最优的性能表现，显著提升了动态场景重建的视觉质量。</td></tr>
<tr><td>2026-02-28</td><td>TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward Pose-free Reconstruction</td><td>[2603.00697](http://arxiv.org/pdf/2603.00697)</td><td>本文提出TokenSplat，一种前馈式框架，用于从无相机姿态的多视角图像中联合进行三维高斯重建与相机姿态估计。其核心贡献与创新点如下：

◆ 提出了令牌对齐的高斯预测模块，直接在特征空间中对齐多视角间的语义对应信息，实现了长距离的跨视角推理。

◆ 利用粗略的令牌位置和融合置信度作为指导，聚合多尺度上下文特征，有效减少了重叠高斯带来的冗余。

◆ 引入了可学习的相机令牌和一个非对称双流解码器，该解码器强制相机令牌与图像令牌进行方向性约束的通信。

◆ 上述设计将视角线索与场景语义解耦，增强了姿态估计的鲁棒性，并在前馈架构中保持了清晰的分解。

◆ 整个系统无需迭代优化即可实现连贯的三维重建和稳定的相机姿态估计，在无姿态设置下取得了更高的重建保真度、新视图合成质量以及更准确的姿态估计结果。</td></tr>
<tr><td>2026-02-28</td><td>Zero-Shot Robotic Manipulation via 3D Gaussian Splatting-Enhanced Multimodal Retrieval-Augmented Generation</td><td>[2603.00500](http://arxiv.org/pdf/2603.00500)</td><td>该论文提出RobMRAG框架，旨在解决机器人零样本操作中几何与空间理解不足的难题。其核心贡献与创新点如下：

◆ 提出一个结合3D高斯泼溅增强的多模态检索增强生成框架，用于零样本机器人操作，弥合高层语义推理与底层几何执行的鸿沟。

◆ 构建了一个多源操作知识库，包含物体接触帧、任务完成帧与位姿参数，为检索提供丰富的多模态先验知识。

◆ 设计了分层多模态检索模块，采用三级优先级混合检索策略，先找任务相关物体原型，再基于像素级相似度与实例匹配距离筛选几何最接近的参考示例。

◆ 创新性地将基于3D高斯泼溅的3D感知位姿优化模块引入框架，能在三维空间中对齐参考物体与目标物体的位姿，提升几何准确性。

◆ 通过将三维对齐结果重投影至图像平面并输入多模态大语言模型，增强了最终位姿参数的生成质量，在30类家庭物体的测试集上显著提升了零样本操作的成功率。</td></tr>
<tr><td>2026-02-27</td><td>UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images</td><td>[2602.24290](http://arxiv.org/pdf/2602.24290)</td><td>本文提出UFO-4D，一个从两张无位姿图像进行密集4D重建的前馈统一框架。其核心贡献与创新点如下：

◆ 提出首个从前馈模型实现统一、密集4D重建的方法，仅需一对无位姿图像，无需耗时的测试时优化。
◆ 引入动态3D高斯溅射作为显式4D表示，以前馈方式直接联合估计一致的3D几何、3D运动和相机位姿。
◆ 核心洞见在于，从单一动态3D高斯表示可微分渲染多种信号，这带来了关键训练优势：实现了自监督的图像合成损失，并紧密耦合了外观、深度与运动。
◆ 由于所有模态共享相同的几何基元，监督其中一项会自然地正则化并提升其他项，这种协同效应有效克服了数据稀缺问题。
◆ 该方法在几何、运动和相机位姿的联合估计上性能显著，超越先前方法达3倍，并能实现跨新视角和时间的高保真4D插值。</td></tr>
<tr><td>2026-02-27</td><td>Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives</td><td>[2602.24136](http://arxiv.org/pdf/2602.24136)</td><td>该论文针对3D高斯溅射（3DGS）模型存在大量冗余、资源消耗高的问题，提出了一个旨在实现紧凑高效3D表示的综合解决方案。其核心贡献与创新点如下：

◆ 提出了一种高效、集成的重建感知自适应剪枝策略。该策略能根据场景重建质量动态决定剪枝时机和优化间隔，在削减模型规模的同时，反而提升了渲染质量。

◆ 引入了一种新型的3D高斯差分（DoG）基元。该基元在一个单独的基元内联合建模正负密度，从而显著增强了高斯基元在紧凑配置下的表达能力。

◆ 通过上述两项创新技术的结合，该方法在实现模型高度紧凑化（高斯数量减少高达90%）的同时，保持了与先进方法相当甚至更优的视觉渲染质量，有效平衡了效率与效果。</td></tr>
<tr><td>2026-02-27</td><td>DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer</td><td>[2602.24096](http://arxiv.org/pdf/2602.24096)</td><td>该论文的核心贡献是提出了DiffusionHarmonizer，一个在线生成式增强框架，旨在解决神经重建场景（如NeRF、3DGS）用于机器人仿真时存在的视觉缺陷与真实感不足问题。

◆ 提出了一个在线生成增强框架，能够将不完美的神经渲染输出，实时转化为时序一致且高真实感的图像，直接服务于在线仿真器。
◆ 其核心是一个单步、时序条件增强器，由预训练的多步图像扩散模型转换而来，实现了在单GPU上的高效在线运行。
◆ 设计了一套定制化的数据筛选与构建流程，能自动生成用于训练的“合成-真实”图像对，重点针对外观融合、伪影校正和光照真实感进行优化。
◆ 最终构建了一个可扩展的系统，显著提升了从现实数据构建的仿真场景的视觉保真度，特别是在新颖视角渲染和动态物体逼真融合方面。
◆ 该工作有效弥合了自动化神经重建与高真实感仿真之间的鸿沟，为自动驾驶等领域的研发与测试提供了更高质量的仿真环境。</td></tr>
<tr><td>2026-02-27</td><td>SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting</td><td>[2602.24020](http://arxiv.org/pdf/2602.24020)</td><td>该论文的核心贡献是提出了SR3R框架，重新思考并革新了基于高斯泼溅的3D超分辨率重建范式，将其从依赖密集输入和逐场景优化的传统方法，转变为一种前馈式、可泛化的直接映射学习。

◆ 提出了一种全新的前馈式框架（SR3R），能够直接从稀疏的低分辨率多视角图像，通过一个学习到的映射网络，一次性预测出高分辨率的3D高斯泼溅表示，实现了高效的单次前向推理。

◆ 从根本上改变了3D超分辨率获取高频知识的途径：让模型能够从大规模多场景数据中自主学习3D特有的高频几何与外观信息，而非依赖从预训练2D超分辨率模型继承的有限先验。

◆ 引入了高斯偏移学习和特征细化等创新模块，以稳定重建过程并锐化高频细节，从而显著提升了重建的保真度。

◆ 该框架具有即插即用和强泛化能力：它可以与任何前馈式3DGS重建主干网络搭配使用，将主干网络提供的低分辨率3DGS“支架”升级为高分辨率版本，并在未见过的场景上实现了强大的零样本泛化性能，甚至超越了针对特定场景进行优化的先进方法。</td></tr>
<tr><td>2026-02-27</td><td>No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency</td><td>[2602.23559](http://arxiv.org/pdf/2602.23559)</td><td>该论文首次系统研究了跨不同传感器模态的视图合成问题，针对一个实际却长期被忽视的难题：获取对齐的RGB-X（如热成像、深度等）数据。现有工作通常假设此类配对数据已存在并专注于模态融合，但实际校准需耗费巨大工程成本。

◆ 核心创新在于提出“匹配-稠密化-巩固”方法，无需对X传感器进行任何3D先验标定，仅需对RGB图像使用近乎零成本的COLMAP进行稀疏重建。
◆ 通过置信度感知的稠密化与自匹配过滤技术，有效提升了跨模态视图合成的质量与鲁棒性。
◆ 最终将优化后的结果整合至3D高斯泼溅（3DGS）框架中，确保了三维空间的一致性。
◆ 该方法旨在消除各类RGB-X传感器间繁琐的校准步骤，为大规模真实世界跨传感器数据采集提供可扩展的解决方案，从而推动跨传感器学习的发展。</td></tr>
<tr><td>2026-02-26</td><td>Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking</td><td>[2602.23172](http://arxiv.org/pdf/2602.23172)</td><td>该论文提出了LaGS方法，用于实现动态场景的4D全景占据栅格跟踪。其核心贡献在于以端到端方式统一了动态目标的时序跟踪与精细的3D结构感知。主要创新点如下：

◆ 提出了一种新颖的潜在高斯泼溅方法，高效地将多视角图像信息聚合到3D体素栅格中，解决了多视图信息融合的关键挑战。
◆ 首先将观测融合为3D高斯分布，以此作为场景的稀疏点云中心潜在表示，实现了场景的紧凑编码。
◆ 随后将聚合的特征泼溅到3D体素网格上，并通过一个基于掩码的分割头进行解码，从而同时输出几何、语义和实例ID信息。
◆ 该方法集成了基于摄像头的端到端跟踪与基于掩码的多视角全景占据预测，实现了对时空场景的整体性理解。
◆ 在Occ3D nuScenes和Waymo数据集上取得了领先的4D全景占据跟踪性能，验证了其有效性。</td></tr>
<tr><td>2026-02-26</td><td>PackUV: Packed Gaussian UV Maps for 4D Volumetric Video</td><td>[2602.23040](http://arxiv.org/pdf/2602.23040)</td><td>该论文的核心贡献是提出了一种名为PackUV的新型4D高斯表示方法，旨在解决大规模4D体视频重建、存储与传输的难题。其核心创新点如下：

◆ 提出Packed Gaussian UV Maps，将4D高斯的所有属性映射到一系列结构化的多尺度UV图集中，实现了紧凑且与图像原生兼容的存储格式。
◆ 开发了PackUV-GS拟合方法，直接在UV域优化高斯参数，确保了长时间序列的时间一致性，并能处理大运动和遮挡解除。
◆ 设计了流引导的高斯标记与视频关键帧模块，能有效识别动态高斯、稳定静态区域，从而在复杂场景下保持时序连贯性。
◆ 其UV图集格式首次实现了与标准视频编解码器的完全兼容，无需质量损失即可利用现有基础设施进行高效流式传输，推动了实用化。
◆ 发布了迄今最大的多视角视频数据集PackUV-2B，包含100个序列和20亿帧，为长时序体视频研究提供了重要基准。实验证明该方法在渲染质量上超越现有基线，并能将高质量重建扩展到长达30分钟的序列。</td></tr>
<tr><td>2026-02-26</td><td>GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation</td><td>[2602.22800](http://arxiv.org/pdf/2602.22800)</td><td>该论文提出了一种名为GSTurb的新框架，用于抑制大气湍流引起的图像退化。其核心贡献与创新点如下：

◆ 首次将3D高斯泼溅技术引入大气湍流图像复原领域，用以建模湍流引起的非等晕性模糊。
◆ 提出一个集成化框架，结合了光流引导的像素位移校正与高斯泼溅的模糊建模，统一优化处理倾斜和模糊两种退化。
◆ 利用高斯参数来共同表示湍流的倾斜和模糊效应，并通过多帧序列进行联合优化，从而提升复原质量。
◆ 在合成数据集ATSyn-static上取得了当前最佳性能，PSNR和SSIM分别显著提升了1.3 dB和0.048。
◆ 在多个真实世界数据集上的实验也验证了其优越性，表明该框架能有效应对合成与真实湍流条件。</td></tr>
<tr><td>2026-02-26</td><td>Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring</td><td>[2602.22731](http://arxiv.org/pdf/2602.22731)</td><td>该论文的核心贡献是提出了一套名为Sapling-NeRF的集成化系统，用于对森林中的幼树进行高精度、可重复且地理定位的三维重建与生态监测。

◆ 提出了一种新颖的三级表征框架，融合了GNSS、激光雷达SLAM和NeRF技术，首次实现了幼树三维模型在真实世界坐标系中的精确地理定位。
◆ 解决了现有隐式重建方法（如NeRF）无法恢复真实尺度和缺乏地理定位能力的核心缺陷，使其能应用于需要长期精确监测的生态学研究。
◆ 通过以对象为中心的NeRF密集重建，显著提升了幼树细薄枝条、密集叶片等精细结构特征的捕捉能力，其精度优于传统地面激光扫描（TLS）。
◆ 整套系统实现了对0.5米至2米高幼树茎干骨架和叶片分布等关键性状的现场定量化测量，为生态学家分析森林动态提供了更丰富的结构数据。
◆ 在牛津和芬兰的真实森林样地进行了实验验证，证明该系统能高精度捕获树高、分枝模式和叶木比等参数，支持可重复的长期生态监测。</td></tr>
<tr><td>2026-02-26</td><td>ArtPro: Self-Supervised Articulated Object Reconstruction with Adaptive Integration of Mobility Proposals</td><td>[2602.22666](http://arxiv.org/pdf/2602.22666)</td><td>该论文提出了一种名为ArtPro的自监督铰接物体重建新框架，其核心贡献在于通过自适应整合运动提议，显著提升了复杂多部件物体重建的鲁棒性和准确性。具体创新点如下：

◆ 提出了自适应整合运动提议的新框架，克服了现有方法对初始部件分割高度敏感的局限。
◆ 采用基于几何特征和运动先验的过分割初始化方法，生成具有合理运动假设的部件提议。
◆ 在优化过程中，通过分析空间邻域间的运动一致性，动态合并这些部件提议。
◆ 引入碰撞感知的运动剪枝机制，有效防止错误的运动学估计，避免优化陷入局部最优。
实验表明，该方法在合成与真实物体数据上均能实现鲁棒重建，在精度和稳定性上显著优于现有方法。</td></tr>
<tr><td>2026-02-26</td><td>BetterScene: 3D Scene Synthesis with Representation-Aligned Generative Model</td><td>[2602.22596](http://arxiv.org/pdf/2602.22596)</td><td>BetterScene的核心目标是利用极稀疏、无约束的照片，提升多样化真实场景的新视角合成质量。其核心贡献与创新点在于：

◆ 首次深入探究并利用了预训练扩散模型的潜在空间，而非仅微调UNet模块，从而更有效地解决细节不一致和伪影问题。

◆ 提出了时间等变性正则化方法，将其应用于SVD流程中的VAE模块，以增强跨视角的时序一致性。

◆ 引入了与视觉基础模型对齐的表征，同样作用于VAE模块，以更好地恢复符合真实世界视觉先验的细节。

◆ 设计了一个集成框架，结合前馈式3D高斯溅射模型来渲染特征，作为SVD增强器的输入，从而生成连续、无伪影且视图一致的新视角。

◆ 在具有挑战性的DL3DV-10K数据集上验证了其优越性，性能超越了现有先进方法。</td></tr>
<tr><td>2026-02-26</td><td>GIFSplat: Generative Prior-Guided Iterative Feed-Forward 3D Gaussian Splatting from Sparse Views</td><td>[2602.22571](http://arxiv.org/pdf/2602.22571)</td><td>该论文提出了一种名为GIFSplat的纯前馈迭代优化框架，用于从稀疏无位姿图像重建3D高斯溅射场景。其核心贡献与创新点如下：

◆ 提出迭代式前馈残差更新机制，通过少量仅前向传播的步骤，利用渲染证据逐步优化3D场景，在效率与质量间取得更好平衡。

◆ 设计了一种无需梯度回传的生成式先验蒸馏方法，将冻结的扩散模型先验转化为高斯级别的提示线索，避免了视图数量的持续增加。

◆ 实现了在保持前馈推理效率（秒级完成）的同时，进行基于生成先验的逐场景自适应，克服了现有方法引入生成先验后推理时间大幅增加的问题。

◆ 整个框架无需输入相机位姿，也无需任何测试时的梯度优化，简化了使用流程。

◆ 在多个数据集上性能显著优于现有前馈方法，PSNR提升最高达+2.1 dB。</td></tr>
<tr><td>2026-02-26</td><td>SwiftNDC: Fast Neural Depth Correction for High-Fidelity 3D Reconstruction</td><td>[2602.22565](http://arxiv.org/pdf/2602.22565)</td><td>SwiftNDC的核心贡献是提出一个快速通用的神经深度校正框架，以解决深度引导三维重建中的几何不一致问题，从而高效生成高保真结果。

◆ 提出了神经深度校正场，能够快速生成跨视角一致的精细化深度图，从根本上改善多视角深度估计的尺度漂移和不一致问题。
◆ 设计了一套包含反投影和重投影误差过滤的流程，从校正后的深度图生成干净、均匀分布的稠密点云，为下游任务提供了优质的几何初始化。
◆ 证明了这种可靠的几何初始化能大幅加速基于3D高斯泼溅的网格重建，仅需极少优化迭代即可获得高质量表面。
◆ 同时提升了3D高斯泼溅在新视角合成中的渲染质量，凸显了强几何初始化对渲染任务的普适益处。
◆ 在涵盖网格重建与新视角合成的五个数据集上进行了全面验证，一致展示了其在提升重建精度、渲染保真度及降低运行时间方面的有效性。</td></tr>
<tr><td>2026-02-25</td><td>AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction</td><td>[2602.22376](http://arxiv.org/pdf/2602.22376)</td><td>该论文提出AeroDGS，一个用于单目无人机视频的物理一致动态高斯溅射框架，旨在解决单视角、大范围空中场景中动态物体重建的深度模糊和运动估计不稳定问题。

其核心创新点如下：
◆ 提出了单目几何提升模块，能够从单一空中序列中重建出可靠的静态与动态几何结构，为动态估计提供了稳健基础。
◆ 设计了物理引导优化模块，引入了可微分的地面支撑、直立稳定性和轨迹平滑性先验，将模糊的图像线索转化为物理一致的运动估计。
◆ 构建了一个联合优化框架，能够同步细化静态背景与动态实体，确保几何稳定性和时间演化的连贯性。
◆ 建立并公开了一个涵盖不同高度与运动条件的真实世界无人机数据集，用于评估动态空中重建任务。
实验表明，该方法在合成与真实无人机场景中均优于现有技术，实现了动态空中环境下的卓越重建保真度。</td></tr>
<tr><td>2026-02-25</td><td>Interactive Augmented Reality-enabled Outdoor Scene Visualization For Enhanced Real-time Disaster Response</td><td>[2602.21874](http://arxiv.org/pdf/2602.21874)</td><td>本论文的核心贡献是开发了一个面向灾害响应的用户中心增强现实（AR）系统，旨在通过直观的可视化与交互提升实时决策与协调效率。

◆ 创新性地将新兴的3D高斯溅射（3DGS）技术用于户外场景的详细重建与AR可视化，在保证高细节度的同时维持了用户的情境感知并降低了认知负荷。
◆ 提出了一种轻量级的交互范式，结合了“微缩世界”（WIM）导航与可过滤的语义兴趣点（POIs），使用户能快速浏览、筛选关键信息，从而支持高效决策。
◆ 设计了一个支持重建数据流式更新的系统架构，确保可视化内容能随场景变化而动态更新，适应灾害响应的实时性要求。
◆ 通过以用户为中心的性能评估验证了系统的高可用性和高接受度，初步用户反馈证实该设计易于使用，并能有效支持实时协调与快速决策。</td></tr>
<tr><td>2026-02-25</td><td>Space-Time Forecasting of Dynamic Scenes with Motion-aware Gaussian Grouping</td><td>[2602.21668](http://arxiv.org/pdf/2602.21668)</td><td>该论文的核心贡献是提出了一个用于动态场景长期时空预测的新框架MoGaF。其核心创新点在于：

◆ 提出了基于4D高斯溅射表示的新框架MoGaF，用于解决从有限观测中预测动态场景的长期演化难题。
◆ 创新性地引入了运动感知的高斯分组机制，以及分组优化策略，从而能够分别处理刚性和非刚性区域，确保其运动在物理上保持一致。
◆ 通过上述结构化表示，构建了一个轻量级的预测模块，专门用于预测未来的运动，实现了真实且时间稳定的场景演化。
◆ 在合成与真实数据集上的实验表明，该方法在渲染质量、运动合理性和长期预测稳定性方面均优于现有基线方法。</td></tr>
<tr><td>2026-02-25</td><td>DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling</td><td>[2602.21644](http://arxiv.org/pdf/2602.21644)</td><td>DAGS-SLAM的核心贡献在于提出了一种面向移动部署、高效且鲁棒的动态场景3D高斯溅射SLAM系统。其创新点可总结如下：

◆ 引入了时空运动概率（MP）作为每个高斯点的状态，以轻量方式持续估计和更新其动态可能性，替代了传统依赖繁重光流或逐帧分割的方法。
◆ 设计了一个不确定性感知的调度器，仅在高斯点不确定性高时按需触发语义分割（如YOLO），大幅减少了计算开销，提升了系统实时性。
◆ 将轻量级实例语义先验与几何线索相融合，共同估计MP，增强了动态判断在光照挑战下的鲁棒性。
◆ 在前端将MP传播用于动态感知的特征点匹配选择，提升了跟踪鲁棒性；在后端通过MP引导的优化抑制动态伪影，改善了重建质量。
◆ 在公开动态RGB-D数据集上的实验表明，该系统在消费级GPU上实现了实时性能，在保持高精度跟踪与重建的同时，为移动部署提供了更优的速度-精度权衡。</td></tr>
<tr><td>2026-02-24</td><td>HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles</td><td>[2602.21333](http://arxiv.org/pdf/2602.21333)</td><td>该论文的核心贡献是提出了一个名为HorizonForge的统一框架，用于实现高保真且精确可控的驾驶场景编辑与生成。

◆ 提出了一种新颖的“高斯-网格”混合三维场景表示方法，将场景重建为可编辑的高斯泼溅与网格，实现了比现有三维表示方法更高的保真度和细粒度的三维操控能力。
◆ 引入了支持语言驱动的车辆插入功能，允许用户通过自然语言指令在场景中添加任意车辆。
◆ 设计了一种噪声感知的视频扩散渲染流程，该流程能有效保证编辑后视频在空间和时间上的一致性，并能通过单次前向传播生成多样化的场景变体，无需针对每条轨迹进行繁琐的优化。
◆ 构建了一个名为HorizonSuite的综合评估基准，涵盖了自车层面和智能体层面的多种编辑任务（如轨迹修改、物体操控），为标准化的性能评估提供了基础。
◆ 通过整合上述创新，该框架在逼真度和可控性上显著超越了现有最佳方法，获得了83.4%的用户偏好提升和25.19%的FID指标改进，为自动驾驶仿真建立了一个强大而简洁的新范式。</td></tr>
<tr><td>2026-02-24</td><td>BrepGaussian: CAD reconstruction from Multi-View Images with Gaussian Splatting</td><td>[2602.21105](http://arxiv.org/pdf/2602.21105)</td><td>该论文的核心贡献是提出了BrepGaussian框架，能够仅从多视角二维图像中直接重建出高质量的CAD边界表示模型。其创新点主要体现在方法框架和策略设计上。

◆ 首创了将高斯泼溅渲染技术应用于从图像中学习三维CAD边界表示的任务，通过可学习特征的高斯泼溅渲染器进行高效建模。
◆ 提出了一种两阶段学习框架，将几何重建与特征学习解耦，先捕捉整体几何与边缘，再细化面片特征，从而提升了重建的清晰度与一致性。
◆ 设计了一种特定的拟合策略，最终输出由修剪曲面、边和角组成的显式参数化B-rep模型，而非传统的点云或隐式表示。
◆ 该方法不依赖于密集且干净的点云作为输入，仅需二维图像，并展现出对新颖形状的良好泛化能力。
实验表明，该方法在性能上超越了现有先进技术。</td></tr>
<tr><td>2026-02-24</td><td>Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting</td><td>[2602.20933](http://arxiv.org/pdf/2602.20933)</td><td>该论文针对稀疏视图下3D高斯泼溅（3DGS）的过拟合问题，提出了名为DropAnSH-GS的创新方法。其核心贡献与创新点如下：

◆ 揭示了现有3DGS Dropout方法中存在的“邻居补偿效应”，即被随机丢弃的高斯单元其功能会被邻近单元弥补，从而削弱了正则化效果。

◆ 提出了一种新颖的基于锚点的Dropout策略。该方法不再独立丢弃高斯单元，而是随机选择锚点高斯，并同步移除其空间邻居，从而有效打破局部冗余，促使模型学习更具鲁棒性的全局表征。

◆ 首次指出并处理了高阶球谐函数系数对颜色过拟合的贡献。将Dropout思想延伸至颜色属性，通过随机丢弃高阶球谐系数，迫使外观信息更集中于低阶系数。

◆ 该方法带来的一个直接优势是支持灵活的训练后模型压缩，即可以直接截断高阶球谐系数以实现模型简化，且性能损失很小。

◆ 所提方法计算开销可忽略，并能轻松集成到多种3DGS变体中，显著提升其在稀疏视图下的性能，实验证明其效果优于现有Dropout方法。</td></tr>
<tr><td>2026-02-24</td><td>RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction</td><td>[2602.20807](http://arxiv.org/pdf/2602.20807)</td><td>该论文提出了RU4D-SLAM框架，旨在解决动态环境中SLAM与场景重建的难题，其核心贡献是将3D高斯泼溅与SLAM结合，并扩展至4D（时空）动态场景重建。主要创新点如下：

◆ 首次将4D高斯泼溅概念引入SLAM系统，在空间3D表示中融入时间维度，实现对动态场景的连续重建。
◆ 集成了运动模糊渲染机制，增强了动态场景的表征能力，并能合成模糊图像以提升系统对真实拍摄条件的适应性。
◆ 扩展了逐像素不确定性建模方法，使其从原本仅适用于静态场景，发展到能有效处理模糊图像，从而提高了跟踪的鲁棒性。
◆ 提出了一种语义引导的重新加权机制，用于动态场景中逐像素不确定性估计，优化了对场景变化的感知。
◆ 引入了可学习的不透明度权重，支持自适应的4D地图构建，使系统能灵活处理动态物体和低质量输入。

实验表明，该方法在轨迹精度和4D场景重建质量上显著优于现有技术，尤其在包含运动物体和低质量图像的动态环境中表现突出。</td></tr>
<tr><td>2026-02-24</td><td>Monocular Endoscopic Tissue 3D Reconstruction with Multi-Level Geometry Regularization</td><td>[2602.20718](http://arxiv.org/pdf/2602.20718)</td><td>该论文针对单目内窥镜场景下可变形组织的三维重建问题，提出了一种基于3D高斯泼溅的创新方法，旨在同时实现高质量平滑表面重建与实时渲染能力。

◆ 核心贡献在于提出了一种结合表面感知重建与几何约束的新框架，基于3D高斯泼溅技术，兼顾了重建质量与渲染速度。
◆ 创新性地引入了表面感知重建机制，首先利用符号距离场方法构建网格，再以此网格约束高斯泼溅的重建过程，从而获得更一致的组织表面。
◆ 为模拟软组织的物理形变，设计了多级几何正则化约束，包括局部刚性约束与全局非刚性约束，以确保形变过程的物理合理性。
◆ 该方法最终实现了快速的渲染过程与平滑的表面外观，在纹理和几何两方面均达到扎实的重建质量，并通过定量与定性实验验证了其优越性。</td></tr>
<tr><td>2026-02-24</td><td>WildGHand: Learning Anti-Perturbation Gaussian Hand Avatars from Monocular In-the-Wild Videos</td><td>[2602.20556](http://arxiv.org/pdf/2602.20556)</td><td>该论文的核心贡献是提出了WildGHand框架，旨在从单目野外视频中学习抗干扰的高斯手部化身，解决了现有方法在复杂真实场景下性能退化的问题。

其核心创新点如下：
◆ 提出了一个基于优化的框架，首次将自适应3D高斯泼溅技术应用于单目野外视频，以重建高保真手部化身。
◆ 设计了一个动态扰动解耦模块，在优化过程中将视频中的扰动（如物体交互、运动模糊）显式建模为3D高斯属性上的时变偏置。
◆ 提出了一种扰动感知的优化策略，通过生成每帧各向异性的加权掩码来指导优化，从而在空间和时间维度上识别并抑制扰动。
◆ 构建了一个包含多种干扰的单目手部视频数据集，为野外手部化身重建任务提供了首个基准。
◆ 在自建数据集和公开数据集上的大量实验表明，该方法实现了最先进的性能，在PSNR等多项指标上相比基线模型有显著提升（例如PSNR相对提升最高达15.8%）。</td></tr>
<tr><td>2026-02-23</td><td>Aesthetic Camera Viewpoint Suggestion with 3D Aesthetic Field</td><td>[2602.20363](http://arxiv.org/pdf/2602.20363)</td><td>本文的核心贡献是提出了一种基于稀疏输入即可在三维空间进行美学推理并高效推荐拍摄视角的新方法。其核心创新点如下：

◆ 首次提出了“三维美学场”的概念，将美学评估从二维图像提升至三维空间，实现了对场景几何结构感知的美学推理。
◆ 提出了一种前馈式三维高斯溅射网络，能够从预训练的二维美学模型中提取高级美学知识并蒸馏到三维空间，仅需稀疏的输入视角即可预测新视角的美学质量。
◆ 设计了一个两阶段搜索流程，结合了粗粒度视角采样与基于梯度的精细化调整，从而高效地找到美学上佳的拍摄视角，避免了密集采集或耗时的强化学习搜索。
◆ 该方法在仅需稀疏捕获的条件下，就能实现与理解场景几何的、三维感知的美学建模，在视角推荐效果上超越了现有方法。</td></tr>
<tr><td>2026-02-23</td><td>Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques</td><td>[2602.20342](http://arxiv.org/pdf/2602.20342)</td><td>该论文的核心贡献是提出了一套完整的端到端系统，能够将无人机实时视频流高效转换为可用于AR/VR的高保真3D场景重建。

其核心创新点包括：
◆ 构建了一个集成实时视频流、传感器融合、位姿估计与3D高斯溅射优化的完整架构，实现了从数据采集到模型更新的端到端低延迟处理。
◆ 成功将新兴的3D高斯溅射技术整合到无人机实时感知系统中，解决了此前该技术与端到端无人机重建系统结合不足的问题。
◆ 系统能够实现模型的连续更新与在交互式可视化环境中的低延迟部署，直接支持沉浸式AR/VR应用。
◆ 实验表明，该系统在保持高视觉保真度（与离线参考结果误差仅4-7%）的同时，渲染性能和端到端延迟显著优于基于NeRF的方法。</td></tr>
<tr><td>2026-02-23</td><td>tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction</td><td>[2602.20160](http://arxiv.org/pdf/2602.20160)</td><td>本文提出tttLRM，一个用于长上下文自回归三维重建的新模型。其核心贡献与创新点如下：
◆ 引入测试时训练层，将多张观测图像高效压缩为该层的快速权重，在隐空间形成隐式三维表示，实现了计算复杂度随输入图像数量线性增长的长上下文处理能力。
◆ 该隐式表示可灵活解码为多种显式格式，如下游应用所需的高斯泼溅，兼顾了表示的通用性与实用性。
◆ 模型支持在线学习变体，能够对连续输入的观测流进行渐进式三维重建与细化，适用于动态或流式场景。
◆ 通过在新视图合成任务上进行预训练，模型可有效迁移至显式三维建模任务，从而提升重建质量并加速收敛。
◆ 大量实验表明，该方法在物体和场景的前馈式三维高斯重建任务上，性能优于现有先进技术。</td></tr>
<tr><td>2026-02-23</td><td>Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting</td><td>[2602.19916](http://arxiv.org/pdf/2602.19916)</td><td>该论文针对3D高斯泼溅技术难以准确建模复杂镜面反射的问题，提出了一个增强型辐射场通用框架。其核心贡献与创新点如下：

◆ 提出了一种新颖的增强型高斯核，通过引入视图依赖的不透明度来显式地建模镜面反射效果，从而有效分离漫反射和镜面分量。
◆ 设计了一种误差驱动的补偿策略，能够有效提升现有3DGS场景的渲染质量，增强了方法的通用性和修复能力。
◆ 开发了一套从2D高斯初始化开始，自适应插入并优化增强高斯核的完整流程，最终构建出增强的辐射场。
◆ 实验证明，该方法不仅在渲染质量上超越了先进的NeRF方法，同时实现了更高的参数效率，兼顾了效果与性能。</td></tr>
<tr><td>2026-02-23</td><td>One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image</td><td>[2602.19766](http://arxiv.org/pdf/2602.19766)</td><td>该论文提出One2Scene框架，从单张图像生成可自由探索的高质量3D场景，核心贡献在于通过分解任务与引入几何一致性先验解决了远视角下的失真问题。

◆ 将复杂问题分解为三个可处理的子任务：全景锚视图生成、3D几何支架构建、以及基于支架的新视图合成，提升了生成过程的稳定性和质量。
◆ 创新性地将单张全景图投影为多个稀疏锚视图，将重建任务重构为多视图立体匹配，从而能够利用大规模多视图数据集学习到的强几何先验。
◆ 设计了一个双向特征融合模块，有效加强了跨视图一致性，生成了高效且几何可靠的显式3D高斯溅射支架。
◆ 提出以该3D一致的几何支架作为强先验来驱动新视图生成，使得即使在较大相机运动下也能稳定输出照片级真实且几何准确的视图，支持沉浸式场景探索。</td></tr>
<tr><td>2026-02-23</td><td>RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing</td><td>[2602.19753](http://arxiv.org/pdf/2602.19753)</td><td>该论文针对3D高斯泼溅技术中原始图元数量庞大、评估效率低的问题，提出了一种名为RAP的快速前馈式重要性预测方法。其核心贡献与创新点如下：

◆ 提出了首个免渲染的3DGS图元重要性预测方法，直接依据高斯属性与局部邻域统计来推断重要性，彻底避免了传统方法依赖多视角渲染和可见性计算的瓶颈。

◆ 设计了一个紧凑的MLP网络，通过联合优化渲染损失、剪枝感知损失和显著性分布正则化损失，来准确预测每个图元的重要性分数。

◆ 该方法计算速度快、效率高，其计算时间不随视角数量增加而线性增长，克服了现有方法计算耗时、对视角选择和数量敏感的缺陷。

◆ 具备良好的泛化能力与易集成性，仅需在少量场景上训练后，即可有效泛化至未见过的数据，并能作为即插即用模块无缝集成到重建、压缩与传输流程中。

◆ 为3DGS处理流程的高效化提供了新思路，在保持重建质量的同时，显著提升了冗余去除、模型压缩等下游任务的效率与可扩展性。</td></tr>
<tr><td>2026-02-22</td><td>DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via Frequency-Aware Filtering</td><td>[2602.19323](http://arxiv.org/pdf/2602.19323)</td><td>该论文针对3D高斯泼溅（3DGS）技术易受对抗性攻击的问题，提出了一种提升其鲁棒性的防御方法。其核心贡献与创新点如下：

◆ 首次通过小波变换系统分析了对抗性扰动在输入图像低频与高频成分中的不同行为，为防御提供了理论依据。
◆ 基于上述分析，设计了一种简单而有效的频率感知防御策略，通过滤除高频噪声同时保留低频内容来重建训练视图。
◆ 该方法能有效抑制对抗性伪影，同时保持原始场景的真实性，且在干净数据上的训练性能不会受到显著损害。
◆ 实现了鲁棒性与干净输入性能之间的理想平衡，并且无需依赖干净的监督数据。
◆ 在多种基准测试和不同攻击强度下进行了广泛实验，验证了该方法能显著增强3DGS的鲁棒性，为构建更安全可靠的3D重建系统开辟了道路。</td></tr>
<tr><td>2026-02-21</td><td>PhysConvex: Physics-Informed 3D Dynamic Convex Radiance Fields for Reconstruction and Simulation</td><td>[2602.18886](http://arxiv.org/pdf/2602.18886)</td><td>该论文的核心贡献是提出了PhysConvex框架，将视觉重建与物理模拟统一于动态3D场景的表示中。其核心创新点如下：

◆ 提出了一种物理驱动的动态凸体表示法，用基于连续介质力学的凸体基元来表征可变形辐射场，实现了视觉外观与物理属性的一致性建模。

◆ 引入了边界驱动的动态凸体表征，通过顶点和表面的动力学来建模变形，能够捕捉空间自适应的非均匀形变以及演化的边界。

◆ 开发了一种高效的降阶凸体模拟方法，利用神经蒙皮本征模态作为形状与材料感知的形变基，在牛顿动力学下以时变降维自由度来平流动态凸体场，从而高效模拟复杂几何与异质材料。

◆ 凸体动力学提供了紧凑且无间隙的体覆盖，这一特性同时增强了几何表达的效率与物理模拟的保真度。

总体而言，该工作通过物理信息与神经表示的深度融合，实现了从视频中高保真重建几何、外观与物理属性，性能优于现有方法。</td></tr>
<tr><td>2026-02-20</td><td>Unifying Color and Lightness Correction with View-Adaptive Curve Adjustment for Robust 3D Novel View Synthesis</td><td>[2602.18322](http://arxiv.org/pdf/2602.18322)</td><td>该论文针对复杂光照下多视角图像采集的亮度与色彩不一致问题，提出了一种鲁棒的3D新视图合成方法Luminance-GS++，其核心贡献与创新点如下：

◆ 提出一个基于3D高斯泼溅的框架，在保持显式表示和实时渲染效率的同时，有效处理低光、过曝及复杂光照变化。
◆ 设计了一种全局视图自适应的亮度调整与局部像素级残差细化相结合的色彩校正机制，实现精确的光照与颜色统一。
◆ 引入了无监督优化目标，联合约束亮度校正与多视角几何及光度一致性，无需依赖人工标注或假设固定光照。
◆ 该方法不修改底层3D表示，保持了3D高斯泼溅的原有优势，在提升重建保真度的同时维持了实时渲染性能。
◆ 在多种挑战性场景下的实验表明，该方法在光照不一致条件下的新视图合成质量达到了先进水平。</td></tr>
<tr><td>2026-02-20</td><td>Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting</td><td>[2602.18314](http://arxiv.org/pdf/2602.18314)</td><td>该论文提出Diff2DGS，一个用于手术场景可靠三维重建的两阶段框架，其核心贡献与创新点如下：

◆ 提出一个新颖的两阶段框架，专门解决手术场景中因器械遮挡导致的重建不可靠问题。
◆ 在第一阶段，引入一个基于扩散模型的视频修复模块，利用时序先验信息，以高时空一致性的方式修复被器械遮挡的组织区域。
◆ 在第二阶段，改进2D高斯泼溅（2DGS）方法，为其配备一个可学习的形变模型（LDM），以精确捕捉动态组织的形变和人体解剖几何结构。
◆ 扩展了评估体系，不仅使用图像质量指标，更在SCARED数据集上进行了定量的深度精度分析，揭示了仅优化图像质量未必能获得最佳三维几何精度这一关键发现。
◆ 通过联合优化深度质量与外观保真度，该方法在EndoNeRF和StereoMIS等多个基准测试中，在外观（PSNR）和几何精度上均超越了现有先进技术。</td></tr>
<tr><td>2026-02-19</td><td>4D Monocular Surgical Reconstruction under Arbitrary Camera Motions</td><td>[2602.17473](http://arxiv.org/pdf/2602.17473)</td><td>该论文针对单目内窥镜视频中因大范围相机运动和场景形变导致的重建难题，提出了Local-EndoGS框架，实现了在任意相机运动下的高质量4D重建。其核心创新点如下：

◆ 提出了一种渐进式、基于窗口的全局表示方法，为每个观测窗口分配局部可变形场景模型，从而能够适应长序列和大范围的相机运动。

◆ 设计了从粗到细的优化策略，融合多视图几何、跨窗口信息与单目深度先验，在不依赖立体深度或精确运动恢复结构初始化的情况下实现鲁棒重建。

◆ 引入了长距离2D像素轨迹约束与物理运动先验，增强了动态形变过程的合理性与真实性。

实验表明，该方法在多个公开数据集上超越了现有技术，在外观质量和几何精度上表现优异。</td></tr>
<tr><td>2026-02-19</td><td>NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting</td><td>[2602.17182](http://arxiv.org/pdf/2602.17182)</td><td>该论文提出了一种用于内窥镜场景的单目非刚性SLAM系统NRGS-SLAM，其核心贡献在于利用可变形3D高斯溅射技术，有效解决了因组织形变导致的相机运动与场景形变耦合难题。具体创新点如下：

◆ 提出一种形变感知的3D高斯地图表示，为每个高斯基元引入可学习的形变概率，无需外部标注，通过贝叶斯自监督策略进行优化，从而显式建模场景形变。
◆ 设计了一个可变形跟踪模块，采用由粗到细的位姿估计策略，优先利用低形变区域进行鲁棒相机跟踪，随后高效更新每帧的形变场。
◆ 开发了一个可变形建图模块，能渐进式扩展和优化地图，在表征能力与计算效率之间取得平衡。
◆ 构建了一个统一的鲁棒几何损失函数，融入外部几何先验，以缓解单目非刚性SLAM固有的病态性问题。
实验表明，该系统在多个内窥镜数据集上实现了更精确的相机位姿估计和更高质量的逼真场景重建，性能优于现有方法。</td></tr>
<tr><td>2026-02-19</td><td>B$^3$-Seg: Camera-Free, Training-Free 3DGS Segmentation via Analytic EIG and Beta-Bernoulli Bayesian Updates</td><td>[2602.17134](http://arxiv.org/pdf/2602.17134)</td><td>该论文提出了一种名为B³-Seg的快速、无需相机和训练的3D高斯溅射开放词汇分割方法。其核心贡献与创新点如下：

◆ 首次在无需预定义相机视角、无需真实标签且无需重新训练的条件下，实现了交互式3D高斯溅射分割，极大提升了实用性。
◆ 提出一种理论严谨的贝叶斯方法，将分割问题重新定义为顺序性的Beta-Bernoulli贝叶斯更新过程，确保了分割概率估计的稳定收敛。
◆ 引入了基于解析计算的期望信息增益来主动选择下一个最佳视图，该策略具有自适应的单调性和子模性。
◆ 从理论上证明了该视图选择策略是一种贪婪算法，能获得(1-1/e)的近似最优保证，从而实现了可证明的信息采集效率。
◆ 实验表明，该方法仅需数秒即可完成端到端分割，效果与高成本监督方法相当，为影视游戏制作提供了真正低延迟的交互编辑工具。</td></tr>
<tr><td>2026-02-19</td><td>3D Scene Rendering with Multimodal Gaussian Splatting</td><td>[2602.17124](http://arxiv.org/pdf/2602.17124)</td><td>本文提出了一种集成射频感知与3D高斯泼溅的多模态场景渲染框架，旨在提升传统视觉方法在恶劣条件下的鲁棒性与效率。其核心贡献与创新点如下：

◆ 首次将射频感知（如汽车雷达）与3D高斯泼溅渲染相结合，构建了一个多模态渲染框架，以应对纯视觉方法在恶劣天气、低光照或遮挡等视觉线索不可靠场景下的局限。

◆ 利用射频信号对天气、光照及遮挡的强鲁棒性，仅需稀疏的射频深度测量数据，即可实现高效的深度预测，从而显著减少了对大量相机视角的依赖。

◆ 通过射频感知生成的高质量3D点云，为多种高斯泼溅架构提供了优异的初始化条件，提升了初始化的效率与质量，降低了传统视觉初始化所需的处理成本。

◆ 该框架在保持高计算与内存效率的同时，通过射频信息提升结构准确性，实现了高保真度的3D场景渲染，为工业监控、机器人及自动驾驶等应用提供了更可靠的解决方案。

◆ 数值实验验证了将射频感知智能融入高斯泼溅流程的优势，证明了其在渲染质量与鲁棒性方面的显著提升。</td></tr>
<tr><td>2026-02-19</td><td>i-PhysGaussian: Implicit Physical Simulation for 3D Gaussian Splatting</td><td>[2602.17117](http://arxiv.org/pdf/2602.17117)</td><td>本文提出i-PhysGaussian框架，核心贡献是将3D高斯泼溅（3DGS）重建技术与隐式物理模拟相结合，以显著提升复杂场景下物理仿真的稳定性和效率。

◆ 首创性地将3D高斯泼溅（3DGS）与隐式材料点法（MPM）积分器相耦合，实现了基于神经辐射场表示的高质量物理仿真。
◆ 采用基于动量平衡残差最小化的隐式牛顿型优化求解，替代传统的显式逐步更新方法，从根本上增强了系统的物理一致性。
◆ 该隐式求解方法大幅降低了对仿真时间步长的敏感性，解决了显式方法在高刚度材料或准静态运动等复杂场景下精度迅速下降的问题。
◆ 实验证明，该框架能使用比显式基线方法大20倍的时间步长保持稳定，在复杂动态过渡中仍能保持结构连贯性与运动平滑性。</td></tr>
<tr><td>2026-02-17</td><td>Semantic-Guided 3D Gaussian Splatting for Transient Object Removal</td><td>[2602.15516](http://arxiv.org/pdf/2602.15516)</td><td>该论文针对3D高斯泼溅重建中瞬态物体导致的伪影问题，提出了一种语义引导的去除方法。其核心贡献与创新点如下：

◆ 提出首个利用视觉-语言模型进行语义感知的瞬态物体去除框架，通过CLIP模型计算渲染视图与干扰物文本提示的相似性，无需依赖易受视差歧义影响的运动启发式方法。
◆ 设计了基于高斯粒子的语义评分累积机制，在训练迭代中为每个高斯积累语义分数，从而精准识别属于瞬态类别的高斯。
◆ 引入了一种校准阈值策略，对超过阈值的高斯进行不透明度正则化和周期性剪枝，有效移除了瞬态物体，同时保持了极低的内存开销和实时渲染性能。
◆ 该方法在RobustNeRF基准测试中，于多个序列上持续提升了重建质量，验证了在干扰物类别可预测场景中，语义指导是一种实用且高效的策略。</td></tr>
<tr><td>2026-02-17</td><td>DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles</td><td>[2602.15355](http://arxiv.org/pdf/2602.15355)</td><td>该论文的核心贡献是提出了DAV-GSWT框架，旨在以极少的输入数据合成用于大规模场景的高质量3D高斯分布Wang Tiles。其核心创新点如下：

◆ 提出一个数据高效框架，结合扩散模型先验与主动视角采样，仅需极少观测数据即可生成高保真高斯分布瓦片。
◆ 引入分层不确定性量化机制，能自主识别最具信息量的新视角，从而优化数据采集。
◆ 利用生成式扩散模型补全缺失的结构细节，确保瓦片拼接边界的过渡自然无缝。
◆ 通过上述技术显著降低了构建大规模虚拟环境所需的数据量，同时保持了视觉真实性与交互性能。</td></tr>
<tr><td>2026-02-16</td><td>Time-Archival Camera Virtualization for Sports and Visual Performances</td><td>[2602.15181](http://arxiv.org/pdf/2602.15181)</td><td>该论文针对体育和视觉表演中的相机虚拟化问题，提出了一种支持高效时间归档的神经渲染新方法。其核心贡献在于解决了现有动态场景新视角合成方法在快速、非刚性运动及多主体独立运动时渲染质量与连贯性不足的问题，并首次实现了对动态场景任意历史时刻的回顾式渲染。

◆ 创新性地采用神经体渲染框架，替代当前主流的动态3D高斯溅射等方法，以提升相机虚拟化的渲染质量与鲁棒性。
◆ 提出通过多视角同步相机间的刚性变换来建模动态场景，有效处理了快速、非刚性及多主体的复杂运动。
◆ 首次为神经渲染方法引入了高效的时间归档能力，用户可回溯到动态场景的任意过去时刻并进行新视角合成。
◆ 所实现的时间归档功能支持对直播事件的回放、分析与存档，填补了现有新视角合成技术在该应用领域的空白。</td></tr>
<tr><td>2026-02-16</td><td>Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery</td><td>[2602.14929](http://arxiv.org/pdf/2602.14929)</td><td>该论文的核心贡献是提出了一个无需配对监督、基于几何的零射样本地到卫星图像地理定位框架，并创建了首个系统性评测基准。

◆ 提出了Wrivinder框架，这是一个零样本、几何驱动的创新方法，通过聚合多张地面照片重建一致的三维场景，并将其与卫星图像对齐。
◆ 该框架融合了SfM重建、3D高斯泼溅、语义接地和单目深度度量线索，生成稳定的天顶视图渲染图，可直接与卫星上下文进行匹配，实现度量精确的相机地理定位。
◆ 针对该任务缺乏合适评测基准的问题，发布了MC-Sat数据集，这是一个精心策划的、连接多视角地面图像与地理注册卫星图块的数据集，涵盖多样户外环境。
◆ Wrivinder与MC-Sat共同为研究以几何为中心、无需配对监督的跨视图对齐任务，提供了首个全面的基线平台和测试基准。
◆ 在零样本实验中，Wrivinder在密集和大范围场景中均实现了低于30米的地理定位精度，验证了基于几何聚合的方法对于实现鲁棒的地面到卫星定位的有效性。</td></tr>
<tr><td>2026-02-16</td><td>Gaussian Mesh Renderer for Lightweight Differentiable Rendering</td><td>[2602.14493](http://arxiv.org/pdf/2602.14493)</td><td>该论文的核心贡献是提出了一种名为高斯网格渲染器（GMR）的新型轻量级可微分渲染器，旨在解决传统基于网格的可微分渲染器优化速度慢或内存占用大的问题。

◆ 核心创新在于将3D高斯抛雪球（3DGS）的高效光栅化流程与三角形网格表示紧密集成，利用前者的速度优势来优化后者。
◆ 提出了一种从网格三角形解析推导出对应高斯图元的方法，确保了网格的结构保真度，并建立了有效的梯度流。
◆ 与传统网格渲染器相比，该方法能产生更平滑的梯度，这尤其有利于在有限内存下使用更小的批量大小进行优化，从而提升了优化效率和稳定性。
◆ 整体上，GMR实现了一种轻量级的可微分渲染框架，为表面重建等任务提供了兼具高保真度、快速渲染和高效优化能力的新工具。</td></tr>
<tr><td>2026-02-15</td><td>Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation</td><td>[2602.14199](http://arxiv.org/pdf/2602.14199)</td><td>本文针对3D高斯泼溅（3DGS）在训练中高斯原语数量激增导致内存消耗过大的问题，提出了一种创新的多级离散小波变换频率调制框架。其核心贡献与创新点如下：

◆ 提出了一个基于多级离散小波变换（DWT）的频率调制框架，通过递归分解低频子带，构建了比现有单层方法更深的训练课程。
◆ 该多级调制机制能在训练早期提供由粗到细的渐进式监督，从而持续且有效地减少最终所需的高斯原语数量。
◆ 研究发现并简化了调制参数，证明仅需学习一个单一的缩放参数即可实现有效调制，而无需学习完整的高通滤波器，降低了优化复杂性。
◆ 该方法缓解了现有方案中因联合优化小波正则化与3D重建而产生的梯度竞争问题，该问题曾导致高斯过度致密化。
◆ 在多个标准数据集上的实验表明，该方法在保持具有竞争力渲染质量的同时，进一步显著降低了高斯数量，优化了存储与计算成本。</td></tr>
<tr><td>2026-02-14</td><td>High-fidelity 3D reconstruction for planetary exploration</td><td>[2602.13909](http://arxiv.org/pdf/2602.13909)</td><td>该论文的核心贡献是提出了一种用于行星探测的高保真三维重建自动化流程，旨在解决传统方法在无结构、低纹理外星环境中面临的挑战。

◆ 创新性地将辐射场方法（NeRF与高斯泼溅）集成到行星机器人重建流程中，以生成兼具光度细节与几何一致性的三维模型。
◆ 开发了一个统一且自动化的处理系统，有效结合了Nerfstudio与COLMAP框架，并兼容ROS2工作流。
◆ 实现了直接处理原始探测车数据（rosbag记录）的能力，仅需最小化的视觉输入即可生成密集、逼真且度量精确的三维场景表达。
◆ 该流程为自主系统在类行星极端环境下的感知与规划提供了增强支持，为未来基于辐射场的行星地图构建研究奠定了基础。</td></tr>
<tr><td>2026-02-14</td><td>Joint Orientation and Weight Optimization for Robust Watertight Surface Reconstruction via Dirichlet-Regularized Winding Fields</td><td>[2602.13801](http://arxiv.org/pdf/2602.13801)</td><td>该论文提出了一种名为DiWR的鲁棒性水密表面重建方法，其核心贡献与创新点如下：

◆ 提出了一种名为Dirichlet Winding Reconstruction (DiWR)的集成化新方法，能够从带有噪声、离群点及非均匀采样的无定向点云中直接重建水密表面。

◆ 创新性地将点方向、逐点面积权重和置信系数这三者的联合优化统一到一个端到端的流程中，无需依赖分离的预处理步骤。

◆ 以广义环绕数场为目标隐式表示，并通过最小化该场的狄利克雷能量及结合基于GWN的约束进行联合优化，使算法具备内在的鲁棒性。

◆ 该方法能有效补偿点云的非均匀采样，降低噪声影响，并在重建过程中自动弱化离群点的贡献，从而处理更具挑战性的数据。

◆ 实验验证表明，DiWR在来自3D高斯泼溅等复杂输入数据上表现优异，其效果超越了传统的多阶段流程和近年来的联合定向-重建方法。</td></tr>
<tr><td>2026-02-14</td><td>Nighttime Autonomous Driving Scene Reconstruction with Physically-Based Gaussian Splatting</td><td>[2602.13549](http://arxiv.org/pdf/2602.13549)</td><td>本文针对自动驾驶仿真中的夜间场景重建问题，提出了一种基于物理渲染的改进型3D高斯溅射方法。其核心贡献与创新点如下：

◆ 首次将物理渲染（PBR）原理集成到3D高斯溅射框架中，专门用于提升夜间自动驾驶场景的重建质量。
◆ 提出了复合场景高斯表示，并联合优化了基于双向反射分布函数（BRDF）的材料属性，以更好地建模夜间复杂光照与外观。
◆ 通过全局光照模块显式建模漫反射分量，并利用各向异性球形高斯函数来建模高光反射分量，从而更精确地分解与再现夜间光线交互。
◆ 该方法在显著提升夜间场景重建的定量指标与视觉质量的同时，保持了实时渲染的效率优势。
◆ 在nuScenes和Waymo两大真实自动驾驶数据集上的广泛实验证实，该方法在多种夜间场景下均优于现有先进技术。</td></tr>
<tr><td>2026-02-13</td><td>FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation</td><td>[2602.13444](http://arxiv.org/pdf/2602.13444)</td><td>该论文的核心贡献是提出了FlowHOI框架，用于生成语义接地、时序连贯的手-物交互序列，以解决现有模型在长视野、高接触任务中因缺乏显式交互表示而失败的问题。

◆ 提出了一种两阶段流匹配框架，能够根据第一视角观察、语言指令和3D高斯溅射场景重建，生成包含手部姿态、物体姿态和接触状态的手-物交互序列。
◆ 创新地将交互生成解耦为以几何为中心的抓取和以语义为中心的操控，后者利用紧凑的3D场景令牌，并通过运动-文本对齐损失确保交互在物理场景和语言指令上的语义接地。
◆ 设计了一个从大规模第一视角视频中恢复对齐的手-物轨迹和网格的重建流程，构建了高质量的手-物交互先验知识，以解决监督数据稀缺的问题。
◆ 在GRAB和HOT3D基准测试中，该方法在动作识别准确率和物理仿真成功率上均优于最强的扩散基线，同时实现了40倍的推理加速，并成功在真实机器人上完成了四项灵巧操控任务的验证。</td></tr>
<tr><td>2026-02-13</td><td>GSM-GS: Geometry-Constrained Single and Multi-view Gaussian Splatting for Surface Reconstruction</td><td>[2602.12796](http://arxiv.org/pdf/2602.12796)</td><td>该论文针对3D高斯泼溅技术在表面重建中因点云不规则性导致高频细节丢失的问题，提出了一个几何约束的单视图与多视图协同优化框架GSM-GS。其核心创新点如下：

◆ 提出单视图自适应子区域加权约束机制，利用图像梯度特征将场景划分为纹理丰富与纹理稀疏区域，并通过深度差异特征引导自适应滤波，以保留关键细节。

◆ 在单视图中采用针对区域纹理变化的双分支约束策略，根据不同区域的纹理特性差异化优化几何细节的表征能力。

◆ 设计多视图几何引导的跨视图点云关联方法，结合动态权重采样策略，在相邻点云帧间构建三维结构法向约束。

◆ 通过上述跨视图约束有效强化多视图一致性，提升复杂表面微观结构的重建保真度与整体几何精度。

实验表明，该方法在公开数据集上实现了优异的渲染质量与几何重建效果。</td></tr>
<tr><td>2026-02-12</td><td>LatentAM: Real-Time, Large-Scale Latent Gaussian Attention Mapping via Online Dictionary Learning</td><td>[2602.12314](http://arxiv.org/pdf/2602.12314)</td><td>该论文提出了LatentAM，一个用于开放词汇机器人感知的在线3D高斯溅射建图框架，其核心贡献与创新点如下：

◆ 提出了一种在线字典学习方法，替代了传统依赖特定模型解码器蒸馏高维视觉语言模型嵌入的方式，实现了模型无关且无需预训练，支持在测试时即插即用不同视觉语言模型。

◆ 为每个高斯图元关联一个紧凑的查询向量，通过一个带有可学习字典的注意力机制，可将其转换为近似的视觉语言模型嵌入，从而构建可扩展的潜在特征地图。

◆ 字典从流式观测中高效初始化，并在信任域正则化下在线优化，以适应动态演变的场景语义。

◆ 设计了一种基于体素哈希的高效地图管理策略，将优化限制在GPU上的活跃局部地图，而全局地图存储和索引在CPU上，确保了GPU内存使用的有界性，从而能够扩展到长轨迹和大规模环境。

实验表明，该方法在公开基准和大规模自定义数据集上，相比现有方法显著提升了特征重建保真度，并在评估数据集上达到了接近实时的速度（12-35 FPS）。</td></tr>
<tr><td>2026-02-12</td><td>3DGSNav: Enhancing Vision-Language Model Reasoning for Object Navigation via Active 3D Gaussian Splatting</td><td>[2602.12159](http://arxiv.org/pdf/2602.12159)</td><td>该论文提出了一种名为3DGSNav的零样本目标导航新框架，其核心贡献在于通过创新的三维场景表示与推理增强方法，提升了智能体在未知环境中寻找目标物体的能力。

◆ 首次将3D高斯泼溅（3DGS）技术作为视觉语言模型（VLM）的持久化记忆嵌入导航框架，增强了模型对环境的长期空间理解与推理能力。
◆ 提出主动感知机制，能够增量式构建环境的3DGS表示，并支持基于轨迹和前沿感知的自由视点渲染，为VLM提供更丰富的视觉上下文。
◆ 设计了结构化的视觉提示，并与思维链（CoT）提示相结合，共同优化了VLM在导航决策中的推理过程。
◆ 引入了一个包含实时目标检测器过滤潜在目标、以及由VLM驱动的主动视点切换以进行目标重验证的协同系统，从而确保了高效可靠的目标识别。
该框架在多个标准测试和真实四足机器人实验中均展现出鲁棒且具有竞争力的性能。</td></tr>
<tr><td>2026-02-12</td><td>GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry</td><td>[2602.11714](http://arxiv.org/pdf/2602.11714)</td><td>GSO-SLAM的核心贡献在于提出了一种新颖的双向耦合单目密集SLAM系统，它通过创新的集成框架，在实时性能下实现了卓越的场景重建与跟踪精度。

◆ 提出了双向耦合的视觉里程计（VO）与高斯泼溅（GS）集成新范式。该方法避免了现有方案中统一场景表示带来的高计算成本，或松散集成带来的冗余问题。

◆ 设计了一个基于期望最大化（EM）框架的联合优化机制。该机制能够同步优化VO产生的半稠密深度估计与GS场景表示，且不引入额外的计算开销。

◆ 创新性地提出了高斯泼溅初始化方法。该方法直接利用VO提供的图像信息、关键帧位姿和像素关联来生成接近最终效果的高斯场景初始状态，从而摒弃了传统依赖启发式方法的初始化流程。

◆ 整个系统在保持实时运行的同时，在公开数据集上验证了其领先的性能。实验表明，该系统在场景重建的几何/光度保真度以及跟踪精度方面均达到了先进水平。</td></tr>
<tr><td>2026-02-12</td><td>TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction</td><td>[2602.11705](http://arxiv.org/pdf/2602.11705)</td><td>该论文提出了TG-Field，一个用于静态和动态CT重建的几何感知高斯变形框架，其核心贡献与创新点如下：

◆ 针对高度稀疏投影和动态运动下的严重伪影问题，提出了一个专为CT重建定制的几何感知高斯变形框架，统一处理静态与动态场景。
◆ 采用多分辨率哈希编码器来捕获局部空间先验，在超稀疏投影设置下有效正则化高斯基元参数，提升了重建的稳定性。
◆ 将框架扩展至动态重建，引入了时间条件表征和一个时空注意力块，以自适应聚合特征，从而解决时空模糊性并增强时间连贯性。
◆ 额外设计了一个运动流网络，用于精细建模呼吸运动，以追踪局部解剖结构变形，进一步提升了动态重建的精度。
◆ 在合成与真实数据集上的实验表明，该方法在高度稀疏视角条件下 consistently 超越现有方法，达到了最先进的重建精度。</td></tr>
<tr><td>2026-02-13</td><td>Variation-aware Flexible 3D Gaussian Editing</td><td>[2602.11638](http://arxiv.org/pdf/2602.11638)</td><td>该论文针对3D高斯泼溅（3DGS）间接编辑方法存在的跨视图不一致、灵活性差和效率低的问题，提出了名为VF-Editor的原生直接编辑框架。其核心贡献与创新点如下：

◆ 提出了VF-Editor，首次实现了对3D高斯基元的原生直接编辑，通过前馈方式预测属性变化，从根本上避免了传统“2D编辑再3D投影”范式带来的不一致性问题。

◆ 设计了一个新颖的“变化预测器”，该预测器通过从2D编辑知识中蒸馏学习而来，能够高效且准确地估计3D高斯属性的变化量。

◆ 该预测器采用统一架构，编码输入后生成一个变化场，并利用两个并行的可学习解码函数，迭代推断每个3D高斯的属性编辑量。

◆ 得益于其统一设计，VF-Editor能够无缝地从多种不同的2D编辑器和编辑策略中蒸馏知识，并集成到单个预测器中，实现了向3D领域灵活且高效的知识迁移，极大地提升了编辑的灵活性和效果。</td></tr>
<tr><td>2026-02-12</td><td>LeafFit: Plant Assets Creation from 3D Gaussian Splatting</td><td>[2602.11577](http://arxiv.org/pdf/2602.11577)</td><td>LeafFit的核心贡献是提出了一套从植物3D高斯泼溅模型生成可编辑、可实例化网格资产的完整流程。其创新点主要体现在以下几个方面：

◆ 首次提出将高保真但无结构的3DGS植物模型，转化为游戏等传统生产流程兼容的轻量级网格资产，解决了3DGS内存占用高、缺乏拓扑结构的关键问题。

◆ 利用植物叶片形状重复的特性，设计了一套从无结构点云中自动分割单叶片的方案，并引入用户交互作为备选，提高了分割的鲁棒性和实用性。

◆ 创新地采用可微移动最小二乘法进行模板拟合，能够将一个代表性的叶片模板精确地变形适配到所有其他叶片上，保证了重建的几何准确性。

◆ 提出高效的运行时渲染方案，通过顶点着色器实时计算变形，极大降低了存储开销，实现了高质量与低存储的平衡。

◆ 整体流程最终实现了参数化编辑能力，允许用户调整模板等参数来批量修改资产，为植物资产的快速创建与迭代提供了新工具。实验证明该方法在分割质量、变形精度和减容方面均优于现有基线。</td></tr>
<tr><td>2026-02-12</td><td>ReaDy-Go: Real-to-Sim Dynamic 3D Gaussian Splatting Simulation for Environment-Specific Visual Navigation with Moving Obstacles</td><td>[2602.11575](http://arxiv.org/pdf/2602.11575)</td><td>该论文提出ReaDy-Go，一个用于视觉导航的实景到仿真动态模拟新方法，旨在解决动态真实环境中导航策略训练的难题。

其核心贡献与创新点如下：
◆ 首创了一个实景到仿真的动态模拟流程，能针对目标部署环境（如家庭、工厂）合成逼真的动态导航场景，克服了以往仿真大多只针对静态场景的局限。
◆ 提出动态3D高斯溅射模拟器，将重建的静态高斯溅射场景与可动画化的人体高斯溅射化身相结合，实现了动态障碍物（行人）在真实场景中的自然插入与运动。
◆ 开发了一套完整的动态环境导航数据集生成方法，利用上述模拟器、专为动态高斯表示设计的机器人专家规划器以及人工规划器，自动生成包含移动障碍物的逼真训练数据。
◆ 通过生成的动态数据集训练导航策略，使得策略在仿真和真实世界实验中均优于基线方法，显著提升了在存在移动障碍物情况下的导航鲁棒性，并成功实现了仿真到现实的迁移与零样本泛化。</td></tr>
<tr><td>2026-02-10</td><td>ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D Gaussian Splatting</td><td>[2602.10278](http://arxiv.org/pdf/2602.10278)</td><td>本文针对单图像3D重建中因遮挡导致信息缺失的难题，提出了一种名为ERGO的自适应优化框架。其核心贡献与创新点如下：

◆ 提出了基于超额风险分解的优化框架，将3D高斯泼溅的损失分解为可优化的超额风险与不可约的贝叶斯误差，从而量化合成监督信号中的噪声。

◆ 通过动态估计视图特定的超额风险，自适应调整优化过程中的损失权重，使模型能有效抵抗合成多视角图像中的几何与纹理不一致噪声。

◆ 引入了几何感知与纹理感知的优化目标，与超额风险权重机制形成互补，构建了一个协同的全局-局部优化范式。

◆ 该框架显著提升了重建3D内容的几何保真度与纹理质量，在多个公开数据集上的实验证明了其优于现有先进方法的性能。</td></tr>
<tr><td>2026-02-10</td><td>XSPLAIN: XAI-enabling Splat-based Prototype Learning for Attribute-aware INterpretability</td><td>[2602.10239](http://arxiv.org/pdf/2602.10239)</td><td>该论文针对3D高斯泼溅（3DGS）模型缺乏可解释性的问题，提出了首个专门为其分类任务设计的原型学习解释框架XSPLAIN。其核心贡献与创新点如下：

◆ 提出了首个面向3DGS分类的事前可解释性框架，填补了该领域空白。
◆ 设计了一种新颖的可逆正交变换，能在保持模型原始决策边界严格不变的前提下，解耦特征通道以实现可解释性。
◆ 采用基于原型的解释方法，将预测关联到有代表性的训练样本，提供直观的“此像彼”式推理，且不牺牲分类性能。
◆ 通过体素聚合的PointNet主干网络有效处理3DGS数据，克服了传统点云解释方法中显著性图模糊、无法捕捉高斯基元体积一致性的缺陷。
◆ 严格的用户研究（51人）证实了其解释的优越性与用户信任度，显著优于基线方法。</td></tr>
<tr><td>2026-02-10</td><td>ArtisanGS: Interactive Tools for Gaussian Splat Selection with AI and Human in the Loop</td><td>[2602.10173](http://arxiv.org/pdf/2602.10173)</td><td>该论文的核心贡献是开发了一套名为ArtisanGS的交互式工具集，旨在解决从非结构化3D高斯泼溅（3DGS）场景中精确选择和分割对象的难题，并支持可控的局部编辑。

◆ 提出了一套以灵活的高斯泼溅选择和分割为核心的交互式工具集，而非专注于全自动或高级编辑，填补了该领域交互工具的空白。
◆ 引入了一种快速的AI驱动方法，能将用户引导的2D选择掩码传播到3DGS选择中，并允许用户在出现错误时进行干预修正。
◆ 结合了灵活的手动选择和分割工具，使得用户能够对非结构化的3DGS场景实现几乎任意的二进制分割。
◆ 开发了一种用户引导的局部编辑方法，利用定制的视频扩散模型，并让用户通过选择工具直接控制AI可修改的区域，实现了下游应用。
◆ 整个工具集无需额外优化即可适用于任何野外捕获的3DGS场景，提升了实用性和易用性。</td></tr>
<tr><td>2026-02-10</td><td>Faster-GS: Analyzing and Improving Gaussian Splatting Optimization</td><td>[2602.09999](http://arxiv.org/pdf/2602.09999)</td><td>本文的核心贡献在于系统性地整合、评估并改进了3D高斯泼溅（3DGS）的优化过程，提出了一个名为Faster-GS的高效新系统。其创新点主要包括：

◆ 系统性地整合与评估了先前3DGS研究中最有效且广泛适用的优化策略，厘清了算法本质改进与工程实现优化的关系，为公平比较建立了基础。

◆ 提出了多项新颖的优化技术，并深入研究了原框架中未被充分探索的方面，如数值稳定性、高斯截断和梯度近似等问题。

◆ 最终实现的Faster-GS系统，在保持视觉质量的前提下，实现了高达5倍的训练加速，为3DGS优化建立了一个新的高性价比、资源高效的基准。

◆ 进一步证明了所提出的优化方案可成功应用于4D高斯重建（即动态非刚性场景），实现了高效的动态场景优化。</td></tr>
<tr><td>2026-02-10</td><td>CompSplat: Compression-aware 3D Gaussian Splatting for Real-world Video</td><td>[2602.09816](http://arxiv.org/pdf/2602.09816)</td><td>本文提出CompSplat框架，旨在解决真实世界长视频在新视角合成任务中因压缩和相机位姿问题导致的渲染质量下降难题。其核心贡献与创新点如下：

◆ 首次在3D高斯泼溅框架中显式建模逐帧压缩特性，直接针对视频压缩带来的帧间不一致性进行优化。
◆ 提出压缩感知的帧加权机制，在训练中区分处理不同压缩程度的帧，以减轻累积几何误差。
◆ 设计自适应修剪策略，增强场景表示的鲁棒性与几何一致性，尤其在重度压缩条件下效果显著。
◆ 整体框架能同时处理长序列、未知相机位姿与多样化压缩模式，克服了现有方法通常只侧重其中单一问题的局限。
◆ 在多个高难度基准测试上实现领先的渲染质量与位姿估计精度，尤其在严重压缩环境下大幅超越现有先进方法。</td></tr>
<tr><td>2026-02-10</td><td>Toward Fine-Grained Facial Control in 3D Talking Head Generation</td><td>[2602.09736](http://arxiv.org/pdf/2602.09736)</td><td>该论文的核心贡献是提出了一个名为FG-3DGS的新框架，旨在解决3D说话头生成中细粒度面部控制不足的难题，特别是唇部同步不准确和面部抖动问题，以生成时序一致且高保真的结果。

其核心创新点包括：
◆ 提出了一种频率感知的解耦策略，根据面部区域的不同运动特性进行显式建模。将面颊、鼻子等低频运动区域与眼睛、嘴巴等高频运动区域分开处理。
◆ 针对高频运动区域（如眼、口），设计了由面部区域掩码引导的专用网络进行独立捕捉，以实现更精细的控制。
◆ 采用高斯增量来表示预测的运动动态，并将其应用于静态高斯点以生成最终帧，保持了表示的灵活性。
◆ 引入了一个高频细化的渲染后对齐机制，该机制通过预训练模型从大规模音视频对中学习，以增强单帧生成质量并实现更精准的唇部同步。</td></tr>
<tr><td>2026-02-10</td><td>Stability and Concentration in Nonlinear Inverse Problems with Block-Structured Parameters: Lipschitz Geometry, Identifiability, and an Application to Gaussian Splatting</td><td>[2602.09415](http://arxiv.org/pdf/2602.09415)</td><td>本文针对具有块结构参数的非线性逆问题，建立了一个统一的算子理论框架，以分析其稳定性和统计集中性。

◆ 提出了一套统一假设，将块状Lipschitz几何、局部可辨识性与次高斯噪声相结合，为一大类高维非线性逆问题提供了普适的分析基础。
◆ 在该框架下，同时建立了确定性稳定性不等式、最小二乘失配函数的全局Lipschitz界，以及非渐近的集中性估计。
◆ 推导出了不依赖于具体重建算法、仅由前向算子本质决定的高概率参数误差界，揭示了算法无关的算子级性能极限。
◆ 将理论具体应用于高斯泼溅渲染算子，验证了其满足所提假设，并导出了控制其Lipschitz连续性和分辨率相关可观测性的显式常数。
◆ 由此揭示了一个根本性的稳定性-分辨率权衡，证明估计误差本质上受图像分辨率与模型复杂度之比的制约。</td></tr>
<tr><td>2026-02-10</td><td>Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields</td><td>[2602.08958](http://arxiv.org/pdf/2602.08958)</td><td>该论文针对植物生长这一独特动态场景，提出了一种新颖的4D重建方法，其核心贡献与创新点如下：

◆ 首创了适用于植物生长的3D高斯流场表示。该方法将生长过程建模为高斯参数（位置、尺度、朝向、颜色、不透明度）随时间变化的导数，从而能够表达非线性和连续时间的生长动态。

◆ 解决了新生几何结构的建模难题。与传统的变形场或4D高斯溅射方法不同，本方法不依赖固定的高斯集合或线性运动轨迹，能够自然地模拟植物在生长过程中几何结构的扩张、分枝与分化。

◆ 提出了“逆向生长”的初始化策略。为了获得足够的高斯基元，该方法先重建成熟植株，然后学习一个逆向生长过程，以此模拟植物的发展历史，为正向生长建模提供了高质量的初始状态。

◆ 在多个植物生长时序数据集上验证了优越性。实验表明，该方法在图像质量和几何精度方面均优于现有方法，为生长中的三维结构的外观建模提供了新途径。</td></tr>
<tr><td>2026-02-09</td><td>Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limit</td><td>[2602.08909](http://arxiv.org/pdf/2602.08909)</td><td>该论文的核心贡献在于对标准多视图优化产生的3D高斯泼溅（3DGS）解，即“渲染最优参考”，进行了系统性分析，并揭示了其内在规律与预测极限。

◆ 首次将标准3DGS优化结果定义为“渲染最优参考”，并系统分析了其统计特性，发现其尺度呈混合结构、辐射度呈双峰分布等跨场景稳定模式。
◆ 通过可学习性探测，揭示了参数可预测性根本取决于点云密度，发现了“密度分层”现象：稠密区域参数与几何相关可无渲染预测，稀疏区域则普遍预测失败。
◆ 通过方差分解进行形式化论证，阐明稀疏区域因可见性异质性导致几何与外观参数强耦合，其协方差主导，难以解耦预测。
◆ 据此阐明了ROR的双重本质：在稠密区是可由点云推断的“几何基元”，在稀疏区则是必须依赖多视图约束的“视图合成基元”。
◆ 基于上述发现，提出了提升训练鲁棒性的密度感知策略，并讨论了能自适应平衡前馈预测与基于渲染优化的系统架构方向。</td></tr>
<tr><td>2026-02-09</td><td>GaussianCaR: Gaussian Splatting for Efficient Camera-Radar Fusion</td><td>[2602.08784](http://arxiv.org/pdf/2602.08784)</td><td>该论文的核心贡献是提出了GaussianCaR，一个用于鸟瞰图分割的端到端相机-雷达融合网络。其创新点主要体现在以下方面：

◆ 首次将高斯泼溅技术重新定位为一种高效的通用视图转换器，用以桥接相机与雷达之间的视图差异。
◆ 提出了一种新颖的融合范式，直接映射原始传感器数据（图像像素和雷达点）到共享的鸟瞰图潜在特征空间，而非依赖中间表示。
◆ 设计了一个结合多尺度融合与Transformer解码器的架构，能够高效地提取鸟瞰图特征。
◆ 在nuScenes数据集上的实验表明，该方法在车辆、道路和车道分隔线的分割任务上达到了领先的性能（IoU分别为57.3%、82.9%和50.1%）。
◆ 在实现优异性能的同时，保持了高达3.2倍的推理速度优势，显著提升了效率。</td></tr>
<tr><td>2026-02-09</td><td>Rotated Lights for Consistent and Efficient 2D Gaussians Inverse Rendering</td><td>[2602.08724](http://arxiv.org/pdf/2602.08724)</td><td>该论文针对基于2D高斯泼溅的逆向渲染中材质反照率估计不准确、存在阴影残留的问题，提出了名为RotLight的创新解决方案。其核心贡献与创新点如下：

◆ 提出一种简单实用的RotLight数据采集设置，仅需在采集过程中旋转物体数次（如两次），即可有效减少反照率估计中的歧义性，缓解颜色失真与阴影残留问题。

◆ 引入一个代理网格模型，该模型不仅能实现精确的入射光线追踪，还通过启用残差约束改善了全局光照的处理，从而提升了逆向渲染的整体精度。

◆ 将上述旋转采集策略与代理网格增强的2D高斯逆向渲染框架相结合，在合成与真实世界数据集上验证了该方法在保持高效计算的同时，实现了更优的反照率分解质量。</td></tr>
<tr><td>2026-02-09</td><td>Informative Object-centric Next Best View for Object-aware 3D Gaussian Splatting in Cluttered Scenes</td><td>[2602.08266](http://arxiv.org/pdf/2602.08266)</td><td>该论文针对杂乱场景中因遮挡导致观测不全的问题，提出了一种面向对象的下一代最佳视角选择方法，并与对象感知的3D高斯泼溅技术结合，以构建更可靠的场景表示。其核心贡献与创新点如下：

◆ 提出了一种实例感知的下一代最佳视角策略，克服了现有方法仅依赖几何线索、忽略操作相关语义以及过度偏向利用而非探索的局限。
◆ 开发了对象感知的3D高斯泼溅表示，能将实例级信息蒸馏为独热对象向量，从而显式编码语义信息。
◆ 利用对象特征计算置信加权的信息增益，以此指导识别与错误及不确定高斯分布相关的区域，优先探索未充分观测的区域。
◆ 该方法可灵活调整为以对象为中心的下一代最佳视角策略，能将视角选择聚焦于目标物体，从而提升对物体摆放位置的重建鲁棒性。
◆ 实验证明，该方法在合成和真实数据集上显著降低了深度误差，并在真实机器人操作任务中验证了有效性。</td></tr>
<tr><td>2026-02-07</td><td>Thermal odometry and dense mapping using learned ddometry and Gaussian splatting</td><td>[2602.07493](http://arxiv.org/pdf/2602.07493)</td><td>本文针对热红外传感器在黑暗、烟尘等恶劣条件下成像稳定的优势，提出了一种新型的热视觉里程计与稠密建图系统TOM-GS。其核心贡献与创新点如下：

◆ 首次将高斯泼溅（Gaussian Splatting）重建技术引入热相机SLAM系统，实现了高效且高质量的稠密三维地图构建。
◆ 提出了一种融合学习式里程计与高斯泼溅建图的完整框架，克服了传统几何方法在多样化数据上易失效且无法生成稠密地图的局限。
◆ 设计了专门的热图像增强模块，优化了热红外图像的输入质量，提升了系统的感知能力。
◆ 集成了单目深度估计信息，增强了在单热相机配置下的场景几何理解与建图效果。
◆ 通过大量实验验证，该系统在运动估计和新视角渲染任务上均优于现有学习方法，证明了学习式流程在热视觉里程计与稠密重建中的优越性。</td></tr>
<tr><td>2026-02-06</td><td>Zero-Shot UAV Navigation in Forests via Relightable 3D Gaussian Splatting</td><td>[2602.07101](http://arxiv.org/pdf/2602.07101)</td><td>该论文的核心贡献是提出一个端到端强化学习框架，实现了无人机在复杂森林环境中的零样本、高速、避障导航。其创新点主要在于：

◆ 提出可重光照的3D高斯溅射技术，将场景的几何外观与光照分离，解决了传统神经渲染中光照与场景耦合的问题。
◆ 构建了一个基于真实数据的高保真仿真环境，并利用上述技术合成从强烈日光到漫射阴天等多种逼真光照条件，用于策略训练。
◆ 设计了一种强化学习策略，能够直接根据单目RGB图像输出连续控制指令，并通过光照增强训练迫使策略学习光照不变的鲁棒视觉特征。
◆ 最终实现了无需微调即可零样本迁移到真实世界，使轻型四旋翼无人机能在复杂森林中以高达10米/秒的速度进行抗光照干扰的稳健导航。</td></tr>
<tr><td>2026-02-06</td><td>DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos</td><td>[2602.06846](http://arxiv.org/pdf/2602.06846)</td><td>◆ Spatial audio is crucial for creating compelling immersive 360-degree video experiences.
◆ However, generating realistic spatial audio, such as first-order ambisonics (FOA), from 360-degree videos in complex acoustic scenes remains challenging.
◆ Existing methods often overlook the dynamic nature and acoustic complexity of 360-degree scenes, fail to fully account for dynamic sound sources, and neglect complex environmental effects such as occlusion, reflections, and reverberation, which are influenced by scene geometries and materials.</td></tr>
<tr><td>2026-02-06</td><td>GaussianPOP: Principled Simplification Framework for Compact 3D Gaussian Splatting via Error Quantification</td><td>[2602.06830](http://arxiv.org/pdf/2602.06830)</td><td>◆ Existing 3D Gaussian Splatting simplification methods commonly use importance scores, such as blending weights or sensitivity, to identify redundant Gaussians.
◆ However, these scores are not driven by visual error metrics, often leading to suboptimal trade-offs between compactness and rendering fidelity.
◆ We present GaussianPOP, a principled simplification framework based on analytical Gaussian error quantification.</td></tr>
<tr><td>2026-02-06</td><td>Uncertainty-Aware 4D Gaussian Splatting for Monocular Occluded Human Rendering</td><td>[2602.06343](http://arxiv.org/pdf/2602.06343)</td><td>◆ High-fidelity rendering of dynamic humans from monocular videos typically degrades catastrophically under occlusions.
◆ Existing solutions incorporate external priors-either hallucinating missing content via generative models, which induces severe temporal flickering, or imposing rigid geometric heuristics that fail to capture diverse appearances.
◆ To this end, we reformulate the task as a Maximum A Posteriori estimation problem under heteroscedastic observation noise.</td></tr>
<tr><td>2026-02-05</td><td>From Blurry to Believable: Enhancing Low-quality Talking Heads with 3D Generative Priors</td><td>[2602.06122](http://arxiv.org/pdf/2602.06122)</td><td>◆ Creating high-fidelity, animatable 3D talking heads is crucial for immersive applications, yet often hindered by the prevalence of low-quality image or video sources, which yield poor 3D reconstructions.
◆ In this paper, we introduce SuperHead, a novel framework for enhancing low-resolution, animatable 3D head avatars.
◆ The core challenge lies in synthesizing high-quality geometry and textures, while ensuring both 3D and temporal consistency during animation and preserving subject identity.</td></tr>
<tr><td>2026-02-05</td><td>NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects</td><td>[2602.05822](http://arxiv.org/pdf/2602.05822)</td><td>◆ We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs.
◆ Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection.
◆ The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation.</td></tr>
<tr><td>2026-02-05</td><td>PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction</td><td>[2602.05190](http://arxiv.org/pdf/2602.05190)</td><td>◆ We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis.
◆ Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames.
◆ These components are integrated into a fully differentiable, end-to-end trainable pipeline.</td></tr>
<tr><td>2026-02-04</td><td>QuantumGS: Quantum Encoding Framework for Gaussian Splatting</td><td>[2602.05047](http://arxiv.org/pdf/2602.05047)</td><td>◆ Recent advances in neural rendering, particularly 3D Gaussian Splatting (3DGS), have enabled real-time rendering of complex scenes.
◆ However, standard 3DGS relies on spherical harmonics, which often struggle to accurately capture high-frequency view-dependent effects such as sharp reflections and transparency.
◆ While hybrid approaches like Viewing Direction Gaussian Splatting (VDGS) mitigate this limitation using classical Multi-Layer Perceptrons (MLPs), they remain limited by the expressivity of classical networks in low-parameter regimes.</td></tr>
<tr><td>2026-02-04</td><td>Nix and Fix: Targeting 1000x Compression of 3D Gaussian Splatting with Diffusion Models</td><td>[2602.04549](http://arxiv.org/pdf/2602.04549)</td><td>◆ 3D Gaussian Splatting (3DGS) revolutionized novel view rendering.
◆ Instead of inferring from dense spatial points, as implicit representations do, 3DGS uses sparse Gaussians.
◆ This enables real-time performance but increases space requirements, hindering applications such as immersive communication.</td></tr>
<tr><td>2026-02-04</td><td>VecSet-Edit: Unleashing Pre-trained LRM for Mesh Editing from Single Image</td><td>[2602.04349](http://arxiv.org/pdf/2602.04349)</td><td>◆ 3D editing has emerged as a critical research area to provide users with flexible control over 3D assets.
◆ While current editing approaches predominantly focus on 3D Gaussian Splatting or multi-view images, the direct editing of 3D meshes remains underexplored.
◆ Prior attempts, such as VoxHammer, rely on voxel-based representations that suffer from limited resolution and necessitate labor-intensive 3D mask.</td></tr>
<tr><td>2026-02-04</td><td>JOintGS: Joint Optimization of Cameras, Bodies and 3D Gaussians for In-the-Wild Monocular Reconstruction</td><td>[2602.04317](http://arxiv.org/pdf/2602.04317)</td><td>◆ Reconstructing high-fidelity animatable 3D human avatars from monocular RGB videos remains challenging, particularly in unconstrained in-the-wild scenarios where camera parameters and human poses from off-the-shelf methods (e.g., COLMAP, HMR2.0) are often inaccurate.
◆ Splatting (3DGS) advances demonstrate impressive rendering quality and real-time performance, they critically depend on precise camera calibration and pose annotations, limiting their applicability in real-world settings.
◆ We present JOintGS, a unified framework that jointly optimizes camera extrinsics, human poses, and 3D Gaussian representations from coarse initialization through a synergistic refinement mechanism.</td></tr>
<tr><td>2026-02-04</td><td>Towards Next-Generation SLAM: A Survey on 3DGS-SLAM Focusing on Performance, Robustness, and Future Directions</td><td>[2602.04251](http://arxiv.org/pdf/2602.04251)</td><td>◆ Traditional Simultaneous Localization and Mapping (SLAM) systems often face limitations including coarse rendering quality, insufficient recovery of scene details, and poor robustness in dynamic environments.
◆ 3D Gaussian Splatting (3DGS), with its efficient explicit representation and high-quality rendering capabilities, offers a new reconstruction paradigm for SLAM.
◆ This survey comprehensively reviews key technical approaches for integrating 3DGS with SLAM.</td></tr>
<tr><td>2026-02-03</td><td>AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting</td><td>[2602.04043](http://arxiv.org/pdf/2602.04043)</td><td>◆ The growing demand for rapid and scalable 3D asset creation has driven interest in feed-forward 3D reconstruction methods, with 3D Gaussian Splatting (3DGS) emerging as an effective scene representation.
◆ While recent approaches have demonstrated pose-free reconstruction from unposed image collections, integrating stylization or appearance control into such pipelines remains underexplored.
◆ Existing attempts largely rely on image-based conditioning, which limits both controllability and flexibility.</td></tr>
<tr><td>2026-02-03</td><td>Constrained Dynamic Gaussian Splatting</td><td>[2602.03538](http://arxiv.org/pdf/2602.03538)</td><td>◆ While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets.
◆ In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training.
◆ Our key insight is to introduce a differentiable budget controller as the core optimization driver.</td></tr>
<tr><td>2026-02-03</td><td>Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization</td><td>[2602.03327](http://arxiv.org/pdf/2602.03327)</td><td>◆ Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity.
◆ However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios.
◆ While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors.</td></tr>
<tr><td>2026-02-03</td><td>WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU</td><td>[2602.03207](http://arxiv.org/pdf/2602.03207)</td><td>◆ We present WebSplatter, an end-to-end GPU rendering pipeline for the heterogeneous web ecosystem.
◆ Unlike naive ports, WebSplatter introduces a wait-free hierarchical radix sort that circumvents the lack of global atomics in WebGPU, ensuring deterministic execution across diverse hardware.
◆ Furthermore, we propose an opacity-aware geometry culling stage that dynamically prunes splats before rasterization, significantly reducing overdraw and peak memory footprint.</td></tr>
<tr><td>2026-02-03</td><td>SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation</td><td>[2602.02989](http://arxiv.org/pdf/2602.02989)</td><td>◆ Novel view synthesis of dynamic scenes is fundamental to achieving photorealistic 4D reconstruction and immersive visual experiences.
◆ Recent progress in Gaussian-based representations has significantly improved real-time rendering quality, yet existing methods still struggle to maintain a balance between long-term static and short-term dynamic regions in both representation and optimization.
◆ To address this, we present SharpTimeGS, a lifespan-aware 4D Gaussian framework that achieves temporally adaptive modeling of both static and dynamic regions under a unified representation.</td></tr>
<tr><td>2026-02-02</td><td>SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation</td><td>[2602.02402](http://arxiv.org/pdf/2602.02402)</td><td>◆ Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions.
◆ Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization.
◆ This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation.</td></tr>
<tr><td>2026-02-02</td><td>UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction</td><td>[2602.02089](http://arxiv.org/pdf/2602.02089)</td><td>◆ While 3D Gaussian Splatting (3DGS) enables high-quality, real-time rendering for bounded scenes, its extension to large-scale urban environments gives rise to critical challenges in terms of geometric consistency, memory efficiency, and computational scalability.
◆ To address these issues, we present UrbanGS, a scalable reconstruction framework that effectively tackles these challenges for city-scale applications.
◆ First, we propose a Depth-Consistent D-Normal Regularization module.</td></tr>
<tr><td>2026-02-03</td><td>SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors</td><td>[2602.02000](http://arxiv.org/pdf/2602.02000)</td><td>◆ Reconstructing 3D scenes from sparse images remains a challenging task due to the difficulty of recovering accurate geometry and texture without optimization.
◆ Recent approaches leverage generalizable models to generate 3D scenes using 3D Gaussian Splatting (3DGS) primitive.
◆ However, they often fail to produce continuous surfaces and instead yield discrete, color-biased point clouds that appear plausible at normal resolution but reveal severe artifacts under close-up views.</td></tr>
<tr><td>2026-02-02</td><td>CloDS: Visual-Only Unsupervised Cloth Dynamics Learning in Unknown Conditions</td><td>[2602.01844](http://arxiv.org/pdf/2602.01844)</td><td>◆ Deep learning has demonstrated remarkable capabilities in simulating complex dynamic systems.
◆ However, existing methods require known physical properties as supervision or inputs, limiting their applicability under unknown conditions.
◆ To explore this challenge, we introduce Cloth Dynamics Grounding (CDG), a novel scenario for unsupervised learning of cloth dynamics from multi-view visual observations.</td></tr>
<tr><td>2026-02-02</td><td>FastPhysGS: Accelerating Physics-based Dynamic 3DGS Simulation via Interior Completion and Adaptive Optimization</td><td>[2602.01723](http://arxiv.org/pdf/2602.01723)</td><td>◆ Extending 3D Gaussian Splatting (3DGS) to 4D physical simulation remains challenging.
◆ Based on the Material Point Method (MPM), existing methods either rely on manual parameter tuning or distill dynamics from video diffusion models, limiting the generalization and optimization efficiency.
◆ Recent attempts using LLMs/VLMs suffer from a text/image-to-3D perceptual gap, yielding unstable physics behavior.</td></tr>
<tr><td>2026-02-02</td><td>VRGaussianAvatar: Integrating 3D Gaussian Avatars into VR</td><td>[2602.01674](http://arxiv.org/pdf/2602.01674)</td><td>◆ We present VRGaussianAvatar, an integrated system that enables real-time full-body 3D Gaussian Splatting (3DGS) avatars in virtual reality using only head-mounted display (HMD) tracking signals.
◆ The system adopts a parallel pipeline with a VR Frontend and a GA Backend.
◆ The VR Frontend uses inverse kinematics to estimate full-body pose and streams the resulting pose along with stereo camera parameters to the backend.</td></tr>
<tr><td>2026-02-02</td><td>MarkCleaner: High-Fidelity Watermark Removal via Imperceptible Micro-Geometric Perturbation</td><td>[2602.01513](http://arxiv.org/pdf/2602.01513)</td><td>◆ Semantic watermarks exhibit strong robustness against conventional image-space attacks.
◆ In this work, we show that such robustness does not survive under micro-geometric perturbations: spatial displacements can remove watermarks by breaking the phase alignment.
◆ Motivated by this observation, we introduce MarkCleaner, a watermark removal framework that avoids semantic drift caused by regeneration-based watermark removal.</td></tr>
<tr><td>2026-02-01</td><td>Radioactive 3D Gaussian Ray Tracing for Tomographic Reconstruction</td><td>[2602.01057](http://arxiv.org/pdf/2602.01057)</td><td>◆ 3D Gaussian Splatting (3DGS) has recently emerged in computer vision as a promising rendering technique.
◆ By adapting the principles of Elliptical Weighted Average (EWA) splatting to a modern differentiable pipeline, 3DGS enables real-time, high-quality novel view synthesis.
◆ Building upon this, R2-Gaussian extended the 3DGS paradigm to tomographic reconstruction by rectifying integration bias, achieving state-of-the-art performance in computed tomography (CT).</td></tr>
<tr><td>2026-01-31</td><td>HPC: Hierarchical Point-based Latent Representation for Streaming Dynamic Gaussian Splatting Compression</td><td>[2602.00671](http://arxiv.org/pdf/2602.00671)</td><td>◆ While dynamic Gaussian Splatting has driven significant advances in free-viewpoint video, maintaining its rendering quality with a small memory footprint for efficient streaming transmission still presents an ongoing challenge.
◆ Existing streaming dynamic Gaussian Splatting compression methods typically leverage a latent representation to drive the neural network for predicting Gaussian residuals between frames.
◆ Their core latent representations can be categorized into structured grid-based and unstructured point-based paradigms.</td></tr>
<tr><td>2026-01-31</td><td>Tune-Your-Style: Intensity-tunable 3D Style Transfer with Gaussian Splatting</td><td>[2602.00618](http://arxiv.org/pdf/2602.00618)</td><td>◆ 3D style transfer refers to the artistic stylization of 3D assets based on reference style images.
◆ Recently, 3DGS-based stylization methods have drawn considerable attention, primarily due to their markedly enhanced training and rendering speeds.
◆ However, a vital challenge for 3D style transfer is to strike a balance between the content and the patterns and colors of the style.</td></tr>
<tr><td>2026-01-30</td><td>EAG-PT: Emission-Aware Gaussians and Path Tracing for Indoor Scene Reconstruction and Editing</td><td>[2601.23065](http://arxiv.org/pdf/2601.23065)</td><td>◆ Recent reconstruction methods based on radiance field such as NeRF and 3DGS reproduce indoor scenes with high visual fidelity, but break down under scene editing due to baked illumination and the lack of explicit light transport.
◆ In contrast, physically based inverse rendering relies on mesh representations and path tracing, which enforce correct light transport but place strong requirements on geometric fidelity, becoming a practical bottleneck for real indoor scenes.
◆ In this work, we propose Emission-Aware Gaussians and Path Tracing (EAG-PT), aiming for physically based light transport with a unified 2D Gaussian representation.</td></tr>
<tr><td>2026-01-30</td><td>Learning Geometrically-Grounded 3D Visual Representations for View-Generalizable Robotic Manipulation</td><td>[2601.22988](http://arxiv.org/pdf/2601.22988)</td><td>◆ Real-world robotic manipulation demands visuomotor policies capable of robust spatial scene understanding and strong generalization across diverse camera viewpoints.
◆ While recent advances in 3D-aware visual representations have shown promise, they still suffer from several key limitations, including reliance on multi-view observations during inference which is impractical in single-view restricted scenarios, incomplete scene modeling that fails to capture holistic and fine-grained geometric structures essential for precise manipulation, and lack of effective policy training strategies to retain and exploit the acquired 3D knowledge.
◆ To address these challenges, we present MethodName, a unified representation-policy learning framework for view-generalizable robotic manipulation.</td></tr>
<tr><td>2026-01-30</td><td>Diachronic Stereo Matching for Multi-Date Satellite Imagery</td><td>[2601.22808](http://arxiv.org/pdf/2601.22808)</td><td>◆ Recent advances in image-based satellite 3D reconstruction have progressed along two complementary directions.
◆ On one hand, multi-date approaches using NeRF or Gaussian-splatting jointly model appearance and geometry across many acquisitions, achieving accurate reconstructions on opportunistic imagery with numerous observations.
◆ On the other hand, classical stereoscopic reconstruction pipelines deliver robust and scalable results for simultaneous or quasi-simultaneous image pairs.</td></tr>
<tr><td>2026-01-30</td><td>PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</td><td>[2601.22046](http://arxiv.org/pdf/2601.22046)</td><td>◆ Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both.
◆ We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner.
◆ This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy.</td></tr>
<tr><td>2026-01-29</td><td>Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy</td><td>[2601.22026](http://arxiv.org/pdf/2601.22026)</td><td>◆ Volumetric medical imaging offers great potential for understanding complex pathologies.
◆ Yet, traditional 2D slices provide little support for interpreting spatial relationships, forcing users to mentally reconstruct anatomy into three dimensions.
◆ Direct volumetric path tracing and VR rendering can improve perception but are computationally expensive, while precomputed representations, like Gaussian Splatting, require planning ahead.</td></tr>
<tr><td>2026-01-29</td><td>Lightweight High-Fidelity Low-Bitrate Talking Face Compression for 3D Video Conference</td><td>[2601.21269](http://arxiv.org/pdf/2601.21269)</td><td>◆ The demand for immersive and interactive communication has driven advancements in 3D video conferencing, yet achieving high-fidelity 3D talking face representation at low bitrates remains a challenge.
◆ Traditional 2D video compression techniques fail to preserve fine-grained geometric and appearance details, while implicit neural rendering methods like NeRF suffer from prohibitive computational costs.
◆ To address these challenges, we propose a lightweight, high-fidelity, low-bitrate 3D talking face compression framework that integrates FLAME-based parametric modeling with 3DGS neural rendering.</td></tr>
<tr><td>2026-01-28</td><td>FreeFix: Boosting 3D Gaussian Splatting via Fine-Tuning-Free Diffusion Models</td><td>[2601.20857](http://arxiv.org/pdf/2601.20857)</td><td>◆ Neural Radiance Fields and 3D Gaussian Splatting have advanced novel view synthesis, yet still rely on dense inputs and often degrade at extrapolated views.
◆ Recent approaches leverage generative models, such as diffusion models, to provide additional supervision, but face a trade-off between generalization and fidelity: fine-tuning diffusion models for artifact removal improves fidelity but risks overfitting, while fine-tuning-free methods preserve generalization but often yield lower fidelity.
◆ We introduce FreeFix, a fine-tuning-free approach that pushes the boundary of this trade-off by enhancing extrapolated rendering with pretrained image diffusion models.</td></tr>
<tr><td>2026-01-28</td><td>GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering</td><td>[2601.20429](http://arxiv.org/pdf/2601.20429)</td><td>◆ 3D Gaussian Splatting has gained widespread adoption across diverse applications due to its exceptional rendering performance and visual quality.
◆ While most existing methods rely on rasterization to render Gaussians, recent research has started investigating ray tracing approaches to overcome the fundamental limitations inherent in rasterization.
◆ However, current Gaussian ray tracing methods suffer from inefficiencies such as bloated acceleration structures and redundant node traversals, which greatly degrade ray tracing performance.</td></tr>
<tr><td>2026-01-28</td><td>GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction</td><td>[2601.20331](http://arxiv.org/pdf/2601.20331)</td><td>◆ 3D Gaussian Splatting enables efficient optimization and high-quality rendering, yet accurate surface reconstruction remains challenging.
◆ Prior methods improve surface reconstruction by refining Gaussian depth estimates, either via multi-view geometric consistency or through monocular depth priors.
◆ However, multi-view constraints become unreliable under large geometric discrepancies, while monocular priors suffer from scale ambiguity and local inconsistency, ultimately leading to inaccurate Gaussian depth supervision.</td></tr>
<tr><td>2026-01-27</td><td>Graphical X Splatting (GraphiXS): A Graphical Model for 4D Gaussian Splatting under Uncertainty</td><td>[2601.19843](http://arxiv.org/pdf/2601.19843)</td><td>◆ We propose a new framework to systematically incorporate data uncertainty in Gaussian Splatting.
◆ Being the new paradigm of neural rendering, Gaussian Splatting has been investigated in many applications, with the main effort in extending its representation, improving its optimization process, and accelerating its speed.
◆ However, one orthogonal, much needed, but under-explored area is data uncertainty.</td></tr>
<tr><td>2026-01-27</td><td>WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater Reconstruction and Restoration</td><td>[2601.19753](http://arxiv.org/pdf/2601.19753)</td><td>◆ Underwater 3D reconstruction and appearance restoration are hindered by the complex optical properties of water, such as wavelength-dependent attenuation and scattering.
◆ Existing Neural Radiance Fields (NeRF)-based methods struggle with slow rendering speeds and suboptimal color restoration, while 3D Gaussian Splatting (3DGS) inherently lacks the capability to model complex volumetric scattering effects.
◆ To address these issues, we introduce WaterClear-GS, the first pure 3DGS-based framework that explicitly integrates underwater optical properties of local attenuation and scattering into Gaussian primitives, eliminating the need for an auxiliary medium network.</td></tr>
<tr><td>2026-01-27</td><td>DiffStyle3D: Consistent 3D Gaussian Stylization via Attention Optimization</td><td>[2601.19717](http://arxiv.org/pdf/2601.19717)</td><td>◆ 3D style transfer enables the creation of visually expressive 3D content, enriching the visual appearance of 3D scenes and objects.
◆ However, existing VGG- and CLIP-based methods struggle to model multi-view consistency within the model itself, while diffusion-based approaches can capture such consistency but rely on denoising directions, leading to unstable training.
◆ To address these limitations, we propose DiffStyle3D, a novel diffusion-based paradigm for 3DGS style transfer that directly optimizes in the latent space.</td></tr>
<tr><td>2026-01-28</td><td>Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction</td><td>[2601.19489](http://arxiv.org/pdf/2601.19489)</td><td>◆ We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge.
◆ The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate).
◆ To robustly handle these heterogeneous settings, we develop a two-stage solution.</td></tr>
<tr><td>2026-01-27</td><td>ClipGS-VR: Immersive and Interactive Cinematic Visualization of Volumetric Medical Data in Mobile Virtual Reality</td><td>[2601.19310](http://arxiv.org/pdf/2601.19310)</td><td>◆ High-fidelity cinematic medical visualization on mobile virtual reality (VR) remains challenging.
◆ Although ClipGS enables cross-sectional exploration via 3D Gaussian Splatting, it lacks arbitrary-angle slicing on consumer-grade VR headsets.
◆ To achieve real-time interactive performance, we introduce ClipGS-VR and restructure ClipGS&#x27;s neural inference into a consolidated dataset, integrating high-fidelity layers from multiple pre-computed slicing states into a unified rendering structure.</td></tr>
<tr><td>2026-01-27</td><td>TIGaussian: Disentangle Gaussians for Spatial-Awared Text-Image-3D Alignment</td><td>[2601.19247](http://arxiv.org/pdf/2601.19247)</td><td>◆ While visual-language models have profoundly linked features between texts and images, the incorporation of 3D modality data, such as point clouds and 3D Gaussians, further enables pretraining for 3D-related tasks, e.g., cross-modal retrieval, zero-shot classification, and scene recognition.
◆ As challenges remain in extracting 3D modal features and bridging the gap between different modalities, we propose TIGaussian, a framework that harnesses 3D Gaussian Splatting (3DGS) characteristics to strengthen cross-modality alignment through multi-branch 3DGS tokenizer and modality-specific 3D feature alignment strategies.
◆ Specifically, our multi-branch 3DGS tokenizer decouples the intrinsic properties of 3DGS structures into compact latent representations, enabling more generalizable feature extraction.</td></tr>
<tr><td>2026-01-27</td><td>UniMGS: Unifying Mesh and 3D Gaussian Splatting with Single-Pass Rasterization and Proxy-Based Deformation</td><td>[2601.19233](http://arxiv.org/pdf/2601.19233)</td><td>◆ Joint rendering and deformation of mesh and 3D Gaussian Splatting (3DGS) have significant value as both representa tions offer complementary advantages for graphics applica tions.
◆ However, due to differences in representation and ren dering pipelines, existing studies render meshes and 3DGS separately, making it difficult to accurately handle occlusions and transparency.
◆ Moreover, the deformed 3DGS still suffers from visual artifacts due to the sensitivity to the topology quality of the proxy mesh.</td></tr>
<tr><td>2026-01-27</td><td>Bridging Visual and Wireless Sensing: A Unified Radiation Field for 3D Radio Map Construction</td><td>[2601.19216](http://arxiv.org/pdf/2601.19216)</td><td>◆ The emerging applications of next-generation wireless networks (e.g., immersive 3D communication, low-altitude networks, and integrated sensing and communication) necessitate high-fidelity environmental intelligence.
◆ 3D radio maps have emerged as a critical tool for this purpose, enabling spectrum-aware planning and environment-aware sensing by bridging the gap between physical environments and electromagnetic signal propagation.
◆ However, constructing accurate 3D radio maps requires fine-grained 3D geometric information and a profound understanding of electromagnetic wave propagation.</td></tr>
<tr><td>2026-01-26</td><td>Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting</td><td>[2601.18633](http://arxiv.org/pdf/2601.18633)</td><td>◆ Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image.
◆ Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations.
◆ We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis.</td></tr>
<tr><td>2026-01-26</td><td>ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection</td><td>[2601.18629](http://arxiv.org/pdf/2601.18629)</td><td>◆ Real-to-Sim-to-Real technique is gaining increasing interest for robotic manipulation, as it can generate scalable data in simulation while having narrower sim-to-real gap.
◆ However, previous methods mainly focused on environment-level visual real-to-sim transfer, ignoring the transfer of interactions, which could be challenging and inefficient to obtain purely in simulation especially for contact-rich tasks.
◆ We propose ExoGS, a robot-free 4D Real-to-Sim-to-Real framework that captures both static environments and dynamic interactions in the real world and transfers them seamlessly to a simulated environment.</td></tr>
<tr><td>2026-01-26</td><td>LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction</td><td>[2601.18475](http://arxiv.org/pdf/2601.18475)</td><td>◆ Free-Viewpoint Video (FVV) reconstruction enables photorealistic and interactive 3D scene visualization; however, real-time streaming is often bottlenecked by sparse-view inputs, prohibitive training costs, and bandwidth constraints.
◆ While recent 3D Gaussian Splatting (3DGS) has advanced FVV due to its superior rendering speed, Streaming Free-Viewpoint Video (SFVV) introduces additional demands for rapid optimization, high-fidelity reconstruction under sparse constraints, and minimal storage footprints.
◆ To bridge this gap, we propose StreamLoD-GS, an LoD-based Gaussian Splatting framework designed specifically for SFVV.</td></tr>
<tr><td>2026-01-25</td><td>Geometry-Grounded Gaussian Splatting</td><td>[2601.17835](http://arxiv.org/pdf/2601.17835)</td><td>◆ Gaussian Splatting (GS) has demonstrated impressive quality and efficiency in novel view synthesis.
◆ However, shape extraction from Gaussian primitives remains an open problem.
◆ Due to inadequate geometry parameterization and approximation, existing shape reconstruction methods suffer from poor multi-view consistency and are sensitive to floaters.</td></tr>
<tr><td>2026-01-25</td><td>Advancing Structured Priors for Sparse-Voxel Surface Reconstruction</td><td>[2601.17720](http://arxiv.org/pdf/2601.17720)</td><td>◆ Reconstructing accurate surfaces with radiance fields has progressed rapidly, yet two promising explicit representations, 3D Gaussian Splatting and sparse-voxel rasterization, exhibit complementary strengths and weaknesses.
◆ 3D Gaussian Splatting converges quickly and carries useful geometric priors, but surface fidelity is limited by its point-like parameterization.
◆ Sparse-voxel rasterization provides continuous opacity fields and crisp geometry, but its typical uniform dense-grid initialization slows convergence and underutilizes scene structure.</td></tr>
<tr><td>2026-01-24</td><td>PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling</td><td>[2601.17354](http://arxiv.org/pdf/2601.17354)</td><td>◆ Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics.
◆ While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory.
◆ We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity.</td></tr>
<tr><td>2026-01-23</td><td>LGDWT-GS: Local and Global Discrete Wavelet-Regularized 3D Gaussian Splatting for Sparse-View Scene Reconstruction</td><td>[2601.17185](http://arxiv.org/pdf/2601.17185)</td><td>◆ We propose a new method for few-shot 3D reconstruction that integrates global and local frequency regularization to stabilize geometry and preserve fine details under sparse-view conditions, addressing a key limitation of existing 3D Gaussian Splatting (3DGS) models.
◆ We also introduce a new multispectral greenhouse dataset containing four spectral bands captured from diverse plant species under controlled conditions.
◆ Alongside the dataset, we release an open-source benchmarking package that defines standardized few-shot reconstruction protocols for evaluating 3DGS-based methods.</td></tr>
<tr><td>2026-01-26</td><td>A Step to Decouple Optimization in 3DGS</td><td>[2601.16736](http://arxiv.org/pdf/2601.16736)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis.
◆ As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such as synchronous weight updating and Adam with the adaptive gradient.
◆ However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization.</td></tr>
<tr><td>2026-01-23</td><td>ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction</td><td>[2601.16672](http://arxiv.org/pdf/2601.16672)</td><td>◆ High-quality 3D garment reconstruction plays a crucial role in mitigating the sim-to-real gap in applications such as digital avatars, virtual try-on and robotic manipulation.
◆ However, existing garment reconstruction methods typically rely on unstructured representations, such as 3D Gaussian Splats, struggling to provide accurate reconstructions of garment topology and sewing structures.
◆ As a result, the reconstructed outputs are often unsuitable for high-fidelity physical simulation.</td></tr>
<tr><td>2026-01-22</td><td>EVolSplat4D: Efficient Volume-based Gaussian Splatting for 4D Urban Scene Synthesis</td><td>[2601.15951](http://arxiv.org/pdf/2601.15951)</td><td>◆ Novel view synthesis (NVS) of static and dynamic urban scenes is essential for autonomous driving simulation, yet existing methods often struggle to balance reconstruction time with quality.
◆ While state-of-the-art neural radiance fields and 3D Gaussian Splatting approaches achieve photorealism, they often rely on time-consuming per-scene optimization.
◆ Conversely, emerging feed-forward methods frequently adopt per-pixel Gaussian representations, which lead to 3D inconsistencies when aggregating multi-view predictions in complex, dynamic environments.</td></tr>
<tr><td>2026-01-22</td><td>ThermoSplat: Cross-Modal 3D Gaussian Splatting with Feature Modulation and Geometry Decoupling</td><td>[2601.15897](http://arxiv.org/pdf/2601.15897)</td><td>◆ Multi-modal scene reconstruction integrating RGB and thermal infrared data is essential for robust environmental perception across diverse lighting and weather conditions.
◆ However, extending 3D Gaussian Splatting (3DGS) to multi-spectral scenarios remains challenging.
◆ Current approaches often struggle to fully leverage the complementary information of multi-modal data, typically relying on mechanisms that either tend to neglect cross-modal correlations or leverage shared representations that fail to adaptively handle the complex structural correlations and physical discrepancies between spectrums.</td></tr>
<tr><td>2026-01-22</td><td>LL-GaussianImage: Efficient Image Representation for Zero-shot Low-Light Enhancement with 2D Gaussian Splatting</td><td>[2601.15772](http://arxiv.org/pdf/2601.15772)</td><td>◆ 2D Gaussian Splatting (2DGS) is an emerging explicit scene representation method with significant potential for image compression due to high fidelity and high compression ratios.
◆ However, existing low-light enhancement algorithms operate predominantly within the pixel domain.
◆ Processing 2DGS-compressed images necessitates a cumbersome decompression-enhancement-recompression pipeline, which compromises efficiency and introduces secondary degradation.</td></tr>
<tr><td>2026-01-22</td><td>LL-GaussianMap: Zero-shot Low-Light Image Enhancement via 2D Gaussian Splatting Guided Gain Maps</td><td>[2601.15766](http://arxiv.org/pdf/2601.15766)</td><td>◆ Significant progress has been made in low-light image enhancement with respect to visual quality.
◆ However, most existing methods primarily operate in the pixel domain or rely on implicit feature representations.
◆ As a result, the intrinsic geometric structural priors of images are often neglected.</td></tr>
<tr><td>2026-01-21</td><td>SplatBus: A Gaussian Splatting Viewer Framework via GPU Interprocess Communication</td><td>[2601.15431](http://arxiv.org/pdf/2601.15431)</td><td>◆ Radiance field-based rendering methods have attracted significant interest from the computer vision and computer graphics communities.
◆ They enable high-fidelity rendering with complex real-world lighting effects, but at the cost of high rendering time.
◆ 3D Gaussian Splatting solves this issue with a rasterisation-based approach for real-time rendering, enabling applications such as autonomous driving, robotics, virtual reality, and extended reality.</td></tr>
<tr><td>2026-01-21</td><td>LuxRemix: Lighting Decomposition and Remixing for Indoor Scenes</td><td>[2601.15283](http://arxiv.org/pdf/2601.15283)</td><td>◆ We present a novel approach for interactive light editing in indoor scenes from a single multi-view scene capture.
◆ Our method leverages a generative image-based light decomposition model that factorizes complex indoor scene illumination into its constituent light sources.
◆ This factorization enables independent manipulation of individual light sources, specifically allowing control over their state (on/off), chromaticity, and intensity.</td></tr>
<tr><td>2026-01-21</td><td>ScenDi: 3D-to-2D Scene Diffusion Cascades for Urban Generation</td><td>[2601.15221](http://arxiv.org/pdf/2601.15221)</td><td>◆ Recent advancements in 3D object generation using diffusion models have achieved remarkable success, but generating realistic 3D urban scenes remains challenging.
◆ Existing methods relying solely on 3D diffusion models tend to suffer a degradation in appearance details, while those utilizing only 2D diffusion models typically compromise camera controllability.
◆ To overcome this limitation, we propose ScenDi, a method for urban scene generation that integrates both 3D and 2D diffusion models.</td></tr>
<tr><td>2026-01-21</td><td>POTR: Post-Training 3DGS Compression</td><td>[2601.14821](http://arxiv.org/pdf/2601.14821)</td><td>◆ 3D Gaussian Splatting (3DGS) has recently emerged as a promising contender to Neural Radiance Fields (NeRF) in 3D scene reconstruction and real-time novel view synthesis.
◆ 3DGS outperforms NeRF in training and inference speed but has substantially higher storage requirements.
◆ To remedy this downside, we propose POTR, a post-training 3DGS codec built on two novel techniques.</td></tr>
<tr><td>2026-01-22</td><td>Structured Image-based Coding for Efficient Gaussian Splatting Compression</td><td>[2601.14510](http://arxiv.org/pdf/2601.14510)</td><td>◆ Gaussian Splatting (GS) has recently emerged as a state-of-the-art representation for radiance fields, combining real-time rendering with high visual fidelity.
◆ However, GS models require storing millions of parameters, leading to large file sizes that impair their use in practical multimedia systems.
◆ To address this limitation, this paper introduces GS Image-based Compression (GSICO), a novel GS codec that efficiently compresses pre-trained GS models while preserving perceptual fidelity.</td></tr>
<tr><td>2026-01-20</td><td>Rig-Aware 3D Reconstruction of Vehicle Undercarriages using Gaussian Splatting</td><td>[2601.14208](http://arxiv.org/pdf/2601.14208)</td><td>◆ Inspecting the undercarriage of used vehicles is a labor-intensive task that requires inspectors to crouch or crawl underneath each vehicle to thoroughly examine it.
◆ Additionally, online buyers rarely see undercarriage photos.
◆ We present an end-to-end pipeline that utilizes a three-camera rig to capture videos of the undercarriage as the vehicle drives over it, and produces an interactive 3D model of the undercarriage.</td></tr>
<tr><td>2026-01-20</td><td>One-Shot Refiner: Boosting Feed-forward Novel View Synthesis via One-Step Diffusion</td><td>[2601.14161](http://arxiv.org/pdf/2601.14161)</td><td>◆ We present a novel framework for high-fidelity novel view synthesis (NVS) from sparse images, addressing key limitations in recent feed-forward 3D Gaussian Splatting (3DGS) methods built on Vision Transformer (ViT) backbones.
◆ While ViT-based pipelines offer strong geometric priors, they are often constrained by low-resolution inputs due to computational costs.
◆ Moreover, existing generative enhancement methods tend to be 3D-agnostic, resulting in inconsistent structures across views, especially in unseen regions.</td></tr>
<tr><td>2026-01-20</td><td>ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins</td><td>[2601.13706](http://arxiv.org/pdf/2601.13706)</td><td>◆ High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP).
◆ Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints.
◆ We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction.</td></tr>
<tr><td>2026-01-19</td><td>GaussExplorer: 3D Gaussian Splatting for Embodied Exploration and Reasoning</td><td>[2601.13132](http://arxiv.org/pdf/2601.13132)</td><td>◆ We present GaussExplorer, a framework for embodied exploration and reasoning built on 3D Gaussian Splatting (3DGS).
◆ While prior approaches to language-embedded 3DGS have made meaningful progress in aligning simple text queries with Gaussian embeddings, they are generally optimized for relatively simple queries and struggle to interpret more complex, compositional language queries.
◆ Alternative studies based on object-centric RGB-D structured memories provide spatial grounding but are constrained by pre-fixed viewpoints.</td></tr>
<tr><td>2026-01-19</td><td>TreeDGS: Aerial Gaussian Splatting for Distant DBH Measurement</td><td>[2601.12823](http://arxiv.org/pdf/2601.12823)</td><td>◆ Aerial remote sensing enables efficient large-area surveying, but accurate direct object-level measurement remains difficult in complex natural scenes.
◆ Recent advancements in 3D vision, particularly learned radiance-field representations such as NeRF and 3D Gaussian Splatting, have begun to raise the ceiling on reconstruction fidelity and densifiable geometry from posed imagery.
◆ Nevertheless, direct aerial measurement of important natural attributes such as tree diameter at breast height (DBH) remains challenging.</td></tr>
<tr><td>2026-01-19</td><td>CSGaussian: Progressive Rate-Distortion Compression and Segmentation for 3D Gaussian Splatting</td><td>[2601.12814](http://arxiv.org/pdf/2601.12814)</td><td>◆ We present the first unified framework for rate-distortion-optimized compression and segmentation of 3D Gaussian Splatting (3DGS).
◆ While 3DGS has proven effective for both real-time rendering and semantic scene understanding, prior works have largely treated these tasks independently, leaving their joint consideration unexplored.
◆ Inspired by recent advances in rate-distortion-optimized 3DGS compression, this work integrates semantic learning into the compression pipeline to support decoder-side applications--such as scene editing and manipulation--that extend beyond traditional scene reconstruction and view synthesis.</td></tr>
<tr><td>2026-01-19</td><td>KaoLRM: Repurposing Pre-trained Large Reconstruction Models for Parametric 3D Face Reconstruction</td><td>[2601.12736](http://arxiv.org/pdf/2601.12736)</td><td>◆ We propose KaoLRM to re-target the learned prior of the Large Reconstruction Model (LRM) for parametric 3D face reconstruction from single-view images.
◆ Parametric 3D Morphable Models (3DMMs) have been widely used for facial reconstruction due to their compact and interpretable parameterization, yet existing 3DMM regressors often exhibit poor consistency across varying viewpoints.
◆ To address this, we harness the pre-trained 3D prior of LRM and incorporate FLAME-based 2D Gaussian Splatting into LRM&#x27;s rendering pipeline.</td></tr>
<tr><td>2026-01-19</td><td>GaussianTrimmer: Online Trimming Boundaries for 3DGS Segmentation</td><td>[2601.12683](http://arxiv.org/pdf/2601.12683)</td><td>◆ With the widespread application of 3D Gaussians in 3D scene representation, 3D scene segmentation methods based on 3D Gaussians have also gradually emerged.
◆ However, existing 3D Gaussian segmentation methods basically segment on the basis of Gaussian primitives.
◆ Due to the large variation range of the scale of 3D Gaussians, large-sized Gaussians that often span the foreground and background lead to jagged boundaries of segmented objects.</td></tr>
<tr><td>2026-01-17</td><td>Active Semantic Mapping of Horticultural Environments Using Gaussian Splatting</td><td>[2601.12122](http://arxiv.org/pdf/2601.12122)</td><td>◆ Semantic reconstruction of agricultural scenes plays a vital role in tasks such as phenotyping and yield estimation.
◆ However, traditional approaches that rely on manual scanning or fixed camera setups remain a major bottleneck in this process.
◆ In this work, we propose an active 3D reconstruction framework for horticultural environments using a mobile manipulator.</td></tr>
<tr><td>2026-01-17</td><td>DIAMOND-SSS: Diffusion-Augmented Multi-View Optimization for Data-efficient SubSurface Scattering</td><td>[2601.12020](http://arxiv.org/pdf/2601.12020)</td><td>◆ Subsurface scattering (SSS) gives translucent materials -- such as wax, jade, marble, and skin -- their characteristic soft shadows, color bleeding, and diffuse glow.
◆ Modeling these effects in neural rendering remains challenging due to complex light transport and the need for densely captured multi-view, multi-light datasets (often more than 100 views and 112 OLATs).
◆ We present DIAMOND-SSS, a data-efficient framework for high-fidelity translucent reconstruction from extremely sparse supervision -- even as few as ten images.</td></tr>
<tr><td>2026-01-15</td><td>RSATalker: Realistic Socially-Aware Talking Head Generation for Multi-Turn Conversation</td><td>[2601.10606](http://arxiv.org/pdf/2601.10606)</td><td>◆ Talking head generation is increasingly important in virtual reality (VR), especially for social scenarios involving multi-turn conversation.
◆ Existing approaches face notable limitations: mesh-based 3D methods can model dual-person dialogue but lack realistic textures, while large-model-based 2D methods produce natural appearances but incur prohibitive computational costs.
◆ Recently, 3D Gaussian Splatting (3DGS) based methods achieve efficient and realistic rendering but remain speaker-only and ignore social relationships.</td></tr>
<tr><td>2026-01-15</td><td>Thinking Like Van Gogh: Structure-Aware Style Transfer via Flow-Guided 3D Gaussian Splatting</td><td>[2601.10075](http://arxiv.org/pdf/2601.10075)</td><td>◆ In 1888, Vincent van Gogh wrote, &quot;I am seeking exaggeration in the essential.&quot; This principle, amplifying structural form while suppressing photographic detail, lies at the core of Post-Impressionist art.
◆ However, most existing 3D style transfer methods invert this philosophy, treating geometry as a rigid substrate for surface-level texture projection.
◆ To authentically reproduce Post-Impressionist stylization, geometric abstraction must be embraced as the primary vehicle of expression.</td></tr>
<tr><td>2026-01-14</td><td>Variable Basis Mapping for Real-Time Volumetric Visualization</td><td>[2601.09417](http://arxiv.org/pdf/2601.09417)</td><td>◆ Real-time visualization of large-scale volumetric data remains challenging, as direct volume rendering and voxel-based methods suffer from prohibitively high computational cost.
◆ We propose Variable Basis Mapping (VBM), a framework that transforms volumetric fields into 3D Gaussian Splatting (3DGS) representations through wavelet-domain analysis.
◆ First, we precompute a compact Wavelet-to-Gaussian Transition Bank that provides optimal Gaussian surrogates for canonical wavelet atoms across multiple scales.</td></tr>
<tr><td>2026-01-14</td><td>TIDI-GS: Floater Suppression in 3D Gaussian Splatting for Enhanced Indoor Scene Fidelity</td><td>[2601.09291](http://arxiv.org/pdf/2601.09291)</td><td>◆ 3D Gaussian Splatting (3DGS) is a technique to create high-quality, real-time 3D scenes from images.
◆ This method often produces visual artifacts known as floaters--nearly transparent, disconnected elements that drift in space away from the actual surface.
◆ This geometric inaccuracy undermines the reliability of these models for practical applications, which is critical.</td></tr>
<tr><td>2026-01-14</td><td>GaussianFluent: Gaussian Simulation for Dynamic Scenes with Mixed Materials</td><td>[2601.09265](http://arxiv.org/pdf/2601.09265)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a prominent 3D representation for high-fidelity and real-time rendering.
◆ Prior work has coupled physics simulation with Gaussians, but predominantly targets soft, deformable materials, leaving brittle fracture largely unresolved.
◆ This stems from two key obstacles: the lack of volumetric interiors with coherent textures in GS representation, and the absence of fracture-aware simulation methods for Gaussians.</td></tr>
<tr><td>2026-01-14</td><td>A$^2$TG: Adaptive Anisotropic Textured Gaussians for Efficient 3D Scene Representation</td><td>[2601.09243](http://arxiv.org/pdf/2601.09243)</td><td>◆ Gaussian Splatting has emerged as a powerful representation for high-quality, real-time 3D scene rendering.
◆ While recent works extend Gaussians with learnable textures to enrich visual appearance, existing approaches allocate a fixed square texture per primitive, leading to inefficient memory usage and limited adaptability to scene variability.
◆ In this paper, we introduce adaptive anisotropic textured Gaussians (A$^2$TG), a novel representation that generalizes textured Gaussians by equipping each primitive with an anisotropic texture.</td></tr>
<tr><td>2026-01-12</td><td>3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing</td><td>[2601.07963](http://arxiv.org/pdf/2601.07963)</td><td>◆ The transformative potential of 3D content creation has been progressively unlocked through advancements in generative models.
◆ Recently, intuitive drag editing with geometric changes has attracted significant attention in 2D editing yet remains challenging for 3D scenes.
◆ In this paper, we introduce 3DGS-Drag -- a point-based 3D editing framework that provides efficient, intuitive drag manipulation of real 3D scenes.</td></tr>
<tr><td>2026-01-13</td><td>ViewMorpher3D: A 3D-aware Diffusion Framework for Multi-Camera Novel View Synthesis in Autonomous Driving</td><td>[2601.07540](http://arxiv.org/pdf/2601.07540)</td><td>◆ Autonomous driving systems rely heavily on multi-view images to ensure accurate perception and robust decision-making.
◆ To effectively develop and evaluate perception stacks and planning algorithms, realistic closed-loop simulators are indispensable.
◆ While 3D reconstruction techniques such as Gaussian Splatting offer promising avenues for simulator construction, the rendered novel views often exhibit artifacts, particularly in extrapolated perspectives or when available observations are sparse.</td></tr>
<tr><td>2026-01-12</td><td>Mon3tr: Monocular 3D Telepresence with Pre-built Gaussian Avatars as Amortization</td><td>[2601.07518](http://arxiv.org/pdf/2601.07518)</td><td>◆ Immersive telepresence aims to transform human interaction in AR/VR applications by enabling lifelike full-body holographic representations for enhanced remote collaboration.
◆ However, existing systems rely on hardware-intensive multi-camera setups and demand high bandwidth for volumetric streaming, limiting their real-time performance on mobile devices.
◆ To overcome these challenges, we propose Mon3tr, a novel Monocular 3D telepresence framework that integrates 3D Gaussian splatting (3DGS) based parametric human modeling into telepresence for the first time.</td></tr>
<tr><td>2026-01-12</td><td>R3-RECON: Radiance-Field-Free Active Reconstruction via Renderability</td><td>[2601.07484](http://arxiv.org/pdf/2601.07484)</td><td>◆ In active reconstruction, an embodied agent must decide where to look next to efficiently acquire views that support high-quality novel-view rendering.
◆ Recent work on active view planning for neural rendering largely derives next-best-view (NBV) criteria by backpropagating through radiance fields or estimating information entropy over 3D Gaussian primitives.
◆ While effective, these strategies tightly couple view selection to heavy, representation-specific mechanisms and fail to account for the computational and resource constraints required for lightweight online deployment.</td></tr>
<tr><td>2026-01-11</td><td>SARA: Scene-Aware Reconstruction Accelerator</td><td>[2601.06831](http://arxiv.org/pdf/2601.06831)</td><td>◆ We present SARA (Scene-Aware Reconstruction Accelerator), a geometry-driven pair selection module for Structure-from-Motion (SfM).
◆ Unlike conventional pipelines that select pairs based on visual similarity alone, SARA introduces geometry-first pair selection by scoring reconstruction informativeness - the product of overlap and parallax - before expensive matching.
◆ A lightweight pre-matching stage uses mutual nearest neighbors and RANSAC to estimate these cues, then constructs an Information-Weighted Spanning Tree (IWST) augmented with targeted edges for loop closure, long-baseline anchors, and weak-view reinforcement.</td></tr>
<tr><td>2026-01-10</td><td>SRFlow: A Dataset and Regularization Model for High-Resolution Facial Optical Flow via Splatting Rasterization</td><td>[2601.06479](http://arxiv.org/pdf/2601.06479)</td><td>◆ Facial optical flow supports a wide range of tasks in facial motion analysis.
◆ However, the lack of high-resolution facial optical flow datasets has hindered progress in this area.
◆ In this paper, we introduce Splatting Rasterization Flow (SRFlow), a high-resolution facial optical flow dataset, and Splatting Rasterization Guided FlowNet (SRFlowNet), a facial optical flow model with tailored regularization losses.</td></tr>
<tr><td>2026-01-09</td><td>NAS-GS: Noise-Aware Sonar Gaussian Splatting</td><td>[2601.06285](http://arxiv.org/pdf/2601.06285)</td><td>◆ Underwater sonar imaging plays a crucial role in various applications, including autonomous navigation in murky water, marine archaeology, and environmental monitoring.
◆ However, the unique characteristics of sonar images, such as complex noise patterns and the lack of elevation information, pose significant challenges for 3D reconstruction and novel view synthesis.
◆ In this paper, we present NAS-GS, a novel Noise-Aware Sonar Gaussian Splatting framework specifically designed to address these challenges.</td></tr>
<tr><td>2026-01-09</td><td>LayerGS: Decomposition and Inpainting of Layered 3D Human Avatars via 2D Gaussian Splatting</td><td>[2601.05853](http://arxiv.org/pdf/2601.05853)</td><td>◆ We propose a novel framework for decomposing arbitrarily posed humans into animatable multi-layered 3D human avatars, separating the body and garments.
◆ Conventional single-layer reconstruction methods lock clothing to one identity, while prior multi-layer approaches struggle with occluded regions.
◆ We overcome both limitations by encoding each layer as a set of 2D Gaussians for accurate geometry and photorealistic rendering, and inpainting hidden regions with a pretrained 2D diffusion model via score-distillation sampling (SDS).</td></tr>
<tr><td>2026-01-09</td><td>FeatureSLAM: Feature-enriched 3D gaussian splatting SLAM in real time</td><td>[2601.05738](http://arxiv.org/pdf/2601.05738)</td><td>◆ We present a real-time tracking SLAM system that unifies efficient camera tracking with photorealistic feature-enriched mapping using 3D Gaussian Splatting (3DGS).
◆ Our main contribution is integrating dense feature rasterization into the novel-view synthesis, aligned with a visual foundation model.
◆ This yields strong semantics, going beyond basic RGB-D input, aiding both tracking and mapping accuracy.</td></tr>
<tr><td>2026-01-09</td><td>GS-DMSR: Dynamic Sensitive Multi-scale Manifold Enhancement for Accelerated High-Quality 3D Gaussian Splatting</td><td>[2601.05584](http://arxiv.org/pdf/2601.05584)</td><td>◆ In the field of 3D dynamic scene reconstruction, how to balance model convergence rate and rendering quality has long been a critical challenge that urgently needs to be addressed, particularly in high-precision modeling of scenes with complex dynamic motions.
◆ To tackle this issue, this study proposes the GS-DMSR method.
◆ By quantitatively analyzing the dynamic evolution process of Gaussian attributes, this mechanism achieves adaptive gradient focusing, enabling it to dynamically identify significant differences in the motion states of Gaussian models.</td></tr>
<tr><td>2026-01-09</td><td>GaussianSwap: Animatable Video Face Swapping with 3D Gaussian Splatting</td><td>[2601.05511](http://arxiv.org/pdf/2601.05511)</td><td>◆ We introduce GaussianSwap, a novel video face swapping framework that constructs a 3D Gaussian Splatting based face avatar from a target video while transferring identity from a source image to the avatar.
◆ Conventional video swapping frameworks are limited to generating facial representations in pixel-based formats.
◆ The resulting swapped faces exist merely as a set of unstructured pixels without any capacity for animation or interactive manipulation.</td></tr>
<tr><td>2026-01-08</td><td>Sketch&amp;Patch++: Efficient Structure-Aware 3D Gaussian Representation</td><td>[2601.05394](http://arxiv.org/pdf/2601.05394)</td><td>◆ We observe that Gaussians exhibit distinct roles and characteristics analogous to traditional artistic techniques -- like how artists first sketch outlines before filling in broader areas with color, some Gaussians capture high-frequency features such as edges and contours, while others represent broader, smoother regions analogous to brush strokes that add volume and depth.
◆ Based on this observation, we propose a hybrid representation that categorizes Gaussians into (i) Sketch Gaussians, which represent high-frequency, boundary-defining features, and (ii) Patch Gaussians, which cover low-frequency, smooth regions.
◆ This semantic separation naturally enables layered progressive streaming, where the compact Sketch Gaussians establish the structural skeleton before Patch Gaussians incrementally refine volumetric detail.</td></tr>
<tr><td>2026-01-08</td><td>ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting</td><td>[2601.04754](http://arxiv.org/pdf/2601.04754)</td><td>◆ We present ProFuse, an efficient context-aware framework for open-vocabulary 3D scene understanding with 3D Gaussian Splatting (3DGS).
◆ The pipeline enhances cross-view consistency and intra-mask cohesion within a direct registration setup, adding minimal overhead and requiring no render-supervised fine-tuning.
◆ Instead of relying on a pretrained 3DGS scene, we introduce a dense correspondence-guided pre-registration phase that initializes Gaussians with accurate geometry while jointly constructing 3D Context Proposals via cross-view clustering.</td></tr>
<tr><td>2026-01-06</td><td>RelightAnyone: A Generalized Relightable 3D Gaussian Head Model</td><td>[2601.03357](http://arxiv.org/pdf/2601.03357)</td><td>◆ 3D Gaussian Splatting (3DGS) has become a standard approach to reconstruct and render photorealistic 3D head avatars.
◆ A major challenge is to relight the avatars to match any scene illumination.
◆ For high quality relighting, existing methods require subjects to be captured under complex time-multiplexed illumination, such as one-light-at-a-time (OLAT).</td></tr>
<tr><td>2026-01-06</td><td>CaricatureGS: Exaggerating 3D Gaussian Splatting Faces With Gaussian Curvature</td><td>[2601.03319](http://arxiv.org/pdf/2601.03319)</td><td>◆ A photorealistic and controllable 3D caricaturization framework for faces is introduced.
◆ We start with an intrinsic Gaussian curvature-based surface exaggeration technique, which, when coupled with texture, tends to produce over-smoothed renders.
◆ To address this, we resort to 3D Gaussian Splatting (3DGS), which has recently been shown to produce realistic free-viewpoint avatars.</td></tr>
<tr><td>2026-01-06</td><td>A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting</td><td>[2601.03200](http://arxiv.org/pdf/2601.03200)</td><td>◆ Developing high-fidelity, interactive digital twins is crucial for enabling closed-loop motion planning and reliable real-world robot execution, which are essential to advancing sim-to-real transfer.
◆ However, existing approaches often suffer from slow reconstruction, limited visual fidelity, and difficulties in converting photorealistic models into planning-ready collision geometry.
◆ We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs.</td></tr>
<tr><td>2026-01-05</td><td>Joint Semantic and Rendering Enhancements in 3D Gaussian Modeling with Anisotropic Local Encoding</td><td>[2601.02339](http://arxiv.org/pdf/2601.02339)</td><td>◆ Recent works propose extending 3DGS with semantic feature vectors for simultaneous semantic segmentation and image rendering.
◆ However, these methods often treat the semantic and rendering branches separately, relying solely on 2D supervision while ignoring the 3D Gaussian geometry.
◆ Moreover, current adaptive strategies adapt the Gaussian set depending solely on rendering gradients, which can be insufficient in subtle or textureless regions.</td></tr>
<tr><td>2026-01-05</td><td>360-GeoGS: Geometrically Consistent Feed-Forward 3D Gaussian Splatting Reconstruction for 360 Images</td><td>[2601.02102](http://arxiv.org/pdf/2601.02102)</td><td>◆ 3D scene reconstruction is fundamental for spatial intelligence applications such as AR, robotics, and digital twins.
◆ Traditional multi-view stereo struggles with sparse viewpoints or low-texture regions, while neural rendering approaches, though capable of producing high-quality results, require per-scene optimization and lack real-time efficiency.
◆ Explicit 3D Gaussian Splatting (3DGS) enables efficient rendering, but most feed-forward variants focus on visual quality rather than geometric consistency, limiting accurate surface reconstruction and overall reliability in spatial perception tasks.</td></tr>
<tr><td>2026-01-04</td><td>Animated 3DGS Avatars in Diverse Scenes with Consistent Lighting and Shadows</td><td>[2601.01660](http://arxiv.org/pdf/2601.01660)</td><td>◆ We present a method for consistent lighting and shadows when animated 3D Gaussian Splatting (3DGS) avatars interact with 3DGS scenes or with dynamic objects inserted into otherwise static scenes.
◆ Our key contribution is Deep Gaussian Shadow Maps (DGSM), a modern analogue of the classical shadow mapping algorithm tailored to the volumetric 3DGS representation.
◆ Building on the classic deep shadow mapping idea, we show that 3DGS admits closed form light accumulation along light rays, enabling volumetric shadow computation without meshing.</td></tr>
<tr><td>2026-01-04</td><td>ParkGaussian: Surround-view 3D Gaussian Splatting for Autonomous Parking</td><td>[2601.01386](http://arxiv.org/pdf/2601.01386)</td><td>◆ Parking is a critical task for autonomous driving systems (ADS), with unique challenges in crowded parking slots and GPS-denied environments.
◆ However, existing works focus on 2D parking slot perception, mapping, and localization, 3D reconstruction remains underexplored, which is crucial for capturing complex spatial geometry in parking scenarios.
◆ Naively improving the visual quality of reconstructed parking scenes does not directly benefit autonomous parking, as the key entry point for parking is the slots perception module.</td></tr>
<tr><td>2026-01-04</td><td>ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery</td><td>[2601.00939](http://arxiv.org/pdf/2601.00939)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a novel paradigm for 3D reconstruction from satellite imagery.
◆ However, in multi-temporal satellite images, prevalent shadows exhibit significant inconsistencies due to varying illumination conditions.
◆ To address this, we propose ShadowGS, a novel framework based on 3DGS.</td></tr>
<tr><td>2026-01-01</td><td>Clean-GS: Semantic Mask-Guided Pruning for 3D Gaussian Splatting</td><td>[2601.00913](http://arxiv.org/pdf/2601.00913)</td><td>◆ 3D Gaussian Splatting produces high-quality scene reconstructions but generates hundreds of thousands of spurious Gaussians (floaters) scattered throughout the environment.
◆ These artifacts obscure objects of interest and inflate model sizes, hindering deployment in bandwidth-constrained applications.
◆ We present Clean-GS, a method for removing background clutter and floaters from 3DGS reconstructions using sparse semantic masks.</td></tr>
<tr><td>2025-12-31</td><td>PhysTalk: Language-driven Real-time Physics in 3D Gaussian Scenes</td><td>[2512.24986](http://arxiv.org/pdf/2512.24986)</td><td>◆ Realistic visual simulations are omnipresent, yet their creation requires computing time, rendering, and expert animation knowledge.
◆ Open-vocabulary visual effects generation from text inputs emerges as a promising solution that can unlock immense creative potential.
◆ However, current pipelines lack both physical realism and effective language interfaces, requiring slow offline optimization.</td></tr>
<tr><td>2025-12-31</td><td>UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning</td><td>[2512.24763](http://arxiv.org/pdf/2512.24763)</td><td>◆ 3D Gaussian Splatting (3DGS) and Neural Radiance Fields (NeRF) have advanced novel-view synthesis.
◆ Recent methods extend multi-view 2D segmentation to 3D, enabling instance/semantic segmentation for better scene understanding.
◆ A key challenge is the inconsistency of 2D instance labels across views, leading to poor 3D predictions.</td></tr>
<tr><td>2025-12-31</td><td>Splatwizard: A Benchmark Toolkit for 3D Gaussian Splatting Compression</td><td>[2512.24742](http://arxiv.org/pdf/2512.24742)</td><td>◆ The recent advent of 3D Gaussian Splatting (3DGS) has marked a significant breakthrough in real-time novel view synthesis.
◆ However, the rapid proliferation of 3DGS-based algorithms has created a pressing need for standardized and comprehensive evaluation tools, especially for compression task.
◆ Existing benchmarks often lack the specific metrics necessary to holistically assess the unique characteristics of different methods, such as rendering speed, rate distortion trade-offs memory efficiency, and geometric accuracy.</td></tr>
<tr><td>2025-12-30</td><td>Improved 3D Gaussian Splatting of Unknown Spacecraft Structure Using Space Environment Illumination Knowledge</td><td>[2512.23998](http://arxiv.org/pdf/2512.23998)</td><td>◆ This work presents a novel pipeline to recover the 3D structure of an unknown target spacecraft from a sequence of images captured during Rendezvous and Proximity Operations (RPO) in space.
◆ The target&#x27;s geometry and appearance are represented as a 3D Gaussian Splatting (3DGS) model.
◆ However, learning 3DGS requires static scenes, an assumption in contrast to dynamic lighting conditions encountered in spaceborne imagery.</td></tr>
<tr><td>2025-12-28</td><td>3D Scene Change Modeling With Consistent Multi-View Aggregation</td><td>[2512.22830](http://arxiv.org/pdf/2512.22830)</td><td>◆ Change detection plays a vital role in scene monitoring, exploration, and continual reconstruction.
◆ Existing 3D change detection methods often exhibit spatial inconsistency in the detected changes and fail to explicitly separate pre- and post-change states.
◆ To address these limitations, we propose SCaR-3D, a novel 3D scene change detection framework that identifies object-level changes from a dense-view pre-change image sequence and sparse-view post-change images.</td></tr>
<tr><td>2025-12-23</td><td>Nebula: Enable City-Scale 3D Gaussian Splatting in Virtual Reality via Collaborative Rendering and Accelerated Stereo Rasterization</td><td>[2512.20495](http://arxiv.org/pdf/2512.20495)</td><td>◆ 3D Gaussian splatting (3DGS) has drawn significant attention in the architectural community recently.
◆ However, current architectural designs often overlook the 3DGS scalability, making them fragile for extremely large-scale 3DGS.
◆ Meanwhile, the VR bandwidth requirement makes it impossible to deliver high-fidelity and smooth VR content from the cloud.</td></tr>
<tr><td>2025-12-23</td><td>Enhancing annotations for 5D apple pose estimation through 3D Gaussian Splatting (3DGS)</td><td>[2512.20148](http://arxiv.org/pdf/2512.20148)</td><td>◆ Automating tasks in orchards is challenging because of the large amount of variation in the environment and occlusions.
◆ One of the challenges is apple pose estimation, where key points, such as the calyx, are often occluded.
◆ Recently developed pose estimation methods no longer rely on these key points, but still require them for annotations, making annotating challenging and time-consuming.</td></tr>
<tr><td>2025-12-22</td><td>WorldWarp: Propagating 3D Geometry with Asynchronous Video Diffusion</td><td>[2512.19678](http://arxiv.org/pdf/2512.19678)</td><td>◆ Generating long-range, geometrically consistent video presents a fundamental dilemma: while consistency demands strict adherence to 3D geometry in pixel space, state-of-the-art generative models operate most effectively in a camera-conditioned latent space.
◆ This disconnect causes current methods to struggle with occluded areas and complex camera trajectories.
◆ To bridge this gap, we propose WorldWarp, a framework that couples a 3D structural anchor with a 2D generative refiner.</td></tr>
<tr><td>2025-12-22</td><td>TwinAligner: Visual-Dynamic Alignment Empowers Physics-aware Real2Sim2Real for Robotic Manipulation</td><td>[2512.19390](http://arxiv.org/pdf/2512.19390)</td><td>◆ The robotics field is evolving towards data-driven, end-to-end learning, inspired by multimodal large models.
◆ However, reliance on expensive real-world data limits progress.
◆ Simulators offer cost-effective alternatives, but the gap between simulation and reality challenges effective policy transfer.</td></tr>
<tr><td>2025-12-21</td><td>EcoSplat: Efficiency-controllable Feed-forward 3D Gaussian Splatting from Multi-view Images</td><td>[2512.18692](http://arxiv.org/pdf/2512.18692)</td><td>◆ Feed-forward 3D Gaussian Splatting (3DGS) enables efficient one-pass scene reconstruction, providing 3D representations for novel view synthesis without per-scene optimization.
◆ However, existing methods typically predict pixel-aligned primitives per-view, producing an excessive number of primitives in dense-view settings and offering no explicit control over the number of predicted Gaussians.
◆ To address this, we propose EcoSplat, the first efficiency-controllable feed-forward 3DGS framework that adaptively predicts the 3D representation for any given target primitive count at inference time.</td></tr>
<tr><td>2025-12-21</td><td>Geometric-Photometric Event-based 3D Gaussian Ray Tracing</td><td>[2512.18640](http://arxiv.org/pdf/2512.18640)</td><td>◆ Event cameras offer a high temporal resolution over traditional frame-based cameras, which makes them suitable for motion and structure estimation.
◆ However, it has been unclear how event-based 3D Gaussian Splatting (3DGS) approaches could leverage fine-grained temporal information of sparse events.
◆ This work proposes a framework to address the trade-off between accuracy and temporal resolution in event-based 3DGS.</td></tr>
<tr><td>2025-12-22</td><td>Chorus: Multi-Teacher Pretraining for Holistic 3D Gaussian Scene Encoding</td><td>[2512.17817](http://arxiv.org/pdf/2512.17817)</td><td>◆ While 3DGS has emerged as a high-fidelity scene representation, encoding rich, general-purpose features directly from its primitives remains under-explored.
◆ We address this gap by introducing Chorus, a multi-teacher pretraining framework that learns a holistic feed-forward 3D Gaussian Splatting (3DGS) scene encoder by distilling complementary signals from 2D foundation models.
◆ Chorus employs a shared 3D encoder and teacher-specific projectors to learn from language-aligned, generalist, and object-aware teachers, encouraging a shared embedding space that captures signals from high-level semantics to fine-grained structure.</td></tr>
<tr><td>2025-12-18</td><td>Animate Any Character in Any World</td><td>[2512.17796](http://arxiv.org/pdf/2512.17796)</td><td>◆ Recent advances in world models have greatly enhanced interactive environment simulation.
◆ Existing methods mainly fall into two categories: (1) static world generation models, which construct 3D environments without active agents, and (2) controllable-entity models, which allow a single entity to perform limited actions in an otherwise uncontrollable environment.
◆ In this work, we introduce AniX, leveraging the realism and structural grounding of static world generation while extending controllable-entity models to support user-specified characters capable of performing open-ended actions.</td></tr>
<tr><td>2025-12-19</td><td>Flying in Clutter on Monocular RGB by Learning in 3D Radiance Fields with Domain Adaptation</td><td>[2512.17349](http://arxiv.org/pdf/2512.17349)</td><td>◆ Modern autonomous navigation systems predominantly rely on lidar and depth cameras.
◆ However, a fundamental question remains: Can flying robots navigate in clutter using solely monocular RGB images?
◆ Given the prohibitive costs of real-world data collection, learning policies in simulation offers a promising path.</td></tr>
<tr><td>2025-12-18</td><td>SDFoam: Signed-Distance Foam for explicit surface reconstruction</td><td>[2512.16706](http://arxiv.org/pdf/2512.16706)</td><td>◆ Neural radiance fields (NeRF) have driven impressive progress in view synthesis by using ray-traced volumetric rendering.
◆ Splatting-based methods such as 3D Gaussian Splatting (3DGS) provide faster rendering by rasterizing 3D primitives.
◆ RadiantFoam (RF) brought ray tracing back, achieving throughput comparable to Gaussian Splatting by organizing radiance with an explicit Voronoi Diagram (VD).</td></tr>
<tr><td>2025-12-17</td><td>Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting</td><td>[2512.15508](http://arxiv.org/pdf/2512.15508)</td><td>◆ Feed-forward 3D Gaussian Splatting (3DGS) models enable real-time scene generation but are hindered by suboptimal pixel-aligned primitive placement, which relies on a dense, rigid grid and limits both quality and efficiency.
◆ We introduce a new feed-forward architecture that detects 3D Gaussian primitives at a sub-pixel level, replacing the pixel grid with an adaptive, &quot;Off The Grid&quot; distribution.
◆ Inspired by keypoint detection, our multi-resolution decoder learns to distribute primitives across image patches.</td></tr>
<tr><td>2025-12-17</td><td>MVGSR: Multi-View Consistent 3D Gaussian Super-Resolution via Epipolar Guidance</td><td>[2512.15048](http://arxiv.org/pdf/2512.15048)</td><td>◆ Scenes reconstructed by 3D Gaussian Splatting (3DGS) trained on low-resolution (LR) images are unsuitable for high-resolution (HR) rendering.
◆ Consequently, a 3DGS super-resolution (SR) method is needed to bridge LR inputs and HR rendering.
◆ Early 3DGS SR methods rely on single-image SR networks, which lack cross-view consistency and fail to fuse complementary information across views.</td></tr>
<tr><td>2025-12-16</td><td>HGS: Hybrid Gaussian Splatting with Static-Dynamic Decomposition for Compact Dynamic View Synthesis</td><td>[2512.14352](http://arxiv.org/pdf/2512.14352)</td><td>◆ Dynamic novel view synthesis (NVS) is essential for creating immersive experiences.
◆ Existing approaches have advanced dynamic NVS by introducing 3D Gaussian Splatting (3DGS) with implicit deformation fields or indiscriminately assigned time-varying parameters, surpassing NeRF-based methods.
◆ However, due to excessive model complexity and parameter redundancy, they incur large model sizes and slow rendering speeds, making them inefficient for real-time applications, particularly on resource-constrained devices.</td></tr>
<tr><td>2025-12-16</td><td>GaussianPlant: Structure-aligned Gaussian Splatting for 3D Reconstruction of Plants</td><td>[2512.14087](http://arxiv.org/pdf/2512.14087)</td><td>◆ We present a method for jointly recovering the appearance and internal structure of botanical plants from multi-view images based on 3D Gaussian Splatting (3DGS).
◆ While 3DGS exhibits robust reconstruction of scene appearance for novel-view synthesis, it lacks structural representations underlying those appearances (e.g., branching patterns of plants), which limits its applicability to tasks such as plant phenotyping.
◆ To achieve both high-fidelity appearance and structural reconstruction, we introduce GaussianPlant, a hierarchical 3DGS representation, which disentangles structure and appearance.</td></tr>
<tr><td>2025-12-15</td><td>Computer vision training dataset generation for robotic environments using Gaussian splatting</td><td>[2512.13411](http://arxiv.org/pdf/2512.13411)</td><td>◆ This paper introduces a novel pipeline for generating large-scale, highly realistic, and automatically labeled datasets for computer vision tasks in robotic environments.
◆ Our approach addresses the critical challenges of the domain gap between synthetic and real-world imagery and the time-consuming bottleneck of manual annotation.
◆ We leverage 3D Gaussian Splatting (3DGS) to create photorealistic representations of the operational environment and objects.</td></tr>
<tr><td>2025-12-12</td><td>Moment-Based 3D Gaussian Splatting: Resolving Volumetric Occlusion with Order-Independent Transmittance</td><td>[2512.11800](http://arxiv.org/pdf/2512.11800)</td><td>◆ The recent success of 3D Gaussian Splatting (3DGS) has reshaped novel view synthesis by enabling fast optimization and real-time rendering of high-quality radiance fields.
◆ However, it relies on simplified, order-dependent alpha blending and coarse approximations of the density integral within the rasterizer, thereby limiting its ability to render complex, overlapping semi-transparent objects.
◆ In this paper, we extend rasterization-based rendering of 3D Gaussian representations with a novel method for high-fidelity transmittance computation, entirely avoiding the need for ray tracing or per-pixel sample sorting.</td></tr>
<tr><td>2025-12-11</td><td>Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from Sparse and Motion-Blurred Views</td><td>[2512.10369](http://arxiv.org/pdf/2512.10369)</td><td>◆ 3D Gaussian Splatting (3DGS) has emerged as a state-of-the-art method for novel view synthesis.
◆ However, its performance heavily relies on dense, high-quality input imagery, an assumption that is often violated in real-world applications, where data is typically sparse and motion-blurred.
◆ These two issues create a vicious cycle: sparse views ignore the multi-view constraints necessary to resolve motion blur, while motion blur erases high-frequency details crucial for aligning the limited views.</td></tr>
<tr><td>2025-12-10</td><td>Splatent: Splatting Diffusion Latents for Novel View Synthesis</td><td>[2512.09923](http://arxiv.org/pdf/2512.09923)</td><td>◆ Radiance field representations have recently been explored in the latent space of VAEs that are commonly used by diffusion models.
◆ This direction offers efficient rendering and seamless integration with diffusion-based pipelines.
◆ However, these methods face a fundamental limitation: The VAE latent space lacks multi-view consistency, leading to blurred textures and missing details during 3D reconstruction.</td></tr>
<tr><td>2025-12-10</td><td>YOPO-Nav: Visual Navigation using 3DGS Graphs from One-Pass Videos</td><td>[2512.09903](http://arxiv.org/pdf/2512.09903)</td><td>◆ Visual navigation has emerged as a practical alternative to traditional robotic navigation pipelines that rely on detailed mapping and path planning.
◆ However, constructing and maintaining 3D maps is often computationally expensive and memory-intensive.
◆ We address the problem of visual navigation when exploration videos of a large environment are available.</td></tr>
<tr><td>2025-12-11</td><td>Relightable and Dynamic Gaussian Avatar Reconstruction from Monocular Video</td><td>[2512.09335](http://arxiv.org/pdf/2512.09335)</td><td>◆ Modeling relightable and animatable human avatars from monocular video is a long-standing and challenging task.
◆ Recently, Neural Radiance Field (NeRF) and 3D Gaussian Splatting (3DGS) methods have been employed to reconstruct the avatars.
◆ However, they often produce unsatisfactory photo-realistic results because of insufficient geometrical details related to body motion, such as clothing wrinkles.</td></tr>
<tr><td>2025-12-10</td><td>MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification</td><td>[2512.09270](http://arxiv.org/pdf/2512.09270)</td><td>◆ Recent advances in 4D Gaussian Splatting (4DGS) have extended the high-speed rendering capability of 3D Gaussian Splatting (3DGS) into the temporal domain, enabling real-time rendering of dynamic scenes.
◆ However, one of the major remaining challenges lies in modeling long-range motion-contained dynamic videos, where a naive extension of existing methods leads to severe memory explosion, temporal flickering, and failure to handle appearing or disappearing occlusions over time.
◆ To address these challenges, we propose a novel 4DGS framework characterized by an Anchor Relay-based Bidirectional Blending (ARBB) mechanism, named MoRel, which enables temporally consistent and memory-efficient modeling of long-range dynamic scenes.</td></tr>
<tr><td>2025-12-09</td><td>OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics</td><td>[2512.08625](http://arxiv.org/pdf/2512.08625)</td><td>◆ Simultaneous Localization and Mapping (SLAM) is a foundational component in robotics, AR/VR, and autonomous systems.
◆ With the rising focus on spatial AI in recent years, combining SLAM with semantic understanding has become increasingly important for enabling intelligent perception and interaction.
◆ Recent efforts have explored this integration, but they often rely on depth sensors or closed-set semantic models, limiting their scalability and adaptability in open-world environments.</td></tr>
<tr><td>2025-12-09</td><td>On-the-fly Large-scale 3D Reconstruction from Multi-Camera Rigs</td><td>[2512.08498](http://arxiv.org/pdf/2512.08498)</td><td>◆ Recent advances in 3D Gaussian Splatting (3DGS) have enabled efficient free-viewpoint rendering and photorealistic scene reconstruction.
◆ While on-the-fly extensions of 3DGS have shown promise for real-time reconstruction from monocular RGB streams, they often fail to achieve complete 3D coverage due to the limited field of view (FOV).
◆ Employing a multi-camera rig fundamentally addresses this limitation.</td></tr>
<tr><td>2025-12-09</td><td>Visionary: The World Model Carrier Built on WebGPU-Powered Gaussian Splatting Platform</td><td>[2512.08478](http://arxiv.org/pdf/2512.08478)</td><td>◆ Neural rendering, particularly 3D Gaussian Splatting (3DGS), has evolved rapidly and become a key component for building world models.
◆ However, existing viewer solutions remain fragmented, heavy, or constrained by legacy pipelines, resulting in high deployment friction and limited support for dynamic content and generative models.
◆ In this work, we present Visionary, an open, web-native platform for real-time various Gaussian Splatting and meshes rendering.</td></tr>
<tr><td>2025-12-09</td><td>Zero-Splat TeleAssist: A Zero-Shot Pose Estimation Framework for Semantic Teleoperation</td><td>[2512.08271](http://arxiv.org/pdf/2512.08271)</td><td>◆ We introduce Zero-Splat TeleAssist, a zero-shot sensor-fusion pipeline that transforms commodity CCTV streams into a shared, 6-DoF world model for multilateral teleoperation.</td></tr>
<tr><td>2025-12-09</td><td>From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images</td><td>[2512.07527](http://arxiv.org/pdf/2512.07527)</td><td>◆ City-scale 3D reconstruction from satellite imagery presents the challenge of extreme viewpoint extrapolation, where our goal is to synthesize ground-level novel views from sparse orbital images with minimal parallax.
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
<tr><td>2025-12-09</td><td>COREA: Coarse-to-Fine 3D Representation Alignment Between Relightable 3D Gaussians and SDF via Bidirectional 3D-to-3D Supervision</td><td>[2512.07107](http://arxiv.org/pdf/2512.07107)</td><td>◆ We present COREA, the first unified framework that jointly learns relightable 3D Gaussians and a Signed Distance Field (SDF) for accurate geometry reconstruction and faithful relighting.
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
<tr><td>2026-03-17</td><td>MessyKitchens: Contact-rich object-level 3D scene reconstruction</td><td>[2603.16868](http://arxiv.org/pdf/2603.16868)</td><td>◆ Monocular 3D scene reconstruction has recently seen significant progress.
◆ Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image.
◆ Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations.</td></tr>
<tr><td>2026-03-17</td><td>WildDepth: A Multimodal Dataset for 3D Wildlife Perception and Depth Estimation</td><td>[2603.16816](http://arxiv.org/pdf/2603.16816)</td><td>◆ Depth estimation and 3D reconstruction have been extensively studied as core topics in computer vision.
◆ Starting from rigid objects with relatively simple geometric shapes, such as vehicles, the research has expanded to address general objects, including challenging deformable objects, such as humans and animals.
◆ However, for the animal, in particular, the majority of existing models are trained based on datasets without metric scale, which can help validate image-only models.</td></tr>
<tr><td>2026-03-17</td><td>$D^3$-RSMDE: 40$\times$ Faster and High-Fidelity Remote Sensing Monocular Depth Estimation</td><td>[2603.16362](http://arxiv.org/pdf/2603.16362)</td><td>◆ Real-time, high-fidelity monocular depth estimation from remote sensing imagery is crucial for numerous applications, yet existing methods face a stark trade-off between accuracy and efficiency.
◆ Although using Vision Transformer (ViT) backbones for dense prediction is fast, they often exhibit poor perceptual quality.
◆ Conversely, diffusion models offer high fidelity but at a prohibitive computational cost.</td></tr>
<tr><td>2026-03-17</td><td>Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation</td><td>[2603.16340](http://arxiv.org/pdf/2603.16340)</td><td>◆ In this paper, we propose \textbf{Iris}, a deterministic framework for Monocular Depth Estimation (MDE) that integrates real-world priors into the diffusion model.
◆ Conventional feed-forward methods rely on massive training data, yet still miss details.
◆ Previous diffusion-based methods leverage rich generative priors yet struggle with synthetic-to-real domain transfer.</td></tr>
<tr><td>2026-03-17</td><td>PureCLIP-Depth: Prompt-Free and Decoder-Free Monocular Depth Estimation within CLIP Embedding Space</td><td>[2603.16238](http://arxiv.org/pdf/2603.16238)</td><td>◆ We propose PureCLIP-Depth, a completely prompt-free, decoder-free Monocular Depth Estimation (MDE) model that operates entirely within the Contrastive Language-Image Pre-training (CLIP) embedding space.
◆ Unlike recent models that rely heavily on geometric features, we explore a novel approach to MDE driven by conceptual information, performing computations directly within the conceptual CLIP space.
◆ The core of our method lies in learning a direct mapping from the RGB domain to the depth domain strictly inside this embedding space.</td></tr>
<tr><td>2026-03-17</td><td>Leveling3D: Leveling Up 3D Reconstruction with Feed-Forward 3D Gaussian Splatting and Geometry-Aware Generation</td><td>[2603.16211](http://arxiv.org/pdf/2603.16211)</td><td>◆ Feed-forward 3D reconstruction has revolutionized 3D vision, providing a powerful baseline for downstream tasks such as novel-view synthesis with 3D Gaussian Splatting.
◆ Previous works explore fixing the corrupted rendering results with a diffusion model.
◆ However, they lack geometric concern and fail at filling the missing area on the extrapolated view.</td></tr>
<tr><td>2026-03-16</td><td>Pointing-Based Object Recognition</td><td>[2603.15403](http://arxiv.org/pdf/2603.15403)</td><td>◆ This paper presents a comprehensive pipeline for recognizing objects targeted by human pointing gestures using RGB images.
◆ As human-robot interaction moves toward more intuitive interfaces, the ability to identify targets of non-verbal communication becomes crucial.
◆ Our proposed system integrates several existing state-of-the-art methods, including object detection, body pose estimation, monocular depth estimation, and vision-language models.</td></tr>
<tr><td>2026-03-16</td><td>Spectral Rectification for Parameter-Efficient Adaptation of Foundation Models in Colonoscopy Depth Estimation</td><td>[2603.15374](http://arxiv.org/pdf/2603.15374)</td><td>◆ Accurate monocular depth estimation is critical in colonoscopy for lesion localization and navigation.
◆ Foundation models trained on natural images fail to generalize directly to colonoscopy.
◆ We identify the core issue not as a semantic gap, but as a statistical shift in the frequency domain: colonoscopy images lack the strong high-frequency edge and texture gradients that these models rely on for geometric reasoning.</td></tr>
<tr><td>2026-03-13</td><td>UE5-Forest: A Photorealistic Synthetic Stereo Dataset for UAV Forestry Depth Estimation</td><td>[2603.15304](http://arxiv.org/pdf/2603.15304)</td><td>◆ Dense ground-truth disparity maps are practically unobtainable in forestry environments, where thin overlapping branches and complex canopy geometry defeat conventional depth sensors -- a critical bottleneck for training supervised stereo matching networks for autonomous UAV-based pruning.
◆ We present UE5-Forest, a photorealistic synthetic stereo dataset built entirely in Unreal Engine 5 (UE5).
◆ One hundred and fifteen photogrammetry-scanned trees from the Quixel Megascans library are placed in virtual scenes and captured by a simulated stereo rig whose intrinsics -- 63 mm baseline, 2.8 mm focal length, 3.84 mm sensor width -- replicate the ZED Mini camera mounted on our drone.</td></tr>
<tr><td>2026-03-16</td><td>Reference-Free Omnidirectional Stereo Matching via Multi-View Consistency Maximization</td><td>[2603.15019](http://arxiv.org/pdf/2603.15019)</td><td>◆ Reliable omnidirectional depth estimation from multi-fisheye stereo matching is pivotal to many applications, such as embodied robotics.
◆ Existing approaches either rely on spherical sweeping with heuristic fusion strategies to build the cost columns or perform reference-centric stereo matching based on rectified views.
◆ However, these methods fail to explicitly exploit geometric relationships between multiple views, rendering them less capable of capturing the global dependencies, visibility, or scale changes.</td></tr>
<tr><td>2026-03-16</td><td>Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3</td><td>[2603.14998](http://arxiv.org/pdf/2603.14998)</td><td>◆ Autonomous navigation in GPS-denied and visually degraded environments remains challenging for unmanned aerial vehicles (UAVs).
◆ To this end, we investigate the use of a monocular thermal camera as a standalone sensor on a UAV platform for real-time depth estimation and simultaneous localization and mapping (SLAM).
◆ To extract depth information from thermal images, we propose a novel pipeline employing a lightweight supervised network with recurrent blocks (RBs) integrated to capture temporal dependencies, enabling more robust predictions.</td></tr>
<tr><td>2026-03-16</td><td>Fractal Autoregressive Depth Estimation with Continuous Token Diffusion</td><td>[2603.14702](http://arxiv.org/pdf/2603.14702)</td><td>◆ Monocular depth estimation can benefit from autoregressive (AR) generation, but direct AR modeling is hindered by the modality gap between RGB and depth, inefficient pixel-wise generation, and instability in continuous depth prediction.
◆ We propose a Fractal Visual Autoregressive Diffusion framework that reformulates depth estimation as a coarse-to-fine, next-scale autoregressive generation process.
◆ A VCFR module fuses multi-scale image features with current depth predictions to improve cross-modal conditioning, while a conditional denoising diffusion loss models depth distributions directly in continuous space and mitigates errors caused by discrete quantization.</td></tr>
<tr><td>2026-03-16</td><td>E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction</td><td>[2603.14684](http://arxiv.org/pdf/2603.14684)</td><td>◆ The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS).
◆ These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting.
◆ Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution.</td></tr>
<tr><td>2026-03-15</td><td>V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning</td><td>[2603.14482](http://arxiv.org/pdf/2603.14482)</td><td>◆ We present V-JEPA 2.1, a family of self-supervised models that learn dense, high-quality visual representations for both images and videos while retaining strong global scene understanding.
◆ The approach combines four key components.
◆ First, a dense predictive loss uses a masking-based objective in which both visible and masked tokens contribute to the training signal, encouraging explicit spatial and temporal grounding.</td></tr>
<tr><td>2026-03-14</td><td>ALTIS: Automated Loss Triage and Impact Scoring from Sentinel-1 SAR for Property-Level Flood Damage Assessment</td><td>[2603.13803](http://arxiv.org/pdf/2603.13803)</td><td>◆ Floods are among the costliest natural catastrophes globally, yet the property and casualty insurance industry&#x27;s post-event response remains heavily reliant on manual field inspection: slow, expensive, and geographically constrained.
◆ Satellite Synthetic Aperture Radar (SAR) offers cloud-penetrating, all-weather imaging uniquely suited to rapid post-flood assessment, but existing research evaluates SAR flood detection against academic benchmarks such as IoU and F1-score that do not capture insurance-workflow requirements.
◆ We present ALTIS: a five-stage pipeline transforming raw Sentinel-1 GRD and SLC imagery into property-level impact scores within 24-48 hours of flood peak.</td></tr>
<tr><td>2026-03-12</td><td>DVD: Deterministic Video Depth Estimation with Generative Priors</td><td>[2603.12250](http://arxiv.org/pdf/2603.12250)</td><td>该论文提出了DVD框架，首次将预训练视频扩散模型确定性地转化为单次深度估计器，解决了生成模型几何幻觉与判别模型数据依赖之间的固有矛盾。

◆ 创新性地将扩散时间步作为结构锚点，平衡全局稳定性与高频细节，实现确定性深度回归。
◆ 提出潜在流形校正技术，通过微分约束缓解回归导致的过度平滑，恢复清晰边界与连贯运动。
◆ 利用全局仿射相干性这一内在特性，约束跨窗口差异，无需复杂时序对齐即可实现长视频无缝推理。
◆ 仅用极少量任务数据（比基线少163倍）成功挖掘了视频基础模型中隐含的几何先验。
◆ 在零样本深度估计基准上达到最先进性能，并完整开源训练流程以促进社区发展。</td></tr>
<tr><td>2026-03-12</td><td>R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection</td><td>[2603.11566](http://arxiv.org/pdf/2603.11566)</td><td>R4Det论文的核心贡献在于提出了一种高性能的4D雷达与相机融合的3D物体检测框架，有效解决了现有方法的关键瓶颈。其创新点具体如下：

◆ 提出了全景深度融合模块，通过联合优化绝对深度与相对深度估计，使两者相互增强，显著提升了深度估计的鲁棒性和准确性，从而改善了3D定位精度。

◆ 设计了可变形门控时序融合模块，该模块不依赖于自车姿态信息，即使在自车位姿缺失或不准确的情况下，也能稳定有效地融合多帧时序信息，增强了系统的实用性。

◆ 构建了实例引导的动态优化模块，利用从2D图像实例中提取的语义原型来动态优化特征，使得在雷达点云极其稀疏或完全缺失（如对小物体）的情况下，系统能有效依赖视觉单模态先验进行检测。

◆ 整体框架在TJ4DRadSet和VoD数据集上实现了领先的3D物体检测性能，验证了所提方法在应对实际自动驾驶复杂场景挑战时的有效性。</td></tr>
<tr><td>2026-03-11</td><td>WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation</td><td>[2603.10703](http://arxiv.org/pdf/2603.10703)</td><td>该论文的核心贡献是提出了WalkGPT模型及配套数据集，以解决现有大视觉语言模型在行人导航中缺乏空间与语义精准接地的问题。其创新点如下：

◆ 提出了“接地导航引导”新任务，要求模型根据行人视角图像和导航查询，生成结合语言对话、分割掩码和相对深度估计的完整导航指导。
◆ 设计了WalkGPT模型，首次将语言推理与分割统一于单一架构，实现了无需用户提示或锚点的深度感知可访问性分析。
◆ 创新了多尺度查询投影器，通过跨空间层次聚合文本令牌来塑造最终图像令牌，以增强细粒度接地能力。
◆ 引入了校准文本投影器及区域对齐损失，将语言嵌入映射为分割感知的表征，从而提升分割与语言的对齐精度。
◆ 构建了PAVE大规模基准数据集，包含4.1万张行人视角图像及与可访问性相关的问题和深度接地答案，支持任务评估。实验证明WalkGPT在接地推理和分割性能上表现优异。</td></tr>
<tr><td>2026-03-11</td><td>AsyncMDE: Real-Time Monocular Depth Estimation via Asynchronous Spatial Memory</td><td>[2603.10438](http://arxiv.org/pdf/2603.10438)</td><td>本文核心贡献是提出了AsyncMDE系统，旨在解决基于基础模型的单目深度估计计算成本高、难以在边缘设备实时部署的问题。

◆ 提出了异步双模型架构，将计算昂贵的基础模型（在后台生成高质量空间特征）与轻量级模型（在前台异步运行）解耦，从而分摊了跨时间的计算开销。
◆ 设计了包含互补融合与自回归更新的空间记忆机制，轻量级模型能异步融合缓存的历史特征与当前观测，实现跨帧特征复用，在保证精度的前提下大幅减少重复计算。
◆ 系统在精度与效率间取得优异平衡，模型参数量仅3.83M，在高端GPU上达到237 FPS，恢复了基础模型77%的精度差距，且参数量减少25倍。
◆ 在静态、动态及极端运动场景的多个基准测试中验证了其鲁棒性，系统在刷新间隔内性能下降平缓，并在Jetson AGX Orin边缘设备上借助TensorRT实现了161 FPS，证明了其实时边缘部署的可行性。</td></tr>
<tr><td>2026-03-10</td><td>SurgFed: Language-guided Multi-Task Federated Learning for Surgical Video Understanding</td><td>[2603.09496](http://arxiv.org/pdf/2603.09496)</td><td>该论文提出了一种用于手术视频理解的多任务联邦学习框架SurgFed，以解决机器人辅助微创手术中的两大挑战。其核心贡献与创新点如下：

◆ 针对组织多样性导致的本地模型适应性问题，设计了语言引导的通道选择模块。该轻量化网络利用预定义文本输入增强站点特异性特征学习，优化了本地模型对特定组织嵌入的捕捉。

◆ 针对任务多样性导致的服务器聚合难题，提出了语言引导的超网络聚合机制。该模块采用分层交叉注意力机制，结合文本输入建模跨站点的任务交互，从而指导超网络生成个性化的参数更新。

◆ 首次实现了跨多种手术类型的联邦学习统一框架，能够同时处理手术场景分割与深度估计两个关键任务，有效探索了跨站点与跨任务的协同学习。

◆ 在五个公开数据集和四种手术类型上的实验表明，该框架性能优于现有最先进方法，验证了其有效性。</td></tr>
<tr><td>2026-03-10</td><td>EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation</td><td>[2603.09385](http://arxiv.org/pdf/2603.09385)</td><td>本文的核心贡献是提出了EventVGGT框架，首次将视觉基础模型中的时空与多视图几何先验知识蒸馏到事件相机领域，以解决事件流单目深度估计中时间不一致和标注稀缺的问题。

◆ 创新性地将事件流建模为连贯的视频序列进行处理，突破了现有方法将事件流视为独立帧的局限。
◆ 首次从视觉几何基础Transformer模型中蒸馏出强大的时空表征和多视图几何先验知识至事件域。
◆ 设计了一个全面的三重蒸馏策略：通过跨模态特征混合在输出层面融合RGB与事件特征；通过时空特征蒸馏在特征层面迁移知识；通过时间一致性蒸馏在时间层面强制跨帧深度变化的对齐，确保了预测的时序一致性。
◆ 该框架在多个数据集上显著超越了现有方法，例如将EventScape数据集上的深度误差降低了超过53%，并展现了强大的零样本泛化能力。</td></tr>
<tr><td>2026-03-10</td><td>SpaceSense-Bench: A Large-Scale Multi-Modal Benchmark for Spacecraft Perception and Pose Estimation</td><td>[2603.09320](http://arxiv.org/pdf/2603.09320)</td><td>该论文的核心贡献是构建了一个大规模、多模态的航天器感知与位姿估计基准数据集SpaceSense-Bench，以解决在轨自主任务中真实数据匮乏和现有合成数据集不足的问题。

◆ 创建了迄今规模最大、目标最多样的航天器感知数据集，包含136个卫星模型，数据量约70GB，极大提升了数据规模和多样性。
◆ 提供了高质量、多模态且时间同步的传感器数据，包括高分辨率RGB图像、毫米级精度深度图和256线激光雷达点云。
◆ 提供了全面且密集的标注信息，包括像素级和点云级的7类部件级语义标签以及精确的6自由度位姿真值，支持细粒度理解。
◆ 设计并实现了基于虚幻引擎5的高保真空间仿真与全自动数据生成流程，确保了数据的高质量和一致性。
◆ 通过系统性地评估五项代表性感知任务，揭示了当前方法在识别小部件和零样本泛化到全新航天器方面的瓶颈，并实证了扩大训练集规模对提升泛化性能的重要价值。</td></tr>
<tr><td>2026-03-09</td><td>Viewpoint-Agnostic Grasp Pipeline using VLM and Partial Observations</td><td>[2603.07866](http://arxiv.org/pdf/2603.07866)</td><td>该论文针对移动腿式机器人在杂乱遮挡环境中抓取的难题，提出了一套端到端的语言引导抓取流程，显著提升了部分观测条件下的鲁棒性。

◆ 构建了从开放词汇目标检测到安全抓取执行的完整端到端流程，实现了通过自然语言指令直接指挥机器人抓取。
◆ 提出基于RGB-D的物体中心点云提取方法，并结合反向投影深度补偿与两阶段点云补全技术，有效改善了遮挡导致的几何信息不可靠问题。
◆ 设计了以安全为导向的启发式抓取选择策略，综合考虑可达性、接近可行性和避障空间，确保生成的6自由度抓取候选可安全执行。
◆ 在真实四足机器人上进行了验证，在杂乱桌面场景中取得了90%的整体成功率，相比依赖固定视角的基线方法（30%）有大幅提升，证明了其对遮挡和部分观测的强大鲁棒性。</td></tr>
<tr><td>2026-03-08</td><td>FrameVGGT: Frame Evidence Rolling Memory for streaming VGGT</td><td>[2603.07690](http://arxiv.org/pdf/2603.07690)</td><td>该论文的核心贡献是提出了FrameVGGT，一个用于流式视觉几何Transformer的帧驱动滚动显式记忆框架，旨在解决长序列处理中内存无限增长的问题。

◆ 核心创新在于从几何支持的新视角重新审视有界内存流式处理，指出在固定内存预算下，传统的令牌级保留会稀释每帧内的证据支持，导致后续融合不稳定。
◆ 提出了帧驱动的记忆框架，将每一帧新增的键值对视为一个连贯的证据块进行处理，而非独立的令牌。
◆ 设计了将每个证据块压缩为紧凑原型的摘要方法，并在严格预算下维护一个固定容量的中期记忆库来存储互补的帧块。
◆ 引入了一个可选的轻量级锚点层级，以应对罕见的长期性能退化情况，增强了系统的鲁棒性。
◆ 在多个长序列3D感知任务（如3D重建、视频深度估计和相机姿态估计）上验证了该方法，在有限内存条件下取得了更优的精度-内存权衡，并在长流中保持了更稳定的几何推理性能。</td></tr>
<tr><td>2026-03-06</td><td>SurgSync: Time-Synchronized Multi-Modal Data Collection Framework and Dataset for Surgical Robotics</td><td>[2603.06919](http://arxiv.org/pdf/2603.06919)</td><td>该论文的核心贡献是提出了一个名为SurgSync的、用于手术机器人研究的同步多模态数据采集框架和数据集，旨在解决人工智能训练数据匮乏的问题。

其核心创新点包括：
◆ 设计了一个兼具离线和在线同步功能的多模态数据采集框架，分别支持算法训练与实时推理。
◆ 开发了双模式（在线匹配/离线匹配）同步记录器，确保了多传感器数据流的时间对齐。
◆ 集成了现代立体内窥镜，使图像质量达到临床系统水平，并增加了侧视摄像头和新颖的电容式接触传感器，后者能提供宝贵的接触地面真值数据。
◆ 提供了一套后处理工具箱，包含深度估计、光流计算以及使用高斯热图的实用运动学重投影方法。
◆ 通过用户研究，利用离体组织采集了包含214个已验证实例的临床现实数据集，并演示了该数据在手术技能评估网络中的应用价值。</td></tr>
<tr><td>2026-03-06</td><td>CHMv2: Improvements in Global Canopy Height Mapping using DINOv3</td><td>[2603.06382](http://arxiv.org/pdf/2603.06382)</td><td>该论文的核心贡献是提出了CHMv2，一种基于DINOv3构建的全球米级分辨率冠层高度地图，其通过高分辨率光学卫星影像估算冠层高度，显著提升了全球森林监测能力。

◆ 首次将先进的DINOv3视觉基础模型应用于全球尺度的冠层高度深度估计任务，提升了模型的特征提取能力。
◆ 通过大规模扩充地理多样化的训练数据，并结合自动化的数据整理与配准流程，为模型提供了更高质量、更全面的训练基础。
◆ 设计了针对冠层高度分布特点的损失函数和数据采样策略，有效减少了高森林区域的估计偏差，并更好地保留了冠层边缘和林隙等细微结构。
◆ 相较于现有产品，CHMv2在精度上取得实质性提升，并通过与独立的机载激光扫描数据以及数千万个GEDI和ICESat-2观测点进行验证，证明了其在主要森林生物群落中性能的稳定性和可靠性。</td></tr>
<tr><td>2026-03-06</td><td>RePer-360: Releasing Perspective Priors for 360$^\circ$ Depth Estimation via Self-Modulation</td><td>[2603.05999](http://arxiv.org/pdf/2603.05999)</td><td>该论文的核心贡献是提出了一种名为RePer-360的自调制框架，旨在高效地将基于透视图像预训练的深度基础模型适配到360度全景图像上，同时最大程度保留其原有的强大先验知识。

其核心创新点如下：
◆ 提出了一个轻量级的几何对齐引导模块，该模块从ERP和CP两种互补投影中提取调制信号，以此引导模型适应全景域，而无需覆盖或破坏其预训练的透视先验知识。
◆ 引入了自条件AdaLN-Zero机制，能够生成像素级的缩放因子，以有效减少透视域与全景域之间的特征分布差异。
◆ 设计了一种立方体贴图域的一致性损失函数，该损失提升了训练的稳定性，并增强了不同投影之间的对齐效果。
◆ 该方法实现了研究重点的转移，从传统的多投影融合转向了在保护预训练先验下的全景域自适应，从而在仅使用1%训练数据的情况下，性能便超越了标准的全微调方法。
◆ 在同等域内训练设置下，该方法取得了约20%的RMSE性能提升，显著提高了单目全景深度估计的精度与数据效率。</td></tr>
<tr><td>2026-03-06</td><td>EventGeM: Global-to-Local Feature Matching for Event-Based Visual Place Recognition</td><td>[2603.05807](http://arxiv.org/pdf/2603.05807)</td><td>该论文提出了EventGeM，一种用于基于事件的视觉位置识别（VPR）的先进方法。其核心贡献在于构建了一个高效且鲁棒的全局到局部特征匹配流程，能够在多种挑战性条件下实现最先进的定位性能，并满足实时性要求。

◆ 提出了一种新颖的全局到局部特征融合流水线，首先利用预训练的视觉变换器（ViT）从事件直方图图像中提取全局特征进行初始匹配。
◆ 创新性地引入预训练的MaxViT骨干网络来检测局部特征关键点，并基于2D单应性变换与RANSAC进行重排序，提升了匹配的几何一致性。
◆ 额外采用预训练的视觉基础模型进行深度估计，通过比较查询图像与参考图像之间的结构相似性，实现了进一步的重排序精化。
◆ 该方法在多个基准数据集和不同光照条件下，性能超越了当前最佳的基于事件的位置识别方法，证明了其卓越的鲁棒性和准确性。
◆ 整个系统设计兼顾效率，可在多种计算架构上实时运行，并成功在真实机器人平台上使用来自事件相机的原始事件流进行了在线定位验证。</td></tr>
<tr><td>2026-03-05</td><td>EmboAlign: Aligning Video Generation with Compositional Constraints for Zero-Shot Manipulation</td><td>[2603.05757](http://arxiv.org/pdf/2603.05757)</td><td>该论文提出EmboAlign框架，旨在解决视频生成模型在零样本机器人操控中存在的物理不合理性和几何重定向误差问题。其核心贡献与创新点如下：

◆ 提出一种无需额外训练数据的推理时对齐框架，通过视觉语言模型自动提取任务相关的组合约束条件，弥补视频生成模型在结构化空间推理上的不足。

◆ 利用视觉语言模型生成组合约束，这些约束捕捉了任务执行的关键物理限制条件，确保操作的安全性与成功率。

◆ 设计两阶段约束应用机制：首先进行约束引导的视频片段选择，从批量生成结果中筛选出最物理合理的候选；随后进行基于约束的轨迹优化，以选定片段为初始解，在相同约束下优化机器人轨迹以纠正重定向误差。

◆ 将视频生成模型的动态捕捉能力与视觉语言模型的空间推理能力互补结合，提升了零样本操控的物理真实性和执行精度。

◆ 在六项真实机器人操控任务上验证了有效性，相比基线方法成功率提升43.3%，且无需任何任务特定训练数据。</td></tr>
<tr><td>2026-03-05</td><td>Any to Full: Prompting Depth Anything for Depth Completion in One Stage</td><td>[2603.05711](http://arxiv.org/pdf/2603.05711)</td><td>该论文提出了一种名为Any2Full的单阶段、领域通用且模式无关的深度补全框架。其核心贡献在于通过创新的提示微调方法，将预训练的单目深度估计模型直接适配于深度补全任务，避免了现有方法的局限性。

◆ 提出单阶段框架：将深度补全重新定义为对预训练单目深度估计模型的提示微调，摒弃了依赖显式相对到度量对齐的两阶段流程，从而提升了效率并减少了结构失真。
◆ 设计尺度感知提示编码器：该模块能从任意稀疏度和不规则空间分布的深度输入中提取尺度线索，并蒸馏为统一的尺度提示，以引导模型预测。
◆ 实现领域通用与模式无关：框架不依赖于特定的训练RGB分布或深度模式，有效保留了预训练模型的几何先验，显著提升了跨域泛化能力和对各种深度模式的鲁棒性。
◆ 取得卓越性能与效率：实验证明，该方法在精度上大幅超越现有方法（如OMNI-DC），并在速度上优于同类方案（如PriorDA），为通用深度补全建立了新范式。</td></tr>
<tr><td>2026-03-04</td><td>LiDAR Prompted Spatio-Temporal Multi-View Stereo for Autonomous Driving</td><td>[2603.03765](http://arxiv.org/pdf/2603.03765)</td><td>这篇论文的核心贡献是提出了一个名为DriveMVS的新型多视图立体视觉框架，旨在解决自动驾驶中深度估计的度量精度、多视图与时间一致性以及跨域泛化等关键挑战。其创新点主要体现在以下三个方面：

◆ 提出了“激光雷达提示”的新思路，将稀疏但度量精确的激光雷达点云作为几何提示，以硬几何先验和软特征引导两种方式嵌入框架，从而将深度估计锚定在绝对尺度上，显著提升了度量精度。

◆ 设计了一个深度融合机制，通过三重线索组合器对多视图立体匹配代价体积、激光雷达提示特征和图像特征进行深度融合，有效解决了深度估计中的模糊性问题，增强了系统的鲁棒性。

◆ 引入了一个时空解码器，该解码器能够联合利用当前帧的多视图立体几何线索和来自相邻帧的时间上下文信息，从而确保了深度估计在时间序列上的平滑性与一致性，提升了时序稳定性。

实验表明，DriveMVS在多个基准测试中取得了领先的性能，尤其在度量精度、时间稳定性和零样本跨域迁移能力上表现突出，证明了其对构建可扩展、可靠自动驾驶系统的实用价值。</td></tr>
<tr><td>2026-03-03</td><td>Confidence-aware Monocular Depth Estimation for Minimally Invasive Surgery</td><td>[2603.03571](http://arxiv.org/pdf/2603.03571)</td><td>该论文的核心贡献是提出了一种用于微创手术的置信感知单目深度估计框架，旨在提升在烟雾、反光等干扰下的深度估计精度，并首次为深度图提供可靠性评估。其创新点主要包括：

◆ 提出了校准置信度目标的方法，利用微调后的立体匹配模型集成来生成像素级的置信度概率，以捕捉视差方差。

◆ 设计了置信度感知的损失函数，在训练基线深度估计模型时，让高置信度的像素主导优化过程，从而提升模型精度。

◆ 引入了一个推理时置信度估计头，仅通过两个卷积层即可在推理阶段预测每个像素的置信度，生成可用于评估深度可靠性的置信度图。

◆ 该框架在内部和公共数据集上得到验证，不仅将深度估计精度提升了约8%，还能鲁棒地量化预测置信度，增强了临床应用的可靠性。</td></tr>
<tr><td>2026-03-03</td><td>The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes</td><td>[2603.02985](http://arxiv.org/pdf/2603.02985)</td><td>该论文的核心贡献是创建了一个用于评估非刚性腹部手术场景4D重建算法的综合性数据集。其创新点主要体现在数据采集、内容设计和应用价值三个方面。

◆ 首次提供了在真实手术条件下（猪尸体实验），配对的腹腔镜视频与高质量结构化光几何数据，用于评估软组织形变的三维重建。
◆ 设计了三种序列类型（整体形变、增量形变和相机移动片段），专门用于系统测试算法对非刚性运动、大形变和视野外更新的鲁棒性。
◆ 数据集内容极为丰富且精细，不仅提供校正后的立体图像，还包含逐帧器械掩膜、立体深度、起始/结束点云，以及经过人工校正的相机位姿和内参。
◆ 通过后期处理实现了多模态数据的精确配准，并提供了器械分割掩膜，使得算法能在可见区域和遮挡区域进行全面的几何精度定量评估。
◆ 该数据集规模庞大，包含超过30万帧图像和369个点云，可作为非刚性SLAM、4D重建和深度估计方法开发与评估的综合性基准。</td></tr>
<tr><td>2026-03-03</td><td>DREAM: Where Visual Understanding Meets Text-to-Image Generation</td><td>[2603.02667](http://arxiv.org/pdf/2603.02667)</td><td>本文提出DREAM框架，旨在统一视觉理解与文本生成图像两大任务。其核心贡献与创新点如下：

◆ 提出首个在单一模型中联合优化判别式与生成式目标的学习框架，实现了视觉表征学习与文本到图像生成的统一。

◆ 引入名为“掩蔽预热”的训练技术，通过渐进式掩蔽策略，初期低掩蔽率促进对比学习以获取优质视觉表征，后期过渡到全掩蔽以稳定生成训练，有效协调了两种目标。

◆ 在推理阶段提出“语义对齐解码”方法，通过将部分掩蔽的图像候选与目标文本对齐并择优解码，显著提升了生成图像与文本的语义一致性，文本-图像对齐度提升6.3%，且无需外部重排序模型。

◆ 仅使用CC12M数据集训练，便在多项任务上取得优异性能：ImageNet线性探测准确率达72.7%（超越CLIP 1.1%），生成质量FID为4.25（超越FLUID 6.2%），并在少样本分类、语义分割和深度估计任务中表现一致领先。

◆ 实证表明判别与生成目标具有协同效应，验证了统一多模态模型可同时在视觉理解与生成领域达到卓越水平。</td></tr>
<tr><td>2026-03-02</td><td>TruckDrive: Long-Range Autonomous Highway Driving Dataset</td><td>[2603.02413](http://arxiv.org/pdf/2603.02413)</td><td>该论文的核心贡献是推出了首个专为重型卡车高速公路长距离自动驾驶设计的大规模多模态数据集TruckDrive，以解决现有数据集在远距离感知方面的不足。

◆ 首创针对高速公路重型卡车长距离感知需求的数据集，填补了现有数据集主要覆盖城市短距离场景的空白。
◆ 搭载了专为远距离感知定制的多模态传感器套件，包括七个长距离FMCW激光雷达、十个4D FMCW雷达和高分辨率相机，能有效感知数百米外的场景。
◆ 提供了大规模、高质量的数据与标注，包含47.5万个样本和16.5万帧密集标注，支持最远1000米的2D检测和400米的3D检测、深度估计、跟踪、规划等任务。
◆ 通过实验揭示了当前前沿自动驾驶模型在150米以外远距离感知性能急剧下降（31%至99%）的系统性缺陷，证明了长距离感知是亟待解决的关键挑战。
◆ 数据集包含长达20秒的高速序列，为研究预测性规划和端到端驾驶提供了重要资源。</td></tr>
<tr><td>2026-03-02</td><td>Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation</td><td>[2603.01999](http://arxiv.org/pdf/2603.01999)</td><td>这篇论文的核心贡献是提出了一种基于视觉的师生框架，用于实现移动机器人的全向导航，以克服传统2D激光雷达在复杂三维环境中感知的局限性。

◆ 创新性地采用师生学习框架，将基于特权2D激光雷达（考虑完整机器人轮廓）训练的教师策略知识，蒸馏到仅使用单目视觉的学生策略中。
◆ 学生策略完全依赖由四个RGB摄像头和微调的Depth Anything V2模型生成的单目深度图进行导航，摒弃了对激光雷达传感器的依赖。
◆ 整个推理流程（包括深度估计、策略执行和电机控制）均可完全在机载计算平台（NVIDIA Jetson）上运行，无需外部算力，实现了高度集成和实用的部署。
◆ 在仿真与实物实验中，该视觉学生策略在成功率和鲁棒性上均显著超越了标准的2D激光雷达基线方法，特别是在应对悬垂物、低矮物等激光雷达扫描平面之外的复杂三维障碍物时优势明显。</td></tr>
<tr><td>2026-03-03</td><td>PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts</td><td>[2603.01650](http://arxiv.org/pdf/2603.01650)</td><td>本文的核心贡献是提出了PromptStereo，一种用于零样本立体匹配的新方法，其核心创新在于设计了一个新颖的迭代优化模块。具体贡献与创新点如下：

◆ 提出了提示循环单元（PRU），这是一个全新的迭代优化模块。它基于单目深度基础模型的解码器构建，替代了传统表示能力有限的GRU架构。

◆ 创新性地将单目结构提示和立体运动提示集成到解码过程中。这种方法能将绝对的立体尺度信息注入模型，同时保留其原有的单目深度先验知识。

◆ 系统性地探索并强化了迭代优化阶段在零样本泛化中的关键作用，解决了现有方法对此阶段关注不足的问题。

◆ 所提出的PromptStereo框架在多个数据集上实现了零样本泛化性能的领先，同时保持了相当甚至更快的推理速度。

◆ 本研究指明了“基于提示引导的迭代优化”是零样本立体匹配领域一个极具前景的新方向。</td></tr>
<tr><td>2026-03-02</td><td>WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments</td><td>[2603.01475](http://arxiv.org/pdf/2603.01475)</td><td>该论文的核心贡献是创建了一个专门针对非结构化自然环境的跨模态机器人感知基准数据集，以弥补现有数据集中于结构化城市场景的不足。

◆ 提出了首个大规模、跨模态的自然环境基准WildCross，专注于地点识别和度量深度估计任务。
◆ 数据集规模庞大且信息丰富，包含超过47.6万帧序列RGB图像，并提供了半稠密深度图、表面法线标注。
◆ 数据质量高，每帧图像都配有精确的6自由度位姿，并与同步的稠密激光雷达子图严格对齐，实现了多模态数据的精确关联。
◆ 通过全面的实验验证了数据集的价值，涵盖了视觉、激光雷达及跨模态的地点识别，以及度量深度估计，证明了其在复杂自然场景中作为挑战性基准的有效性。
◆ 公开了完整的代码库和数据集，旨在推动非结构化环境中多模态机器人感知研究的发展。</td></tr>
<tr><td>2026-03-05</td><td>Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving</td><td>[2603.01007](http://arxiv.org/pdf/2603.01007)</td><td>该论文提出Dr.Occ框架，旨在解决自动驾驶中基于环视相机的3D语义占据预测的两大核心难题：几何错位与空间类别不平衡。其核心贡献与创新点如下：

◆ 提出深度引导的2D到3D视图变换器（D²-VFormer），利用高质量稠密深度信息构建可靠的几何先验，实现了体素特征的精确几何对齐，解决了因深度估计不准导致的视图变换几何错位问题。

◆ 提出区域引导的专家变换器（R/R²-EFormer），受混合专家（MoE）框架启发，该模块自适应地分配区域特定专家，以专注处理不同空间区域，有效缓解了语义类别在空间中分布不均（空间各向异性）的问题。

◆ 深度引导与区域引导两大组件形成互补：前者确保几何结构的准确性，后者增强对空间语义变化的建模能力，共同提升场景理解的完整性。

◆ 在纯视觉设定下，该方法在Occ3D-nuScenes基准上显著超越强基线BEVDet4D，mIoU提升7.43%，IoU提升3.09%，验证了其有效性。</td></tr>
<tr><td>2026-02-27</td><td>Altitude-Aware Visual Place Recognition in Top-Down View</td><td>[2602.23872](http://arxiv.org/pdf/2602.23872)</td><td>本文针对航空视觉位置识别在高度变化下的挑战，提出了一种创新的纯视觉解决方案。其核心贡献与创新点如下：

◆ 提出了一种高度自适应的视觉位置识别方法，通过分析图像中地面特征的密度来估计平台的相对高度，无需依赖气压计或飞行时间传感器等额外硬件。
◆ 设计了基于相对高度的图像裁剪策略，生成标准化的查询图像，以应对大幅高度变化带来的视角与尺度差异。
◆ 构建了一个分类式视觉位置识别框架，将估计的高度信息整合到检索流程中，显著提升了定位精度和鲁棒性。
◆ 该方法在多种地形和高度条件下验证有效，实验表明其高度估计误差远低于传统单目度量深度估计方法，并在检索精度上带来显著提升。
◆ 最终成果是一个即插即用的纯视觉三维位置识别系统，为资源有限的中小型空中平台在复杂环境中实现精准定位提供了实用且可扩展的解决方案。</td></tr>
<tr><td>2026-02-26</td><td>EndoDDC: Learning Sparse to Dense Reconstruction for Endoscopic Robotic Navigation via Diffusion Depth Completion</td><td>[2602.21893](http://arxiv.org/pdf/2602.21893)</td><td>该论文针对内窥镜手术机器人导航中深度估计的挑战，提出了一种名为EndoDDC的新型深度补全方法。其核心贡献与创新点如下：

◆ 提出了一种专为内窥镜环境设计的深度补全方法（EndoDDC），旨在解决该场景下弱纹理和光线反射导致的深度估计难题。
◆ 创新性地融合了内窥镜图像、稀疏深度信息以及深度梯度特征，为深度重建提供了更丰富的多模态线索。
◆ 首次将扩散模型引入内窥镜深度补全任务，通过其强大的生成优化能力，将稀疏重建结果优化为高精度、高鲁棒性的稠密深度图。
◆ 在两个公开内窥镜数据集上的实验表明，该方法在深度准确性和鲁棒性上均超越了现有先进模型，验证了其有效性。
◆ 该方法不依赖精确的深度标注，降低了数据获取门槛，并有望减少复杂内窥镜环境中的视觉误差，提升手术导航安全性。</td></tr>
<tr><td>2026-02-25</td><td>Structure-to-Image: Zero-Shot Depth Estimation in Colonoscopy via High-Fidelity Sim-to-Real Adaptation</td><td>[2602.21740](http://arxiv.org/pdf/2602.21740)</td><td>该论文针对结肠镜单目深度估计中仿真与真实图像间的域适应难题，提出了创新解决方案。其核心贡献在于提出了一种“结构到图像”的新范式，显著提升了零样本深度估计的精度。

◆ 首创了“结构到图像”的新范式，将深度图从后置约束转变为主动生成的基石，从根本上改变了域适应的思路。
◆ 首次将相位一致性引入结肠镜域适应任务，有效捕捉对光照变化不敏感的结构信息。
◆ 设计了一种跨层级结构约束，能够协同优化整体几何结构与血管纹理等细粒度细节，平衡真实感与结构一致性。
◆ 所提出的方法在公开数据集上的零样本评估中表现优异，使深度估计模型的RMSE最大降低了44.18%，远超现有方法。
◆ 通过高质量仿真到真实的适应，有效克服了传统图像翻译方法产生的结构扭曲和高光伪影问题。</td></tr>
<tr><td>2026-02-24</td><td>Pip-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching</td><td>[2602.20496](http://arxiv.org/pdf/2602.20496)</td><td>该论文针对基于迭代优化的立体匹配算法在边缘设备部署时的效率瓶颈，提出了Pip-Stereo系统，其核心贡献与创新点如下：

◆ 提出渐进式迭代剪枝策略，通过分析发现视差更新具有时空冗余性，从而抑制冗余迭代步骤，将递归计算压缩至接近单次前向推理，极大提升效率。

◆ 设计协同式单目先验迁移框架，无需引入独立的单目深度编码器即可隐式嵌入深度先验，在保持精度的同时显著减少了计算负担。

◆ 开发硬件感知的FlashGRU循环神经网络算子，利用结构化稀疏和I/O优化设计，在2K分辨率下相比原生ConvGRU实现了7.28倍加速、76.6%内存峰值降低和80.9%全局内存请求减少。

◆ 整套方法使高精度迭代立体匹配能在边缘硬件上实时运行，如在NVIDIA Jetson Orin NX上仅需75毫秒处理320×640图像，精度媲美大型迭代模型，且泛化能力和精度远超现有实时方法。</td></tr>
<tr><td>2026-02-27</td><td>One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image</td><td>[2602.19766](http://arxiv.org/pdf/2602.19766)</td><td>该论文提出One2Scene框架，从单张图像生成可自由探索的高质量3D场景，核心贡献在于通过分解任务与引入几何一致性先验解决了远视角下的失真问题。

◆ 将复杂问题分解为三个可处理的子任务：全景锚视图生成、3D几何支架构建、以及基于支架的新视图生成，提升了生成过程的稳定性和质量。
◆ 提出将输入的全景图投影为多个稀疏锚视图，并将重建任务重新定义为多视图立体匹配，从而能够利用大规模多视角数据集学习到的强几何先验。
◆ 设计了一个双向特征融合模块，用于增强跨视图一致性，从而构建出高效且几何可靠的显式3D高斯溅射支架。
◆ 创新性地使用该3D一致的几何支架作为强先验，来驱动新颖视图生成器，使得即使在较大相机运动下也能生成照片般真实且几何准确的视图，支持沉浸式场景探索。</td></tr>
<tr><td>2026-02-23</td><td>Training Deep Stereo Matching Networks on Tree Branch Imagery: A Benchmark Study for Real-Time UAV Forestry Applications</td><td>[2602.19763](http://arxiv.org/pdf/2602.19763)</td><td>本文首次针对真实树木枝干图像训练并测试了十种深度立体匹配网络，旨在为无人机林业应用提供实时深度估计方案。研究基于Canterbury Tree Branches数据集，使用DEFOM生成的视差图作为训练基准，系统评估了各类网络在植被场景下的性能。

◆ 首次在真实树木枝干立体图像数据集上对十种深度立体匹配网络进行全面的训练与测试，填补了该领域针对性研究的空白。
◆ 采用DEFOM生成的视差图作为高质量训练目标，确保了植被场景下监督信号的可靠性。
◆ 综合运用感知指标（SSIM、LPIPS、ViTScore）与结构指标（SIFT/ORB特征匹配）进行多维评估，发现BANet-3D综合质量最优，而RAFT-Stereo在场景理解上最佳。
◆ 在无人机搭载的Jetson Orin硬件上进行实时性测试，指出AnyNet在1080P下可达6.99 FPS，是唯一的近实时方案，而BANet-2D在质量与速度间取得了最佳平衡。
◆ 对比了720P与1080P的处理耗时，为实际林业无人机系统的分辨率选择提供了实用指导。</td></tr>
<tr><td>2026-02-21</td><td>Marginalized Bundle Adjustment: Multi-View Camera Pose from Monocular Depth Estimates</td><td>[2602.18906](http://arxiv.org/pdf/2602.18906)</td><td>该论文的核心贡献是提出了一种名为“边缘化光束法平差”的新方法，成功将单目深度估计集成到传统运动恢复结构中，解决了其深度图误差方差大的挑战。

◆ 创新性地提出了边缘化光束法平差方法，其灵感来源于现代RANSAC估计器，旨在利用单目深度估计的密度优势来抑制其较大的误差方差。
◆ 首次系统性地论证了单目深度估计图虽然存在噪声，但其精度已足以支持高质量的多视图相机位姿估计，为运动恢复结构任务提供了新的数据源。
◆ 该方法在性能上实现了突破，在运动恢复结构和相机重定位任务中达到了先进或具有竞争力的水平。
◆ 通过大量实验验证了方法的鲁棒性和广泛适用性，其性能在不同规模的数据集上均表现一致，既能处理少量图像，也能处理包含数千张图像的大型系统。
◆ 整体工作凸显了单目深度估计技术在多视图三维视觉中的巨大应用潜力，为领域发展提供了新方向。</td></tr>
<tr><td>2026-02-20</td><td>A Single Image and Multimodality Is All You Need for Novel View Synthesis</td><td>[2602.17909](http://arxiv.org/pdf/2602.17909)</td><td>该论文的核心贡献在于提出了一种利用极稀疏多模态测距数据来显著提升单图像新视角合成质量的方法，以解决传统单目深度估计在复杂场景中不可靠的问题。

◆ 创新性地将极稀疏的雷达或激光雷达等测距数据引入单图像新视角合成任务，以多模态融合克服纯视觉深度估计在弱纹理、遮挡等场景下的局限。
◆ 提出一种多模态深度重建框架，采用基于角域的局部高斯过程建模，能够利用稀疏数据生成密集深度图并明确量化不确定性。
◆ 该方法重建的深度与不确定性可直接作为几何条件，无缝替换现有基于扩散模型的渲染流程中的单目深度估计器，无需改动生成模型本身。
◆ 在真实世界多模态驾驶场景上的实验证明，该方案能大幅提升单图像新视角视频生成的几何一致性和视觉质量，凸显了可靠几何先验的重要性。
◆ 研究结果证明了即使在数据极端稀疏的情况下，多模态感知也能为基于扩散模型的视图合成带来显著的实用效益。</td></tr>
<tr><td>2026-02-19</td><td>Multi-Modal Monocular Endoscopic Depth and Pose Estimation with Edge-Guided Self-Supervision</td><td>[2602.17785](http://arxiv.org/pdf/2602.17785)</td><td>该论文提出了一种名为PRISM的自监督学习框架，用于解决单目内窥镜深度与姿态估计的难题。其核心贡献在于通过引入解剖和光照先验来引导几何学习，显著提升了在纹理缺失、光照复杂场景下的性能。

◆ 创新性地整合了边缘检测与亮度解耦，为几何学习提供结构引导。使用学习型边缘检测器获取高频薄边缘图，同时通过本征分解模块分离着色与反射成分。
◆ 利用分解得到的着色线索辅助深度估计，使模型能更有效地利用光照信息。
◆ 通过全面的训练数据消融研究，得出了两项重要实践发现：在真实数据上的自监督训练优于在仿真模型数据上的有监督训练；视频帧率是影响性能的关键因素，需根据数据集进行特定采样。
◆ 在多个真实与合成数据集上实现了最先进的性能，验证了框架的有效性。</td></tr>
<tr><td>2026-02-18</td><td>StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation</td><td>[2602.16915](http://arxiv.org/pdf/2602.16915)</td><td>该论文的核心贡献是提出了一种名为StereoAdapter-2的新型水下立体深度估计框架，旨在解决因光线衰减和散射导致的水下域适应难题。

◆ 提出了一种创新的ConvSS2D更新算子，以替代传统ConvGRU。该算子基于选择性状态空间模型，能以线性计算复杂度实现高效的长距离空间传播。
◆ 设计了一种四向扫描策略，该策略与对极几何自然对齐，并能捕捉垂直结构一致性，从而在单次更新中有效处理大视差和无纹理区域。
◆ 构建了大规模合成水下立体数据集UW-StereoDepth-80K。该数据集通过结合语义感知风格迁移和几何一致的新视图合成的两阶段生成流程创建，包含多样化的基线、衰减和散射参数。
◆ 继承了动态LoRA适配机制，并结合上述创新，使框架在多个水下基准测试上取得了零样本状态最优性能，并在真实BlueROV2平台上验证了其鲁棒性。</td></tr>
<tr><td>2026-02-18</td><td>Breaking the Sub-Millimeter Barrier: Eyeframe Acquisition from Color Images</td><td>[2602.16281](http://arxiv.org/pdf/2602.16281)</td><td>该论文的核心贡献是提出了一种基于人工智能视觉的全新方法，用于从彩色图像中高精度地获取眼镜框轮廓数据，以替代传统低效的机械测量方式。

◆ 创新性地采用基于人工智能视觉的多视图信息处理方法，取代了依赖精密机械工具的传统镜框测量技术。
◆ 提出了一套完整的自动化处理流程，包括图像采集、镜框分割、深度估计以及多视图融合处理，直接从静态彩色图像中获取三维空间信息。
◆ 通过将分割后的RGB图像与深度数据集成，实现了对镜框轮廓的精确测量，精度达到亚毫米级，满足了光学镜片配装的高标准要求。
◆ 该方法省去了对专用追踪设备和复杂校准流程的依赖，显著简化了验光师的工作流程，提高了工作效率。
◆ 在真实数据上对所提算法的不同配置和变体进行了验证与分析，结果表明其性能可与现有解决方案竞争，同时更具便捷性。</td></tr>
<tr><td>2026-02-16</td><td>AnchorWeave: World-Consistent Video Generation with Retrieved Local Spatial Memories</td><td>[2602.14941](http://arxiv.org/pdf/2602.14941)</td><td>该论文针对长时序相机可控视频生成中空间世界一致性的难题，提出了一种新的记忆增强框架AnchorWeave。其核心贡献与创新点如下：

◆ 摒弃了易受多视角错位影响的全局三维场景重建方法，转而采用多个干净的局部几何记忆作为条件。
◆ 提出了覆盖度驱动的局部记忆检索机制，根据目标相机轨迹动态选择最相关的局部记忆片段。
◆ 设计了多锚点编织控制器，在生成过程中学习协调并融合所选局部记忆之间的跨视角不一致性。
◆ 通过局部几何条件、多锚点控制和覆盖度检索的协同作用，有效减少了累积误差，显著提升了长时序场景的空间一致性。
◆ 大量实验证明，该方法在保持高视觉质量的同时，大幅改善了长期一致性，并通过消融研究验证了各关键组件的有效性。</td></tr>
<tr><td>2026-02-15</td><td>DenseMLLM: Standard Multimodal LLMs are Intrinsic Dense Predictors</td><td>[2602.14134](http://arxiv.org/pdf/2602.14134)</td><td>该论文的核心贡献在于提出了一种名为DenseMLLM的方法，它成功地将标准多模态大语言模型直接用于密集预测任务，而无需依赖传统的任务特定解码器。

◆ 提出了一种新颖的范式，挑战了现有需要为密集任务添加复杂专用解码器的做法，证明了标准MLLM架构本身具备密集预测的内在潜力。
◆ 设计了一种创新的视觉令牌监督策略，使模型能够同时处理多种标签和任务，从而在一个统一框架内支持如语义分割、深度估计等多种密集预测。
◆ 保持了模型的极简和通用性设计，无需针对不同任务进行架构专门化，这强化了MLLM作为通用智能体的实用性。
◆ 该方法在多个密集预测和视觉语言基准测试中取得了极具竞争力的性能，验证了其有效性。
◆ 这项工作为构建更统一、实用的多模态基础模型指明了方向，减少了模型复杂性和碎片化。</td></tr>
<tr><td>2026-02-10</td><td>Sim2Radar: Toward Bridging the Radar Sim-to-Real Gap with VLM-Guided Scene Reconstruction</td><td>[2602.13314](http://arxiv.org/pdf/2602.13314)</td><td>Sim2Radar的核心贡献是提出了一种从单张RGB图像直接合成毫米波雷达训练数据的端到端框架，旨在解决雷达感知任务中真实数据稀缺且标注成本高昂的瓶颈。其创新点主要体现在：

◆ 首创了从单视角RGB图像到雷达数据合成的完整流程，无需人工3D场景建模，实现了可扩展的雷达数据生成。
◆ 采用视觉语言模型进行场景理解与推理，结合单目深度估计与分割技术，重建出具有材料感知能力的3D场景。
◆ 利用基于物理的射线追踪器进行毫米波传播模拟，并通过ITU-R电磁参数化的菲涅尔反射模型来配置，实现了高保真的雷达信号合成。
◆ 通过迁移学习验证了合成数据的有效性：在合成数据上预训练雷达点云目标检测模型，再于少量真实数据上微调，显著提升了在真实室内场景中的3D检测性能。
◆ 证明了基于物理的视觉驱动雷达仿真能为雷达学习提供有效的几何先验，在真实数据有限的情况下可显著提升感知模型的定位精度与整体性能。</td></tr>
<tr><td>2026-02-12</td><td>GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry</td><td>[2602.11714](http://arxiv.org/pdf/2602.11714)</td><td>GSO-SLAM的核心贡献是提出了一种新颖的双向耦合单目密集SLAM系统，它通过创新的集成框架，在实时运行的同时实现了卓越的场景重建与跟踪精度。

◆ 核心创新在于提出了视觉里程计（VO）与高斯泼溅（GS）场景表示之间的双向耦合机制，克服了现有方法在计算成本或冗余集成上的缺陷。
◆ 该方法在期望最大化（EM）框架内制定了联合优化，能够同步优化VO产生的半稠密深度估计与GS表示，且不引入额外计算开销。
◆ 提出了高斯泼溅初始化技术，直接利用VO提供的图像信息、关键帧位姿和像素关联来生成接近最终效果的高斯场景初始状态，从而避免了对启发式方法的依赖。
◆ 整个系统在保持实时运行效率的前提下，在场景重建的几何/光度保真度以及跟踪精度方面均达到了先进水平。</td></tr>
<tr><td>2026-02-11</td><td>MDE-VIO: Enhancing Visual-Inertial Odometry Using Learned Depth Priors</td><td>[2602.11323](http://arxiv.org/pdf/2602.11323)</td><td>该论文提出了一种将学习式深度先验融入视觉惯性里程计的新方法，旨在解决传统单目VIO在低纹理环境中因特征稀疏导致的性能下降问题。

◆ 核心创新在于将学习到的稠密深度先验直接集成到经典的VINS-Mono优化后端中，而非使用计算复杂的端到端模型，从而在边缘设备上实现了实时运行。
◆ 提出了一种新颖的优化框架，该框架通过强制施加仿射不变深度一致性约束和成对顺序约束，有效利用了学习深度的几何信息。
◆ 设计了基于方差的选通机制，能够显式地过滤学习深度图中不稳定的伪影，增强了系统的鲁棒性。
◆ 整个方法在严格遵循边缘设备算力限制的前提下，鲁棒地恢复了轨迹的度量尺度。
实验表明，该方法在挑战性场景中能防止系统发散，并在TartanGround和M3ED数据集上显著提升了精度，绝对轨迹误差最高降低了28.3%。</td></tr>
<tr><td>2026-02-11</td><td>PuriLight: A Lightweight Shuffle and Purification Framework for Monocular Depth Estimation</td><td>[2602.11066](http://arxiv.org/pdf/2602.11066)</td><td>该论文提出名为PuriLight的轻量级自监督单目深度估计框架，旨在同时解决计算效率与细节保留的双重挑战。其核心贡献在于设计了一个三阶段架构，通过三个创新模块协同工作，实现了在极低参数量下保持高精度与高效率。

◆ 提出Shuffle-Dilation Convolution（SDC）模块，用于高效的局部特征提取，平衡感受野与计算负担。
◆ 设计Rotation-Adaptive Kernel Attention（RAKA）模块，通过自适应注意力机制增强分层特征表示能力。
◆ 引入Deep Frequency Signal Purification（DFSP）模块，在频域进行全局特征纯化，提升细节保留与结构精度。
◆ 整体框架将上述模块有效整合，实现了轻量化与高精度特征处理的统一，在自监督条件下减少了对真实标注数据的依赖。
◆ 大量实验验证该框架以极少的训练参数达到了先进性能，同时保持了优异的计算效率，推动了实用化轻量深度估计模型的发展。</td></tr>
<tr><td>2026-02-11</td><td>Interpretable Vision Transformers in Monocular Depth Estimation via SVDA</td><td>[2602.11005](http://arxiv.org/pdf/2602.11005)</td><td>该论文的核心贡献是将一种新型的、可解释的注意力机制引入单目深度估计任务，重新定义了Transformer模型在该领域的透明度。

◆ 提出了SVD启发式注意力机制，首次为密集预测任务提供了具有光谱结构的注意力公式化表达。
◆ 该机制通过在学习到的对角矩阵中嵌入归一化的查询-键交互，将方向对齐与光谱调制解耦，使注意力图本身具有内在可解释性，而非事后近似。
◆ 在保持甚至略微提升预测精度的同时，该方法仅增加了微小的计算开销，在KITTI和NYU-v2数据集上得到验证。
◆ 其关键创新在于衍生出六个可量化的光谱指标，用于评估注意力机制的熵、秩、稀疏性、对齐性、选择性和鲁棒性。
◆ 这些指标揭示了训练过程中注意力组织方式具有一致的跨数据集和深度维度模式，这是标准Transformer模型无法提供的洞察。
◆ 总体而言，该方法将注意力的角色从不透明的机制转变为可量化的描述符，为构建透明的密集预测模型开辟了新途径。</td></tr>
<tr><td>2026-02-11</td><td>AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models</td><td>[2602.10698](http://arxiv.org/pdf/2602.10698)</td><td>该论文针对现有视觉-语言-动作模型因依赖二维图像训练而导致三维空间理解不足的问题，提出了一种融合深度信息以增强三维特征表征的新框架。其核心贡献与创新点如下：

◆ 提出将深度估计集成到视觉-语言-动作模型中的新框架，利用名为VGGT的深度估计基线，从标准RGB输入中提取几何感知的三维线索，从而在无需大规模三维数据集的情况下隐式恢复三维结构信息。

◆ 引入一个称为“动作助手”的新模块，该模块利用动作先验知识对学习到的三维表征进行约束，确保其与下游控制任务保持一致，从而提升了深度特征的可信度与实用性。

◆ 通过将增强后的三维特征与传统的二维视觉标记相融合，显著提高了模型在复杂三维环境中的泛化能力和鲁棒性，有效弥补了二维观测与三维感知决策之间的差距。

实验结果表明，该方法不仅增强了模型在几何模糊场景中的感知能力，还实现了更优的动作预测精度。这项工作凸显了深度驱动数据增强与辅助专家监督在机器人系统二维到三维感知升级中的潜力。</td></tr>
<tr><td>2026-02-10</td><td>VersaViT: Enhancing MLLM Vision Backbones via Task-Guided Optimization</td><td>[2602.09934](http://arxiv.org/pdf/2602.09934)</td><td>该论文针对MLLM视觉编码器在密集预测任务上表现不足的问题，提出了一种增强视觉主干网络通用性的方法。其核心贡献与创新点如下：

◆ 揭示了当前多模态大语言模型（MLLMs）中视觉编码器的局限性，即其虽然具备高层语义对齐能力，但在语义分割、深度估计等需要密集像素级理解的任务上表现欠佳。

◆ 提出了VersaViT，一个全面的视觉Transformer架构。它实例化了一个新颖的多任务协作后训练框架，旨在提升视觉主干网络的通用能力。

◆ 该框架通过轻量级的任务头，并利用多粒度监督信号，来协同优化视觉主干网络，从而使其能够同时适应以语言为中介的推理任务和像素级的视觉理解任务。

◆ 通过在下游多种任务上的广泛实验，验证了该方法的有效性，最终得到了一个既服务于高级语义对齐又能可靠执行经典视觉任务的通用视觉主干网络。</td></tr>
<tr><td>2026-02-10</td><td>RAD: Retrieval-Augmented Monocular Metric Depth Estimation for Underrepresented Classes</td><td>[2602.09532](http://arxiv.org/pdf/2602.09532)</td><td>该论文提出了一种检索增强的单目度量深度估计框架RAD，旨在解决复杂场景中少数类别物体深度估计不准的难题。其核心贡献与创新点如下：

◆ 提出检索增强框架，通过检索相似场景的RGB-D样本作为几何代理，模拟多视图立体视觉的优势，增强对少数类别的深度感知。

◆ 设计不确定性感知检索机制，能自动识别输入图像中低置信度区域，并针对性地检索包含相似语义内容的RGB-D上下文样本。

◆ 采用双流网络架构处理输入图像与检索样本，并通过匹配交叉注意力模块进行融合，该模块仅依赖可靠的点对应关系传递几何信息，避免错误匹配干扰。

◆ 在多个标准数据集上验证了框架的有效性，显著提升了少数类别的深度估计精度，如在NYU Depth v2上相对绝对误差降低29.2%，同时保持主流类别的竞争力。</td></tr>
<tr><td>2026-02-09</td><td>Forest canopy height estimation from satellite RGB imagery using large-scale airborne LiDAR-derived training data and monocular depth estimation</td><td>[2602.06503](http://arxiv.org/pdf/2602.06503)</td><td>◆ Large-scale, high-resolution forest canopy height mapping plays a crucial role in understanding regional and global carbon and water cycles.
◆ Spaceborne LiDAR missions, including the Ice, Cloud, and Land Elevation Satellite-2 (ICESat-2) and the Global Ecosystem Dynamics Investigation (GEDI), provide global observations of forest structure but are spatially sparse and subject to inherent uncertainties.
◆ In contrast, near-surface LiDAR platforms, such as airborne and unmanned aerial vehicle (UAV) LiDAR systems, offer much finer measurements of forest canopy structure, and a growing number of countries have made these datasets openly available.</td></tr>
<tr><td>2026-02-06</td><td>Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels</td><td>[2602.06382](http://arxiv.org/pdf/2602.06382)</td><td>◆ Achieving robust vision-based humanoid locomotion remains challenging due to two fundamental issues: the sim-to-real gap introduces significant perception noise that degrades performance on fine-grained tasks, and training a unified policy across diverse terrains is hindered by conflicting learning objectives.
◆ To address these challenges, we present an end-to-end framework for vision-driven humanoid locomotion.
◆ For robust sim-to-real transfer, we develop a high-fidelity depth sensor simulation that captures stereo matching artifacts and calibration uncertainties inherent in real-world sensing.</td></tr>
<tr><td>2026-02-05</td><td>AnyThermal: Towards Learning Universal Representations for Thermal Perception</td><td>[2602.06203](http://arxiv.org/pdf/2602.06203)</td><td>◆ We present AnyThermal, a thermal backbone that captures robust task-agnostic thermal features suitable for a variety of tasks such as cross-modal place recognition, thermal segmentation, and monocular depth estimation using thermal images.
◆ Existing thermal backbones that follow task-specific training from small-scale data result in utility limited to a specific environment and task.
◆ Unlike prior methods, AnyThermal can be used for a wide range of environments (indoor, aerial, off-road, urban) and tasks, all without task-specific training.</td></tr>
<tr><td>2026-02-11</td><td>Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation</td><td>[2602.06032](http://arxiv.org/pdf/2602.06032)</td><td>◆ Vision Foundation Models (VFMs) have achieved remarkable success when applied to various downstream 2D tasks.
◆ Despite their effectiveness, they often exhibit a critical lack of 3D awareness.
◆ To this end, we introduce Splat and Distill, a framework that instills robust 3D awareness into 2D VFMs by augmenting the teacher model with a fast, feed-forward 3D reconstruction pipeline.</td></tr>
<tr><td>2026-02-05</td><td>Depth as Prior Knowledge for Object Detection</td><td>[2602.05730](http://arxiv.org/pdf/2602.05730)</td><td>◆ Detecting small and distant objects remains challenging for object detectors due to scale variation, low resolution, and background clutter.
◆ Safety-critical applications require reliable detection of these objects for safe planning.
◆ Depth information can improve detection, but existing approaches require complex, model-specific architectural modifications.</td></tr>
<tr><td>2026-02-05</td><td>UniSurg: A Video-Native Foundation Model for Universal Understanding of Surgical Videos</td><td>[2602.05638](http://arxiv.org/pdf/2602.05638)</td><td>◆ While foundation models have advanced surgical video analysis, current approaches rely predominantly on pixel-level reconstruction objectives that waste model capacity on low-level visual details - such as smoke, specular reflections, and fluid motion - rather than semantic structures essential for surgical understanding.
◆ We present UniSurg, a video-native foundation model that shifts the learning paradigm from pixel-level reconstruction to latent motion prediction.
◆ Built on the Video Joint Embedding Predictive Architecture (V-JEPA), UniSurg introduces three key technical innovations tailored to surgical videos: 1) motion-guided latent prediction to prioritize semantically meaningful regions, 2) spatiotemporal affinity self-distillation to enforce relational consistency, and 3) feature diversity regularization to prevent representation collapse in texture-sparse surgical scenes.</td></tr>
<tr><td>2026-02-05</td><td>Depth estimation of a monoharmonic source using a vertical linear array at fixed distance</td><td>[2602.05560](http://arxiv.org/pdf/2602.05560)</td><td>◆ Estimating the depth of a monoharmonic sound source at a fixed range using a vertical linear array (VLA) is challenging in the absence of seabed environmental parameters, and relevant research remains scarce.
◆ The orthogonality constrained modal search based depth estimation (OCMS-D) method is proposed in this paper, which enables the estimation of the depth of a monoharmonic source at a fixed range using a VLA under unknown seabed parameters.
◆ Using the sparsity of propagating normal modes and the orthogonality of mode depth functions, OCMS-D estimates the normal mode parameters under a fixed source-array distance at first.</td></tr>
<tr><td>2026-02-05</td><td>NeVStereo: A NeRF-Driven NVS-Stereo Architecture for High-Fidelity 3D Tasks</td><td>[2602.05423](http://arxiv.org/pdf/2602.05423)</td><td>◆ In modern dense 3D reconstruction, feed-forward systems (e.g., VGGT, pi3) focus on end-to-end matching and geometry prediction but do not explicitly output the novel view synthesis (NVS).
◆ Neural rendering-based approaches offer high-fidelity NVS and detailed geometry from posed images, yet they typically assume fixed camera poses and can be sensitive to pose errors.
◆ As a result, it remains non-trivial to obtain a single framework that can offer accurate poses, reliable depth, high-quality rendering, and accurate 3D surfaces from casually captured views.</td></tr>
<tr><td>2026-02-05</td><td>PoseGaussian: Pose-Driven Novel View Synthesis for Robust 3D Human Reconstruction</td><td>[2602.05190](http://arxiv.org/pdf/2602.05190)</td><td>◆ We propose PoseGaussian, a pose-guided Gaussian Splatting framework for high-fidelity human novel view synthesis.
◆ Human body pose serves a dual purpose in our design: as a structural prior, it is fused with a color encoder to refine depth estimation; as a temporal cue, it is processed by a dedicated pose encoder to enhance temporal consistency across frames.
◆ These components are integrated into a fully differentiable, end-to-end trainable pipeline.</td></tr>
<tr><td>2026-02-03</td><td>Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal</td><td>[2602.04053](http://arxiv.org/pdf/2602.04053)</td><td>◆ We present SeeingThroughClutter, a method for reconstructing structured 3D representations from single images by segmenting and modeling objects individually.
◆ Prior approaches rely on intermediate tasks such as semantic segmentation and depth estimation, which often underperform in complex scenes, particularly in the presence of occlusion and clutter.
◆ We address this by introducing an iterative object removal and reconstruction pipeline that decomposes complex scenes into a sequence of simpler subtasks.</td></tr>
<tr><td>2026-02-03</td><td>Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements</td><td>[2602.03209](http://arxiv.org/pdf/2602.03209)</td><td>◆ Autonomous field robots operating in unstructured environments require robust perception to ensure safe and reliable operations.
◆ Recent advances in monocular depth estimation have demonstrated the potential of low-cost cameras as depth sensors; however, their adoption in field robotics remains limited due to the absence of reliable scale cues, ambiguous or low-texture conditions, and the scarcity of large-scale datasets.
◆ To address these challenges, we propose a depth completion model that trains on synthetic data and uses extremely sparse measurements from depth sensors to predict dense metric depth in unseen field robotics environments.</td></tr>
<tr><td>2026-02-02</td><td>Multi-Task Learning for Robot Perception with Imbalanced Data</td><td>[2602.01899](http://arxiv.org/pdf/2602.01899)</td><td>◆ Multi-task problem solving has been shown to improve the accuracy of the individual tasks, which is an important feature for robots, as they have a limited resource.
◆ However, when the number of labels for each task is not equal, namely imbalanced data exist, a problem may arise due to insufficient number of samples, and labeling is not very easy for mobile robots in every environment.
◆ We propose a method that can learn tasks even in the absence of the ground truth labels for some of the tasks.</td></tr>
<tr><td>2026-02-01</td><td>OASIS-DC: Generalizable Depth Completion via Output-level Alignment of Sparse-Integrated Monocular Pseudo Depth</td><td>[2602.01268](http://arxiv.org/pdf/2602.01268)</td><td>◆ Recent monocular foundation models excel at zero-shot depth estimation, yet their outputs are inherently relative rather than metric, limiting direct use in robotics and autonomous driving.
◆ We leverage the fact that relative depth preserves global layout and boundaries: by calibrating it with sparse range measurements, we transform it into a pseudo metric depth prior.
◆ Building on this prior, we design a refinement network that follows the prior where reliable and deviates where necessary, enabling accurate metric predictions from very few labeled samples.</td></tr>
<tr><td>2026-01-30</td><td>Deep in the Jungle: Towards Automating Chimpanzee Population Estimation</td><td>[2601.22917](http://arxiv.org/pdf/2601.22917)</td><td>◆ The estimation of abundance and density in unmarked populations of great apes relies on statistical frameworks that require animal-to-camera distance measurements.
◆ In practice, acquiring these distances depends on labour-intensive manual interpretation of animal observations across large camera trap video corpora.
◆ This study introduces and evaluates an only sparsely explored alternative: the integration of computer vision-based monocular depth estimation (MDE) pipelines directly into ecological camera trap workflows for great ape conservation.</td></tr>
<tr><td>2026-01-30</td><td>Diachronic Stereo Matching for Multi-Date Satellite Imagery</td><td>[2601.22808](http://arxiv.org/pdf/2601.22808)</td><td>◆ Recent advances in image-based satellite 3D reconstruction have progressed along two complementary directions.
◆ On one hand, multi-date approaches using NeRF or Gaussian-splatting jointly model appearance and geometry across many acquisitions, achieving accurate reconstructions on opportunistic imagery with numerous observations.
◆ On the other hand, classical stereoscopic reconstruction pipelines deliver robust and scalable results for simultaneous or quasi-simultaneous image pairs.</td></tr>
<tr><td>2026-01-30</td><td>High-Definition 5MP Stereo Vision Sensing for Robotics</td><td>[2601.22445](http://arxiv.org/pdf/2601.22445)</td><td>◆ High-resolution (5MP+) stereo vision systems are essential for advancing robotic capabilities, enabling operation over longer ranges and generating significantly denser and accurate 3D point clouds.
◆ However, realizing the full potential of high-angular-resolution sensors requires a commensurately higher level of calibration accuracy and faster processing -- requirements often unmet by conventional methods.
◆ This study addresses that critical gap by processing 5MP camera imagery using a novel, advanced frame-to-frame calibration and stereo matching methodology designed to achieve both high accuracy and speed.</td></tr>
<tr><td>2026-01-29</td><td>MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources</td><td>[2601.22054](http://arxiv.org/pdf/2601.22054)</td><td>◆ Scaling has powered recent advances in vision foundation models, yet extending this paradigm to metric depth estimation remains challenging due to heterogeneous sensor noise, camera-dependent biases, and metric ambiguity in noisy cross-source 3D data.
◆ We introduce Metric Anything, a simple and scalable pretraining framework that learns metric depth from noisy, diverse 3D sources without manually engineered prompts, camera-specific modeling, or task-specific architectures.
◆ Central to our approach is the Sparse Metric Prompt, created by randomly masking depth maps, which serves as a universal interface that decouples spatial reasoning from sensor and camera biases.</td></tr>
<tr><td>2026-01-29</td><td>Belief Propagation Converges to Gaussian Distributions in Sparsely-Connected Factor Graphs</td><td>[2601.21935](http://arxiv.org/pdf/2601.21935)</td><td>◆ Belief Propagation (BP) is a powerful algorithm for distributed inference in probabilistic graphical models, however it quickly becomes infeasible for practical compute and memory budgets.
◆ Many efficient, non-parametric forms of BP have been developed, but the most popular is Gaussian Belief Propagation (GBP), a variant that assumes all distributions are locally Gaussian.
◆ GBP is widely used due to its efficiency and empirically strong performance in applications like computer vision or sensor networks - even when modelling non-Gaussian problems.</td></tr>
<tr><td>2026-01-28</td><td>GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction</td><td>[2601.20331](http://arxiv.org/pdf/2601.20331)</td><td>◆ 3D Gaussian Splatting enables efficient optimization and high-quality rendering, yet accurate surface reconstruction remains challenging.
◆ Prior methods improve surface reconstruction by refining Gaussian depth estimates, either via multi-view geometric consistency or through monocular depth priors.
◆ However, multi-view constraints become unreliable under large geometric discrepancies, while monocular priors suffer from scale ambiguity and local inconsistency, ultimately leading to inaccurate Gaussian depth supervision.</td></tr>
<tr><td>2026-01-28</td><td>Physically Guided Visual Mass Estimation from a Single RGB Image</td><td>[2601.20303](http://arxiv.org/pdf/2601.20303)</td><td>◆ Estimating object mass from visual input is challenging because mass depends jointly on geometric volume and material-dependent density, neither of which is directly observable from RGB appearance.
◆ Consequently, mass prediction from pixels is ill-posed and therefore benefits from physically meaningful representations to constrain the space of plausible solutions.
◆ We propose a physically structured framework for single-image mass estimation that addresses this ambiguity by aligning visual cues with the physical factors governing mass.</td></tr>
<tr><td>2026-01-28</td><td>Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction</td><td>[2601.19489](http://arxiv.org/pdf/2601.19489)</td><td>◆ We present a fast 3DGS reconstruction pipeline designed to converge within one minute, developed for the SIGGRAPH Asia 3DGS Fast Reconstruction Challenge.
◆ The challenge consists of an initial round using SLAM-generated camera poses (with noisy trajectories) and a final round using COLMAP poses (highly accurate).
◆ To robustly handle these heterogeneous settings, we develop a two-stage solution.</td></tr>
<tr><td>2026-01-27</td><td>Towards Gold-Standard Depth Estimation for Tree Branches in UAV Forestry: Benchmarking Deep Stereo Matching Methods</td><td>[2601.19461](http://arxiv.org/pdf/2601.19461)</td><td>◆ Autonomous UAV forestry operations require robust depth estimation with strong cross-domain generalization, yet existing evaluations focus on urban and indoor scenarios, leaving a critical gap for vegetation-dense environments.
◆ We present the first systematic zero-shot evaluation of eight stereo methods spanning iterative refinement, foundation model, diffusion-based, and 3D CNN paradigms.
◆ All methods use officially released pretrained weights (trained on Scene Flow) and are evaluated on four standard benchmarks (ETH3D, KITTI 2012/2015, Middlebury) plus a novel 5,313-pair Canterbury Tree Branches dataset ($1920 \times 1080$).</td></tr>
<tr><td>2026-01-27</td><td>MIRAGE: Enabling Real-Time Automotive Mediated Reality</td><td>[2601.19385](http://arxiv.org/pdf/2601.19385)</td><td>◆ Traffic is inherently dangerous, with around 1.19 million fatalities annually.
◆ Automotive Mediated Reality (AMR) can enhance driving safety by overlaying critical information (e.g., outlines, icons, text) on key objects to improve awareness, altering objects&#x27; appearance to simplify traffic situations, and diminishing their appearance to minimize distractions.
◆ However, real-world AMR evaluation remains limited due to technical challenges.</td></tr>
<tr><td>2026-01-27</td><td>Instance-Guided Radar Depth Estimation for 3D Object Detection</td><td>[2601.19314](http://arxiv.org/pdf/2601.19314)</td><td>◆ Accurate depth estimation is fundamental to 3D perception in autonomous driving, supporting tasks such as detection, tracking, and motion planning.
◆ However, monocular camera-based 3D detection suffers from depth ambiguity and reduced robustness under challenging conditions.
◆ Radar provides complementary advantages such as resilience to poor lighting and adverse weather, but its sparsity and low resolution limit its direct use in detection frameworks.</td></tr>
<tr><td>2026-01-26</td><td>On the Role of Depth in Surgical Vision Foundation Models: An Empirical Study of RGB-D Pre-training</td><td>[2601.18929](http://arxiv.org/pdf/2601.18929)</td><td>◆ Vision foundation models (VFMs) have emerged as powerful tools for surgical scene understanding.
◆ However, current approaches predominantly rely on unimodal RGB pre-training, overlooking the complex 3D geometry inherent to surgical environments.
◆ Although several architectures support multimodal or geometry-aware inputs in general computer vision, the benefits of incorporating depth information in surgical settings remain underexplored.</td></tr>
<tr><td>2026-01-25</td><td>SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation</td><td>[2601.17657](http://arxiv.org/pdf/2601.17657)</td><td>◆ Contrastive Language-Image Pre-training (CLIP) has accomplished extraordinary success for semantic understanding but inherently struggles to perceive geometric structure.
◆ Existing methods attempt to bridge this gap by querying CLIP with textual prompts, a process that is often indirect and inefficient.
◆ This paper introduces a fundamentally different approach using a dual-pathway decoder.</td></tr>
<tr><td>2026-01-24</td><td>AsterNav: Autonomous Aerial Robot Navigation In Darkness Using Passive Computation</td><td>[2601.17550](http://arxiv.org/pdf/2601.17550)</td><td>◆ Autonomous aerial navigation in absolute darkness is crucial for post-disaster search and rescue operations, which often occur from disaster-zone power outages.
◆ Yet, due to resource constraints, tiny aerial robots, perfectly suited for these operations, are unable to navigate in the darkness to find survivors safely.
◆ In this paper, we present an autonomous aerial robot for navigation in the dark by combining an Infra-Red (IR) monocular camera with a large-aperture coded lens and structured light without external infrastructure like GPS or motion-capture.</td></tr>
<tr><td>2026-01-24</td><td>Cross360: 360° Monocular Depth Estimation via Cross Projections Across Scales</td><td>[2601.17271](http://arxiv.org/pdf/2601.17271)</td><td>◆ 360° depth estimation is a challenging research problem due to the difficulty of finding a representation that both preserves global continuity and avoids distortion in spherical images.
◆ Existing methods attempt to leverage complementary information from multiple projections, but struggle with balancing global and local consistency.
◆ Their local patch features have limited global perception, and the combined global representation does not address discrepancies in feature extraction at the boundaries between patches.</td></tr>
<tr><td>2026-01-20</td><td>Atomic Depth Estimation From Noisy Electron Microscopy Data Via Deep Learning</td><td>[2601.17046](http://arxiv.org/pdf/2601.17046)</td><td>◆ We present a novel approach for extracting 3D atomic-level information from transmission electron microscopy (TEM) images affected by significant noise.
◆ The approach is based on formulating depth estimation as a semantic segmentation problem.
◆ We address the resulting segmentation problem by training a deep convolutional neural network to generate pixel-wise depth segmentation maps using simulated data corrupted by synthetic noise.</td></tr>
<tr><td>2026-01-26</td><td>AnchoredDream: Zero-Shot 360° Indoor Scene Generation from a Single View via Geometric Grounding</td><td>[2601.16532](http://arxiv.org/pdf/2601.16532)</td><td>◆ Single-view indoor scene generation plays a crucial role in a range of real-world applications.
◆ However, generating a complete 360° scene from a single image remains a highly ill-posed and challenging problem.
◆ Recent approaches have made progress by leveraging diffusion models and depth estimation networks, yet they still struggle to maintain appearance consistency and geometric plausibility under large viewpoint changes, limiting their effectiveness in full-scene generation.</td></tr>
<tr><td>2026-01-21</td><td>RayRoPE: Projective Ray Positional Encoding for Multi-view Attention</td><td>[2601.15275](http://arxiv.org/pdf/2601.15275)</td><td>◆ We study positional encodings for multi-view transformers that process tokens from a set of posed input images, and seek a mechanism that encodes patches uniquely, allows SE(3)-invariant attention with multi-frequency similarity, and can be adaptive to the geometry of the underlying scene.
◆ We find that prior (absolute or relative) encoding schemes for multi-view attention do not meet the above desiderata, and present RayRoPE to address this gap.
◆ RayRoPE represents patch positions based on associated rays but leverages a predicted point along the ray instead of the direction for a geometry-aware encoding.</td></tr>
<tr><td>2026-01-18</td><td>HOT-POT: Optimal Transport for Sparse Stereo Matching</td><td>[2601.12423](http://arxiv.org/pdf/2601.12423)</td><td>◆ Stereo vision between images faces a range of challenges, including occlusions, motion, and camera distortions, across applications in autonomous driving, robotics, and face analysis.
◆ Due to parameter sensitivity, further complications arise for stereo matching with sparse features, such as facial landmarks.
◆ To overcome this ill-posedness and enable unsupervised sparse matching, we consider line constraints of the camera geometry from an optimal transport (OT) viewpoint.</td></tr>
<tr><td>2026-01-16</td><td>studentSplat: Your Student Model Learns Single-view 3D Gaussian Splatting</td><td>[2601.11772](http://arxiv.org/pdf/2601.11772)</td><td>◆ Recent advance in feed-forward 3D Gaussian splatting has enable remarkable multi-view 3D scene reconstruction or single-view 3D object reconstruction but single-view 3D scene reconstruction remain under-explored due to inherited ambiguity in single-view.
◆ We present \textbf{studentSplat}, a single-view 3D Gaussian splatting method for scene reconstruction.
◆ To overcome the scale ambiguity and extrapolation problems inherent in novel-view supervision from a single input, we introduce two techniques: 1) a teacher-student architecture where a multi-view teacher model provides geometric supervision to the single-view student during training, addressing scale ambiguity and encourage geometric validity; and 2) an extrapolation network that completes missing scene context, enabling high-quality extrapolation.</td></tr>
<tr><td>2026-01-16</td><td>SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models</td><td>[2601.11729](http://arxiv.org/pdf/2601.11729)</td><td>◆ Visual Foundation Models (VFMs), such as DINO and CLIP, excel in semantic understanding of images but exhibit limited spatial reasoning capabilities, which limits their applicability to embodied systems.
◆ As a result, recent work incorporates some 3D tasks (such as depth estimation) into VFM training.
◆ However, VFM performance remains inconsistent across other spatial tasks, raising the question of whether these models truly have spatial awareness or overfit to specific 3D objectives.</td></tr>
<tr><td>2026-01-20</td><td>SurfSLAM: Sim-to-Real Underwater Stereo Reconstruction For Real-Time SLAM</td><td>[2601.10814](http://arxiv.org/pdf/2601.10814)</td><td>◆ Localization and mapping are core perceptual capabilities for underwater robots.
◆ Stereo cameras provide a low-cost means of directly estimating metric depth to support these tasks.
◆ However, despite recent advances in stereo depth estimation on land, computing depth from image pairs in underwater scenes remains challenging.</td></tr>
<tr><td>2026-01-16</td><td>NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration</td><td>[2601.09823](http://arxiv.org/pdf/2601.09823)</td><td>◆ Latent diffusion models such as Stable Diffusion 1.5 offer strong generative priors that are highly valuable for image restoration, yet their full pipelines remain too computationally heavy for deployment on edge devices.
◆ Existing lightweight variants predominantly compress the denoising U-Net or reduce the diffusion trajectory, which disrupts the underlying latent manifold and limits generalization beyond a single task.
◆ We introduce NanoSD, a family of Pareto-optimal diffusion foundation models distilled from Stable Diffusion 1.5 through network surgery, feature-wise generative distillation, and structured architectural scaling jointly applied to the U-Net and the VAE encoder-decoder.</td></tr>
<tr><td>2026-01-13</td><td>CogniMap3D: Cognitive 3D Mapping and Rapid Retrieval</td><td>[2601.08175](http://arxiv.org/pdf/2601.08175)</td><td>◆ We present CogniMap3D, a bioinspired framework for dynamic 3D scene understanding and reconstruction that emulates human cognitive processes.
◆ Our approach maintains a persistent memory bank of static scenes, enabling efficient spatial knowledge storage and rapid retrieval.
◆ CogniMap3D integrates three core capabilities: a multi-stage motion cue framework for identifying dynamic objects, a cognitive mapping system for storing, recalling, and updating static scenes across multiple visits, and a factor graph optimization strategy for refining camera poses.</td></tr>
<tr><td>2026-01-18</td><td>UDPNet: Unleashing Depth-based Priors for Robust Image Dehazing</td><td>[2601.06909](http://arxiv.org/pdf/2601.06909)</td><td>◆ Image dehazing has witnessed significant advancements with the development of deep learning models.
◆ However, a few methods predominantly focus on single-modal RGB features, neglecting the inherent correlation between scene depth and haze distribution.
◆ Even those that jointly optimize depth estimation and image dehazing often suffer from suboptimal performance due to inadequate utilization of accurate depth information.</td></tr>
<tr><td>2026-01-20</td><td>GeoSurDepth: Harnessing Foundation Model for Spatial Geometry Consistency-Oriented Self-Supervised Surround-View Depth Estimation</td><td>[2601.05839](http://arxiv.org/pdf/2601.05839)</td><td>◆ Accurate surround-view depth estimation provides a competitive alternative to laser-based sensors and is essential for 3D scene understanding in autonomous driving.
◆ While prior studies have proposed various approaches that primarily focus on enforcing cross-view constraints at the photometric level, few explicitly exploit the rich geometric structure inherent in both monocular and surround-view setting.
◆ In this work, we propose GeoSurDepth, a framework that leverages geometry consistency as the primary cue for surround-view depth estimation.</td></tr>
<tr><td>2026-01-08</td><td>Pixel-Perfect Visual Geometry Estimation</td><td>[2601.05246](http://arxiv.org/pdf/2601.05246)</td><td>◆ Recovering clean and accurate geometry from images is essential for robotics and augmented reality.
◆ However, existing geometry foundation models still suffer severely from flying pixels and the loss of fine details.
◆ In this paper, we present pixel-perfect visual geometry models that can predict high-quality, flying-pixel-free point clouds by leveraging generative modeling in the pixel space.</td></tr>
<tr><td>2026-01-15</td><td>Bayesian Monocular Depth Refinement via Neural Radiance Fields</td><td>[2601.03869](http://arxiv.org/pdf/2601.03869)</td><td>◆ Monocular depth estimation has applications in many fields, such as autonomous navigation and extended reality, making it an essential computer vision task.
◆ However, current methods often produce smooth depth maps that lack the fine geometric detail needed for accurate scene understanding.
◆ We propose MDENeRF, an iterative framework that refines monocular depth estimates using depth information from Neural Radiance Fields (NeRFs).</td></tr>
<tr><td>2026-01-07</td><td>IDESplat: Iterative Depth Probability Estimation for Generalizable 3D Gaussian Splatting</td><td>[2601.03824](http://arxiv.org/pdf/2601.03824)</td><td>◆ Generalizable 3D Gaussian Splatting aims to directly predict Gaussian parameters using a feed-forward network for scene reconstruction.
◆ Among these parameters, Gaussian means are particularly difficult to predict, so depth is usually estimated first and then unprojected to obtain the Gaussian sphere centers.
◆ Existing methods typically rely solely on a single warp to estimate depth probability, which hinders their ability to fully leverage cross-view geometric cues, resulting in unstable and coarse depth maps.</td></tr>
<tr><td>2026-01-06</td><td>Guardians of the Hair: Rescuing Soft Boundaries in Depth, Stereo, and Novel Views</td><td>[2601.03362](http://arxiv.org/pdf/2601.03362)</td><td>◆ Soft boundaries, like thin hairs, are commonly observed in natural and computer-generated imagery, but they remain challenging for 3D vision due to the ambiguous mixing of foreground and background cues.
◆ This paper introduces Guardians of the Hair (HairGuard), a framework designed to recover fine-grained soft boundary details in 3D vision tasks.
◆ Specifically, we first propose a novel data curation pipeline that leverages image matting datasets for training and design a depth fixer network to automatically identify soft boundary regions.</td></tr>
<tr><td>2026-01-06</td><td>VLM4VLA: Revisiting Vision-Language-Models in Vision-Language-Action Models</td><td>[2601.03309](http://arxiv.org/pdf/2601.03309)</td><td>◆ Vision-Language-Action (VLA) models, which integrate pretrained large Vision-Language Models (VLM) into their policy backbone, are gaining significant attention for their promising generalization capabilities.
◆ This paper revisits a fundamental yet seldom systematically studied question: how VLM choice and competence translate to downstream VLA policies performance?
◆ We introduce VLM4VLA, a minimal adaptation pipeline that converts general-purpose VLMs into VLA policies using only a small set of new learnable parameters for fair and efficient comparison.</td></tr>
<tr><td>2026-01-06</td><td>InfiniDepth: Arbitrary-Resolution and Fine-Grained Depth Estimation with Neural Implicit Fields</td><td>[2601.03252](http://arxiv.org/pdf/2601.03252)</td><td>◆ Existing depth estimation methods are fundamentally limited to predicting depth on discrete image grids.
◆ Such representations restrict their scalability to arbitrary output resolutions and hinder the geometric detail recovery.
◆ This paper introduces InfiniDepth, which represents depth as neural implicit fields.</td></tr>
<tr><td>2026-01-06</td><td>Reinforcement Learning for Follow-the-Leader Robotic Endoscopic Navigation via Synthetic Data</td><td>[2601.02798](http://arxiv.org/pdf/2601.02798)</td><td>◆ Autonomous navigation is crucial for both medical and industrial endoscopic robots, enabling safe and efficient exploration of narrow tubular environments without continuous human intervention, where avoiding contact with the inner walls has been a longstanding challenge for prior approaches.
◆ We present a follow-the-leader endoscopic robot based on a flexible continuum structure designed to minimize contact between the endoscope body and intestinal walls, thereby reducing patient discomfort.
◆ To achieve this objective, we propose a vision-based deep reinforcement learning framework guided by monocular depth estimation.</td></tr>
<tr><td>2026-01-06</td><td>StableDPT: Temporal Stable Monocular Video Depth Estimation</td><td>[2601.02793](http://arxiv.org/pdf/2601.02793)</td><td>◆ Applying single image Monocular Depth Estimation (MDE) models to video sequences introduces significant temporal instability and flickering artifacts.
◆ We propose a novel approach that adapts any state-of-the-art image-based (depth) estimation model for video processing by integrating a new temporal module - trainable on a single GPU in a few days.
◆ Our architecture StableDPT builds upon an off-the-shelf Vision Transformer (ViT) encoder and enhances the Dense Prediction Transformer (DPT) head.</td></tr>
<tr><td>2026-01-06</td><td>AnyDepth: Depth Estimation Made Easy</td><td>[2601.02760](http://arxiv.org/pdf/2601.02760)</td><td>◆ Monocular depth estimation aims to recover the depth information of 3D scenes from 2D images.
◆ Recent work has made significant progress, but its reliance on large-scale datasets and complex decoders has limited its efficiency and generalization ability.
◆ In this paper, we propose a lightweight and data-centric framework for zero-shot monocular depth estimation.</td></tr>
<tr><td>2026-01-05</td><td>Adapting Depth Anything to Adverse Imaging Conditions with Events</td><td>[2601.02020](http://arxiv.org/pdf/2601.02020)</td><td>◆ Robust depth estimation under dynamic and adverse lighting conditions is essential for robotic systems.
◆ Currently, depth foundation models, such as Depth Anything, achieve great success in ideal scenes but remain challenging under adverse imaging conditions such as extreme illumination and motion blur.
◆ These degradations corrupt the visual signals of frame cameras, weakening the discriminative features of frame-based depths across the spatial and temporal dimensions.</td></tr>
<tr><td>2026-01-05</td><td>DisCo-FLoc: Using Dual-Level Visual-Geometric Contrasts to Disambiguate Depth-Aware Visual Floorplan Localization</td><td>[2601.01822](http://arxiv.org/pdf/2601.01822)</td><td>◆ Since floorplan data is readily available, long-term persistent, and robust to changes in visual appearance, visual Floorplan Localization (FLoc) has garnered significant attention.
◆ Existing methods either ingeniously match geometric priors or utilize sparse semantics to reduce FLoc uncertainty.
◆ However, they still suffer from ambiguous FLoc caused by repetitive structures within minimalist floorplans.</td></tr>
<tr><td>2026-01-04</td><td>Language as Prior, Vision as Calibration: Metric Scale Recovery for Monocular Depth Estimation</td><td>[2601.01457](http://arxiv.org/pdf/2601.01457)</td><td>◆ Relative-depth foundation models transfer well, yet monocular metric depth remains ill-posed due to unidentifiable global scale and heightened domain-shift sensitivity.
◆ Under a frozen-backbone calibration setting, we recover metric depth via an image-specific affine transform in inverse depth and train only lightweight calibration heads while keeping the relative-depth backbone and the CLIP text encoder fixed.
◆ Since captions provide coarse but noisy scale cues that vary with phrasing and missing objects, we use language to predict an uncertainty-aware envelope that bounds feasible calibration parameters in an unconstrained space, rather than committing to a text-only point estimate.</td></tr>
<tr><td>2026-01-02</td><td>AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction</td><td>[2601.00796](http://arxiv.org/pdf/2601.00796)</td><td>◆ Reconstructing dynamic 3D scenes from monocular videos requires simultaneously capturing high-frequency appearance details and temporally continuous motion.
◆ Existing methods using single Gaussian primitives are limited by their low-pass filtering nature, while standard Gabor functions introduce energy instability.
◆ Moreover, lack of temporal continuity constraints often leads to motion artifacts during interpolation.</td></tr>
<tr><td>2025-12-31</td><td>Projection-based Adversarial Attack using Physics-in-the-Loop Optimization for Monocular Depth Estimation</td><td>[2512.24792](http://arxiv.org/pdf/2512.24792)</td><td>◆ Deep neural networks (DNNs) remain vulnerable to adversarial attacks that cause misclassification when specific perturbations are added to input images.
◆ This vulnerability also threatens the reliability of DNN-based monocular depth estimation (MDE) models, making robustness enhancement a critical need in practical applications.
◆ To validate the vulnerability of DNN-based MDE models, this study proposes a projection-based adversarial attack method that projects perturbation light onto a target object.</td></tr>
<tr><td>2025-12-30</td><td>Guided Diffusion-based Generation of Adversarial Objects for Real-World Monocular Depth Estimation Attacks</td><td>[2512.24111](http://arxiv.org/pdf/2512.24111)</td><td>◆ Monocular Depth Estimation (MDE) serves as a core perception module in autonomous driving systems, but it remains highly susceptible to adversarial attacks.
◆ Errors in depth estimation may propagate through downstream decision making and influence overall traffic safety.
◆ Existing physical attacks primarily rely on texture-based patches, which impose strict placement constraints and exhibit limited realism, thereby reducing their effectiveness in complex driving environments.</td></tr>
<tr><td>2025-12-29</td><td>Leveraging Synthetic Priors for Monocular Depth Estimation in Specular Surgical Environments</td><td>[2512.23786](http://arxiv.org/pdf/2512.23786)</td><td>◆ Accurate Monocular Depth Estimation (MDE) is critical for robotic surgery but remains fragile in specular, fluid-filled endoscopic environments.
◆ Existing self-supervised methods, typically relying on foundation models trained with noisy real-world pseudo-labels, often suffer from boundary collapse on thin surgical tools and transparent surfaces.
◆ In this work, we address this by leveraging the high-fidelity synthetic priors of the Depth Anything V2 architecture, which inherently captures precise geometric details of thin structures.</td></tr>
<tr><td>2025-12-28</td><td>With Great Context Comes Great Prediction Power: Classifying Objects via Geo-Semantic Scene Graphs</td><td>[2512.23024](http://arxiv.org/pdf/2512.23024)</td><td>◆ Humans effortlessly identify objects by leveraging a rich understanding of the surrounding scene, including spatial relationships, material properties, and the co-occurrence of other objects.
◆ In contrast, most computational object recognition systems operate on isolated image regions, devoid of meaning in isolation, thus ignoring this vital contextual information.
◆ This paper argues for the critical role of context and introduces a novel framework for contextual object classification.</td></tr>
<tr><td>2025-12-28</td><td>Depth Anything in $360^\circ$: Towards Scale Invariance in the Wild</td><td>[2512.22819](http://arxiv.org/pdf/2512.22819)</td><td>◆ Panoramic depth estimation provides a comprehensive solution for capturing complete $360^\circ$ environmental structural information, offering significant benefits for robotics and AR/VR applications.
◆ However, while extensively studied in indoor settings, its zero-shot generalization to open-world domains lags far behind perspective images, which benefit from abundant training data.
◆ This disparity makes transferring capabilities from the perspective domain an attractive solution.</td></tr>
<tr><td>2025-12-27</td><td>Visual Autoregressive Modelling for Monocular Depth Estimation</td><td>[2512.22653](http://arxiv.org/pdf/2512.22653)</td><td>◆ We propose a monocular depth estimation method based on visual autoregressive (VAR) priors, offering an alternative to diffusion-based approaches.
◆ Our method adapts a large-scale text-to-image VAR model and introduces a scale-wise conditional upsampling mechanism with classifier-free guidance.
◆ Our approach performs inference in ten fixed autoregressive stages, requiring only 74K synthetic samples for fine-tuning, and achieves competitive results.</td></tr>
<tr><td>2025-12-26</td><td>iOSPointMapper: RealTime Pedestrian and Accessibility Mapping with Mobile AI</td><td>[2512.22392](http://arxiv.org/pdf/2512.22392)</td><td>◆ Accurate, up-to-date sidewalk data is essential for building accessible and inclusive pedestrian infrastructure, yet current approaches to data collection are often costly, fragmented, and difficult to scale.
◆ We introduce iOSPointMapper, a mobile application that enables real-time, privacy-conscious sidewalk mapping on the ground, using recent-generation iPhones and iPads.
◆ The system leverages on-device semantic segmentation, LiDAR-based depth estimation, and fused GPS/IMU data to detect and localize sidewalk-relevant features such as traffic signs, traffic lights and poles.</td></tr>
<tr><td>2025-12-26</td><td>Bab_Sak Robotic Intubation System (BRIS): A Learning-Enabled Control Framework for Safe Fiberoptic Endotracheal Intubation</td><td>[2512.21983](http://arxiv.org/pdf/2512.21983)</td><td>◆ Endotracheal intubation is a critical yet technically demanding procedure, with failure or improper tube placement leading to severe complications.
◆ Existing robotic and teleoperated intubation systems primarily focus on airway navigation and do not provide integrated control of endotracheal tube advancement or objective verification of tube depth relative to the carina.
◆ This paper presents the Robotic Intubation System (BRIS), a compact, human-in-the-loop platform designed to assist fiberoptic-guided intubation while enabling real-time, objective depth awareness.</td></tr>
<tr><td>2025-12-26</td><td>StereoVLA: Enhancing Vision-Language-Action Models with Stereo Vision</td><td>[2512.21970](http://arxiv.org/pdf/2512.21970)</td><td>◆ Stereo cameras closely mimic human binocular vision, providing rich spatial cues critical for precise robotic manipulation.
◆ Despite their advantage, the adoption of stereo vision in vision-language-action models (VLAs) remains underexplored.
◆ In this work, we present StereoVLA, a VLA model that leverages rich geometric cues from stereo vision.</td></tr>
<tr><td>2025-12-24</td><td>CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models</td><td>[2512.19083](http://arxiv.org/pdf/2512.19083)</td><td>◆ Autonomous navigation for Unmanned Aerial Vehicles faces key challenges from limited onboard computational resources, which restrict deployed deep neural networks to shallow architectures incapable of handling complex environments.
◆ Offloading tasks to remote edge servers introduces high latency, creating an inherent trade-off in system design.
◆ To address these limitations, we propose CoDrone - the first cloud-edge-end collaborative computing framework integrating foundation models into autonomous UAV cruising scenarios - effectively leveraging foundation models to enhance performance of resource-constrained unmanned aerial vehicle platforms.</td></tr>
<tr><td>2025-12-22</td><td>CETCAM: Camera-Controllable Video Generation via Consistent and Extensible Tokenization</td><td>[2512.19020](http://arxiv.org/pdf/2512.19020)</td><td>◆ Achieving precise camera control in video generation remains challenging, as existing methods often rely on camera pose annotations that are difficult to scale to large and dynamic datasets and are frequently inconsistent with depth estimation, leading to train-test discrepancies.
◆ We introduce CETCAM, a camera-controllable video generation framework that eliminates the need for camera annotations through a consistent and extensible tokenization scheme.
◆ CETCAM leverages recent advances in geometry foundation models, such as VGGT, to estimate depth and camera parameters and converts them into unified, geometry-aware tokens.</td></tr>
<tr><td>2025-12-21</td><td>A Study of Finetuning Video Transformers for Multi-view Geometry Tasks</td><td>[2512.18684](http://arxiv.org/pdf/2512.18684)</td><td>◆ This paper presents an investigation of vision transformer learning for multi-view geometry tasks, such as optical flow estimation, by fine-tuning video foundation models.
◆ Unlike previous methods that involve custom architectural designs and task-specific pretraining, our research finds that general-purpose models pretrained on videos can be readily transferred to multi-view problems with minimal adaptation.
◆ The core insight is that general-purpose attention between patches learns temporal and spatial information for geometric reasoning.</td></tr>
<tr><td>2025-12-20</td><td>EndoStreamDepth: Temporally Consistent Monocular Depth Estimation for Endoscopic Video Streams</td><td>[2512.18159](http://arxiv.org/pdf/2512.18159)</td><td>◆ This work presents EndoStreamDepth, a monocular depth estimation framework for endoscopic video streams.
◆ It provides accurate depth maps with sharp anatomical boundaries for each frame, temporally consistent predictions across frames, and real-time throughput.
◆ Unlike prior work that uses batched inputs, EndoStreamDepth processes individual frames with a temporal module to propagate inter-frame information.</td></tr>
<tr><td>2025-12-19</td><td>Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting</td><td>[2512.17908](http://arxiv.org/pdf/2512.17908)</td><td>◆ Monocular depth estimation remains challenging as recent foundation models, such as Depth Anything V2 (DA-V2), struggle with real-world images that are far from the training distribution.
◆ We introduce Re-Depth Anything, a test-time self-supervision framework that bridges this domain gap by fusing DA-V2 with the powerful priors of large-scale 2D diffusion models.
◆ Our method performs label-free refinement directly on the input image by re-lighting predicted depth maps and augmenting the input.</td></tr>
<tr><td>2025-12-19</td><td>Long-Range depth estimation using learning based Hybrid Distortion Model for CCTV cameras</td><td>[2512.17784](http://arxiv.org/pdf/2512.17784)</td><td>◆ Accurate camera models are essential for photogrammetry applications such as 3D mapping and object localization, particularly for long distances.
◆ Various stereo-camera based 3D localization methods are available but are limited to few hundreds of meters&#x27; range.
◆ This is majorly due to the limitation of the distortion models assumed for the non-linearities present in the camera lens.</td></tr>
<tr><td>2025-12-19</td><td>SAVeD: A First-Person Social Media Video Dataset for ADAS-equipped vehicle Near-Miss and Crash Event Analyses</td><td>[2512.17724](http://arxiv.org/pdf/2512.17724)</td><td>◆ The advancement of safety-critical research in driving behavior in ADAS-equipped vehicles require real-world datasets that not only include diverse traffic scenarios but also capture high-risk edge cases such as near-miss events and system failures.
◆ However, existing datasets are largely limited to either simulated environments or human-driven vehicle data, lacking authentic ADAS (Advanced Driver Assistance System) vehicle behavior under risk conditions.
◆ To address this gap, this paper introduces SAVeD, a large-scale video dataset curated from publicly available social media content, explicitly focused on ADAS vehicle-related crashes, near-miss incidents, and disengagements.</td></tr>
<tr><td>2025-12-18</td><td>Infinite-Homography as Robust Conditioning for Camera-Controlled Video Generation</td><td>[2512.17040](http://arxiv.org/pdf/2512.17040)</td><td>◆ Recent progress in video diffusion models has spurred growing interest in camera-controlled novel-view video generation for dynamic scenes, aiming to provide creators with cinematic camera control capabilities in post-production.
◆ A key challenge in camera-controlled video generation is ensuring fidelity to the specified camera pose, while maintaining view consistency and reasoning about occluded geometry from limited observations.
◆ To address this, existing methods either train trajectory-conditioned video generation model on trajectory-video pair dataset, or estimate depth from the input video to reproject it along a target trajectory and generate the unprojected regions.</td></tr>
<tr><td>2025-12-18</td><td>Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation</td><td>[2512.16913](http://arxiv.org/pdf/2512.16913)</td><td>◆ In this work, we present a panoramic metric depth foundation model that generalizes across diverse scene distances.
◆ We explore a data-in-the-loop paradigm from the view of both data construction and framework design.
◆ We collect a large-scale dataset by combining public datasets, high-quality synthetic data from our UE5 simulator and text-to-image models, and real panoramic images from the web.</td></tr>
<tr><td>2025-12-18</td><td>N3D-VLM: Native 3D Grounding Enables Accurate Spatial Reasoning in Vision-Language Models</td><td>[2512.16561](http://arxiv.org/pdf/2512.16561)</td><td>◆ While current multimodal models can answer questions based on 2D images, they lack intrinsic 3D object perception, limiting their ability to comprehend spatial relationships and depth cues in 3D scenes.
◆ In this work, we propose N3D-VLM, a novel unified framework that seamlessly integrates native 3D object perception with 3D-aware visual reasoning, enabling both precise 3D grounding and interpretable spatial understanding.
◆ Unlike conventional end-to-end models that directly predict answers from RGB/RGB-D inputs, our approach equips the model with native 3D object perception capabilities, enabling it to directly localize objects in 3D space based on textual descriptions.</td></tr>
<tr><td>2025-12-17</td><td>In Pursuit of Pixel Supervision for Visual Pre-training</td><td>[2512.15715](http://arxiv.org/pdf/2512.15715)</td><td>◆ At the most basic level, pixels are the source of the visual information through which we perceive the world.
◆ Pixels contain information at all levels, ranging from low-level attributes to high-level concepts.
◆ Autoencoders represent a classical and long-standing paradigm for learning representations from pixels or other raw inputs.</td></tr>
<tr><td>2025-12-16</td><td>DASP: Self-supervised Nighttime Monocular Depth Estimation with Domain Adaptation of Spatiotemporal Priors</td><td>[2512.14536](http://arxiv.org/pdf/2512.14536)</td><td>◆ Self-supervised monocular depth estimation has achieved notable success under daytime conditions.
◆ However, its performance deteriorates markedly at night due to low visibility and varying illumination, e.g., insufficient light causes textureless areas, and moving objects bring blurry regions.
◆ To this end, we propose a self-supervised framework named DASP that leverages spatiotemporal priors for nighttime depth estimation.</td></tr>
<tr><td>2025-12-16</td><td>Elastic3D: Controllable Stereo Video Conversion with Guided Latent Decoding</td><td>[2512.14236](http://arxiv.org/pdf/2512.14236)</td><td>◆ The growing demand for immersive 3D content calls for automated monocular-to-stereo video conversion.
◆ We present Elastic3D, a controllable, direct end-to-end method for upgrading a conventional video to a binocular one.
◆ Our approach, based on (conditional) latent diffusion, avoids artifacts due to explicit depth estimation and warping.</td></tr>
<tr><td>2025-12-16</td><td>Robust Single-shot Structured Light 3D Imaging via Neural Feature Decoding</td><td>[2512.14028](http://arxiv.org/pdf/2512.14028)</td><td>◆ We consider the problem of active 3D imaging using single-shot structured light systems, which are widely employed in commercial 3D sensing devices such as Apple Face ID and Intel RealSense.
◆ Traditional structured light methods typically decode depth correspondences through pixel-domain matching algorithms, resulting in limited robustness under challenging scenarios like occlusions, fine-structured details, and non-Lambertian surfaces.
◆ Inspired by recent advances in neural feature matching, we propose a learning-based structured light decoding framework that performs robust correspondence matching within feature space rather than the fragile pixel domain.</td></tr>
<tr><td>2025-12-16</td><td>Deep Learning Perspective of Scene Understanding in Autonomous Robots</td><td>[2512.14020](http://arxiv.org/pdf/2512.14020)</td><td>◆ This paper provides a review of deep learning applications in scene understanding in autonomous robots, including innovations in object detection, semantic and instance segmentation, depth estimation, 3D reconstruction, and visual SLAM.
◆ It emphasizes how these techniques address limitations of traditional geometric models, improve depth perception in real time despite occlusions and textureless surfaces, and enhance semantic reasoning to understand the environment better.
◆ When these perception modules are integrated into dynamic and unstructured environments, they become more effective in decisionmaking, navigation and interaction.</td></tr>
<tr><td>2025-12-15</td><td>StarryGazer: Leveraging Monocular Depth Estimation Models for Domain-Agnostic Single Depth Image Completion</td><td>[2512.13147](http://arxiv.org/pdf/2512.13147)</td><td>◆ The problem of depth completion involves predicting a dense depth image from a single sparse depth map and an RGB image.
◆ Unsupervised depth completion methods have been proposed for various datasets where ground truth depth data is unavailable and supervised methods cannot be applied.
◆ However, these models require auxiliary data to estimate depth values, which is far from real scenarios.</td></tr>
<tr><td>2025-12-13</td><td>BokehDepth: Enhancing Monocular Depth Estimation through Bokeh Generation</td><td>[2512.12425](http://arxiv.org/pdf/2512.12425)</td><td>◆ Bokeh and monocular depth estimation are tightly coupled through the same lens imaging geometry, yet current methods exploit this connection in incomplete ways.
◆ High-quality bokeh rendering pipelines typically depend on noisy depth maps, which amplify estimation errors into visible artifacts, while modern monocular metric depth models still struggle on weakly textured, distant and geometrically ambiguous regions where defocus cues are most informative.
◆ We introduce BokehDepth, a two-stage framework that decouples bokeh synthesis from depth prediction and treats defocus as an auxiliary supervision-free geometric cue.</td></tr>
<tr><td>2025-12-17</td><td>ProbeMDE: Uncertainty-Guided Active Proprioception for Monocular Depth Estimation in Surgical Robotics</td><td>[2512.11773](http://arxiv.org/pdf/2512.11773)</td><td>◆ Monocular depth estimation (MDE) provides a useful tool for robotic perception, but its predictions are often uncertain and inaccurate in challenging environments such as surgical scenes where textureless surfaces, specular reflections, and occlusions are common.
◆ To address this, we propose ProbeMDE, a cost-aware active sensing framework that combines RGB images with sparse proprioceptive measurements for MDE.
◆ Our approach utilizes an ensemble of MDE models to predict dense depth maps conditioned on both RGB images and on a sparse set of known depth measurements obtained via proprioception, where the robot has touched the environment in a known configuration.</td></tr>
<tr><td>2025-12-11</td><td>Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching</td><td>[2512.11130](http://arxiv.org/pdf/2512.11130)</td><td>◆ Stereo foundation models achieve strong zero-shot generalization but remain computationally prohibitive for real-time applications.
◆ Efficient stereo architectures, on the other hand, sacrifice robustness for speed and require costly per-domain fine-tuning.
◆ To bridge this gap, we present Fast-FoundationStereo, a family of architectures that achieve, for the first time, strong zero-shot generalization at real-time frame rate.</td></tr>
<tr><td>2025-12-11</td><td>Empowering Dynamic Urban Navigation with Stereo and Mid-Level Vision</td><td>[2512.10956](http://arxiv.org/pdf/2512.10956)</td><td>◆ The success of foundation models in language and vision motivated research in fully end-to-end robot navigation foundation models (NFMs).
◆ NFMs directly map monocular visual input to control actions and ignore mid-level vision modules (tracking, depth estimation, etc) entirely.
◆ While the assumption that vision capabilities will emerge implicitly is compelling, it requires large amounts of pixel-to-action supervision that are difficult to obtain.</td></tr>
<tr><td>2025-12-11</td><td>Video Depth Propagation</td><td>[2512.10725](http://arxiv.org/pdf/2512.10725)</td><td>◆ Depth estimation in videos is essential for visual perception in real-world applications.
◆ However, existing methods either rely on simple frame-by-frame monocular models, leading to temporal inconsistencies and inaccuracies, or use computationally demanding temporal modeling, unsuitable for real-time applications.
◆ These limitations significantly restrict general applicability and performance in practical settings.</td></tr>
<tr><td>2025-12-11</td><td>SpaceDrive: Infusing Spatial Awareness into VLM-based Autonomous Driving</td><td>[2512.10719](http://arxiv.org/pdf/2512.10719)</td><td>◆ End-to-end autonomous driving methods built on vision language models (VLMs) have undergone rapid development driven by their universal visual understanding and strong reasoning capabilities obtained from the large-scale pretraining.
◆ However, we find that current VLMs struggle to understand fine-grained 3D spatial relationships which is a fundamental requirement for systems interacting with the physical world.
◆ To address this issue, we propose SpaceDrive, a spatial-aware VLM-based driving framework that treats spatial information as explicit positional encodings (PEs) instead of textual digit tokens, enabling joint reasoning over semantic and spatial representations.</td></tr>
<tr><td>2025-12-11</td><td>Robust Shape from Focus via Multiscale Directional Dilated Laplacian and Recurrent Network</td><td>[2512.10498](http://arxiv.org/pdf/2512.10498)</td><td>◆ Shape-from-Focus (SFF) is a passive depth estimation technique that infers scene depth by analyzing focus variations in a focal stack.
◆ Most recent deep learning-based SFF methods typically operate in two stages: first, they extract focus volumes (a per pixel representation of focus likelihood across the focal stack) using heavy feature encoders; then, they estimate depth via a simple one-step aggregation technique that often introduces artifacts and amplifies noise in the depth map.
◆ To address these issues, we propose a hybrid framework.</td></tr>
<tr><td>2025-12-09</td><td>Scale-invariant and View-relational Representation Learning for Full Surround Monocular Depth</td><td>[2512.08700](http://arxiv.org/pdf/2512.08700)</td><td>◆ Recent foundation models demonstrate strong generalization capabilities in monocular depth estimation.
◆ However, directly applying these models to Full Surround Monocular Depth Estimation (FSMDE) presents two major challenges: (1) high computational cost, which limits real-time performance, and (2) difficulty in estimating metric-scale depth, as these models are typically trained to predict only relative depth.
◆ To address these limitations, we propose a novel knowledge distillation strategy that transfers robust depth knowledge from a foundation model to a lightweight FSMDE network.</td></tr>
<tr><td>2025-12-09</td><td>Accuracy Does Not Guarantee Human-Likeness in Monocular Depth Estimators</td><td>[2512.08163](http://arxiv.org/pdf/2512.08163)</td><td>◆ Monocular depth estimation is a fundamental capability for real-world applications such as autonomous driving and robotics.
◆ Although deep neural networks (DNNs) have achieved superhuman accuracy on physical-based benchmarks, a key challenge remains: aligning model representations with human perception, a promising strategy for enhancing model robustness and interpretability.
◆ Research in object recognition has revealed a complex trade-off between model accuracy and human-like behavior, raising a question whether a similar divergence exist in depth estimation, particularly for natural outdoor scenes where benchmarks rely on sensor-based ground truth rather than human perceptual estimates.</td></tr>
<tr><td>2025-12-10</td><td>More than Segmentation: Benchmarking SAM 3 for Segmentation, 3D Perception, and Reconstruction in Robotic Surgery</td><td>[2512.07596](http://arxiv.org/pdf/2512.07596)</td><td>◆ The recent Segment Anything Model (SAM) 3 has introduced significant advancements over its predecessor, SAM 2, particularly with the integration of language-based segmentation and enhanced 3D perception capabilities.
◆ SAM 3 supports zero-shot segmentation across a wide range of prompts, including point, bounding box, and language-based prompts, allowing for more flexible and intuitive interactions with the model.
◆ In this empirical evaluation, we assess the performance of SAM 3 in robot-assisted surgery, benchmarking its zero-shot segmentation with point and bounding box prompts and exploring its effectiveness in dynamic video tracking, alongside its newly introduced language prompt segmentation.</td></tr>
<tr><td>2025-12-07</td><td>Generalized Geometry Encoding Volume for Real-time Stereo Matching</td><td>[2512.06793](http://arxiv.org/pdf/2512.06793)</td><td>◆ Real-time stereo matching methods primarily focus on enhancing in-domain performance but often overlook the critical importance of generalization in real-world applications.
◆ In contrast, recent stereo foundation models leverage monocular foundation models (MFMs) to improve generalization, but typically suffer from substantial inference latency.
◆ To address this trade-off, we propose Generalized Geometry Encoding Volume (GGEV), a novel real-time stereo matching network that achieves strong generalization.</td></tr>
<tr><td>2025-12-07</td><td>CoT4Det: A Chain-of-Thought Framework for Perception-Oriented Vision-Language Tasks</td><td>[2512.06663](http://arxiv.org/pdf/2512.06663)</td><td>◆ Large Vision-Language Models (LVLMs) have demonstrated remarkable success in a broad range of vision-language tasks, such as general visual question answering and optical character recognition (OCR).
◆ However, their performance on perception-centric tasks -- such as object detection, semantic segmentation, and depth estimation -- remains significantly inferior to that of task-specific expert models.
◆ For example, Qwen2.5-VL-7B-Instruct achieves only 19% mAP on COCO2017 val, particularly struggling with dense scenes and small object recall.</td></tr>
<tr><td>2025-12-09</td><td>HuPrior3R: Incorporating Human Priors for Better 3D Dynamic Reconstruction from Monocular Videos</td><td>[2512.06368](http://arxiv.org/pdf/2512.06368)</td><td>◆ Monocular dynamic video reconstruction faces significant challenges in dynamic human scenes due to geometric inconsistencies and resolution degradation issues.
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
> 更新于: 2026.03.18
