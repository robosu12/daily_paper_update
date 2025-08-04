# SLAM领域最新论文 (2025.08.04)

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

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='visual-slam'>Visual SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='loop-closure'>Loop Closure</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='image-matching'>Image Matching</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='3dgs'>3DGS</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='depth-estimation'>Depth Estimation</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2025.08.04
