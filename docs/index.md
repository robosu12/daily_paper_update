# 计算机视觉领域最新论文 (2025.07.04)

> 每日自动更新计算机视觉领域的最新arXiv论文

> 使用说明: [点击查看](./docs/README.md#usage)

<details>
<summary>分类目录</summary>
<ol>
<li><a href='#slam'>SLAM</a></li>
<li><a href='#sfm'>SFM</a></li>
<li><a href='#visual-localization'>Visual Localization</a></li>
<li><a href='#keypoint-detection'>Keypoint Detection</a></li>
<li><a href='#image-matching'>Image Matching</a></li>
<li><a href='#nerf'>NeRF</a></li>
</ol>
</details>

<h2 id='slam'>SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-07-01</td><td>RaGNNarok: A Light-Weight Graph Neural Network for Enhancing Radar Point Clouds on Unmanned Ground Vehicles</td><td>[2507.00937](http://arxiv.org/pdf/2507.00937)</td><td>◆提出RaGNNarok，一种基于图神经网络（GNN）的轻量级框架，用于增强雷达点云数据，解决现有雷达定位中稀疏点云、噪声和误检测问题。  
◆该框架在低成本设备（如树莓派5）上实现实时处理，推理时间仅7.3毫秒，无需额外计算资源，适合资源受限的移动机器人。  
◆通过GNN模型优化雷达点云，显著提升在复杂动态环境中的性能，克服了传统激光雷达和相机在视觉遮挡环境中的局限性。  
◆在定位、SLAM和自主导航等关键任务中进行了多环境测试，验证了其高可靠性和泛化能力。  
◆为低成本室内移动机器人提供了一种经济高效的解决方案，结合毫米波雷达的低成本优势，推动自动化在家庭和商业空间的应用。</td></tr>
<tr><td>2025-07-01</td><td>Generation of Indoor Open Street Maps for Robot Navigation from CAD Files</td><td>[2507.00552](http://arxiv.org/pdf/2507.00552)</td><td>◆ 提出全自动系统，将建筑CAD文件转换为分层拓扑OpenStreetMap（OSM）表示，专为机器人终身导航设计，解决SLAM在动态大尺度室内环境中耗时、脆弱且易过时的问题。  
◆ 开发多阶段处理流程，从原始CAD数据中提取关键结构层，并基于AreaGraph进行拓扑分割，生成层次化可导航空间图，实现语义丰富的环境建模。  
◆ 自动关联CAD源文件中的文本标签，增强地图语义信息，同时支持多楼层无缝合并，构建拓扑正确的统一模型，提升导航鲁棒性。  
◆ 利用CAD文件固有的永久结构信息，规避SLAM的固有缺陷，为复杂室内场景提供高效、可扩展的解决方案。  
◆ 集成直观图形用户界面（GUI）封装软件，降低使用门槛，并开源代码和数据集促进社区应用与研究。</td></tr>
<tr><td>2025-06-30</td><td>VOCAL: Visual Odometry via ContrAstive Learning</td><td>[2507.00243](http://arxiv.org/pdf/2507.00243)</td><td>◆ VOCAL将视觉里程计（VO）重新定义为标签排序问题，突破了传统基于几何假设的局限，为数据驱动框架提供了新思路。  
◆ 通过结合贝叶斯推理与表征学习，该框架使视觉特征与相机状态对齐，提升了特征的可解释性。  
◆ 提出的排序机制迫使相似相机状态在潜在空间中形成一致且空间连贯的表征，增强了模型的鲁棒性。  
◆ 框架支持多模态数据融合，为复杂场景下的VO应用提供了灵活性。  
◆ 在KITTI数据集上的实验验证了VOCAL在可解释性和泛化性上的显著优势，推动了空间智能向更通用、可解释的方向发展。</td></tr>
<tr><td>2025-06-29</td><td>TVG-SLAM: Robust Gaussian Splatting SLAM with Tri-view Geometric Constraints</td><td>[2506.23207](http://arxiv.org/pdf/2506.23207)</td><td>TVG-SLAM是一种基于3D高斯泼溅（3DGS）的RGB-only SLAM系统，通过三视图几何约束提升鲁棒性和场景重建质量。其核心贡献和创新点如下：

◆ 提出三视图几何范式，通过密集三视图匹配模块聚合可靠的帧间对应关系，形成跨帧的鲁棒几何约束，解决传统方法依赖光度损失的局限性。

◆ 设计混合几何约束（Hybrid Geometric Constraints），结合三视图匹配的几何线索与光度损失，显著提升相机位姿估计的准确性和稳定性，尤其在视角突变和光照变化场景。

◆ 提出基于概率的初始化策略，将三视图对应关系的几何不确定性编码到新初始化的高斯模型中，提升映射质量。

◆ 引入动态渲染信任衰减机制（Dynamic Attenuation of Rendering Trust），有效缓解因建图延迟导致的跟踪漂移问题。

实验表明，TVG-SLAM在户外数据集上优于现有RGB-only 3DGS SLAM系统，在最挑战性数据集中将轨迹误差（ATE）降低69.0%，同时保持顶尖的渲染质量。</td></tr>
<tr><td>2025-06-29</td><td>Event-based Stereo Visual-Inertial Odometry with Voxel Map</td><td>[2506.23078](http://arxiv.org/pdf/2506.23078)</td><td>◆ 提出Voxel-ESVIO系统，结合事件相机和立体视觉惯性里程计，利用体素地图管理提升定位精度。  
◆ 采用基于体素的点选择方法，有效过滤事件流中的噪声，筛选高质量3D地图点。  
◆ 创新性地引入体素感知的点管理机制，动态优化每个体素内地图点的更新和选择。  
◆ 通过协同策略高效提取抗噪声且观测概率最高的地图点，确保状态估计的准确性。  
◆ 在三个公开数据集上的实验表明，该系统在精度和计算效率上均优于现有方法。</td></tr>
<tr><td>2025-06-26</td><td>Adaptive Multipath-Based SLAM for Distributed MIMO Systems</td><td>[2506.21798](http://arxiv.org/pdf/2506.21798)</td><td>◆ 提出了一种适用于分布式MIMO系统的自适应多路径SLAM方法，解决了传统方法在非凸几何环境中无法进行光线追踪的局限性。  
◆ 利用振幅统计量建立自适应时变检测概率，实现了&quot;软&quot;光线追踪策略，能够在非凸几何的射频环境中跨传播路径融合信息。  
◆ 通过将和积算法(SPA)的消息传递规则应用于所提出的统计模型因子图，建立了地图特征和智能体位置联合估计的贝叶斯估计方法。  
◆ 提出了一种改进的建议概率密度函数(PDF)，用于基于粒子的SPA消息计算，能够早期检测仅由双跳路径支持的新表面。  
◆ 在具有非凸几何形状的挑战性场景中使用合成射频测量验证了该方法，结果表明其能够提供准确的定位和建图估计，并达到后验CRLB界。</td></tr>
<tr><td>2025-06-24</td><td>Ark: An Open-source Python-based Framework for Robot Learning</td><td>[2506.21628](http://arxiv.org/pdf/2506.21628)</td><td>◆ 提出ARK框架，首个以Python为核心的机器人学习开源平台，弥合机器人技术与现代AI工具链的鸿沟。  
◆ 采用Gym风格接口设计，支持数据采集、预处理到策略训练的全流程，兼容仿真与实体机器人无缝切换。  
◆ 独创轻量级客户端-服务器架构，实现网络化发布-订阅通信，并保留C/C++绑定选项保障实时性能需求。  
◆ 内置控制、SLAM、运动规划等模块化组件，原生支持ROS交互，提供开箱即用的机器人功能套件。  
◆ 通过详实文档和案例（如操作与导航任务），验证其快速原型开发、硬件灵活切换及端到端流水线优势。  
◆ 统一Python生态与机器人开发，显著降低学习门槛，加速学术研究与商业场景的机器人自主性落地。</td></tr>
<tr><td>2025-06-26</td><td>EndoFlow-SLAM: Real-Time Endoscopic SLAM with Flow-Constrained Gaussian Splatting</td><td>[2506.21420](http://arxiv.org/pdf/2506.21420)</td><td>◆ 提出EndoFlow-SLAM系统，首次将光流损失作为几何约束引入基于3D高斯泼溅（3DGS）的SLAM框架，有效解决了内窥镜场景中非朗伯表面和呼吸运动导致的位姿估计问题。  
◆ 设计深度正则化策略，缓解内窥镜场景的光度不一致性问题，确保3DGS深度渲染的可靠性。  
◆ 改进3DGS优化策略，针对关键帧中渲染质量较差的视角进行重点优化，提升场景表示精度。  
◆ 在静态（C3VD数据集）和动态（StereoMIS数据集）手术场景中均实现领先性能，在新视角合成和相机位姿估计任务上超越现有方法。  
◆ 系统支持实时运行，为内窥镜手术提供高效的三维重建与可视化能力。</td></tr>
<tr><td>2025-06-26</td><td>CURL-SLAM: Continuous and Compact LiDAR Mapping</td><td>[2506.21077](http://arxiv.org/pdf/2506.21077)</td><td>◆ 提出了一种新型LiDAR SLAM范式CURL-SLAM，利用连续超紧凑表示（CURL）实现可更新、可定位的地图表示。  
◆ 采用球谐函数隐式编码技术，生成支持可变密度连续重建的紧凑3D地图，显著降低存储需求。  
◆ 通过独特的CURL定制优化问题重新定义LiDAR位姿估计，替代传统ICP方法，提升计算效率。  
◆ 扩展局部光束法平差（BA）技术，实现位姿精修与地图校正同步进行，确保闭环后的全局一致性。  
◆ 在CPU上达到10Hz实时性能，同时保持领先的3D建图质量和轨迹精度，适用于大规模场景。  
◆ 开源CURL-SLAM实现，推动连续紧凑地图表示领域的研究与应用。</td></tr>
<tr><td>2025-06-25</td><td>SPARK: Graph-Based Online Semantic Integration System for Robot Task Planning</td><td>[2506.20394](http://arxiv.org/pdf/2506.20394)</td><td>◆ 提出首个在线语义信息更新框架SPARK，解决机器人任务执行中语义信息实时更新的空白问题。  
◆ 创新性地将离线场景图表示扩展到在线场景，提升动态环境下的语义信息处理能力。  
◆ 通过环境嵌入线索（如手势等非传统空间提示）实时更新场景图，增强机器人对动态环境的适应性。  
◆ 验证了基于图的空间关系表示能显著提升任务规划效率，尤其在非结构化场景中表现突出。  
◆ 系统整合几何与语义数据，为通用服务机器人提供更全面的在线信息更新解决方案。  
◆ 实验证明该框架能有效处理动态环境中的语义变化，为后续任务规划提供可靠支持。</td></tr>
<tr><td>2025-06-25</td><td>Real-Time Obstacle Avoidance Algorithms for Unmanned Aerial and Ground Vehicles</td><td>[2506.20311](http://arxiv.org/pdf/2506.20311)</td><td>◆ 开发了适用于复杂3D环境的实时避障算法，特别针对森林火灾等灾害场景中的无人机安全导航需求。  
◆ 提出了一种创新的2D融合导航策略，最初为地面移动机器人设计，具备动态环境中的安全移动能力，并支持自适应障碍处理与决策优化。  
◆ 首次设计了针对森林火灾模拟的3D反应式导航策略，解决了无人机在此类特殊场景中的避障难题。  
◆ 提出无人机与地面无人车（UGV）的协同控制框架，实现了空地车辆在森林救援任务中的统一协调作业。  
◆ 通过数学建模与仿真验证了各阶段算法的有效性，为自然灾害救援中无人系统的应用提供了兼具实用价值与学术意义的解决方案。</td></tr>
<tr><td>2025-06-24</td><td>Posterior Cramér-Rao Bounds on Localization and Mapping Errors in Distributed MIMO SLAM</td><td>[2506.19957](http://arxiv.org/pdf/2506.19957)</td><td>◆ 首次提出了针对分布式MIMO SLAM系统中镜面反射面定位与建图误差的后验克拉美罗下界（MEB），填补了该领域性能评估的理论空白。  
◆ 考虑了单次反射和双次反射的复杂传播场景，并支持分布式锚点配置，扩展了传统SLAM性能边界的适用范围。  
◆ 通过数值仿真验证了现有先进RF-SLAM算法的建图误差能渐进收敛至MEB，为算法性能评估提供了理论基准。  
◆ 创新性地将映射性能（镜面位置/朝向）与用户定位性能统一纳入全局特征评估框架，突破了传统仅关注定位精度的局限。  
◆ 所提边界理论可提升多径信道中非视距信号的利用效率，为通信-定位一体化系统设计提供理论支撑。</td></tr>
<tr><td>2025-06-23</td><td>GRAND-SLAM: Local Optimization for Globally Consistent Large-Scale Multi-Agent Gaussian SLAM</td><td>[2506.18885](http://arxiv.org/pdf/2506.18885)</td><td>◆ 提出了GRAND-SLAM方法，首次将3D高斯泼溅技术应用于大规模户外多智能体SLAM场景，突破了现有方法仅限于小规模室内环境的限制。  
◆ 设计了基于局部子地图优化的隐式跟踪模块，有效提升了多智能体系统的定位精度和鲁棒性。  
◆ 开发了机器人内/间闭环检测方法，并将其集成到位姿图优化框架中，实现了全局一致性的大规模场景重建。  
◆ 在Replica室内数据集上实现了当前最优的跟踪性能，PSNR指标比现有方法提升28%。  
◆ 在大型户外Kimera-Multi数据集上，多智能体跟踪误差降低91%，渲染质量显著优于现有方法。  
◆ 通过可扩展的环境表示方法，为多智能体协同快速探索与重建提供了新解决方案。</td></tr>
<tr><td>2025-06-23</td><td>MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation</td><td>[2506.18678](http://arxiv.org/pdf/2506.18678)</td><td>◆ 提出首个分布式多智能体协作神经SLAM框架MCN-SLAM，结合混合隐式神经场景表示，解决传统单智能体SLAM在大场景和长序列中的局限性。  
◆ 创新设计三平面-网格联合场景表示方法，显著提升场景重建质量，优于现有神经隐式表示方案。  
◆ 开发新型&quot;内部-跨智能体&quot;闭环检测机制，首次实现单智能体局部与多智能体全局一致性协同优化。  
◆ 提出在线蒸馏方法实现多子地图融合，通过分布式通信优化解决NeRF类系统带宽受限问题。  
◆ 发布首个真实世界密集SLAM数据集DES，涵盖单/多智能体场景，提供连续轨迹与高精度3D网格真值，填补领域空白。  
实验证明该方法在建图、定位和通信效率上均优于现有技术，代码与数据集将开源推动SLAM与三维重建研究发展。</td></tr>
<tr><td>2025-06-24</td><td>Multimodal Fusion SLAM with Fourier Attention</td><td>[2506.18204](http://arxiv.org/pdf/2506.18204)</td><td>◆ 提出FMF-SLAM方法，通过快速傅里叶变换（FFT）提升多模态SLAM的算法效率，解决传统光流SLAM计算资源消耗大的问题。  
◆ 创新设计基于傅里叶的自注意力与跨注意力机制，有效融合RGB和深度信号的特征提取。  
◆ 引入多尺度跨模态知识蒸馏技术，增强多模态特征间的交互与互补性。  
◆ 结合GNSS-RTK全局定位模块与全局Bundle Adjustment，实现安全机器人的实时应用验证。  
◆ 在TUM、TartanAir及真实场景数据集上验证性能，在噪声、光照变化和黑暗条件下达到领先水平。  
◆ 公开代码与数据集，推动多模态SLAM领域的可复现研究。</td></tr>
<tr><td>2025-06-22</td><td>ADA-DPM: A Neural Descriptors-based Adaptive Noise Point Filtering Strategy for SLAM</td><td>[2506.18016](http://arxiv.org/pdf/2506.18016)</td><td>◆ 提出ADA-DPM自适应噪声过滤策略，在动态物体干扰和噪声环境下同时提升SLAM定位精度与系统鲁棒性。  
◆ 设计动态分割头（Dynamic Segmentation Head），通过预测特征点类别主动剔除动态特征点，减少动态干扰。  
◆ 引入全局重要性评分头（Global Importance Scoring Head），自适应筛选高贡献特征点并抑制噪声干扰，优化特征选择。  
◆ 构建跨层图内卷积模块（GLI-GCN），融合多尺度邻域结构，增强重叠特征的判别能力。  
◆ 在多个公开数据集上验证有效性，实验结果表明该方法性能优于现有技术。</td></tr>
<tr><td>2025-06-21</td><td>Optimizing Exploration with a New Uncertainty Framework for Active SLAM Systems</td><td>[2506.17775](http://arxiv.org/pdf/2506.17775)</td><td>◆提出不确定性地图（UM）框架，通过概率分布量化地图不确定性，为主动SLAM系统建立新型环境建模方法。  
◆定义不确定性边界（UF）作为探索-开发的关键目标与停止准则，解决传统方法中探索终止条件模糊的问题。  
◆创新性引入基于KL散度的符号相对熵（SiREn），首次实现覆盖度与不确定性的联合度量，仅需单一参数即可平衡探索与开发。  
◆设计传感器无关的通用架构，兼容相机、激光雷达及多传感器融合系统，突破现有方法对特定SLAM配置的依赖。  
◆结合UF的路径规划系统首次实现开放空间的自主探索能力，填补了主动SLAM文献中该行为的空白。  
◆开源ROS节点与完整数据集，推动方法验证与社区应用，增强研究可复现性。</td></tr>
<tr><td>2025-06-18</td><td>MCOO-SLAM: A Multi-Camera Omnidirectional Object SLAM System</td><td>[2506.15402](http://arxiv.org/pdf/2506.15402)</td><td>◆ 提出MCOO-SLAM系统，首次将多相机全景配置引入物体级SLAM，解决传统单目或RGB-D系统视场窄、遮挡敏感和深度感知受限的问题。  
◆ 融合点特征与开放词汇语义增强的物体级地标，实现复杂户外场景中更鲁棒且语义丰富的建图。  
◆ 设计语义-几何-时序多模态融合策略，显著提升跨视角物体关联的准确性，改善物体建模一致性。  
◆ 创新全景闭环检测模块，通过场景级描述符实现视角无关的地点识别，增强系统在动态环境中的稳定性。  
◆ 构建分层3D场景图谱抽象地图，为机器人高层推理任务提供结构化语义支持。  
实验证明该系统在遮挡、位姿变化和复杂环境下的定位精度与可扩展性均优于现有方法。</td></tr>
<tr><td>2025-06-24</td><td>RA-NeRF: Robust Neural Radiance Field Reconstruction with Accurate Camera Pose Estimation under Complex Trajectories</td><td>[2506.15242](http://arxiv.org/pdf/2506.15242)</td><td>◆ 提出RA-NeRF方法，能够在复杂相机轨迹下实现高精度的相机位姿估计，解决了传统NeRF和3DGS依赖准确位姿先验的问题。  
◆ 采用增量式重建流程，结合光度一致性约束和光流驱动的位姿调节机制，提升了初始化和定位阶段的鲁棒性。  
◆ 创新性地引入隐式位姿滤波器，通过捕捉相机运动模式有效消除位姿估计中的噪声干扰。  
◆ 在Tanks&amp;Temple和NeRFBuster两个数据集上验证了方法的有效性，其中NeRFBuster包含极具挑战性的相机轨迹场景。  
◆ 实验结果表明，RA-NeRF在相机位姿估计精度和场景重建视觉质量上均达到最先进水平，尤其在复杂轨迹条件下表现突出。</td></tr>
<tr><td>2025-06-18</td><td>SHeRLoc: Synchronized Heterogeneous Radar Place Recognition for Cross-Modal Localization</td><td>[2506.15175](http://arxiv.org/pdf/2506.15175)</td><td>◆ 提出SHeRLoc，首个专为异构雷达设计的深度网络，填补了跨模态雷达定位研究的空白。  
◆ 采用RCS极坐标匹配技术，有效对齐多模态雷达数据，解决异构传感器数据融合难题。  
◆ 提出基于分层最优传输的特征聚合方法，生成具有旋转鲁棒性的多尺度描述符。  
◆ 结合FFT相似性数据挖掘和自适应边界三元组损失，实现视场感知的度量学习。  
◆ 在公开数据集上实现召回率@1从不足0.1提升至0.9，性能超越现有最佳方法一个数量级。  
◆ 扩展性强，可应用于LiDAR等传感器，为跨模态地点识别和异构SLAM开辟新途径。</td></tr>
<tr><td>2025-06-18</td><td>VIMS: A Visual-Inertial-Magnetic-Sonar SLAM System in Underwater Environments</td><td>[2506.15126](http://arxiv.org/pdf/2506.15126)</td><td>◆ 提出VIMS系统，首次将视觉-惯性-磁力-声纳多模态融合用于水下SLAM，解决传统视觉-惯性方法在水下环境中的尺度估计和闭环难题。  
◆ 创新性引入低成本单波束声纳，有效提升水下尺度估计精度，克服纯视觉方法因水体折射导致的尺度漂移问题。  
◆ 利用高采样率磁力计配合经济型磁场线圈生成磁特征，实现基于磁场指纹的场所识别，填补水下无纹理区域的感知空白。  
◆ 设计分层式视觉-磁力混合闭环检测框架，通过多模态数据互补增强闭环鲁棒性，显著降低误匹配率。  
◆ 优化前端计算负载，平衡局部特征跟踪与全局描述子匹配，在不增加前端负担的前提下实现高效闭环。  
◆ 实验验证系统在复杂水下环境中的优越性，相比传统方法定位精度提升30%以上，为低成本水下自主导航提供新方案。</td></tr>
<tr><td>2025-06-16</td><td>Slanted light-sheet array microscopy for large volume imaging at rates exceeding 100 Hz</td><td>[2506.13664](http://arxiv.org/pdf/2506.13664)</td><td>◆ 开发了倾斜光片阵列显微镜（SLAM），实现了超过100 Hz的超快速大体积成像，突破了传统成像速度限制。  
◆ 基于标准宽场复合显微镜进行简单改造，仅需对照明光路进行最小化修改，便于集成和推广。  
◆ 支持大范围多维度高分辨率成像（横向超过500像素，深度超过200层），同时保持光学切片和局部光化学能力。  
◆ 结合深度学习（条件去噪扩散概率模型），实现了各向同性分辨率提升，优化了图像质量。  
◆ 兼容常规生物样本制备协议，适用于多种生物医学研究场景，具有广泛的应用潜力。  
◆ 在高速成像的同时兼顾了空间分辨率、信噪比和大视场需求，为动态生物过程观测提供了新工具。</td></tr>
<tr><td>2025-06-16</td><td>Cognitive Synergy Architecture: SEGO for Human-Centric Collaborative Robots</td><td>[2506.13149](http://arxiv.org/pdf/2506.13149)</td><td>◆ 提出SEGO（语义图谱本体）认知映射架构，首次将几何感知、语义推理和解释生成整合为统一框架，实现人机协作机器人的认知协同。  
◆ 构建动态认知场景图，突破传统SLAM仅关注空间几何的局限，同时表征环境中的语义关系和本体一致性。  
◆ 创新性地融合基于SLAM的定位、深度学习物体检测跟踪与本体驱动推理三大模块，实现实时语义连贯的环境建模。  
◆ 通过本体论约束确保语义推理的逻辑一致性，使机器人能理解&quot;桌子上的杯子&quot;等复杂语义关系。  
◆ 支持可解释性输出，机器人可生成对人类友好的场景解释，显著提升人机协作的透明度和信任度。  
◆ 该架构为人类中心协作机器人提供标准化认知处理流程，在工业装配、家庭服务等场景展现应用潜力。</td></tr>
<tr><td>2025-06-16</td><td>A Novel ViDAR Device With Visual Inertial Encoder Odometry and Reinforcement Learning-Based Active SLAM Method</td><td>[2506.13100](http://arxiv.org/pdf/2506.13100)</td><td>◆ 提出了一种新型ViDAR设备，结合视觉、惯性和电机编码器，构建紧耦合的视觉-惯性-编码器里程计（VIEO），显著提升了SLAM系统的主动能力和视野范围。  
◆ 设计了ViDAR校准方法，确保VIEO算法的精确初始化，解决了多传感器融合中的标定难题。  
◆ 首次将电机编码器引入SLAM系统，以极低的成本和结构复杂度增强了跨帧共视关系，提高了状态估计精度。  
◆ 提出基于深度强化学习（DRL）的平台运动解耦主动SLAM方法，能够自主优化运动策略以增加特征点多样性。  
◆ 实验证明，所提VIEO算法相比传统VIO算法在共视关系和估计精度上均有显著提升，且DRL主动SLAM进一步优化了系统性能。  
◆ 为复杂环境下主动SLAM系统的平台设计和运动解耦方法提供了新思路，兼具理论创新和实用价值。</td></tr>
<tr><td>2025-06-16</td><td>SuperPoint-SLAM3: Augmenting ORB-SLAM3 with Deep Features, Adaptive NMS, and Learning-Based Loop Closure</td><td>[2506.13089](http://arxiv.org/pdf/2506.13089)<br><a href=''>[代码]</a></td><td>◆ 用自监督的SuperPoint特征检测-描述子替代传统ORB特征，提升了在极端视角、尺度和光照变化下的鲁棒性。  
◆ 引入自适应非极大值抑制(ANMS)技术，实现空间分布更均匀的关键点提取，增强场景覆盖度。  
◆ 集成轻量级NetVLAD模块作为学习式回环检测器，显著改善了传统词袋模型的识别能力。  
◆ 在KITTI数据集上将平均平移误差从4.15%降至0.34%，旋转误差从0.0027度/米降至0.0010度/米。  
◆ 在EuRoC MAV数据集上所有序列误差降低约50%（如V2_03从1.58%降至0.79%）。  
◆ 保持ORB-SLAM3实时性能的同时，通过深度学习特征与学习式回环的融合实现了精度突破。</td></tr>
<tr><td>2025-06-12</td><td>LRSLAM: Low-rank Representation of Signed Distance Fields in Dense Visual SLAM System</td><td>[2506.10567](http://arxiv.org/pdf/2506.10567)</td><td>◆ 提出LRSLAM模型，采用低秩张量分解方法（Six-axis和CP分解）优化稠密视觉SLAM系统，显著提升计算效率和内存利用率。  
◆ 通过低秩表示有符号距离场（SDF），解决了传统神经隐式表示的高计算成本和内存占用问题，适合大规模场景。  
◆ 相比现有方法（如ESLAM的平面张量分解），进一步降低了内存增长压力，同时保持高精度重建与定位能力。  
◆ 在多种室内RGB-D数据集上验证，LRSLAM在参数效率、处理速度和准确性方面均优于当前最优方法。  
◆ 实现了更快的收敛速度和更高的系统鲁棒性，为自动驾驶、移动机器人等实时应用提供可行解决方案。  
◆ 代码将开源，促进相关领域研究发展。</td></tr>
<tr><td>2025-06-11</td><td>VAULT: A Mobile Mapping System for ROS 2-based Autonomous Robots</td><td>[2506.09583](http://arxiv.org/pdf/2506.09583)</td><td>◆ 提出VAULT原型系统，基于ROS 2框架，专为户外自主机器人设计，解决复杂环境下的实时定位与建图难题。  
◆ 创新性融合多传感器数据（GNSS、VIO、IMU）与扩展卡尔曼滤波（EKF），生成高可靠性3D里程计，提升户外定位鲁棒性。  
◆ 结合视觉SLAM（VSLAM）技术，构建精细3D点云地图，弥补传统2D LiDAR在户外场景的局限性。  
◆ 实现室内外环境通用性，通过多模态传感器协同，适应农业、林业等无结构化户外场景需求。  
◆ 提供开源ROS 2解决方案，为自主机器人社区提供可扩展、模块化的移动测绘系统（MMS）参考框架。</td></tr>
<tr><td>2025-06-10</td><td>UFM: A Simple Path towards Unified Dense Correspondence with Flow</td><td>[2506.09278](http://arxiv.org/pdf/2506.09278)</td><td>◆ 提出统一流与匹配模型（UFM），首次实现宽基线场景和光流估计的统一训练，突破传统分而治之的局限。  
◆ 采用简单通用的Transformer架构直接回归(u,v)流，避免传统 coarse-to-fine 代价体积的复杂性，训练更高效且对大位移更精准。  
◆ 在光流任务上精度超越当前最优方法（Unimatch）28%，在宽基线匹配任务上误差降低62%且速度提升6.7倍（对比RoMa）。  
◆ 首次证明统一训练模型可同时在光流和宽基线匹配两个领域超越专用方法，为通用稠密对应开辟新路径。  
◆ 通过共可见像素的统一数据训练，为多模态、长距离和实时对应任务提供新思路。</td></tr>
<tr><td>2025-06-10</td><td>Princeton365: A Diverse Dataset with Accurate Camera Pose</td><td>[2506.09035](http://arxiv.org/pdf/2506.09035)</td><td>◆ 提出了Princeton365数据集，包含365个多样化视频，提供高精度的相机位姿，填补了当前SLAM基准在精度和数据多样性之间的空白。  
◆ 设计了一种新颖的地面真值采集框架，结合校准板和360度相机，实现了室内、室外和物体扫描视频的多模态同步采集（单目/立体RGB视频和IMU）。  
◆ 提出了一种基于光流的场景尺度感知SLAM评估指标，克服了传统指标（如ATE）无法跨场景比较的局限性，便于分析算法失败模式。  
◆ 构建了具有挑战性的新视角合成（NVS）基准，涵盖当前基准未涉及的场景（如非朗伯表面和360度相机轨迹）。  
◆ 公开了完整的数据集、代码和提交平台（https://princeton365.cs.princeton.edu），推动SLAM和NVS领域的标准化研究。</td></tr>
<tr><td>2025-06-10</td><td>Planar Collisionless Shock Simulations with Semi-Implicit Particle-in-Cell Model FLEKS</td><td>[2506.08384](http://arxiv.org/pdf/2506.08384)</td><td>◆ 验证了半隐式粒子网格代码FLEKS在无碰撞激波模拟中的适用性，特别针对全球磁层建模相关参数范围。  
◆ 开发了精细化算法，使FLEKS能够在电子惯性长度量级的网格分辨率下精确模拟激波结构。  
◆ 成功捕捉了激波关键特征，包括激波结构（脚部、陡坡、过冲和欠冲）、上下游波动（快磁声波、哨声波、阿尔芬离子回旋波和镜像模）以及非麦克斯韦粒子分布。  
◆ 揭示了二维模拟对准确重现准垂直激波下游波动物理和准平行激波复杂动力学（如表面波纹、激波子、SLAMS和喷流）的必要性。  
◆ 通过参数研究阐明了质量比和网格分辨率对激波物理的影响，为半隐式PIC代码的物理和数值参数选择提供了重要指导。  
◆ 为将动力学激波过程整合到MHD-AEPIC模型的大尺度空间等离子体模拟中奠定了基础。</td></tr>
<tr><td>2025-06-09</td><td>ZeroVO: Visual Odometry with Minimal Assumptions</td><td>[2506.08005](http://arxiv.org/pdf/2506.08005)</td><td>ZeroVO是一种无需预训练即可泛化至不同相机和环境的视觉里程计算法，其核心贡献与创新点如下：

◆ 提出无需标定的几何感知网络结构，能有效处理深度估计和相机参数中的噪声，摆脱传统方法对固定标定配置的依赖。

◆ 引入基于语言的先验知识，通过语义信息增强特征提取的鲁棒性，显著提升模型在未知领域的泛化能力。

◆ 开发半监督训练框架，利用未标注数据迭代适应新场景，进一步强化模型在真实复杂场景中的适应能力。

◆ 在KITTI、nuScenes和Argoverse 2等标准数据集及GTA合成数据上验证性能，相对现有方法提升超过30%。

◆ 无需微调或相机标定的特性，使得该技术具备大规模实际部署的潜力，极大扩展了视觉里程计的应用范围。</td></tr>
<tr><td>2025-06-08</td><td>Faster than Fast: Accelerating Oriented FAST Feature Detection on Low-end Embedded GPUs</td><td>[2506.07164](http://arxiv.org/pdf/2506.07164)</td><td>这篇论文的核心贡献是通过优化低端嵌入式GPU上的Oriented FAST特征检测，显著提升了视觉SLAM系统的实时处理能力。具体创新点如下：

◆ 提出了一种二进制级编码策略，快速确定FAST特征点候选点，大幅减少了计算复杂度。  
◆ 设计了一种可分离的Harris角点检测策略，结合底层GPU硬件指令优化，提高了计算效率。  
◆ 在Jetson TX2嵌入式GPU上实现了平均7.3倍的速度提升，远超现有OpenCV的GPU加速方案。  
◆ 通过优化FAST特征点检测和Harris角点检测这两个最耗时的步骤，解决了移动平台实时处理的瓶颈问题。  
◆ 为资源受限环境下的实时SLAM应用提供了高效解决方案，具有广泛的移动和嵌入式应用潜力。</td></tr>
<tr><td>2025-06-08</td><td>UNO: Unified Self-Supervised Monocular Odometry for Platform-Agnostic Deployment</td><td>[2506.07013](http://arxiv.org/pdf/2506.07013)</td><td>◆ 提出UNO框架，实现跨平台、跨环境的统一自监督单目视觉里程计，无需针对特定场景或设备进行调优。  
◆ 采用混合专家策略（Mixture-of-Experts），通过多个专用解码器分别处理不同类别的运动模式，提升泛化能力。  
◆ 设计可微分的Gumbel-Softmax模块，动态构建帧间关联图并选择最优解码器，同时剔除错误估计。  
◆ 结合预训练的尺度无关深度先验与轻量级捆绑调整（bundling adjustment），后端统一优化几何一致性。  
◆ 在KITTI（自动驾驶）、EuRoC-MAV（无人机）和TUM-RGBD（手持设备）三大数据集上验证，性能达到SOTA。</td></tr>
<tr><td>2025-06-06</td><td>GS4: Generalizable Sparse Splatting Semantic SLAM</td><td>[2506.06517](http://arxiv.org/pdf/2506.06517)</td><td>◆ 提出了首个基于可泛化高斯泼溅（GS）的语义SLAM算法，通过学习网络实现跨场景的3D地图构建，摆脱传统方法依赖单场景优化的限制。  
◆ 采用RGB-D图像识别主干网络，直接从降采样和反向投影的图像位置预测高斯参数，实现高效增量式地图更新。  
◆ 创新性地将3D语义分割集成到GS框架中，通过共享主干网络统一3D建图与识别任务，提升语义理解能力。  
◆ 提出仅需1次迭代的全局定位后优化策略，有效解决定位漂移和漂浮物问题，显著提升系统鲁棒性。  
◆ 在ScanNet数据集上实现SOTA性能，所用高斯数量比同类方法少一个数量级，并在NYUv2和TUM RGB-D上展示零样本泛化能力。</td></tr>
<tr><td>2025-06-06</td><td>Enhancing Situational Awareness in Underwater Robotics with Multi-modal Spatial Perception</td><td>[2506.06476](http://arxiv.org/pdf/2506.06476)</td><td>◆ 提出多模态感知融合框架，整合摄像头、IMU和声学设备数据，解决水下视觉SLAM因光线衰减和低对比度导致的失效问题。  
◆ 突破传统单目/双目视觉限制，支持多摄像头配置，提升系统在复杂水下环境中的可扩展性。  
◆ 结合几何方法与学习技术，引入语义分析增强场景理解，实现更鲁棒的状态估计和3D重建。  
◆ 通过真实海域实验验证（特隆赫姆峡湾），首次展示多模态系统在恶劣水下条件下的实时可靠性能。  
◆ 系统分析传感器标定等工程挑战，指出基于学习方法的局限性，为未来大规模水下作业研究指明方向。</td></tr>
<tr><td>2025-06-06</td><td>Dy3DGS-SLAM: Monocular 3D Gaussian Splatti...</td><td>[2506.05965](http://arxiv.org/pdf/2506.05965)</td><td>◆ 提出了首个基于单目RGB输入的动态场景3D高斯泼溅SLAM系统Dy3DGS-SLAM。  
◆ 通过融合光流掩码和深度掩码的概率模型生成动态掩码，仅需单次网络迭代即可优化跟踪尺度和几何渲染。...</td></tr>
<tr><td>2025-06-06</td><td>Analysis of points outcome in ATP Grand Slam Tenni...</td><td>[2506.05866](http://arxiv.org/pdf/2506.05866)</td><td>◆ 该论文创新地利用大数据和机器学习方法（如逻辑回归、随机森林等）预测网球大满贯赛事中每一分的胜负，并结合球员排名、历史数据等因素分析影响得分的关键战略因素。  
◆ 研究基于2016-2020...</td></tr>
<tr><td>2025-06-05</td><td>On-the-fly Reconstruction for Large-Scale Novel Vi...</td><td>[2506.05558](http://arxiv.org/pdf/2506.05558)</td><td>◆提出实时重建方法，在拍摄后立即生成相机位姿和训练好的3D高斯泼溅模型，解决了传统方法耗时过长的问题。  
◆结合快速初始位姿估计和直接采样高斯基元位置/形状的技术，显著加速联合优化过程。  
...</td></tr>
<tr><td>**2025-06-05**</td><td>**Deep Learning Reforms Image Matching: A Survey and Outlook**</td><td>[2506.04619](http://arxiv.org/abs/2506.04619)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-04**</td><td>**cuVSLAM: CUDA accelerated visual odometry**</td><td>[2506.04359](http://arxiv.org/abs/2506.04359)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-04**</td><td>**Seeing in the Dark: Benchmarking Egocentric 3D Vision with the Oxford Day-and-Night Dataset**</td><td>[2506.04224](http://arxiv.org/abs/2506.04224)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-03**</td><td>**LEG-SLAM: Real-Time Language-Enhanced Gaussian Splatting for SLAM**</td><td>[2506.03073](http://arxiv.org/abs/2506.03073)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-03**</td><td>**Online Performance Assessment of Multi-Source-Localization for Autonomous Driving Systems Using Subjective Logic**</td><td>[2506.02932](http://arxiv.org/abs/2506.02932)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-03**</td><td>**VTGaussian-SLAM: RGBD SLAM for Large Scale Scenes with Splatting View-Tied 3D Gaussians**</td><td>[2506.02741](http://arxiv.org/abs/2506.02741)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-03**</td><td>**GeneA-SLAM2: Dynamic SLAM with AutoEncoder-Preprocessed Genetic Keypoints Resampling and Depth Variance-Guided Dynamic Region Removal**</td><td>[2506.02736](http://arxiv.org/abs/2506.02736)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-03**</td><td>**Olfactory Inertial Odometry: Methodology for Effective Robot Navigation by Scent**</td><td>[2506.02373](http://arxiv.org/abs/2506.02373)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-06-01**</td><td>**Globally Consistent RGB-D SLAM with 2D Gaussian Splatting**</td><td>[2506.00970](http://arxiv.org/abs/2506.00970)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-30**</td><td>**Black-box Adversarial Attacks on CNN-based SLAM Algorithms**</td><td>[2505.24654](http://arxiv.org/abs/2505.24654)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**Semantic Exploration and Dense Mapping of Complex Environments using Ground Robots Equipped with LiDAR and Panoramic Camera**</td><td>[2505.22880](http://arxiv.org/abs/2505.22880)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians**</td><td>[2505.22859](http://arxiv.org/abs/2505.22859)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**UP-SLAM: Adaptively Structured Gaussian SLAM with Uncertainty Prediction in Dynamic Environments**</td><td>[2505.22335](http://arxiv.org/abs/2505.22335)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-27**</td><td>**HS-SLAM: A Fast and Hybrid Strategy-Based SLAM Approach for Low-Speed Autonomous Driving**</td><td>[2505.20906](http://arxiv.org/abs/2505.20906)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-27**</td><td>**ProBA: Probabilistic Bundle Adjustment with the Bhattacharyya Coefficient**</td><td>[2505.20858](http://arxiv.org/abs/2505.20858)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**ADD-SLAM: Adaptive Dynamic Dense SLAM with Gaussian Splatting**</td><td>[2505.19420](http://arxiv.org/abs/2505.19420)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-25**</td><td>**VPGS-SLAM: Voxel-based Progressive 3D Gaussian SLAM in Large-Scale Scenes**</td><td>[2505.18992](http://arxiv.org/abs/2505.18992)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-23**</td><td>**CU-Multi: A Dataset for Multi-Robot Data Association**</td><td>[2505.17576](http://arxiv.org/abs/2505.17576)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-22**</td><td>**TAT-VPR: Ternary Adaptive Transformer for Dynamic and Efficient Visual Place Recognition**</td><td>[2505.16447](http://arxiv.org/abs/2505.16447)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-20**</td><td>**A Methodological Framework for Measuring Spatial Labeling Similarity**</td><td>[2505.14128](http://arxiv.org/abs/2505.14128)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-22**</td><td>**Place Recognition: A Comprehensive Review, Current Challenges and Future Directions**</td><td>[2505.14068](http://arxiv.org/abs/2505.14068)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-19**</td><td>**eStonefish-scenes: A synthetically generated dataset for underwater event-based optical flow prediction tasks**</td><td>[2505.13309](http://arxiv.org/abs/2505.13309)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-23**</td><td>**VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold**</td><td>[2505.12549](http://arxiv.org/abs/2505.12549)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-18**</td><td>**Is Semantic SLAM Ready for Embedded Systems ? A Comparative Survey**</td><td>[2505.12384](http://arxiv.org/abs/2505.12384)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-18**</td><td>**Structureless VIO**</td><td>[2505.12337](http://arxiv.org/abs/2505.12337)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-16**</td><td>**EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video**</td><td>[2505.11709](http://arxiv.org/abs/2505.11709)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-16**</td><td>**Improved Bag-of-Words Image Retrieval with Geometric Constraints for Ground Texture Localization**</td><td>[2505.11620](http://arxiv.org/abs/2505.11620)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-16**</td><td>**Robust 2D lidar-based SLAM in arboreal environments without IMU/GNSS**</td><td>[2505.10847](http://arxiv.org/abs/2505.10847)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-15**</td><td>**TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation**</td><td>[2505.10696](http://arxiv.org/abs/2505.10696)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-15**</td><td>**A hybrid SLAM-Payne framework for atmospheric parameter and abundance determination of early-type Stars from LAMOST DR9 low-resolution Spectra**</td><td>[2505.10310](http://arxiv.org/abs/2505.10310)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-15**</td><td>**Large-Scale Gaussian Splatting SLAM**</td><td>[2505.09915](http://arxiv.org/abs/2505.09915)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-13**</td><td>**Automated Meta Prompt Engineering for Alignment with the Theory of Mind**</td><td>[2505.09024](http://arxiv.org/abs/2505.09024)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-13**</td><td>**MDF: Multi-Modal Data Fusion with CNN-Based Object Detection for Enhanced Indoor Localization Using LiDAR-SLAM**</td><td>[2505.08388](http://arxiv.org/abs/2505.08388)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-13**</td><td>**SKiD-SLAM: Robust, Lightweight, and Distributed Multi-Robot LiDAR SLAM in Resource-Constrained Field Environments**</td><td>[2505.08230](http://arxiv.org/abs/2505.08230)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-12**</td><td>**RDD: Robust Feature Detector and Descriptor using Deformable Transformer**</td><td>[2505.08013](http://arxiv.org/abs/2505.08013)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-12**</td><td>**Ranking-aware Continual Learning for LiDAR Place Recognition**</td><td>[2505.07198](http://arxiv.org/abs/2505.07198)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-07**</td><td>**Scalable Aerial GNSS Localization for Marine Robots**</td><td>[2505.04095](http://arxiv.org/abs/2505.04095)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-06**</td><td>**Thermal-LiDAR Fusion for Robust Tunnel Localization in GNSS-Denied and Low-Visibility Conditions**</td><td>[2505.03565](http://arxiv.org/abs/2505.03565)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-06**</td><td>**AquaticVision: Benchmarking Visual SLAM in Underwater Environment with Events and Frames**</td><td>[2505.03448](http://arxiv.org/abs/2505.03448)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-06**</td><td>**LiftFeat: 3D Geometry-Aware Local Feature Matching**</td><td>[2505.03422](http://arxiv.org/abs/2505.03422)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-05**</td><td>**LiDAR-Inertial SLAM-Based Navigation and Safety-Oriented AI-Driven Control System for Skid-Steer Robots**</td><td>[2505.02598](http://arxiv.org/abs/2505.02598)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-04**</td><td>**Robust Localization, Mapping, and Navigation for Quadruped Robots**</td><td>[2505.02272](http://arxiv.org/abs/2505.02272)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-04**</td><td>**SafeNav: Safe Path Navigation using Landmark Based Localization in a GPS-denied Environment**</td><td>[2505.01956](http://arxiv.org/abs/2505.01956)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-03**</td><td>**GauS-SLAM: Dense RGB-D SLAM with Gaussian Surfels**</td><td>[2505.01934](http://arxiv.org/abs/2505.01934)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-02**</td><td>**Tightly Coupled Range Inertial Odometry and Mapping with Exact Point Cloud Downsampling**</td><td>[2505.01017](http://arxiv.org/abs/2505.01017)</td><td>摘要生成中...</td></tr>
</tbody>
</table>
</div>

<h2 id='sfm'>SFM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-06-30</td><td>Towards Initialization-free Calibrated Bundle Adjustment</td><td>[2506.23808](http://arxiv.org/pdf/2506.23808)</td><td>◆ 提出了一种无需初始化的标定束调整方法，能够在初始重建阶段直接利用相机标定信息，生成接近度量精度的重建结果（仅差一个相似变换）。  
◆ 创新性地引入具有标定信息的成对相对旋转估计，这些旋转估计仅对相似变换保持不变，从而推动解保持真实场景的度量特征。  
◆ 将旋转平均技术整合到伪物体空间误差（pOSE）框架中，实现了标定信息与初始化无关的SfM（运动恢复结构）流程。  
◆ 实验证明该方法能够可靠优化目标函数，即使从随机初始解出发也能高概率收敛到全局最优，获得精确的接近度量重建。  
◆ 相比现有基于pOSE的方法（仅能获得射影变换解且需要更多数据），新方法显著提升了重建精度和效率。</td></tr>
<tr><td>2025-06-30</td><td>AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention</td><td>[2506.23611](http://arxiv.org/pdf/2506.23611)</td><td>◆ 提出AttentionGS框架，首次实现无需高质量初始点云的3D高斯泼溅重建，突破传统3DGS对SfM点云的强依赖。  
◆ 创新性引入几何注意力机制，在训练初期快速恢复场景全局结构，解决随机初始化导致的收敛难题。  
◆ 设计渐进式纹理注意力模块，在训练后期精细化局部细节，显著提升纹理缺失场景的渲染质量。  
◆ 开发不透明度加权梯度策略，优化高斯分布致密化过程，实现更精准的表面重建。  
◆ 在标准数据集上全面超越现有方法，尤其在低纹理/受限视角场景中表现突出，验证了方案的鲁棒性。  
◆ 为实际应用中复杂场景的3D重建提供新思路，扩展了3DGS技术的适用边界。</td></tr>
<tr><td>2025-06-27</td><td>Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras</td><td>[2506.22069](http://arxiv.org/pdf/2506.22069)</td><td>◆ 提出了一种基于单扫描线投影交点的新方法，用于估计滚动快门相机间的相对位姿，无需显式建模相机运动。  
◆ 创新性地实现了单视图内扫描线的相对位姿估计，扩展了滚动快门相机的应用场景。  
◆ 该方法作为滚动快门运动恢复结构（SfM）的基础模块，支持独立计算每条扫描线的位姿，且无需运动模型假设。  
◆ 在已知内参和无镜头畸变的条件下，分类了通用和特定场景（如平行线和已知重力方向）的最小求解器。  
◆ 针对平行线场景，开发了带/不带重力先验的最小求解器，通过将其与1D相机的2D结构估计问题关联实现创新求解。  
◆ 在Fastec数据集上的实验验证了该方法用于滚动快门SfM初始化的可行性，展现了进一步开发的潜力。</td></tr>
<tr><td>2025-06-24</td><td>ICP-3DGS: SfM-free 3D Gaussian Splatting for Large-scale Unbounded Scenes</td><td>[2506.21629](http://arxiv.org/pdf/2506.21629)</td><td>◆ 提出了一种无需SfM预处理的方法ICP-3DGS，将迭代最近点（ICP）与基于优化的位姿细化相结合，解决了大范围无边界场景中相机位姿估计的难题。  
◆ 创新性地将ICP引入3D高斯泼溅（3DGS）框架，实现了在大幅度相机运动下的高精度位姿估计，突破了传统神经渲染对SfM先验的依赖。  
◆ 设计了基于体素的场景致密化策略，有效指导大规模场景的重建过程，提升了场景覆盖率和几何细节的完整性。  
◆ 在室内外多种尺度场景的实验中，ICP-3DGS在相机位姿估计和新视角合成任务上均优于现有方法，证明了其鲁棒性和泛化能力。  
◆ 开源了完整代码，为后续研究提供了可复现的基础，推动了无预计算位姿的神经渲染技术的发展。</td></tr>
<tr><td>2025-06-26</td><td>Wild refitting for black box prediction</td><td>[2506.21460](http://arxiv.org/pdf/2506.21460)</td><td>◆ 提出了一种名为&quot;wild refitting&quot;的高效计算流程，仅需单次数据集和预测方法的黑箱访问，通过残差计算、对称化和缩放三个步骤，为惩罚非参数估计提供实例级均方预测误差的高概率上界。  
◆ 创新性地采用Rademacher残差对称化技术（类似wild bootstrap变体），通过预定义缩放因子ρ调整残差，构建以当前估计为中心的修正预测问题。  
◆ 在允许噪声异质性的较温和条件下，理论证明了该方法性能：当wild噪声尺度ρ选择适当时，wild refit能确保预测误差上界的有效性。  
◆ 为实际应用提供关键设计指导，包括残差构建方法、wild子问题中噪声缩放量的选择依据，以及黑箱程序局部稳定性的分析框架。  
◆ 展示了方法在多个领域的适用性，如基于结构化矩阵惩罚的非刚性运动恢复、深度神经网络先验的即插即用图像修复，以及核方法的随机草图技术。</td></tr>
<tr><td>2025-06-24</td><td>Experimental Assessment of Neural 3D Reconstruction for Small UAV-based Applications</td><td>[2506.19491](http://arxiv.org/pdf/2506.19491)</td><td>◆ 提出了一种将神经三维重建（N3DR）技术与小型无人机系统结合的新方法，用于精细三维数字重建静态小物体。  
◆ 设计并实现了一套基于N3DR的流程，整合了Instant-ngp、Nerfacto和Splatfacto等先进模型，显著提升了重建质量。  
◆ 通过多无人机协同采集图像，解决了小型无人机在动态飞行和功耗限制下的自主性与任务能力问题。  
◆ 采用多种图像和点云指标评估模型性能，并与传统运动恢复结构（SfM）算法对比，验证了N3DR的优越性。  
◆ 实验证明该方案能支持高精度三维建模和异常检测，拓展了小型无人机在受限环境中的应用潜力。  
◆ 整体研究展示了N3DR技术在提升微型无人机系统能力方面的广阔前景。</td></tr>
<tr><td>2025-06-23</td><td>ViDAR: Video Diffusion-Aware 4D Reconstruction From Monocular Inputs</td><td>[2506.18792](http://arxiv.org/pdf/2506.18792)</td><td>◆ 提出ViDAR框架，首次将个性化扩散模型引入单目视频的4D重建任务，通过生成伪多视角监督信号解决单目输入的结构-运动歧义问题。  
◆ 创新性地利用场景特定特征进行条件扩散，在保持外观细节的同时有效缓解单目模糊性导致的伪影问题。  
◆ 设计扩散感知损失函数，专门处理扩散生成视图的时空不一致性，提升合成视图与真实几何的对齐精度。  
◆ 提出相机位姿优化策略，动态调整合成视角与底层场景几何的匹配关系，增强动态区域的几何一致性。  
◆ 在极端视角变化的DyCheck基准测试中全面超越现有方法，尤其在运动丰富区域重建质量上取得显著提升。  
◆ 发布新评测基准，首次针对场景中高动态部分的重建性能进行系统化比较，推动领域评估标准发展。</td></tr>
<tr><td>2025-06-23</td><td>Room temperature spin injection into commercial VCSELs at non-resonant wavelengths</td><td>[2506.18376](http://arxiv.org/pdf/2506.18376)</td><td>◆ 首次在室温下实现了对商用垂直腔面发射激光器（VCSEL）的非共振波长自旋注入，突破了传统共振波长限制。  
◆ 通过794 nm和810 nm光泵浦实验，观察到20%和5%的最大圆偏振度差异，揭示了波长对自旋注入效率的影响机制。  
◆ 结合量子阱光学取向研究，证实长波长激发会导致自旋注入效率降低，为器件优化提供理论依据。  
◆ 扩展自旋翻转模型（SFM），首次纳入实际激发条件，使理论模型能准确复现实验观测趋势。  
◆ 该成果为自旋激光器的低阈值、高速调制和全光数据处理等应用提供了新的实现路径。</td></tr>
<tr><td>2025-06-11</td><td>OWSM-Biasing: Contextualizing Open Whisper-Style Speech Models for Automatic Speech Recognition with Dynamic Vocabulary</td><td>[2506.09448](http://arxiv.org/pdf/2506.09448)</td><td>◆ 提出了一种将上下文偏置（CB）方法与预训练的开放Whisper风格语音模型（OWSM v3.1）结合的新方法，无需微调预训练参数。  
◆ 通过利用预训练语音基础模型（SFMs）的嵌入知识，即使在小数据集上也能有效提升罕见词和未登录词的识别准确率。  
◆ 该方法在保持SFMs原有优势的同时，显著降低了偏置词错误率（B-WER），在LibriSpeech测试集上提升11.6个百分点。  
◆ 整体词错误率（WER）改善0.9个百分点，同时实时因子（RTF）减少7.5%，兼顾性能与效率。  
◆ 实验证明，该方法优于从头训练的CB方法，凸显了预训练模型知识迁移的重要性。</td></tr>
<tr><td>2025-06-06</td><td>SurGSplat: Progressive Geometry-Constrained Gaussian Splatting for Surgical Scene Reconstruction</td><td>[2506.05935](http://arxiv.org/pdf/2506.05935)</td><td>◆提出SurGSplat新范式，通过渐进式几何约束优化3D高斯泼溅(3DGS)技术，解决内窥镜场景稀疏特征和光照不均导致的传统SfM方法重建失败问题。  
◆首创将几何约束融入3DGS优化过程，实现血管等关键解剖结构的高精度重建，显著提升手术场景的视觉清晰度。  
◆开发渐进式优化框架，逐步细化重建细节，在保持实时性的同时突破现有方法在复杂手术环境中的性能瓶颈。  
◆实验证明该方法在新型视角合成(NVS)和位姿估计精度上均超越现有技术，为术中导航提供高保真重建解决方案。  
◆通过专属几何约束机制有效克服内窥镜图像特征稀疏的固有挑战，为微创手术提供更可靠的3D场景理解支持。  
◆开源项目网站提供完整技术细节和可视化结果，推动手术导航领域的可重复研究。</td></tr>
<tr><td>2025-06-05</td><td>On-the-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images</td><td>[2506.05558](http://arxiv.org/pdf/2506.05558)</td><td>◆ 提出实时重建方法，能够在图像采集完成后立即生成相机位姿和训练好的3D高斯泼溅模型，显著缩短传统方法所需的分钟到小时级计算时间。  
◆ 针对大场景和宽基线图像序列，设计了快速初始位姿估计方案，结合学习特征和GPU友好的小型束调整，提升处理效率。  
◆ 创新性地采用高斯图元位置与形状的直接采样方法，通过增量式生成图元加速训练过程，实现位姿与高斯图元的快速联合优化。  
◆ 提出可扩展的辐射场构建技术，通过渐进式聚类将3DGS图元存储在锚点中并从GPU卸载，有效管理大规模场景的内存需求。  
◆ 引入动态图元合并机制，根据视点需求自适应调整3DGS规模，保持渲染质量的同时优化计算资源使用。  
◆ 实验验证该方法能实时处理多种采集场景和不同规模的数据集，在速度、图像质量或两者兼备方面优于仅针对特定场景的现有方法。</td></tr>
<tr><td>2025-06-05</td><td>SupeRANSAC: One RANSAC to Rule Them All</td><td>[2506.04803](http://arxiv.org/pdf/2506.04803)<br><a href=''>[代码]</a></td><td>◆ SupeRANSAC提出了一种统一的RANSAC框架，解决了传统RANSAC在不同视觉任务中性能不稳定的问题。  
◆ 通过系统分析RANSAC在特定视觉任务（如单应性矩阵、基础矩阵、位姿估计等）中的有效技术，优化了整体流程。  
◆ 相比现有最佳方法，SupeRANSAC在基础矩阵估计任务中平均提升了6个AUC点，表现出更高的准确性。  
◆ 该框架克服了现有库（如OpenCV和PoseLib）在不同任务中表现不一致的缺陷，实现了跨任务的稳定高性能。  
◆ 论文提供了详细的实现细节和任务特定优化，为鲁棒估计领域提供了可复现的高效解决方案。  
◆ 开源代码便于社区验证和应用，已在多个数据集和问题上展示了显著的性能提升。</td></tr>
<tr><td>2025-06-04</td><td>Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation</td><td>[2506.04225](http://arxiv.org/pdf/2506.04225)</td><td>◆ Voyager提出了一种新颖的视频扩散框架，能够从单张图像生成用户自定义相机路径下的3D点云序列，实现端到端的世界一致场景生成，无需依赖传统3D重建流程。  
◆ 该框架首次整合了RGB与深度视频的联合生成，通过现有世界观测条件确保全局一致性，解决了长序列生成中的累积误差问题。  
◆ 创新性地采用带点云剔除的世界缓存机制和自回归推理方法，支持上下文感知的迭代场景扩展，实现超长距离（&gt;100米）的3D场景探索。  
◆ 开发了可扩展的数据引擎，通过自动化相机位姿估计和深度预测，构建大规模无人工标注的训练数据集，显著降低数据获取成本。  
◆ 在视觉质量和几何精度上超越现有方法，支持虚拟现实、游戏开发等需要动态3D场景构建的应用场景。  
◆ 整体架构摒弃了多阶段处理流程，首次实现单模型直接输出几何一致的可探索3D场景，为生成式3D建模开辟新方向。</td></tr>
<tr><td>2025-06-04</td><td>Accelerating SfM-based Pose Estimation with Dominating Set</td><td>[2506.03667](http://arxiv.org/pdf/2506.03667)</td><td>◆ 提出基于支配集的预处理技术，显著加速基于SfM的位姿估计过程，适用于AR/VR和机器人等实时应用场景。  
◆ 首次将图论中的支配集概念引入SfM模型优化，在不显著损失精度前提下实现计算效率提升。  
◆ 在OnePose数据集上验证了方法的普适性，兼容多种SfM位姿估计技术，展现广泛适用性。  
◆ 实现1.5-14.48倍的加速效果，同时将参考图像数量和点云规模分别缩减17-23倍和2.27-4倍。  
◆ 通过平衡速度与精度，为实时3D位姿估计提供了高效解决方案，突破现有技术瓶颈。</td></tr>
<tr><td>2025-06-03</td><td>Nearby dwarf galaxies with extreme star formation rates: a window into dwarf-galaxy evolution in the early Universe</td><td>[2506.03265](http://arxiv.org/pdf/2506.03265)</td><td>◆ 研究发现附近低光度矮星系（质量10^7-10^8太阳质量）存在极端恒星形成率（0.1-3太阳质量/年），可作为早期宇宙（z~5.5）矮星系的类比样本。  
◆ 通过对比正常矮星系样本，发现极端恒星形成率并非由星系结构紧凑性或特殊环境（如靠近节点/纤维结构）驱动。  
◆ 揭示具有极端恒星形成率的矮星系中相互作用星系和早型形态比例显著升高（分别增加约5.6倍和9倍），表明星系相互作用是关键触发机制。  
◆ 指出当前基于中低红移数据的主序星形成率演化模型会低估早期宇宙（z~5.5）矮星系的恒星形成率。  
◆ 提出早期宇宙矮星系通过更高气体丰度与频繁相互作用的共同作用，驱动其恒星质量快速累积的新演化图景。</td></tr>
<tr><td>2025-06-02</td><td>Fast and Robust Rotation Averaging with Anisotropic Coordinate Descent</td><td>[2506.01940](http://arxiv.org/pdf/2506.01940)</td><td>◆ 提出了一种快速且鲁棒的各向异性旋转平均方法，通过分析块坐标下降法家族，简化了原有和弦距离优化的复杂形式。  
◆ 首次将各向异性扩展应用于块坐标下降法，开发出一个通用的快速求解器，显著提升了计算效率。  
◆ 将该求解器集成到大规模鲁棒旋转平均流程中，解决了传统方法在问题规模增大时计算效率低下的问题。  
◆ 通过实验验证，该方法在公开的结构运动数据集上达到了最先进的性能表现。  
◆ 克服了传统局部方法对初始化的敏感性，避免了最小生成树方法中常见的漂移累积和局部极小值陷阱问题。  
◆ 在全局最优性、鲁棒性和效率之间取得了良好平衡，为各向异性旋转平均提供了实用解决方案。</td></tr>
<tr><td>2025-06-03</td><td>Improving Multilingual Speech Models on ML-SUPERB 2.0: Fine-tuning with Data Augmentation and LID-Aware CTC</td><td>[2505.24200](http://arxiv.org/pdf/2505.24200)</td><td>◆ 提出多种微调策略（冻结上游训练、部分微调、低秩适应）优化多语言语音基础模型（SFM）在ML-SUPERB 2.0上的表现。  
◆ 采用数据增强技术缓解少样本场景下的性能下降问题，提升模型在资源受限条件下的鲁棒性。  
◆ 创新性地引入语言识别（LID）感知的CTC损失函数作为正则化手段，联合优化LID和ASR任务。  
◆ 在ML-SUPERB 2.0基准上实现显著提升：LID准确率相对提高14%，ASR字错误率（CER）相对降低30%。  
◆ 综合方法在Interspeech 2025 ML-SUPERB 2.0挑战赛中斩获第二名，验证了策略的有效性。</td></tr>
<tr><td>2025-05-29</td><td>Rooms from Motion: Un-posed Indoor 3D Object Detection as Localization and Mapping</td><td>[2505.23756](http://arxiv.org/pdf/2505.23756)</td><td>◆ 提出Rooms from Motion (RfM)方法，首次实现无需先验相机位姿的室内3D物体检测，通过物体中心化框架同时完成定位与建图。  
◆ 创新性地用图像衍生的3D定向包围盒替代传统基于2D关键点的匹配器，从而估计相机位姿并生成全局语义3D物体地图。  
◆ 在已有相机位姿时，通过全局3D包围盒优化显著提升地图质量，优于依赖点云或多视图的过参数化方法。  
◆ 实现稀疏定位与参数化建图，其计算复杂度仅与场景中物体数量成正比，效率更高。  
◆ 在CA-1M和ScanNet++数据集上，RfM的定位性能与地图质量均超越基于点云和密集体素的领先方法。  
◆ 扩展Cubify Anything至全场景，建立通用的物体中心化表征，为场景理解提供新范式。</td></tr>
<tr><td>2025-05-30</td><td>FAMA: The First Large-Scale Open-Sc...</td><td>[2505.22759](http://arxiv.org/pdf/2505.22759)<br><a href=''>[代码]</a></td><td>◆FAMA是首个面向英语和意大利语的大规模开源语音基础模型，填补了语音领域开放科学的空白。  
◆创新性地使用15万+小时开源语音数据训练，并发布了包含1.6万小时清洗和伪标注数据的新数据集。 ...</td></tr>
<tr><td>**2025-05-28**</td><td>**UAVPairs: A Challenging Benchmark for Match Pair Retrieval of Large-scale UAV Images**</td><td>[2505.22098](http://arxiv.org/abs/2505.22098)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**Fast Feature Matching of UAV Images via Matrix Band Reduction-based GPU Data Schedule**</td><td>[2505.22089](http://arxiv.org/abs/2505.22089)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-30**</td><td>**Towards Robust Assessment of Pathological Voices via Combined Low-Level Descriptors and Foundation Model Representations**</td><td>[2505.21356](http://arxiv.org/abs/2505.21356)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-27**</td><td>**Intern-GS: Vision Model Guided Sparse-View 3D Gaussian Splatting**</td><td>[2505.20729](http://arxiv.org/abs/2505.20729)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**Robust fine-tuning of speech recognition models via model merging: application to disordered speech**</td><td>[2505.20477](http://arxiv.org/abs/2505.20477)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-29**</td><td>**Sparse2DGS: Sparse-View Surface Reconstruction using 2D Gaussian Splatting with Dense Point Cloud**</td><td>[2505.19854](http://arxiv.org/abs/2505.19854)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-25**</td><td>**Improving Novel view synthesis of 360$^\circ$ Scenes in Extremely Sparse Views by Jointly Training Hemisphere Sampled Synthetic Images**</td><td>[2505.19264](http://arxiv.org/abs/2505.19264)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-24**</td><td>**Token-Level Logits Matter: A Closer Look at Speech Foundation Models for Ambiguous Emotion Recognition**</td><td>[2505.18484](http://arxiv.org/abs/2505.18484)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-22**</td><td>**Tracking the Flight: Exploring a Computational Framework for Analyzing Escape Responses in Plains Zebra (Equus quagga)**</td><td>[2505.16882](http://arxiv.org/abs/2505.16882)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-21**</td><td>**A Taxonomy of Structure from Motion Methods**</td><td>[2505.15814](http://arxiv.org/abs/2505.15814)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-18**</td><td>**Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis**</td><td>[2505.12226](http://arxiv.org/abs/2505.12226)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-15**</td><td>**Mapping Semantic Segmentation to Point Clouds Using Structure from Motion for Forest Analysis**</td><td>[2505.10751](http://arxiv.org/abs/2505.10751)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-13**</td><td>**Unveiling the Best Practices for Applying Speech Foundation Models to Speech Intelligibility Prediction for Hearing-Impaired People**</td><td>[2505.08215](http://arxiv.org/abs/2505.08215)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-12**</td><td>**RDD: Robust Feature Detector and Descriptor using Deformable Transformer**</td><td>[2505.08013](http://arxiv.org/abs/2505.08013)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-12**</td><td>**Geometric Prior-Guided Neural Implicit Surface Reconstruction in the Wild**</td><td>[2505.07373](http://arxiv.org/abs/2505.07373)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-11**</td><td>**Symmetry in Fundamental Parameters of Galaxies on the Star-forming Main Sequence**</td><td>[2505.06868](http://arxiv.org/abs/2505.06868)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-10**</td><td>**TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge For Interpretability and Kinematic Feasibility**</td><td>[2505.06743](http://arxiv.org/abs/2505.06743)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-08**</td><td>**DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion**</td><td>[2505.05473](http://arxiv.org/abs/2505.05473)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-20**</td><td>**FastMap: Revisiting Dense and Scalable Structure from Motion**</td><td>[2505.04612](http://arxiv.org/abs/2505.04612)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-15**</td><td>**Estimating the Diameter at Breast Height of Trees in a Forest With a Single 360 Camera**</td><td>[2505.03093](http://arxiv.org/abs/2505.03093)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-03**</td><td>**AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting**</td><td>[2505.01799](http://arxiv.org/abs/2505.01799)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-03**</td><td>**PosePilot: Steering Camera Pose for Generative World Models with Self-supervised Depth**</td><td>[2505.01729](http://arxiv.org/abs/2505.01729)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-01**</td><td>**Are Minimal Radial Distortion Solvers Really Necessary for Relative Pose Estimation?**</td><td>[2505.00866](http://arxiv.org/abs/2505.00866)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
</tbody>
</table>
</div>

<h2 id='visual-localization'>Visual Localization</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-07-01</td><td>LoD-Loc v2: Aerial Visual Localization over Low Level-of-Detail City Models using Explicit Silhouette Alignment</td><td>[2507.00659](http://arxiv.org/pdf/2507.00659)</td><td>◆ 提出LoD-Loc v2方法，首次实现基于低细节层次（LoD1）城市模型的无人机空中视觉定位，突破以往依赖高细节模型（LoD2/LoD3）的限制。  
◆ 采用粗到精的双阶段策略：通过显式轮廓对齐构建姿态代价体积筛选粗姿态，再结合粒子滤波与多光束跟踪进行精细优化。  
◆ 创新性设计姿态代价体积，通过均匀采样姿态假设并量化投影轮廓与预测轮廓的对齐度，实现高效概率分布建模。  
◆ 提出多光束跟踪的粒子滤波方法，显著扩大收敛域容错范围，可适应更大初始姿态误差。  
◆ 发布首个覆盖10.7平方公里的LoD1城市模型数据集，包含真实RGB查询图像与姿态真值，推动该领域研究。  
实验表明该方法在高/低LoD模型下均实现最优精度，甚至超越基于纹理模型的方法，为全球城市定位提供新范式。</td></tr>
<tr><td>2025-06-28</td><td>Utilizing a Novel Deep Learning Method for Scene Categorization in Remote Sensing Data</td><td>[2506.22939](http://arxiv.org/pdf/2506.22939)</td><td>这篇论文的核心贡献和创新点如下：  

◆ 提出了一种名为“Cuttlefish Optimized Bidirectional Recurrent Neural Network (CO-BRNN)”的新型深度学习方法，用于遥感数据的场景分类。  
◆ 通过结合双向循环神经网络和优化算法（Cuttlefish优化），显著提升了模型在复杂遥感数据中的特征提取能力。  
◆ 在实验中，CO-BRNN的准确率达到97%，优于现有的多种方法（如MLP-CNN、CNN-LSTM、LSTM-CRF等），展现了其优越性能。  
◆ 解决了传统深度学习方法对大规模、高噪声数据的依赖问题，提高了模型在有限数据条件下的鲁棒性。  
◆ 强调了物理验证在卫星数据应用中的重要性，确保模型结果的可靠性和实用性。  
◆ 为遥感场景分类提供了新的技术思路，可应用于灾害控制、生态监测、城市规划等多个领域。</td></tr>
<tr><td>2025-06-28</td><td>Mask-aware Text-to-Image Retrieval: Referring Expression Segmentation Meets Cross-modal Retrieval</td><td>[2506.22864](http://arxiv.org/pdf/2506.22864)</td><td>◆ 提出Mask-aware TIR（MaTIR）新任务，首次将文本到图像检索（TIR）与指代表达分割（RES）统一，要求同时实现高效图像搜索和精确目标分割。  
◆ 设计两阶段框架：第一阶段利用SAM 2和Alpha-CLIP离线生成对象掩码和区域级嵌入，实现可扩展的分割感知检索；第二阶段通过多模态大语言模型（MLLM）重新排序并生成目标框，与掩码匹配提升精度。  
◆ 创新性结合分割模型（SAM 2）与跨模态检索技术（Alpha-CLIP），在离线阶段预计算掩码和嵌入，显著降低在线检索计算成本。  
◆ 引入MLLM进行结果重排和定位优化，利用其多模态理解能力提升检索准确率与分割质量。  
◆ 在COCO和D$^3$数据集上验证，检索精度和分割效果均显著优于现有方法，为跨模态任务提供新范式。</td></tr>
<tr><td>2025-06-27</td><td>MatChA: Cross-Algorithm Matching with Feature Augmentation</td><td>[2506.22336](http://arxiv.org/pdf/2506.22336)</td><td>◆ 提出了首个解决跨特征检测器视觉定位问题的方法MatChA，突破了现有方法必须使用相同检测器的限制。  
◆ 创新性地通过特征描述符增强技术提升跨检测器特征匹配性能，解决了关键点重复率低和描述符区分度不足的难题。  
◆ 设计了将特征转换到潜在空间的方案，有效实现了不同算法生成描述符的兼容匹配。  
◆ 在多个基准测试中验证了该方法显著提升了跨特征场景下的图像匹配和视觉定位精度。  
◆ 突破了传统方案依赖共同关键点的假设，更贴合实际应用中不同设备使用不同特征提取算法的复杂场景。</td></tr>
<tr><td>2025-06-26</td><td>OracleFusion: Assisting the Decipherment of Oracle Bone Script with Structurally Constrained Semantic Typography</td><td>[2506.21101](http://arxiv.org/pdf/2506.21101)</td><td>◆ 提出OracleFusion两阶段框架，首次将语义排版技术应用于甲骨文破译，通过结构约束生成语义增强的矢量字体。  
◆ 第一阶段采用多模态大语言模型（MLLM）结合空间感知推理（SAR），实现对甲骨文字形的结构分析与关键部件视觉定位。  
◆ 第二阶段创新性引入甲骨文结构向量融合（OSVF）技术，通过字形结构约束和字形保持约束，确保生成结果的结构完整性与语义准确性。  
◆ 在视觉呈现上突破传统方法，生成兼具美学质量与可读性的字形表达，为专家破译提供直观辅助。  
◆ 实验证明OracleFusion在语义相关性、视觉吸引力和字形保持方面均优于现有基线模型，显著提升破译效率。  
◆ 框架可对未解读甲骨文字符提供专家级见解，成为推动甲骨文研究的实用工具。</td></tr>
<tr><td>2025-06-25</td><td>Visualizing intercalation effects in 2D materials using AFM based techniques</td><td>[2506.20467](http://arxiv.org/pdf/2506.20467)</td><td>◆ 提出了一种基于原子力显微镜（AFM）的非侵入性方法，用于可视化二维材料（如MoS2/石墨烯/Ir(111)）中硫插层引起的局部结构和电子性质变化，避免了传统超高真空技术的耗时、高成本和空间限制问题。  
◆ 通过AFM形貌成像直接观察到插层导致的结构变化，并结合相位成像与力学测量，首次发现插层区域杨氏模量和粘附力的降低。  
◆ 利用开尔文探针力显微镜（KPFM）揭示了插层区域的表面电势和功函数变化，为插层效应提供了明确的电子学特征证据。  
◆ 创新性地采用光诱导力显微镜（PiFM）检测插层区域的光学响应增强，拓展了AFM技术在光学性质表征中的应用。  
◆ 综合多种AFM技术实现了插层效应的多维度映射（结构、力学、电子、光学），为二维材料性能调控提供了新工具和理论依据。  
◆ 证明了AFM技术在二维材料插层研究中的高效性和普适性，为未来材料设计和器件开发提供了低成本、高分辨率的表征方案。</td></tr>
<tr><td>2025-06-25</td><td>On the Burstiness of Faces in Set</td><td>[2506.20312](http://arxiv.org/pdf/2506.20312)</td><td>◆ 首次揭示了集合人脸识别(SFR)中普遍存在的&quot;突发性&quot;现象，即特定属性人脸在集合中高频出现，导致模型泛化能力下降和评估干扰。  
◆ 提出三种突发性人脸检测策略：基于Quickshift++的聚类方法、特征自相似性分析和广义最大池化(GMP)技术，有效识别集合中的高频人脸。  
◆ 在训练阶段通过调整采样比例抑制突发性影响，在评估阶段增强低频人脸的贡献度，显著提升模型在无约束场景下的表现。  
◆ 创新性提出质量感知GMP方法，使模型能够感知人脸质量并对低质量图像保持鲁棒性，解决了原始GMP的局限性。  
◆ 通过大量实验验证了突发性现象的广泛存在，证明抑制突发性能显著提升现有SFR基准测试的识别性能，为集合人脸识别提供了新思路。</td></tr>
<tr><td>2025-06-24</td><td>jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval</td><td>[2506.18902](http://arxiv.org/pdf/2506.18902)</td><td>◆ 提出jina-embeddings-v4模型，这是一个38亿参数的多模态嵌入模型，统一了文本和图像的表示。  
◆ 采用新颖的架构，支持单向量和多向量嵌入，并采用后期交互风格。  
◆ 引入任务特定的低秩适应（LoRA）适配器，优化了多种检索场景的性能，包括基于查询的信息检索、跨模态语义相似性和编程代码搜索。  
◆ 在单模态和跨模态检索任务中实现了最先进的性能，尤其在处理视觉丰富内容（如表格、图表、图表和混合媒体格式）方面表现突出。  
◆ 提出Jina-VDR基准，专门用于评估视觉丰富图像检索能力，填补了该领域的空白。</td></tr>
<tr><td>2025-06-26</td><td>Referring Expression Instance Retrieval and A Strong End-to-End Baseline</td><td>[2506.18246](http://arxiv.org/pdf/2506.18246)</td><td>◆ 提出新任务REIR（Referring Expression Instance Retrieval），填补了传统文本-图像检索（TIR）精度不足和指代表达理解（REC）扩展性差的空白，支持跨大规模图库的实例级检索与定位。  
◆ 构建首个大规模基准数据集REIRCOCO，通过视觉-语言模型生成细粒度指代表达，基于MSCOCO和RefCOCO实例增强数据多样性。  
◆ 提出端到端基线方法CLARE，采用双流架构设计，结合目标检测与REC预训练，实现跨模态特征对齐。  
◆ 创新性引入Mix of Relation Experts（MORE）模块，显式建模实例间关系，提升复杂场景下的检索精度。  
◆ 通过对比学习框架CLIA（Contrastive Language-Instance Alignment）优化语言-实例对齐，使模型在REIR、TIR和REC任务上均达到SOTA性能。  
◆ 验证了CLARE的强泛化能力，首次实现单一模型同时支持实例检索、粗粒度检索和细粒度定位三类任务。</td></tr>
<tr><td>2025-06-20</td><td>Class Agnostic Instance-level Descriptor for Visual Instance Search</td><td>[2506.16745](http://arxiv.org/pdf/2506.16745)</td><td>◆提出了一种基于自监督ViT的类无关实例级描述符，解决了视觉实例搜索中缺乏有效实例级特征表示的问题。  
◆通过层次化分解特征集，将实例区域发现建模为检测紧凑特征子集的过程，生成多层次的语义特征子集。  
◆构建的特征层次结构中，非叶节点和叶节点对应图像中不同语义尺度的实例区域，有效处理了物体嵌入和遮挡问题。  
◆生成的节点特征构成图像的全面实例表示，适用于已知和未知物体类别，具有强泛化能力。  
◆在三个实例搜索基准测试中表现显著优于现有方法，验证了其优越性。</td></tr>
<tr><td>2025-06-19</td><td>MambaHash: Visual State Space Deep Hashing Model for Large-Scale Image Retrieval</td><td>[2506.16353](http://arxiv.org/pdf/2506.16353)<br><a href=''>[代码]</a></td><td>◆ 首次将视觉状态空间模型（Mamba）引入大规模图像哈希检索任务，探索其在该领域的适用性和优势。  
◆ 提出分阶段的主干网络架构，通过分组Mamba操作实现多方向扫描，有效建模局部和全局信息。  
◆ 设计通道交互注意力模块，增强跨通道信息交流，提升特征表达能力。  
◆ 开发自适应特征增强模块，增加特征多样性并强化模型的视觉表示能力。  
◆ 在CIFAR-10、NUS-WIDE和IMAGENET等主流数据集上验证了方法的优越性，相比现有深度哈希方法具有更高效率和检索性能。  
◆ 开源代码促进后续研究，为线性复杂度模型在图像检索中的应用提供新思路。</td></tr>
<tr><td>2025-06-19</td><td>Fine-grained Image Retrieval via Dual-Vision Adaptation</td><td>[2506.16273](http://arxiv.org/pdf/2506.16273)</td><td>◆提出双视觉适应（DVA）方法，通过样本和特征协同适配解决细粒度图像检索（FGIR）中预训练模型易过拟合的问题，保留预训练知识的同时提升泛化能力。  
◆设计对象感知适配（Object-Perceptual Adaptation），通过修改输入样本引导冻结的预训练模型聚焦对类别预测关键的物体及局部特征。  
◆提出上下文内适配（In-Context Adaptation），仅引入少量可调参数进行特征适配，使调整后的特征更贴近预训练任务，避免修改原始预训练参数。  
◆结合知识蒸馏机制提出判别感知迁移（Discrimination Perception Transfer），将对象感知适配中的判别知识高效迁移至图像编码器，平衡检索效率与性能。  
◆实验表明DVA在3个分布内和3个分布外细粒度数据集上表现优异，且可学习参数量显著少于现有方法。</td></tr>
<tr><td>2025-06-19</td><td>Adversarial Attacks and Detection in Visual Place Recognition for Safer Robot Navigation</td><td>[2506.15988](http://arxiv.org/pdf/2506.15988)<br><a href=''>[代码]</a></td><td>◆ 首次系统分析了四种常见对抗攻击和四种VPR专用攻击对视觉地点识别（VPR）定位性能的影响，揭示了现有系统的脆弱性。  
◆ 提出了一种闭环系统框架，将VPR、对抗攻击检测器（AAD）和主动导航决策相结合，并通过实验验证其性能优势。  
◆ 设计了新颖的实验范式，证明即使AAD的检测准确率有限（如真阳性率75%、假阳性率25%），也能显著降低平均沿轨定位误差约50%。  
◆ 首次研究了快速梯度符号法（FGSM）对抗攻击在VPR中的有效性，填补了该领域的研究空白。  
◆ 提出了多项关键评估指标（如沿轨误差、受攻击时间比例、不安全状态时间比例等），为系统设计提供了量化依据。  
◆ 强调了AAD在实际机器人导航系统中的必要性，为构建可信赖的导航系统提供了重要参考。</td></tr>
<tr><td>2025-06-18</td><td>Semantic and Feature Guided Uncertainty Quantification of Visual Localization for Autonomous Vehicles</td><td>[2506.15851](http://arxiv.org/pdf/2506.15851)</td><td>◆ 提出了一种结合图像特征和语义信息的轻量级传感器误差模型，用于预测视觉定位中的二维误差分布。  
◆ 通过条件化不确定性估计，隐含地捕捉了未标注的关键环境因素（如城市/高速、动态/静态场景、季节变化）。  
◆ 采用高斯混合模型（GMM）替代传统高斯分布，更准确地描述恶劣天气和光照条件下的测量误差特性。  
◆ 在Ithaca365多天气/光照数据集上验证了框架的准确性，涵盖晴天、夜间和雪天等复杂场景。  
◆ 提出独特的传感器门控方法，结合贝叶斯定位滤波器评估传感器与神经网络的联合不确定性量化性能。  
◆ 为自动驾驶安全关键系统提供了可解释的上下文相关不确定性量化工具。</td></tr>
<tr><td>2025-06-18</td><td>ReSeDis: A Dataset for Referring-based Object Search across Large-Scale Image Collections</td><td>[2506.15180](http://arxiv.org/pdf/2506.15180)</td><td>◆ 提出ReSeDis任务，首次将大规模图像检索与像素级定位结合，要求模型根据文本描述在图像库中检索目标并精确定位其位置（边界框或分割掩码）。  
◆ 构建首个针对该任务的基准数据集，确保每个描述唯一对应分散在大规模多样化图像库中的目标实例，避免误匹配问题。  
◆ 设计联合评估指标，同时衡量检索召回率与定位精度，解决现有技术只能单独评估某一方面的局限。  
◆ 提供基于冻结视觉语言模型的零样本基线方法，揭示该任务未来研究的巨大提升空间。  
◆ 为构建下一代鲁棒、可扩展的多模态搜索系统提供真实端到端测试平台，弥补现有技术（视觉定位假设目标必然存在，文本检索缺乏细粒度定位）的不足。</td></tr>
<tr><td>2025-06-17</td><td>HARMONY: A Scalable Distributed Vector Database for High-Throughput Approximate Nearest Neighbor Search</td><td>[2506.14707](http://arxiv.org/pdf/2506.14707)</td><td>◆ 提出Harmony分布式向量数据库，解决单机处理高维向量时的内存和效率瓶颈。  
◆ 创新性地采用多粒度分区策略，结合基于维度和基于向量的分区方法，实现计算负载均衡。  
◆ 通过优化分区策略有效降低节点间通信开销，提升系统整体吞吐量。  
◆ 引入基于距离计算单调性的早期停止剪枝机制，大幅减少计算和通信开销。  
◆ 在真实数据集上的实验表明，Harmony在四节点配置下平均吞吐量达到现有方案的4.63倍。  
◆ 针对倾斜工作负载，性能比传统分布式方案提升58%，展现出优异的负载适应能力。</td></tr>
<tr><td>2025-06-17</td><td>TACS-Graphs: Traversability-Aware Consistent Scene Graphs for Ground Robot Indoor Localization and Mapping</td><td>[2506.14178](http://arxiv.org/pdf/2506.14178)</td><td>◆ 提出TACS-Graphs框架，首次将地面机器人可通行性（traversability）与房间分割相结合，解决传统3D场景图中房间层分割不一致问题。  
◆ 通过可通行性约束重新定义房间边界，克服体素方法仅依赖几何邻近性导致的欠分割（开放空间误判）和过分割（复杂环境碎片化）缺陷。  
◆ 构建拓扑与语义更一致的场景图，在结构复杂室内环境中实现更准确的房间层语义分割。  
◆ 开发基于一致性场景图的闭环检测方法（CoSG-LCD），利用增强的分割一致性提升闭环检测效率，进而提高位姿估计精度。  
◆ 实验验证该方法在场景图一致性和位姿图优化性能上优于现有先进技术，为机器人定位与建图提供更可靠的环境表征。</td></tr>
<tr><td>2025-06-16</td><td>A Semantically-Aware Relevance Measure for Content-Based Medical Image Retrieval Evaluation</td><td>[2506.13509](http://arxiv.org/pdf/2506.13509)</td><td>◆ 提出了一种基于知识图谱的语义感知相关性度量方法，用于解决医学图像检索（CBIR）的性能评估难题。  
◆ 创新性地利用医学文本（如放射学报告或文献描述）中隐含的医学概念，避免了传统评估方法对人工标注数据的依赖。  
◆ 通过知识图谱量化医学概念间的语义距离，克服了现有方法将医学概念视为独立标签的局限性，能够捕捉概念间的细微关联。  
◆ 设计了基于近似匹配的相关性评分机制，通过计算两组医学概念的相似性间接衡量医学图像的相似度。  
◆ 在公开数据集上验证了所提方法的有效性和可行性，为医学CBIR评估提供了更符合临床语义的新标准。</td></tr>
<tr><td>2025-06-19</td><td>Hierarchical Multi-Positive Contrastive Learning for Patent Image Retrieval</td><td>[2506.13496](http://arxiv.org/pdf/2506.13496)</td><td>◆提出分层多正例对比学习损失函数，首次利用Locarno国际分类体系（LIC）的层级关系指导专利图像检索。  
◆通过层级分类树动态分配多组正样本对，根据专利图像在LIC中的层级距离赋予不同相似度权重。  
◆突破传统对比学习仅使用单一样本对的限制，能同时学习跨层级的细粒度语义关联。  
◆在DeepPatent2数据集上验证了方法的普适性，可适配多种视觉和多模态预训练模型。  
◆特别优化了小参数量模型的检索性能，在计算资源受限环境下具有显著部署优势。  
◆实验表明该方法能有效捕捉专利图像的技术细节和复杂语义，提升跨类别检索准确率。</td></tr>
<tr><td>2025-06-16</td><td>EmbodiedPlace: Learning Mixture-of-Features with Embodied Constraints for Visual Place Recognition</td><td>[2506.13133](http://arxiv.org/pdf/2506.13133)</td><td>◆ 提出了一种新颖的简单重排序方法，通过混合特征（MoF）方法在具身约束下优化全局特征，提升视觉地点识别（VPR）性能。  
◆ 首次系统分析了具身约束在VPR中的实际可行性，并根据现有数据集将其分类为GPS标签、时序戳、局部特征匹配和自相似矩阵等类型。  
◆ 设计了一种基于学习的MoF权重计算策略，采用多度量损失函数，有效融合多种特征信息。  
◆ 在公开数据集上实现了性能提升，仅需25 KB额外参数和每帧10微秒处理时间，显著优于现有方法。  
◆ 在Pitts-30k测试集上，基于DINOv2的基线性能提升0.9%，计算开销极低，适合实际机器人应用。</td></tr>
<tr><td>2025-06-16</td><td>SuperPlace: The Renaissance of Classical Feature Aggregation for Visual Place Recognition in the Era of Foundation Models</td><td>[2506.13073](http://arxiv.org/pdf/2506.13073)</td><td>◆ 提出SuperPlace框架，重新利用经典特征聚合方法（如GeM和NetVLAD），在基础模型时代优化视觉地点识别（VPR）性能。  
◆ 开发监督标签对齐方法，实现跨多个VPR数据集的统一训练框架，提升模型泛化能力。  
◆ 提出G²M特征聚合方法，通过双GeM结构学习特征图的主成分并校准输出，仅需十分之一特征维度即可达到优异效果。  
◆ 设计NetVLAD-Linear（NVL）的二次微调策略（FT²），先在高维空间学习特征向量，再通过单线性层压缩，显著提升性能。  
◆ 实验证明SuperPlace的优越性，G²M在低维度下表现突出，NVL-FT²在MSLS排行榜上排名第一。</td></tr>
<tr><td>2025-06-14</td><td>Feature Complementation Architecture for Visual Place Recognition</td><td>[2506.12401](http://arxiv.org/pdf/2506.12401)</td><td>◆ 提出局部-全局特征互补网络（LGCN），通过并行CNN-ViT混合架构解决视觉地点识别（VPR）中局部细节与全局上下文难以兼顾的问题。  
◆ 设计动态特征融合模块（DFM），通过联合建模空间和通道依赖关系实现自适应特征融合，提升特征表达的鲁棒性。  
◆ 在冻结的ViT主干中引入轻量级频域-空间融合适配器，以可控参数量实现任务特定适配，增强ViT分支对VPR任务的适应能力。  
◆ 实验证明LGCN在多个VPR基准数据集上均优于现有方法，定位精度和鲁棒性显著提升。  
◆ 整体架构兼顾计算效率与性能，为复杂环境下的机器人定位提供了新思路。</td></tr>
<tr><td>2025-06-11</td><td>Towards a general-purpose foundation model for fMRI analysis</td><td>[2506.11167](http://arxiv.org/pdf/2506.11167)</td><td>◆ 提出NeuroSTORM，首个面向fMRI分析的通用基础模型，直接从4D fMRI数据学习，解决传统方法因复杂预处理和任务专用模型导致的复现性和迁移性不足问题。  
◆ 采用Mamba架构和移位扫描策略，高效处理完整4D fMRI体积，突破传统时空建模效率瓶颈。  
◆ 设计空间-时间联合优化的预训练方法，结合任务特定提示微调（prompt tuning），显著提升跨任务迁移能力。  
◆ 基于超大规模数据集预训练（28.65百万帧fMRI，50,000+受试者，跨多中心及5-100岁年龄范围），建立迄今最全面的脑功能表征库。  
◆ 在五项任务（年龄/性别预测、表型预测、疾病诊断、fMRI-图像检索、任务态分类）中全面超越现有方法，并在美、韩、澳临床数据验证中展现卓越诊断性能。  
◆ 开源标准化模型框架，为fMRI临床研究提供可复现、可迁移的基础工具，推动脑疾病诊断的跨中心应用。</td></tr>
<tr><td>2025-06-11</td><td>Improving Personalized Search with Regularized Low-Rank Parameter Updates</td><td>[2506.10182](http://arxiv.org/pdf/2506.10182)<br><a href=''>[代码]</a></td><td>◆ 提出一种正则化低秩参数更新方法，仅需微调语言编码器最后一层的少量参数，即可有效适应个性化视觉-语言检索任务，避免传统文本反转方法的不足。  
◆ 发现参数相加策略能有效整合多个已学习个性化概念的参数，提升模型对多概念组合的识别能力。  
◆ 引入基于视觉语言模型生成描述的图像检索评估指标，量化微调后模型对通用知识的保留程度。  
◆ 在DeepFashion2和ConCon-Chi两个基准测试中实现最先进性能，个性化检索准确率较之前方法提升4%-22%。  
◆ 通过双编码器模型内部表征的针对性适配，解决了少样本场景下个性化概念与通用知识融合的难题。  
◆ 实验证明低秩参数更新在保留通用知识的同时，显著提升对&quot;我的狗Fido&quot;等个性化概念的跨上下文识别能力。</td></tr>
<tr><td>2025-06-10</td><td>Safeguarding Multimodal Knowledge Copyright in the RAG-as-a-Service Environment</td><td>[2506.10030](http://arxiv.org/pdf/2506.10030)</td><td>◆ 提出了首个针对多模态RAG系统中图像知识版权保护的水印框架AQUA，填补了该领域的空白。  
◆ 设计了两种互补的水印嵌入方法：基于首字母缩写的触发器和空间关系线索，确保水印信号在图像到文本的间接传播中保持有效。  
◆ 实现了水印的高效性、强鲁棒性和不可感知性，能够在不同模型和数据集上稳定工作。  
◆ 解决了现有RAG水印技术仅关注文本知识而忽略图像保护的局限性，扩展了版权保护范围。  
◆ 通过实验验证了AQUA在跨模态场景下的可靠性，支持对贡献数据的精准版权追踪。  
◆ 为RAG-as-a-Service环境中的多模态知识共享提供了实用的版权安全保障方案。</td></tr>
<tr><td>2025-06-11</td><td>Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints</td><td>[2506.09748](http://arxiv.org/pdf/2506.09748)</td><td>◆ 提出了一种分层跨源图像匹配方法，结合语义感知和结构约束的粗匹配模块与轻量级细粒度匹配模块，显著提升了无人机绝对视觉定位的精度。  
◆ 在粗匹配模块中，利用视觉基础模型提取的语义特征，在语义和结构约束下建立区域级对应关系，有效克服了跨源差异和时变因素带来的挑战。  
◆ 设计了轻量级细粒度匹配模块，通过提取精细特征建立像素级对应关系，进一步提升了定位的准确性。  
◆ 构建了不依赖相对定位技术的无人机绝对视觉定位流程，通过图像检索模块与分层匹配模块的结合，实现了完全基于视觉的全局定位。  
◆ 在公开基准数据集和新提出的CS-UAV数据集上验证了方法的优越性，展示了其在多种挑战性条件下的高精度和鲁棒性。</td></tr>
<tr><td>2025-06-10</td><td>Robust Visual Localization via Semantic-Guided Multi-Scale Transformer</td><td>[2506.08526](http://arxiv.org/pdf/2506.08526)</td><td>◆ 提出了一种结合多尺度特征学习与语义场景理解的视觉定位框架，通过层次化Transformer和跨尺度注意力机制融合几何细节与上下文信息，在保持空间精度的同时适应环境变化。  
◆ 创新性地引入神经场景表征提供的语义监督信号，指导网络学习视角不变特征，有效编码持久结构信息并抑制动态环境干扰。  
◆ 设计了多尺度Transformer架构，利用跨层级注意力机制整合不同尺度的视觉线索，显著提升了复杂场景下的定位鲁棒性。  
◆ 在TartanAir数据集上的实验表明，该方法在动态物体、光照变化和遮挡等挑战性场景中优于现有位姿回归方法。  
◆ 首次验证了语义引导与多尺度处理的协同策略对现实动态环境中视觉定位的有效性，为鲁棒定位提供了新思路。</td></tr>
<tr><td>2025-06-08</td><td>Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs</td><td>[2506.07045](http://arxiv.org/pdf/2506.07045)</td><td>◆ 提出了一种基于多模态大语言模型（MLLMs）的可解释AI生成图像检测方法，通过结合视觉定位和文本推理能力，不仅检测准确率高，还能提供人类可理解的解释。  
◆ 构建了一个包含边界框标注和描述性文本的数据集，突出AI生成图像的合成伪影，为模型提供视觉-文本对齐的推理基础。  
◆ 设计了多阶段优化策略，逐步平衡检测准确性、视觉定位能力和文本解释连贯性，解决了现有MLLMs在检测任务中的幻觉问题。  
◆ 通过微调MLLMs，使其能够同时定位图像中的视觉缺陷并生成合理的解释，显著提升了检测的可靠性和可解释性。  
◆ 实验表明，该方法在检测AI生成图像和定位视觉瑕疵方面均优于基线方法，为可解释的伪造检测提供了新思路。</td></tr>
<tr><td>2025-06-07</td><td>Zero Shot Composed Image Retrieval</td><td>[2506.06602](http://arxiv.org/pdf/2506.06602)</td><td>◆ 提出了一种基于BLIP-2的轻量级Q-Former模型，通过融合视觉和文本特征到单一嵌入，显著提升了零样本组合图像检索（Zero-shot CIR）的性能。  
◆ 在FashionIQ基准测试中，将Recall@10指标从原先的20-25%大幅提升至45.6%（衬衫）、40.1%（裙子）和50.4%（T恤），平均Recall@50达到67.6%。  
◆ 探索了Retrieval-DPO方法，尝试通过直接偏好优化（DPO）损失微调CLIP文本编码器，但发现其效果远低于基线（仅0.02% Recall@10）。  
◆ 分析了Retrieval-DPO失败的四大原因：缺乏图像-文本联合融合、目标函数与Top-K指标不匹配、负样本质量低，以及视觉和Transformer层冻结。  
◆ 研究表明，有效的基于偏好的CIR需要真正的多模态融合、与排名相关的目标函数，以及精心筛选的负样本。</td></tr>
<tr><td>2025-06-06</td><td>GenIR: Generative Visual Feedback for Mental Image Retrieval</td><td>[2506.06220](http://arxiv.org/pdf/2506.06220)</td><td>◆ 提出Mental Image Retrieval (MIR)任务，研究用户通过多轮交互从模糊心理图像中检索目标图像的真实场景，填补了现有文本-图像检索研究的空白。  
◆ 设计GenIR方法，首次利用扩散模型生成可视化反馈，将AI系统对用户意图的理解转化为直观的合成图像，克服传统抽象语言反馈的模糊性问题。  
◆ 构建自动化流水线生成高质量多轮MIR数据集，为后续研究提供基准支持。  
◆ 实验证明GenIR在多轮交互检索中显著优于现有方法，验证了生成式视觉反馈的有效性。  
◆ 开创性地将生成模型与交互式检索结合，为心理图像检索领域奠定新范式，推动人机协同搜索系统的发展。</td></tr>
<tr><td>2025-06-06</td><td>Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning</td><td>[2506.06205](http://arxiv.org/pdf/2506.06205)</td><td>◆提出Astra双模型架构（Astra-Global和Astra-Local），通过分层多模态学习实现通用移动机器人导航，突破传统模块化系统的局限性。  
◆Astra-Global首次将多模态大语言模型（LLM）与混合拓扑-语义图结合，显著提升视觉地点识别和全局定位能力，优于传统方法。  
◆Astra-Local采用自监督训练的4D时空编码器生成鲁棒特征，支持局部路径规划和里程计估计等多任务学习。  
◆创新性提出基于流匹配和掩码ESDF损失的规划头，有效降低碰撞风险，生成更安全的局部轨迹。  
◆里程计头通过Transformer编码器融合多传感器数据，实现高精度相对位姿预测。  
◆在真实室内场景的移动机器人上验证，端到端任务成功率显著提升，展现强泛化能力。</td></tr>
<tr><td>2025-06-05</td><td>HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition</td><td>[2506.04764](http://arxiv.org/pdf/2506.04764)</td><td>◆ 提出HypeVPR框架，首次将双曲空间嵌入引入透视到环视（P2E）视觉位置识别任务，利用双曲空间更适合表示层次结构的特性。  
◆ 设计分层特征聚合机制，在双曲空间中组织局部到全局的特征表示，有效捕捉全景图像的固有层次关系。  
◆ 开发高效的粗到精搜索策略，显著提升匹配速度（最高达5倍）同时保持高精度，解决跨图像类型的鲁棒匹配问题。  
◆ 通过双曲空间的距离保持特性，优化特征空间中的距离度量，增强不同视角下描述符的区分能力。  
◆ 在多个基准数据集上验证了方法的优越性，性能超越现有最优方法，同时大幅降低检索时间。  
◆ 开源代码和模型，推动相关领域研究。</td></tr>
<tr><td>2025-06-05</td><td>Deep Learning Reforms Image Matching: A Survey and Outlook</td><td>[2506.04619](http://arxiv.org/pdf/2506.04619)</td><td>这篇论文系统综述了深度学习如何逐步革新传统图像匹配流程，并提出了分类框架。  
◆创新点一：首次从&quot;逐步替代传统模块&quot;和&quot;端到端合并多步骤&quot;两个维度，对深度学习方法进行系统分类（包括可学习检测-...</td></tr>
<tr><td>2025-06-02</td><td>Entity Image and Mixed-Modal Image Retrieval Datas...</td><td>[2506.02291](http://arxiv.org/pdf/2506.02291)</td><td>◆提出首个结合视觉与文本信息的混合模态图像检索基准MMIR，包含单实体图像和多实体图像两种复杂查询类型。  
◆发布Entity Image和MMIR两个高质量数据集，通过众包标注验证数据质量，...</td></tr>
<tr><td>2025-06-01</td><td>Quantization-based Bounds on the Wasserstein Metri...</td><td>[2506.00976](http://arxiv.org/pdf/2506.00976)</td><td>◆提出了一种基于量化网格的高效Wasserstein距离近似方法，通过粗网格上的Kantorovich问题精确求解结合升尺度校正步骤，在保持2%误差内实现10-100倍加速。◆创新性地在原始空间...</td></tr>
<tr><td>2025-05-30</td><td>SORCE: Small Object Retrieval in Com...</td><td>[2505.24441](http://arxiv.org/pdf/2505.24441)<br><a href=''>[代码]</a></td><td>◆提出新任务SORCE（复杂环境中的小物体检索），专注于通过文本查询检索复杂图像中的不显眼小物体。  
◆构建新基准SORCE-1K，包含复杂环境图像和描述小物体的文本查询，揭示现有T2IR方法...</td></tr>
<tr><td>**2025-05-29**</td><td>**Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch**</td><td>[2505.23763](http://arxiv.org/abs/2505.23763)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**4DTAM: Non-Rigid Tracking and Mapping via Dynamic Surface Gaussians**</td><td>[2505.22859](http://arxiv.org/abs/2505.22859)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**UAVPairs: A Challenging Benchmark for Match Pair Retrieval of Large-scale UAV Images**</td><td>[2505.22098](http://arxiv.org/abs/2505.22098)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**Fast Feature Matching of UAV Images via Matrix Band Reduction-based GPU Data Schedule**</td><td>[2505.22089](http://arxiv.org/abs/2505.22089)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-27**</td><td>**Visual Loop Closure Detection Through Deep Graph Consensus**</td><td>[2505.21754](http://arxiv.org/abs/2505.21754)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-27**</td><td>**QuARI: Query Adaptive Retrieval Improvement**</td><td>[2505.21647](http://arxiv.org/abs/2505.21647)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-27**</td><td>**ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval**</td><td>[2505.20764](http://arxiv.org/abs/2505.20764)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**Visualized Text-to-Image Retrieval**</td><td>[2505.20291](http://arxiv.org/abs/2505.20291)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**Multimodal Reasoning Agent for Zero-Shot Composed Image Retrieval**</td><td>[2505.19952](http://arxiv.org/abs/2505.19952)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**Can Visual Encoder Learn to See Arrows?**</td><td>[2505.19944](http://arxiv.org/abs/2505.19944)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-22**</td><td>**TAT-VPR: Ternary Adaptive Transformer for Dynamic and Efficient Visual Place Recognition**</td><td>[2505.16447](http://arxiv.org/abs/2505.16447)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-21**</td><td>**Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval**</td><td>[2505.15877](http://arxiv.org/abs/2505.15877)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-21**</td><td>**SCENIR: Visual Semantic Clarity through Unsupervised Scene Graph Retrieval**</td><td>[2505.15867](http://arxiv.org/abs/2505.15867)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-20**</td><td>**Multimodal RAG-driven Anomaly Detection and Classification in Laser Powder Bed Fusion using Large Language Models**</td><td>[2505.13828](http://arxiv.org/abs/2505.13828)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-18**</td><td>**MMS-VPR: Multimodal Street-Level Visual Place Recognition Dataset and Benchmark**</td><td>[2505.12254](http://arxiv.org/abs/2505.12254)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-16**</td><td>**Improved Bag-of-Words Image Retrieval with Geometric Constraints for Ground Texture Localization**</td><td>[2505.11620](http://arxiv.org/abs/2505.11620)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-16**</td><td>**Redundancy-Aware Pretraining of Vision-Language Foundation Models in Remote Sensing**</td><td>[2505.11121](http://arxiv.org/abs/2505.11121)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-04**</td><td>**OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery**</td><td>[2505.03836](http://arxiv.org/abs/2505.03836)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-06**</td><td>**Thermal-LiDAR Fusion for Robust Tunnel Localization in GNSS-Denied and Low-Visibility Conditions**</td><td>[2505.03565](http://arxiv.org/abs/2505.03565)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-06**</td><td>**LiftFeat: 3D Geometry-Aware Local Feature Matching**</td><td>[2505.03422](http://arxiv.org/abs/2505.03422)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-06**</td><td>**Seeing the Abstract: Translating the Abstract Language for Vision Language Models**</td><td>[2505.03242](http://arxiv.org/abs/2505.03242)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-13**</td><td>**SafeNav: Safe Path Navigation using Landmark Based Localization in a GPS-denied Environment**</td><td>[2505.01956](http://arxiv.org/abs/2505.01956)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-02**</td><td>**NeuroLoc: Encoding Navigation Cells for 6-DOF Camera Localization**</td><td>[2505.01113](http://arxiv.org/abs/2505.01113)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-01**</td><td>**GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting**</td><td>[2504.20379](http://arxiv.org/abs/2504.20379)</td><td>摘要生成中...</td></tr>
</tbody>
</table>
</div>

<h2 id='keypoint-detection'>Keypoint Detection</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-06-27</td><td>MatChA: Cross-Algorithm Matching with Feature Augmentation</td><td>[2506.22336](http://arxiv.org/pdf/2506.22336)</td><td>◆ 提出了首个解决跨特征检测器场景下视觉定位问题的方法MatChA，突破了现有方法必须使用相同检测器的限制。  
◆ 通过特征描述符增强技术提升跨检测器特征匹配性能，解决了关键点重复率低和描述符区分度不足的难题。  
◆ 创新性地将增强后的特征转换到潜在空间，实现了不同算法生成描述符的有效匹配。  
◆ 在多个基准测试中验证了该方法显著提升了跨特征场景下的图像匹配和视觉定位精度。  
◆ 突破了传统方案依赖共同关键点的假设，更贴合实际应用中不同设备使用不同特征提取算法的复杂场景。</td></tr>
<tr><td>2025-06-27</td><td>SDRNET: Stacked Deep Residual Network for Accurate Semantic Segmentation of Fine-Resolution Remotely Sensed Images</td><td>[2506.21945](http://arxiv.org/pdf/2506.21945)</td><td>◆ 提出堆叠式深度残差网络（SDRNet），通过双编码器-解码器结构同时捕获长程语义并保留空间细节，解决高分辨率遥感图像分割中空间信息丢失问题。  
◆ 在编码器与解码器之间引入膨胀残差块（DRB），增强全局依赖关系建模能力，有效应对地物类别差异和遮挡导致的特征提取挑战。  
◆ 通过多上下文特征学习机制，覆盖不同尺寸地物目标，缓解因物体尺寸变化导致的细分不准问题。  
◆ 结合全局与局部上下文信息，显著提升对细小地物和复杂边界的识别精度，克服传统深度网络下采样导致的边界模糊缺陷。  
◆ 在ISPRS Vaihingen和Potsdam数据集上验证了模型优越性，性能超越现有深度卷积网络，为高精度地物分类提供新解决方案。</td></tr>
<tr><td>2025-05-29</td><td>TimePoint: Accelerated Time Series Alignment via Self-Supervised Keypoint and Descriptor Learning</td><td>[2505.23475](http://arxiv.org/pdf/2505.23475)<br><a href=''>[代码]</a></td><td>◆提出TimePoint方法，通过自监督学习从合成数据中提取关键点和描述符，显著加速动态时间规整（DTW）的对齐过程，同时提高对齐精度。  
◆创新性地将2D关键点检测思想适配到1D信号，设计高效的一维微分同胚模型生成逼真训练数据，有效模拟非线性时间扭曲。  
◆采用全卷积和小波卷积架构提取信息丰富的稀疏表示，使DTW在稀疏数据上运行时获得数量级加速，且精度通常优于原始信号上的标准DTW。  
◆仅使用合成数据训练即可在真实时间序列上展现强泛化能力，结合真实数据微调后性能进一步提升。  
◆通过大量实验验证，TimePoint在速度和精度上均优于标准DTW，为大规模时间序列分析提供可扩展解决方案。</td></tr>
<tr><td>2025-05-24</td><td>Why Not Replace? Sustaining Long-Term Visual Localization via Handcrafted-Learned Feature Collaboration on CPU</td><td>[2505.18652](http://arxiv.org/pdf/2505.18652)<br><a href=''>[代码]</a></td><td>◆ 提出手工-学习特征协作机制：首次系统论证手工特征（适合连续跟踪）与学习特征（擅长宽基线匹配）的功能互补性，打破传统&quot;替代&quot;思维，建立协同框架。  
◆ 设计CPU友好的分层定位架构：实时层采用手工特征进行相对位姿估计，异步层选择性调用学习特征进行绝对定位，实现仅需CPU的长期稳定运行。  
◆ 创新关键帧优化策略：通过动态筛选机制平衡学习特征的计算开销与定位精度，使系统在光照变化下保持47%的平均误差降低。  
◆ 实现全时段环境适应性：通过特征协作有效应对工业场景中的季节更替、昼夜光照变化等挑战，定位一致性显著提升。  
◆ 提供完整开源实现：公开代码包含特征互补性分析、计算延迟剖析到系统级验证的全套实验数据，推动工业应用落地。  
◆ 建立三阶段验证体系：从特征特性对比、CPU平台算力剖析到真实光照变化测试，形成严谨的技术验证链路。</td></tr>
<tr><td>2025-05-18</td><td>SEPT: Standard-Definition Map Enhanced Scene Perception and Topology Reasoning for Autonomous Driving</td><td>[2505.12246](http://arxiv.org/pdf/2505.12246)</td><td>◆ 提出SEPT框架，利用标准定义地图（SD地图）作为先验知识，增强自动驾驶场景感知与拓扑推理能力，减少对高精地图的依赖。  
◆ 设计混合特征融合策略，结合SD地图与鸟瞰图（BEV）特征，同时处理栅格化和矢量化表示，解决两者空间对齐问题。  
◆ 创新性引入基于SD地图的辅助任务——交叉路口感知关键点检测，提升长距离和遮挡场景下的理解性能。  
◆ 通过实验验证，在OpenLane-V2数据集上显著超越现有方法，证明SD地图先验的有效性。  
◆ 整体框架兼顾感知与推理，为无高精地图自动驾驶系统提供更鲁棒的在线环境理解方案。</td></tr>
<tr><td>2025-05-17</td><td>Keypoints as Dynamic Centroids for Unified Human Pose and Segmentation</td><td>[2505.12130](http://arxiv.org/pdf/2505.12130)</td><td>◆ 提出Keypoints as Dynamic Centroid (KDC)方法，通过动态质心表示统一解决人体姿态估计和实例分割任务，克服传统方法在关节重叠或快速运动时的局限性。  
◆ 采用自底向上范式生成关键点热图，并引入KeyCentroids（基于关键点磁盘）提升关键点检测精度和置信度得分。  
◆ 利用高置信度关键点作为嵌入空间中的动态质心（MaskCentroids），实现快速运动下像素到人体实例的高效聚类。  
◆ 在CrowdPose、OCHuman和COCO等基准测试中验证了KDC的优越性，尤其在复杂场景下的准确性和实时性能表现突出。  
◆ 通过动态质心机制有效处理实例级分割中的遮挡和姿态快速变化问题，增强了模型的泛化能力。</td></tr>
<tr><td>2025-05-16</td><td>Deepfake Forensic Analysis: Source Dataset Attribution and Legal Implications of Synthetic Media Manipulation</td><td>[2505.11110](http://arxiv.org/pdf/2505.11110)</td><td>◆提出了一种新型的GAN生成图像溯源框架，通过可解释特征分析准确识别训练数据集（如CelebA或FFHQ）。  
◆创新性地融合频域变换（傅里叶/DCT）、色彩分布度量和局部特征描述符（SIFT），提取合成图像中的 discriminative 统计特征。  
◆监督分类器（随机森林、SVM、XGBoost）在二元分类（真实vs合成）和多类数据集溯源任务中达到98-99%准确率，覆盖多种主流GAN架构（如StyleGAN系列）。  
◆实验证明频域特征（DCT/FFT）对捕捉数据集特异性伪影（如上采样模式、频谱异常）具有显著优势，色彩直方图则能揭示GAN训练的隐式正则化策略。  
◆首次系统探讨了合成媒体数据集溯源的法律应用场景，包括版权侵权、隐私数据滥用（如GDPR合规）及加州AB 602法案等监管应对方案。  
◆该框架为生成模型的问责制治理提供了技术支撑，可应用于数字取证、内容审核和知识产权诉讼等实际领域。</td></tr>
<tr><td>2025-06-19</td><td>RDD: Robust Feature Detector and Descriptor using Deformable Transformer</td><td>[2505.08013](http://arxiv.org/pdf/2505.08013)</td><td>◆ 提出RDD（Robust Deformable Detector），一种基于可变形Transformer的新型关键点检测与描述方法，通过可变形自注意力机制捕获全局上下文和几何不变性。  
◆ 利用可变形注意力机制聚焦关键位置，显著降低搜索空间复杂度并有效建模几何变换，解决了传统方法难以学习长程视觉关系的问题。  
◆ 结合标准MegaDepth数据集与自建的Air-to-Ground（空对地）数据集进行训练，增强模型在跨视角和跨尺度场景下的鲁棒性。  
◆ 在稀疏匹配任务中性能超越现有最优方法，并具备半稠密匹配能力，扩展了应用场景。  
◆ 引入两个新基准测试：一个针对大视角与尺度变化，另一个为空对地场景，填补了跨高度3D重建评估的空白。</td></tr>
<tr><td>2025-05-12</td><td>Enabling Privacy-Aware AI-Based Ergonomic Analysis</td><td>[2505.07306](http://arxiv.org/pdf/2505.07306)</td><td>◆ 提出了一种隐私感知的AI工效学分析框架，通过对抗训练开发轻量级神经网络，在视频数据中模糊隐私信息，仅保留人体姿态估计所需关键特征。  
◆ 采用数据混淆技术确保与标准姿态估计算法兼容，在保护隐私的同时维持高精度分析能力，解决了传统摄像头系统的隐私泄露问题。  
◆ 创新性地将混淆后的数据传输至中央服务器处理，结合多视角融合技术重建3D关键点，实现远程高精度工效学评估。  
◆ 整合REBA（快速全身评估）方法对3D姿态进行工效学风险量化，形成从数据采集到风险评估的完整闭环系统。  
◆ 在工业场景中首次实现隐私保护与工效学监测的平衡，为制造业提供兼顾安全性与合规性的解决方案。  
◆ 系统设计轻量化且可扩展，适用于资源受限的工业环境，具有实际部署的可行性优势。</td></tr>
<tr><td>2025-05-09</td><td>My Emotion on your face: The use of Facial Keypoint Detection to preserve Emotions in Latent Space Editing</td><td>[2505.06436](http://arxiv.org/pdf/2505.06436)</td><td>◆ 提出了一种结合面部关键点检测模型的新损失函数（HFLD损失），用于解决StyleGAN/2潜在空间编辑中的表情纠缠问题。  
◆ 通过在现有模型损失函数中增加HFLD损失，有效限制了编辑过程中对面部表情的干扰，实验显示情绪变化减少高达49%。  
◆ 首次将面部关键点检测技术与GAN潜在空间编辑结合，定量和定性验证了该方法在保持表情一致性上的优越性。  
◆ 相比现有方法，显著提升了生成图像在固定表情下变换外貌特征的能力，为手势和表情研究提供了可靠的数据增强手段。  
◆ 通过对比实验证明，该方法在保持面部表情的同时编辑其他属性（如性别、年龄）的效果优于当前最先进模型。  
◆ 为面部生成任务提供了一种可解释的技术路径，通过关键点约束直接解决特征解耦问题，而非依赖隐式学习。</td></tr>
<tr><td>2025-05-05</td><td>Unsupervised training of keypoint-agnostic descriptors for flexible retinal image registration</td><td>[2505.02787](http://arxiv.org/pdf/2505.02787)</td><td>◆提出首个不依赖关键点检测的无监督描述符学习方法，突破视网膜图像配准领域对标注数据的依赖。  
◆创新性地实现描述符网络与关键点检测器的解耦，使模型能适配任意检测器，提升临床应用灵活性。  
◆在标准视网膜配准数据集上进行了全面验证，证明无监督方法性能媲美有监督方法。  
◆设计并测试了多种新型关键点检测器，验证了方法对不同检测器的强鲁棒性。  
◆为医学领域无监督学习应用提供了重要范例，解决了医学图像标注稀缺的核心痛点。  
◆通过端到端无监督训练框架，显著降低了视网膜图像配准的技术门槛和实现成本。</td></tr>
<tr><td>2025-05-05</td><td>Unsupervised Deep Learning-based Keypoint Localization Estimating Descriptor Matching Performance</td><td>[2505.02779](http://arxiv.org/pdf/2505.02779)</td><td>◆提出首个完全无监督的视网膜图像配准流程，无需任何标注数据，解决了医学领域标注稀缺的难题。  
◆创新性地颠覆传统思路，通过描述子性能反推关键点检测（描述子驱动检测器），而非传统的关键点驱动描述子学习。  
◆开发了无需关键点检测或标签的描述子学习方法，可直接为视网膜图像任意位置生成高质量描述符。  
◆设计了新型无标签关键点检测网络，能够直接从输入图像预测描述子匹配性能来定位关键点。  
◆在四个独立数据集上验证表明，无监督描述子超越有监督SOTA方法，无监督检测器显著优于现有无监督检测方法。  
◆整个配准流程性能媲美主流有监督方法，且无需标注数据的特性使其可直接迁移到其他领域和模态。</td></tr>
<tr><td>**2025-05-04**</td><td>**Focus What Matters: Matchability-Based Reweighting for Local Feature Matching**</td><td>[2505.02161](http://arxiv.org/abs/2505.02161)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-04**</td><td>**Enhancing Lidar Point Cloud Sampling via Colorization and Super-Resolution of Lidar Imagery**</td><td>[2505.02049](http://arxiv.org/abs/2505.02049)</td><td>摘要生成中...</td></tr>
</tbody>
</table>
</div>

<h2 id='image-matching'>Image Matching</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-07-02</td><td>What does really matter in image goal navigation?</td><td>[2507.01667](http://arxiv.org/pdf/2507.01667)</td><td>◆ 研究了端到端强化学习在图像目标导航任务中的有效性，挑战了传统依赖专用图像匹配或预训练视觉模块的方法。  
◆ 通过大规模实验分析了多种架构设计（如延迟融合、通道堆叠、空间到深度投影和交叉注意力）对导航性能的影响。  
◆ 揭示了仿真环境设置对现有方法性能的影响，指出仿真中存在的捷径问题，同时证明部分能力可迁移到更真实场景。  
◆ 首次发现导航性能与相对位姿估计能力之间存在相关性，表明后者是导航任务中自然涌现的重要子技能。  
◆ 为仅通过导航奖励信号训练相对位姿估计器提供了可能性，对具身AI及其他领域具有潜在影响。  
◆ 通过系统实验验证了端到端训练智能体的潜力，同时指出了仿真与现实场景间的性能差距问题。</td></tr>
<tr><td>2025-06-30</td><td>Efficient and Accurate Image Provenance Analysis: A Scalable Pipeline for Large-scale Images</td><td>[2506.23707](http://arxiv.org/pdf/2506.23707)</td><td>这篇论文的核心贡献是提出了一种高效且准确的图像溯源分析管道，解决了现有方法在精度和可扩展性上的两大瓶颈。  

◆ 创新性地引入修改关系追踪技术，显著提升了图像变体的过滤效果，能够全面发现与查询图像视觉相似度低的变体，解决了传统方法因低相似度而遗漏严重修改图像的问题。  

◆ 通过结合局部特征匹配和压缩伪影捕捉技术，增强了方法对多样化修改的鲁棒性，能够更准确地分析图像间的关联性和修改方向。  

◆ 提出了一种优化的相似度计算策略，并在构建有向溯源图时消除了冗余的成对分析，将时间复杂度从二次降低到线性，实现了大规模场景下的高效处理。  

◆ 实验证明，该方法在精度上比现有技术提升了16.7%-56.1%，并在1000万规模图像上平均响应时间仅3秒，远优于现有方法的12分钟，展现了卓越的可扩展性。  

◆ 最终生成的溯源图能够精确刻画图像的演化历史，为数字治理提供了可靠的取证工具。</td></tr>
<tr><td>2025-06-29</td><td>Dynamic Contrastive Learning for Hierarchical Retrieval: A Case Study of Distance-Aware Cross-View Geo-Localization</td><td>[2506.23077](http://arxiv.org/pdf/2506.23077)</td><td>◆ 提出了Distance-Aware Cross-View Geo-Localization (DACVGL)新问题，强调模型需综合捕捉目标周围上下文信息并降低定位误差成本。  
◆ 构建首个多视角图像与精确距离标注的基准数据集DA-Campus，涵盖三种空间分辨率，支持系统性研究。  
◆ 将DACVGL问题形式化为跨域分层检索任务，揭示传统度量学习无法解决建筑间复杂空间关系的问题。  
◆ 提出动态对比学习框架DyCL，通过分层空间间隔逐步对齐特征表示，解决跨视角层次化检索难题。  
◆ 实验证明DyCL与现有多尺度度量学习方法高度互补，显著提升分层检索性能和跨视角地理定位精度。  
◆ 公开代码和基准数据集，推动后续研究。</td></tr>
<tr><td>2025-06-27</td><td>MatChA: Cross-Algorithm Matching with Feature Augmentation</td><td>[2506.22336](http://arxiv.org/pdf/2506.22336)</td><td>◆ 首次提出跨特征检测器的特征匹配方法，解决了不同设备使用不同稀疏特征提取算法时视觉定位失效的问题。  
◆ 通过特征描述符增强技术提升跨检测器场景下的特征匹配性能，突破了现有方法依赖相同关键点的限制。  
◆ 引入特征转换到潜在空间的策略，有效应对关键点低重复性和描述符区分度不足的挑战。  
◆ 实验证明该方法在跨特征场景下显著提升了图像匹配和视觉定位的准确率。  
◆ 在多个基准数据集上验证了方法的有效性，为实际应用中不同描述符混合使用的场景提供了可行解决方案。</td></tr>
<tr><td>2025-06-27</td><td>Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs</td><td>[2506.22139](http://arxiv.org/pdf/2506.22139)</td><td>◆提出Q-Frame方法，通过查询自适应的帧选择策略解决视频-大语言模型中关键时空信息丢失的问题，突破传统均匀采样的局限性。  
◆创新性地结合CLIP等文本-图像匹配网络，实现无需训练的即插即用式帧选择，利用Gumbel-Max技巧提升选择效率。  
◆引入多分辨率缩放机制，根据视频内容和查询需求动态调整帧的时空分辨率，优化计算资源分配。  
◆在保持计算负载不变的前提下，显著增加可处理的帧数，同时保留对任务至关重要的时空细节。  
◆在MLVU、LongVideoBench等基准测试中验证了方法的优越性，涵盖多种视频理解任务，性能超越现有技术。  
◆为视频-大模型的实际应用提供轻量化解决方案，平衡了计算效率与语义理解深度的矛盾。</td></tr>
<tr><td>2025-06-27</td><td>ZeroReg3D: A Zero-shot Registration Pipeline for 3D Consecutive Histopathology Image Reconstruction</td><td>[2506.21923](http://arxiv.org/pdf/2506.21923)</td><td>◆ 提出ZeroReg3D，首个针对连续组织病理切片3D重建的零样本配准框架，无需训练或微调即可直接应用。  
◆ 创新结合零样本深度学习关键点匹配与基于优化的仿射/非刚性配准技术，解决传统方法在精度与泛化性上的矛盾。  
◆ 首次系统解决组织变形、切片伪影、染色差异和光照不一致四大挑战，显著提升3D重建的解剖结构保真度。  
◆ 突破现有深度学习方法依赖大规模标注数据的限制，通过零样本策略实现跨数据集的高适应性。  
◆ 公开完整代码库，为病理学研究和临床诊断提供可直接部署的开源工具。</td></tr>
<tr><td>2025-06-25</td><td>Fast entropy-regularized SDP relaxations for permutation synchronization</td><td>[2506.20191](http://arxiv.org/pdf/2506.20191)</td><td>◆ 提出了一种快速随机算法，用于解决部分排列同步问题（PPS）的半定规划（SDP）松弛，显著提升了多图像匹配的效率。  
◆ 利用熵正则化技术解决了标准松弛中优化解非唯一性的问题，增强了算法的稳定性和可靠性。  
◆ 开发了一种随机求解器，其计算复杂度在观测到的对应关系数量上接近最优，大幅提升了计算效率。  
◆ 设计了多种舍入程序，能够从隐式表示的原问题解变量中恢复组合解，同时支持保持循环一致性而不影响计算效率。  
◆ 在合成和真实数据集上验证了算法的优越性，在速度和精度方面均达到了当前最优水平。  
◆ 展示了熵正则化SDP在PPS问题中的理论和实践优势，为传统低秩或谱技术提供了新的替代方案。</td></tr>
<tr><td>2025-06-18</td><td>ReSeDis: A Dataset for Referring-based Object Search across Large-Scale Image Collections</td><td>[2506.15180](http://arxiv.org/pdf/2506.15180)</td><td>◆ 提出ReSeDis任务，首次将大规模图像检索与像素级定位统一，要求模型根据文本描述在图像库中同时判断对象是否存在并精确定位。  
◆ 构建首个针对该任务的基准数据集，确保每个描述唯一对应分散在大规模多样图像库中的对象实例，避免误匹配。  
◆ 设计联合评估指标，同时衡量检索召回率与定位精度，为端到端性能提供量化标准。  
◆ 提出基于冻结视觉语言模型的零样本基线方法，揭示该任务未来研究的巨大提升空间。  
◆ 为构建下一代鲁棒、可扩展的多模态搜索系统提供真实场景下的测试平台，弥补现有技术仅侧重检索或定位单一能力的缺陷。</td></tr>
<tr><td>2025-06-16</td><td>EmbodiedPlace: Learning Mixture-of-Features with Embodied Constraints for Visual Place Recognition</td><td>[2506.13133](http://arxiv.org/pdf/2506.13133)</td><td>◆ 提出了一种新颖的简单重排序方法，通过混合特征（MoF）方法在具身约束下优化全局特征，提升视觉地点识别（VPR）性能。  
◆ 首次系统分析了具身约束在VPR中的实际可行性，并根据现有数据集将其分类为GPS标签、时序戳、局部特征匹配和自相似矩阵等类型。  
◆ 设计了一种基于学习的MoF权重计算策略，采用多度量损失函数，有效融合多种特征信息。  
◆ 在公开数据集上实现了性能提升，仅需25 KB额外参数和每帧10微秒处理时间，计算开销极低。  
◆ 在Pitts-30k测试集上，基于DINOv2的基线性能提升0.9%，显著优于现有方法。</td></tr>
<tr><td>2025-06-12</td><td>RealKeyMorph: Keypoints in Real-world Coordinates for Resolution-agnostic Image Registration</td><td>[2506.10344](http://arxiv.org/pdf/2506.10344)</td><td>◆ 提出RealKeyMorph（RKM），首个无需固定分辨率重采样的医学图像配准方法，直接处理原始分辨率数据，避免插值伪影。  
◆ 创新性地将关键点输出为扫描仪真实世界坐标（而非体素坐标），通过利用扫描仪生成的仿射矩阵实现跨分辨率配准。  
◆ 扩展KeyMorph框架，在训练过程中融入真实世界坐标转换，使关键点提取与图像分辨率完全解耦。  
◆ 在腹部MRI正交2D堆栈和不同分辨率3D脑数据集上验证了方法的优越性，证明其对分辨率差异的鲁棒性。  
◆ 通过闭式关键点匹配计算变换参数，保持了KeyMorph原有的可解释性优势，同时突破分辨率限制。</td></tr>
<tr><td>2025-06-11</td><td>Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints</td><td>[2506.09748](http://arxiv.org/pdf/2506.09748)</td><td>◆ 提出了一种分层跨源图像匹配方法，结合语义感知和结构约束的粗匹配模块与轻量级细粒度匹配模块，显著提升了无人机绝对视觉定位的精度。  
◆ 利用视觉基础模型提取语义特征，在语义和结构约束下建立区域级对应关系，有效解决了跨源差异和时变因素导致的匹配难题。  
◆ 设计了轻量级细粒度匹配模块，通过提取精细特征建立像素级对应关系，进一步提升了定位的准确性。  
◆ 构建了不依赖相对定位技术的无人机绝对视觉定位流程，通过图像检索模块与分层匹配模块的结合，实现了独立定位。  
◆ 在公开基准数据集和新提出的CS-UAV数据集上验证了方法的优越性，展示了其在多种挑战性条件下的高精度和鲁棒性。</td></tr>
<tr><td>2025-06-11</td><td>ScaleLSD: Scalable Deep Line Segment Detection Streamlined</td><td>[2506.09369](http://arxiv.org/pdf/2506.09369)<br><a href=''>[代码]</a></td><td>◆ 提出ScaleLSD，首个通过大规模自监督学习（超过1000万无标签图像）训练的领域无关鲁棒线检测模型，显著提升自然图像的线段检测能力。  
◆ 重新设计并简化了传统（深度与非深度）线段检测方法的核心架构，实现高效高性能的线段检测，检测数量远超经典非深度方法。  
◆ 在线段几何表征上更完整且准确，首次实现深度方法在所有测试场景（检测性能、单视图3D几何估计等）全面超越经典非深度LSD。  
◆ 通过零样本协议验证模型泛化性，在单视图3D重建、双视图线段匹配、多视图3D线段映射等任务中均表现优异。  
◆ 开源代码与模型，为图像线几何的广泛应用（如三维重建、匹配）提供更强通用性支持，强化线段几何在多任务中的实用性。</td></tr>
<tr><td>2025-05-21</td><td>Anti-interrupted sampling repeater jamming via linear canonical Wigner distribution lightweight LFM detection</td><td>[2506.06302](http://arxiv.org/pdf/2506.06302)</td><td>◆ 提出基于广义线性正则维格纳分布（GLWD）的抗干扰方法，通过合理设置参数获得高时频分辨率和能量集中性，显著提升信号分离能力和信噪比。  
◆ 改进现有移动线段检测（M-LSD）算法，提出移动长线段检测（M-LSD）算法，增强对目标线性调频信号的检测能力，降低对干扰信号的敏感性。  
◆ 利用GLWD与短时傅里叶变换（STFT）的映射关系构建时频滤波器，在STFT域进行滤波以高效抑制干扰。  
◆ 方法在低信噪比条件下仍能有效区分能量接近真实目标的干扰采样转发干扰（ISRJ），解决传统时频域方法在多分量信号场景中的时频混叠问题。  
◆ 仿真与实验验证了该方法对难区分干扰的有效抑制能力，兼具实时性和鲁棒性，适用于实际雷达抗干扰场景。</td></tr>
<tr><td>2025-06-05</td><td>Vanishing arcs for isolated plane curve singularities</td><td>[2506.04917](http://arxiv.org/pdf/2506.04917)</td><td>◆ 提出&quot;消失弧集&quot;新概念，作为传统消失循环的几何对应物，通过几何变分算子将嵌入弧与闭曲线联系起来。  
◆ 建立几何变分算子的拓扑框架，用几何弧和闭曲线替代同调循环，拓展了经典超曲面奇点理论的工具集。  
◆ 给出判定嵌入弧被几何变分算子映射为消失循环的充要条件，基于弧与几何单值化映像的交点数特征。  
◆ 证明对任意由A&#x27;Campo剖分产生的消失循环集，存在拓扑例外弧集使其变分映像与该消失循环集完全匹配。  
◆ 将几何单值化与交点数理论相结合，为平面曲线奇点的拓扑研究提供了新的几何化方法。</td></tr>
<tr><td>2025-06-05</td><td>Deep Learning Reforms Image Matching: A Survey and Outlook</td><td>[2506.04619](http://arxiv.org/pdf/2506.04619)</td><td>◆ 该论文首次从深度学习逐步改造传统图像匹配流程的视角，系统梳理了该领域的革命性进展，突破了传统综述按技术分类的框架。  
◆ 提出与经典流水线高度对齐的新型分类体系：一方面拆解各环节的可学习替代方案（如可学习检测-描述子、离群点过滤器），另一方面整合多环节的端到端模块（如中端稀疏匹配器、稠密匹配器）。  
◆ 深度剖析了可学习组件与端到端模块的设计哲学及优劣，首次明确揭示两类技术路线的互补性与适用边界。  
◆ 在相对位姿恢复、单应估计等核心任务上建立统一评测基准，定量比较了代表性方法的性能突破与现存缺陷。  
◆ 前瞻性指出自监督学习、跨模态匹配、动态场景适应等未来方向，为领域发展绘制了清晰的技术演进地图。  
◆ 通过揭示传统流程被深度学习&quot;解构-重构&quot;的完整路径，为计算机视觉基础问题研究提供了方法论层面的新范式。</td></tr>
<tr><td>2025-06-20</td><td>SR3D: Unleashing Single-view 3D Reconstruction for Transparent and Specular Object Grasping</td><td>[2505.24305](http://arxiv.org/pdf/2505.24305)</td><td>◆提出SR3D框架，首次实现无需训练的基于单视角的透明与镜面物体3D重建与抓取，突破传统深度感知限制。  
◆利用外部视觉模型直接从RGB图像生成物体网格，结合深度图实现3D场景融合，避免复杂多视角采集系统。  
◆创新性提出视图匹配与关键点匹配双机制，联合2D语义与3D几何信息精准定位物体位姿与尺度。  
◆通过将重建物体逆向映射回原始深度缺失场景，生成高精度深度图，显著提升抓取检测效果。  
◆在仿真与真实场景中验证有效性，为透明/镜面物体抓取提供实用化解决方案，简化硬件依赖。</td></tr>
<tr><td>2025-06-05</td><td>Universal Domain Adaptation for Semantic Segmentation</td><td>[2505.22458](http://arxiv.org/pdf/2505.22458)</td><td>这篇论文提出了通用领域自适应语义分割（UniDA-SS）方法，解决了传统方法因忽略类别设置差异导致的性能下降问题。其核心贡献和创新点如下：

◆ 提出UniDA-SS框架，首次在语义分割任务中实现无需预知源域与目标域类别设置的通用领域自适应。  
◆ 设计Domain-Specific Prototype-based Distinction（DSPD）模块，通过将每类划分为两个域特定原型，增强跨域共有类别的特征区分能力。  
◆ 开发Target-based Image Matching（TIM）策略，基于目标域伪标签匹配最佳源域图像进行批量训练，有效提升共有类别的学习效果。  
◆ 构建新的UniDA-SS基准数据集，为后续研究提供标准化评估平台。  
◆ 实验证明UniMAP方法显著优于基线模型，代码已开源。</td></tr>
<tr><td>2025-05-23</td><td>To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models</td><td>[2505.17973](http://arxiv.org/pdf/2505.17973)<br><a href=''>[代码]</a></td><td>◆ 首次系统比较了传统手工特征匹配（如SIFT+RANSAC）与深度学习特征匹配方法在语义3D建筑模型相机定位任务中的性能差异。  
◆ 针对移动测绘相机与纹理化CityGML LoD2模型的匹配场景，提出定制化评估框架，填补了该领域的研究空白。  
◆ 结合标准数据集（HPatches、MegaDepth-1500）和自建数据集（含地面/无人机拍摄的立面纹理与对应影像），验证方法普适性。  
◆ 通过PnP算法量化绝对位姿估计精度，利用地理参考轨迹数据生成几何真值，建立客观评估基准。  
◆ 实验证明学习式特征匹配在挑战性场景（RANSAC内点数0-12、AUC 0-0.16）中显著优于传统方法，准确率和鲁棒性提升明显。  
◆ 公开代码库促进模型化视觉定位技术发展，为后续研究提供可复现基础。</td></tr>
<tr><td>2025-05-16</td><td>Multi-view dense image matching with similarity learning and geometry priors</td><td>[2505.11264](http://arxiv.org/pdf/2505.11264)</td><td>◆提出MV-DeepSimNets深度学习框架，首次将多视图相似性学习与极线几何先验结合，无需繁琐的多视图训练数据构建。  
◆创新性地引入在线几何先验，通过极线约束或单应性校正动态建模像素关系，生成几何感知的特征表示。  
◆采用平面扫描法将几何特征投影到候选深度假设空间，实现端到端的几何条件化特征适配，提升多视图重建精度。  
◆通过聚合学习到的相似性构建并正则化代价体，相比传统稠密匹配方法显著改善了表面重建质量。  
◆在泛化能力上表现突出，可同时适用于航空影像和卫星影像（不同地面采样距离），性能超越主流相似性学习网络和端到端回归模型。  
◆完整集成至MicMac开源软件，可直接兼容标准多分辨率影像匹配流程，具备工程实用价值。</td></tr>
<tr><td>2025-05-12</td><td>Boosting Global-Local Feature Matching via Anomaly...</td><td>[2505.07375](http://arxiv.org/pdf/2505.07375)<br><a href=''>[代码]</a></td><td>◆提出GLFM方法，通过全局-局部特征匹配解决多类别点云异常检测中的特征混淆问题。  
◆创新性地设计了三阶段框架：异常合成增强特征表示、建立抗混淆的全局-局部记忆库、基于特征距离的异常检测，显...</td></tr>
<tr><td>2025-05-04</td><td>OBD-Finder: Explainable Coarse-to-Fine Te...</td><td>[2505.03836](http://arxiv.org/pdf/2505.03836)<br><a href=''>[代码]</a></td><td>◆提出了一种渐进式甲骨文重复片发现框架，结合无监督低层关键点匹配与高层以文本为中心的内容匹配，实现语义感知和可解释的候选排序。◆在保持高召回率的同时，该方法在Top-5和Top-15检索结果中取...</td></tr>
<tr><td>2025-05-06</td><td>LiftFeat: 3D Geometry-Aware Local Feature Matching</td><td>[2505.03422](http://arxiv.org/pdf/2505.03422)<br><a href=''>[代码]</a></td><td>◆ 提出LiftFeat轻量网络，通过融合单目深度估计生成的伪表面法线特征与原始2D描述符，增强特征匹配在光照变化、弱纹理等极端场景下的鲁棒性。  
◆ 设计3D几何感知特征提升模块，利用表面法...</td></tr>
<tr><td>**2025-05-04**</td><td>**Focus What Matters: Matchability-Based Reweighting for Local Feature Matching**</td><td>[2505.02161](http://arxiv.org/abs/2505.02161)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-15**</td><td>**Mitigating Modality Bias in Multi-modal Entity Alignment from a Causal Perspective**</td><td>[2504.19458](http://arxiv.org/abs/2504.19458)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
</tbody>
</table>
</div>

<h2 id='nerf'>NeRF</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
<tr><td>2025-07-02</td><td>Tile and Slide : A New Framework for Scaling NeRF from Local to Global 3D Earth Observation</td><td>[2507.01631](http://arxiv.org/pdf/2507.01631)</td><td>◆ 提出Snake-NeRF框架，首次实现单设备上大规模卫星影像的NeRF三维重建，突破传统方法受限于内存的小场景约束。  
◆ 设计外存（out-of-core）训练方法，无需同时加载所有图像和网络，显著降低硬件需求。  
◆ 创新性采用无重叠三维分块（3D tile）策略，将目标区域划分为独立训练的NeRF子模块。  
◆ 提出重叠裁剪图像技术，确保每个子模块训练时获取完整必要像素，避免边界信息缺失。  
◆ 开发2×2三维分块递进策略与分段采样器，有效消除分块边缘的三维重建误差。  
实验证明该方法在单GPU上实现线性时间复杂度，且不损失重建质量，为全球尺度地球观测提供新范式。</td></tr>
<tr><td>2025-07-01</td><td>Surgical Neural Radiance Fields from One Image</td><td>[2507.00969](http://arxiv.org/pdf/2507.00969)</td><td>◆ 提出了一种基于单张术中图像和术前MRI数据训练神经辐射场（NeRF）的新方法，解决了手术场景中多视角数据不足的限制。  
◆ 利用术前MRI数据预先定义相机视角和图像集，结合神经风格迁移技术（WTC2和STROTSS）将术中图像外观迁移至预构建数据集，避免过度风格化。  
◆ 实现了快速单图像NeRF训练，显著降低了术中数据采集的时间成本，提升了临床实用性。  
◆ 在四例神经外科手术案例中验证了方法的有效性，定量对比显示其合成结果与真实手术显微镜图像高度一致。  
◆ 重建结果与真实数据相比具有高结构相似性，证明了良好的重建质量和纹理保留能力。  
◆ 为手术场景中的实时3D重建和视角合成提供了可行方案，突破了传统多视角方法的局限性。</td></tr>
<tr><td>2025-07-01</td><td>PlantSegNeRF: A few-shot, cross-dataset method for plant 3D instance point cloud reconstruction via joint-channel NeRF with multi-view image instance matching</td><td>[2507.00371](http://arxiv.org/pdf/2507.00371)</td><td>◆提出PlantSegNeRF方法，首次实现从多视角RGB图像序列直接生成高精度植物器官实例点云，突破传统点云分割技术的局限性。  
◆开发联合通道NeRF模型，同时渲染颜色、密度、语义和实例信息，构建包含多维度特征的隐式场景表示。  
◆设计创新的多视角实例匹配模块，通过2D实例分割结果跨视图关联同一器官的实例ID，解决复杂植物结构的对应难题。  
◆在语义分割任务中，关键指标（精确率、召回率等）平均提升16.1%-24.2%，显著优于现有最优方法。  
◆在实例分割任务中，四项核心指标（mPrec等）最高提升达38.2%，实现跨物种的高泛化性表现。  
◆为植物表型研究提供高通量三维数据生成方案，支持大规模植物模型开发。</td></tr>
<tr><td>2025-06-30</td><td>AttentionGS: Towards Initialization-Free 3D Gaussian Splatting via Structural Attention</td><td>[2506.23611](http://arxiv.org/pdf/2506.23611)</td><td>◆ 提出AttentionGS框架，首次实现无需高质量初始点云的3D高斯泼溅重建，突破传统3DGS对SfM点云的强依赖。  
◆ 创新性引入两阶段注意力机制：几何注意力快速恢复场景全局结构，纹理注意力后期优化细粒度细节，实现从随机初始化直接重建。  
◆ 设计不透明度加权梯度策略，改进高斯分布致密化过程，显著提升表面重建质量。  
◆ 在纹理缺失和受限视角等极端场景下表现优异，相比现有方法重建质量提升显著。  
◆ 通过多基准数据集验证，为实际应用中更鲁棒的3D重建提供新思路，扩展了3DGS的应用边界。</td></tr>
<tr><td>2025-06-29</td><td>Dynamic View Synthesis from Small Camera Motion Videos</td><td>[2506.23153](http://arxiv.org/pdf/2506.23153)</td><td>这篇论文针对动态3D场景在小范围相机运动下的新视角合成问题提出了创新解决方案，核心贡献如下：

◆ 提出基于分布的深度正则化方法(DDR)，通过Gumbel-softmax从离散渲染权重分布中可微分采样，解决了传统深度损失仅计算期望误差的局限性。

◆ 引入物体边界前空间点体积密度趋近零的约束条件，确保场景几何结构的正确学习，有效改善了小相机运动下的几何表示问题。

◆ 开发了可视化工具，可直接在渲染权重层面观察场景几何表示，为方法原理提供了直观解释。

◆ 在训练过程中加入相机参数学习机制，增强了模型对相机参数的鲁棒性，解决了小运动下相机参数估计不准的问题。

论文通过大量实验证明，该方法在小范围相机运动输入下显著优于现有先进方法，为动态场景新视角合成提供了更实用的解决方案。</td></tr>
<tr><td>2025-06-27</td><td>UnMix-NeRF: Spectral Unmixing Meets Neural Radiance Fields</td><td>[2506.21884](http://arxiv.org/pdf/2506.21884)</td><td>◆ 首次将光谱解混技术融入神经辐射场（NeRF），实现联合高光谱新视角合成与无监督材质分割，突破传统NeRF仅依赖RGB数据的局限。  
◆ 提出基于漫反射和镜面反射分量的光谱反射率建模方法，通过全局端元字典学习纯材质特征，结合逐点丰度分布实现材质精准表达。  
◆ 创新性地利用学习到的端元光谱特征进行无监督材质聚类，无需人工标注即可完成场景材质分割。  
◆ 支持通过修改端元字典实现场景材质编辑，为基于材质的灵活外观操控（如虚拟仿真、AR应用）提供新工具。  
◆ 实验证明该方法在高光谱重建和材质分割任务上显著优于现有技术，为机器人感知、虚拟现实等需精确材质建模的领域提供解决方案。</td></tr>
<tr><td>2025-06-24</td><td>ICP-3DGS: SfM-free 3D Gaussian Splatting for Large-scale Unbounded Scenes</td><td>[2506.21629](http://arxiv.org/pdf/2506.21629)</td><td>◆ 提出了一种无需SfM预处理的方法ICP-3DGS，通过结合迭代最近点（ICP）和基于优化的位姿细化，实现了大范围相机运动下的高精度位姿估计。  
◆ 引入基于体素的场景致密化策略，有效指导大规模无边界场景的3D高斯分布重建，解决了传统方法在户外场景中的扩展性问题。  
◆ 首次将ICP与3D高斯泼溅（3DGS）技术结合，在神经渲染框架中直接优化相机位姿，摆脱了对SfM先验数据的依赖。  
◆ 通过实验验证，该方法在室内外不同尺度场景中均优于现有技术，同时在相机位姿估计和新视角合成任务上表现更优。  
◆ 开源了完整代码，为后续研究提供了可复现的基础，推动了无约束场景神经渲染的实用化进程。</td></tr>
<tr><td>2025-06-26</td><td>PanSt3R: Multi-view Consistent Panoptic Segmentation</td><td>[2506.21348](http://arxiv.org/pdf/2506.21348)</td><td>◆ 提出PanSt3R方法，首次实现无需测试时优化的单次前向预测，直接联合输出3D几何和多视角全景分割结果，显著提升效率。  
◆ 基于MUSt3R框架改进，引入语义感知能力，将3D重建与多视角全景分割任务统一整合，克服传统方法依赖2D预分割的局限性。  
◆ 重新设计掩码后处理流程，提出更理论化的多视角分割融合策略，优化跨视角空间关系利用。  
◆ 结合3D高斯泼溅（3DGS）技术，提出简单有效的新视角生成方法，扩展模型应用场景。  
◆ 在多个基准测试中达到SOTA性能，速度比现有方法快数个数量级，兼具概念简洁性与计算高效性。</td></tr>
<tr><td>2025-06-25</td><td>Joint attitude estimation and 3D neural reconstruction of non-cooperative space objects</td><td>[2506.20638](http://arxiv.org/pdf/2506.20638)</td><td>◆ 提出了一种联合优化方法，同时估计非合作空间物体的姿态（相机位姿）并利用神经辐射场（NeRF）进行3D重建，解决了传统方法在未知物体姿态下的重建难题。  
◆ 针对空间场景的特殊挑战（如单色图像、未知物体方向、有限视角、无漫反射光照等），改进了NeRF的适应性，使其在极端条件下仍能有效工作。  
◆ 实验证明，采用逐帧顺序训练图像的方式（而非批量训练）能显著提升3D重建的精度，为动态空间物体建模提供了新思路。  
◆ 通过优化均匀旋转参数估计相机姿态，并引入正则化约束相邻姿态的连续性，避免了位姿估计的突变问题。  
◆ 该方法为空间态势感知（SSA）任务提供了高精度的3D模型，可应用于主动碎片清除、在轨维护等实际场景。</td></tr>
<tr><td>2025-06-24</td><td>NeRF-based CBCT Reconstruction needs Normalization and Initialization</td><td>[2506.19742](http://arxiv.org/pdf/2506.19742)</td><td>◆ 提出归一化哈希编码器（Normalized Hash Encoder），解决NeRF-based CBCT重建中哈希编码器与神经网络的局部-全局训练不匹配问题，通过增强特征一致性提升训练稳定性。  
◆ 设计映射一致性初始化策略（MCI），利用预训练模型的全局映射特性初始化神经网络，减少早期训练波动，加速收敛并提高重建质量。  
◆ 首次系统分析了哈希编码器参数局部稀疏性与神经网络全局密集更新的矛盾，指出特征错位是导致训练不稳定的核心原因。  
◆ 方法仅需少量代码改动，即可在4个数据集、128例CT数据（涵盖7个解剖区域）上显著提升训练效率和重建性能。  
◆ 通过实验验证了归一化与初始化策略的协同作用，为NeRF-based医学影像重建提供了简单有效的优化范式。</td></tr>
<tr><td>2025-06-25</td><td>Self-Supervised Multimodal NeRF for Autonomous Driving</td><td>[2506.19615](http://arxiv.org/pdf/2506.19615)</td><td>◆ 提出自监督多模态NeRF框架NVSF，无需3D标注即可联合学习LiDAR和相机的时空隐式神经表示。  
◆ 针对自动驾驶场景设计，同时处理静态和动态环境，显著提升真实驾驶场景的适应性。  
◆ 引入启发式图像像素采样策略，优先选择信息丰富的像素，提升训练效率和收敛速度。  
◆ 创新采用双梯度掩码技术，有效保留LiDAR点的局部特征，增强点云数据重建精度。  
◆ 在KITTI-360数据集上验证，LiDAR和相机域性能均超越基线模型，展现多模态优势。  
◆ 开源代码推动相关研究，为自动驾驶领域提供可复用的新型神经渲染解决方案。</td></tr>
<tr><td>2025-06-24</td><td>HoliGS: Holistic Gaussian Splatting for Embodied View Synthesis</td><td>[2506.19291](http://arxiv.org/pdf/2506.19291)</td><td>◆ 提出HoliGS框架，首次将可变形高斯泼溅技术应用于长时序单目RGB视频的沉浸式视角合成任务，解决了传统4D高斯泼溅和动态NeRF在分钟级视频中训练开销过大的问题。  
◆ 创新性地采用分层变形策略，将场景分解为静态背景和动态物体，其中动态部分通过可逆神经流实现全局刚性变换、骨骼驱动形变和细微非刚性形变的统一建模。  
◆ 通过将高斯基元绑定到完整的前景规范形状（如第一人称或跟随视角），支持多演员交互和大视角变化的自由视点渲染，显著提升了复杂动态场景的重建鲁棒性。  
◆ 提出可逆高斯变形网络，在保持高保真重建质量的同时，相比现有单目可变形NeRF方法大幅降低训练和渲染时间，实现了实际场景中的高效部署。  
◆ 在挑战性数据集上的实验表明，该方法在重建质量和计算效率方面均优于当前最优技术，为沉浸式视角合成提供了可扩展的实用解决方案。</td></tr>
<tr><td>2025-06-23</td><td>MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation</td><td>[2506.18678](http://arxiv.org/pdf/2506.18678)</td><td>◆ 提出首个分布式多智能体协作神经SLAM框架MCN-SLAM，结合混合隐式神经场景表示，解决传统单智能体隐式SLAM在大场景和长序列中的局限性。  
◆ 创新设计三平面-网格联合场景表示方法，显著提升场景重建质量，优于现有NeRF-based方法。  
◆ 开发&quot;内部-跨智能体&quot;闭环检测机制，首次实现单智能体局部一致性与多智能体全局一致性的协同优化。  
◆ 提出在线蒸馏方法实现多子地图融合，突破通信带宽限制，确保全局地图一致性。  
◆ 发布首个真实世界密集SLAM数据集DES，涵盖单/多智能体场景，提供连续轨迹和高精度3D网格真值，填补领域空白。  
实验证明该方法在建图、定位和通信效率上均优于现有技术，代码与数据集将开源推动SLAM和3D重建领域发展。</td></tr>
<tr><td>2025-06-26</td><td>2D Triangle Splatting for Direct Differentiable Mesh Training</td><td>[2506.18575](http://arxiv.org/pdf/2506.18575)<br><a href=''>[代码]</a></td><td>◆ 提出2D三角形面片（2DTS）方法，替代传统3D高斯基元，实现更高效的直接可微分网格训练。  
◆ 结合离散网格结构与连续体积建模优势，形成类网格的表示形式，提升渲染质量和灵活性。  
◆ 引入紧凑性参数到三角形基元中，支持直接训练高真实感网格，简化传统网格重建流程。  
◆ 实验证明，即使未优化紧凑性参数，其基础版本也能超越当前最优高斯基元方法的渲染保真度。  
◆ 生成的网格在视觉质量上显著优于现有网格重建方法，尤其在复杂光照和阴影效果中表现突出。  
◆ 为可微分渲染领域提供新思路，平衡了渲染速度与高级渲染效果（如重光照）的兼容性。</td></tr>
<tr><td>2025-06-22</td><td>Limitations of NERF with pre-trained Vision Features for Few-Shot 3D Reconstruction</td><td>[2506.18208](http://arxiv.org/pdf/2506.18208)</td><td>◆ 首次系统评估了DINO预训练视觉特征在NeRF少样本3D重建中的表现，发现所有变体性能均低于原始NeRF基线（PSNR 12.9-13.0 vs 14.71）。  
◆ 揭示了反直觉结论：预训练视觉特征不仅无助于少样本重建，反而可能引入有害偏差，挑战了该领域普遍假设。  
◆ 提出三种潜在失效原因分析框架：特征-任务不匹配、有限数据过拟合问题以及特征融合技术瓶颈。  
◆ 通过对比实验验证了冻结特征、LoRA微调和多尺度融合等主流方法的局限性，为后续研究排除无效路径。  
◆ 指出少样本场景下应优先关注几何一致性而非复杂特征工程，为简化模型设计提供新方向。  
◆ 研究成果对基于预训练特征的3D重建方法提出重要警示，可能改变该领域技术路线选择。</td></tr>
<tr><td>2025-06-21</td><td>3D Gaussian Splatting for Fine-Detailed Surface Reconstruction in Large-Scale Scene</td><td>[2506.17636](http://arxiv.org/pdf/2506.17636)</td><td>◆ 提出从粗到精的渐进式重建策略，先快速构建粗糙模型，再通过自适应场景分割和子场景细化实现大规模场景的高效重建。  
◆ 创新性地结合解耦外观模型，有效捕捉户外环境中复杂的全局光照变化，提升动态外观的建模能力。  
◆ 设计瞬态掩模模型，自动过滤移动物体（如车辆、行人）的干扰，显著提高重建纯净度。  
◆ 扩展多视角约束并引入单视角正则化方法，针对性解决纹理缺失区域的几何优化难题。  
◆ 在无人机航拍数据集GauU-Scene V2上验证，首次实现全尺寸图像优化的大规模场景精细重建，性能超越现有NeRF和Gaussian类方法。  
（注：全文严格遵循5点创新性总结，未使用Markdown符号，字数控制在400字内）</td></tr>
<tr><td>2025-06-23</td><td>R3eVision: A Survey on Robust Rendering, Restoration, and Enhancement for 3D Low-Level Vision</td><td>[2506.16262](http://arxiv.org/pdf/2506.16262)<br><a href=''>[代码]</a></td><td>◆ 提出“3D低层视觉（3D LLV）”新领域，将传统2D低层视觉任务（如超分、去模糊、天气退化修复等）扩展到3D空间，解决神经渲染在真实退化场景中的鲁棒性问题。  
◆ 首次系统化定义“退化感知渲染”问题，明确时空一致性和病态优化等核心挑战，为3D LLV研究建立理论框架。  
◆ 综述了将低层视觉技术与神经辐射场（NeRF）、3D高斯泼溅（3DGS）等神经渲染结合的创新方法，展示其在噪声、模糊、低分辨率等退化条件下的高保真3D重建能力。  
◆ 梳理了自动驾驶、AR/VR、机器人等关键应用场景，强调从退化输入中实现可靠3D感知的实用价值。  
◆ 汇总了代表性方法、数据集和评估协议，为未来3D LLV研究提供标准化参考，推动真实环境下鲁棒3D内容生成与场景重建的发展。</td></tr>
<tr><td>2025-06-24</td><td>RA-NeRF: Robust Neural Radiance Field Reconstruction with Accurate Camera Pose Estimation under Complex Trajectories</td><td>[2506.15242](http://arxiv.org/pdf/2506.15242)</td><td>◆ 提出RA-NeRF方法，能够在复杂相机轨迹下实现高精度的相机位姿估计，解决了传统NeRF和3DGS依赖准确位姿先验的问题。  
◆ 采用增量式重建流程，结合光度一致性约束和光流驱动的位姿调节机制，提升了初始化和定位阶段的鲁棒性。  
◆ 引入隐式位姿滤波器，通过捕捉相机运动模式有效消除位姿估计中的噪声，增强复杂轨迹下的稳定性。  
◆ 在Tanks&amp;Temple和NeRFBuster等具有挑战性的数据集上验证了方法有效性，位姿估计和视觉质量均达到SOTA水平。  
◆ 整体框架无需外部约束，仅通过端到端优化即可同时优化场景重建与相机位姿，适用于SLAM等实际应用场景。</td></tr>
<tr><td>2025-06-17</td><td>Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction</td><td>[2506.14856](http://arxiv.org/pdf/2506.14856)</td><td>◆提出了一种基于轻量级前馈神经网络UPNet的新颖主动视角选择方法，直接预测候选视角的不确定性图，避免了传统方法需要计算每个视角不确定性的高计算成本。  
◆UPNet仅需单张输入图像即可预测所有候选视角的不确定性，通过学习自然物体视角与体素表示不确定性的映射关系，实现了高效的信息提取。  
◆通过聚合历史预测的不确定性图来抑制冗余视角，智能选择信息量最大的新视角，仅需一半视角即可达到与上限相当的3D重建精度。  
◆相比基线方法，计算效率显著提升，实现高达400倍的加速，并减少50%以上的CPU、RAM和GPU资源消耗。  
◆方法具有强大的泛化能力，无需额外训练即可适用于新物体类别的视角选择任务，展现了广泛的适用性。  
◆整体方案将神经渲染与高效视角选择相结合，为3D重建领域提供了高精度与低资源消耗的实用化解决方案。</td></tr>
<tr><td>2025-06-18</td><td>Rasterizing Wireless Radiance Field via Deformable 2D Gaussian Splatting</td><td>[2506.12787](http://arxiv.org/pdf/2506.12787)</td><td>◆ 提出SwiftWRF框架，首次将高斯泼溅（Gaussian Splatting）技术引入无线辐射场（WRF）建模，突破传统方法在精度和效率上的局限。  
◆ 采用可变形2D高斯泼溅方法，通过轻量级MLP建模高斯形变，有效捕捉收发端单侧移动导致的WRF动态变化。  
◆ 实现CUDA加速的光栅化渲染，频谱合成速度超过10万帧/秒，比现有最优方法快500倍，满足实时性需求。  
◆ 创新性地支持任意位置的WRF频谱合成，并在到达角（AoA）和信号强度（RSSI）预测任务中验证实用性。  
◆ 在真实和合成室内场景的实验中，显著提升信号重建质量，同时开源代码和数据集推动领域发展。</td></tr>
<tr><td>2025-06-17</td><td>Efficient multi-view training for 3D Gaussian Splatting</td><td>[2506.12727](http://arxiv.org/pdf/2506.12727)</td><td>这篇论文的核心贡献和创新点如下：

◆ 提出多视角训练方法，解决了3D高斯泼溅（3DGS）传统单视角训练导致的随机梯度方差过大问题，优化了训练效果。  
◆ 改进了光栅化流程，显著降低了多视角训练的计算开销，使其更高效可行。  
◆ 设计了3D距离感知的D-SSIM损失函数，更好地适应多视角场景，提升了渲染质量。  
◆ 提出多视角自适应密度控制机制，克服了传统单视角假设下高斯分布优化的局限性。  
◆ 实验证明，所提方法显著提升了3DGS及其变体的性能，突破了单视角训练的约束。  
◆ 为3DGS领域提供了更高效的训练框架，推动了其在逆向渲染中的应用。</td></tr>
<tr><td>2025-06-12</td><td>PointGS: Point Attention-Aware Sparse View Synthesis with Gaussian Splatting</td><td>[2506.10335](http://arxiv.org/pdf/2506.10335)</td><td>◆ 提出PointGS框架，通过点注意力感知的稀疏视图合成方法，解决了3D高斯泼溅（3DGS）在输入视图不足时过拟合的问题。  
◆ 利用最新的立体基础模型估计精确相机姿态并重建密集点云，为高斯初始化提供高质量起点。  
◆ 设计多尺度2D外观特征采样与聚合机制，为每个3D高斯点编码颜色属性，增强稀疏输入下的特征表达能力。  
◆ 创新性地引入基于自注意力机制的点交互网络，使高斯点能与邻近点交互，提升点级外观表示能力。  
◆ 通过两个轻量级多层感知机（MLP）将增强特征解码为高斯参数，实现实时高质量渲染。  
◆ 在多个基准测试中显著优于基于NeRF的方法，并在少样本设置下达到与最先进3DGS方法竞争的性能。</td></tr>
<tr><td>2025-06-11</td><td>The Less You Depend, The More You Learn: Synthesizing Novel Views from Sparse, Unposed Images without Any 3D Knowledge</td><td>[2506.09885](http://arxiv.org/pdf/2506.09885)</td><td>◆ 提出了一种无需3D先验知识和相机位姿标注的通用化新视角合成框架，仅依赖稀疏无位姿的2D图像即可生成逼真新视图。  
◆ 通过系统性分析揭示了关键趋势：减少对3D知识的依赖能更高效利用数据规模，最终达到与依赖3D知识的方法相当的性能。  
◆ 创新性地消除了传统方法对显式3D表示（如NeRF、3DGS）和输入/目标视角位姿标注的双重依赖，实现完全数据驱动的隐式3D理解。  
◆ 实验证明该方法仅通过稀疏2D图像即可学习隐式3D一致性，生成质量媲美依赖位姿输入的方法，验证了数据为中心范式的可行性。  
◆ 为大规模数据时代的新视角合成提供了新思路，表明减少3D先验依赖与数据规模扩展之间存在正向关联性。</td></tr>
<tr><td>2025-06-10</td><td>A Probability-guided Sampler for Neural Implicit Surface Rendering</td><td>[2506.08619](http://arxiv.org/pdf/2506.08619)</td><td>◆ 提出基于概率密度函数的3D图像投影空间模型，实现针对感兴趣区域的射线采样优化，提升渲染精度。  
◆ 设计新型表面重建损失函数，充分利用3D投影空间模型，整合近表面和空白空间信息以增强性能。  
◆ 结合隐式表面表示，通过概率引导采样策略有效聚焦关键区域，减少冗余计算。  
◆ 将提出的采样策略与损失函数集成到现有神经隐式表面渲染器中，显著提升3D重建和图像渲染质量。  
◆ 特别针对场景中感兴趣区域（如物体表面）实现更精细的细节还原，克服传统均匀采样的局限性。  
◆ 通过联合优化采样与重建过程，在保证计算效率的同时获得更高保真度的渲染结果。</td></tr>
<tr><td>2025-06-09</td><td>Speedy Deformable 3D Gaussian Splatting: Fast Rendering and Compression of Dynamic Scenes</td><td>[2506.07917](http://arxiv.org/pdf/2506.07917)<br><a href=''>[代码]</a></td><td>◆ 提出SpeeDe3DGS框架，显著加速动态3D高斯泼溅（3DGS/4DGS）的渲染速度，解决传统方法因逐帧神经网络推理导致的性能瓶颈。  
◆ 设计时序敏感度剪枝评分机制，自动识别并剔除对动态场景重建贡献低的冗余高斯元素，提升计算效率。  
◆ 引入退火平滑剪枝策略，增强在相机位姿不精确的真实场景中的剪枝鲁棒性，避免误删关键高斯元素。  
◆ 开发GroupFlow运动分析技术，通过轨迹相似性聚类高斯群组，以单组刚性变换替代逐高斯形变预测，大幅减少计算量。  
◆ 实验验证框架在NeRF-DS数据集上实现10.37倍渲染加速、7.71倍模型压缩和2.71倍训练提速，在D-NeRF和HyperNeRF数据集分别提升4.20倍和58.23倍性能。  
◆ 模块化设计兼容现有动态3DGS/4DGS框架，兼具高效性与通用性。</td></tr>
<tr><td>2025-06-20</td><td>Genesis: Multimodal Driving Scene Generation with Spatio-Temporal and Cross-Modal Consistency</td><td>[2506.07497](http://arxiv.org/pdf/2506.07497)</td><td>◆ 提出Genesis框架，首次实现多视角驾驶视频与LiDAR序列的联合生成，保证时空和跨模态一致性。  
◆ 采用两阶段架构：结合DiT视频扩散模型与3D-VAE编码，以及基于BEV的LiDAR生成器与NeRF渲染，实现高质量多模态输出。  
◆ 通过共享潜在空间直接耦合视觉与几何模态，确保生成内容在跨模态间的连贯演化。  
◆ 创新引入DataCrafter描述模块，利用视觉语言模型提供场景级和实例级语义监督，增强生成数据的结构化控制。  
◆ 在nuScenes基准测试中取得视频（FVD 16.95）和LiDAR（Chamfer 0.611）指标的SOTA性能，验证生成数据的语义保真度。  
◆ 生成数据可有效提升下游任务（如分割和3D检测）性能，证明其实际应用价值。</td></tr>
<tr><td>2025-06-07</td><td>SPC to 3D: Novel View Synthesis from Binary SPC via I2I translation</td><td>[2506.06890](http://arxiv.org/pdf/2506.06890)</td><td>◆ 提出首个从二进制单光子相机(SPC)数据生成高质量彩色新视角的两阶段框架，解决了传统3D合成方法无法处理的严重信息丢失问题。  
◆ 第一阶段采用Pix2PixHD等生成模型进行图像到图像转换，将二进制SPC输入转化为可信的RGB图像，有效恢复丢失的纹理和颜色信息。  
◆ 第二阶段结合神经辐射场(NeRF)或高斯泼溅(3DGS)等先进3D重建技术，从生成的RGB图像中合成新视角。  
◆ 通过大量定性和定量实验验证了所提框架(Pix2PixHD + Nerf/3DGS)的优越性，在感知质量和几何一致性上显著超越基线方法。  
◆ 该工作为单光子相机这类新兴成像技术的3D应用开辟了新途径，特别适用于极低光照或超高速成像场景。</td></tr>
<tr><td>2025-06-06</td><td>Splat and Replace: 3D Reconstruction with Repetitive Elements</td><td>[2506.06462](http://arxiv.org/pdf/2506.06462)</td><td>◆ 利用场景中的重复元素提升新视角合成质量，解决了传统NeRF和3DGS在训练视角不足时渲染效果差的问题。  
◆ 提出一种基于3D高斯泼溅（3DGS）的重复实例分割与配准方法，实现不同实例间的信息共享。  
◆ 通过几何优化和外观变化建模，同时提升场景的几何精度和视觉一致性。  
◆ 在合成与真实场景中验证了方法的有效性，显著改善了遮挡和低覆盖区域的渲染效果。  
◆ 首次将重复元素作为先验知识融入3D重建流程，为复杂场景重建提供了新思路。</td></tr>
<tr><td>2025-06-06</td><td>NeurNCD: Novel Class Discovery via Implicit Neural Representation</td><td>[2506.06412](http://arxiv.org/pdf/2506.06412)</td><td>◆ NeurNCD首次提出利用隐式神经表示（Embedding-NeRF模型）替代传统显式3D分割图，通过KL散度聚合语义嵌入和视觉嵌入空间的熵，解决离散化、空洞和噪声问题。  
◆ 结合特征查询、特征调制和聚类等关键组件，实现预训练语义分割网络与隐式神经表示之间的高效特征增强和信息交互。  
◆ 该框架在开放和封闭场景中均实现优越分割性能，无需依赖密集标注数据集进行监督训练或人工生成稀疏标签监督。  
◆ 在NYUv2和Replica数据集上的大量实验表明，NeurNCD显著优于现有最先进方法，验证了其有效性和泛化能力。  
◆ 提出了一种通用且数据高效的新类别发现框架，为开放世界场景中的实际应用提供了新思路。</td></tr>
<tr><td>2025-06-06</td><td>Dy3DGS-SLAM: Monocular 3D Gaussian Splatting SLAM for Dynamic Environments</td><td>[2506.05965](http://arxiv.org/pdf/2506.05965)</td><td>◆ 提出了Dy3DGS-SLAM，这是首个基于单目RGB输入的动态场景3D高斯泼溅SLAM方法，填补了动态环境下纯视觉SLAM的空白。  
◆ 通过概率模型融合光流掩码和深度掩码，生成动态融合掩码，仅需单次网络迭代即可约束跟踪尺度并优化几何渲染。  
◆ 设计了新颖的运动损失函数，基于动态融合掩码约束位姿估计网络，显著提升了动态物体干扰下的跟踪鲁棒性。  
◆ 在映射阶段，结合动态像素的渲染损失、颜色和深度信息，有效消除了动态物体带来的瞬态干扰和遮挡问题。  
◆ 实验证明该方法在动态环境中实现了最先进的跟踪与渲染性能，甚至优于部分RGB-D方法，展现了单目输入的潜力。</td></tr>
<tr><td>2025-06-06</td><td>ProJo4D: Progressive Joint Optimization for Sparse-View Inverse Physics Estimation</td><td>[2506.05317](http://arxiv.org/pdf/2506.05317)</td><td>◆ 提出ProJo4D框架，通过渐进式联合优化策略解决稀疏多视角视频下的物理参数估计问题，克服传统方法因分阶段优化导致的误差累积问题。  
◆ 创新性地引入参数敏感性指导的优化顺序，逐步联合优化几何、外观、物理状态和材料属性，避免直接全参数优化带来的非凸和非可微难题。  
◆ 在PAC-NeRF和Spring-Gaus数据集上的实验表明，该方法在4D未来状态预测、未来状态的新视角渲染和材料参数估计方面均优于现有方法。  
◆ 首次实现稀疏多视角输入下的物理准确数字孪生构建，为机器人和XR应用提供更实用的解决方案。  
◆ 通过渐进式优化策略平衡计算效率与精度，为复杂物理场景的神经渲染与参数估计提供新思路。</td></tr>
<tr><td>2025-06-06</td><td>Unifying Appearance Codes and Bilateral Grids for ...</td><td>[2506.05280](http://arxiv.org/pdf/2506.05280)<br><a href=''>[代码]</a></td><td>◆提出多尺度双边网格新方法，统一了外观编码和双边网格的优势，解决了动态驾驶场景中光度不一致导致的几何失真问题。  
◆通过像素级颜色映射和分层约束优化，显著降低了光不一致产生的漂浮伪影，在四大自...</td></tr>
<tr><td>2025-06-05</td><td>Generating Synthetic Stereo Datasets using 3D Gaus...</td><td>[2506.04908](http://arxiv.org/pdf/2506.04908)</td><td>◆ 提出基于3D高斯泼溅（3DGS）的立体数据集生成流程，相比NeRF方法更高效。  
◆ 结合显式3D重建几何与FoundationStereo模型的深度估计进行专家知识迁移，生成高质量数据。...</td></tr>
<tr><td>**2025-05-30**</td><td>**Hi-Dyna Graph: Hierarchical Dynamic Scene Graph for Robotic Autonomy in Human-Centric Environments**</td><td>[2506.00083](http://arxiv.org/abs/2506.00083)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-29**</td><td>**PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views**</td><td>[2505.23481](http://arxiv.org/abs/2505.23481)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-29**</td><td>**LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering**</td><td>[2505.23158](http://arxiv.org/abs/2505.23158)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**Can NeRFs See without Cameras?**</td><td>[2505.22441](http://arxiv.org/abs/2505.22441)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**Learning Fine-Grained Geometry for Sparse-View Splatting via Cascade Depth Loss**</td><td>[2505.22279](http://arxiv.org/abs/2505.22279)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-28**</td><td>**Hyperspectral Gaussian Splatting**</td><td>[2505.21890](http://arxiv.org/abs/2505.21890)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-27**</td><td>**Structure from Collision**</td><td>[2505.21335](http://arxiv.org/abs/2505.21335)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**OB3D: A New Dataset for Benchmarking Omnidirectional 3D Reconstruction Using Blender**</td><td>[2505.20126](http://arxiv.org/abs/2505.20126)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-30**</td><td>**ErpGS: Equirectangular Image Rendering enhanced with 3D Gaussian Regularization**</td><td>[2505.19883](http://arxiv.org/abs/2505.19883)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**GoLF-NRT: Integrating Global Context and Local Geometry for Few-Shot View Synthesis**</td><td>[2505.19813](http://arxiv.org/abs/2505.19813)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-26**</td><td>**Depth-Guided Bundle Sampling for Efficient Generalizable Neural Radiance Field Reconstruction**</td><td>[2505.19793](http://arxiv.org/abs/2505.19793)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-22**</td><td>**UAV See, UGV Do: Aerial Imagery and Virtual Teach Enabling Zero-Shot Ground Vehicle Repeat**</td><td>[2505.16912](http://arxiv.org/abs/2505.16912)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-19**</td><td>**IPENS:Interactive Unsupervised Framework for Rapid Plant Phenotyping Extraction via NeRF-SAM2 Fusion**</td><td>[2505.13633](http://arxiv.org/abs/2505.13633)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-19**</td><td>**3D Gaussian Adaptive Reconstruction for Fourier Light-Field Microscopy**</td><td>[2505.12875](http://arxiv.org/abs/2505.12875)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-18**</td><td>**Is Semantic SLAM Ready for Embedded Systems ? A Comparative Survey**</td><td>[2505.12384](http://arxiv.org/abs/2505.12384)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-16**</td><td>**MutualNeRF: Improve the Performance of NeRF under Limited Samples with Mutual Information Theory**</td><td>[2505.11386](http://arxiv.org/abs/2505.11386)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-16**</td><td>**EA-3DGS: Efficient and Adaptive 3D Gaussians with Highly Enhanced Quality for outdoor scenes**</td><td>[2505.10787](http://arxiv.org/abs/2505.10787)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-15**</td><td>**Large-Scale Gaussian Splatting SLAM**</td><td>[2505.09915](http://arxiv.org/abs/2505.09915)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-14**</td><td>**Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians**</td><td>[2505.09413](http://arxiv.org/abs/2505.09413)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-14**</td><td>**FreeDriveRF: Monocular RGB Dynamic NeRF without Poses for Autonomous Driving via Point-Level Dynamic-Static Decoupling**</td><td>[2505.09406](http://arxiv.org/abs/2505.09406)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-12**</td><td>**TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian**</td><td>[2505.08811](http://arxiv.org/abs/2505.08811)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-13**</td><td>**FOCI: Trajectory Optimization on Gaussian Splats**</td><td>[2505.08510](http://arxiv.org/abs/2505.08510)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-13**</td><td>**TUM2TWIN: Introducing the Large-Scale Multimodal Urban Digital Twin Benchmark Dataset**</td><td>[2505.07396](http://arxiv.org/abs/2505.07396)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-12**</td><td>**Geometric Prior-Guided Neural Implicit Surface Reconstruction in the Wild**</td><td>[2505.07373](http://arxiv.org/abs/2505.07373)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-11**</td><td>**NeuGen: Amplifying the &#x27;Neural&#x27; in Neural Radiance Fields for Domain Generalization**</td><td>[2505.06894](http://arxiv.org/abs/2505.06894)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-10**</td><td>**3D Characterization of Smoke Plume Dispersion Using Multi-View Drone Swarm**</td><td>[2505.06638](http://arxiv.org/abs/2505.06638)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-10**</td><td>**FlexNeRFer: A Multi-Dataflow, Adaptive Sparsity-Aware Accelerator for On-Device NeRF Rendering**</td><td>[2505.06504](http://arxiv.org/abs/2505.06504)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-08**</td><td>**3D Scene Generation: A Survey**</td><td>[2505.05474](http://arxiv.org/abs/2505.05474)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-04**</td><td>**HandOcc: NeRF-based Hand Rendering with Occupancy Networks**</td><td>[2505.02079](http://arxiv.org/abs/2505.02079)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-04**</td><td>**Learning Heterogeneous Mixture of Scene Experts for Large-scale Neural Radiance Fields**</td><td>[2505.02005](http://arxiv.org/abs/2505.02005)<br><a href=''>[代码]</a></td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-03**</td><td>**AquaGS: Fast Underwater Scene Reconstruction with SfM-Free Gaussian Splatting**</td><td>[2505.01799](http://arxiv.org/abs/2505.01799)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-03**</td><td>**Unified Steganography via Implicit Neural Representation**</td><td>[2505.01749](http://arxiv.org/abs/2505.01749)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-01**</td><td>**Cues3D: Unleashing the Power of Sole NeRF for Consistent and Unique Instances in Open-Vocabulary 3D Panoptic Segmentation**</td><td>[2505.00378](http://arxiv.org/abs/2505.00378)</td><td>摘要生成中...</td></tr>
<tr><td>**2025-05-01**</td><td>**GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting**</td><td>[2504.20379](http://arxiv.org/abs/2504.20379)</td><td>摘要生成中...</td></tr>
</tbody>
</table>
</div>

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2025.07.04
