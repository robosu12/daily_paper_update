# SLAM领域最新论文 (2025.07.26)

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
</ol>
</details>

<h2 id='lidar-slam'>LiDAR SLAM</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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
</tbody>
</table>
</div>

<div align='right'><a href='#top'>↑ 返回顶部</a></div>

<h2 id='loop-closure'>Loop Closure</h2>

<div class="table-container">
<table>
<thead><tr><th>日期</th><th>标题</th><th>论文与代码</th><th>摘要</th></tr></thead>
<tbody>
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
<tr><td>2025-07-11</td><td>Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection</td><td>[2507.07994](http://arxiv.org/pdf/2507.07994)</td><td>◆ 提出首个基于草图的无源少样本关键点检测框架，利用人类手绘草图作为跨模态监督信号，解决了传统方法在查询数据分布未知时的局限性。  
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

---
> 本列表自动生成 | [反馈问题](https://github.com/your-repo/issues)
> 更新于: 2025.07.26
