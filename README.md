# SLAM领域最新论文 (2025.07.23)

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
> 更新于: 2025.07.23
