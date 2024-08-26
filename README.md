# DTCLMapper
DTCLMapper: Dual Temporal Consistent Learning for Vectorized HD Map Construction [[PDF]](https://arxiv.org/pdf/2405.05518), IEEE T-ITS, 2024.

## Motivation
<div align=center>
<img src="https://github.com/lynn-yu/DTCLMapper/blob/main/framework.png" >
</div>


### Abstract
Temporal information plays a pivotal role in Bird’sEye-View (BEV) driving scene understanding, which can alleviate the visual information sparsity. However, the indiscriminate temporal fusion method will cause the barrier of feature redundancy
when constructing vectorized High-Definition (HD) maps. In this paper, we revisit the temporal fusion of vectorized HD maps, focusing on temporal instance consistency and temporal map consistency learning. To improve the representation of
instances in single-frame maps, we introduce a novel method, DTCLMapper. This approach uses a dual-stream temporal consistency learning module that combines instance embedding with geometry maps. In the instance embedding component, our approach integrates temporal Instance Consistency Learning (ICL), ensuring consistency from vector points and instance features aggregated from points. A vectorized points pre-selection module is employed to enhance the regression efficiency of vector points from each instance. Then aggregated instance features obtained from the vectorized points preselection module are grounded in contrastive learning to realize temporal consistency, where positive and negative samples are selected based on position and semantic information. The geometry mapping component introduces Map Consistency Learning (MCL) designed with self-supervised learning. The MCL enhances the generalization capability of our consistent learning approach by concentrating on the global location and distribution constraints of the instances.

### Update

2023.05 Init repository.
2024.08 Update

### Data
Download  [nuScenes](https://www.nuscenes.org/) and [Argoverse2](https://www.argoverse.org/). And change the dataset path in the code.

### Environment
Following install.md

### Training
./tools/dist_train.sh --config ./projects/configs/DTCLMapper/v2maptr_tiny_r50_24e.py 

### Acknowledgement
The code framework of this project is based on ![MapTR](https://github.com/hustvl/MapTR), thanks to this excellent work.

### Contact

Feel free to contact me if you have additional questions or have interests in collaboration. Please drop me an email at  lsynn@hnu.edu.cn

### 🤝 Publication
Please consider referencing this paper if you use the ```code``` from our work.
Thanks a lot :)

```
@article{li2024dtclmapper,
  title={DTCLMapper: Dual Temporal Consistent Learning for Vectorized HD Map Construction},
  author={Li, Siyu and Lin, Jiacheng and Shi, Hao and Zhang, Jiaming and Wang, Song and Yao, You and Li, Zhiyong and Yang, Kailun},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024}
}
```
