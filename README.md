

# Dynamic PET Image Prediction Using a Network Combining Reversible and Irreversible Modules    
Jie Sun, Qian Xia, Chuanfu Sun, Yumei Chen, Huafeng Liu, Wentao Zhu, Qiegen Liu   

Abstract:      
Dynamic positron emission tomography (PET) images can reveal the distribution of tracers in the organism and the dynamic processes involved in biochemical reactions, and it is widely used in clinical practice. Dynamic PET imaging is useful in analyzing the kinetics and metabolic processes of radiotracers. Prolonged scan times can cause discomfort for both patients and medical personnel. This study proposes a frame prediction method for dynamic PET imaging, reducing PET scanning time by applying a multi-module deep learning framework composed of reversible and irreversible modules. The network can predict kinetic parameter images based on the earlier frames of dynamic PET images, and then generate complete dynamic PET images. In the experiments with simulated data and the generalization experiments with clinical data, the dynamic PET images predicted by our network have higher SSIM and PSNR and lower MSE than its counterparts. The generalization performance of this network in clinical data experiments indicates that the proposed method has potential in the application of dynamic PET.     

## The training pipeline of M²-Net

 ![fig1](https://github.com/yqx7150/MM-Net/blob/main/fig/fig1.jpg)

## The detailed architecture of M²-Net

 ![fig2](https://github.com/yqx7150/MM-Net/blob/main/fig/fig2.jpg)

## Visualization results of several comparison methods

 ![fig5](https://github.com/yqx7150/MM-Net/blob/main/fig/fig5.jpg)


# Dataset

You need to prepare at least one type of dynamic PET data, as shown in the "data" folder, which includes "CP", "test", and "train". The "train" and "test" folders contain the original noise-free data "fdg_3D", as well as the data with added noise "fdg_3D_noise", the kinetic parameters k1, k2, k3, k4, and the corresponding ki and vb. Similarly, you need to make references at the corresponding positions in the training file. For example, at the "--root1" position, you need to fill in the address of the training dataset, and at the "--root2" position, you need to fill in the address of the test dataset.

#  Train

```python
python train.fdg.py 
```

##  resume training:

To fine-tune a pre-trained model, or resume the previous training, use the --resume flag

# Test

Set CP_PATH, root2, sampling_intervals, and ckpt in test_fdg.py to the corresponding CP address, test dataset address, sampling protocol, and the address of the model to be tested. The model address will be automatically generated during training.
Set os.environ['CUDA_VISIBLE_DEVICES'] = "2" to the graphics card you want to select.

```python
# Test the generated images (Img) and k values.
python test_fdg_img_k.py 
# Test the generated ki and vb values.
python test_fdg_ki_vb.py 
```

# FMZ
Set up FMZ in the same way as the above FDG settings, and at least change the file names.


### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/algorithm-overview.png" width = "800" height = "500"> </div>
 Some examples of invertible and variable augmented network: IVNAC, VAN-ICC, iVAN and DTS-INN.          
           
     
  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
 * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)         
 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    
  * Variable Augmented Network for Invertible Decolorization (基于辅助变量增强的可逆彩色图像灰度化)  [<font size=5>**[Paper]**</font>](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT221205?viewType=HTML)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VA-IDN)        
  * Synthetic CT Generation via Variant Invertible Network for Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10666843)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)      
  * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/IVPSQA/)
  * Temporal Image Sequence Separation in Dual-tracer Dynamic PET with an Invertible Network [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10542421)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DTS-INN/)
