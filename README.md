# A3DCF: Robust Visual Object Tracking via Adaptive Attribute-Aware Discriminative Correlation Filters

Official implementation of the paper Robust Visual Object Tracking via Adaptive Attribute-Aware Discriminative Correlation Filters

## [Download the Paper](https://ieeexplore.ieee.org/abstract/document/9318537/)
>@article{zhu2021robust,
  title={Robust Visual Object Tracking via Adaptive Attribute-Aware Discriminative Correlation Filters},
  author={Zhu, Xue-Feng and Wu, Xiao-Jun and Xu, Tianyang and Feng, Zhenhua and Kittler, Josef},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}

![image](https://github.com/xuefeng-zhu5/A3DCF/tree/master/img.png)

#### Dependencies:
- [PDollar Toolbox](https://pdollar.github.io/toolbox)
- mtimesx (https://github.com/martin-danelljan/ECO/tree/master/external_libs/mtimesx)
- mexResize (https://github.com/martin-danelljan/ECO/tree/master/external_libs/mexResize) 
- MatConvNet. Please download the latest [MatConvNet](http://www.vlfeat.org/matconvnet/) in './tracker_exter/matconvnet' (Set 'opts.enableGpu = true' in 'matconvnet/matlab/vl_compilenn.m'). Please download the [ResNet50](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) and put it in /path/to/AAADCF/tracker_featu/networks/.

Installation: Run install.m file to compile the libraries and download networks.

test: run_demo.m



#### References:
- [1] Henriques, João F., et al. "High-speed tracking with kernelized correlation filters." IEEE Transactions on Pattern Analysis and Machine Intelligence 37.3 (2015): 583-596.
- [2] Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." Computer Vision and Pattern Recognition, 2005. CVPR 2005. 
- [3] Van De Weijer, Joost, et al. "Learning color names for real-world applications." IEEE Transactions on Image Processing 18.7 (2009): 1512-1523.
- [4] Danelljan Martin, et al. "Eco: Efficient convolution operators for tracking." Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [5] Xu T, Feng Z H, Wu X J, et al. Joint group feature selection and discriminative filter learning for robust visual object tracking[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 7950-7960.

#### Raw Results:[Google](https://drive.google.com/drive/folders/1ghxZ6cRgMoMqd2plBhd1D6tiQ2YP3r6j?usp=sharing)
OTB100, UAV123, GOT10K, LaSOT, TrackingNet, DTB70, VOT2018.

