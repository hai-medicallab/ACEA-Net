
	


## ACEA-Net: Weakly Supervised Prostate 3D MRI Image Segmentation via Advanced Prompt Points

<div align="center"> 
</div>
<div style="text-align:center;">
  <img src="/acea-net/figs/ACEA-Net.png" style="max-width:60%;">
</div>

## Requirements
We have only tested in the following environments. Please ensure that the version of each package is not lower than that listed below. 
* Set up a virtual environment (e.g. conda or virtualenv) with Python == 3.8.10
* Follow official guidance to install [Pytorch][torch_link] with torch == 1.9.1+cu111
* Install other requirements using: 
```bash
pip install -r requirements.txt
```

[torch_link]: https://pytorch.org/


## Usage

### The Stage of ACEANet: Learning from SeedGeo

Execute the script to expand the annotation seeds based on seed-geodesic distance transform: 
```bash
# VS_MSD-not-spcaing dataset
python ./data/VS_MSD-not-spcaing/generate_geodesic_labels.py \
--dataset_split ./splits/split_VS_MSD.csv \
--path_images ./data/VS_MSD-not-spcaing/image_crop/ \
--image_postfix T2 \
--path_labels ./data/VS_MSD-not-spcaing/label_crop/ \
--label_postfix Label \
--path_anno_7points ./data/VS_MSD-not-spcaing/annotation_7points/ \
--anno_7points_postfix 7points \
--path_geodesic ./data/VS_MSD-not-spcaing/geodesic/ \
--geodesic_weight 0.5 \
--geodesic_threshold 0.2


```

Execute the script to train an initial model using the expanded seeds.
```bash
# VS_MSD-not-spcaing dataset
python train_gatedcrfloss3d22d_multiview_varianceloss.py \
--model_dir ./models/VS_MSD-not-spcaing/gatedcrfloss3d22d_multiview_varianceloss-0.31-unet1-0.3075-unt-0.3125-unet-0.31-unet-0.31-unet1-0.305-unet-0.315-unet-0.32-unet-0.31-unet-0.5-0.275-0.25-unet-0.3-unet-0.325-unet-0.5-0.325-0.5-0.32-0.5-0.33-0.5-0.34-0.5-0.36-0.5-0.25-0.5-0.4-0.5-0.35-0.5-0.2-0.5-0.3/ \
--network U_Net2D5 \
--batch_size 1 \
--max_epochs 300 \
--rampup_epochs 30 \
--dataset_split ./splits/split_VS_MSD.csv \
--path_images ./data/VS_MSD-not-spcaing/image_crop/ \
--image_postfix T2 \
--path_labels ./data/VS_MSD-not-spcaing/label_crop/ \
--label_postfix Label \
--path_geodesic_labels ./data/VS_MSD-not-spcaing/geodesic/weight0.5_threshold0.2/geodesic_label/ \
--geodesic_label_postfix GeodesicLabel \
--learning_rate 1e-2 \
--spatial_shape 128 128 48 \
--weight_gatedcrf 1.0 \
--down_size 64 64 48 \
--kernel_radius 5 5 -1 \
--weight_variance 0.1


```

Perform inference on the test dataset using `inference.py` to obtain segmentation results. Then, execute `utilities/scores.py` to obtain evaluation metrics such as "dice" and "assd" for the segmentation results.
```bash
# VS_MSD-not-spcaing dataset
python inference.py \
--model_dir ./models/VS_MSD-not-spcaing/gatedcrfloss3d22d_multiview_varianceloss-0.31-unet1-0.3075-unt-0.3125-unet-0.31-unet-0.31-unet1-0.305-unet-0.315-unet-0.32-unet-0.31-unet-0.5-0.275-0.25-unet-0.3-unet-0.325-unet-0.5-0.325-0.5-0.32-0.5-0.33-0.5-0.34-0.5-0.36-0.5-0.25-0.5-0.4-0.5-0.35-0.5-0.2-0.5-0.3/ \
--network U_Net2D5 \
--dataset_split ./splits/split_VS_MSD.csv \
--path_images ./data/VS_MSD-not-spcaing/image_crop/ \
--image_postfix T2 \
--phase inference \
--spatial_shape 128 128 48 \
--epoch_inf best

python utilities/scores.py \
--model_dir ./models/VS_MSD-not-spcaing/gatedcrfloss3d22d_multiview_varianceloss-0.31-unet1-0.3075-unt-0.3125-unet-0.31-unet-0.31-unet1-0.305-unet-0.315-unet-0.32-unet-0.31-unet-0.5-0.275-0.25-unet-0.3-unet-0.325-unet-0.5-0.325-0.5-0.32-0.5-0.33-0.5-0.34-0.5-0.36-0.5-0.25-0.5-0.4-0.5-0.35-0.5-0.2-0.5-0.3/ \
--network U_Net2D5 \
--dataset_split ./splits/split_VS_MSD.csv \
--image_postfix T2 \
--phase inference \
--path_labels ./data/VS_MSD-not-spcaing/label_crop/ \
--label_postfix Label


```



## Acknowledgement
This code is adapted from [InExtremIS](https://github.com/ReubenDo/InExtremIS). We thank Dr. Reuben Dorent for his elegant and efficient code base.

## Citation
```
@ARTICLE{10893699,
  author={Zou, Jie and Huang, Mengxing and Zhang, Yu and Zhang, Zhiyuan and Zhou, Wenjie and Bhatti, Uzair Aslam and Chen, Jing and Bai, Zhiming},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={ACEA-Net: Weakly Supervised Prostate 3D MRI Image Segmentation via Advanced Prompt Points}, 
  year={2025},
  volume={},
  number={},
  pages={1-12},
  keywords={Annotations;Three-dimensional displays;Noise measurement;Noise;Magnetic resonance imaging;Image edge detection;Accuracy;Training;Semantic segmentation;Labeling;Weakly Supervised Segmentation;Geodetic Distance Transform;Pseudo-label Generation;Learning with Noisyt Label},
  doi={10.1109/JBHI.2025.3543444}}
```