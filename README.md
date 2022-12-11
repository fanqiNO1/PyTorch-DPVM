# PyTorch Implementation of Deep Photometric Visibility Metric (DPVM)

This is a PyTorch implementation of the [Predicting Visible Image Differences Under Varying Display Brightness and Viewing Distance](https://openaccess.thecvf.com/content_CVPR_2019/html/Ye_Predicting_Visible_Image_Differences_Under_Varying_Display_Brightness_and_Viewing_CVPR_2019_paper.html) in CVPR 2019. The DPVM can be used as image quality metrics or be used for evaluating compressed image qualities.

## Code Usage:

`python example.py --ref reference.png --dist distorted.png --ppd 60 --lumin 110 --load_path path/to/model.pth --output output.png`

The trained model weights are available [here](https://drive.google.com/file/d/1ikCXlb_QxytuZ32pU2aOOQsFY9aEOwP_/view?usp=share_link).

## Sample Results:

The reference image is:

<img src="asserts/ref.png" width="480px">

And the distorted image is:

<img src="asserts/dis.png" width="480px">

The prediction is:

<img src="asserts/pred_color.png" width="480px">

The image is from `marking_aliasing/Attic_Aliasing_l1_aliasing`. The ppd is 40 and the luminance is 110.


## Notes:
We have implemented network architecture used in the paper. In the future, we will migrate the loss function from tensorflow to PyTorch. This can serve as a basic implementation of the paper and a footfold for future works.

## Citation:
If you find any part of this code useful, please cite the paper.

 
    @inproceedings{photometricnn2019,
        title={Predicting visible image differences under varying display brightness and viewing distance},
        author={N. Ye and K. Wolski and R. K. Mantiuk},
        booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition 2019 (CVPR 2019)}, 
        year={2019}
        keywords={Photometric visibility metric, Deep learning},  
    }
