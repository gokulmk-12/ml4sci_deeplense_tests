# ML4SCI_DeepLense_Tests

## Contact Information
- **Name**    : Gokul M K
- **Email**   : ed21b026@smail.iitm.ac.in
- **Github**  : [gokulmk-12](https://github.com/gokulmk-12)
- **Website** : [Portfolio](https://gokulmk-12.github.io/)
- **LinkedIn**: [gokul-m-k](https://www.linkedin.com/in/gokul-m-k-886a93263/)
- **Location**: Chennai, Tamil Nadu, India
- **Timezone**: IST (UTC +5:30)

## Education Details
- **University**: Indian Institute of Technology Madras
- **Degree**: Bachelors in Engineering Design (B.Tech), Dual Degree in Robotics (IDDD)
- **Major**: Robotics and Artificial Intelligence
- **Expected Graduation**: May, 2026

## Background
Hi, I’m Gokul, a Dual Degree student in Engineering Design with a specialization in Robotics at the Indian Institute of Technology, Madras. I’m interested in developing Learning and Control based solutions for Robotic Challenges. To achieve this, I have gathered knowledge in Reinforcement Learning, Control Engineering, Deep Learning and Generative AI. I am proficient in **C**, **C++**, **Python**, and **MATLAB**, along with frameworks such as **PyTorch**, **Torch Lightning**, **Keras**, **TensorFlow**, and **JAX**. Lately, I have been delving into astrophysics, exploring topics like gravitational lensing of massive stars and galaxies. My strong foundation in Deep Learning, combined with my interest in astrophysics, has led me to the DeepLense Project.

# Test Details

## 1) Common Test: Multi-Class Classification
- **Goal**: Classify the strong lensing images with no substructure, subhalo substructure, vortex substructure
- **Evaluation**: ROC Curve, AUC
- **Library**: torch, torchvision, sklearn
- **Specific Reference**: [Deep Learning the Morphology of Dark Matter Substructure](https://iopscience.iop.org/article/10.3847/1538-4357/ab7925)

### Plan & Results
My plan was to use state-of-the-art image classification neural networks pretrained on imagenet. The table below provides the list of models used, and their corresponding hyperparameters and accuracies.

<table>
  <tr>
    <td>
      <table>
        <tr><th>Model</th><th>Training Accuracy</th></tr>
        <tr><td>AlexNet</td><td>32.79 %</td></tr>
        <tr><td>VGGNet-16</td><td>34.56 %</td></tr>
        <tr><td>ResNet-18</td><td>99.04 %</td></tr>
        <tr><td>EfficientNet-B7</td><td>94.86 %</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>Learning Rate</td><td>0.0001</td></tr>
        <tr><td>Optimizer</td><td>Adam</td></tr>
        <tr><td>Batch Size</td><td>32</td></tr>
        <tr><td>Iterations</td><td>30</td></tr>
      </table>
    </td>
  </tr>
</table>

ResNet-18 with weights pretrained on ImageNet achieved the best results with a training accuracy of **99.04%** with 30 iterations of training (18 minutes 30 seconds)

![roc_auc](https://github.com/user-attachments/assets/36e3d7bf-e554-414f-a754-a43235347099)

## 2) Specific Test: Image Super Resolution
- **Goal**: A deep learning model to upscale low-resolution strong lensing images
- **Evaluation**: MSE, SSIM, PSNR
- **Library**: torch, torchvision, sklearn
- **Specific Reference**: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155)

### Plan & Results
#### Task 3A
I implemented the architecture recommended in the above reference paper. The loss function comprises 3 components: **per-pixel MSE loss** (weight: 100), **feature reconstruction loss** (weight: 100), and **total variation loss** (weight: 1e-7). When using only MSE loss, the generated images appeared blurred. Incorporating the feature reconstruction loss with a pretrained VGG network resulted in sharper and cleaner images while preserving PSNR and SSIM. Additionally, total variation loss was applied to eliminate unwanted noise introduced during prediction.

<table>
  <tr>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>Per-pixel Loss Weight</td><td>100</td></tr>
        <tr><td>Feature Reconstruction Loss Weight</td><td>100</td></tr>
        <tr><td>Total Variation Loss Weight</td><td>1e-7</td></tr>
        <tr><td>Train-Test Split</td><td>90:10</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>Learning Rate</td><td>0.0001</td></tr>
        <tr><td>Optimizer</td><td>Adam</td></tr>
        <tr><td>Batch Size</td><td>16</td></tr>
        <tr><td>Iterations</td><td>30</td></tr>
      </table>
    </td>
  </tr>
</table>

![task_3a_result](https://github.com/user-attachments/assets/65eb822e-bb69-4e2d-a429-26317206bf78)

![task_3a_eval](https://github.com/user-attachments/assets/9dae4256-876b-4657-96f3-6f6b7a84e30c)

#### Task 3B
This task focused on training a super-resolution neural network with a limited dataset. I applied transfer learning using the model from Task 3A, incorporating a few residual blocks along with downsampling and upsampling layers to align the input and output image dimensions. During training, the base model's weights (from Task 3A) were kept frozen. Data augmentation had little impact on reducing the loss. Most parameters remained unchanged, except for the number of training iterations.

<table>
  <tr>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>Per-pixel Loss Weight</td><td>100</td></tr>
        <tr><td>Feature Reconstruction Loss Weight</td><td>100</td></tr>
        <tr><td>Total Variation Loss Weight</td><td>1e-7</td></tr>
        <tr><td>Train-Test Split</td><td>90:10</td></tr>
      </table>
    </td>
    <td>
      <table>
        <tr><th>Hyperparameter</th><th>Value</th></tr>
        <tr><td>Learning Rate</td><td>0.0001</td></tr>
        <tr><td>Optimizer</td><td>Adam</td></tr>
        <tr><td>Batch Size</td><td>16</td></tr>
        <tr><td>Iterations</td><td>100</td></tr>
      </table>
    </td>
  </tr>
</table>

![task_3b_result](https://github.com/user-attachments/assets/9e083f78-620d-40ed-b834-4db33d2bacd3)

![task_3b_eval](https://github.com/user-attachments/assets/b2085ec8-c90f-42e1-bde9-6e9c070dc069)

## 3) Specific Test: Diffusion Models
- **Goal**: A generative model to simulate realistic strong gravitational lensing images
- **Evaluation**: FID
- **Library**: torch, torchvision, sklearn
- **Specific Reference**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)
  
### Plan & Results

## 4) Specific Test: Lens Finding
- **Goal**: A model to identify lenses from the given image.
- **Evaluation**: ROC, AUC Curve
- **Library**: torch, torchvision, sklearn
- **Specific Reference**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)
  
### Plan & Results



