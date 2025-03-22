# ML4SCI_DeepLense_Tests

## Contact Information
- **Name**:     Gokul M K
- **Email**:    ed21b026@smail.iitm.ac.in
- **Github**:   [gokulmk-12](https://github.com/gokulmk-12)
- **Website**:  [Portfolio](https://gokulmk-12.github.io/)
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
        <tr><th>Model</th><th>Accuracies</th></tr>
        <tr><td>AlexNet</td><td>List all new or modified files</td></tr>
        <tr><td>VGGNet</td><td>Show file differences that haven't been staged</td></tr>
        <tr><td>ResNet-18</td><td>Show file differences that haven't been staged</td></tr>
        <tr><td>ViT-b-16</td><td>Show file differences that haven't been staged</td></tr>
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

ResNet-18 with pretrained weights achieved the best results with a training accuracy of **99.04%** with 30 iterations of training (18 minutes 30 seconds)

![roc_auc](https://github.com/user-attachments/assets/36e3d7bf-e554-414f-a754-a43235347099)

## 2) Specific Test: Diffusion Models
- **Goal**: A generative model to simulate realistic strong gravitational lensing images
- **Evaluation**: FID
- **Library**: torch, torchvision, sklearn
- **Specific Reference**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)
  
### Plan & Results

## 3) Specific Test: Image Super Resolution
- **Goal**: A deep learning model to upscale low-resolution strong lensing images
- **Evaluation**: MSE, SSIM, PSNR
- **Library**: torch, torchvision, sklearn
- **Specific Reference**: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155)

### Plan & Results
I implemented the architecture recommended in the reference paper. The loss function comprises three components: **per-pixel MSE loss** (weight: 100), **feature reconstruction loss** (weight: 100), and **total variation loss** (weight: 1e-7). When using only the MSE loss, the generated images appeared blurred. Incorporating the feature reconstruction loss with a pretrained VGG network resulted in sharper and cleaner images while preserving PSNR and SSIM. Additionally, total variation loss was applied to eliminate unwanted noise introduced during prediction.

| Hyperparameters | Values |
| --- | --- |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Batch Size | 16 |
| Iterations | 30 |

![task_3a_result](https://github.com/user-attachments/assets/65eb822e-bb69-4e2d-a429-26317206bf78)




