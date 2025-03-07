# DualAttendMed: A Dual-Stage Attention Approach for Disease Localization and Classification in Medical Images

## Overview
Welcome to the official GitHub repository for **DualAttendMed**, a cutting-edge dual-stage attention framework developed to improve disease localization and classification in medical images. This project, authored by Junaid Abbas, Danyal Badar, and Liu Li from Chongqing University, addresses critical challenges in medical image analysis, particularly for diabetic retinopathy (DR) detection using retinal fundus images. The framework integrates advanced techniques such as ResNet-152, Channel Attention, Attention-Assisted Data Augmentation (AADA), and Bilinear Attention Pooling (BAP), achieving state-of-the-art results.

## Paper
The detailed methodology, experiments, and results are documented in the accompanying paper:
- **Title:** DualAttendMed: A Dual-Stage Attention Approach for Disease Localization and Classification in Medical Images
- **Authors:** Junaid Abbas¹, Danyal Badar², Liu Li¹
- **Affiliations:** 
  - ¹School of Big Data and Software Engineering, Chongqing University
  - ²College of Computer Science, Chongqing University
- **Date:** March 2025
- **Abstract:** Accurately diagnosing medical images is crucial in healthcare, but challenges like precise localization, imbalanced class distributions, and complex disease patterns limit accuracy. DualAttendMed introduces a novel dual-stage attention framework, utilizing ResNet-152 for feature extraction, Channel Attention for refining disease-relevant regions, and AADA for interpretable attention maps. Validated on APTOS, DDR, and Messidor-2 datasets, it achieves classification accuracies of 92.50%, 87.10%, and 88.70%, with IoU values of 0.85, 0.80, and 0.75, enhancing clinical confidence and early detection.
- **Note:** The paper is currently under review. Upon acceptance, the full code and additional resources will be made publicly available in this repository.

## Features
- **Dual-Stage Attention:** Iteratively refines focus on diagnostically relevant regions.
- **Attention-Assisted Data Augmentation (AADA):** Dynamically augments data to address class imbalance and enhance generalization.
- **Composite Loss Function:** Balances classification accuracy, attention alignment, and feature diversity.
- **Clinically Interpretable Outputs:** Provides visual explanations rated 4.7/5 by experts.
- **State-of-the-Art Performance:** Outperforms leading CNN, attention, and hybrid methods on retinal fundus datasets.

## Installation
### Prerequisites
- Python 3.8+
- PyTorch 1.6+
- Other dependencies: `numpy`, `torchvision`, `matplotlib`, `scikit-learn`
- **Note:** Detailed installation instructions and code will be provided after the paper is accepted and published.

## Upcoming Content
- **Code:** Full implementation in PyTorch will be released post-acceptance.
- **Datasets:** Preparation guidelines for APTOS, Messidor-2, and DDR datasets.
- **Documentation:** Comprehensive usage guides, training, and visualization scripts.

## Contributing
We welcome feedback and collaboration! Please feel free to open issues for discussions. Detailed contribution guidelines will be added after code release.

## License
This project will be licensed under the [MIT License](LICENSE) upon code publication - see the `LICENSE` file (to be added) for details.

## Acknowledgments
- The authors thank Chongqing University for research support.
- Special thanks to the ophthalmologists who provided clinical feedback.

## Contact
For questions or collaborations, contact:
- Junaid Abbas: [email@example.com]
- Danyal Badar: [email@example.com]
- Liu Li: [email@example.com]

## Future Work
- Extend DualAttendMed to other imaging modalities (e.g., X-rays, MRIs).
- Refine multi-lesion attention fusion techniques.
- Broaden validation across diverse populations.
