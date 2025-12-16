# Residual Networks for Fine-Grained Image Classification

This repository contains a comprehensive project exploring deep learning techniques for fine-grained image classification, developed for the Multimedia Information Retrieval and Computer Vision course at the University of Pisa.

The core of the project is a comparative study based on the ResNet architecture, systematically applying and evaluating a series of techniques on the **Oxford-IIIT Pet dataset**. The notebook covers:

*   Training a ResNet18 from scratch.
*   Fine-tuning with pre-trained ImageNet weights (Transfer Learning).
*   Knowledge Distillation using a ResNet34 teacher model to improve a ResNet18 student.
*   Model compression techniques for deployment (Post-Training Quantization and Iterative Pruning).
*   Model interpretation and visualisation (t-SNE, filter analysis, confusion matrices).

The notebook can be easily opened and run in Google Colab, via the link at the beginning of the notebook itself, reported here for convenience: https://colab.research.google.com/github/gabrielebilliciani/ResNet-Oxford-Pet/blob/main/CV_project.ipynb. The Colab notebook is already set to use a GPU runtime for faster training.

## Summary of Results

The experiments demonstrate a clear and effective progression of techniques, with the final model combining fine-tuning and knowledge distillation to achieve the best performance.

| Method                    | Architecture    | Test Accuracy (%) | Key Takeaway                                                    |
| :------------------------ | :-------------- | :---------------- | :-------------------------------------------------------------- |
| Baseline (From Scratch)   | ResNet18        | 78.00%            | Shows the difficulty of training from scratch on this dataset.  |
| Fine-Tuning               | ResNet18        | 89.62%            | Transfer learning provides the single biggest performance leap. |
| Teacher Model             | ResNet34        | 91.28%            | Establishes a strong performance ceiling for distillation.      |
| KD (From Scratch)         | ResNet18        | 86.21%            | A powerful alternative when pre-trained weights are unavailable.|
| **Fine-Tuning + KD**      | **ResNet18**    | **90.49%**        | **Optimal Performance:** Combining techniques yields the best result.|
| Quantisation (PTQ)        | ResNet18 (INT8) | 90.38%            | **Most Practical:** Near-optimal accuracy with major efficiency gains.|
| Pruning (Unstructured)    | ResNet18        | (various)         | High sparsity is achievable but with limited practical benefits.  |
