# DeepBag: CNN Bagging Ensemble for Fashion-MNIST Classification

An end-to-end deep learning project that applies **Bootstrap Aggregation (Bagging)** to Convolutional Neural Networks (CNNs) for image classification. DeepBag compares a single CNN baseline against an ensemble of three independently trained CNNs and provides an interactive Streamlit application for image predictions.

## Overview

Deep learning models can make different prediction errors even when trained on the same dataset. **Bagging** reduces the impact of individual model errors by training multiple models on bootstrap samples and combining their predictions.

In this project, three CNN classifiers are trained on different bootstrap samples of the Fashion-MNIST training set. Their predicted class probabilities are averaged to generate the final ensemble prediction.


live deployed: [https://bootstrap-aggregation-with-cnnsclassifier-3bnoeiqwgvgpg9px9npl.streamlit.app/]
### Key Features

- Fashion-MNIST image classification across 10 clothing categories.
- CNN baseline for single-model performance comparison.
- Bootstrap sampling **with replacement** to train three CNN models.
- Soft-voting ensemble using averaged class probabilities.
- Evaluation on a held-out test set.
- Individual CNN and ensemble prediction comparison.
- Interactive Streamlit interface for image uploads and predictions.
- Visual confidence and class-probability comparisons.

## Results

<img width="963" height="322" alt="Screenshot 2026-10-18 063254" src="https://github.com/user-attachments/assets/fd59d485-c629-4351-b212-e87fe627309a" />

<img width="702" height="628" alt="Screenshot 2026-10-18 063437" src="https://github.com/user-attachments/assets/37df2b1c-b42d-4d22-b567-cdf3f2c6e304" />




The following results were obtained during the experiment:

| Model | Test Accuracy |
|---|---:|
| CNN 1 | 90.96% |
| CNN Bagging Ensemble (3 CNNs) | **91.57%** |
| Absolute improvement over the best individual CNN | **+0.61 percentage points** |

The ensemble achieved a 91.57% test accuracy, compared with 90.96% for the best individual CNN.

The corresponding classification error decreased from 9.04% to 8.43%, representing an approximately **6.78% relative error reduction**.

*These results are specific to this experiment and training configuration. Performance can vary with random seeds, bootstrap samples, model initialization and training settings.*

## Dataset

**Fashion-MNIST** is a dataset of grayscale clothing images designed as a classification benchmark.

- Training images: 60,000
- Test images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10
- Image type: Grayscale

The training set was divided into 55,000 training images and 5,000 validation images. The official test set was reserved for final evaluation.

The dataset is loaded directly through TensorFlow/Keras, so a separate manual download is not required.

**Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.

## Methodology

### 1. Data preprocessing

- Load Fashion-MNIST using TensorFlow/Keras.
- Normalize pixel values from the range 0–255 to 0–1.
- Reshape images to include the single grayscale channel.
- Create separate training and validation sets.
- Keep the test set separate for final evaluation.

### 2. CNN baseline

Train a single CNN to establish a reference performance before introducing bagging.

### 3. Bootstrap sampling

Generate three bootstrap datasets by sampling training examples **with replacement**. Each CNN is trained on its own sampled dataset, allowing the models to learn from different training distributions.

### 4. CNN architecture

Each CNN uses the following architecture:

- Conv2D — 32 filters, 3 × 3 kernel, ReLU activation
- MaxPooling2D
- Conv2D — 64 filters, 3 × 3 kernel, ReLU activation
- MaxPooling2D
- Flatten
- Dense — 128 units, ReLU activation
- Dropout — 0.3
- Dense — 10 units, Softmax activation

**Training configuration:** Adam optimizer and sparse categorical cross-entropy loss.

### 5. Soft-voting ensemble

Each CNN outputs a probability distribution over the 10 classes. The ensemble averages the three probability distributions:

\[
P_{\text{ensemble}}(y\mid x)
=\frac{1}{3}\sum_{i=1}^{3}P_i(y\mid x)
\]

The class with the highest averaged probability becomes the final prediction.

### 6. Evaluation

Evaluate the individual CNNs and ensemble on the held-out test set. Compare their accuracies to assess whether combining the models improves generalization.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** NumPy
- **Visualization:** Matplotlib, Plotly (where used in the application)
- **Web Application:** Streamlit
- **Development Environment:** Google Colab and Visual Studio Code
- **Dataset:** Fashion-MNIST

## Project Structure

```text
Bootstrap-Aggregation-with-CNNs_Classifier/
│
├── app.py                  # Streamlit prediction application
├── requirements.txt        # Python dependencies
├── class_names.json        # Class label mapping
├── baseline_cnn.keras      # Trained baseline CNN
├── cnn_1.keras             # Trained bootstrap CNN 1
├── cnn_2.keras             # Trained bootstrap CNN 2
├── cnn_3.keras             # Trained bootstrap CNN 3
├── README.md               # Project documentation
├── .gitignore
├── LICENSE
└── notebook/
    └── deepbag.ipynb       # Training and evaluation notebook
```

*Adjust the tree to match the files and folders actually committed to the repository.*

## Getting Started

### Prerequisites

- Python 3.10 or another Python version compatible with your installed TensorFlow release
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Pranav123221/Bootstrap-Aggregation-with-CNNs_Classifier.git
cd Bootstrap-Aggregation-with-CNNs_Classifier
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

Ensure that `app.py`, `class_names.json` and the required trained `.keras` model files are available in the configured artifact directory.

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open it in your browser to use the application.

## Using the Application

1. Launch the Streamlit application.
2. Upload a compatible clothing image.
3. Review the baseline CNN prediction.
4. Compare predictions from the three individual CNN models.
5. View the ensemble prediction and class probabilities.

**Important:** Fashion-MNIST contains small, grayscale, 28 × 28 images. Predictions on real-world photographs may be less reliable because their backgrounds, colors, scale and image characteristics differ from the training data. Confidence scores are model probabilities, not guarantees of correctness.

## Running the Experiment

Open the training notebook in Google Colab or Jupyter and execute the cells in order.

The notebook should cover:

1. Dataset loading and preprocessing.
2. Train/validation separation.
3. CNN baseline training.
4. Bootstrap sample generation.
5. Training the three CNN models.
6. Test-set predictions and evaluation.
7. Soft-voting ensemble construction.
8. Saving trained models and supporting artifacts.

A GPU can accelerate CNN training, but the notebook can also run on a CPU with longer training times.

## Why Bagging?

A single neural network may be sensitive to the training examples and initialization it receives. Bagging introduces diversity through bootstrap sampling and combines the resulting predictions.

Potential benefits include:

- Reduced sensitivity to individual training samples.
- More robust predictions when individual models make different errors.
- An opportunity to measure whether an ensemble outperforms a single model.

**Trade-off:** Three CNNs require more storage, inference computation and maintenance than one CNN. The ensemble is useful only if its performance improvement justifies this additional cost.

## Limitations

- The experiment uses Fashion-MNIST, a relatively small and simple image benchmark.
- Only three CNN models are included in the ensemble.
- Accuracy alone does not explain every type of classification error.
- Soft-voting probabilities may not be well calibrated.
- The ensemble's performance advantage depends on diversity between model errors.
- The current experiment does not establish production-level performance on real-world clothing images.

## Future Improvements

- Add confusion matrices, precision, recall and macro F1-score.
- Compare soft voting with majority voting.
- Measure prediction latency, model size and inference cost.
- Test different bootstrap sizes and ensemble sizes.
- Explore data augmentation and learning-rate tuning.
- Add automated tests and continuous integration.
- Containerize the application with Docker.
- Deploy the Streamlit app publicly.
- Evaluate calibration and robustness on more challenging image datasets.

## Key Learnings

This project explores:

- Convolutional Neural Networks for image classification.
- Bootstrap sampling with replacement.
- Ensemble learning and soft voting.
- Train, validation and test set separation.
- Comparative model evaluation.
- Saving and loading Keras models.
- Integrating trained deep learning models into a Streamlit application.

## Author

**Pranav Sharma**  
B.Tech — Computer Science (AI/ML)

- GitHub: [Pranav123221](https://github.com/Pranav123221)
- Project Repository: [Bootstrap Aggregation with CNNs Classifier](https://github.com/Pranav123221/Bootstrap-Aggregation-with-CNNs_Classifier)

## License

This project is distributed under the license included in the repository. See the `LICENSE` file for details.
