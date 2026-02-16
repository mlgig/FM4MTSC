# FM4MTSC
This repository contains the code for the paper "An Empirical Evaluation of Foundation Models for Multivariate Time Series Classification" accepted at ECML-PKDD 2025.

## Abstract
Foundation models have recently emerged as a promising approach for time series analysis, adapting transformer architectures originally designed for natural language processing to handle continuous temporal data. While these models demonstrate strong performance across various time series tasks, their handling of multivariate time series, particularly inter-channel dependencies, remains underexplored. In this paper, we present a comprehensive analysis of current foundation models for time series, including tokenization-based, patch-based, and shape-based approaches, focusing on their mechanisms and data representations for capturing relationships between channels. Our analysis shows that even though these models have advanced architectures, they mostly process channels independently, which may prevent them from fully capturing cross-channel patterns. We examine this limitation across different model families and discuss its implications for multivariate time series analysis. Our empirical evaluation shows that foundation models perform well on simpler tasks but exhibit diminished effectiveness as channel dependencies increase, with specialized time series methods consistently outperforming them on complex datasets. These findings highlight the critical need for channel-aware architectures and more effective strategies for modeling inter-channel relationships in foundation models.

## Resources
* [Full Paper (ResearchGate)]([[https://www.researchgate.net/publication/392760978_...](https://www.researchgate.net/publication/392760978_An_Empirical_Evaluation_of_Foundation_Models_for_Multivariate_Time_Series_Classification)])
* [Research Poster (PDF)](./docs/Poster.pdf)
* [Technical Presentation (PDF)](./docs/1221-An_Empirical_Evaluation_of_Foundation_Models_for_MTSC_Pinar_Sungu_Isiacik.pptx)


# Datasets

Datasets are available [HERE](https://drive.google.com/drive/folders/10Hvep0hk9naj_nRk4JMiXHMeV1vw0lcd?usp=sharing)

# Time Series Classification Methods

This repository contains a collection of various methods for time series classification, organized into distinct modules for better maintainance and comparison.

## Repository Structure

```
├── deep_learning_methods/          # Deep learning based models
│   ├── src/                        # Source code
│   └── scripts/                    # Training scripts
│
├── foundation_models/              # Foundation model implementations
│   ├── aLLM4TS/                    # aLLM4TS model
│   ├── chronos/                    # Chronos model
│   ├── mantis/                     # Mantis model
│   ├── moment/                     # Moment model
│   ├── one-fits-all/               # One-Fits-All model
│   └── VQShape/                    # VQShape model
│
├── time_series_methods/            # Time series specific methods
│   ├── src/                        # Source code
│   └── scripts/                    # Training scripts
│
├── traditional_ml_methods/         # Traditional ML based models
│   ├── src/                        # Source code
│   └── scripts/                    # Training scripts
│
└── transformer_based_methods/      # Transformer based models
│   ├── src/                        # Source code
│   └── scripts/                    # Training scripts
└──
```

## Methods Overview

### Traditional ML Methods
- Classical machine learning algorithms adapted for time series data
- Random Forest, SVM, KNN, Gradient Boosting, Logistic Regression, etc.

### Time Series Methods
- Specialized methods for time series classification from the Aeon library
- Methods like ROCKET, MiniRocket, QUANT, Hydra, and Catch22

### Deep Learning Methods
- **Aeon CNN**: CNN classifier from the Aeon library
- **Custom CNN**: Custom implementation of CNN for time series classification
- **TimesNet**: Implementation of the TimesNet architecture

### Transformer-Based Methods
- **TSLANet**: TSLANet: Rethinking Transformers for Time Series Representation Learning
- **ConvTran**: Improving Position Encoding of Transformers for Multivariate Time Series Classification (ConvTran)

### Foundation Models

- **aLLM4TS**: [Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning](https://arxiv.org/pdf/2402.04852)

- **Chronos**: [Chronos: Learning the Language of Time Series](https://arxiv.org/pdf/2403.07815)

- **Mantis**: [Mantis: Lightweight Calibrated Foundation Model for User-Friendly Time Series Classification](https://arxiv.org/pdf/2502.15637)

- **MOMENT**: [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/pdf/2402.03885)

- **One-Fits-All**: [One Fits All: Power General Time Series Analysis by Pretrained LM](https://arxiv.org/pdf/2302.11939)

- **VQShape**: [Abstracted Shapes as Tokens - A Generalizable and Interpretable Model for Time-series Classification](https://arxiv.org/pdf/2411.01006)


## Usage

Each module has its own README file with specific instructions, but the general workflow is:

1. Install the required dependencies for the specific method you want to use
2. Prepare your data in the expected format (typically NumPy `.npy` files)
3. Run the training scripts or use the provided notebooks to train and evaluate models



## Data Format

The expected data format is a NumPy `.npy` file containing a dictionary with:
- `train` key with `X` (features) and `y` (labels) keys
- `test` key with `X` (features) and `y` (labels) keys

Example structure:
```python
{
    "train": {
        "X": np.ndarray,  # Shape: [n_samples, n_channels, n_timesteps]
        "y": np.ndarray 
    },
    "test": {
        "X": np.ndarray,  # Shape: [n_samples, n_channels, n_timesteps]
        "y": np.ndarray  
    }
}
```

## References and Original Repositories

### Time Series Methods
- **Aeon Library**: [GitHub](https://github.com/aeon-toolkit/aeon)

### Deep Learning Methods
- **Aeon CNN**: [Aeon Library](https://github.com/aeon-toolkit/aeon)
- **TimesNet**: [Original Implementation](https://github.com/thuml/TimesNet)

### Transformer-Based Methods
- **TSLANet**: [Original Implementation](https://github.com/emadeldeen24/TSLANet)
- **ConvTran**: [Original Implementation](https://github.com/Navidfoumani/ConvTran)

### Foundation Models
- **aLLM4TS**: [Original Implementation](https://github.com/yxbian23/aLLM4TS)
- **Chronos**: [Original Implementation](https://github.com/amazon-science/chronos-forecasting)
- **Mantis**: [Original Implementation](https://github.com/vfeofanov/mantis)
- **Moment**: [Original Implementation](https://github.com/moment-timeseries-foundation-model/moment-research)
- **one-fits-all**: [Original Implementation](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)
- **VQShape**: [Original Implementation](https://github.com/YunshiWen/VQShape)



## License

Each method may come with its own license. Please refer to the original repositories for specific license information.

## Citation

If you use this repository in your research, please cite as:
```
@misc{pinar2025fm4mtsc,
    title={An Empirical Evaluation of Foundation Models for Multivariate Time Series Classification},
    author={Pinar Sungu Isiacik and Thach Le Nguyen and Timilehin Aderinola and Georgiana Ifrim},
    year={2025},
    conference={ECMLPKDD},
    eprint={tba},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
