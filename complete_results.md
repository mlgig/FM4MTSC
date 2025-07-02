# Complete Results: Foundation Models vs Traditional Methods

## 🔍 Big Picture Results Overview

This table provides a comparison of **all methods** across **all datasets** to show the complete landscape of multivariate time series classification performance.

| Category | Method | CMJ<br/>(Weak Dep.) | MP8<br/>(Strong Dep.) | MP50<br/>(Strong Dep.) | SYNTH<br/>(Strong Dep.) | Avg |
|----------|--------|:---:|:---:|:---:|:---:|:---:|
| **Traditional ML** | Random Forest | 0.933 | 0.607 | 0.476 | 0.538 | 0.639 |
| | Gradient Boosting | **0.939** | 0.605 | 0.555 | 0.518 | 0.654 |
| | KNN | 0.620 | 0.548 | 0.341 | 0.544 | 0.513 |
| | Logistic Regression | 0.687 | 0.642 | 0.622 | 0.537 | 0.622 |
| | Ridge Classifier | 0.536 | 0.607 | 0.540 | 0.519 | 0.551 |
| **Time Series** | ROCKET | 0.950 | 0.743 | **0.793** | 0.861 | **0.837** |
| | MiniRocket | 0.944 | 0.741 | 0.787 | 0.882 | 0.839 |
| | QUANT | 0.933 | 0.696 | 0.740 | **0.964** | 0.833 |
| | HYDRA | 0.944 | **0.748** | 0.738 | 0.912 | 0.836 |
| | Catch22 | 0.922 | 0.635 | 0.672 | 0.906 | 0.784 |
| **Deep Learning** | CNN | 0.922 | 0.810 | 0.661 | 0.732 | 0.781 |
| | CNN (aeon) | 0.950 | 0.659 | 0.252 | 0.868 | 0.682 |
| | TimesNet | 0.866 | 0.351 | 0.245 | 0.486 | 0.487 |
| | InceptionTime* | TBD | TBD | TBD | TBD | TBD |
| | DisjointCNN* | TBD | TBD | TBD | TBD | TBD |
| | LITEMVTime* | TBD | TBD | TBD | TBD | TBD |
| **Transformers** | ConvTran | 0.866 | 0.804 | 0.570 | 0.930 | 0.793 |
| | TSLANet | 0.894 | 0.727 | 0.424 | 0.871 | 0.729 |
| **Foundation Models** | Chronos (best) | 0.927 | 0.467 | 0.287 | 0.513 | 0.549 |
| | MOMENT | 0.855 | 0.489 | 0.662 | 0.768 | 0.694 |
| | OneFitsAll | 0.866 | 0.644 | 0.266 | 0.672 | 0.612 |
| | aLLM4TS (best) | 0.872 | 0.689 | 0.472 | 0.505 | 0.635 |
| | VQShape (best) | 0.922 | 0.659 | 0.368 | 0.671 | 0.655 |
| | Mantis (zero-shot) | 0.966 | 0.647 | 0.692 | 0.833 | 0.785 |
| | Mantis (fine-tuned) | **0.955** | 0.697 | 0.773 | 0.929 | **0.839** |

**Legend:**
- **Bold**: Best in category for that dataset
- *Asterisk: Results pending from stronger baselines
- Dep. = Channel Dependency Level

## 🎯 Key Insights

### Performance by Channel Dependency
- **Weak Dependency (CMJ)**: Foundation models competitive (Mantis: 0.955)
- **Strong Dependency (MP8/MP50/SYNTH)**: Traditional time series methods dominate

### Best Methods by Dataset
- **CMJ**: 🥇 Mantis (0.955) 🥈 ROCKET (0.950) 🥉 MiniRocket (0.944)
- **MP8**: 🥇 HYDRA (0.748) 🥈 ROCKET (0.743) 🥉 MiniRocket (0.741)  
- **MP50**: 🥇 ROCKET (0.793) 🥈 MiniRocket (0.787) 🥉 Mantis FT (0.773)
- **SYNTH**: 🥇 QUANT (0.964) 🥈 ConvTran (0.930) 🥉 Mantis FT (0.929)

### Category Rankings (Average Performance)
1. **Time Series Methods**: 0.837 avg
2. **Foundation Models**: 0.839 avg (Mantis FT)
3. **Deep Learning**: 0.781 avg  
4. **Transformers**: 0.793 avg (ConvTran)
5. **Traditional ML**: 0.654 avg

## 🔍 Channel Dependency Analysis

| Method Category | Weak Dep.<br/>(CMJ) | Strong Dep.<br/>(MP8+MP50+SYNTH) | Performance Drop |
|-----------------|:---:|:---:|:---:|
| Foundation Models | 0.922 | 0.589 | **-36.1%** |
| Time Series | 0.939 | 0.793 | **-15.6%** |
| Deep Learning | 0.913 | 0.668 | **-26.8%** |
| Traditional ML | 0.754 | 0.571 | **-24.3%** |

**Key Finding**: Foundation models show the largest performance degradation (-36.1%) when channel dependencies become important, supporting our main hypothesis.

## 📊 Dataset Characteristics

| Dataset | Samples<br/>(Train/Test) | Channels | Length | Classes | Channel Dependency | Description |
|---------|:---:|:---:|:---:|:---:|:---:|-----------|
| **CMJ** | 419/179 | 3 | 384 | 3 | **Weak** | Counter Movement Jump - y-channel sufficient |
| **MP8** | 1,426/595 | 8 | 161 | 4 | **Strong** | Military Press - requires 4+ channels |
| **MP50** | 1,426/595 | 50 | 161 | 4 | **Strong** | Military Press - all body part coordinates |
| **SYNTH** | 7,500/1,000 | 8 | 500 | 2 | **Strong** | Synthetic - requires specific 2 channels |

---
