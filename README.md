# Neural Simulation-Based Inference for space-time Gaussian processes

**Alexandre Loret**¹\*, **Lucia Clarotto**², **Nicolas Desassis**¹, **Thomas Romary**¹  

¹ *STIM, Mines Paris – PSL*  
² *MIA Paris-Saclay, AgroParisTech / INRAE – Université Paris-Saclay*  

---

In spatio-temporal data analysis, estimating covariance parameters for Gaussian random fields via maximum likelihood estimation (MLE) is computationally prohibitive for large datasets. 
Following recent advances in simulation-based inference techniques, we propose two neural network architectures — Neural Bayes Estimation (NBE) and Neural Ratio Estimation (NRE) — to learn the parameters of Gaussian spatio-temporal Gneiting-Matérn models with advection. 
The networks are trained on simulations generated directly from the underlying Gaussian random field model. 
A thorough simulation study demonstrates strong performance across one-dimensional time and both one-dimensional space and two-dimensional space, yielding high accuracy and reliable uncertainty quantification for both NBE and NRE. 
Through extensive offline training, the framework achieves fully amortized inference, replacing costly optimization with a single forward pass. Finally, we apply our framework to solar radiation prediction and short-term forecasting. 
By adopting a sliding-window approach, the model accounts for temporal non-stationarity while retaining this key advantage: achieving near-instantaneous parameter inference and forecasting.

---

# Set up environment

The easiest option is to use [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Depending on CUDA version, KeOps might cause crashes during MLE or kriging prediction. The parameter `use_KeOps` can be set to `False`.

# Scripts

Two important scripts : 

- `workflow_classic.py`
Used to perform the complete simulation studies : train set and test set simulations + training and testing of NBE and NRE + MLE + summary graphs.

- `workflow_sun.py`
Used to perform the inferences on the solar radiations data : Sliding-windows over time + kriging and conditional simulations. The network is already trained.

---

### Acknowledgments
\* The authors would like to thank the [Geolearning Chair](https://chaire-geolearning.org/) for its financial support.
