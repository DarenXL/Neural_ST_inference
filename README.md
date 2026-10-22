# Neural Simulation-Based Inference Methods for Complex Spatio-Temporal Models

**Alexandre Loret**¹\*, **Lucia Clarotto**², **Nicolas Desassis**¹, **Thomas Romary**¹  

¹ *STIM, Mines Paris – PSL*  
² *MIA Paris-Saclay, AgroParisTech / INRAE – Université Paris-Saclay*  

---

In spatio-temporal data analysis, estimating the parameters of spatio-temporal covariance functions of Gaussian random fields is usually performed by maximum likelihood estimation (MLE). However, in practice, this method quickly becomes computationally prohibitive as the size of the dataset increases.
Our work aims at developing alternatives to this traditional approach. One of the main challenges lies in the complex spatio-temporal dependencies of the models, which may arise from physical mechanisms, such as advection, or from non-separability between space and time.
Following recent studies that address this issue using Simulation-Based Inference (SBI) techniques, we propose two neural network architectures designed to learn the parameters of Gaussian spatio-temporal Gneiting-Matérn models. The networks are trained on multiple simulations generated from the Gaussian random field model itself. We compare different architectures in terms of estimation % and prediction
accuracy, uncertainty quantification and computational cost. The proposed approaches have the advantage of being amortized: unlike classical MLE, once training is completed, parameter inference is almost instantaneous.

---

# Set up environment

The easiest option is to use [uv](https://docs.astral.sh/uv/).

```
uv sync
```

# Scripts

Two important scripts : 

- `workflow_classic.py`
Used to perform the complete simulation studies : train set and test set simulations + training and testing of NBE and NRE + MLE + summary graphs.

- `workflow_sun.py`
Used to perform the inferences on the solar radiations data : Sliding windows over time + kriging and conditional simulations. The network is already trained.

---

### Acknowledgments
\* The authors would like to thank the [Geolearning Chair](https://chaire-geolearning.org/) for its financial support.
