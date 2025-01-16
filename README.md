# Forecasting-Outside-the-Box

This repo presents the code and a small summary of our work "Forecasting Outside the Box: Application-Driven Optimal Pointwise Forecasts for Stochastic Optimization" by PhD. Tito Homem-de-Mello, Juan Valencia, PhD. Felipe Lagos and PhD. Guido Lagos. The latest version of the paper can be found [here](https://arxiv.org/abs/2411.03520).

This paper is in the context of "Machine Learning for Optimization" and "Optimization under Uncertainty," where the decision maker tries to solve an optimization problem that considers uncertainty. The decision maker uses machine learning methods to address and predict what the uncertainty will be and uses it as input for the optimization problem that the practitioner would like to solve. The enormous growth in data availability in recent years—more specifically, the presence of contextual information in the data (also called covariates or features in the literature)—has led to the development of new models in stochastic optimization. In such models, the uncertainty represented by ξ can be predicted to some extent by the available contextual information. Thus, the goal is to optimize the expected value of a certain function conditionally on a given value of the contextual information, henceforth called the contextual information of interest.

In this setting, practitioners typically use classical forecasting methods that do not consider the specific application of the forecasts. This oversight can lead to assumptions—such as the prediction error symmetry inherent in least squares—that may not be appropriate for problems with asymmetric outcomes.

Our work operates within a framework known as SPO (Smart "Predict, then Optimize"). The central concept of the SPO approach is to change the training metric for models from a "least squares metric" to a "regret or loss metric." This shift allows us to measure the decision error caused by the estimation or prediction and assess the performance of the prediction based on its impact on the objective function, rather than relying on traditional error criteria like least squares. As a result, we can create biased predictors that enhance optimization by minimizing the impact of incorrect predictions

The "cool" thing about doing this is that the predictions can be out of the support of the data, meaning that we can give random predictions that work well :) (See the paper for results).

We have tested our methods in three different problems: A resource allocation problem [ (link)](https://github.com/rohitkannan/DD-SAA/blob/master/Data-Driven%20SAA%20with%20Covariate%20Information%20(R1).pdf), a Shipment planning problem[ (link)](https://arxiv.org/abs/1904.11637), and a real-world problem of bike sharing allocation in San Francisco[ (link)](https://www.semanticscholar.org/paper/Stochastic-programming-models-for-distribution-and-Cavagnini/1e758d7a0422c10057726d4752cd01b4a7d4ba53).
