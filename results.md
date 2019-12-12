# Results on public datasets

We measure the performance of the generative models on the mentioned public datasets by evaluating the downstream learners which are learned on synthetic data produced by these models and testing them on real test data.



Experiments are performed for three different privacy levels i.e low privacy (epsilon = 8), medium privacy (epsilon = 5) and high privacy (epsilon = 2). Delta is fixed to be 10<sup>-5</sup>. <br/>
Since the datasets we use are class imbalanced, we resort to using the AUROC score to evaluate the learners.

### A note on choosing privacy parameters:
* `sigma` : This determines the amount of noise added during noisy SGD training for DP-WGAN and IMLE. For a given privacy budget, a larger `sigma` would lead to more number of training epochs compared to a smaller `sigma`. For our purposes, we chose sigma such that the number of training epochs are close to what a non-private training of the model would require for satisfactory performance.
* `lap_scale` :  This is the inverse laplace noise scale multiplier and determines the amount of noise added when performing private aggregation of teacher ensembles (PATE) in PATE-GAN. For a given privacy budget, a larger `lap_scale` would lead to less number of training epochs compared to a smaller `lap_scale`. We chose `lap_scale` the same way we chose `sigma`.
* `num_teachers` : The number of teacher discriminators to be used in the PATE-GAN setup. Increasing the number of teachers makes the training more resilient to noise added but also needs more data for the predictions of indivdual teachers to remain reliable. We use 10 teacher discriminators for all our experiments.
* `clip_coeff` : The coefficient to clip the gradients to before adding noise for SGD training for DP-WGAN and IMLE. We use a value of 0.1 for all our experiments.
* `micro_batch_size` : Gradients are averaged for a micro-batch and then clipped before adding noise during noisy SGD training for DP-WGAN and IMLE. Increasing the micro-batch size will reduce utility but improve run-time. We use a micro-batch size of 8 and a mini-batch size of 64 for all our experiments.

Choosing these parameters for a new dataset or a different privacy budget is an art in itself. We provide the results below for the datasets we used along with the values of these parameters to aid the user.

### Adult Census Dataset:

Dataset Shape : 33,916 X 12  
Column type : All continuous  
Task : predict whether a person earns more than $ 50k a year or not

Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
Real Data | inf/inf | - | 0.836 | 0.837 | 0.872 | 0.809 | 0.883
DPWGAN | 8/1e-5 | sigma=0.8 | 0.741 | 0.731 | 0.754 | 0.758 | 0.747
PATEGAN | 8/1e-5 | lap_scale=3e-4 | 0.595 | 0.582 | 0.560 | 0.619 | 0.490
RONGauss | 8/1e-5 | - | 0.587 | 0.553 | 0.652 | 0.710 | 0.568
IMLE | 8/1e-5 | sigma=0.6 | 0.806 | 0.808 | 0.789 | 0.708 | 0.779


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 5/1e-5 | sigma=0.9 | 0.690 | 0.789 | 0.5 | 0.683 | 0.761
PATEGAN | 5/1e-5 | lap_scale=3e-4 | 0.633 | 0.673 | 0.5 | 0.559 | 0.624
RONGauss | 5/1e-5 | - | 0.54 | 0.675 | 0.516 | 0.759 | 0.603
IMLE | 5/1e-5 | sigma=0.7 | 0.818 | 0.817 | 0.798 | 0.744 | 0.768


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 2/1e-5 | sigma=1.0 | 0.7 | 0.697 | 0.5 | 0.66 | 0.5
PATEGAN | 2/1e-5 | lap_scale=1e-4 | 0.658 | 0.618 | 0.5 | 0.5 | 0.5
RONGauss | 2/1e-5 | - | 0.559 | 0.583 | 0.647 | 0.721 | 0.556
IMLE | 2/1e-5 | sigma=0.85 | 0.812 | 0.745 | 0.775 | 0.610 | 0.769


### NHANES Diabetes Dataset:

Dataset Shape : 4,412 X 48  
Column type : All continuous  
Task : predict the onset of type II diabetes

Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
Real Data | inf/inf | - | 0.802 | 0.814 | 0.754 | 0.729 | 0.831
DPWGAN | 8/1e-5 | sigma=1.9 | 0.545 | 0.53 | 0.494 | 0.54 | 0.493
PATEGAN | 8/1e-5 | lap_scale=1e-4 | 0.5 | 0.727 | 0.567 | 0.603 | 0.525
RONGauss | 8/1e-5 | - | 0.496 | 0.51 | 0.523 | 0.525 | 0.748
IMLE | 8/1e-5 | sigma=1.2 | 0.720 | 0.712 | 0.713 | 0.723 | 0.730


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 5/1e-5 | sigma=2.2 | 0.683 | 0.757 | 0.615 | 0.756 | 0.576
PATEGAN | 5/1e-5 | lap_scale=1e-4 | 0.720 | 0.766 | 0.560 | 0.5 | 0.43
RONGauss | 5/1e-5 | - | 0.61 | 0.563 | 0.642 | 0.601 | 0.748
IMLE | 5/1e-5 | sigma=1.9 | 0.711 | 0.648 | 0.721 | 0.720 | 0.687


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 2/1e-5 | sigma=7.0 | 0.495 | 0.513 | 0.5 | 0.505 | 0.556
PATEGAN | 2/1e-5 | lap_scale=1e-4 | 0.43 | 0.368 | 0.553 | 0.501 | 0.496
RONGauss | 2/1e-5 | - | 0.605 | 0.670 | 0.391 | 0.585 | 0.554
IMLE | 2/1e-5 | sigma=4.0 | 0.447 | 0.472 | 0.486 | 0.607 | 0.541


### Give Me Some Credit Dataset:

Dataset Shape : 90,201 X 11  
Column type : All continuous  
Task : predict the experience of financial distress in the next 2 years

Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
Real Data | inf/inf | - | 0.842 | 0.845 | 0.759 | 0.837 | 0.849
DPWGAN | 8/1e-5 | sigma=0.5 | 0.553 | 0.554 | 0.684 | 0.49 | 0.559
PATEGAN | 8/1e-5 | lap_scale=3e-4 | 0.742 | 0.654 | 0.499 | 0.471 | 0.5
RONGauss | 8/1e-5 | - | 0.477 | 0.569 | 0.445 | 0.61 | 0.65
IMLE | 8/1e-5 | sigma=0.6 | 0.787 | 0.785 | 0.808 | 0.761 | 0.814


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 5/1e-5 | sigma=0.7 | 0.668 | 0.625 | 0.348 | 0.539 | 0.569
PATEGAN | 5/1e-5 | lap_scale=3e-4 | 0.598 | 0.634 | 0.383 | 0.737 | 0.667
RONGauss | 5/1e-5 | - | 0.508 | 0.473 | 0.589 | 0.695 | 0.702
IMLE | 5/1e-5 | sigma=0.7 | 0.821 | 0.800 | 0.760 | 0.811 | 0.677


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 2/1e-5 | sigma=2 | 0.489 | 0.492 | 0.475 | 0.721 | 0.521
PATEGAN | 2/1e-5 | lap_scale=1e-4 | 0.484 | 0.541 | 0.505 | 0.786 | 0.683
RONGauss | 2/1e-5 | - | 0.508 | 0.473 | 0.589 | 0.695 | 0.702
IMLE | 2/1e-5 | sigma=0.9 | 0.770 | 0.780 | 0.771 | 0.502 | 0.695



### Home Credit Risk Dataset:

Dataset Shape : 230,633 X 403  
Column type : All continuous  
Task : predict if the client will repay their loan

Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
Real Data | inf/inf | - | 0.754 | 0.758 | 0.633 | 0.614 | 0.761
DPWGAN | 8/1e-5 | sigma=0.55 | 0.541 | 0.523 | 0.502 | 0.5 | 0.518
PATEGAN | 8/1e-5 | lap_scale=3e-4 | 0.54 | 0.512 | 0.508 | 0.51 | 0.518
RONGauss | 8/1e-5 | - | 0.487 | 0.494 | 0.492 | 0.505 | 0.574
IMLE | 8/1e-5 | sigma=0.5 | 0.478 | 0.479 | 0.506 | 0.480 | 0.496


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 5/1e-5 | sigma=0.7 | 0.507 | 0.471 | 0.511 | 0.5 | 0.505
PATEGAN | 5/1e-5 | lap_scale=2e-4 | 0.503 | 0.540 | 0.497 | 0.535 | 0.547
RONGauss | 5/1e-5 | - | 0.496 | 0.5 | 0.506 | 0.5 | 0.57
IMLE | 5/1e-5 | sigma=0.55 | 0.475 | 0.511 | 0.5 | 0.525 | 0.493


Method | Eps/Delta | Privacy parameter | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- |--- 
DPWGAN | 2/1e-5 | sigma=1 | 0.478 | 0.452 | 0.504 | 0.5 | 0.555
PATEGAN | 2/1e-5 | lap_scale=1e-4 | 0.472 | 0.499 | 0.537 | 0.547 | 0.56
RONGauss | 2/1e-5 | - | 0.496 | 0.5 | 0.506 | 0.5 | 0.57
IMLE | 2/1e-5 | sigma=0.95 | 0.532 | 0.497 | 0.476 | 0.5 | 0.471



### Adult Categorical Dataset:

Same as the Adult Census dataset  
Column type : All categorical

Method | Eps/Delta | LR | NN | RF | GNB | GB 
--- | --- | --- | --- |--- |--- |--- 
Real Data | inf/inf |  0.888 | 0.893 | 0.877 | 0.852 | 0.915
PrivatePGM | 8/1e-5 | 0.882 | 0.882 | 0.867 | 0.850 | 0.896
PrivatePGM | 5/1e-5 | 0.881 | 0.884 | 0.865 | 0.850 | 0.896
PrivatePGM | 2/1e-5 | 0.881 | 0.881 | 0.869 | 0.850 | 0.896


