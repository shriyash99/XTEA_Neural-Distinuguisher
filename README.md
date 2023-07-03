# XTEA_Neural-Distinguisher
Neural-Cryptanalysis of XTEA cipher
The analysis of the XTEA encryption algorithm is crucial for assessing its security and identifying potential vulnerabilities. Our research focuses on accurately distinguishing between random and nonrandom outputs of the XTEA encryption algorithm using neural network-based distinguishers. We introduce novel Conv1D and Conv2D architectures with residual blocks and utilize k-fold cross-validation for model training. Evaluating the models on a dataset of random and non-random XTEA sequences, we use accuracy as the main evaluation metric. Our experimental results demonstrate the successful performance of both Conv1D and Conv2D architectures up to the 7th round of XTEA, effectively distinguishing between random and non-random outputs. This work contributes to enhancing the security analysis of XTEA by leveraging advanced machine learning techniques, highlighting the potential of neural network-based distinguishers in improving the analysis of encryption algorithms.

## Results of Conv1D neural distinguisher:-
![Alt text](Results/combine.png?raw=true "Conv1D")

## Results of Conv2D neural distinguisher:-
![Alt text](Results/combine_2D.png?raw=true "Conv2D")
