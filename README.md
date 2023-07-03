# XTEA_Neural-Distinguisher
Neural-Cryptanalysis of XTEA cipher
The analysis of the XTEA encryption algorithm is crucial for assessing its security and identifying potential vulnerabilities. Our research focuses on accurately distinguishing between random and nonrandom outputs of the XTEA encryption algorithm using neural network-based distinguishers. We introduce novel Conv1D and Conv2D architectures with residual blocks and utilize k-fold cross-validation for model training. Evaluating the models on a dataset of random and non-random XTEA sequences, we use accuracy as the main evaluation metric. Our experimental results demonstrate the successful performance of both Conv1D and Conv2D architectures up to the 7th round of XTEA, effectively distinguishing between random and non-random outputs. This work contributes to enhancing the security analysis of XTEA by leveraging advanced machine learning techniques, highlighting the potential of neural network-based distinguishers in improving the analysis of encryption algorithms.

## Results of Conv1D neural distinguisher:-
![Alt text](Results/combine.png?raw=true "Conv1D")

## Results of Conv2D neural distinguisher:-
![Alt text](Results/combine_2D.png?raw=true "Conv2D")

### Comparison with existing working models:-
Throughout our analysis, we employed both the Conv1D and Conv2D architectures to differentiate between the rounds of XTEA. By utilizing these designs, we achieved a flawless accuracy of 100% for the fifth round. Additionally, we extended our investigation up to the seventh round. Comparing our findings with those of Bellini et al. in their publication "A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input Differences," we discovered that they obtained an accuracy of 0.5978% for the fifth round using the DBitNet model. Our results clearly demonstrate a significant improvement of 40% in accuracy over the DBitNet model proposed by Bellini et al. Furthermore, while Bellini et al. did not provide results beyond the fifth round, our study successfully distinguished up to the seventh round with high accuracy.
![Alt text](Results/comparison.png?raw=true "comparison")
