Aidan Goeschel
Adam Gans
Frank Xu
CS 179 Final Write Up

1.	Introduction
For our project, we chose to create a lip-reading model that outputs text from an input video of someone speaking. Our main approach centers on using a Variational Autoencoder (VAE) built on top of a pre-trained 3D convolutional encoder (ResNet3D_18). We fine-tuned this encoder as part of the VAE on our lip reading dataset, so that it learns features specifically useful for visual speech recognition, namely a mean and a variance for each element in the latent space. From these latent distributions, we sample 5 points per distribution. We then use those samples to train a series of classifiers. We compared the performance of a deep neural network, a multi-layer perceptron network, and a gradient boosted forest, specifically XGBoost.

We are using a subset of the GRID dataset (Glasgow University's Audio-Visual Speech Databases). The entire dataset consists of videos of 34 human subjects speaking 53 different words. The subset we are using consists of 10 people speaking all 53 words. We also tested on 2 other subsets that were not in the training dataset to test the out-of-distribution performance of our model.

While our VAE + MLP/DNN/XGBoost pipeline may not match the performance of state-of-the-art end-to-end CNN+RNN or Transformer models in lipreading research, it is a valid and practical approach for word-level lipreading, especially on medium-sized datasets.

2.	Resources
We used the dataset from https://spandh.dcs.shef.ac.uk/gridcorpus/ to train and test our model. It includes short MPGs (videos) of various individuals saying a variety of random words, with the alignments indicating which words are being said during which timestamps. 

To process our videos into code, we use the CV2 library created by OpenCV. Specifically, we use the function cv2.VideoCapture(), which turns each video frame into a 360x288 matrix where each value is the RGB color of that pixel. We save a list of frames along with the corresponding word that is said during those frames. Then, to save ourselves a lot of computation, we use a pretrained model to create a feature vector of the given collection of frames. Specifically, we use r3d_18 from Torchvision Models.   
3.	Training & Evaluation
To assess the efficacy of our lipreading word prediction pipeline, we employed a combination of quantitative error metrics and empirical timing analysis. The evaluation focused on both the feature extraction (VAE) and classification (MLP/DNN/XGBoost) stages.

Encoder
Error Metrics 
●	Accuracy: The primary metric was classification accuracy, defined as the proportion of correctly predicted words over the total number of samples. We report both training and validation accuracy to assess model generalization.
●	Loss curve: Tracking the loss (reconstruction, KL divergence, and classification loss) during VAE training helps diagnose underfitting or overfitting.
                         

Time Information
●	Training time: 93 Minutes

At the beginning of our project, we used a pre-trained encoder based on the R(3D)18 (r3d_18) architecture from torchvision. This model was used without any fine-tuning on our lipreading dataset. As a result, the extracted features were not well-suited for our specific task, and the downstream classification performance was poor. 

To address this, we implemented a Variational Autoencoder (VAE) framework. By training the VAE on our dataset, we were able to fine-tune the encoder so that it learned features more relevant to lipreading. However, even after this step, the classification accuracy on the training set was still low (around 30%).

To further improve performance, we added a classifier head (a linear layer) to the VAE. This allowed the encoder to learn features that are not only good for reconstruction but also discriminative for word classification. After this modification, we observed a significant improvement: the MLP classifier trained on the VAE’s latent features achieved around 80% accuracy on the training set.

Additionally, we applied batch processing during training. This not only made the training process much faster but also helped stabilize the optimization and improve generalization.

Classifier
DNN
For the DNN, we tried many different configurations (i.e., different layer sizes and number of dropout layers), but we ended up settling on the architecture seen to the left. We used a cross-entropy loss function.

The model kept overfitting, which we can see in the divergence of the validation loss and training loss below. To try to mitigate this, we increased the number of samples per latent distribution to increase the training dataset size for the classifier, and also added dropout layers after each block of dense layers and ReLU. The overall test accuracy for the best model was 24.87%, which, while still significantly higher than chance, could not be used for accurate lip reading transcription. We found that in our training set, nearly a quarter of the tokens were “sil,” so the model learned to always predict that token, which could have led to the subpar performance of the model.

XGBoost
Next, we decided to completely switch architectures. We figured that decision trees may be able to better predict the words based on points sampled from the latent distribution. We used the same augmented dataset that we used for the DNN to train it. The training accuracy was 27.4%, and the testing accuracy was 24.6%. We had similar problems to the DNN, where it only predicted “sil”.

MLP
In an effort to understand the level of accuracy we should expect, we originally created a simple MLP model to directly classify the feature vectors given from the pretrained r3d_18 model. Because this model achieved higher accuracy than our latent space classifiers, we decided to try viewing our latent space values as 256-value vectors and training the MLP with these. With our original encoder, this only resulted in slightly better performance compared to the DNN/XGBoost methods, at ~30% (with single-digit out-of-distribution accuracy). However, after we trained the encoder, the performance of the MLP classification jumped to well over 70% accuracy. After a full training of the MLP, up to 1000 epochs, the final test accuracy was 77.3%, and the out-of-distribution accuracy was 52.4%.

4.	Conclusion
While sampling from our latent distributions may have seemed to be a good way to generate a large dataset, it did not yield good results in terms of classification accuracy. We found that the best performance was using the mu and log variance themselves as inputs into an MLP classifier rather than sampling from the distributions. Therefore, it stands to reason that these classification models are able to more effectively learn from the structured, deterministic features encoded in the latent means and variances, rather than the noise introduced by sampling. This suggests that the latent parameters themselves carry the most discriminative information for classification tasks, and that sampling may obscure this signal with unnecessary stochasticity.

5.	Works Cited/Appendix
Barker, Jon. “Grid Corpus Dataset.” The GRID Audiovisual Sentence Corpus, University of Sheffield, 18 Mar. 2013, spandh.dcs.shef.ac.uk/gridcorpus/. 
Bentalb, Mohamed. “Lipreading Dataset.” Kaggle, 13 May 2023, www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset. 
Cooke, Martin, et al. “The Grid Audio-Visual Speech Corpus.” Zenodo, Zenodo, 22 July 2024, zenodo.org/records/3625687. 
“PYTORCH Documentation.” PyTorch Documentation - PyTorch 2.7 Documentation, docs.pytorch.org/docs/stable/index.html. Accessed 11 June 2025.
 
Graph of the difference in validation and training loss for the deep neural network. The vertical line marks the epoch with the smallest overall validation loss.
