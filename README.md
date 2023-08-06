# NLP_Internship
An NLP project based on LSTM Models for predicting restaurant reviews using ULMFit.
ULMFiT NLP Transfer Learning --> 
Sentiment analysi-->

 ### To build the text classification model, there are three stages:

1. General-Domain LM Pretraining
A pretrained AWD-LSTM SequentialRNN is imported, which works as a sequence generator (i.e. predicts the next word) for a general-domain corpus, in this case the WikiText103 dataset.

2. Target Task LM Fine-Tuning
The AWD-LSTM Language Model is fine-tuned on the domain-specific corpus (Yelp reviews), to be able to generate fake restaurant reviews.

3. Target Task Classifier
The embeddings learnt from these first two steps are imported into a new classifier model, which is then fine-tuned on the target task (star ratings) with gradual unfreezing of the final layers.


### Synthetic Text Generation
After stage 2 of the process is complete, the AWD-LSTM RNN language model can now be used for synthetic text generation. The original RNN model was trained to predict the next word in the WikiText103 dataset, and we have fine-tuned this with our yelp corpus to predict the next word in a restaurant review.

learn.predict("I really loved the restaurant, the food was")
I really loved the restaurant, the food was authentic

learn.predict("I hated the restaurant, the food tasted")
I hated the restaurant, the food tasted bad

You can generate reviews of any length. The output generally has a believable sentence structure, but they tend to lack higher-order coherency within a paragraph. This is because the RNN has no memory of the start of the sentence by the time it reaches the end of it. Larger transformer attention models like OpenAI GPT-2 or BERT do a better job at this.

learn.predict("The food is good and the staff", words=30, temperature=0.75)
The food is good and the staff is very friendly. We had the full menu and the Big Lots of Vegas. The food was ok, but there was nothing and this isn't a Chinese place.

### Classifier: Predicting the Star-value of a Review 
The overall accuracy of the trained classifier was 0.672, which means that giving the model and un-seen restaurant review it can predict its rating (1-5 stars) correctly 67.2% of the time.

Examples

Prediction: 5 | Actual: 5
(INPUT 25816) You can count on excellent quality and fresh baked goods daily. The patisseries are refined and always delicious. I am addicted to their home made salads and strong coffee. \nYou can order customized cakes and impress your guests. Everything here is made with the finest ingredients. It never disappoints. \n\nThe service is formal. You are always treated with respect. Sometimes I don't mind when they call me Madame but I always correct them and ask to be called \"Mademoiselle, SVP!\"\n\nI guarantee you will return here many times.

Prediction: 4 | Actual: 3
(INPUT 28342) 8 of us just finished eating here.  Service was very friendly, prices were definitely reasonable, and we all really enjoyed our meals. \n\nI would come back again for sure!\n\nUnfortunately I didn't snap any photos of our food, but here are a few of the place.

Prediction: 2 | Actual: 2
(INPUT 43756) The food was not all that.  The customer service was just okay. Don't get what all the rave is about??

Results
Plotting an Actual vs. Predicted matrix gives us a visual representation of the accuracy of the model. True positives are highlighted on the diagonal. So even when it makes the prediction wrong - the error usually is only off by only 1 star![Screenshot 2023-08-06 151936](https://github.com/adityaraipat/NLP_Internship/assets/75754921/87a04e41-4ced-4015-992a-5a572a836c79)



## Improvements
In the paper MultiFiT: Efficient Multi-lingual Language Model Fine-tuning (2019), the transfer learning language model is improved using

1.Subword Tokenization, which uses a mixture of character, subword and word tokens, depending on how common they are. These properties allow it to fit much better to multilingual models (non-english languages).



2.Updates the AWD-LSTM base RNN network with a Quasi-Recurrent Neural Network (QRNN). The QRNN benefits from attributes from both a CNN and an LSTM:
It can be parallelized across time and minibatch dimensions like a CNN (for performance boost)
It retains the LSTM’s sequential bias (the output depends on the order of elements in the sequence).
"In our experiments, we obtain a 2-3x speed-up during training using QRNNs"


"We find that our monolingual language models fine-tuned only on 100 labeled examples of the corresponding task in the target language outperform zero-shot inference (trained on 1000 examples in the source language) with multilingual BERT and LASER. MultiFit also outperforms the other methods when all models are fine-tuned on 1000 target language examples."

