# Swedish Survival Kit
Authors: Tori Leatherman & Eva Engel

## Introduction - Scalable ML System

This repository contains a scalable ML system service that provides transcription of swedish language audio. It utilises Whisper, a pre-trained Hugging Face Transformer that is fine-tuned for the data set 'Common Voice'. Common Voice is a series of crowd-sourced datasets where speakers record text from Wikipedia in various languages. We'll use the latest edition of the Common Voice dataset (version 11), choosing the subset containing only swedish. 

As stated previously, Whisper is a pre-trained model for automatic speech recognition (ASR) published in September 2022 by the authors Alec Radford et al. from OpenAI. It was pre-trained on labelled audio-transcription data, with 680,000 hours of training data. It is available in over 96 languages; however, we chose the language swedish to focus on. As foreigners in Sweden and novices to the Swedish language, we thought it would be valuable to create a transcription tool for Swedish audio. This transcription coupled with a translation pipeline allows us to transcribe swedish audio or video, then automatically translate it into english. This allows us to transcribe and translate swedish in real-time, which is invaluable for those who have low comprehension of the language. 

This work is based on the blog post https://huggingface.co/blog/fine-tune-whisper which details a step-by-step guide of fine-tuning Whisper using Hugging Face Transformers. We have implemented the steps contained here into two different pipelines in Google Colab: a Feature Pipeline and a Training Pipeline. After which, we uploaded and utilized our trained model in HuggingFace to create a UI that allows a user to speak into the microphone, or upload a video URL which will then be transcribed and translated to english. 

#### Try it yourself here: https://huggingface.co/spaces/torileatherman/whisper_swedish_to_english

The pipelines and corresponding descriptions can be found in the notebook files.

## Model Experiments & Improvements

Although we use a pre-trained transformer model, there are various possibilities to improve the model’s performance. One approach focuses on the model including its architecture and its hyperparameter. The other approach addresses the data basis and handling.

### Model-Centric
Starting with model-centric improvements, we could change the model architecture in general. In our case, OpenAI provides 5+ different transformer sizes:

![OpenAI Models](https://miro.medium.com/max/1400/1*_xEqgvZDExxNMaeojLQlDg.png) 

Depending on the size of the available data, one can choose a bigger architecture that consists of more parameters. However, this might lead to an overfit or to not having enough resources to train the model.
Secondly, the results could likely be improved by optimising the training hyperparameters, such as learning rate and dropout. Hyperparameters can be tested using a grid search or a random search in order to try different values in combination that may improve the model performance. 

In our case, we attempted model-centric improvement using two different methods previously described: altering model architecture, and tuning model hyperparameters.

### Model Architecture

#### Baseline model
Given the hyperparameters in the provided notebook, we fine-tuned the whisper-small model. Our model details can be seen on huggingface here: https://huggingface.co/torileatherman/whisper_small_sv.  For 4000 steps, we get the following evaluation results: 
| Training loss | Validation Loss | Wer |
| --- | --- |  --- |
| 0.0060 |0.320301 |  19.762 |

Worth mentioning is the validation loss starts to increase indicating a overfit of the training data.




The next model we will address is whisper-medium, which can be found through openai here: https://huggingface.co/openai/whisper-medium. For this model, we used the same hyperparameters as above with the small model, with the exception of using 6000 steps instead of 4000. We achieved the following training results after 6000 steps:

| Training loss | Validation Loss | Wer |
| --- | --- |  --- |
| 0.0016 |0.712412|  30.577 |

For the changes in model architecture, we also utilized a dashboard Weights and Bias to track the performance in various metrics. These can be viewed here: 
As we can see, the Wer score is higher than the small model which indicates a worse performance. Thus in order to achieve the same performance as the small model, we need sufficiently more steps.  Looking at the wand dashboard that tracks the performance in various metrics https://wandb.ai/two_data_scientists/test-project?workspace=user-eengel7, we see that both the training and validation loss remain to decrease. We can infer that using some more training steps, we would improve the performance of the model. Note that, since there are 769M parameters in the medium model, compared to 244M parameters in the small model, we will most likely overfit at some point. Given constrained resources and already 10h of training, we were not able to continue training the model.

The next model we tried is whisper-tiny, which can be found here: https://huggingface.co/openai/whisper-tiny. In this case, we had the same hyperparameters as the small model; however, we terminated the training process after 1000 steps. The reasoning for this will become clear after looking at the following training results: 
| Training loss | Validation Loss | Wer |
| --- | --- |  --- |
| 0.6874 |0.714224|   109.8679|

With previous models, after 1000 steps the Wer score was already significantly close to the end result at the end of all 4000 or 6000 steps. Thus, we concluded that there was a statistical impossibility that the tiny model would outperform the small model which had a Wer score of close to 19.

### Model Hyperparameters
We tried unsuccessfully to alter the model architecture in order to achieve a higher Wer score than our small model. We now tried to tune hyperparameters in the following ways.

The first method we chose was to alter the learning rate from 1e-05 to 2e-05. This resulted in the following training results after 4000 steps:
| Training loss | Validation Loss | Wer |
| --- | --- |  --- |
| 0.0357 |0.279927|   99.36274|

As this was unsuccessful, we chose to alter the type of learning rate from a linear rate to a cosine learning rate. Therefore, we implemented a Scheduler class that adjusts the learning rate. In our case, we used get_cosine_schedule_with_warmup that creates a schedule with a learning rate that decreases following the values of the cosine function between the initial learning rate set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the initial learning rate set in the optimizer.
The dashboard https://wandb.ai/two_data_scientists/hypertuning_learning_rate?workspace=user-eengel7 indicates that hypertuning the learning rate using this approach will most likely not result in an improvement of model performance.


### Data-Centric

Data-centric improvement means adding valuable data to our model to improve training. Since we chose to evaluate Swedish audio, we found that the https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-56/ had relevant and reputable data. This source includes Swedish audio files and associated transcriptions. Due to time constraints, we chose to focus our efforts on model-centric improvements, and did not add any additional data in an attempt to improve performance. 
In addition, as stated in the above blog post, we could improve the pre-processing method by adjusting the sampling rate. As computer devices expect finite arrays, we discretise the speech signals by sampling values from our signal at fixed time steps.  However, this approach loses information if the sampling rate is low. Having better resources and storage, one could try to increase the sampling rate to obtain higher quality data.


