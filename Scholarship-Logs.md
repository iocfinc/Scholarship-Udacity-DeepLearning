# Deep Learning Scholarship Journal

## Hello, World

Hello, I am Ira and I have just been accepted to the PyTorch Deep Learning Nanodegree course by Udacity in partnership with Facebook.

## Deliverables

Looking forward to building projects again with the use of PyTorch. Particularly interested in deploying the sentiment analysis model through a website.

Here are the five projects, listed for convenience:

        Predicting Bike-Sharing Data
        Dog Breed Classifier
        Generate TV Scripts
        Generate Faces
        Deploy a Sentiment Analysis Model

## Day 1: January 31, 2019

I am starting late for the program. A week has passed and I am just getting in to the rhythm of doing additional work every day. I did a quick check again today on the program home to check the modules that are not yet 100% completed. My last Nanodegree course got credited so there are some projects that are already checked.

### Capsule Networks

I have been reading on [Capsule Networks](https://cezannec.github.io/Capsule_Networks/) today. Its some additional lesson added on this version of the Nanodegree that is unique. First off, what are capsule networks? Capsule networks are made up of capsules which in turn are collection of nodes which contain information about a specific part or property. To explain better lets have a use case. Capsules are suited for recognition problems in CNN. In CNN usually we do pooling so that our image gets to a smaller feature-level representation. Simply put it throws pixels away. Now for classification problems this would be okay to use. But in recognition this would be problematic. In recognition we want to detect a certain pattern for example a face. If we use pooling some data, spatial information, would be lost. Meaning the location and orientation of the eyes, ears, nose, etc. would no longer matter. This would be problematic because if we know that it does not look for spatial information (the location of the eyes relative to the nose and the ears) we can trick the model to still recognize a face by simply throwing a garbled image as long as it has the correct features. This is where the use of capsules come in handy. Capsules will look for the parts __*AND*__ represent the spatial relationship of those parts.

![PyTorch Capsule Netowrks](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bfdc81c_screen-shot-2018-11-27-at-2.40.04-pm/screen-shot-2018-11-27-at-2.40.04-pm.png)

In the image above we see the structure of a capsule network. The model will still detect a face but it knows that the face is made up of eyes and nose & mouth pairs which are in turn made up of other nodes.

![PyTorch Capsule](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bfdc9ca_screen-shot-2018-11-27-at-2.48.28-pm/screen-shot-2018-11-27-at-2.48.28-pm.png)

In the example above we see a sample representation of a capsule when used on a cat. As we can see, the output of the network would be a **Vector** which will have a magnitude and direction value.

* Magnitude (m) - The probability that a part exists; ranges from 0 to 1.
* Orientation (theta) = The state of the part properties.

#### Existence

Under the topic of Capsules is existence. This is the probability of a part being in the image or is detected. If we say the existence is 0.9 then we are saying that we are 90% confident that the part is in the image (for example the face). How do Capsule compute for existence? It would look at the different properties of the part (for a face it could be the position, width, orientation, color, height etc.) then treat these properties as an indicator of whether the part is in the image. Come to think of it, its like a sub-neural network or an NN-inception since the existence value is a function of the properties computed based on the detection of individual properties.

:book: [Cezanne Camacho's Capsule Networks post](https://cezannec.github.io/Capsule_Networks/)

Capsule networks are indeed interesting and there is more to them than what I have here. Its interesting to see new improvements to an old (although CNNs are supposed to be new) implementation of CNNs in facial recognition. Capsules have *dynamic routing* that was not discussed in Udacity but its implementation is similar to how we build a neural network.

## Day 2: February 1, 2019

Mostly doing some video watching on the Nanodegree. I was going over individual cards to see which are not yet 100% completed. Started to work on a side project that I have been putting off ever since I started to learn Deep Reinforcement Learning last year. Hopefully this time it works.

I have a lot on my plate right now and I am feeling a bit overwhelmed. But regardless we have to push on. The problems won't solve themselves so taking in the small wins are key to staying afloat. End of the log for now.

## Day 3: February 2, 2019

At it again with my side-project. Minimal coursework for today. I know I have to put in a few hours every week. Its just that I am focusing my attention on the side-project right now. Sorry Udacity. I'll be back again some time next week. Just to update I have made some progress in the side-project, its private for now but its shaping up quite well. Looking forward to being able to use it publicly. Next up is I am also finishing up Fast.ai's Lesson #2 for deep learning. Fast.ai lessons was one of the recommendations presented by my co-scholar Joan Cabezas. He had one of the highest accuracy score in the PyTorch scholarship challenge and I had a quick chat with him in LinkedIn on what other courses he had taken. Fast.ai was one of his recommendations. So here I am, finishing up Fast.ai in parallel with my Nanodegree and side-projects and work and life. 

Sometimes I feel like my head is a :bomb:. But then I just recall :dart:'s that I have and its a little bit easier. Going to take the :crown: one day but first we start as :beginner:s.

**Plus Esse, Plus Ultra**

Now off to do some house chores. :wink:

## Day 4: February 3, 2019

I have a flight today so I am squeezing in some lessons. For this day its **Transfer Learning**. It has been covered previously and its the approach that I used for the phase 1 challenge. In this lesson we went over how transfer learning works and what approach to take.

Transfer learning is basically using a pre-trained model and re-purposing it to the problem that we are trying to solve. The idea of transfer learning is that instead of re-training a whole new model from scratch we would rather make use of an already existing model although it has been trained for something generic. For example, the ImageNet classification problem models are trained over 1000 labels of images. Then we have a problem that we have to classify a dog breed. Knowing that the ImageNet did cover some images of dogs then we can think of just re-purposing that model into a tailored model for our problem. Below is the general approach as laid out in the course.

![Guide on Transfer Learning Approach](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5baa60db_screen-shot-2018-09-25-at-9.22.35-am/screen-shot-2018-09-25-at-9.22.35-am.png)(source: Udacity)

While it is true that transfer learning will help speed up the creation of models for specific problems, we also have to consider that its not a magic wand to wave around and expect it to solve the problem by itself. The background of what our problem is (what we are trying to do), the basis of our pre-trained model (what it was trained to do) as well as the understanding of how to approach the problem is needed so that the implementation of transfer learning is successful(in my opinion). Simply loading the model and feeding it our data *MIGHT* work, knowing the basis of the transfer learning approach would help us create a better model overall. I guess this is where wisdom separates itself from knowledge. It would probably take some experimentation on our part to fully figure out what works and what doesn't but having a cheat sheet like the one above surely won't hurt.

As is the case for all DL models, its all about the data we have at our disposal for training. We can be dealing with a large data set or a small data set. Then we can also have a model that is trained to solve a similar problem or a different problem. All of these contribute to the way we choose which approach to make. Generally, on what I have observed, small data sets tend to suffer from over-fitting (since we are mostly going to repeat them over the batches) so we freeze as much of the layers we can (fixed-feature extraction). For large data sets we have the luxury of re-training the model completely (we make use of the architecture but random weights initialized) or use the model as a starting point to improve the weights (fine-tuning).

:gem: Additional reads for transfer learning: [Paper on the transferability of features of pre-trained CNNs](https://arxiv.org/pdf/1411.1792.pdf)

## Day 8: February 7,2019

I am back from a trip. Took a break from work and from the nanodegree since Day 4. Went on a planned trip to visit my sister. Now I am back. I have no other trips lined up for this quarter so focus is now on this and on rendering the rest of my time before I line up another job that is actually related to AI-Deep Learning.

So I just got invited to the student leader's private group in Slack. There will be updates on the group so I am going to be more active. That is one of my goal for this Nanodegree, be more active especially in taking initiatives and have a good time with others in the area as well.

What is next for now? I am continuing Transfer Learning module for now, its 88% left its just the notebook for now.

:dart: 15 days to the Deadline of Project 3: LinkedIn update.

### Tangent: Forecasting at Uber

I have been reading up on this [article](https://eng.uber.com/forecasting-introduction/) from eng.uber.com regarding the application of forecasting to their business. Their use case includes both back-office (hardware capacity forecasting in their infrastructure) and front-office (ridership prediction).

![Uber's Ridership forecasting](http://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/image4.gif)

The example above is the result of Uber's marketing model for the California (SF bay area) ridership. The heat map shows where the demands are and this model helps Uber direct their drivers to hot spots. It serves two ways, one is that it helps the passenger get a ride faster. Second is that it allows drivers to pick up passengers quicker. Over-provisioning of drivers would lead to areas under-served as well as drivers which will pick up no rider. Under-provisioning would lead to unhappy customers since the bookings will take longer. These will result to reduced trust between Uber, the user and the drivers.

The article also brought to light the challenges that Uber is experiencing on forecasting. For one thing, similar to the bike-ridership data set, the seasonality of the ridership is one challenge. Spikes in ridership happen on weekends, holidays and the December (related to the holidays). Uber's model should be reliable enough that it can still work even with these challenges. Another set of challenge for Uber is that it also has to deal with physicality in its approach for modelling. I think the physical difficulty is very different and somehow unique to Uber or to ride-hailing apps for that matter. What we mean by physicality is that Uber would have to deal with the physical constraints of the roads and the area it serves. It has to share these constraints to variables outside its control. For example, the roads can only have a certain amount of vehicles on them. Uber has to account for the fact that these roads will most likely also get filled during peak hours (rush-hours) and holidays(shopping traffic). Through this unique set of challenge Uber's forecasting model not only has to deal with temporal predictions but spatio-temporal.

## Day 10: February 9,2019

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [ ] Personal Project (1 hour at least)
* [ ] Nanodegree project (LinkedIn and Github)
* [ ] One Chapter Story Telling with Data

Back to the tangent on forecasting at Uber, we have already learned about what Uber is using forecasting for and what challenges they are facing that is unique to them. Now we move on to the approach they are making for their forecasting. They usually make s of model-based or causal classical, statistical methods and machine learning approaches. Model-based approach is used when the underlying mechanism, or physics, of the problem is known. As the name suggests, there has been a pre-determined causality between the inputs to the model and the inputs (variables) are already known. An example would be a fare-matrix where the estimated amount displayed on booking is determined by the approximate distance between the points of travel as well as the estimated time taken for the trip plus the service charge. If all those variables are present then the fare estimate can be displayed on the booking screen.

When the underlying mechanisms are not know or too complicated, the use of a simple statistical model is usually better. Examples of the use case would be the stock market (price, preceding price, current news etc.) or retail sales (demand, supply, time of purchase, seasonality, current trends). Classical methods that are applicable for this are ARIMA (autoregressive integrated moving average), exponential smoothing methods (Holt-winters), the Theta method. These methods have been proven, by Uber, to be enough and applicable to their data.

With the advent of machine learning, new approaches have been tried. For example Quantile Regression Forests(QRF), has been added. Recurrent Neural Networks (RNN) have also been useful, but only with sufficient data is available especially exogenous regressors. Exogenous here meaning the outliers or those events that has zero correlation to the data, in time series this could be a Concert or an accident, a sale or a payday rush, some holidays. These are not really normal but RNNs can be used if enough of these regressors are made available. As expected, these machine learning models tend to be black-boxes and Uber only uses them when interpretability is not a requirement.

__*The bottom line is that Uber, or us for that matter, does not know for sure which model or approach would result in best performance*__. This necessitates the comparison of models and their performance on multiple approaches. For this, Uber has created its own framework on comparing these forecasting methods.

Chronological testing is important for this matter since the ordering of the data in time series is relevant. We should not cut out a certain sample or random samples in the middle of the data set and use that as our test data and then train the model on the remaining data. The data set requires that the training be done only on data prior to the time of the test data.

![Windowing technique](https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/image6-e1536165830511-696x184.png)

The image above shows the two main approach used when testing and training data, one is the _sliding window_ and the other is _expanding window_. For sliding window a set of time is used as the training data and then a forecast is determined. After the sliding window has been trained, the window shifts (slides) to include the most recent data and from there update the model for forecasting. The previous data already outside the window are dropped and are not used in training the current model. For the expanding window, there is no dropping of data. All previous data are still used to come up with the updated model after the latest data is made available. For sliding window we lose memory by dropping previous data, for expanding data we introduce complexity in computation since all data points are used. When there is a limited amount of data to work with, expanding window is preferred. The implementation suggested is the marrying of the two methods, expanding and sliding. At the first points and expanding window is used then after a set length has been achieved the window becomes sliding. Thing of it a the same as convolution without padding but not quite.

In terms of evaluation metrics there are a lot of possible values suggested, absolute errors and percentage errors. One useful approach would be to check the model's performance against the naive forecast. For non-seasonal series, this is going to be equal to the last value. For a periodic time series, the forecast estimate is equal to the previous seasonal value (for a daily time series on a monthly period its going to be the value of the same date at the previous month). For Uber's case, they are going to be testing many models for this and they have come up with [Omphalos](https://eng.uber.com/omphalos/) which is a back testing framework for rapid iterations and comparisons of forecasting methodologies.

Determining the best method for the use case is just half of the equation, the prediction interval should also be considered. This is going to be the degree of error we provide for our model under a high probability, think of it as a standard deviation allocation for our data. The higher the prediction interval we use, the larger reserve we have to allocate for our estimates. Prediction intervals are of the same importance as you point forecast (the actual value we have forecasted) and should be included in the graph.

![Uncertainty Estimations](https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/image2-696x203.png)


In closing, forecasting is critical for building better products overall. It will lead to better user experience and ensure success of global businesses like Uber. The next part of this forecasting series is the introduction to pre-processing.

Additional Materials :books: [Forecasting: Principles and Practice](https://otexts.com/fpp2/)

## Day 11: February 10,2019

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [x] Personal Project (1 hour at least)
* [ ] Nanodegree project (LinkedIn and Github)
* [x] One Chapter Story Telling with Data

Finished Chapter two of Story Telling with Data today. It talks about the most commonly used graphing tools or charts and their use cases. It share the good pointers to remember when dealing with these charts and what not to do. It also shows important examples of how to use more powerful charts being able to tell a better story than the default ones in Excel. Looking forward to the next chapter on this book. Up next is some time on my personal project but first I have to do some house chores.

I am now working on my personal project. I can't provide details as it is not in line with this. I am simply logging it in so that I can track the time I have spent outside of the nanodegree but within the bounds of AI-DL-ML.

## Day 13: February 12,2019

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [x] Personal Project (1 hour at least)
* [ ] Nanodegree project (LinkedIn and Github) :dart:
* [x] One Chapter Story Telling with Data

Right now I am watching Weight Initialization lesson for the nanodegree. I have a timeline to finish this within March 2nd week so more focus would be added to this. Allocation of hours would be increased so that I would meet the deadline I set for myself. This is going to be doable.

So back to Weight Initialization lesson. I am learning about how to set the initial weights for a model (un-trained). There are different effects on the training portion of the model depending on how the weights are initialized, more on this later. The idea is that to set weights initially, we can do them at random, start at zero or start at a very big number.

First we discus the case of all zeros and all ones initially. The resulting graph (in the notebook) showed that for all ones initialization the initial loss was very high but it gradually decreased over time. In the case of all zero initialization, the loss was at zero and remained there constantly. While its great to see that our loss was very low, it actually meant that the model was not able to learn anything. It would just constantly a guess over and over again. 

**This is actually a new learning for me in this topic.**  The idea is that having an initialized
value for the weights kept at a constant means that the model is already set to fail. Recall the
idea of the perceptrons as gates or switches. They are used to control how much effect a single
input has on the final output. Now imagine if you were the learning algorithm for this model and
you see that all the values of your switches are the same. It like giving you a switch board
without any labels on what each switch does. This obviously is not a good scenario to be in.
Obviously you will have to check each switch and figure out how they affect the final output of
the model you are trying to fix. As discussed in the results of the notebook for this lesson,
the all one initialization had a hard time figuring out which switch does what initially and
therefore had its loss increase instead of decrease. In terms of the all zeros, it would seem
that the learning algorithm plainly gave up and just picked an output and give that as the
result for every other input it is given, it did not learn. So obviously a constant value
initialization is bad for the model and giving it a value of zeros is worse.

A quick tangent on how I see the zero vs. one initialization. Obviously there is a polarized result when the model was initialized as zero and when it was initialized as one. When it was all ones the model had a hard time figuring out what the weights should be so it had to experiment for a bit and figure out which weight (perceptron actually) was contributing to the loss on the model. Eventually though, the model with all ones did start to learn and decrease its loss value as the epoch went on. In terms of all zeros the model simply did not learn and instead chose one output and repeated it for every possible input. This is how I see it. Having all ones is like starting out a job and all the training modules are dumped on your lap and you have to read all of it. Not yet having the experience to know which action should correspond to the required solution you have a hard time figuring out which Knowledge manual to use. Sure you will make mistakes but over time you get to improve because you get to know which manual goes to what issue. While its not perfect, its also not bad in the sense that you at least have a chance to figure out which information to keep and which to discard. In terms of having an initialization of zeros, this would be the same as being hired for a job and already be asked to solve complex problems without prior training and access to the knowledge base. How would you feel? For the all ones at least they have all the manuals, they just have to get experience to figure out which to use. For the case of all zeros, they have nothing and the programs that they are (find the easiest way to do things) they would simply pick an action and repeat that. That is actually worse that guessing by the way. But that is how I picture it. But back to the topic at hand.

To solve the problem exposed by uniform value initialization we have to introduce a randomness in our weight initialization. In this way, even if some weights might be the same at first they should be in a number that will be manageable to the system. Think of it as having two switches that are un-labeled, that would be easier to deduce which switch does what instead of having them all un-labeled. In this way, our model has a chance to learn from its mistakes. This is done through the use of randomly setting weights. We can choose to distribute the weights uniformly or via normal distribution.

Now the question would be the possible range of values that we can use for the initial values. Note that choosing a very large value is not recommended for the model. Instead, the range of values is actually correlated to the number of inputs for the perceptron, in this case inversely correlated. The recommended range would be $[-y,y]$ where $y = 1/\sqrt{n}$ and $n$ is the number of inputs to that given neuron. So it is going to be a relatively small number but also not zero ($0$) otherwise it would also not train.

Now on to the distribution of the weights from the range. There are two types of distribution we can look at, **uniform** and **normal**. From what I recall in *engineering statistics* uniform distribution would be the same as equal distribution and will tend to have a histogram plot more of a rectangle. For normal distribution, the distribution is more of a bell curve. Meaning that majority of our values would be centered near the mean of the range which in this case of $[-y,y]$ would be at zero. From the model in the notebook there are some slight benefits on using normal distribution in selecting the weights when compared to uniform distribution. This does scale up the more complex our model becomes so the advantage actually goes to the normal distributions instead of uniform distribution.

Now that we have discussed the relationship of the range and the initialization of weights as well as the distribution of weights for the model I can start summarizing it in my own words. While it might seem that it is a not a good idea to provide have an all zero initialization for the weights, the normal distribution is preferred which is somehow a contradiction considering that the center of the distribution for a range would be the mean and with a $+/-$ range that would be at zero. This is could be clarified by the wording. We are actually distributing the values centered around the mean (0) but we are not initializing them all at zero. We want an initial weight that is small but not zero and normal distribution gives us exactly that. More values would be near the mean (0) for normal distribution. Also, we talk about the range values of the weights for a perceptron. This is inversely correlated (not proportional since its $1/\sqrt{n}$) to the number of inputs for that perceptron. So for a huge network with perceptron and input values of 256 or 512 or 1024 or more this would be a very small number indeed.

Next up would be auto-encoders but that is for later in the evening. For now I need to rest a bit. :smile: Or else my head will be a :bomb:. 


## Day 14: February 13,2019

**:smirk: TARGET END: March 2nd week. :gem:**

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [x] Personal Project (1 hour at least)
* [ ] Nanodegree project (LinkedIn and Github) :dart:
* [x] One Chapter Story Telling with Data

The lesson right now is Autoencoders. Simply put its going to be a compression and decompression using neural networks. This is building up on the idea that multiple features passed through a neural network can be represented by a reduced representation of it. Two things should be clear after this, we can reverse the order by decompressing a neural-network-reduced input and that we can use neural networks to reduce the representation of a data with minimal loss.

Recall style transfer, the idea there was that the CNN was looking for the style of the base art through the weights after passing through some layers of perceptron. Basically it was simply finding out what features was relevant to the original image, like finding out what hue was the dominant one used by the artist or are there any shapes that feature distinctly on the artwork or how was the stroke of the brush applied on the canvas. These things can be represented by the set of weights in the CNN. In style transfer the next step after finding these representation was to apply the *style* of the original image to the new image by passing through the new image as input and adjusting it in such a way that the weights match to achieve the same effect as the original artwork. In autoencoders however, instead of applying the new image to the same network we are going to take the reduced representation of the original input and pipe that through another network that has the same setting of weights but in the reverse order (decompression). What would happen is that the dimensionality of the reduced representation should increase and since its just the same network in reverse it should come out as the same size as the original. The idea here therefore is that we should reduce the total loss of the network from input to output. As with anything else, there will always be a loss in the transaction seeing that we did perform a process on it. The computations would eventually introduce some errors.

Now there are multiple examples of autoencoders available today. JPEG, MP3, MP4, MOV these are basically autoencoders. They take in the original data and reduce it in a simpler/reduced sized form which can be decompressed later for accessing. The main difference is that these formats are written explicitly by humans while our NN autoencoders have their weights set by the algorithm. One very good use case for use of NN autoencoders would be de-noising and image transformation. For example, we can make an autoencoder that will take on a black and white image and output a colored version of the same image.

## Day 17: February 16,2019

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [x] Personal Project (1 hour at least)
* [ ] Nanodegree project (LinkedIn and Github) :dart:
* [x] One Chapter Story Telling with Data

So I have finished the Autoencoder section, that completes the convolutional neural network portion. What's left now is the project which is to critique my GitHub profile for a good portfolio representation.

Back to autoencoders, the last topic was on Denoising application of autoencoders. In this application we are able to input a noisy image and through the autoencoder we should be able to transform it to a less noisy and legible version of that image. The pipeline for this model was that the target images are the same as the input images, the only difference is that the inputs are first processed to add random noise to them. Once the noisy inputs are created it is then fed into the autoencoder which takes care of the processing of the input to the desired target. Now it is important to note that the random noise generated is not part of the trained network, only the autoencoder is trained. What happens is that the auto-encoder will compress the images and then decompress it. The idea behind it is that the majority of the magic will happen in the decompression. By knowing the target images, the weights in the decompression portion of the auto-encoder gets trained to reproduce the original image and account for the losses or noise in the input. This is almost similar to a simple autoencoder discussed earlier where the data is transformed and then recreated and both the input and target are the same images only with some losses but with a size reduction in the middle. The use of autoencoder on the former excercise was mainly for compression in this exercise it is mainly for denoising.

The use case for this would be, if I understand the brochures correctly, the AI correction of images in smartphones right now. We all know the branding of phones as AI camera. This is a use case of autoencoders. A distinct example is in fixing low-light images which are some of the selling points of the AI-camera's today. The AI portion is that the processing of images were trained to produce an image that will heighten the features supposedly in the images and reducing the low light effects and highlighting the subject.

## Day 18: February 17,2019

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [x] Personal Project (1 hour at least)
* [ ] Nanodegree project (LinkedIn and Github) :dart:
* [x] One Chapter Story Telling with Data

I had some progress in the Objectives. I was able to submit my GitHub page for review. Received feedback to Correct my Readme files so that it shows how to use the projects. After that not much else done for the day.

## Day 19: February 18,2019

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [x] Personal Project (1 hour at least)
* [ ] Nanodegree project (LinkedIn and Github) :dart:
* [x] One Chapter Story Telling with Data

Received a response regarding an application I had. I did not get considered. But it was eye opening to me. The fact that I struggled to produce an output shows that there are gaps in my skills and those I need to work on. Onwards. Keep moving.

For now I am dissecting a paper on AI to fix night images. The GitHub page is [here](https://github.com/cchen156/Learning-to-See-in-the-Dark). Also, there is a paper and its quite interesting. It details how the researchers defined the pipeline and how their method compares to others. Imagine using this and building on it. We can possibly create a better processing pipeline for video feeds at night/low light. If we can achieve better fidelity on those feeds then we can possibly achieve higher accuracy results for example in object detection and identification.

I am actually enjoying this paper. I see an application for it. Plus it is a good showcase of autoencoders and deep learning so it would be a big boost when I can implement it in PyTorch. I have to schedule it sometime next week. For now I have to finish first on my LinkedIn Project so that I can tick it off my objectives for days 10-20.

## Day 20: February 19,2019

OBJECTIVES FOR THIS WEEK (10-20):

* [x] Finish up the tangent on Uber's forecasting
* [x] Personal Project (1 hour at least)
* [x] Nanodegree project (LinkedIn and Github) :dart:
* [x] One Chapter Story Telling with Data

Completed the review of my LinkedIn page by the careers office. They have a lot of changes they want me to push to my page. Its all fine as they are for the improvement of my profile. I will gradually roll them out to both the GitHub page and the LinkedIn page as well. For now I shall have a new set of objectives for day 21-30.

I am just about to finish reading the paper on "Seeing in the Dark". What they have done is take the RAW file from the camera sensor and apply it to a CNN-autoencoder which ultimately outputs an sRGB image that has been corrected. Their result is seen below, which was taken from their github repository for the project.

![Sample Results from Author's Github repository](https://raw.githubusercontent.com/cchen156/Learning-to-See-in-the-Dark/master/images/fig1.png)

So why did they start doing this project to begin with? Basically they want to solve a problem in the digital photography community where low-light images are very difficult to take. Low-light images are those that are either taken at night or taken when the subject does not have enough sources of light illuminating it. What happens is that due to the lack of photons, its difficult for the features and colors of the subject to come up in the resulting image. There are some ways to workaround this issue. One would be to increase the ISO setting of the camera but that would lead to color degredation and the resulting image's color is not the same as the original subject. Another would be to increase the exposure time for the image but that has a problem of blurring and adding noise to the output. Also, the current existing options to resolve the issue cannot be transferred to video. The researcher's solution was to create a pipeline that will take in the RAW input from the camera and then output an sRGB image that will have a high fidelity to the original subject and give as much detail as well.

There is also a technical discussion on the paper as to why low-light images are hard to process. One is that with low-light there is low photon counts that will lead to less activation in the sensor, photons are what transmits luminance and chrominance to our eyes or in this case the sensor. Another challenge with low light image is the low Signal-to-Noise(SNR) ratio for the data, meaning that it is much harder to figure out which values are signal and which were just noise introduced to the data.

The pipeline as introduced by the researchers would take in the input via Bayer raw data, from there the red-green-blue-green layers are taken and the black levels are subtracted from the layers. From there an amplification $\gamma$ is introduced which would increase the value of the original data. From there it is fed into the Convolutional network which would output the RGB layers, from there the image is merged to create the final RGB output. Note that the number of layers as well as the dimensions are changed as it goes through the pipeline. The original dimensions for the RAW file are $H \times W \times 1$ in the Bayer form. From there the individual components are taken and the dimensions are half of the original $H\over{2}$$\times$ $W\over{2}$$\times 4$, the 4 here indicating the 4 sensors from the Bayer filter of RGBG. The resulting output of the ConvNet is $H\over{2}$$\times$ $W\over{2}$$\times 12$, the 12 here is now in the form of RGB with 4 sets for a total of 12 channels. Finally, the 12 channels are transformed into $H \times W \times 3$, in this case the original dimensions of the image is back but with the layers in RGB.

Using the researcher's pipeline did show a qualitative improvement from the current processing method. What they did to compare was present the image outputs of the control method and their own pipeline and used a mechanical turk to rate which image produced a better result. There are still improvements to this area of research and the authors did leave a link for the dataset for future researchers to use. One interesting study for this research is on testing the pipeline to generic sensor inputs. For this paper what they used was only Sony and Fuji sensor results but they did try generalizing the model and using RAW files from other cameras with promising results. Great read and very promising research in my opinion :thumbsup: :clap: :bulb:

:bomb: Do a code review on the source code on how they implemented it. I already see the autoencoder portion during the first scan of the train module.

## Day 21: February 20,2019

Another 10 day period for this course. All focus on finishing up the objectives. I again have slowed down progress on the personal project I was working on earlier. :s Below are the target objectives for the coming days, most of them would be to cover ground for 100% completion of the nanodegree. Then I have also added some portion for updates on my presence in LinkedIn to get additional views and possibly get interviews. I do not have a personal website so I have to make LinkedIn work for me.

* [ ] :dart: Complete the autoencoder paper on "Seeing in the Dark"
* [ ] :dart: Re-read "jobs in Deep Learning" section for better targeting and updates on the LinkedIn page.
* [ ] :dart: Complete the Recurrent Neural Network stragglers: **"Implementation of RNN&LSTM, Embeddings,Sentiment Analysis, Attention"**
* [ ] :dart: Update experiences and projects in LinkedIn Pages.

## Day 22: February 21,2019

OBJECTIVES FOR THIS WEEK (21-30):

* [x] :dart: Complete the autoencoder paper on "Seeing in the Dark"
* [ ] :dart: Re-read "jobs in Deep Learning" section for better targeting and updates on the LinkedIn page.
* [ ] :dart: Complete the Recurrent Neural Network stragglers: **"Implementation of RNN&LSTM, Embeddings,Sentiment Analysis, Attention"**
* [ ] :dart: Update experiences and projects in LinkedIn Pages.

[Paper](https://arxiv.org/abs/1707.09482)

First off for this day, I need to get more organized. I have a lot of objectives to fulfill for the next three months. This is both on the professional and personal commitments. I will be having some increased workload at the office in preparation for a move. This will be on top of my current workload which is erratic at best so I there would be less time to spend doing something on the side like reading papers and code while in the office. This would mean that I have to budget my time outside of it. For now I have 3 main objectives personally, one is to complete the Nanodegree scholarship, another is to continue looking for a job and preparing for an interview, finally I want to finish my side-project. So my idea would be to stick to a schedule for the coming weeks so that I can slowly tick of one of the objective which is to finish my Nanodegree. Once we clear that up I would be able to spend some more on the remaining two objectives. This will be tight but I have made it work before, I'll make it work again.

I have just marked off my task on autoencoder paper "Seeing in the Dark". Earlier I have finished the *Implementation of RNN & LSTM* portion. For now since I can't open the Udacity site to proceed with the stragglers I will be doing my lesson plan for the transition.

Back to the nanodegree, this lesson would be about Embeddings and Word2Vec, so its mostly Natural Language Processing. First up would be embeddings or Word Embeddings where the model learns to map vocabularies through their vector values. For this lesson the focus is more on learning Word2Vec embedding which focus on mapping words to embeddings that contain semantic meaning. Embeddings can learn the relationship between tense or relationship between gendered words. Embeddings are simply getting the mathematical relationship between words in a vocabulary. One thing to note when using embeddings is that it is highly reliant on the source text, specifically its the biased or incorrect mappings from the source text that would lead to errors in the mappings.

Embeddings are just building on the earlier concept of learning through numerical data that has been used in ANN and CNN (pixel values). In this case it is just expanded to the application of words. Earlier in the course we have been able to process text by using one-hot encoding. The problem with one-hot encoding is that it will lead to a very large matrix size (depending on how big the vocabulary for the soruce text is) but only one will have a value with the rest being zeros. This would be computationally inefficient. Honestly, I am having a hard time understanding this part. What generally happens is that from an input word, it gets tagged into a set of weights (is that it?). I have to pause for a while and read some more on embeddings.

## Day 22: February 21,2019

OBJECTIVES FOR THIS WEEK (21-30):

* [x] :dart: Complete the autoencoder paper on "Seeing in the Dark"
* [x] :dart: Re-read "jobs in Deep Learning" section for better targeting and updates on the LinkedIn page.
* [ ] :dart: Complete the Recurrent Neural Network stragglers: **"Implementation of RNN&LSTM, Embeddings,Sentiment Analysis, Attention"**
* [ ] :dart: Update experiences and projects in LinkedIn Pages.

I had to stop the embeddings portion earlier, I had a hard time understanding how it works conceptually. For now I am reading an [answer in Quora](https://www.quora.com/How-does-word2vec-work-Can-someone-walk-through-a-specific-example) for this problem. Reading that post as well as the reference post used in that answer from ["Why word2vec works?"](http://andyljones.tumblr.com/post/111299309808/why-word2vec-works) I sort of got the idea. What generally happens is word2vec is looking for probabilities. Suppose that we have gone over the entire corpus of text as our dataset, we would be able to get an idea of how much a word has appeared and who it appeared closely to. *This is how I understood it:* Since we now know how much a word has appeared we can get the probability of a word in relation to another word (context) which makes is a conditional probability. From here we can get the conditional probability of a word appearing WITH another word. To simplify I'll also use the same example as the one in the posts, the probability of word $w$ appearing in relation to *king, queen, man and woman*. For a word $w$ and another word $w'$, the probability of $w$ appearing close to $w'$ can be denoted as $P(w'|w)$. This is where it gets interesting. Now let us define our problem *"king is to queen as man is to woman"*, we can interpret this statement as:
```math
{P(w'|king)\over{P(w'|queen)}} \approx {P(w'|man)\over{P(w'|woman)}}
```

This is assuming that the text corpus was not biased and the context is not skewed. As we can see here, we can actually now define the relationship between the words just by getting the conditional probability by going through the text. How does this help us? For example in verb tenses or other context? Simply put, we just have to balance the equation. For example we did not know what word would relate to man or *"A is to B as C is to ________"*. What we need now is to find from our text data the words that will make the equation true.

```math
{P(w'|A)\over{P(w'|B)}} \approx {P(w'|C)\over{P(w'| ? )}}
```

Since we already have the values of conditional probabilities of $w'$ with respect to $A$ and $B$ we can evaluate:

```math
{P(w'|A)\over{P(w'|B)}} = X
```

From there we can therefore find the value of $P(w'|?)$ from the entire vocabulary or data set that will satisfy the equality:

```math
X \approx {P(w'|C)\over{P(w'| ? )}}
```

## Day 26: February 25,2019

OBJECTIVES FOR THIS WEEK (21-30):

* [x] :dart: Complete the autoencoder paper on "Seeing in the Dark"
* [x] :dart: Re-read "jobs in Deep Learning" section for better targeting and updates on the LinkedIn page.
* [ ] :dart: Complete the Recurrent Neural Network stragglers: **"Implementation of RNN&LSTM, Embeddings,Sentiment Analysis, Attention"**
* [ ] :dart: Update experiences and projects in LinkedIn Pages.

What do you do when you are on 24 hour standby? :smirk: Now I am reading through (again) the post on how to land a job in deep learning. One thing they recommend is have a routine. Similar to Deep Work. In this case we would want to read up on the latest on the field, Twitter is a good way to start with all the recent developments. Next would be to browse through the applications of Deep learning to common Machine Learning roles. Deep learning is built on top of Machine Learning. This part of the routine allows us to find out how our deep learning skills can be applied and also audit our current roles such that we can study which skills we have to work on. Basically, look for postings on skills instead of a specific role/job title. Third on our routine should be on building our own deep learning application and share it. We can create a model and then deploy that to an app either via a website or as a service for an Android or iOS app. To one up this step we can write about our process in creating the project, the reason or objective for the project and what we have learned during doing the project. Finally, in line with item 1 is that we want to be reading up on recent papers on the field and write our own take on it or at least try to explain it on our own way. Maybe we can find an implementation or use case for the latest development in the paper. It works 2 ways, first is that you learn more about the latest development and improve your skill. The second is that you show others in the field as well as prospective recruiters that you have communication skills.

Moving on, I submitted some applications in Monster and TechinAsia website. Mostly Deep Learning, AI, Machine Learning roles. Hopefully I get some responses. I did some review of my Repo in GitHub and there are some improvements. First of would have to be to create a simple file/python script that will just load the model/trained path instead of actually running the entire program/notebook to demonstrate the function of the project.

I also did some sample coding for auto-form fill up script using PyAutoGui and Pandas. It will simply load the file(csv) and write it to the corresponding field. Its actually part of a freelance job that I am trying to take to have additional income. Later today I would also be handling the transition training of my backfill so I have to review some tickets.

Back to the nanodegree, I am now on RNN. I viewed the remaining coursework and decided how to reach the 2nd week of March target. First off would be to finish the remaining topics in RNN which is currently ongoing. The next part would be to do the deployment of model first and skip GANS completely. I already made the project in GANS so the remaining parts there are on the difference in PyTorch implementation. In my case the deployment of model to AWS is more important.

So what did we cover in RNN today? Its mostly on subsampling. Its basically reducing the "noise" of our data by removing the fill words that are often not adding any context to our data. Examples of these words would be _is, the, it, are, were_. The process of subsampling has been defined by the paper of Mikolov et.al. and the general formula would be:

```math
P(w') = 1 - \sqrt{t\over{f(w')}}
```

The value of $P(w')$ is the probability of the word $w'$ to be dropped from the vocabulary. The notation $f(w')$ corresponds to the frequency of the word $w'$. This would make $f(w') =$ $w'\over{len(vocab)}$. The higher the frequency of the word $w'$ the more likely it is to be dropped based on the subsampling formula. The value of $t$ is usually set between 0.0001 to 0.00001 and this corresponds to the threshold we are setting for our dataset vocabulary.

Next up would be Context Word Targets. In here we get to know how to set our context, which is actually just setting a window size that we think would best describe how specific our context would be. First off is that we define our **target word**, in this case **it is the word where we want to check the context of**. Since this is an RNN this target word is going to actually be an index of the word ID we want to check. **From that target word the idea is we get context from words prior to it as well as words succeeding it**. This is where the Context window concept appears. To note here is that we cannot set our window too big as that will return words that are no longer in context with our target word.

Okay, now I think I have the idea of the Word2Vec model. This would be building on top of the concept of Context window. The input for Word2Vec, if I understood it correctly, is the Context window. The target of the Word2Vec model would then be the Target word. Between the Context Window words and the Target word is the embedding which are the weights which allows us to get the relationship and context between the words. The target words are on softmax activation so we want their probabilities to show and we want the model to produce the highest probability which is done in backprogpagation. Once we have trained the full embeddings over the text we can drop the output layer (softmax) and make use of the input and embedding layer. This is actually a portion of the autoencoder model, in this case we are just taking up half of the autoencoder. We either use multiple input until the representation (middle) or the representation (middle) to the output layer. Basically one-to-many, for generation, and many-to-one, for context and text labelling.

![mccormick ML Word2Vec Network](http://mccormickml.com/assets/word2vec/skip_gram_net_arch.png)

So the above is the general network representation of a Word2Vec model. The input words are 1-hot encoding of the target word (middle word) and the output layer is a softmax activated 1-hot encoding of the context word. Recall that we had a window before of the words appearing before and after the target words. The input and output of the Word2Vec is simply the pairing of these words. The sampling and how the window works is described in the image below (taken from [mccormick posts](http://www.mccormickml.com))

![mccormick ML Word2Vec Sampling](http://mccormickml.com/assets/word2vec/training_data.png)

The window would simply go over the entire text corpus, or rather the ID set, and log the target and context words from there. Then we simply feed the target word and the output pair to the input and output slots of our model. We then train the model using these word pairing which represents which words come up close to the target indicating they are of the same context.

In training the idea is we want to have the highest probability of the softmax to the output paired word, for the example (brown,the) the word 'the' should have the highest probability for that input-output pairing. This scenario is for training only though. The idea we have for Word2Vec is that we want to simply make use of the weights and we would actually get the topK probabilities during evaluation. Meaning we would want to output an actual subset of words closely associated with the target word. In the example above we should at least have the subset (quick,brown,over,jumps) when we evaluate the top words related to (fox).

## Day 27: February 26,2019

OBJECTIVES FOR THIS WEEK (21-30):

* [x] :dart: Complete the autoencoder paper on "Seeing in the Dark"
* [x] :dart: Re-read "jobs in Deep Learning" section for better targeting and updates on the LinkedIn page.
* [x] :dart: Complete the Recurrent Neural Network stragglers: **"Implementation of RNN&LSTM, Embeddings,Sentiment Analysis, Attention"**
* [ ] :dart: Update experiences and projects in LinkedIn Pages.

>I might have made a mistake in my understanding of embedding. The input is still the entire vocabulary, similar to one-hot. The embedding is what we can change the dimensions of. So what we get is the weights of the hidden layer which is the embedding. After we have the weights, we can actually validate our data by providing the **cosine similarity** between the words. In this case cosine similarity is simply measuring how close or opposite two words are (in the data set) based on their relative vectors towards each other. -- **Deprecated**

![mccormick ML Vector representation](http://mccormickml.com/assets/word2vec/word2vec_weight_matrix_lookup_table.png)

We have said previously that the goal or word2vec is to get the vector representation of the vocabulary. The image above actually shows how this was achieved. We were able to get the vector representation by thinking of our words and weights as pairs. But this is a matrix right? Why did we name it vector representation? The answer is that the one-hot encoded words are exactly that one-hot. Only one value (the word ID) is one and the rest is zeros. If we multiply them we are simply getting the vector (weights) corresponding to that word. What we virtually have is a vector lookup table for our words in the vocabulary similar to the one below.

![mccormick ML word vector](http://mccormickml.com/assets/word2vec/matrix_mult_w_one_hot.png)

One important thing to note here is that the original text corpus plays a very important role in the creation of a robust model. As with anything in deep learning or machine learning, our model would only be as good as our data. When our data is biased to begin with we will see that it our model. Note that its not just the bias that will lead the model to provide different results, we should also consider representation or more specifically under representation in the words in our data. We simply cannot expect it to have great context and relations built when the word is seldom used. What I am trying to say in here is that we should set our expectations accordingly since deep learning is not like a magic potion that will solve our problems. It will have its limitation and by knowing these limitations beforehand we can have a better understanding of our models.

Now on to the lesson, the next one is on *Negative Sampling*. The authors of Word2Vec actually addressed the problem of training a huge Neural Network by releasing another paper. Their first suggestion was to use subsampling which we have discussed yesterday where the more frequent words have a higher chance of being removed from the vocabulary. Another improvement they did was to introduce the concept of negative sampling. When we look at the output target only 1 target word should have a value of 1 and the remaining words zero this is similar to one-hot encoding where our output is simply the length of the vocabulary. If we were to use backpropagation on the entire vocabulary output length then we would be having millions of weight to update (assuming we have 100 vocabulary words and 1000 neurons on the embedding layer). Since only one target word is going to have to be one only a few of the neurons need to be updated dramatically but we still have to back propagate the entire neuron network. With Negative Sampling we simply choose a few words to represent the incorrect answers and perform the weight updates only on the target word and the negative samples. Note also that the negative sampling is referring to the fact that we are sampling only a few of the negative words together with our target positive word. [Here is an article with a simple explanation of what Negative Sampling is](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/). In the paper, it there is also a suggestion on how much words we sample. For big data sets we can get away with 5 words. For smaller data sets we have to get anywhere within 5-20 samples.

How do we select which samples are in Negative Sampling? The negative samples are chosen from a "Unigram Distribution", where more frequent words are more likely to be selected. In this case its a function of the frequency of the current word in the corpus over summation of frequency of words on the corpus. The researchers did some testing on the unigram distribution model and they found that raising the terms to 3/4 provides the best results (see equation below).

![https://towardsdatascience.com/hierarchical-softmax-and-negative-sampling-short-notes-worth-telling-2672010dbe08](https://cdn-images-1.medium.com/max/1600/1*1K2rTosOIXe4iB9PMOQuIQ.gif)

## Day 28: February 27,2019

OBJECTIVES FOR THIS WEEK (21-30):

* [x] :dart: Complete the autoencoder paper on "Seeing in the Dark"
* [x] :dart: Re-read "jobs in Deep Learning" section for better targeting and updates on the LinkedIn page.
* [x] :dart: Complete the Recurrent Neural Network stragglers: **"Implementation of RNN&LSTM, Embeddings,Sentiment Analysis, Attention"**
* [ ] :dart: Update experiences and projects in LinkedIn Pages.

To round out the word2vec topic here is a [post](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/) from mccormick ML about some applications of Word2Vec on advertising and recommenders.

Moving on we move to the topic of attention in Deep Learning.

References:

McCormick, C. (2016, April 19). Word2Vec Tutorial - The Skip-Gram Model. Retrieved from http://www.mccormickml.com

## Day 33 March 4,2019

OBJECTIVES FOR THIS WEEK (21-30):

* [x] :dart: Complete the autoencoder paper on "Seeing in the Dark"
* [x] :dart: Re-read "jobs in Deep Learning" section for better targeting and updates on the LinkedIn page.
* [x] :dart: Complete the Recurrent Neural Network stragglers: **"Implementation of RNN&LSTM, Embeddings,Sentiment Analysis, Attention"**
* [ ] :dart: Update experiences and projects in LinkedIn Pages.

First off I'm applying to some possible offerings in ML/DL. Also, finished attention finally. I did some other applications in ML/DL hopefully I get some callbacks. Right now, since I am a bit behind, my goal would be to move towards AWS deployment with SageMaker. Model Deployment for this week. Build and deploy to production. Learn a few techniques on how to do it.

Its Day 33, and its March. 65 Days to go. I am excited to be out and free again. Pursuing what I would like to do. Hopefully I get to build things and be happier than right now. That's all we could ever ask for really, right? :smirk:

OBJECTIVES FOR THIS WEEK (33-40):

* [ ] :gem: Update experiences and projects in LinkedIn Pages.
* [ ] :gem: Finish *Introduction to Deployment*
* [ ] :gem: Finish *Building a Model with Sagemaker*
* [ ] :gem: Proceed to *Deploying and Building a Model*
* [ ] :gem: Follow that with *Hyperparameter Tuning*
* [ ] :gem: Lastly *Updating a model*
* [ ] :bomb: All leading up to **Deploying a Sentiment Analysis Model**

Here is an interesting article on [pricing algorithms learning to collude](https://www.popularmechanics.com/technology/robots/a26309827/left-to-their-own-devices-pricing-algorithms-resort-to-collusion/). The article is based on [this study](https://voxeu.org/article/artificial-intelligence-algorithmic-pricing-and-collusion) and it shows that AI price setting algorithms can and will collude to get the best possible outcome at the expense of the consumer. The study was about the effects of Reinforcement Learning Agents learning to agree to a certain price for a product hinting on collusion between the two. The problem that this highlights is that the AI is now able to learn how to fix prices so that they can make the most sales which hurts the consumers. The idea here is that the agents in the controlled environment showed signs that they were able to learn the basics of capitalism and learned how to maximize their profits. This is one way to look at it. The other thing that I can see here is that the agents can actually learn to interact with each other without any interface between them and just observing the environment. That would be great if we can transfer that to self-driving cars and later to other applications as well. Imagine when your car learns how to actually maximize the road through sharing? That would be great right?

Here is another article about [predictive maintenance using Big Data](https://www.popularmechanics.com/military/aviation/a25995189/big-data-saves-warplanes/). What the article writes was that the US Air Force had mined the data from their older planes and from this data created machine learning applications on preventive maintenance. They were actually getting the data from various critical value components like the engines, the landing gear and input it to their machine learning model. The model would then flag components that might be due to fail or are no longer running at an acceptable level and alert the maintenance crew to check them. This initiative comes from the successive failures in components that plagued the aircraft some of which has led to loss of life. The source from the air force said that they intend to develop this initiative and proceed to applying it to their fighter fleet which is huge and very active.

## Day 34 March 5,2019

OBJECTIVES FOR THIS WEEK (33-40):

* [ ] :gem: Update experiences and projects in LinkedIn Pages.
* [ ] :gem: Finish *Introduction to Deployment*
* [ ] :gem: Finish *Building a Model with Sagemaker*
* [ ] :gem: Proceed to *Deploying and Building a Model*
* [ ] :gem: Follow that with *Hyperparameter Tuning*
* [ ] :gem: Lastly *Updating a model*
* [ ] :bomb: All leading up to **Deploying a Sentiment Analysis Model**

Now proceeding with Introduction to Deployment topic. First up we are given the problem that can be solved by Machine Learning, Boston housing prices. The idea is that we have to develop a model and deploy it to a cellphone app. The first discussion then transitions to the Machine Learning workflow. I am very excited to finish this topic. Previously, all I have done was simply train the model and test it. Now I can actually apply the models to real world applications and that for me is more important. So on to the Machine Learning Workflow.

### Machine Learning Workflow

The machine learning workflow is cyclical and therefore iterative as well. It starts from the gathering of data and goes through the modeling of a solution and finally proceeds to the deployment in the system. Generally, there are three distinct steps in the workflow: **Explore & Process Data**, **Modeling** and **Deployment**. In this example we are asked to work on a Kaggle Dataset. So we go over the workflow in details below.

#### Explore & Process the data

First thing to do is to Explore and Process the data for the given problem. This section would include **Retrieval of Data**, **Clean and Explore** and **Prepare and Transform**. For the example of a Kaggle Dataset, data retrieval is simple, we have to go to Kaggle and download the dataset directly. There are other ways to do this like web scraping and using other data sources/repository. Once the data is retrieved, we can proceed with cleaning the data and data exploration. In this step we would be able to plot the data and see if there would be missing values in the data set and if there are outliers that need to be handled. It would also help if we can explore the data so that we can get an idea of what model might work best. Last step for this section is to prepare and transform data. This would include normalizing the data set values and splitting the data set into the training, validation and testing subsets.

#### Modeling

Once we have prepared our data we can then proceed with creating out model to train. We can have multiple models which we can train. The idea should be that we will choose the model that provides the best result in our given problem. In the modeling section we are going to run training on the model and we also get to validate the model for losses and accuracy. Once we have trained our model we can then perform the testing using the test data set to get the performance metrics of our model.

#### Deployment

The focus of this lesson and the rest of the nanodegree would be in this section, Deployment. In this example we are going to use our model in a mobile app. This section would include how we are going to use the trained model into an application and also on how to account for changes in the data or on updating the model with the new data.

There are different samples of workflow available from the different vendors of ML solutions, we have one from AWS, another from Azure and also one from GCP. The links to their Workflow are below:

https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-mlconcepts.html
https://docs.microsoft.com/en-us/azure/machine-learning/service/overview-what-is-azure-ml
https://cloud.google.com/ml-engine/docs/tensorflow/ml-solutions-overview

We are now done with the Workflow of a machine learning project, we then move to Cloud computing. First up would be to define Cloud computing. Simply put, its transforming the IT product into a service. Think of file storage and portability. The old product based solution would be to use a flash drive to store the files. With cloud computing we transfer that physical product into a managed service where in we simply rent a space in the provider's infrastructure and we upload our files. This is a simple example of Cloud computing. There are many more possible types of Cloud computing like Infrastructure-as-a-service, Software-as-a-service and Platform-as-a-service all catering to the wide array of traditional services used to be handled individually by companies.

There are many benefits of using Cloud computing. One good example is in capacity management. The graph below shows the differences between the traditional infrastructure capacity and the cloud based capacity. If you have been in operations for a while, you would notice that most of the demand for a service is usually not uniform. There are days wherein many users would simultaneously login to the system. Traditionally this has been solved by adding new infrastructure and then setting it up to split the load. This can lead to times where we have infrastructure that is underutilized due to the seasonal demand for our service. This would obviously lead to cost. Cloud computing providers have a solution for this where they would automatically adjust the capacity of the service based on demand. It can scale up or down depending on the traffic to our system. This solution makes it easier to meet demands during peak hours without needing to invest in infrastructure that would be idle on off-peak hours.

![Udacity-Cloud Computing Capacity over time curve](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bea396e_capacityutilizationcurve3/capacityutilizationcurve3.png)

Obviously there are many benefits that can be gained from using Cloud Computing services aside from Costs. But there are also risks when using cloud services. This is usually not highlighted by providers but some examples would be security, oversight and governance, portability(migration).

In terms of machine learning context, cloud computing is usually used for deployment of the model. But that does not mean that Cloud computing is not used on other parts of the machine learning workflow. For example, the training portion could benefit from additional computing power of cloud based computers. This is especially true for larger and complex datasets. As such, there are also services being sold by the cloud providers to cater to the need for virtual machines and instances.

Machine learning applications would represent the deployment of the model. So far, we have only made models and trained them in this course. We were more focused on the theory behind the models. In the succeeding lessons we would be diving into how we can make services out of these models. First up is we define what we mean by deployment, or deployment to production. Simply put, deployment to production means that our machine learning model is integrated into an existing production environment so that the model can be used to make decisions or predictions based upon data input to the model. In the next discussions we are going to assume that the machine learning model was developed using Python.

Now that we have a working model we want to deploy the next question would be in what way can we deploy these model, from a modelling component to the deployment component. In Udacity's notes there are *three* primary methods to transfer the model.

1. Python model is *recoded* into the programming language of the production environment.
2. Model is coded in *Predictive Model Markup Language (PMML)* or *Portable Format Analytics(PFA)*.
3. Python model is *converted* into a format that can be used in the production environment.

The methods above are also arranged by how common they are in usage. Method 3 is similar to the way a model is deployed for Amazon SageMaker. We now go slightly deeper into the methods.

**Recoding Model into Programming language of Production environment**

This methods involves recoding the trained model in Python into the language used in Prod. This could often be in Java or C++. This method is now rarely used since it takes time to recode a model, train and test and achieve the same prediction as the original Python model had.

**Model coded in PMML or PFA**

In this method the model is coded in Predictive Model Markup Language (PMML) or Portable Format for Analytics (PFA), The two are complementary standards and aim to simplify moving predictive models into the production environment. Both language are developed by the Data Mining Group to provide vendor-neutral executable model specifications for certain predictive models used in data mining and machine learning.

**Model is converted into Format that is used in the Production Environment**

This is the most common method used in deployment of a model into production. The Python model that has been created is then converted into code that can be used in the production environment usually through the use of libraries and methods. Most popular machine learning frameworks like PyTorch and TensorFlow have methods that will convert the Python models into intermediate standard format like ONNX (Open Neural Network Exchange format). This intermediate standard format can then be converted into the software native to the production environment. There are also technologies like containers, endpoints and APIs (Application Programming Interfaces) that can help ease the work required for deploying a model into the production environment.

### Machine Learning Workflow and DevOps

![Udacity - Machine learning workflow and DevOps](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bea5c84_mlworkflow-devops-1/mlworkflow-devops-1.png)

As we can see in the workflow much of Explore and Process Data section as well as Modeling are closely related to each other. The deployment section is also seen as a standalone section that is distinct in objective from the other section. As distinctly identified in the workflow DevOps model above, we can clearly see where development in the Machine Learning workflow happens and where the Operations portion of the Machine Learning workflow happens. In this context, those that are involved in Development are commonly referred to as Analysts while those that handle operations are usually software developers. This line separating the duties of Development and Operations are now more blurred and there can be instances where the analyst can transition to a software developer and vice versa. This is aided by the use containers, endpoints and APIs. With the softening of the division between operations and development allows analysts to handle certain aspects of deployment and enable faster updates to models.

### Endpoints & REST APIs

The **Endpoint** is the interface to the model. This interface will facilitate the communication between the model and the application. It will take in the input data sent by the user through the application. It also serves as the receiver for the prediction result of the model that would be sent back to the user after getting the data. Below is a sample breakdown of code that shows the interaction between the endpoint, the application and the model.

![Udacity - Endpoint Breakdown](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bea640c_endpointprogram-2/endpointprogram-2.png)

In the example above, the endpoint is line `predictions = ml_model(input_user_data)`. In this case the the input interface for the endpoint is via the `input_user_data` argument. The model in this case is the `ml_model` function call. The output for the model, which is also part of the endpoint, is then going to be stored in `predictions`. In this case, the entire Python script is the application.

Communication between the application, by extension the user, and the machine learning model will therefore be under the control of the endpoint since it acts as the interface between the two components. In this case the endpoint is considered as an Application Programming Interface (API). Note that APIs are not just about machine learning models, they can also used to process data, or do another task. The idea is that the API will take in the data, will do the operation inside that API and output the result of action back to the application. An easy way to think of an **API** is a set of rules that enable programs, the application and the model in this case, to communicate with each other. In this context, our API is using the Representational State Transfer, **REST*, architecture. REST provides a framework for the set of rules and constraints that must be adhered to for communication between the two programs. This REST API is one that uses HTTP Requests and responses to enable communication between the application and the model through the endpoint.

The HTTP request sent from the application to the model (note that only the application can use the request) is composed of four parts:

* Endpoint - which is in the form of a URL, Uniform Resource Locator, or the web address.
* HTTP Method - there are various HTTP methods that can be used for example *get*,*post*, *put*, *delete*. In the case of deployment our application will use the *POST* method only.
* HTTP Headers - this contains additional information regarding the message, like the data format expected in the message. These information are placed in the headers to be passed to the receiving program.
* Message (Data or Body) - This is the final part of the entire message. For deployment purposes, this will contain the user's data which would be used as input for the model.

The HTTP response sent from the model to the application (only the model will respond) is composed of three parts:

* HTTP Status Code - this is the status of the last message received from the application. If the message was successfully sent and the data was received the status code would start with a *2*, like 200.
* HTTP Headers - again, this contains the additional data regarding the message in the response. This would therefore include the format of the data that is sent back to the receiver.
* Message - this is going to contain the data that was returned from the model. So in our case this will be the predictions result.

The prediction from the user's input data is then presented back to the user via the application. The endpoint is the interface that enables the communication between the application and the model using a REST API. In the future we will learn to use RESTful API and we will come to realize that the application's responsibility would be the following:

* Format the user's data in a way that can be placed in the HTTP request message and be usable on the model's end.
* Translate the HTTP response message of the prediction in a way that is readable/understandable for the end user.

## Day 35 March 6,2019

OBJECTIVES FOR THIS WEEK (33-40):

* [ ] :gem: Update experiences and projects in LinkedIn Pages.
* [ ] :gem: Finish *Introduction to Deployment*
* [ ] :gem: Finish *Building a Model with Sagemaker*
* [ ] :gem: Proceed to *Deploying and Building a Model*
* [ ] :gem: Follow that with *Hyperparameter Tuning*
* [ ] :gem: Lastly *Updating a model*
* [ ] :bomb: All leading up to **Deploying a Sentiment Analysis Model**

Now moving to **Containers**. We have previously discussed that the deployment usually has an *application* where the users can go to and a *model* that does takes in data from the users and outputs predictions based on the input. Managing the communication between the model and application is the *endpoint*. Both the model and application are programs and therefore require a computing environment setup for them to operate on. *Containers* is one way to create this environment and maintain it. Both model and application can be each run on a container computing environment. Containers are created via scripts that contains the setup of the environment which typically includes the software packages, libraries and other computing attributes that would be required to run the software application which in this case would be the model and application. *A container is a standardized collection of software that is used to run an application*. Think of it as a virtual environment we have in Anaconda where we only add dependencies based on the function of the application we have to write. This is similar in concept to the container. One common container software is Docker, and due to its commonality it is used interchangeably sometimes to refer to a container. **To be clear, containers are not full virtual machines.** They do not require the OS to be included only the software which means there is lower overhead to the computing resources. One good example of containers would be the workspaces in Udacity. The containers allow the developers to ensure that only relevant dependencies for the experiment are used. This means that they can update and fix issues on specific containers without it affecting possibly other workspaces. For example, if there is a bug in the workspace for auto-encoders then only the auto-encoders container needs to be edited.

### Software container and Shipping container

One way to think of the container is in the similarly named containers in shipping. The containers in shipping can carry different kinds of goods, bulk goods, dry goods, frozen food, live animals, cars etc. Even though the contents of the containers are not the same, they are standardized that they can be handled universally. There is a standard size for the containers and the tools required to operate and use them are also similar worldwide. The containers can be stacked since they have fixed sizes, they can be grabbed by cranes, the truck carrying doesn't change anywhere in the world because the sizes are standard. This is the same for the concept of containers/docker. The software they run inside them can do different things but one thing they have is that the tools used to operate them are the same and standardized. This makes it easier to transport and deploy the containers of software and interchange them between development and deployment.

### Container structure

The composition of a container is outlined with the image below. The bottom layer would be the computational infrastructure, which could be a cloud based sever, a local data center or a personal computer. The next layer would then be the operating system which is what is used to run the infrastructure. Above that would be the container engine, this would be Docker or Kubernetes etc. Basically they are the software installed on the Operating System that allows the management of the containers. On top of the container engine would then be the actual container where our application would reside. This would be made up of two parts. First would be the Bins/Libraries which are the softwares or dependencies that are required to run and manage the application we have. Finally we have the actual application itself which could be a web service or an app.

![Udacity - Container Structure](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bea67ba_container-1/container-1.png)

### Advantages of Containers

* 1. Increased security due to the isolation of individual applications
* 2. Tailored dependencies/libraries ensure that only relevant software needed to run the application is running per container. This leads to efficient usage of computational resources.
* 3. Ease of management of application lifecycle. Due to the modular approach of the containers it is easier to manage them through out their lifecycle. Containers allow easier creation, replication, deletion and maintenance.
* 4. Containers make it simpler to replicate, save and share containers. Since containers can be created by following a script, this makes it easier to share the setup to others.

Just to build on the fourth advantage, this is similar to the yaml file in anaconda. The file stores the current libraries installed in the environment which can then be used by others with anaconda to create the same environment on their end. The image below shows what information would be contained in the Container script. This could include the configuration of the software in the container as well as the versioning to ensure that the application can be run and allow others to receive the same environment and setup as the original environment.

![Udactiy - Containers script file](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5bea67c6_container-2/container-2.png)

### Characteristics of Modeling and Deployment

For modeling, the characteristics are simply the hyperparameters of our model. This would need to be provided in order for the model to train. There are tools like scikit-learn or open.ai that allows us to get the appropriate hyperparameter for the model we are training. For deployment there are more characteristics, namely *versioning*, *monitoring*, *updating and routing*, and *predictions*. These are set to aid in managing the deployed model. First would be versioning which is simply the model version identifier that would be part of the metadata. The deployment platform should allow the owners to set a version number for the model deployed. The second characteristic would be monitoring, which basically means that the cloud platform would send back metrics on the performance of our model. The metrics could include the accuracy of the model, the traffic to the model, the latency. These metrics allow the owner to monitor the performance of the model and plan out improvements in the model or in the service plan to meet target expectations. Model updating and routing is the third characteristic of the deployment. The ability to update the model should be available for the owner. To aid in the development of an update, there should also be an option to route a portion of a request to a new version(test) and the current version to allow for evaluation of results. Finally there is the characteristic of predictions which can be split into two: On-demand predictions and Batch predictions. For On-demand predictions, we expect our model to respond to requests real-time. Expectations for On-demand predictions would be low latency between the request-response and it should allow high variability in request volume (seasonality). The predictions are returned via the HTTP response and would be in the form of JSON or XML for use of the application. *On-demand predictions* are often used to provide users with real-time online responses. The second type of prediction would be Batch predictions. *Batch predictions* are normally used to provide insights and help make business decisions since it covers a wider range of date. Examples of this would be weekly and monthly forecasts based on the previous month's usage, for this example the model would more likely output a result file instead of an XML or JSON type. For batch predictions/asynchronous the expectation is that there would be a high volume of request but with a more periodic submission (scheduled) therfore latency would not be an issue.