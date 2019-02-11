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