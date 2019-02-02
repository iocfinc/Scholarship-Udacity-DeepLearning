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