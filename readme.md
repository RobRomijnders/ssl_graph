# Semi supervised learning for classification problems
This repo aims to do semi supervised learning (SSL) for classification problems. We view the data as nodes on a graph. The repo implements two methods to learn a classifier on this graph of both labeled and unlabeled data. The first approach uses a [random walk](https://en.wikipedia.org/wiki/Random_walk) between the unlabeled and labeled data. The second approach uses the [label propagation algorithm](https://en.wikipedia.org/wiki/Label_Propagation_Algorithm).

# What is semi supervised learning
In SSL we have both labeled and unlabeled data. Our intuition follows that the unlabeled data can somehow be useful to improve a classifier. Think of this as follows: let's say we work in a production line making parts for planes. We are interested to classify a part as being OK or if it needs additional care. We measure some properties from this part and use those as features for a classifier. Unfortunately, it's expensive to hire an expert mechanic to judge the quality of the parts, so we have only 20 labeled samples. But from the production line, we have measured 100.000 parts. Our objective is to make the best classifier using the 100.020 data samples and the 20 labels.

I could also recommend the pictures in [rinuboney's blog](http://rinuboney.github.io/2016/01/19/ladder-network.html) where he explains SSL in feature space using some nice diagrams.

# Assumptions

To learn also from our unlabeled data, we need to make assumptions.

  * __The unlabeled data carry information that is useful for classification__ This assumption states that some information from learning <img alt="$p(x)$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/c9ea84eb1460d2895e0cf5125bd7f7b5.svg?invert_in_darkmode" align=middle width="30.45108pt" height="24.6576pt"/> will tell us something about the <img alt="$p(y|x)$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/fc76db86ea6c427fdd05067ba4835daa.svg?invert_in_darkmode" align=middle width="43.666425pt" height="24.6576pt"/>. This assumption underlies all SSL models. I want to emphasize that the assumption might not always be the case. Chapter 4 of [this](https://mitpress.mit.edu/books/semi-supervised-learning) book shows suprisingly simple problems where unlabeled data actually hurts the classifier.
  * __The data lies near on a low-dimensional manifold__ This assumption states that data crowds together near a low dimensional manifold. This helps SLL, because unlabeled data can inform us about this manifold, regardless of the labels.
  * __The different classes lie near sub-manifolds__ This assumption states that the data of the different classes crowd together near sub manifolds. This will help SSL, because we can learn about the various sub manifolds from our data points, regardless of the labels.
  * __The decision boundary lies in low density regions in the input space.__ This assumption follows from the second and third assumption. If we assume that all data lie near a manifold and the classes lie near sub manifolds, then it follows that the regions between these sub manifolds have low density. This helps SSL, because we can infer these low density regions from the unlabeled data.

Roughly, I like to summarize these assumptions into a smoothness property. From these assumptions, it follows that any model for this data has a smoothness property. Nearby points must have nearby labels. The notion of _near_ is relative to the manifold. The manifold might be curved in some directions, but on any small region, we may interpret it as an Euclidean space. (I like the intro of [this](https://en.wikipedia.org/wiki/Manifold) wikipeddia article)
 

# Transductive or inductive learning
Label propagation and the random walk algorithm are examples of transductive learning. On the opposite, we have inductive learning algorithms like random forests, SVM's and neural nets. So what is the difference between transductive and inductive learning?

An __inductive__ learning algorithm aims to learn a function, <img alt="$f: \mathcal{X} \rightarrow \mathcal{Y}$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/c8f94eeeec742f7fa1d6d8172c70e47a.svg?invert_in_darkmode" align=middle width="75.556965pt" height="22.83138pt"/>. Predictions will be made by evaluating <img alt="$f(x_i)$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/92a2f148344252edc0cc4112ae131688.svg?invert_in_darkmode" align=middle width="37.470675pt" height="24.6576pt"/> for all <img alt="$x_i$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/9fc20fb1d3825674c6a279cb0d5ca636.svg?invert_in_darkmode" align=middle width="14.045955pt" height="14.15535pt"/> in the test set. In our familiar python code, this looks like
```python
model.fit(X_train, y_train)
y_pred_test = model.evaluate(X_test)
```

An __transductive__ learning algorithm aims to learn a function, <img alt="$f: \mathcal{X}_{train} \times \mathcal{Y}_{train} \times \mathcal{X}_{test}  \rightarrow \mathcal{Y}_{pred}$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/a744f2014cf1dd99818285d77a521803.svg?invert_in_darkmode" align=middle width="248.506005pt" height="22.83138pt"/>. Predictions now always follow from this function. In Python code, we have
```python
y_pred_test = model.predict(X_train, y_train, X_test)
```

Both the random walk algorithm and label propagation are transductive learning algorithms. Their interface looks like the template we just discussed
```python
y_pred = label_propagation(y_labeled, X_labeled, X_unlabeled, X_train, gaussian_kernel(std), mu)
y_pred = random_walk(y_labeled, X_labeled, X_unlabeled, X_test, gaussian_kernel(std))
```

# How to convert a dataset to a graph?
A graph consist of nodes and edges. An example of a graph looks like this:
![Example of a graph](https://github.com/RobRomijnders/ssl_graph/blob/master/im/graph_example.png?raw=true)

We go from data set to graph in three steps

  * Our samples will be the nodes in this graph. Each data sample will be represented by one node in our graph. 
  * We connect two nodes with an edge if the sample in within the k nearest neighbors of the corresponding data sample. 
  * We put weight on this edge according to a kernel function. A kernel function takes two data samples and outputs a measure of similarity. In our implementation, we use a Gaussian kernel. (not to be confused with a Gaussian distribution)

Now we have a graph, we look back at our hypotheses we made for semi supervised learning. It stated that nearby points will have similar labels. In terms of our graph, this translates to the following: *if an edge contains a high weight, its two nodes will have similar labels.* This insight underpins both algorithms to come. 

## Kernels
Kernels are central in our conversion from data samples to the graph. Therefore, I'd like to spend a few words on it. A kernel measures the similarity between two objects. These objects might be vectors, but could also be objects that we don't immediately imagine as vectors. For example, people have designed kernels for proteins, texts and even graphs themselves. In general, a kernel maps any two objects to a positive number, so it implements <img alt="$f: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}^+$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/c2ea507b252afab7be5aa945f051ac82.svg?invert_in_darkmode" align=middle width="119.40621pt" height="26.17758pt"/>.

# The algorithms
## Label propagation
Label propagation literally implements the above statement. If an edge contains high weight, its two nodes will have similar labels. So for any labeled node, we can *propagate* its label to neighboring unlabeled nodes according to the weight. We repeat this step for many times. Then eventually, the labels on the unlabeled nodes will reach an equilibruim. That will be our prediction for these nodes. For more details, you might like to watch [this](https://www.youtube.com/watch?v=HZQOvm0fkLA) lecture.

## Random walk
The random walk algorithm is a little bit more involved. It starts from taking random walk on our graph as we defined it above. Below, we'll follow a random walker that walks on our graph

  1. Let's say I jump on an unlabeled node, i, in our graph. 
  2. Then I randomly *walk* to neighboring nodes. My choice of edges will be determined by their weight, so I am more likely to take edges with higher weight.
  3. When I have jumped to a labeled node, I will stay there and stop my walk
  4. Now the question follows: for each of the labeled nodes, what is my probability of ending at that particular node

This random walker will give rise to a probability distribution over all the labeled nodes. Let's imagine that as a vector, <img alt="$p_i \in \mathbb{R}^{N_{labeled}}$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/025951fd6d5bcb498fe3956197e34c95.svg?invert_in_darkmode" align=middle width="92.794185pt" height="27.65697pt"/>. Then <img alt="$p_{ij}$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/200cdf959030dfee4638a263f63db3ae.svg?invert_in_darkmode" align=middle width="19.025985pt" height="14.15535pt"/> indicates the probability that we start at unlabeled node, i, and end at the labeled node, j.

Repeating these steps for each unlabeled node and stacking those vectors in a matrix results in a matrix, <img alt="$P \in \mathbb{R}^{N_{unlabeled} \times N_{labeled} }$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/cb3c4afebb4ad7461522519d69a88833.svg?invert_in_darkmode" align=middle width="164.265255pt" height="27.65697pt"/>. In discrete mathematics, people will refer to this matrix at the infinity matrix. It got this name, because it follows from allowing the random walker an infinite amount of time to reach the labeled nodes.

Now how does this help us with semi supervised learning? To make a prediction on any of the unlabeled nodes, we can use the infinity matrix as a linear smoother. So our prediction is <img alt="$y_{pred} = P \ y_{labeled}$" src="https://rawgit.com/RobRomijnders/ssl_graph/master/svgs/17bb8f95d7db30df4b82835e7de29780.svg?invert_in_darkmode" align=middle width="124.164315pt" height="22.46574pt"/>

## Graph based learning
Let's grab back at our hypothese and see how these algorithms used them

  * __The unlabeled data carry information that is useful for classification__ In both algoritms, we used unlabeled data to benefit the predictions. In label propagation, unlabeled data might act as a highway to propagate labels. In the random walk, the random walker can walk via unlabeled nodes to reach labeled nodes.
  * __The data lies near on a low-dimensional manifold__ Both algorithm relied on kernels. Kernels only depend on local similarity. Therefore, it can model the, possibly curved, manifold that the data crowds on.
  * __The different classes lie near sub-manifolds__ In both algorithm, the choices are made based on the weights on the edges. Labels are more likely propagated over edges with high weight. The random walker will more probably walk over edges with high weight. For any sub manifold, this ensures that the labels are more likely to be propagated over the sub manifold
  * __The decision boundary lies in low density regions in the input space.__ As the label propagation and random walker make their decisions based on the weight, labels are less likely to cross low density regions where the weights will be low.

# Results
To visualize the inner workings of the algorithms, we work on 2D data. The code contains function `generate_data()` and `generate_data2()`. They result in the following diagrams:
![first datagen](https://github.com/RobRomijnders/ssl_graph/blob/master/im/data2.png?raw=true)

![second datagen](https://github.com/RobRomijnders/ssl_graph/blob/master/im/data.png?raw=true)
The problem consist of classifying the two classes.

The diagram below displays the results as more unlabeled data is used. We also plot the mean absolute error to see how confident the algorithm is in a prediction.

![results](https://github.com/RobRomijnders/ssl_graph/blob/master/im/figure_2.png?raw=true)

# Discussion
For the random walk, we see that the performance improves as we supply more unlabeled data. However, for the label propagation, we observe a constant performance. So somehow, I think the hypotheses don'd work for label propagation. I am curious to learn where the reasoning breaks. Please reach out if you have any clues.



# Further reading

Some interesting sources for further reading are

  * [Semi supervised learning, book, by Chapelle, Scholkopf and Zien](https://mitpress.mit.edu/books/semi-supervised-learning)
  * [Graph based semi-supervised learning, a lecture by Zoubin Ghahramani](https://www.youtube.com/watch?v=HZQOvm0fkLA)

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com