---
title: "NLPCA"
date: 2026-02-27
layout: post
tags: [tech, thoughts]
---
# Nonlinear Principal Component Analysis Using Auto-associative Neural Networks
## By Mark A. Kramer


Hello and welcome to my understanding of this very scary sounding paper.
If this is your first time reading a research paper (it was mine too), and you feel like “it kind of makes sense… but I still have so many questions,” this blog is for you. If it feels overwhelming, I relate. The goal here is not to rewrite the paper, but to walk through the key ideas you must understand to feel comfortable with it.

Let’s start by breaking down the title itself.

What are neural networks?
Neural networks refer to a computational network inspired by the brain. Here the nodes – neurons – process and transmit “signals”. Like any network, their purpose is to handle processing information so the successor can do whatever is necessary. How they do this is a subject for another blog but for now let’s know the basic workings. They do so by a function called activation. What actually unfolds is that an input is taken, its weighted sum is computed and then a bias is added. Then the activation function is applied to it to calculate the desired output (by usage of sigmoid or tanh functions). This should give you a vague idea, which is enough for now.
So, why are we doing this? Simple - we want machines to imitate our brains – to learn relationships without having to explicitly write rules and conditions for it to make sense of data. Neural networks are powerful function approximators — they can model very complicated mappings between inputs and outputs. Here, though, we don’t want predictions, we want to be able to reproduce our inputs.
This is where auto associative neural networks come into factor.
They are a special type of neural network that reproduces its own input. This is required when we want to understand and interpret data, not just use it for computation.
How we achieve this is by the use of a hidden layer called the bottleneck layer where data is compressed to find our true parameter. This is what helps us to find the hidden structure. 

Now, before we dive into the paper, there are 3 things we need to understand. They are: feature space, feature variables and factors.
Feature variables are the measured quantities. They are the coordinates of feature space. Essentially the data points that are plotted. If we measure skulls, our variables would be skull height, width, depth and eye socket width. Each measurement here represents 1 dimension.
Feature space is the mathematical space defined by these parameters. It represents the coordinates of the (data) measurements taken. In this paper, there are 2 feature spaces at play. One is the original measurement space and the other is the discovered reduced space that we are trying to find.
Factors are the hidden causes that generate variation in the data.  Here, the true underlying factors might be: overall skull size and shape proportion.  Instead of 6 independent measurements, there may only be 2 true parameters.
PCA and NLPCA attempt to uncover those hidden factors.

PCA: Principal Component Analysis
What does it mean?  It is used for linear mapping (higher) of multidimensional data into lower dimensions with  minimal loss of information.  All the values are represented in a matrix of (n x m) dimensions, n being the number of observations and m being the number of variables.
Now, with the help of 2 other matrices : T called scores matrix and P called the loading matrix, we are calculating the output which is essentially the input. Here another matrix, called the residual, is being computed for an error approximation/rectification and we arrive at an equation.
This:  
Y = TP<sup>T</sup> + E
Before this equation is explained, a core topic must be explored.
What are eigenvectors? 
For this, we need to understand what a covariance matrix is. Now as we can see from the name they are a type of matrix which tells us how 2 things will vary.   What do I mean by that? To put it simply, it is used to describe the relationship between two variables. If two variables tend to increase together, they have high positive covariance. If one goes up while the other goes down, it is called negative covariance. So the matrix captures all the pairwise relationships between all variables at once. Hence, it is a matrix that explains how multiple variables vary together. It shows the variance for each pair of features in a database. Here it is used for finding directions of maximum variations. 
With this in mind, what are eigenvectors? 
For any matrix, an eigenvector is a special direction such that when the matrix acts on it, the vector only stretches or shrinks i.e., it doesn't rotate. The amount of stretching is the eigenvalue.
In a covariance matrix, it tells the direction of maximum variance. How is that helpful? Well, the direction of maximum variance often corresponds to the dominant underlying structure. Isn’t that exciting? Just by knowing where there is maximum variance you can try to figure what are the factors driving a 100 dimension data point! Of course, it is not necessarily the true physical parameter so a distribution must be referred to. However, reducing 100 correlated measurements to 20 or even fewer meaningful dimensions is still a powerful simplification.

Now that you understand this, we go back to the equation.
Here, 
Y is the original dataset or what has been measured. 
In my example, it would contain all the raw skull measurements we collected.
P are the coefficients of transformation, put in a matrix of (m x f) dimensions. They are the eigenvectors that define new axes of the reduced feature space. They answer "what combination of original variables makes up each new axis?"
Each column of P represents a new axis. Here, it could be 2 factors overall skull size and shape variation. Each entry in P would tell how much each original measurement in Y contributes to them.
T represents the scores matrix of n x f dimensions. It is the coordinates of the hidden variables in the low-dimension matrix or in simpler words, the discovered internal parameters. Now, each skull is measured by 2 factors: skull size and skull shape. So T would give each skull’s position along these 2 factors. 
E is the error or residual matrix, accounting for the noise.Some of the dimensions would still not be accurate, so E would capture the variation in skull measurements that are not explained by the 2 hidden parameters.
Now that definitions are out of the way, let’s see what exactly each term is doing.   P<sup>T</sup> is for converting the low-dimensional representation back to measurement space.   TP<sup>T</sup> rebuilds the original data using the discovered hidden parameters.

So, PCA is essentially a technique for finding the most important directions in your data. In the skull example, a whole lot of measurements - height, width, eye socket dimensions and so on — are all crammed into one big matrix. Now here's the thing, not all of these measurements are independent of each other. A wider skull probably also means a taller skull. Larger eye sockets probably come with a larger overall skull. They move together. PCA looks at all of this and asks - what is the single direction in this data that explains the most variation across all your skull samples? That direction is called the first principal component. Then it finds a second direction, completely independent of the first, that explains the next most variation. Then a third, and so on. The exciting part is that these directions often end up corresponding to something physically meaningful, maybe the first component turns out to basically be overall skull size, and the second turns out to be skull shape, even though it was never explicitly measured. By keeping only the first few of these components instead of all the original measurements, the same data has now been represented in far fewer dimensions while still preserving most of the meaningful structure. That is the whole point - less numbers, same story.  The mathematical machinery behind this, the covariance matrix and its eigenvectors, described earlier, is just a rigorous way of finding those directions of maximum spread. However, the fundamental limitation is that every component must be a straight-line direction in PCA. PCA can find a diagonal axis through a data cluster, but if the data lies along a curve (like a circle or a spiral) no straight line can capture it well. This is precisely where NLPCA steps in, replacing straight-line axes with curved ones learned by the neural network.

Taking this forward, the next section is for explaining the encoding-decoding part with where the non-linear part comes in.  Nonlinear PCA replaces the linear projection Y = TPT with a nonlinear function Y≈ g(f(Y)), where both f and g are learned by a neural network.
G is the Mapping function where every input is connected to a mapping node and every mapping node is connected to each bottleneck node. 
One mapping node does this:  Input —> weighted sum —> sigmoid —> produce single number between 0 and 1
Why is the sigmoid function being applied here? Simple, because sigmoid function is bending the linear relation captured by the weighted sum such that it correctly captures curved relations. 
How?
Well, it basically squashes any real number output in the range of 0 to 1. It is biologically inspired where it mimics how a neuron transmits and thus introduces nonlinearity so the network can approximate nonlinear manifolds.   Side note: Manifolds are the region where the actual data lies in. Essentially where a complex higher dimension structure is reduced or represented by a smaller dimension projection.   { Weak signal : don’t do anything i.e., small input - o/p near 0  Strong signal: immediate reaction i.e., big input - o/p near 1  Intermediate signal: partial reaction i.e., med i/p - smooth transition}  So the weights decide which inputs matter and how much and the sigmoid then warps the result nonlinearly before passing it to the next layer.

Going back, H represents the demapping function where it takes the compressed bottleneck representation and attempts to reconstruct the original data. In other words, the network first compresses the skull measurements into a few hidden factors using the mapping function G, and then expands those factors back into full measurements using H.
You must be wondering why there is a need for all of this hassle when a simple neural network with a single hidden layer is often sufficient for prediction tasks. It is precisely because one hidden layer can only map, it cannot compress and then reconstruct. Since G and H are two separate nonlinear operations, each needs its own independent hidden layer. That gives us three hidden layers in total: one for encoding, one bottleneck, one for decoding. Now, once we have established this architecture in place, we need to figure out how the learning is happening. How do the weights figure out what to do? That is where backpropagation comes in.

Backpropagation is an algorithm through which neural networks are trained by updating weights to reduce prediction error where it calculates how much each weight contributed to the error and adjusts accordingly.
The entire process can be explained by the following flowchart.
Forward pass: feed Y in, get Y' out
                      ↓
Compute error: E = Σ(Y - Y')²
   	          ↓
Backward pass: trace how much each weight
contributed to the error
        		↓
Adjust weights slightly to reduce error
         		↓
Repeat thousands of times
        		↓
Eventually weights stabilize at values
that minimizes E.

Through repeated backward and forward passes, the network is able to arrive at a point where the bottleneck layer has the best compact composition for reconstruction but how is it actually going to go through the factors? What if we want it to analyse more than 1 factor? What will it do then? It can’t randomly pick and choose.
This leads us to two important variants of the method - sequential NLPCA and hierarchical  NLPCA.
In S-NLPCA, all factors are extracted at once. The bottleneck layer is given as many nodes as the total number of factors desired — say 2 nodes for 2 factors — and the entire network is trained in one go to find both factors simultaneously. This is straightforward but carries a risk: during early training, multiple bottleneck nodes can drift toward representing the same dominant factor, because the dominant variance direction produces the largest gradients during training. The network, therefore, wastes the bottleneck layer’s meaning by having 2 nodes describing the same thing, thereby being unable to find a second independent factor.
H-NLPCA addresses this by extracting factors one at a time in a hierarchy. A first network with a single bottleneck node is trained to find the primary factor which explains the most variance. Once trained, the residual is computed: the difference between the original data and what the first network reconstructed. This residual is then fed as input to a second network, also with one bottleneck node, which finds the primary factor of what remains. The process repeats for as many factors as needed. As each network has only one bottleneck node, it can’t have duplicate factors. Each network is forced to find something new since the most important structure was already removed in the previous step. The trade-off is that errors can accumulate across stages, and hence rescaling the residual between steps becomes important since each successive residual gets progressively smaller in magnitude.

In conclusion, what this paper ultimately demonstrates is that the structure hidden inside complex, high-dimensional data need not be linear to be discoverable. PCA, for all its mathematical elegance, is constrained to finding straight-line relationships, a limitation that undermines its usefulness in the real world, where things are often non-linear. Kramer's contribution was recognizing that an autoassociative neural network, trained simply to reproduce its own input through a narrow bottleneck, will naturally discover whatever compact representation best describes the data. In the batch reactor example, a network that was never told initial temperature or impurity concentration existed managed to reconstruct a map of the process that preserved exactly those two variables as its axes. 
It found the physics without knowing any physics, purely by minimising the reconstruction error. 
What makes this paper particularly significant in retrospect is that it arrived at the architecture now called an autoencoder, which is one of the foundational building blocks of what we know as modern deep learning years before these ideas were made mainstream by the branches dedicated to them.