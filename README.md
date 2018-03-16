# Reuters_Classification
Implementation of a Simple Neural Network Model for Document Classification
Dataset Used: Reuters Neswire Classification Dataset
Keras library provides a set of Data sets to play around. Reuters Neswire Classification Dataset is one among those.
Dataset of 11,228 newswires from Reuters, labeled over 46 topics
In this dataset, Documents have been preprocessed, and each review is encoded as a sequence of word indexes (integers). Words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".
