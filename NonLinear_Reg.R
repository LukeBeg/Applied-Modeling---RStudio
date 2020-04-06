'Code and some explanations taken from "Applied Predictive Modelling" book by Kuhn & Johnson. I tried to reproduce and explain the code on my understanding.'

# Generally speaking non-linearity can be factored in by adding (manually) additional model terms like squared, cube terms....
# this way of operating works well if the structure of the linearity is known. 

# In case the non-linearity is not known then one must use other way to estimate the non-linear relationship, where the exact form of the linearity 
# does not need to be known in advance: 
# - neural networks ~ to PLS
# - multivariate adaptive regression (MARS)
# - support vector machines (SVM)
# - K-nearest neighbors (KNN)



'Neural Networks'

# similar to partial least squared (PLS) where the outcome is derived by interaction with HIDDEN LAYERS
# which are linear combinations of (some or all) original predictors

# the linear combination is transformed for instance in the interval 0-1 for probabilities by a non-linear function such as:
# Logistic regression (sigmoidal) -- to connect the predictor to the hidden layer
# Linear combination  -- to connect the hidden layer to the output

#BACKPROPAGATION#
# As a non-linear regression model, paramters are optimized via min the sum of squared residuals. 
# to facilitate this task a back-propagation algorith works with derivatives to find the optimal parameters
# OUTCOME - not a global solution so you can average different model results


#OVERFITTING# 
# due to the larger number of coef --> overfitting. 
#Earlystopping can limit this issue
# weight decay -- penalization method (elastic net)


####### coding ##########
library(AppliedPredictiveModeling)
library(caret)
library(MASS)
library(nnet) # supports 1 hidden layer neural networks
library(neural)
library(ggplot2)

# example - NN -
nnetFit <- nnet(predictors, outcome, # standardize the PREDICTORS to be on the same scale
                size = 5,
                decay = 0.01, 
                linout = TRUE, # linear relationship between the hidden units and the prediction
                trace = FALSE, # reduce the amount of printed output
                maxit = 500,   # expand the number of iterations to find parameter estimates...
                MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1) # number of parameters used in the model
# H(P + 1) + H + 1
# P predictors, H hidden layers



## using model averaging - between different neural networks - 

nnetAvg <- avNNet(predictors, outcome, 
                  size = 5, 
                  decay = 0.01,
                  repeats = 5,  # how many models to average
                  linout = TRUE,
                  trace = FALSE, 
                  maxit = 500, 
                  MaxNWts = 5 * (ncol(predictors) + 1) + 5 + 1)


# new samples 
predict(nnetFit, newdata)
predict(nnetAvg, newdata)



#################### with data ##########################################################

data(solubility)
ctrl <- trainControl(method = "cv", number = 10) 
# to chose the number of hidden units and the amount of weight decay via resampling 
# you can use the train function with NN

tooHigh <- findCorrelation(cor(solTrainXtrans), cutoff = 0.75) # findCorrelation takes a correlation matrix and determines 
# the column NUMBERS that should be removed to obtain a pairwise correlation btw
# predictors less than 0.75
trainXnnet <- solTrainXtrans[,-tooHigh]
testnXnnet <- solTestXtrans[, -tooHigh]

nnetGrid <- expand.grid(.decay = c(0,0.01, 0.1),
                        .size = c(1:4),
                        .bag = FALSE) # we do not use bagging, but it could replace different random seeds

set.seed(100)

nnetTune <- train(solTrainXtrans, solTrainY, 
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl, 
                  preProcess = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE, 
                  MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1,
                  maxit = 5)


nnetTune # it shows the result of the model with NN

'Multivariate Adaptive Regression Splines'

# exhaustive explanation : http://uc-r.github.io/mars

# The algo uses the two new features for each predictor and makes a linear regression on those new features to predict 
# our variable of interest based on a cutoff point. how? the cut off point which gives rise to the lowest model error is 
# chosen. After another knot point is found using the same methodology. This gives raise to a non-linear pattern
# as more knot points are inserted 
# When all knots are created, you can prune them by keeping only those contributing to predictive accuracy
# I guess the same process can be repeated for other variable

# Pruning --> asses each predictor variable and estimate how much the error rate is decreased by including it in the model

# MARS can build models where the features invole multiple predictors at once. With a second degree MARS model the algorithm 
# would conduct the same search of a single term improving the model, after creating the initial pairs of features, would
# instigate another search to create a sort of interaction term such as : 
# h(Year_Built-2003)*h(Gr_Liv_Area-2274) which is a second degree term

library(AppliedPredictiveModeling)
data(solubility)




marsFit <- earth::earth(solTrainXtrans, solTrainY) # MARS model with pruning

marsFit

summary(marsFit) # contains more information. h() is the hinge function. 
# h(MolWeight - 5.77) is zero when the molecular weight is less than 5.77 
# the mirror image hinge function is shown as h(5.77 - MolWeight)


#### now we show improvement as the process goes along

marsGrid1 <- expand.grid(.degree = 1:2, # 1 degree and 2 degree process
                         .nprune = 2:38)
set.seed(100)
marsTuned <- train(solTrainXtrans, solTrainY, 
                   method = "earth",
                   tuneGrid = marsGrid1,
                   trControl = trainControl(method = "cv"))
marsTuned

head(predict(marsTuned, solTestXtrans))

varImp(marsTuned) # this gives the importance of each predictor
# results range from 0 to 100. ( the closer to 100 the more important the feature is )



















'Support Vector Machines (SVM)'

# focus on  epsilon-"insensitive regression"
# SSE can be influenced by outliers 

# logic: given a threshold set by the user (epsilon), data points with residuals within the threshold DO NOT contribute
#        to the regression fit, while data points with absolute difference greater than the threshold DO contribute 
#        a linear-scale amount. But we do not use squared residuals, so large outliers have a limited effect on the regression 
#        equation.

# outliers are the only points defining a regression line -- POORLY PREDICTED POINTS DEFINE THE LINE --

# when estimating the model the individual training set data points x_(ij) are required for new predictions. of course, 
# this makes the prediction equations quite huge, but for some percentage of the training set samples, the alpha parameters 
# will be exactly zero, indicating that they have not impact on the prediction equation. The data points associated with 
# an alpha parameter of zero are those with no impact on the prediction equation, so are withing the epsilon bound (+/-) epsilon
# THEREFORE only a subset of training set data points where alpha not zero will be used for prediction = Support vectors


# to estimate the model parameters the SVM uses the epsilon loss function but also adds a penalty

library(AppliedPredictiveModeling)
library(kernlab)
library(caret)

data(solubility)

trainingData <- solTrainXtrans # create a dataset of predictors
## add the solubility outcome (the y)
trainingData$Solubility <- solTrainY

svmFit <- ksvm(Solubility~., data = trainingData, # radial basis is the default kernel function
               kernel = "rbfdot", kpar = "automatic",
               C = 1, epsilon = 0.1) # cost value C

svmRTuned <- train(solTrainXtrans, solTrainY, 
                   method = "svmRadial",
                   preProc = c("center", "scale"),
                   tuneLength = 14, # default grid search of 14 cost values
                   trControl = trainControl(method = "cv"))

svmRTuned # gives you the values of C and sigma

svmRTuned$finalModel # we see that the model has used observation from the training set (637/951) = 67% as support vectors












'K - Nearest Neighbors'

# KNN predicts a new sample using the K-closest samples from the training set.
# the neighbors are the K points close to the new sample we are trying to identify. Euclidean distance
# the predicted response for the new sample is then the mean of the K neighbors' responses.

# data with predictors that are on a vastly different scale will generate distances that are weighted toward predictors 
# that have a largest scale. so predictors with the largest scales will contribute most to the distance between samples

### so all predictors be centered and scales (normalized or standardize) prior to performing KNN, otherwise it will focus
 # on areas with a larger range.

# optimal K --> small values usually overfit (not good for predictions) and large values underfit.

# computational demanding -> the k-d- tree which partitions a predictor space using a tree approach. after 
#                            the tree has grown a new sample is placed through the structure. Distances are 
#                            only computed for those training obs in the tree that are close to the new sample

# KNN can have poor predictive performance --> when the local predictor structure is not relevant to the response (same as PCA).
#                                              it might be due to irrelevant or noisy predictors ( that is why we scale) and 
#                                              alternatively we can remove those noise predictors or weight the neighbors contribution
#                                              to the prediction of a new sample based on their distance. Training sample
#                                              that are closer to the new sample contribute more to the predicted response, 
#                                              while those which are farther away contribute to a lower extent. 




library(caret)
library(AppliedPredictiveModeling)
data(solubility)

knnDescr <- solTrainXtrans[, -nearZeroVar(solTrainXtrans)] # this removes column with little variance, like they have all zeros
set.seed(100)

#b <- nearZeroVar(solTrainXtrans)
knnTune <- train(knnDescr, 
                 solTrainY,
                 method = "knn", 
                 preProcess = c("center", "scale"), 
                 tuneGrid = data.frame(.k = 1:10),
                 trControl = trainControl(method = "cv"))
knnTune # K = 4
