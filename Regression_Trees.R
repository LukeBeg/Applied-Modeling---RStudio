
'Code and some explanations taken from "Applied Predictive Modelling" book by Kuhn & Johnson. I tried to reproduce and explain the code based on my own understanding.'


# tree-based models consist of one or more nested if-then clauses for the predictors that parition the data: 

#     if Predictor A >= 1.7 then
#      | if Predictor B >= 202.1 then Outcome = 1.3
#      | else Outcome = 5.6 
#     else Outcome = 2.5

# to obtain the prediction of a new sample, one has to follow the tree (if-then statements)
# using the values from that sample predictors until the terminal node is reached. The model 
# formula in the terminal node would then be used to generate the prediction. The terminal node 
# can be defined as a specific function...

# simple tree has 2 weaknesses: 
#     - model instability -> slight change in data can change the structure of the tree and the interpretation
#     - less-than-optimal prediction -> these models define rectangular regions and if the relationship btw the 
#                                       predictors and the response cannot be defined by such rectangular subspaces then 
#      solution ->  (ENSAMBLE METHODS)  the tree-based model will have large prediction errors.



'Basic Regression Trees'

# to achieve outcome homogeneity regression trees determine :
#     - the predictor to split on and value of the split
#     - the depth or complexity of the tree
#     - the prediction equation in the terminal nodes (in this section the model in the terminal nodes are simple constants)


# to build a regression tree 
#    - CART -> which begins with the entire dataset and searches across all predictors to find which predictor and at 
#              what value to separate the data in two groups (S_1 and S_2) such that the overall sum of squared residuals is minimized.
#           (recursive partitioning)
#   the process continues WITHIN S_1 and S_2 until the number of samples in the splits falls below some threshold (arbitrarily set)
# once the tree has fully grown, it may overfit the training set --> pruning - by for instance penalizing the error using the size of the tree

# A tree with no splits means that no predictor explains enough of this variation in the outcome at the chosen value 
# of the complexity parameter (which to my understanding it is fixed)

# surrogate splits are to deal with missing data


# To determine the importance of each predictor consider the reduction in optimization criteria (SSE). Predictors that 
# occur higher in the tree or those that appear more than once will be more important.


library(AppliedPredictiveModeling)
library(caret)
library(rpart)
library(party)
data(solubility)


ls(pattern = "^solT") # show the object in the environment with "^solT". it is like a search function inside the environment
#each column of the data corresponds to a predictor (chemical descriptor) 
# and the rows correspond to compounds
set.seed(2)

sample(names(solTrainX),8) # random sample of column names
# the FP columns corresponds to the binary 0/1 fingerprint predictors that are associated with the presence or absence of a particular
# chemical structure

trainingData <- solTrainXtrans # create a dataset of predictors
## add the solubility outcome (the y)
trainingData$Solubility <- solTrainY

rpartTree <- rpart(Solubility ~., data = trainingData) # CART methodlogy
ctree <- ctree(Solubility ~., data = trainingData) # conditional inference framework

set.seed(100)
rpartTune <- train(solTrainXtrans, solTrainY,
                   method = "rpart2", # tuning over the maximum node depth
                   tuneLength = 10, 
                   trControl = trainControl(method = "cv"))

library(partykit)
rpartTree2 <- as.party(rpartTree) # convert the rpart object into a party object
plot(rpartTree2) # plot the converted tree object to see the actual tree



'Regression Model Trees'

# one limitation of simple regression trees is that each terminal node uses the average of the training set outcomes 
# in that node for prediction. As a consequence, these models may not do a good job predicting samples whose outcomes are 
# extremely high or low.


            ### model trees: use a differnt estimator for the terminal nodes
# the splitting criterion is different
# terminal nodes predict the outcome using a linear model (as opposed to simple average)
# when a sample is predicted it is the combination of the predictions from differnt models along the same path though the tree

# IN PRACTICE : the initial split is found using a search over the predictors and training obs but the method 
#               uses the expected reduction in the node's error rate is used. Basically, we observe the variation 
#               (SD) of the dataset as a whole SD(S)and compare it with the weighted (by sample size) standard deviation 
#               in the splits. The split that which produces the largest reduction in error is chosen and a linear 
#               model is created within the partitions using the split variable in the model. Then the process is repeated 
#               for other splitting itereations. The error associated with each linear model split is used in place of SD(S)
#               to establish the expected reduction in the error rate for the next split. (after the first split, S as a whole 
#               does not exist anymore).

## once the tree has grown with all the linear model, each undergoes a procedure to potentially drop some of the terms. For 
## a given model an adjusted error rate is computed (difference betwen y and y_hat multiplied by a term penalizing models 
## with large numbers of parameters). Each model term is dropped and the adjusted error rate is computed. Terms are dropped 
## from the model as long as the adjusted error rate decreases. The adjusted error rate is used also for pruning the tree:
#  starting from the bottom the adjusted error rate with and without the sub-tree is computed. If the error tree does not
# reduce the adjusted error rate, it is pruned away. This process keeps going until no sub-tree can be removed

# then there is a smoothing function -- the goes bottom-up once the model has been predicted. this smoothing function 
# basically walks back the path of the tree from the end and combines all the models in the path the new sample falls through.

# why smoothing? 
# 1) number of training sample decreases at each split -- very different models as outcome. 
# 2) to mitigate collinearity between two predictors which might be both used for the split and become candidates for linear 
#   model. There would be two terms in the linear model for one piece of information.


# when the tree is not pruned, smoothing significantly improves the error rate.

site_path = R.home(component = "home")
fname = file.path(site_path, "etc", "Rprofile.site")
file.exists(fname)
file.exists("~/.Rprofile")


file.edit("~/.Rprofile")

library(RWeka)
library(caret)
m5tree <- M5P(y ~., data = trainingData)
m5tree <- MP5(y ~., data = trainingData, 
              control = Weka_control(M=10)) # the min number of training set points needed to create additional splits was 
                                            # increased (from 4 (default) to 10)
m5rules <- M5Rules(y ~ ., data = trainingData)


set.seed(100)

m5Tune <- train(solTrainXtrans, solTrainY,
                method = "M5", 
                trControl = trainControl(method = "cv"),
                control = Weka_control(M=10))

plot(m5Tune)












'Bagged Trees'

# bagging = boostrap aggregation
    #  for i=1 to m do
    #    Generate a bootstrap sample of the original data
    #    Train an unpruned tree model on this sample
    #  end

# each model in the ensemble is then used to generate a prediction for a new sample ( bootstrapped samples) and these m 
# predictions are averaged to give to give the bagged model's prediction. The final average prediction has lower variance 
# than the variance across the individual predictions.

# bagging models provide their own internal estimate of predictive performance
#   when doing bootstrapped samples for each model in the ensamble, certain samples are left out. These are called 
#   out-of-bag samples and can be used to assess the predictive performance of that specific model since they are not 
#   used to build the model. Every model in the ensamble generates a measure of predictive performance... the average
#   of the out-of-bag predictive performance can be used to understand the predictive performance of the entire ensamble

# usually m<10 gives the best performance...if not there are other models (RF and boosting).



library(ipred)
baggedTree <- ipredbagg(solTestY, solTestXtrans) # different way of writing Y,x
baggedTree <- bagging(Solubility ~., data = trainingData)

# conditional inference trees can also be bagged using the cforest in the party package IFF the mtry is equal to the number
# of predictors.

library(party)
bagCtrl <- cforest_control(mtry = ncol(trainingData)-1)
baggedTree <- cforest(Solubility ~ ., data = trainingData, controls = bagCtrl)















'Random Forest'

# Generating bootstrap samples introduces a random component into the tree building process, which 
# induces a distribution of trees and thefore also a distribution of predicted values for each sample. 

# Trees in bagging are not however completely independent of each other since all of the original predictors are considered 
# at every split of every tree. So if we start with a large number of samples and a relationship between predictors
# and response that can be modeled by a tree, then trees from different samples may have similar structures to each other
# especially at the top due to this underlying relationship between predictors. (TREE CORRELATION issue)


# Select the number of models to build, m 
'
for i=1 to m do
    Generate a bootstrap sample of the original data
    Train a tree model on this sample
 for each split do
   Randomly select k (< P ) of the original predictors
   Select the best predictor among the k predictors and partition the data
 end
 Use typical tree model stopping criteria to determine when a tree is complete (but do not prune)
enD
'


# since the algo randomly selects predictors at each split, tree correlation will be lower
# usually the number of selected predictor is P/3

# Compared to bagging, random forests is more computationally efficient on a tree-by-tree basis 
# since the tree building process only needs to evaluate a fraction of the original predictors at each split, 
# although more trees are usually required by random forests.


'THE ENSEMBLE NATURE OF THE RANDOM FOREST MAKES IT IMPOSSIBLE TO GAIN AN UNDERSTANDING OF THE RELATIONSHIP
BETWEEN PREDICTORS AND RESPONSE, BUT YOU CAN ESTIMATE THE IMPORTANCE OF EACH PREDICTOR in terms of improvement in 
node purity.'


library(randomForest)

rfModel <- randomForest(solTrainXtrans, solTrainY)
# or 
#rfModel <- randomForest(Solubility ~., data = trainingData)


# more elaborate
rfModel <- randomForest(solTrainXtrans, solTrainY,
                        importance = TRUE, # importance metric by default is not computed to save time
                        ntrees = 1000) # mtry is by default P/3
                                       # 1000 bootstrap to provide stable reproducible result. so if you run the model 
                                       # tmrw you would get the same predictions more or less.

rfModel                                      









'Boosting Trees'


# weak learner (regression trees) are combined (boosted) to produce an ensamble classifier with a superior accuracy

# AdaBoost --> it was shown to be a powerful prediction tool, usually outperforming any individual model 
#              and can be interpreted as a forward stagewise additive model that minimizes exponential loss
#          --> 'gradient boosting machines'
#                 -- given a loss function and a weak learner, the algo seeks to find an additive model 
#                    that minimizes the loss function. The algo is initialized with the best guess of the response 
#                    (the mean of the response in regression). The gradient (the residual) is calculated and 
#                    a model is fit to the residuals to minimize the loss function. The model is added to the 
#                    previous model, and the procedure continues for a user-specified number of iterations.


# differences with RF 
#         - in RF the trees are independent, max depth and contribute equally to the final model 
#         - in Boosting trees are dependent on past trees have min depth and contribute UNequally to the final outcome


            ####Modification to Gradient Boosting
# the "Shrinkage" was added in Gradient boosting in form of a learning rate -- high impact in reducing RMSE
# the bagging --> random sample a fraction of the training data --> Stochastic  gradient boosting


            ### Variable importance in Boosting
# Variable importance for boosting is a function of the reduction in squared error. The improvement in 
# squared error due to each predictor is summed within each tree in the ensamble. The improvement values for 
# each predictor are then averaged across the entire ensemble to yield an overall importance value. Importance
# of each variable for Boosting considers that the trees are dependent on each other and hence will have 
# correlated structures.


library(gbm)
library(caret)
library(AppliedPredictiveModeling)
data(solubility)
gbmModel <- gbm.fit(solTrainXtrans, solTrainY, distribution = "gaussian")
gbmModel <- gbm(Solubility ~ ., data = trainingData, distribution = "gaussian")

# to tune we define a grid
gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(100, 1000, by = 50),
                        shrinkage = c(0.01, 0.1),
                       n.minobsinnode =10)      # this must be added or it does not work


set.seed(100)

gbmTune <- train(solTrainXtrans, solTrainY, 
                 method = "gbm", # gbm function produces a lot of output so put verbose = False
                 tuneGrid = gbmGrid,
                 verbose = FALSE)

gbmTune

