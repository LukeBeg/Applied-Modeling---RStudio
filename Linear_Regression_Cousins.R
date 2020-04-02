
'Code and some explanations taken from "Applied Predictive Modelling" book by Kuhn & Johnson. I tried to reproduce and explain the code on my understanding.'


library(AppliedPredictiveModeling)
library(caret)
library(MASS)
data(solubility)
## the data objects begin with "sol"
ls(pattern = "^solT") # show the object in the environment with "^solT". it is like a search function inside the environment
#each column of the data corresponds to a predictor (chemical descriptor) 
# and the rows correspond to compounds
set.seed(2)

sample(names(solTrainX),8) # random sample of column names
# the FP columns corresponds to the binary 0/1 fingerprint predictors that are associated with the presence or absence of a particular
# chemical structure


'Ordinary Linear Regression'

trainingData <- solTrainXtrans # create a dataset of predictors
## add the solubility outcome (the y)
trainingData$Solubility <- solTrainY

lmfitallpredictors <- lm(Solubility ~ ., data = trainingData) # fit a model with all predictors, intercept is automatically added

summary(lmfitallpredictors)
              

lmPred1 <- predict(lmfitallpredictors, solTestXtrans) # predictions
head(lmPred1) 


lmValues1 <- data.frame(obs = solTestY, pred = lmPred1) # creates a dataframe with "obs" and "pred" as columns
defaultSummary(lmValues1) # this works with the package caret
                          # you can see that when confronting obs and preds the precision of the model diminishes (it was optimistic)


'Robust linear model'
# rlm()

rlmFitAllPredictors <- rlm(Solubility ~ ., data = trainingData)

ctrl <- trainControl(method = "cv", number = 10) # the function trainControl specifies the type of resampling, 10 fold crossvalidation

set.seed(100)

lmFit1 <- train(x = solTrainXtrans, y = solTrainY, 
                method = "lm", trControl = ctrl) # train is the function to make cross validation on the "lm" model

lmFit1


# ''' Plots to see how the model explains the data '''

xyplot(solTrainY ~ predict(lmFit1), 
       type = c("p", "g"), #plot the points (type = 'p') and a background grid ('g')
       xlab = "Predicted", 
       ylab = "Observed") # observed values vs predicted

xyplot(resid(lmFit1) ~ predict(lmFit1), 
       type = c("p", "g"), 
       xlab = "Predicted", 
       ylab = "Residuals") # if the model has been well specified, this plot should look like a random cloud of points (as it is shown)




'Perform PCA to ensure that predictors are not correlated'
# the problem with PCA
  # dimension reduction with PCA does not necessarely produce new predictors explaining the response. There can be 
  # two predictors which are correlated like x_1, x_2 and PCA summarizes the relationship using the direction of max variability
  # however, after PCA it could be that the first PCA direction contains no predictive information about the response.
    # PCA does not consider any aspects of the response when it selects its components, it "simply" chases the variability present across predictors.
    # if that variability happens to be to be related to the response variability, the PCA is likely to identify a predictive relationship. This is what 
      # happens in the majority of cases. However, when the variability in the predictor space is not related to the variability of the response then PCR can have 
      # issues in identifying a predictive relationship even if the relationship exists.
set.seed(100)
rlmPCA <- train(solTrainXtrans, solTrainY, method = "rlm", # here it uses a robust LM (RLM)
                preProcess = "pca", trControl = ctrl)
rlmPCA





'Partial Least Squared'
# This means that PLS finds components that maximally summarize the variation of the predictors while simultaneously requiring 
# these components to have maximum correlation with the response

library(pls)
plsFit <- plsr(Solubility ~., data = trainingData) # this uses kernel algorith by default

predict(plsFit, solTestXtrans[1:5,], ncomp = 1:2) # this takes only five rows
                                                  # here we specify the predictions using 1 and 2 components


set.seed(100)
plsTune <- train(solTrainXtrans, solTrainY,
                 methods = "pls",
                 tuneLength = 2, # the default tuning grid evaluates components 1 until tunLength
                 trControl = ctrl,
                 preProcess = c("center", "scale"))

plsTune



'Penalized Regression Models'

# original least squared regression finds parameter estimates to minimize the sum of squared errors : SSE
# however, when the model overfits the data or when there is collinearity between predictors, estimates are inflated (large), so the responsiveness
# is due to correlation and not by data.
# so we add a penalty to the SSE if the estimate become large so that we control the magnitude of these estimates to reduce SSE
# Ridge and Lasso regression add this penalty


# RIDGE  regression 
  # adds a penalty to the sum of the squared regression parameter. basically the equation is all positive and the penalty lambda is also positive
  # so that means adding the penalty will increase the SSE (generally speaking) so the estimates from being inflated will have to be 'shrinked'
  # a bit -- "shrinkage methods"

  # this model shrinks toward zero, but it does not set the value to 0 for any value of the penalty --> NO FEATURE SELECTION

# LASSO regression
  # same logic as above but the penalty added to ABSOLUTE values of the regression parameter. 
  # some values are actually set to zero --> FEATURE SELECTION ... it picks one correlated predictor and drops the other correlated to the one chosen


'Elastic nets combine the ridge and lasso model '
library(elasticnet) # with has both ridge and lasso penalties inside with 2 separate parameters
ridgeModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY, lambda = 0.001) # here you only have the fixed ridge penalty


ridgePred <- predict(ridgeModel, newx = as.matrix(solTestXtrans), 
                     s = 1, mode = "fraction", type = "fit") # predict generates predictions for 
                                                             # one or more values of the lasso penalty using s and mode
                                                             # for RIDGE we want lasso penalty = 0, so we want the full solution
                                                             # and we define s = 1 and mode = 'fraction'. so the fraction of the estimate/1 = estimate, i.e. full solution
                                                            # so the value of 1 corresponds to a fraction of 1

head(ridgePred)



'tune the penalty'
ridgeGrid <- data.frame(.lambda = seq(0, .1, length = 15))
set.seed(100)
ridgeRegFit <- train(solTrainXtrans, solTrainY,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = ctrl,
                     preProcess = c("center", "scale")) # put the predictor on the same scale

ridgeRegFit                   



#lasso 

enetModel <- enet(x = as.matrix(solTrainXtrans), 
                  y = solTrainY, lambda = 0.00, normalize = TRUE) # normalize is to standardize
                                                                  # set the lambda = 0 to get the lasso 


# the lasso penalty does not need to be specified until the time of prediction

enetPred <- predict(enetModel, newx = as.matrix(solTestXtrans), 
                     s = 0.1, mode = "fraction", # here the s = 0.1 so value of the estimate shrinked is 0.9 of the original value
                     type = "fit")
names(enetPred) # the fit component has the predicted values 


enetPred <- predict(enetModel, newx = as.matrix(solTestXtrans), 
                    s = 0.0, mode = "fraction", 
                    type = "coefficients") # to see which predictors are actually used in the model

tail(enetPred$coefficients)


# tuning elastic net 

enetGrid <- expand.grid(.lambda = c(0, 0.01, .1), # ridge penalty
                        .fraction = seq(.05, 1, length = 20)) # lasso penalty

set.seed(100)
        
enetTune <- train(solTrainXtrans, solTrainY,
                     method = "enet",
                     tuneGrid = enetGrid,
                     trControl = ctrl,
                     preProcess = c("center", "scale"))

plot(enetTune)



