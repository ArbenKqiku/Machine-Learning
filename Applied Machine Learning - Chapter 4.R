# Applied Machine Learning - Chapter 4

# Packages
install.packages("rms")
install.packages("ipred")
install.packages("MASS")
install.packages("mvtnorm")
install.packages("RANN")
library("AppliedPredictiveModeling")
library("caret")
library("corrplot")
library("e1071")
library("ipred")
library("rms")
library("MASS")
library("mvtnorm")
library(dplyr)
library(tidyr)
library(pls)
library(RANN)

# load data
data(twoClassData)

# see predictors
str(predictors)
dim(predictors)
names(predictors)
View(predictors)

# see outcome variable
classes

# set seed so that we can always have the same repartition of data. This is useful
# to have reproducible results
set.seed(1)

# matrix of row lines is created
trainingRows = createDataPartition(classes, p = 0.8, list = FALSE)

# create train predictors and train classes
trainPredictors = predictors %>% slice(trainingRows)
trainClasses = classes[trainingRows]
testPredictors = predictors %>% slice(-trainingRows)
testClasses = classes[-trainingRows]

# split the data multiple times
repeatedSplits = createDataPartition(trainClasses, p = 0.8, times = 3)

# you can do the same for bootstrapping
bootstrapSamples = createResample(trainClasses, times = 4)

# R has function such as createResamples (bootstrapping), createFolds (for k-fold cross validation)
# and createMultiFolds (for repeated cross validation)
set.seed(1)
cvSplits = createFolds(trainClasses, k = 10, returnTrain = TRUE)

# get the first 90 % of the data
cvPredictors1 = trainPredictors[cvSplits$Fold01,]
nrow(trainPredictors)
nrow(cvPredictors1) # 90 % of the data

# use k-nearest neighbors to estimate classe
knnFit = knn3(x = trainPredictors, y = trainClasses, k = 5)

# test predictions, with argument type = class, it will assign a dichotomous variable class,
# otherwise, a distribution between classes
testPredictions = predict(object = knnFit, newdata = testPredictors, type = "class")


# try to predict Credit Score
data(GermanCredit)

## First, remove near-zero variance predictors then get rid of a few predictors 
## that duplicate values. For example, there are two possible values for the 
## housing variable: "Rent", "Own" and "ForFree". So that we don't have linear
## dependencies, we get rid of one of the levels (e.g. "ForFree"). A vector
## is linear dependent if it can be described by using 2 other vectors. In
## regression, this means that there is not a uniquely defined solution to 
## estimate coefficients.

GermanCredit <- GermanCredit[, -nearZeroVar(GermanCredit)]
GermanCredit$CheckingAccountStatus.lt.0 <- NULL
GermanCredit$SavingsAccountBonds.lt.100 <- NULL
GermanCredit$EmploymentDuration.lt.1 <- NULL
GermanCredit$EmploymentDuration.Unemployed <- NULL
GermanCredit$Personal.Male.Married.Widowed <- NULL
GermanCredit$Property.Unknown <- NULL
GermanCredit$Housing.ForFree <- NULL

## Split the data into training (80%) and test sets (20%)
set.seed(100)
inTrain <- createDataPartition(GermanCredit$Class, p = .8)[[1]]
GermanCreditTrain <- GermanCredit[ inTrain, ]
GermanCreditTest  <- GermanCredit[-inTrain, ]

# tuneLength determines the tune parameters to evaluate. In this case, between 2^-2 and 2^7.
# by default, basic bootstrap is applied. Use trainControl to specify resampling
set.seed(1056)
svmFit = train(Class ~ .,
               data = GermanCreditTrain,
               method = "svmRadial",
               preProc = c("center", "scale"),
               tuneLength = 10,
               trControl = trainControl(method = "repeatedcv",
                                        repeats = 5))


# plot cost functions
plot(svmFit, scales = list(x = list(log = 2)))

# predict values
predictedClassesSvm = predict(svmFit, newdata = GermanCreditTest)

# logistic regression
set.seed(1056)

logisticReg = train(Class ~ .,
               data = GermanCreditTrain,
               method = "glm",
               trControl = trainControl(method = "repeatedcv",
                                        repeats = 5))

predictedClassLog = predict(logisticReg, newdata = GermanCreditTest)

# compare models
resamp = resamples(list(SVM = svmFit, Logistic = logisticReg))
summary(resamp)

modelDifferences <- diff(resamp)
summary(modelDifferences)
modelDifferences$statistics # p-value is big, no difference between models

## *** EXERCISE 1

data(ChemicalManufacturingProcess)

# have the same random factor
set.seed(123)

# dimensions, names, and structure
dim(ChemicalManufacturingProcess)
names(ChemicalManufacturingProcess)
str(ChemicalManufacturingProcess)

# see if there are any missing values
sum(is.na(ChemicalManufacturingProcess))

# see how many missing values there are per column, to see if it is random or if there is a pattern
apply(ChemicalManufacturingProcess, 2, function(x)sum(is.na(x)))

training.samples = ChemicalManufacturingProcess$Yield %>% 
                    createDataPartition(p = 0.8, list = FALSE)

train.data = ChemicalManufacturingProcess[training.samples,]
test.data = ChemicalManufacturingProcess[-training.samples,]

ChemicalManufacturingProcess = ChemicalManufacturingProcess %>% mutate_if(is.character, factor)

# create preprocessing method
preproc = preProcess(train.data, method = c("knnImpute", "scale", "center", "BoxCox"))

# apply preprocessing method to train data and test data
train.data.preproc = predict(preproc, train.data)
test.data.preproc = predict(preproc, test.data)

model = train(form = Yield ~ .,
              data = train.data.preproc,
              trControl = trainControl(method = "repeatedcv", repeats = 10),
              tuneLength = 10, method = "pls")

# predict outcome values
model.pred = predict(model, test.data.preproc)

# see how accurate those predictions are
postResample(model.pred, test.data.preproc$Yield)

# EXPLANATIONS oneSE and tolerance: https://rdrr.io/cran/caret/man/oneSE.html
# use the one standard error method, try to maximise the R squared metric, repeat
# this process 10 times. Keep the simplest method within one standard error of 
# accuracy
oneSE(model$results, metric = "Rsquared", maximize = TRUE, num = 10)

# if a tolerance of 10% is acceptable, what is the best amount of components
tolerance(x = model$results, metric = "Rsquared", tol = 10, maximize = TRUE)

results = model$results[,c("ncomp", "Rsquared", "RsquaredSD")]
results$RSquaredSEM = results$Rsquared/sqrt(length(model$control$index))

library(ggplot2)

oneSE = ggplot(results, aes(ncomp, Rsquared, ymin = Rsquared - RSquaredSEM,
                            ymax = Rsquared))

# linerange requires a ymin and ymax
# http://sape.inf.usi.ch/quick-reference/ggplot2/geom_linerange
oneSE + geom_linerange() + geom_pointrange() + theme_bw()

# which component has the highest R^2
bestR2 = subset(results, ncomp == which.max(results$Rsquared))

# calculate tolerance from the best result. In %, how far is each component from the 
# best result? 
results$tolerance = (results$Rsquared - bestR2$Rsquared)/bestR2$Rsquared * 100

# with the qplot, you can visually see how far away certain points are, you can have a better
# understanding of the tolerance
qplot(data = results, x = ncomp, y = tolerance)

## EXERCISE 2

data(oil)
str(oilType)

# a) Use the sample function in base R to create a completely random sample
# of 60 oils. How closely do the frequencies of the random sample match
# the original samples? Repeat this procedure several times of understand
# the variation in the sampling process.

oilSample = sample(oilType, size = 60, replace = TRUE)

# percentage of frequencies of the sample
round(table(oilSample)/length(oilSample)*100,2)

# % of freq of the actual data
round(table(oilType)/length(oilType)*100,2)

# (b) Use the caret package function createDataPartition to create a stratified
# random sample. How does this compare to the completely random samples?
# with a stratified sample you get a sample that closely matched the original sample

oilStratLines = createDataPartition(y = oilType, p = 0.61, list = FALSE)
oilStratSample = oilType[oilStratLines]

# percentage of frequencies of the sample
round(table(oilStratSample)/length(oilStratSample)*100,2)

# % of freq of the actual data
round(table(oilType)/length(oilType)*100,2)

# let's examine confidence intervals for overall accuracy
getWidth = function(values){
  binom.test(x = floor(values["size"]*values["accuracy"])+1,
            n = values["size"])$conf.int    
}

# go from .7 to .95 for accuracy and each time raise size by 1
cfInfo = expand.grid(size = 10:30, accuracy = seq(.7, .95, by = 0.01))
cfWidth = t(apply(cfInfo, 1, getWidth))

cfInfo$length = cfWidth[,2] - cfWidth[,1]

# this plot shows that as the size increases, the level of accuracy increases as well,
levelplot(length ~ size * accuracy, data = cfInfo)

  
