




#*************************************************** IMPORTANT INFO **********************************************************************
  
  
#  Due to low configuration of my laptop(4 GB RAM), I was unable work with the full dataset using R Studio. While I was working in
#  jupyter using python it was taking some time but I was getting the result, so I have completed all the points that are need to 
#  be done in Python. But in R studio I was unable to do so, so I reduce the size of the dataset and worked on it. So I request you kindly
#  to keep this in mind about the situation. And also I have completed all the thing what should be done according to the points given 
#  in PDF using in R Studio.

#  In Python it absolutly fine and Completed. 

#  In R all things are same as in Python, only the database is small. For that I apologies.

#  I hope you can understand my situation, Thank You.	

#**************************************************** IMPORTANT INFO ***********************************************************************


rm = (list = ls())

setwd('C:/Users/user/Desktop/Santender Project')

library(devtools)
library(caTools)
library(caret)
library(e1071)
library(randomForest)
library(ggplot2)
library(data.table)
library(ggthemes)
library(speedglm)
library(MASS)
library(MASS, quietly = TRUE)
library(bigmemory)
library(ff)
library(tidyr)
library(dplyr)
library(lazyeval)
library(DMwR)
library(pROC)


#let load the dataset
santenderdb <- read.csv(file.path("C:/Users/user/Desktop/Santender Project", "train.csv"))

santenderdb_test <- read.csv(file.path("C:/Users/user/Desktop/Santender Project", "test.csv"))


#I have reduce the dataset to 2000 rows due Memory issue. If I run whole dataset 
#I got an "ERROR : Cannot allocate vector size 76.4GB". For that I reduce to a small size. But I took random sample
#throught out the dataset, to get a natural result.

data_prep <- santenderdb[sample(1:nrow(santenderdb), 2000,
                          replace=FALSE),]


data_prep_test <- santenderdb_test[sample(1:nrow(santenderdb_test), 2000,
                                replace=FALSE),]

#Want to know the columns names
colnames(data_prep)

#Checking the missing Value
sapply(data_prep , function(x) sum(is.na(x)))

#In test set
sapply(data_prep_test , function(x) sum(is.na(x)))


table(data_prep$target)

##  0    1 
## 1793  207 
# This is a Imbalanced target competition.


#Visualized the target columns
target_df <- data.frame(table(data_prep$target))
colnames(target_df) <- c("target", "freq")
ggplot(data=target_df, aes(x=target, y=freq, fill=target)) +
geom_bar(position = 'dodge', stat='identity', alpha=0.5) +
scale_fill_manual("legend", values = c("1" = "dodgerblue", "0"="firebrick1"))



set.seed(123)
split = sample.split(data_prep$target, SplitRatio = 0.5)
training_set = subset(data_prep, split == TRUE)
test_set = subset(data_prep, split == FALSE)


#Modelling
X_train <- data.frame(scale(subset(data_prep, select=-c(data_prep$target, data_prep$ID_code))))
X_test <- data.frame(scale(subset(data_prep, select=-c(data_prep$target, data_prep$ID_code))))


#************************************** Logistic Regression ***************************************
#Fitting Logistic Regression to training set
logisticRegression = glm(formula = target ~ ., family = binomial("logit"), data = training_set[,-1])

summary(logisticRegression)

#Predicting test set results
prob_pred = predict(logisticRegression, type = "response", newdata = test_set[,-2])

Y_prep = ifelse(prob_pred > 0.5, 1, 0)



#Making the Confusion Matrix
caret::confusionMatrix(as.factor(Y_prep), as.factor(test_set$target),
                       positive="1", mode="everything")

#Accuracy was good but F1 Score is too low with 24 percent as we know our dataset is skewed.


cm = table(test_set[, 2], Y_prep)

auc <- roc(test_set[, 2], Y_prep)
print(auc)

#Area under the curve: 0.6196

#    0   1
#0  414  34
#1  40  12

#**********************************************  AUC ROC ***********************************************************
#Lets work on AUC ROC Curve
library(ROCR)
#pred <- predict(logisticRegression, prob_pred, type = "prob")
pred <- prediction(prob_pred, test_set$target)
eval <- performance(pred, "acc")
plot(eval)

#Now to make best cutoff line in the chart
abline(h = 0.78, v = 0.10)



#************************************** Lets Plot some density graph *********************************************

feature_groups <- 3:22
col_names <- colnames(data_prep)[c(2, feature_groups)]
temp <- gather(data_prep[,col_names], key="features", value="value", -target)
temp$target <- factor(temp$target)
temp$features <- factor(temp$features, levels=col_names[-1], labels=col_names[-1])
ggplot(data=temp, aes(x=value)) +
geom_density(aes(fill=target, color=target), alpha=0.3) +
scale_color_manual(values = c("1" = "dodgerblue", "0"="firebrick1")) +
theme_classic() +
facet_wrap(~ features, ncol = 4, scales = "free")


#****************************************** SMOTE **********************************************************

#Because of skewedness and imbalanced dataset we need to use SMOTE technique to balance it

training_set$target <- as.factor(training_set$target)
training_set <- SMOTE(target ~ ., training_set, perc.over = 100, perc.under = 200)
#tran_target <- training_set$target


#***************************************** Logistic Regression after SMOTE ************************************


#Also the dataset taking a very long time to train the data. For that I have used PCA technique
#to reduce the dimension. And again try to use logistic regression


pca <- prcomp((training_set[, -(1:3)]), scale = TRUE)
plot(pca$x[, 1], pca$x[, 2])

leading_score  <- pca$rotation[, 1]

#var_score <- abs(leading_score)

var_score_rank <- sort(leading_score, decreasing = TRUE)

top_70_var <- names(var_score_rank[1:70])

data_prep_Select <- subset(training_set[,-1], select=c(top_70_var, 'target'))

#Modeling After PCA
set.seed(123)
split_pca = sample.split(data_prep_Select$target, SplitRatio = 0.5)

training_set_pca_lr = subset(data_prep_Select, split_pca == TRUE)
#training_target_pca = subset(data_prep_Select, split_pca == TRUE)

test_set_pca_lr = subset(data_prep_Select, split_pca == FALSE)
#test_target_pca = subset(data_prep_Select[, 51], split_pca == FALSE)



#Fitting Logistic Regression to training set
logisticR_smote = glm(formula = training_set_pca_lr$target ~ ., family = binomial, data = training_set_pca_lr)

summary(logisticR_smote)

#Predicting test set results
prob_pred_sm = predict(logisticR_smote, type = "response", newdata = test_set_pca_lr)

Y_prep_sm = ifelse(prob_pred_sm > 0.7, 1, 0)



#Making the Confusion Matrix
caret::confusionMatrix(as.factor(Y_prep_sm), as.factor(test_set_pca_lr$target),
                       positive="1", mode="everything")

#Accuracy was good but F1 Score is 78% with 68% percent of Percision and recall of 82 percent


cm = table(test_set_pca_lr$target, Y_prep_sm)
auc <- roc(test_set_pca_lr$target, Y_prep_sm)
print(auc)

#Area under the curve: 0.72

#Y_prep
#    0   1
#0  65  39
#1  18  86

#*************************************** Random Forest ******************************************************

#Stablishing the training parameters for the random forest, no tune needed.

control <- trainControl(method="repeatedcv", number=20, repeats=2, search="grid", verboseIter=T)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(training_set[,-1]))
tunegrid <- expand.grid(.mtry=mtry)
rf <- train(make.names(target)~., data=training_set[,-1], method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, allowParallel=T, do.trace=25, ntree=100)

probs <- predict(rf, newdata=test_set[,-(1:2)] , type='prob')

#Y_classiy = colnames(probs)[apply(probs, 1, which.max)]

Y_prep_rf = ifelse(probs$X0 > 0.6, 0, 1)


#Confusion MAtrix for random Forest
cm_rf = table(as.factor(test_set$target), as.factor(Y_prep_rf))
auc_rf <- roc(test_set[, 2], Y_prep_rf)
print(auc_rf)

#Area under the curve: 0.56

#   0   1
#0 389  61
#1  31  19

#Making the Confusion Matrix
caret::confusionMatrix(as.factor(Y_prep_rf), as.factor(test_set$target),
                       positive="1", mode="everything")

#F1 score and Precision is low



#*********************************** SVM With PCA *************************************************

pca <- prcomp(training_set[, -(1:3)], scale = TRUE)
plot(pca$x[, 1], pca$x[, 2])

leading_score  <- pca$rotation[, 1]

#var_score <- abs(leading_score)

var_score_rank <- sort(leading_score, decreasing = TRUE)

top_70_var <- names(var_score_rank[1:70])

data_prep_Select <- subset(training_set[,-1], select=c(top_70_var, 'target'))

#Modeling After PCA
set.seed(123)
split_pca = sample.split(data_prep_Select$target, SplitRatio = 0.5)

training_set_pca = subset(data_prep_Select, split_pca == TRUE)
#training_target_pca = subset(data_prep_Select[, 51], split_pca == TRUE)

test_set_pca = subset(data_prep_Select, split_pca == FALSE)
#test_target_pca = subset(data_prep_Select[, 51], split_pca == FALSE)



classifier_svm = svm(formula = target ~ ., 
                 data = training_set_pca, 
                 type = 'C-classification', 
                 kernel = 'linear')



y_pred_svm = predict(classifier_svm, newdata = test_set_pca)

#Confusion Matrix for random Forest
cm_svm = table(test_set_pca$target, y_pred_svm)

#y_pred_svm
#   0   1
#0  67   37
#1  25   79


auc_svm <- roc(test_set_pca$target, as.numeric(y_pred_svm))
print(auc_svm)

#Area under the curve: 0.70


#Making the Confusion Matrix
caret::confusionMatrix(as.factor(y_pred_svm), as.factor(test_set_pca$target),
                       positive="1", mode="everything")

#Here Precision is 68 percent and accuracy is good and also F1 score is 71 percent


#*********************************** KNN *************************************************
library(class)
y_pred_knn = knn(train = training_set[,-1], test = test_set[, -1], 
                 cl = training_set$target, k = 5)

#Confusion Matrix for random Forest
cm_knn = table(test_set$target, y_pred_knn)

#y_pred_knn
#    0   1
#0 449   1
#1  50   0

#Area under the curve: 0.5189

Y_prep_knn = as.numeric(as.character(y_pred_knn))

auc_knn <- roc(test_set$target, as.numeric(Y_prep_knn))
print(auc_knn)

caret::confusionMatrix(as.factor(Y_prep_knn), as.factor(test_set$target),
                       positive="1", mode="everything")

#Here Accuracy and Precision is low and even Area Under Curve is also low




#********************************************************************************************************
# I have used 4 classification techinque and they are:

#1. Logistic Regression
#2. Random Forest Classifier
#3. KNN
#4. SVC

#After running different technique we had a algorithm which is a good fit for our business module and 
#that mean Logistic Regression is the best choice with good Precision, Recall and F1Score.


#Lets predict the test set
probs <- predict(logisticR_smote, newdata=data_prep_test[,2:201], type='response')

data_prep_test$target = ifelse(probs > 0.5, 1, 0)

submission <- data_prep_test[,c('ID_code','target')]



#*******************************************************************************************************
## save this model

saveRDS(logisticR_smote, "logtisR_model.R")


## check if model exists? :
my_model <- readRDS("logtisR_model.R")



#Free of the memory
gc()
memory.size(max=F)

