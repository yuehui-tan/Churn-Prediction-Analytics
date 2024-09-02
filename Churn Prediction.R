library(data.table)
library(caTools)
library(randomForest)
library(ggplot2)
library(car)
library(mltools)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(igraph)
library(arules)
library(arulesViz)
library(dplyr)
library(caret)


df <- fread("Churn Prediction Dataset.csv", stringsAsFactors = T)
View(df)
summary(df)
#================= Data Cleaning & EDA ================================================ 

#Senior Citizen should be Factor instead of Int
str(df)
#Converting Senior Citizen to factor
df$SeniorCitizen <- factor(df$SeniorCitizen)


#11 rows with missing values
sum(is.na(df))
na.df <- df[rowSums(is.na(df)) > 0,]
View(na.df)

View(df[tenure==0]) 
#Currently TotalCharges is NA, so we set to 0
df[tenure == 0, TotalCharges:= 0]
df[tenure==0,]

#dropping customerID col
df <- df[,-1,]
summary(df)
View(df)


selected_df = df[,c("tenure","MonthlyCharges", "TotalCharges")]
cor(selected_df)

#Total Charges, Monthly Charges and Tenure seem to be correlated by logic, so we check by vif() function
multi_col <- glm(Churn~tenure+MonthlyCharges+TotalCharges, data=df, family=binomial)
vif(multi_col)

#Remove Total Charges,
df <- select(df, -c("TotalCharges"))

# both VIFs decreased to under 2
multi_col2 <- glm(Churn~tenure+MonthlyCharges, data=df, family=binomial)
vif(multi_col2)

# ------------------------------------------VISUALISATION-------------------------------------------------
# 01 Demographics 
# Gender x Churn
ggplot(df, aes(x = gender, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Gender", x = "Gender", y = "Count") +
  theme_minimal()

# Senior Citizen x Churn
ggplot(df, aes(x = SeniorCitizen, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Senior Citizen Status", x = "Senior Citizen", y = "Count") +
  theme_minimal()

# Partner x Churn
ggplot(df, aes(x = Partner, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Partner", x = "Partner", y = "Count") +
  theme_minimal()

# Dependents x Churn
# Whether the customer has dependents or not
ggplot(df, aes(x = Dependents, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Dependents", x = "Dependents", y = "Count") +
  theme_minimal()


# 02 Service Usage
# Tenure x Churn
# Number of months the customer has stayed with the company
ggplot(df, aes(x = tenure, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Tenure Length", x = "tenure", y = "Count") +
  theme_minimal()

# Contract x Churn
ggplot(df, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Contract Type", x = "Contract Type", y = "Count") +
  theme_minimal()

# Contract x InternetService
ggplot(df, aes(x = InternetService, fill = Contract)) +
  geom_bar(position = "dodge") +
  labs(title = "Distribution of Contract Types by Internet Service",
       x = "Internet Service",
       y = "Count") +
  theme_minimal()

# InternetService x Churn
ggplot(df, aes(x = InternetService, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Internet Service Type", x = "Different Internet Service Provided", y = "Count") +
  theme_minimal()

# InternetService x Charges x Churn
ggplot(df, aes(x = InternetService, y = MonthlyCharges, fill = Churn)) +
  geom_col() +
  labs(title = "Monthly Charges by Internet Service",
       x = "Internet Service",
       y = "Monthly Charges") +
  theme_minimal()

# TechSupport x Churn
ggplot(df, aes(x = TechSupport, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Availability of Tech Support", x = "Availability of Tech Support", y = "Count") +
  theme_minimal()

# TechSupport x Charges
ggplot(df, aes(x = TechSupport, y = MonthlyCharges, fill = Churn)) +
  geom_col() +
  labs(title = "Monthly Charges by Availability of Tech Support", x = "Availability of Tech Support", y = "Monthly Charges") +
  theme_minimal()

# PaymentMethod x Churn
ggplot(df, aes(x = PaymentMethod, fill = Churn)) +
  geom_bar(position = "dodge") +
  labs(title = "Churn by Payment Method Available", x = "Payment Method Offered", y = "Count") +
  theme_minimal()

# ----------------------------------------ASSOCIATION RULES-----------------------------------------------

set.seed(2023)

#Select Categorical Variables for Association Rules
df2 <- select(df,"gender","SeniorCitizen","Partner","Dependents","OnlineSecurity","MultipleLines","PhoneService","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Churn")

summary(df2)
#All factors
str(df2)
#View(df2)

#Remove no internet service,focus on people with internet service
df2 <- subset(df2, !(OnlineSecurity %in% c("No internet service")))
summary(df2)

#Switch to transaction data type
trans <- as(df2, "transactions")

rules <- apriori (trans, parameter = list(minlen = 2),
                         appearance=list(rhs='Churn=No'))
inspect(rules)
defaultrules.df <- as(rules, "data.frame")
#Generated 166 rules, too many rules
defaultrules.df

#Change the parameters for Support & Confidence
#Support = 0.11, Confidence = 0.90 and Min Length = 5, higher support and higher Confidence = lower No. of Rules
rules2 <- apriori (trans, parameter = list(supp=0.11,conf=0.9,minlen = 5),
                    appearance=list(rhs='Churn=No'))

inspect(rules2) 
summary(rules2)

rules2.df <- as(rules2, "data.frame")
#9 rules generated
rules2.df

#Sort rules according to confidence  (highest to lowest) 
rules.by.conf1 <- sort(rules2, by="confidence")
rules.by.conf.df1 <- as(rules.by.conf1, "data.frame")
rules.by.conf.df1 

#Plot of association rules
plot(rules2, method = "graph",  engine = "htmlwidget")


#=================Train-Test Split============================================
#Train-test split

set.seed(8)
split = sample.split(df$Churn, SplitRatio = 0.7)
train = subset(df, split == T)
test = subset(df, split == F)
View(train)


#===========Creating a balanced Train Set=============================
#Number of Churn = Yes is only 26.53%
table(train$Churn)
nrow(train[train$Churn == "Yes",])/nrow(train)
View(train)


#======== Upsampling ==========================================
train.bal = upSample(x=train[,-19], y=train$Churn)
table(train.bal$Class)
View(train.bal)

#=============================Logistic Regression (UNBALANCED TRAIN)============================
#Unbalanced Dataset
View(train)
log.m = glm(Churn~., data=train, family=binomial)
summary(log.m)

log.m = glm(Churn~SeniorCitizen+tenure+InternetService+MultipleLines+DeviceProtection+StreamingTV+StreamingMovies+Contract+PaperlessBilling+PaymentMethod+MonthlyCharges, data =train, family = binomial)
summary(log.m)

train_predicted <- predict(log.m, newdata = train, type = "response")

# Convert train predicted probabilities to binary predictions
train_predicted_class <- ifelse(train_predicted > 0.5, "Yes", "No")

# Make predictions on the test set
test_predicted <- predict(log.m, newdata = test, type = "response")

# Convert test predicted probabilities to binary predictions
test_predicted_class <- ifelse(test_predicted > 0.5, "Yes", "No")

# Calculate accuracy
accuracy <- mean(test_predicted_class == test$Churn)
accuracy
#$0.8031

table(test$Churn, test_predicted_class, deparse.level = 2)

#test$Churn   No  Yes
#No         1386  166
#Yes        250   311

F1_Score(test$Churn, test_predicted_class)
#0.8695



#=============================Logistic Regression (BALANCED)========================================
log.m2 = glm(Class~., data=train.bal, family=binomial)
summary(log.m2)

train.bal_predicted <- predict(log.m2, newdata = train.bal, type = "response")

# Convert train predicted probabilities to binary predictions
train.bal_predicted_class <- ifelse(train.bal_predicted > 0.5, "Yes", "No")

# Make predictions on the test set
test.bal_predicted <- predict(log.m2, newdata = test, type = "response")

# Convert test predicted probabilities to binary predictions
test.bal_predicted_class <- ifelse(test.bal_predicted > 0.5, "Yes", "No")

# Calculate accuracy
accuracy <- mean(test.bal_predicted_class == test$Churn)
accuracy


table(test$Churn, test.bal_predicted_class, deparse.level = 2)

#test$Churn   No  Yes
#No         1399  153
#Yes        245  316

F1_Score(test$Churn, test.bal_predicted_class)


#Accuracy and F1-Score of Balanced Train Dataset lower than Imbalanced train Dataset


#============================= CART ===================================================
set.seed(2020)
cart.tree1 <- rpart(Churn ~ ., data = train, method = "class",
                    control = rpart.control(minsplit = 2, cp = 0))

printcp(cart.tree1)
plotcp(cart.tree1)
CVerror.cap <- cart.tree1$cptable[which.min(cart.tree1$cptable[,"xerror"]), "xerror"] +
  cart.tree1$cptable[which.min(cart.tree1$cptable[,"xerror"]), "xstd"]
i <- 1; j<- 4
while (cart.tree1$cptable[i,j] > CVerror.cap) {
  i <- i+1
}
cp.opt = ifelse(i>1, sqrt(cart.tree1$cptable[i,1]* cart.tree1$cptable[i-1,1]), 1)

cp.opt
cart.tree2 <- prune(cart.tree1, cp = cp.opt)
print(cart.tree2)
rpart.plot(cart.tree2)

names(cart.tree2$variable.importance)

# 11 important variables
scaledVarImpt <- round(100*cart.tree2$variable.importance/sum(cart.tree2$variable.importance))
scaledVarImpt
# Top 3 - Contract, tenure, OnlineSecurity

tree.y_hat <- predict(cart.tree2, newdata = test, type = "class")
c_mat <- table(test$Churn, tree.y_hat)
c_mat
mean(test$Churn == tree.y_hat)
#78.845%
F1_Score(test$Churn, tree.y_hat)
#0.8653


#=============================================RANDOM FOREST======================================================================
set.seed(2024)
model_rf <- randomForest(Churn~ ., data = train, na.action = na.omit, importance = T)
#model_rf
importance(model_rf)
model_rf
plot(model_rf)

model_rf2 <- randomForest(Churn~ .-gender, data = train, na.action = na.omit, importance = T)
importance(model_rf2)
model_rf2

varImpPlot(model_rf2)
importance(model_rf2)
plot(model_rf2)

y_hat <- predict(model_rf2, newdata = test)
c_mat <- table(test$Churn, y_hat)
c_mat
mean(test$Churn == y_hat)

F1_Score(test$Churn, y_hat)

#========================== XG BOOST =============================================
#Have to do One-hot encoding, all variables needed to be numeric

#One-hot encoding on train set
lab = train[,19]
dummy = dummyVars("~.", data = train[,-19])
newdata = data.frame(predict(dummy, newdata=train[,-19]))
data_train = cbind(newdata, lab)
View(data_train)


#One-hot encoding on test set
lab_test = test[, 19]
dummy = dummyVars("~.", data = test[,-19])
newdata = data.frame(predict(dummy, newdata=test[,-19]))
data_test = cbind(newdata, lab_test)
View(data_test)

# ====================== UpSampling ==================================
#Imbalanced dataset for trainset
table(data_train$Churn)
View(data_train)
#Remove Gender columns 1 and 2
data_train = data_train[, -c(1, 2)]

train_up_sample = upSample(x=data_train[,-44], y=data_train$Churn)
table(train_up_sample$Class)

#================XGBoost Balanced Data==============================
install.packages("xgboost")
library(xgboost)

set.seed(2024)
grid_tune = expand.grid(
  nrounds = c(100,300,500), #number of trees
  max_depth = c(2,4,6,8),
  eta = c(0.01,0.05,0.1,0.2,0.3), #Learning Rate
  gamma = 0,  #pruning, can tune gamma
  colsample_bytree = 1,
  min_child_weight = 1, #the larger, the more conservative the model is
  subsample = 1
)

grid_tune = expand.grid(
  nrounds = 100, #number of trees
  max_depth = c(2),
  eta = c(0.05), #Learning Rate
  gamma = 0,  #pruning, can tune gamma
  colsample_bytree = 1,
  min_child_weight = 1, #the larger, the more conservative the model is
  subsample = 1
)
train_control = trainControl(method = "cv",
                             number = 3,
                             verboseIter = TRUE,
                             allowParallel = TRUE)

xgb_tune = train(x = train_up_sample[,-44],
                 y = train_up_sample[,44],
                 trControl = train_control,
                 tuneGrid = grid_tune,
                 method = "xgbTree",
                 verbose = TRUE)

xgb_tune

#Writing out our Best Model
train_control = trainControl(method = "none",
                             verboseIter = TRUE,
                             allowParallel = TRUE)

#Using bestTune to optimise our parameters
final_grid = expand.grid(
  nrounds = xgb_tune$bestTune$nrounds, #number of trees
  max_depth = xgb_tune$bestTune$max_depth,
  eta = xgb_tune$bestTune$eta, #Learning Rate
  gamma = xgb_tune$bestTune$gamma,  #pruning, can tune gamma
  colsample_bytree = xgb_tune$bestTune$colsample_bytree,
  min_child_weight = xgb_tune$bestTune$min_child_weight, #the larger, the more conservative the model is
  subsample = xgb_tune$bestTune$subsample
)


xgb_model = train(x = train_up_sample[,-44],
                  y = train_up_sample[,44],
                  trControl = train_control,
                  tuneGrid = final_grid,
                  method = "xgbTree",
                  verbose = TRUE)


importance <- xgboost::xgb.importance(model = xgb_model$finalModel)
importance

xgb.pred = predict(xgb_model, data_test)

#Confusion Matrix
table(data_test$Churn, xgb.pred)

#F1-Score
F1_Score(data_test$Churn, xgb.pred)

confusionMatrix(as.factor(as.numeric(xgb.pred)),
                as.factor(as.numeric(data_test$Churn)))


#=================== XG Boost Imbalanced Data ===================================
set.seed(2024)
View(data_train)
grid_tune = expand.grid(
  nrounds = c(100,300,500), #number of trees
  max_depth = c(2,4,6,8),
  eta = c(0.01,0.05,0.1,0.2,0.3), #Learning Rate, lower value prevents overfitting
  gamma = 0,  #pruning, can tune gamma
  colsample_bytree = 1,
  min_child_weight = 1, #the larger, the more conservative the model is
  subsample = 1
)

train_control = trainControl(method = "cv",
                             number = 3,  #3-fold cross validation
                             verboseIter = TRUE,
                             allowParallel = TRUE)

xgb_tune = train(x = data_train[,-44],
                 y = data_train[,44],
                 trControl = train_control,
                 tuneGrid = grid_tune,
                 method = "xgbTree",
                 verbose = TRUE)

xgb_tune
xgb_tune$bestTune

#Highest Accuracy Parameters:
# nrounds   max_depth   eta   
#   100         2       0.05       

#Writing out our Best Model
train_control = trainControl(method = "none",
                             verboseIter = TRUE,
                             allowParallel = TRUE)

final_grid = expand.grid(
  nrounds = xgb_tune$bestTune$nrounds, #number of trees
  max_depth = xgb_tune$bestTune$max_depth,
  eta = xgb_tune$bestTune$eta, #Learning Rate
  gamma = xgb_tune$bestTune$gamma,  #pruning, can tune gamma
  colsample_bytree = xgb_tune$bestTune$colsample_bytree,
  min_child_weight = xgb_tune$bestTune$min_child_weight, #the larger, the more conservative the model is
  subsample = xgb_tune$bestTune$subsample
)


xgb_model = train(x = data_train[,-44],
                  y = data_train[,44],
                  trControl = train_control,
                  tuneGrid = final_grid,
                  method = "xgbTree",
                  verbose = TRUE)

#Important Variables
importance <- xgboost::xgb.importance(model = xgb_model$finalModel)
importance
#Gain is improvement in accuracy by a feature to the branches it is on

#Variable importance
varImp(xgb_model)
xgb_imp <- xgb.importance(feature_names = xgb_model$finalModel$feature_names,
                          model = xgb_model$finalModel)
xgb.plot.importance(xgb_imp)

xgb.pred = predict(xgb_model, data_test)

#Confusion Matrix
confusionMatrix(as.factor(as.numeric(xgb.pred)),
                as.factor(as.numeric(data_test$Churn)))
table(data_test$Churn, xgb.pred)

#F1-Score
F1_Score(data_test$Churn, xgb.pred)



















