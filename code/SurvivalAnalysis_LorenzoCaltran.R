

source("Preprocessing & Relaxed Lasso - Careddu.R")

########### SURVIVAL ANALYSIS ############


library(survival)

for (i in 1:nrow(mb)){
  if (mb$overall_survival[i]==1){
    mb$overall_survival[i] <- 0
  }
  else{
    mb$overall_survival[i] <- 1
  }
}

#overall survival = 1 equals to death of the patient, otherwise it's 0

mb[mb == ''] <- NA
status <- mb$overall_survival
y <- mb$overall_survival_months

fitKM <- survfit(Surv(y, event=status) ~ 1)
plot(fitKM, mark.time=T, xlab="Time", ylab="Survival Probability")


library(ggfortify)
library(dplyr)

#We want to verify that older patients have a higher probability of dying

new_data <- mutate(mb, age = ifelse((mb$age_at_diagnosis > 60), "Over 60", "Lower than 60"),
              age = factor(age))

age_fit <- survfit(Surv(y, status) ~ age, data=new_data)
autoplot(age_fit, censor.shape = "*", ylab="Survival Probability")


#We want to verify that patients with cancer at later stages have a higher probability of dying


mb[!mb$tumor_stage %in% c(1, 2, 3, 4), ]$tumor_stage <- NA

stage_fit <- survfit(Surv(y, status) ~ mb$tumor_stage, data=mb)
autoplot(stage_fit, conf.int=FALSE, censor.shape = "*", surv.size=2, ylab="Survival Probability")


#We want to verify that patients with cancer at later stages have a higher probability of dying


mb[!mb$tumor_stage %in% c(1, 2, 3, 4), ]$tumor_stage <- NA

stage_fit <- survfit(Surv(y, status) ~ mb$tumor_stage, data=mb)
autoplot(stage_fit, conf.int=FALSE, censor.shape = "*", surv.size=2, ylab="Survival Probability")

#We want to verify that patients with tumor of bigger size have a higher probability of dying

tumor_measure <- rep(NA, nrow(mb))
new_data$tumor_measure <- tumor_measure

for (i in 1:nrow(mb)){
  if (mb$tumor_size[i] > 30){
    new_data$tumor_measure[i] <- "Big (More than 30)"
  }
  if (mb$tumor_size[i] < 30 && mb$tumor_size[i] > 20){
    new_data$tumor_measure[i] <- "Medium (Between 20 and 30)"
  }
  if (mb$tumor_size[i] < 20){
    new_data$tumor_measure[i] <- "Small (Less than 30)"
  }
}

size_fit <- survfit(Surv(y, status) ~ new_data$tumor_measure, data=new_data)
autoplot(size_fit, censor.shape = "*", surv.size=2, ylab="Survival Probability")


#We want to investigate which combination of therapies is more effective

therapy <- rep(NA, nrow(mb))
new_data <- cbind(mb, therapy)

for (i in 1:nrow(mb)){
  if (mb$chemotherapy[i]==1 && mb$hormone_therapy[i]==1 & mb$radio_therapy[i]==1){
    new_data$therapy[i] <- "All three"
  }
  if (mb$chemotherapy[i]==0 && mb$hormone_therapy[i]==0 & mb$radio_therapy[i]==0){
    new_data$therapy[i] <- "None"
  }
  if (mb$chemotherapy[i]==1 && mb$hormone_therapy[i]==0 & mb$radio_therapy[i]==0){
    new_data$therapy[i] <- "Chemo"
  } 
  if (mb$chemotherapy[i]==0 && mb$hormone_therapy[i]==1 & mb$radio_therapy[i]==0){
    new_data$therapy[i] <- "Hormone"
  } 
  if (mb$chemotherapy[i]==0 && mb$hormone_therapy[i]==0 & mb$radio_therapy[i]==1){
    new_data$therapy[i] <- "Radio"
  }
  if (mb$chemotherapy[i]==1 && mb$hormone_therapy[i]==1 & mb$radio_therapy[i]==0){
    new_data$therapy[i] <- "Chemo-Hormone"
  }
  if (mb$chemotherapy[i]==0 && mb$hormone_therapy[i]==1 & mb$radio_therapy[i]==1){
    new_data$therapy[i] <- "Hormone-Radio"
  }
  if (mb$chemotherapy[i]==1 && mb$hormone_therapy[i]==0 & mb$radio_therapy[i]==1){
    new_data$therapy[i] <- "Chemo-Radio"
  }

}

therapy_fit <- survfit(Surv(y, status) ~ new_data$therapy, data=new_data)
autoplot(therapy_fit, conf.int=FALSE, censor.shape = "*", surv.size=2, ylab="Survival Probability")


boxplot(new_data$age_at_diagnosis ~ new_data$therapy, xlab="Therapy", ylab="Age")

boxplot(new_data$tumor_size ~ new_data$therapy, xlab="Therapy", ylab="Tumor size")


#We want to see which type of surgery is more effective

surgery_fit <- survfit(Surv(y, status) ~ mb$type_of_breast_surgery, data=mb)
autoplot(surgery_fit, censor.shape = "*", ylab="Survival Probability")

#We now extract the genomic data to apply the various survival models

library(tidytable)
x <- mb[, 30:691]

#NA filling


for(i in 1:ncol(x)){
  x[is.na(x[,i]), i] <- mean(x[,i], na.rm = TRUE)
}

x <- get_dummies(x)
is.num <- sapply(x, is.numeric)
x <- x[, ..is.num]


train_data <- x[train_indices, ]
test_data <- x[-train_indices, ]

train_response <- y[train_indices]
test_response <- y[-train_indices]

#COX MODEL WITH LASSO PENALIZATION

m1 <- glmnet(train_data, Surv(train_response, event=status[train_indices]), family="cox", alpha=1)
plot(m1)
plot(m1, xvar="lambda")



#Apply cross validation to find the best model

m1.cv <- cv.glmnet(as.matrix(train_data), Surv(train_response, event=status[train_indices]), family="cox", alpha=1, nfolds=5, parallel=TRUE)
plot(m1.cv)
best_lambda <- m1.cv$lambda.min
pred1 <- predict(m1.cv, newx = as.matrix(test_data), type="response", s=best_lambda)
Cindex1 <- Cindex(pred1, Surv(test_response, status[-train_indices]))
Cindex1



#COX MODEL WITH RIDGE PENALIZATION

m2 <- glmnet(as.matrix(train_data), Surv(train_response, event=status[train_indices]), family="cox", alpha=0)
plot(m2)
plot(m2, xvar="lambda")



#Apply cross validation to find the best model

m2.cv <- cv.glmnet(as.matrix(train_data), Surv(train_response, event=status[train_indices]), family="cox", alpha=0, nfolds=5, parallel=TRUE)
plot(m2.cv)
best_lambda <- m2.cv$lambda.min
pred2 <- predict(m2.cv, newx = as.matrix(test_data), type="response", s=best_lambda)
Cindex2 <- Cindex(pred2, Surv(test_response, status[-train_indices]))
Cindex2


#COX MODEL WITH ELASTIC NET REGULARIZATION (ALPHA=0.5)

m3 <- glmnet(as.matrix(train_data), Surv(train_response, event=status[train_indices]), family="cox", alpha=0.5)
plot(m3)
plot(m3, xvar="lambda")



#Apply cross validation to find the best model

m3.cv <- cv.glmnet(as.matrix(train_data), Surv(train_response, event=status[train_indices]), family="cox", alpha=0.5, nfolds=5, parallel=TRUE)
plot(m3.cv)
best_lambda <- m3.cv$lambda.min
pred3 <- predict(m3.cv, newx = as.matrix(test_data), type="response", s=best_lambda)
Cindex3 <- Cindex(pred3, Surv(test_response, status[-train_indices]))
Cindex3
