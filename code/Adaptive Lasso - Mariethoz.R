library(ggplot2)
library(gridExtra)
library(Matrix)


#We source only the preprocessing part (not the relaxed lasso)
source2 <- function(file, start, end) {
  file.lines <- scan(file, what=character(), skip=start-1, nlines=end-start+1, sep='\n')
  file.lines.collapsed <- paste(file.lines, collapse='\n')
  source(textConnection(file.lines.collapsed))
}

setwd("C:/Users/zacma/Documents/Rstud/SMforHDD/SMHDD-Project-Repo-main/SMHDD-Project-Repo-main/code")
source2("Preprocessing & Relaxed Lasso - Careddu.R",1,135)


#The code takes a lot of time to run (especially when using the weights obtained by performing Ridge)
#The results are already shown as comments
#When there are multiple gamma for which cv error is minimum, we decide to pick the gamma with corresponding number of non-zero coefficient minimum (for lambda_min,afterwards, for lambda_1se, we consider the biggest instead)





#Remove the intercept

x_matrix <- as.matrix(train_data)[,2:8314]

x_matrix_test <- as.matrix(test_data)[,2:8314]


#Normal Lasso to compare the results
library(doParallel)
np_cores <- detectCores() -1
registerDoParallel(np_cores)


set.seed(345)
normal_lasso <- cv.glmnet(x_matrix,train_response,family="binomial",type.measure = "class",alpha=1, parallel=TRUE)
normal_lambda_min <- normal_lasso$lambda.min
cv.error_normal <- normal_lasso$cvm[which(normal_lasso$lambda == normal_lambda_min)]
cv.error_normal
#0.2465483

pred_min_normal <- predict(normal_lasso, newx = x_matrix_test, type="class", s=normal_lambda_min)
accuracy_normal <- mean(pred_min_normal==test_response)
accuracy_normal
#0.7611549

coef_normal_lasso <- coef(normal_lasso,s = normal_lambda_min)
number_of_coeff_normal <-  length(coef_normal_lasso@i[-1]+1)
number_of_coeff_normal
#646


##### Adaptive Lasso #####


set.seed(345)
cv.ridge <- cv.glmnet(x_matrix, train_response, family='binomial',type.measure = "class" ,alpha=0, parallel=TRUE)

err.min <- c()
accuracies_1se <- c()
accuracies_min <- c()
number_of_coeff_1se <- c()
number_of_coeff_min <- c()
powers <- 1.5 + (c(-50:50)/100)


for (i in powers)  {
  w1 <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)[, 1][2:(ncol(x_matrix)+1)] ))^i
  w1[w1[,1] == Inf] <- 999999999
  
  set.seed(345)
  cv.lasso <- cv.glmnet(x_matrix, train_response, family='binomial',type.measure="class", alpha=1, parallel=TRUE, penalty.factor=w1)
  
  lmbda_1se <- cv.lasso$lambda.1se
  lmbda_min <- cv.lasso$lambda.min
  
  #mean square error minimum
  err.min <- c(err.min,cv.lasso$cvm[which(cv.lasso$lambda == lmbda_min)])
  
  pred_1se <- predict(cv.lasso, newx = x_matrix_test, type="class", s=lmbda_1se)
  pred_min <- predict(cv.lasso, newx = x_matrix_test, type="class", s=lmbda_min)
  
  #accuracy
  accuracies_1se <- c(accuracies_1se,mean(pred_1se == test_response))
  accuracies_min <- c(accuracies_min,mean(pred_min == test_response))
  
  coef_adp_1se <- coef(cv.lasso,s = lmbda_1se)
  selected_attributes_1se <- (coef_adp_1se@i[-1]+1)
  
  coef_adp_min <- coef(cv.lasso,s = lmbda_min)
  selected_attributes_min <- (coef_adp_min@i[-1]+1)
  
  number_of_coeff_1se <- c(number_of_coeff_1se, length(selected_attributes_1se))
  number_of_coeff_min <- c(number_of_coeff_min, length(selected_attributes_min))
  }


ridge.min <- min(err.min)
ridge.min
#0.2991453

ridge.power <- powers[which(err.min==min(err.min))]
ridge.power
#1.15 1.16

ridge.acc_1se <- accuracies_1se[which(err.min==min(err.min))]
ridge.acc_1se
#0.6220472 0.6246719

ridge.acc_min <- accuracies_min[which(err.min==min(err.min))]
ridge.acc_min
#0.6325459 0.6325459

ridge.coeff_1se <- number_of_coeff_1se[which(err.min==min(err.min))]
ridge.coeff_1se
#3994 4021

ridge.coeff_min <- number_of_coeff_min[which(err.min==min(err.min))]
ridge.coeff_min
#4137 4141

ridge_sel <- 1.15

set.seed(345)
cv.las <- cv.glmnet(x_matrix, train_response, family='binomial', type.measure = "class" , alpha=1, parallel=TRUE)

err.min <- c()
accuracies_1se <- c()
accuracies_min <- c()
number_of_coeff_1se <- c()
number_of_coeff_min <- c()
powers <- 1 + (c(-50:50)/100)

for (i in powers)  {
  w2 <- 1/abs(matrix(coef(cv.las, s=cv.las$lambda.min)[, 1][2:(ncol(x_matrix)+1)] ))^i
  w2[w2[,1] == Inf] <- 999999999
  
  set.seed(345)
  cv.lasso <- cv.glmnet(x_matrix, train_response, family='binomial', type.measure = "class", alpha=1, parallel=TRUE, penalty.factor=w2)
  
  lmbda_1se <- cv.lasso$lambda.1se
  lmbda_min <- cv.lasso$lambda.min
  
  #mean square error minimum
  err.min <- c(err.min,cv.lasso$cvm[which(cv.lasso$lambda == lmbda_min)])
  
  pred_1se <- predict(cv.lasso, newx = x_matrix_test, type="class", s=lmbda_1se)
  pred_min <- predict(cv.lasso, newx = x_matrix_test, type="class", s=lmbda_min)
  
  #accuracy
  accuracies_1se <- c(accuracies_1se,mean(pred_1se == test_response))
  accuracies_min <- c(accuracies_min,mean(pred_min == test_response))
  
  coef_adp_1se <- coef(cv.lasso,s = lmbda_1se)
  selected_attributes_1se <- (coef_adp_1se@i[-1]+1)
  
  coef_adp_min <- coef(cv.lasso,s = lmbda_min)
  selected_attributes_min <- (coef_adp_min@i[-1]+1)
  
  number_of_coeff_1se <- c(number_of_coeff_1se, length(selected_attributes_1se))
  number_of_coeff_min <- c(number_of_coeff_min, length(selected_attributes_min))
}


lass.min <- min(err.min)
lass.min
#0.1900066

lass.power <- powers[which(err.min==min(err.min))]
lass.power
#0.88

lass.acc_1se <- accuracies_1se[which(err.min==min(err.min))]
lass.acc_1se
#0.7532808

lass.acc_min <- accuracies_min[which(err.min==min(err.min))]
lass.acc_min
#0.7454068

lass.coeff_1se <- number_of_coeff_1se[which(err.min==min(err.min))]
lass.coeff_1se
#233

lass.coeff_min <- number_of_coeff_min[which(err.min==min(err.min))]
lass.coeff_min
#245

lass_sel <- 0.88

set.seed(345)
cv.elast <- cv.glmnet(x_matrix, train_response, family='binomial', type.measure = "class", alpha=0.5, parallel=TRUE)

err.min <- c()
accuracies_1se <- c()
accuracies_min <- c()
number_of_coeff_1se <- c()
number_of_coeff_min <- c()
powers <- 1 + (c(-50:50)/100)

for (i in powers)  {
  w3 <- 1/abs(matrix(coef(cv.elast, s=cv.elast$lambda.min)[, 1][2:(ncol(x_matrix)+1)] ))^i
  w3[w3[,1] == Inf] <- 999999999
  
  set.seed(345)
  cv.lasso <- cv.glmnet(x_matrix, train_response, family='binomial', type.measure = "class", alpha=1, parallel=TRUE, penalty.factor=w3)
  
  lmbda_1se <- cv.lasso$lambda.1se
  lmbda_min <- cv.lasso$lambda.min
  
  #mean square error minimum
  err.min <- c(err.min,cv.lasso$cvm[which(cv.lasso$lambda == lmbda_min)])
  
  pred_1se <- predict(cv.lasso, newx = x_matrix_test, type="class", s=lmbda_1se)
  pred_min <- predict(cv.lasso, newx = x_matrix_test, type="class", s=lmbda_min)
  
  #accuracy
  accuracies_1se <- c(accuracies_1se,mean(pred_1se == test_response))
  accuracies_min <- c(accuracies_min,mean(pred_min == test_response))
  
  coef_adp_1se <- coef(cv.lasso,s = lmbda_1se)
  selected_attributes_1se <- (coef_adp_1se@i[-1]+1)
  
  coef_adp_min <- coef(cv.lasso,s = lmbda_min)
  selected_attributes_min <- (coef_adp_min@i[-1]+1)
  
  number_of_coeff_1se <- c(number_of_coeff_1se, length(selected_attributes_1se))
  number_of_coeff_min <- c(number_of_coeff_min, length(selected_attributes_min))
}


elast.min <- min(err.min)
elast.min
#0.2163051

elast.power <- powers[which(err.min==min(err.min))]
elast.power
#0.50 0.51 0.52

elast.acc_1se <- accuracies_1se[which(err.min==min(err.min))]
elast.acc_1se
#0.7611549 0.7611549 0.7611549

elast.acc_min <- accuracies_min[which(err.min==min(err.min))]
elast.acc_min
#0.7637795 0.7664042 0.7664042

elast.coeff_1se <- number_of_coeff_1se[which(err.min==min(err.min))]
elast.coeff_1se
#76 76 76

elast.coeff_min <- number_of_coeff_min[which(err.min==min(err.min))]
elast.coeff_min
#77 77 77

elast_sel <- 0.52


#Check values of best models

w1 <- 1/abs(matrix(coef(cv.ridge, s=cv.ridge$lambda.min)[, 1][2:(ncol(x_matrix)+1)] ))^ridge_sel
w1[w1[,1] == Inf] <- 999999999

w2 <- 1/abs(matrix(coef(cv.las, s=cv.las$lambda.min)[, 1][2:(ncol(x_matrix)+1)] ))^lass_sel
w2[w2[,1] == Inf] <- 999999999

w3 <- 1/abs(matrix(coef(cv.elast, s=cv.elast$lambda.min)[, 1][2:(ncol(x_matrix)+1)] ))^elast_sel
w3[w3[,1] == Inf] <- 999999999

best_ridge.cv <- cv.glmnet(x_matrix, train_response, family='binomial', type.measure = "class", alpha=1, parallel=TRUE, penalty.factor=w1)

best_lass.cv <- cv.glmnet(x_matrix, train_response, family='binomial', type.measure = "class", alpha=1, parallel=TRUE, penalty.factor=w2)

best_elast.cv <- cv.glmnet(x_matrix, train_response, family='binomial', type.measure = "class", alpha=1, parallel=TRUE, penalty.factor=w3)

best_lambda_ridge <- best_ridge.cv$lambda.min
best_lambda_lass <- best_lass.cv$lambda.min
best_lambda_elast <- best_elast.cv$lambda.min

pred_best_ridge <- predict(best_ridge.cv, newx = x_matrix_test, type="class", s=best_lambda_ridge)
pred_best_lass <- predict(best_lass.cv, newx = x_matrix_test, type="class", s=best_lambda_lass)
pred_best_elast <- predict(best_elast.cv, newx = x_matrix_test, type="class", s=best_lambda_elast)

# C-V errors

cv.error_normal #0.2465483

ridge.min #0.2991453

lass.min #0.1900066

elast.min #0.2163051

#accuracies

accuracy_normal #0.7611549

mean(pred_best_ridge == test_response) #0.6325459

mean(pred_best_lass == test_response) #0.7454068

mean(pred_best_elast == test_response) #0.7664042

#Sensitivity

mean((pred_min_normal==test_response)[test_response==1]) #0.6686747

mean((pred_best_ridge==test_response)[test_response==1]) #0.4036145

mean((pred_best_lass==test_response)[test_response==1]) #0.686747

mean((pred_best_elast==test_response)[test_response==1]) #0.7168675

#Specificity

mean((pred_min_normal==test_response)[test_response==0]) #0.8325581

mean((pred_best_ridge==test_response)[test_response==0]) #0.8093023

mean((pred_best_lass==test_response)[test_response==0]) #0.7906977

mean((pred_best_elast==test_response)[test_response==0]) #0.8046512

#N. non-zero coefficients

number_of_coeff_normal #646

coef_best_ridge <- coef(best_ridge.cv,s = best_lambda_ridge)
number_of_coeff_best_ridge <-  length(coef_best_ridge@i[-1]+1)
number_of_coeff_best_ridge #4137

coef_best_lass <- coef(best_lass.cv,s = best_lambda_lass)
number_of_coeff_best_lass <-  length(coef_best_lass@i[-1]+1)
number_of_coeff_best_lass #245

coef_best_elast <- coef(best_elast.cv,s = best_lambda_elast)
number_of_coeff_best_elast <-  length(coef_best_elast@i[-1]+1)
number_of_coeff_best_elast #77


first_20_norm <- order(abs(coef_normal_lasso),decreasing = TRUE)[1:20]
first_20_ridge <- order(abs(coef_best_ridge),decreasing = TRUE)[1:20]
first_20_lass <- order(abs(coef_best_lass),decreasing = TRUE)[1:20]
first_20_elast <- order(abs(coef_best_elast),decreasing = TRUE)[1:20]

tab_first_20 <- data.frame(Normal_Lasso = colnames(x_matrix)[first_20_norm],
                           Ridge = colnames(x_matrix)[first_20_ridge],
                           Lasso = colnames(x_matrix)[first_20_lass],
                           Elastic_Net = colnames(x_matrix)[first_20_elast]
                           )
View(tab_first_20)
