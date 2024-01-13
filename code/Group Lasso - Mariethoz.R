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


#RData file to avoid wasting time
load("gg_lasso.RData")

#Change to TRUE if you want to run everything (it takes a lot of time, the values are already stored in gg_lasso_class.RData)
run_everything <- FALSE



########### Group Lasso ##########


#transform data into matrix, remove the intercept and change the response variable into {1,-1} (gglasso accepts only factors with levels 1,-1)
x_matrix <- as.matrix(train_data)[,2:8314]
train_response_class <- ifelse(train_response==1,1,-1)

x_matrix_test <- as.matrix(test_data)[,2:8314]
test_response_class <- ifelse(test_response==1,1,-1)


#Normal Lasso to compare the results
set.seed(345)
if (run_everything) {normal_lasso <- cv.glmnet(x_matrix,train_response,family="binomial",type.measure = "class")}
normal_lambda_min <- normal_lasso$lambda.min
cv.error_normal <- normal_lasso$cvm[which(normal_lasso$lambda == normal_lambda_min)]
cv.error_normal
#0.2465483

pred_min_normal <- predict(normal_lasso, newx = x_matrix_test, type="class", s=normal_lambda_min)
accuracy_normal <- mean(pred_min_normal==test_response)
accuracy_normal
#0.7612


library(gglasso)



### Grouping 1 ###



data_levels_length <- sapply(seq(1, ncol(predictors)), function(x){
  length(levels(predictors[,x]))}) -1
data_levels_length <- replace(data_levels_length,data_levels_length==-1,1)


#We start by considering the factors as groups
groups_1 <- c()
for (i in 1:length(data_levels_length)) {groups_1 <- c(groups_1,rep(i,data_levels_length[i]))}


if (run_everything) {fit_group_1 <- gglasso(x=x_matrix,y=train_response_class,group=groups_1,lambda.factor = 0.0001,loss = "logit",eps = 1e-4,maxit = 1e9)}
plot(fit_group_1)


set.seed(345)
if (run_everything) {fit_group_1.cv <- cv.gglasso(x=x_matrix,y=train_response_class,group=groups_1,nfolds=10,lambda.factor = 0.0001,loss = "logit",pred.loss = "misclass",eps = 1e-4,maxit = 1e9)}
plot(fit_group_1.cv)

#Best lambdas
lmbda1se_1 <- fit_group_1.cv$lambda.1se # 0.03360139
lmbdamin_1 <- fit_group_1.cv$lambda.min # 0.01325308

cv.error_1 <- fit_group_1.cv$cvm[which(fit_group_1.cv$lambda == lmbdamin_1)]
cv.error_1
#0.2458909

plot(fit_group_1)
abline(v=log(lmbda1se_1),lty=2,col=2)
abline(v=log(lmbdamin_1),lty=2,col=3)
legend("topright",col = c("green","red"),legend = c("min","1se"),lty = 2)

#Check how many variables are not zero
if (run_everything) {fit1se_1 <- gglasso(x_matrix,train_response_class,lambda = lmbda1se_1,group = groups_1,loss = "logit")}
coef1se_1 <- coef(fit1se_1)[2:8314]
length(coef1se_1[coef1se_1!=0])
# 23

if (run_everything) {fitmin_1 <- gglasso(x_matrix,train_response_class,lambda = lmbdamin_1,group = groups_1,loss = "logit")}
coefmin_1 <- coef(fitmin_1)[2:8314]
length(coefmin_1[coefmin_1!=0])
# 82

#Predictions
pred_1se_1 <- predict(fit_group_1.cv, newx = x_matrix_test, type="class", s=lmbda1se_1)
pred_min_1 <- predict(fit_group_1.cv, newx = x_matrix_test, type="class", s=lmbdamin_1)


#Accuracies
acc1se_1 <- mean(pred_1se_1==test_response_class)
print(acc1se_1)
#0.7428

accmin_1 <- mean(pred_min_1==test_response_class)
print(accmin_1)
#0.7454



### Grouping 2 ###



#We now try to consider all the genes as a single group, for the other variables we will do the same as we did in the last grouping

groups_2 <- groups_1
groups_2[69:557] <- rep(28,489)
groups_2[558:8313] <- groups_2[558:8313] - 488



if (run_everything) {fit_group_2 <- gglasso(x=x_matrix,y=train_response_class,group=groups_2,lambda.factor = 0.0001,loss = "logit",eps = 1e-4,maxit = 1e9)}
plot(fit_group_2)


set.seed(345)
if (run_everything) {fit_group_2.cv <- cv.gglasso(x=x_matrix,y=train_response_class,group=groups_2,nfolds=10,lambda.factor = 0.0001,loss = "logit",pred.loss="misclass",eps = 1e-4,maxit = 1e9)}
plot(fit_group_2.cv)

#Best lambdas
lmbda1se_2 <- fit_group_2.cv$lambda.1se # 0.01325308
lmbdamin_2 <- fit_group_2.cv$lambda.min # 0.007583899

cv.error_2 <- fit_group_2.cv$cvm[which(fit_group_2.cv$lambda == lmbdamin_2)]
cv.error_2
#0.2419461


plot(fit_group_2)
abline(v=log(lmbda1se_2),lty=2,col=2)
abline(v=log(lmbdamin_2),lty=2,col=3)
legend("topright",col = c("green","red"),legend = c("min","1se"),lty = 2)

#Check how many variables are not zero
if (run_everything) {fit1se_2 <- gglasso(x_matrix,train_response_class,lambda = lmbda1se_2,group = groups_2,loss = "logit")}
coef1se_2 <- coef(fit1se_2)[2:8314]
length(coef1se_2[coef1se_2!=0])
# 498

if (run_everything) {fitmin_2 <- gglasso(x_matrix,train_response_class,lambda = lmbdamin_2,group = groups_2,loss = "logit")}
coefmin_2 <- coef(fitmin_2)[2:8314]
length(coefmin_2[coefmin_2!=0]) 
# 502



#Predictions
pred_1se_2 <- predict(fit_group_2.cv, newx = x_matrix_test, type="class", s=lmbda1se_2)
pred_min_2 <- predict(fit_group_2.cv, newx = x_matrix_test, type="class", s=lmbdamin_2)


#Accuracies
acc1se_2 <- mean(pred_1se_2==test_response_class)
print(acc1se_2)
#0.7533

accmin_2 <- mean(pred_min_2==test_response_class)
print(accmin_2)
#0.7664



### Grouping 3 ###



# We group together the mutated genes with their respective gene.We leave the other groups as in the first grouping
groups_assigned <- groups_1

for (i in 1:489) {
  temp <- grepl( sort(colnames(x_matrix)[69:557])[i] , colnames( x_matrix ) )
  temp <- grepl("mut",colnames(x_matrix)) & temp
  groups_assigned[temp] <- rep(groups_assigned[which(colnames(x_matrix) == sort(colnames(x_matrix)[69:557])[i])],length(groups_assigned[temp]))
  }

wrong_numbers <- c(591,623,631,679,684)
index = 516
for (i in wrong_numbers) {
  index <- index + 1
  groups_assigned[which(groups_assigned==i)] <- rep(index,length(which(groups_assigned==i)))
  }

#For gglasso, the variables has to be ordered so that the groups are clustered together in an increasing way
groups_length_3 <- rle(sort(groups_assigned))

new_order_3 <- c()
for (i in groups_length_3$values) {new_order_3 <- c(new_order_3,which(groups_assigned==i))}

x_matrix_3 <- x_matrix[,new_order_3]

groups_3 <- c()
for (i in groups_length_3$values) {groups_3 <- c(groups_3,rep(i,groups_length_3$lengths[i]))}



if (run_everything) {fit_group_3 <- gglasso(x=x_matrix_3,y=train_response_class,group=groups_3,lambda.factor = 0.0001,loss = "logit",eps = 1e-4,maxit = 1e9)}
plot(fit_group_3)



set.seed(345)
if (run_everything) {fit_group_3.cv <- cv.gglasso(x=x_matrix_3,y=train_response_class,group=groups_3,nfolds=10,lambda.factor = 0.0001,loss = "logit",pred.loss="misclass",eps = 1e-4,maxit = 1e9)}
plot(fit_group_3.cv)

#Best lambdas
lmbda1se_3 <- fit_group_3.cv$lambda.1se # 0.02789646
lmbdamin_3 <- fit_group_3.cv$lambda.min # 0.01002547

cv.error_3 <- fit_group_3.cv$cvm[which(fit_group_3.cv$lambda == lmbdamin_3)]
cv.error_3
#0.2504931

plot(fit_group_3)
abline(v=log(lmbda1se_3),lty=2,col=2)
abline(v=log(lmbdamin_3),lty=2,col=3)
legend("topright",col = c("green","red"),legend = c("min","1se"),lty = 2)

#Check how many variables are not zero
if (run_everything) {fit1se_3 <- gglasso(x_matrix_3,train_response_class,lambda = lmbda1se_3,group = groups_3,loss = "logit")}
unord_coef1se_3 <- coef(fit1se_3)[2:8314]
coef1se_3 <- c(1:8313)
for (i in 1:8313) {coef1se_3[new_order_3[i]] <- unord_coef1se_3[i]}
length(coef1se_3[coef1se_3!=0])
# 29


if (run_everything) {fitmin_3 <- gglasso(x_matrix_3,train_response_class,lambda = lmbdamin_3,group = groups_3,loss = "logit")}
unord_coefmin_3 <- coef(fitmin_3)[2:8314]
coefmin_3 <- c(1:8313)
for (i in 1:8313) {coefmin_3[new_order_3[i]] <- unord_coefmin_3[i]}
length(coefmin_3[coefmin_3!=0])
# 92


#Predictions
pred_1se_3 <- predict(fit_group_3.cv, newx = x_matrix_test[,new_order_3], type="class", s=lmbda1se_3)
pred_min_3 <- predict(fit_group_3.cv, newx = x_matrix_test[,new_order_3], type="class", s=lmbdamin_3)

#pred_1se_3 <- ifelse(pred1se_3 >= 0.5 , 1,0)
#pred_min_3 <- ifelse(predmin_3 >= 0.5 , 1,0)


#Accuracies
acc1se_3 <- mean(pred_1se_3==test_response_class)
print(acc1se_3)
#0.7507



accmin_3 <- mean(pred_min_3==test_response_class)
print(accmin_3)
#0.7559






#The 20 more significant coefficients for each model
first_20_1 <- order(abs(coefmin_1),decreasing = TRUE)[1:20]
first_20_2 <- order(abs(coefmin_2),decreasing = TRUE)[1:20]
first_20_3 <- order(abs(coefmin_3),decreasing = TRUE)[1:20]

table_first_20 <- data.frame(Grouping_1 = colnames(x_matrix)[first_20_1],
                             Grouping_2 = colnames(x_matrix)[first_20_2],
                             Grouping_3 = colnames(x_matrix)[first_20_3])

View(table_first_20)

library(xtable)

#We analyze only the variables until the genes(no mutated genes)
par(mfrow = c(3,1))
barplot(abs(coefmin_1)[1:557])
abline(v=69,lty = 2,col = 2,xpd=T)
title(main = "Grouping 1")
barplot(abs(coefmin_2)[1:557])
abline(v=69,lty = 2,col = 2,xpd=T)
title(main = "Grouping 2")
barplot(abs(coefmin_3)[1:557])
abline(v=69,lty = 2,col = 2,xpd=T)
title(main = "Grouping 3")
axis(1,at = c(30,370),labels = c("Clinical","Genetic"),tick = F)

par(mfrow = c(1,1))


sens <- c(mean((pred_1se_1==test_response_class)[test_response==1]),mean((pred_min_1==test_response_class)[test_response==1]),mean((pred_1se_2==test_response_class)[test_response==1]),mean((pred_min_2==test_response_class)[test_response==1]),mean((pred_1se_3==test_response_class)[test_response==1]),mean((pred_min_3==test_response_class)[test_response==1]))
spec <- c(mean((pred_1se_1==test_response_class)[test_response==0]),mean((pred_min_1==test_response_class)[test_response==0]),mean((pred_1se_2==test_response_class)[test_response==0]),mean((pred_min_2==test_response_class)[test_response==0]),mean((pred_1se_3==test_response_class)[test_response==0]),mean((pred_min_3==test_response_class)[test_response==0]))

sens
spec


