library(glmnet)
library(Matrix)

library(foreach)
library(doParallel)
#save(list = ls(), file = "preprocessing_data.RData")
load("preprocessing_data.RData")





####RIDGE REGRESSION
custom_lambda_sequence <- 10^seq(1, -4, length = 100)#default lambda sequence doesn't include the minimum 
fit_ridge<-glmnet(x=train_data,y=train_response,family="binomial",alpha=0,lambda=custom_lambda_sequence)



plot(fit_ridge, xvar = "lambda")
#plot(fit_ridge,xvar="dev")



cvfit_ridge<-cv.glmnet(x=train_data,y=train_response,family="binomial",alpha=0,lambda=custom_lambda_sequence)


bestlam_ridge <- cvfit_ridge$lambda.min
bestlam_ridge
abline(v=log(bestlam_ridge),col="red",lty=2)

# lambda.1se 
bestlam_ridge.1se <- cvfit_ridge$lambda.1se
bestlam_ridge.1se
abline(v=log(bestlam_ridge.1se),col="green",lty=2)

plot(cvfit_ridge)


#glm with Lambda equal 0,No regularization
#pred <- predict(fit_ridge, s = 0, newx = test_data,
#                exact = TRUE, type="class",x = train_data, y = train_response)
#pred_accuracy<-mean(pred==test_response)
#pred_accuracy



ridge_pred=predict(cvfit_ridge,newx=test_data,type="class", s="lambda.min")
ridge_pred_accuracy<-mean(ridge_pred==test_response)
ridge_pred_accuracy



ridge_1se_pred=predict(cvfit_ridge,newx=test_data,type="class", s="lambda.1se")
ridge_1se_pred_accuracy<-mean(ridge_pred==test_response)
ridge_1se_pred_accuracy

conf_matrix <- table(Actual =test_response, Predicted = ridge_1se_pred)
print(conf_matrix)



# Load necessary library for AUC-ROC
library(pROC)

# Confusion Matrix


# Precision
precision_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
precision_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Recall
recall_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
recall_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# F1 Score
f1_score_class_0 <- 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
f1_score_class_1 <- 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)

# Specificity
specificity_class_0 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity_class_1 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

# AUC-ROC
ridge_1se_pred_numeric <- as.numeric(as.character(ridge_1se_pred))

# Create ROC curve
roc_curve <- roc(test_response, ridge_1se_pred_numeric)
roc_curve <- roc(test_response, ridge_1se_pred)
auc_roc <- auc(roc_curve)

# Create a dataframe
metrics_ridge_df <- data.frame(
  Class = c("Class 0", "Class 1", "Weighted Average"),
  Precision = c(precision_class_0, precision_class_1, (precision_class_0 + precision_class_1) / 2),
  Recall = c(recall_class_0, recall_class_1, (recall_class_0 + recall_class_1) / 2),
  F1_Score = c(f1_score_class_0, f1_score_class_1, (f1_score_class_0 + f1_score_class_1) / 2),
  Specificity = c(specificity_class_0, specificity_class_1, (specificity_class_0 + specificity_class_1) / 2),
  AUC_ROC = c(NA, NA, auc_roc)
)

# Print the dataframe
print(metrics_ridge_df)



#get_coefs <- function(alpha_value, indices) {
#  as.matrix(coef(glmnet(train_data[indices,],train_response[indices],family="binomial",alpha=alpha_value,lambda=bestlam_lasso.1se,parallel=TRUE)))
#}

get_coefs <- function(alpha_value, indices) {
  if (alpha_value == 1) {
    # Use best lambda for lasso
    lambda_value <- bestlam_lasso.1se
  } else if (alpha_value == 0) {
    # Use best lambda for ridge
    lambda_value <- bestlam_ridge
  } else if(alpha_value>0 & alpha_value<1) {
          lambda_value<-best_lam_elastic_net
  }else {
    stop("Invalid alpha_value. It should be between 0 and 1.")
  }
  
  # Perform the glmnet fitting with the selected lambda
  model <- glmnet(
    x = train_data[indices, ],
    y = train_response[indices],
    family = "binomial",
    alpha = alpha_value,
    lambda = lambda_value,
    parallel = TRUE
  )
  
  # Extract and return the coefficients as a matrix
  return(as.matrix(coef(model)))
}


#BOOTSTRAP PROCEDURE 


registerDoParallel(cores = 8)
# Number of bootstrap samples
num_bootstraps <- 1000
set.seed(1)

# Define the bootstrap procedure as a function
get_bootstrap_sample <- function(alph) {
  library(glmnet)
  get_coefs(alph, sample(1:nrow(train_data), nrow(train_data), replace = TRUE))
}

# Run the bootstrap procedure in parallel
ridge_coefs_matrix <- foreach(i = 1:num_bootstraps, .combine = 'cbind') %dopar% {#%dopar% indicates that loop should be done in parallel
  get_bootstrap_sample(0)
}
stopImplicitCluster()

best_ridge<-glmnet(x=train_data,y=train_response,family="binomial",alpha=0,lambda=bestlam_ridge)

sd_ridge_coefs<- apply(ridge_coefs_matrix, 1, function(x) sd(x))

any(is.na(coef(best_ridge)))
any(is.na(sd_ridge_coefs))
# Calculate t-values
t_values_ridge <- coef(best_ridge) / (sd_ridge_coefs+1e-7)

any(is.na(t_values_ridge))
#some t values have NA possibly due to zero variability in predictor or some other unknown reason, so i am filtering them out

na_positions <- which(is.na(t_values_ridge))

print(na_positions)
#na_positions <- which(is.na(t_values_ridge))


# Calculate relative p-values
df <- num_bootstraps  # Degrees of freedom

non_zero_indices <- which(coef(best_ridge) != 0)

p_values_ridge <- 2 * (1 - pt(abs(t_values_ridge[!is.na(t_values_ridge)]), df))


significant_p_values_ridge <- p_values_ridge <= 0.05

# Identify non-zero coefficients
ridge_coef <- predict(cvfit_ridge,type="coefficients",s=bestlam_ridge)
head(ridge_coef)


# Create a data frame for non-zero coefficients
result_ridge_df <- data.frame(
  #Coefficient = rownames(coef(best_ridge))[non_zero_indices],
  Value = coef(best_ridge)[non_zero_indices],
  Standard_deviation = sd_ridge_coefs[non_zero_indices],
  T_Value = t_values_ridge[non_zero_indices],
  P_Value = p_values_ridge[non_zero_indices],
  P_value_significance = significant_p_values_ridge[non_zero_indices]
)


result_ridge_df_ordered <- result_ridge_df[order(result_ridge_df$P_Value), ]

# Print out the top 20 rows
result_ridge_df_ordered<-head(result_ridge_df_ordered, 20)
print(result_ridge_df_ordered)

n_significant<-sum(result_ridge_df_ordered$P_value_significance)
n_significant



# Sort the dataframe by the  values of the 'Value' column
result_ridge_df_ordered <- result_ridge_df_ordered[order(result_ridge_df_ordered$Value, decreasing = FALSE),]




# Assuming you have the 'result_ridge_df_ordered' dataframe

# Plotting without y-axis
plot(result_ridge_df_ordered$Value, seq_along(result_ridge_df_ordered$Value),
     col = ifelse(result_ridge_df_ordered$Value > 0, "blue", "red"),
     pch = 16,
     main = "The 20 most significant Ridge Coefficients",
     xlab = "Coefficient Values",
     ylab = "",  # Suppress default y-axis
     xlim = c(min(result_ridge_df_ordered$Value) - 1, max(result_ridge_df_ordered$Value) + 1),
     yaxt = "n")  # Suppress y-axis

# Adding labels to points and custom y-axis
text(result_ridge_df_ordered$Value, seq_along(result_ridge_df_ordered$Value),
     labels = rownames(result_ridge_df_ordered), pos = 4, cex = 0.8,col="black")

#text(result_ridge_df_ordered$Value, seq_along(result_ridge_df_ordered$Value),
#     labels = rownames(result_ridge_df_ordered), pos = 4, cex = 0.8, col = "green")

#axis(2, at = seq_along(result_ridge_df_ordered$Value), labels = rownames(result_ridge_df_ordered))
#axis(2, at = seq_along(result_ridge_df_ordered$Value), labels = rownames(result_ridge_df_ordered), las = 1,cex.axis=0.5)








####LASSO REGRESSION

fit_lasso<-glmnet(train_data,train_response,family="binomial",alpha=1)

plot(fit_lasso,xvar="lambda")

cvfit_lasso <- cv.glmnet(train_data, train_response, family = "binomial", alpha = 1)




bestlam_lasso <- cvfit_lasso$lambda.min
bestlam_lasso

bestlam_lasso.1se <- cvfit_lasso$lambda.1se
bestlam_lasso.1se

abline(v=log(bestlam_lasso),col="red",lty=2)
abline(v=log(bestlam_lasso.1se),col="green",lty=2)


plot(fit_lasso, xvar = "lambda", label = TRUE, ylim = c(-1, 1), xlim = c(-4, -2))

plot(cvfit_lasso)

best_lasso<-glmnet(train_data,train_response,family="binomial",alpha=1,lambda=bestlam_lasso.1se)
best_lasso
non_zero_coefficients<-coef(cvfit_lasso,s=bestlam_lasso.1se)[coef(cvfit_lasso,s=bestlam_lasso.1se)!=0]
non_zero_coefficients
length(non_zero_coefficients)
length(coef(cvfit_lasso,s=bestlam_lasso.1se))
#

# Function to extract coefficients from glmnet object
#get_coefs <- function(alpha_value, indices) {
#  as.matrix(coef(glmnet(train_data[indices,],train_response[indices],family="binomial",alpha=alpha_value),s=bestlam_lasso.1se))
#}

# Bootstrap procedure
set.seed(1)

registerDoParallel(cores = 8)

#load("bootstrap_lasso_coefficients.RData")
#lasso_coefs <- replicate(num_bootstraps, get_coefs(1, sample(1:nrow(train_data), nrow(train_data), replace = TRUE)))

lasso_coefs_matrix <- foreach(i = 1:num_bootstraps, .combine = 'cbind') %dopar% {#%dopar% indicates that loop should be done in parallel
  get_bootstrap_sample(1)
}
stopImplicitCluster()


# Calculate standard errors
#se_lasso_coefs <- apply(lasso_coefs_matrix, 1, function(x) sd(x) / sqrt(num_bootstraps))
#mean_lasso_coefs<-apply(lasso_coefs_matrix, 1, function(x) mean(x))
sd_lasso_coefs<- apply(lasso_coefs_matrix, 1, function(x) sd(x))

# Calculate t-values
t_values_lasso <- coef(best_lasso) / (sd_lasso_coefs+1e-7)

t_values_lasso[coef(best_lasso)!=0]

# Calculate relative p-values
df <- num_bootstraps  # Degrees of freedom

non_zero_indices <- which(coef(best_lasso) != 0)

p_values_lasso <- 2 * (1 - pt(abs(t_values_lasso[non_zero_indices]), df))


significant_p_values <- round(p_values_lasso,2) <= 0.05

# Identify non-zero coefficients
lasso_1se_coef <- predict(cvfit_lasso,type="coefficients",s=bestlam_lasso.1se)
#lasso_1se_coef
lasso_1se_coef[lasso_1se_coef!=0]

# Create a data frame for non-zero coefficients
result_lasso_df <- data.frame(
  #Coefficient = rownames(lasso_1se_coef)[non_zero_indices],
  Value = coef(best_lasso)[non_zero_indices],
  Standard_deviation = sd_lasso_coefs[non_zero_indices],
  T_Value = t_values_lasso[non_zero_indices],
  P_Value = p_values_lasso,
  P_value_significance=significant_p_values 
)


print(result_lasso_df)




result_lasso_df_ordered <- result_lasso_df[order(result_lasso_df$Value), ]
point_colors <- ifelse(result_lasso_df_ordered$Value > 0, "blue", "red")
text_colors <- ifelse(result_lasso_df_ordered$P_value_significance, "green", "grey")

# Plotting
plot(result_lasso_df_ordered$Value, seq_along(result_lasso_df_ordered$Value),
     col = point_colors,
     pch = 16,
     main = "Non-zero Lasso Coefficients",
     xlab = "Coefficient Values",
     ylab = "Coefficient Names",
     xlim = c(min(result_lasso_df_ordered$Value) - 1, max(result_lasso_df_ordered$Value) + 1),
     yaxt = "n")  # Suppress y-axis

# Adding labels to points and custom y-axis
text(result_lasso_df_ordered$Value, seq_along(result_lasso_df_ordered$Value),
     labels = rownames(result_lasso_df_ordered), pos = 4, cex = 0.7, col = text_colors)

#axis(2, at = seq_along(result_lasso_df_ordered$Value), labels = rownames(result_lasso_df_ordered), las = 1, cex.axis = 0.7)





#fit_lasso<-glmnet(train_data,train_response,family="binomial",alpha=1)
#summary(fit_lasso)

lasso_pred<-predict(cvfit_lasso,newx=test_data,type="class", s="lambda.1se")
lasso_pred_accuracy<-mean(lasso_pred==test_response)
lasso_pred_accuracy

#lasso_coef <- predict(cvfit_lasso,type="coefficients",s=bestlam_lasso)
#lasso_coef
#lasso_coef[lasso_coef!=0]



conf_matrix <- table(Actual = test_response, Predicted = lasso_pred)
print(conf_matrix)

# Load necessary library for AUC-ROC
library(pROC)

# Confusion Matrix


# Precision
precision_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
precision_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Recall
recall_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
recall_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# F1 Score
f1_score_class_0 <- 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
f1_score_class_1 <- 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)

# Specificity
specificity_class_0 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity_class_1 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

# AUC-ROC
lasso_pred_numeric <- as.numeric(as.character(lasso_pred))

# Create ROC curve
roc_curve <- roc(test_response, lasso_pred_numeric)
#roc_curve <- roc(test_response, ridge_1se_pred)
auc_roc <- auc(roc_curve)

# Create a dataframe
metrics_lasso_df <- data.frame(
  Class = c("Class 0", "Class 1", "Weighted Average"),
  Precision = c(precision_class_0, precision_class_1, (precision_class_0 + precision_class_1) / 2),
  Recall = c(recall_class_0, recall_class_1, (recall_class_0 + recall_class_1) / 2),
  F1_Score = c(f1_score_class_0, f1_score_class_1, (f1_score_class_0 + f1_score_class_1) / 2),
  Specificity = c(specificity_class_0, specificity_class_1, (specificity_class_0 + specificity_class_1) / 2),
  AUC_ROC = c(NA, NA, auc_roc)
)

# Print the dataframe
print(metrics_lasso_df)

####ELASTIC NET


set.seed(1)
# Initialize empty lists
alpha_list <- numeric()
error_min_list <- numeric()
error_1se_list <- numeric()
lambda_min_list <- numeric()
lambda_1se_list <- numeric()

custom_lambda_sequence <- 10^seq(1, -6, length = 100)
for (a in seq(0, 1, by = 0.1)) {
  
  cvfit_elastic_net <- cv.glmnet(x = train_data, y = train_response, family = "binomial", alpha = a,lambda=custom_lambda_sequence)
  plot(cvfit_elastic_net)
  title(main = bquote(alpha == .(a)))
  # Extract alpha, lambda.min, lambda.1se, and cross-validated errors
  alpha_value <- cvfit_elastic_net$glmnet.fit$alpha
  lambda_min <- cvfit_elastic_net$lambda.min
  lambda_1se <- cvfit_elastic_net$lambda.1se
  error_min <- cvfit_elastic_net$cvm[cvfit_elastic_net$lambda == lambda_min]
  error_1se <- cvfit_elastic_net$cvm[cvfit_elastic_net$lambda == lambda_1se]
  
  # Append values to lists
  alpha_list <- c(alpha_list, a)
  error_min_list <- c(error_min_list, error_min)
  error_1se_list <- c(error_1se_list, error_1se)
  lambda_min_list <- c(lambda_min_list, lambda_min)
  lambda_1se_list <- c(lambda_1se_list, lambda_1se)
}

# Create a dataframe from the lists
results_df <- data.frame(
  alpha = seq(0, 1, by = 0.1),
  error_min = error_min_list,
  error_1se = error_1se_list,
  lambda_min = lambda_min_list,
  lambda_1se = lambda_1se_list
)

# Print the dataframe
print(results_df)

plot(results_df$alpha, results_df$error_min,ylim = c(min(results_df$error_min)-0.2, max(results_df$error_1se) + 0.1),
     xlab = "Alpha", ylab = "Minimum binomial deviance", main = "Min binomial deviance vs. Alpha",
     pch = 16, col = "red", cex = 1.5)

# Add error bars (standard deviation)
arrows(results_df$alpha, results_df$error_min - (results_df$error_1se-results_df$error_min), results_df$alpha, results_df$error_1se,
       angle = 90, code = 3, length = 0.05, col = "black")

# Optionally, add points to highlight individual data points
points(results_df$alpha[which.min(results_df$error_min)], results_df$error_min[which.min(results_df$error_min)], pch = 16, col = "blue", cex = 1.5)


alpha_chosen<-results_df$alpha[which.min(results_df$error_min)]
#ALPHA=0.6 has the minimum error

best_lam_elastic_net<-results_df$lambda_1se[which.min(results_df$error_min)]

#Using the Lambda 1se for alpha 0.6
best_elastic_net<-glmnet(train_data,train_response,family="binomial",alpha=alpha_chosen,lambda=results_df$lambda_1se[which.min(results_df$error_min)])
#best_elastic_net
non_zero_coefficients_elastic_net<-coef(best_elastic_net)[coef(best_elastic_net)!=0]
non_zero_coefficients_elastic_net
length(non_zero_coefficients_elastic_net)

#


# Bootstrap procedure

registerDoParallel(cores = 8)

#load("bootstrap_lasso_coefficients.RData")
#lasso_coefs <- replicate(num_bootstraps, get_coefs(1, sample(1:nrow(train_data), nrow(train_data), replace = TRUE)))

elastic_net_coefs_matrix <- foreach(i = 1:num_bootstraps, .combine = 'cbind') %dopar% {#%dopar% indicates that loop should be done in parallel
  get_bootstrap_sample(alpha_chosen)
}
stopImplicitCluster()

#save(list=c("elastic_net_coefs_matrix"),file="elastic_net_coefs_matrix.RData")
#load("elastic_net_coefs_matrix.RData")

# Calculate standard errors
#se_elastic_net_coefs <- apply(elastic_net_coefs_matrix, 1, function(x) sd(x) / sqrt(num_bootstraps))

sd_elastic_net_coefs<- apply(elastic_net_coefs_matrix, 1, function(x) sd(x))

# Calculate t-values
t_values_elastic_net <- coef(best_elastic_net) / (sd_elastic_net_coefs+1e-7)

t_values_elastic_net[coef(best_elastic_net)!=0]

# Calculate relative p-values
df <- num_bootstraps  # Degrees of freedom

non_zero_indices_en <- which(coef(best_elastic_net) != 0)

p_values_elastic_net <- 2 * (1 - pt(abs(t_values_elastic_net[non_zero_indices_en]), df))


significant_p_values_en <- p_values_elastic_net <= 0.05

# Identify non-zero coefficients
elastic_net_coef <- predict(best_elastic_net,type="coefficients")
#lasso_1se_coef
elastic_net_coef[elastic_net_coef!=0]

# Create a data frame for non-zero coefficients
result_elastic_net_df <- data.frame(
  #Coefficient = rownames(elastic_net_coef)[non_zero_indices_en],
  Value = coef(best_elastic_net)[non_zero_indices_en],
  Standard_deviation = sd_elastic_net_coefs[non_zero_indices_en],
  T_Value = t_values_elastic_net[non_zero_indices_en],
  P_Value = p_values_elastic_net,
  P_value_significance=significant_p_values_en
)


print(result_elastic_net_df)

result_elastic_net_df_ordered <- result_elastic_net_df[order(result_elastic_net_df$P_Value), ]

# Print out the top 20 rows
result_elastic_net_df_ordered<-head(result_elastic_net_df_ordered, 20)
print(result_elastic_net_df_ordered)

result_elastic_net_df_ordered <- result_elastic_net_df[order(result_elastic_net_df$Value), ]

# Create a color vector for points based on the sign of the values
point_colors_en <- ifelse(result_elastic_net_df_ordered$Value > 0, "blue", "red")

# Create a color vector for text based on P_value_significance
text_colors_en <- ifelse(result_elastic_net_df_ordered$P_value_significance, "green", "grey")

# Plotting
plot(result_elastic_net_df_ordered$Value, seq_along(result_elastic_net_df_ordered$Value),
     col = point_colors_en,
     pch = 16,
     main = "Non-zero Elastic Net Coefficients",
     xlab = "Coefficient Values",
     ylab = "Coefficient Names",
     xlim = c(min(result_elastic_net_df_ordered$Value) - 1, max(result_elastic_net_df_ordered$Value) + 1),
     yaxt = "n")  # Suppress y-axis

# Adding labels to points and custom y-axis
text(result_elastic_net_df_ordered$Value, seq_along(result_elastic_net_df_ordered$Value),
     labels = rownames(result_elastic_net_df_ordered), pos = 4, cex = 0.5, col = text_colors_en)













elastic_net_pred<-predict(best_elastic_net,newx=test_data,type="class")
elastic_net_pred_accuracy<-mean(elastic_net_pred==test_response)
elastic_net_pred_accuracy

conf_matrix <- table(Actual = test_response, Predicted = elastic_net_pred)
print(conf_matrix)


# Precision
precision_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
precision_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Recall
recall_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
recall_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# F1 Score
f1_score_class_0 <- 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
f1_score_class_1 <- 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)

# Specificity
specificity_class_0 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity_class_1 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

# AUC-ROC
elastic_net_pred_numeric <- as.numeric(as.character(elastic_net_pred))

# Create ROC curve
roc_curve <- roc(test_response, elastic_net_pred_numeric)

auc_roc <- auc(roc_curve)

# Create a dataframe
metrics_elastic_net_df <- data.frame(
  Class = c("Class 0", "Class 1", "Weighted Average"),
  Precision = c(precision_class_0, precision_class_1, (precision_class_0 + precision_class_1) / 2),
  Recall = c(recall_class_0, recall_class_1, (recall_class_0 + recall_class_1) / 2),
  F1_Score = c(f1_score_class_0, f1_score_class_1, (f1_score_class_0 + f1_score_class_1) / 2),
  Specificity = c(specificity_class_0, specificity_class_1, (specificity_class_0 + specificity_class_1) / 2),
  AUC_ROC = c(NA, NA, auc_roc)
)

# Print the dataframe
print(metrics_elastic_net_df)


#SPARSE SVM


library(sparseSVM)
#--------------
set.seed(1)
lasso.svm <- sparseSVM(train_data, train_response)
plot(lasso.svm, xvar="norm")
plot(lasso.svm, xvar="lambda")
#
cv.svm <- cv.sparseSVM(train_data,train_response)
abline(v=log(cv.svm$lambda[cv.svm$min]), lty=2)
#1se rule

lambda_ssvm.1se <- min(cv.svm$lambda[cv.svm$cve<=min(cv.svm$cve)+cv.svm$cvse[cv.svm$min]])
lambda_ssvm.1se
max_l<-max(cv.svm$lambda[cv.svm$cve<=min(cv.svm$cve)+cv.svm$cvse[cv.svm$min]])
max_l
abline(v=log(max_l),col="green",lty="dashed",lwd=3)
#abline(v=log(lambda_ssvm.1se), col=3, lty=2)


cv.svm$lambda.min



plot(lasso.svm, ylim=c(-1,1))
abline(v = log(cv.svm$lambda.min), col = "red", lty = "dashed",lwd=3)
abline(v = log(max_l), col = "green", lty = "dashed",lwd=3)


plot(cv.svm )
abline(v = log(cv.svm$lambda.min), col = "red", lty = "dashed")
abline(v = log(max_l), col = "green", lty = "dashed")

best_sparse_svm<-sparseSVM(train_data, train_response,lambda=max_l)

sparse_svm_pred<-predict(best_sparse_svm,X=test_data,type="class")
sparse_svm_pred_accuracy<-mean(sparse_svm_pred==test_response)
sparse_svm_pred_accuracy

conf_matrix <- table(Actual = test_response, Predicted = sparse_svm_pred)
print(conf_matrix)

non_zero_coefficients_sparse_svm<-coef(best_sparse_svm)[coef(best_sparse_svm)!=0]
non_zero_coefficients_sparse_svm
length(non_zero_coefficients_sparse_svm)

#Bootstrap sampling

registerDoParallel(cores = 8)

num_bootstraps<-200
set.seed(1)
lasso_svm_coefs_matrix <- foreach(i = 1:num_bootstraps, .combine = 'cbind') %dopar% {#%dopar% indicates that loop should be done in parallel
  indices<-sample(1:nrow(train_data), nrow(train_data), replace = TRUE)
  
  # Fit lasso SVM and extract coefficients
  
  library(sparseSVM)
  model<-sparseSVM(train_data[indices, ],train_response[indices],lambda=max_l)
  coef_matrix_svm <- as.matrix(coef(model))
  
  return(coef_matrix_svm)
}
stopImplicitCluster()









#compute_ssvm_coefs_matrix <- function(data, indices) {
#  #sampled_data <- data[indices,]
#  sampled_X <- data$X[indices, , drop = FALSE]
#  sampled_y <- data$y[indices]
#  coef_matrix_svm <- as.matrix(coef(sparseSVM(sampled_X,
#                                              sampled_y
#  ),lambda=lambda_ssvm.1se))
#  
#  return(coef_matrix_svm)
#}

# Number of bootstrap samples
#num_bootstraps <- 1000
#data<- list(X = train_data, y = train_response)

# Perform bootstrap resampling using boot function
#boot_lasso_svm_coefs_matrix <- boot(data, statistic = compute_ssvm_coefs_matrix, R = num_bootstraps)

# Print the result
#print(boot_sd_lasso_svm_result)

#save(list=c("lasso_svm_coefs_matrix"),file="lasso_svm_coefs_matrix")
load("lasso_svm_coefs_matrix")


# Calculate standard errors


sd_lasso_svm_coefs<- apply(lasso_svm_coefs_matrix, 1, function(x) sd(x))

# Calculate t-values
t_values_lasso_svm <- as.matrix(best_sparse_svm$weights) / (sd_lasso_svm_coefs+1e-7)

t_values_lasso_svm[best_sparse_svm$weights!=0]

# Calculate relative p-values
df <- num_bootstraps  # Degrees of freedom

non_zero_indices_ssvm <- which(best_sparse_svm$weights != 0 )

p_values_lasso_ssvm <- 2 * (1 - pt(abs(t_values_lasso_svm[non_zero_indices_ssvm]), df))


significant_p_values_ssvm <- p_values_lasso_ssvm <= 0.05



# Create a data frame for non-zero coefficients
result_lasso_svm_df <- data.frame(
  #Coefficient = rownames(lasso_1se_coef)[non_zero_indices],
  Value = best_sparse_svm$weights[non_zero_indices_ssvm],
  Standard_deviation = sd_lasso_svm_coefs[non_zero_indices_ssvm],
  T_Value = t_values_lasso_svm[non_zero_indices_ssvm],
  P_Value = p_values_lasso_ssvm,
  P_value_significance=significant_p_values_ssvm 
)


print(result_lasso_svm_df)

result_lasso_svm_df_ordered <- result_lasso_svm_df[order(result_lasso_svm_df$Value), ]

# Create a color vector for points based on the sign of the values
point_colors_svm <- ifelse(result_lasso_svm_df_ordered$Value > 0, "blue", "red")

# Create a color vector for text based on P_value_significance
text_colors_svm <- ifelse(result_lasso_svm_df_ordered$P_value_significance, "green", "grey")

# Plotting
plot(result_lasso_svm_df_ordered$Value, seq_along(result_lasso_svm_df_ordered$Value),
     col = point_colors_svm,
     pch = 16,
     main = "Non-zero sparse SVM Coefficients",
     xlab = "Coefficient Values",
     ylab = "Coefficient Names",
     xlim = c(min(result_lasso_svm_df_ordered$Value) - 1, max(result_lasso_svm_df_ordered$Value) + 1),
     yaxt = "n")  # Suppress y-axis

# Adding labels to points and custom y-axis
text(result_lasso_svm_df_ordered$Value, seq_along(result_lasso_svm_df_ordered$Value),
     labels = rownames(result_lasso_svm_df_ordered), pos = 4, cex = 0.7, col = text_colors_svm)

# Precision
precision_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
precision_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Recall
recall_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
recall_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# F1 Score
f1_score_class_0 <- 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
f1_score_class_1 <- 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)

# Specificity
specificity_class_0 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity_class_1 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

# AUC-ROC
sparse_svm_pred_numeric <- as.numeric(as.character(sparse_svm_pred))

# Create ROC curve
roc_curve <- roc(test_response, sparse_svm_pred_numeric)

auc_roc <- auc(roc_curve)

# Create a dataframe
metrics_lasso_svm_df <- data.frame(
  Class = c("Class 0", "Class 1", "Weighted Average"),
  Precision = c(precision_class_0, precision_class_1, (precision_class_0 + precision_class_1) / 2),
  Recall = c(recall_class_0, recall_class_1, (recall_class_0 + recall_class_1) / 2),
  F1_Score = c(f1_score_class_0, f1_score_class_1, (f1_score_class_0 + f1_score_class_1) / 2),
  Specificity = c(specificity_class_0, specificity_class_1, (specificity_class_0 + specificity_class_1) / 2),
  AUC_ROC = c(NA, NA, auc_roc)
)

print(metrics_lasso_svm_df)

###### SVM squared hinge
library(gcdnet)
fit_sqsvm <- gcdnet(train_data,train_response, method="sqsvm")
plot(fit_sqsvm)

#
#cvfit_sqsvm <- cv.gcdnet(train_data, train_response, method="sqsvm") 
#cvfit_sqsvm$lambda.min
#plot(cvfit_sqsvm)
cvfit_sqsvm<- cv.gcdnet(train_data, train_response, method="sqsvm", lambda.factor=0.000001) #10 fold cv
plot(cvfit_sqsvm)
lambda_min_sqsvm<-cvfit_sqsvm$lambda.min
lambda_min_sqsvm
lambda_sqsvm.1se<-cvfit_sqsvm$lambda.1se
lambda_sqsvm.1se
plot(fit_sqsvm, xvar="lambda")
fit_sqsvm <- gcdnet(train_data,train_response, method="sqsvm", lambda.factor=0.000001)
plot(fit_sqsvm, xvar="lambda")

abline(v = log(cvfit_sqsvm$lambda.min), col = "red", lty = "dashed")
abline(v = log(cvfit_sqsvm$lambda.1se), col = "green", lty = "dashed")

plot(fit_sqsvm, xvar="lambda",ylim=c(-0.1,0.1))

abline(v = log(cvfit_sqsvm$lambda.min), col = "red", lty = "dashed",lwd=3)
abline(v = log(cvfit_sqsvm$lambda.1se), col = "green", lty = "dashed",lwd=3)


#coefficient with minimum lambda
coef(fit_sqsvm, s=cvfit_sqsvm$lambda.min)[coef(fit_sqsvm, cvfit_sqsvm$lambda.min)!=0]
#
coef.est_sqsvm<- as.matrix(coef(fit_sqsvm, s=cvfit_sqsvm$lambda.1se))
coef.est_sqsvm[coef.est_sqsvm!=0]
length(coef.est_sqsvm[coef.est_sqsvm!=0])
best_sqsvm<-gcdnet(train_data, train_response,method="sqsvm",lambda=lambda_sqsvm.1se)

sqsvm_pred<-predict(best_sqsvm,test_data,type="class")
sqsvm_pred_binary <- ifelse(sqsvm_pred == 1, 1, 0)
sqsvm_pred_accuracy<-mean(sqsvm_pred_binary==test_response)
sqsvm_pred_accuracy

conf_matrix <- table(Actual = test_response, Predicted = sqsvm_pred_binary)
print(conf_matrix)
set.seed(1)
#bootstrap

load("sqsvm_coefs_matrix")
coefs<-sqsvm_coefs_matrix

registerDoParallel(cores = 8)

num_bootstraps<-500

sqsvm_coefs_matrix <- foreach(i = 1:num_bootstraps, .combine = 'cbind') %dopar% {#%dopar% indicates that loop should be done in parallel
  indices<-sample(1:nrow(train_data), nrow(train_data), replace = TRUE)
  #print(i)
  library(gcdnet)
  model<-gcdnet(train_data[indices, ],train_response[indices],lambda=lambda_sqsvm.1se)
  coef_matrix_sqsvm <- as.matrix(coef(model,s=lambda_sqsvm.1se))
  #cat(i)
  return(coef_matrix_sqsvm)
}
stopImplicitCluster()



save(list=c("sqsvm_coefs_matrix"),file="sqsvm_coefs_matrix")


#dimsqsvm_coefs_matrix<-cbind(sqsvm_coefs_matrix,as.matrix(coef(best_sqsvm)))


sd_sqsvm_coefs<- apply(sqsvm_coefs_matrix, 1, function(x) sd(x))

sd_sqsvm_coefs[sd_sqsvm_coefs!=0]
# Calculate t-values
t_values_sqsvm <- coef(best_sqsvm) / (sd_sqsvm_coefs+1e-7)

t_values_sqsvm[coef(best_sqsvm)!=0]


df <- num_bootstraps  # Degrees of freedom

non_zero_indices_sqsvm <- which((coef(best_sqsvm) != 0) )

p_values_sqsvm <- 2 * (1 - pt(abs(t_values_sqsvm[non_zero_indices_sqsvm]), df))


significant_p_values_sqsvm <- p_values_sqsvm <= 0.05



# Create a data frame for non-zero coefficients
result_sqsvm_df <- data.frame(
  #Coefficient = rownames(lasso_1se_coef)[non_zero_indices],
  Value = coef(best_sqsvm)[non_zero_indices_sqsvm],
  Standard_deviation = sd_sqsvm_coefs[non_zero_indices_sqsvm],
  T_Value = t_values_sqsvm[non_zero_indices_sqsvm],
  P_Value = p_values_sqsvm,
  P_value_significance=significant_p_values_sqsvm 
)
print(result_sqsvm_df)




result_sqsvm_df_ordered <- result_sqsvm_df[order(result_sqsvm_df$Value), ]

# Create a color vector for points based on the sign of the values
point_colors_sqsvm <- ifelse(result_sqsvm_df_ordered$Value > 0, "blue", "red")

# Create a color vector for text based on P_value_significance
text_colors_sqsvm <- ifelse(result_sqsvm_df_ordered$Standard_deviation!=0, "yellow", "black")

# Plotting
plot(result_sqsvm_df_ordered$Value, seq_along(result_sqsvm_df_ordered$Value),
     col = point_colors_sqsvm,
     pch = 16,
     main = "Non-zero SQ-SVM Coefficients",
     xlab = "Coefficient Values",
     ylab = "Coefficient Names",
     xlim = c(min(result_sqsvm_df_ordered$Value) - 1, max(result_sqsvm_df_ordered$Value) + 1),
     yaxt = "n")  # Suppress y-axis

# Adding labels to points and custom y-axis
text(result_sqsvm_df_ordered$Value, seq_along(result_sqsvm_df_ordered$Value),
     labels = rownames(result_sqsvm_df_ordered), pos = 4, cex = 0.7, col = text_colors_sqsvm)#text_colors_sqsvm

#axis(2, at = seq_along(result_sqsvm_df_ordered$Value),
#     labels = rownames(result_sqsvm_df_ordered), las = 1, cex.axis = 0.7)



detach(package:gcdnet)

# Precision
precision_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
precision_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Recall
recall_class_0 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
recall_class_1 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# F1 Score
f1_score_class_0 <- 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0)
f1_score_class_1 <- 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1)

# Specificity
specificity_class_0 <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity_class_1 <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

# AUC-ROC
sqsvm_pred_binary_numeric <- as.numeric(as.character(sqsvm_pred_binary))

# Create ROC curve
roc_curve <- roc(test_response, sqsvm_pred_binary_numeric)

auc_roc <- auc(roc_curve)

# Create a dataframe
metrics_sqsvm_df <- data.frame(
  Class = c("Class 0", "Class 1", "Weighted Average"),
  Precision = c(precision_class_0, precision_class_1, (precision_class_0 + precision_class_1) / 2),
  Recall = c(recall_class_0, recall_class_1, (recall_class_0 + recall_class_1) / 2),
  F1_Score = c(f1_score_class_0, f1_score_class_1, (f1_score_class_0 + f1_score_class_1) / 2),
  Specificity = c(specificity_class_0, specificity_class_1, (specificity_class_0 + specificity_class_1) / 2),
  AUC_ROC = c(NA, NA, auc_roc)
)

print(metrics_sqsvm_df)



accuracy_data <- data.frame(
  Model = c("Logistic Ridge", "Logistic Lasso", "Elastic Net","Lasso SVM","Squared SVM"),
  Accuracy = c(ridge_pred_accuracy, lasso_pred_accuracy, elastic_net_pred_accuracy,sparse_svm_pred_accuracy,sqsvm_pred_accuracy)
)


cat("Accuracy Values for Different Models:\n")
print(accuracy_data)






