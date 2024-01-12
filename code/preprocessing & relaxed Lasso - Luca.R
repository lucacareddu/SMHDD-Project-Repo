library(ggplot2)
library(gridExtra)
library(Matrix)

path <- "/home/luca/Documents/Unipd/Courses/II year/II y I s/SMHDD/Proj/metabric.csv"
mb <- read.csv(path)

############# NA CHECK #############

# For numeric columns
empty_values_count <- sort(colSums(is.na(mb)), decreasing = TRUE)
columns_with_empty_values <- names(empty_values_count[empty_values_count > 0])
for (col_name in columns_with_empty_values) {
  count <- empty_values_count[col_name]
  cat("Column:", col_name, "\tEmpty Values Count:", count, "\n")
}

# For character columns
empty_strings_count <- sort(colSums(mb == ""), decreasing = TRUE)
columns_with_empty_strings <- names(empty_strings_count[empty_strings_count > 0])
for (col_name in columns_with_empty_strings) {
  count <- empty_strings_count[col_name]
  cat("Column:", col_name, "\tEmpty Strings Count:", count, "\n")
}

############ FURTHER CHECKS ############

single_value_columns <- sapply(mb, function(col) length(unique(col)) == 1)
names(single_value_columns[single_value_columns])

table(mb$cancer_type)

############# CLEANING #############

mb <- mb[, -which(names(mb) == "patient_id")]

mb <- mb[!(mb$cancer_type == "Breast Sarcoma"),]
mb <- mb[, -which(names(mb) == "cancer_type")]

mb <- mb[!(mb$death_from_cancer == ""),]

############# EXPLORATION #############

mutgenes_columns <- grep("_mut", names(mb), value = TRUE)
mutgenes <- mb[, mutgenes_columns]

image(1:nrow(mutgenes), 1:ncol(mutgenes), z=(mutgenes != 0),
      col = c("darkgreen", "white"),
      ylab = "Mutated Genes", xlab = "Patients", main = "Greens are zeros.")


ggplot(mb, aes(overall_survival_months, fill = as.factor(death_from_cancer))) + 
  geom_histogram(binwidth = 30, boundary = 0, color = "white", size = 0.1, show.legend = FALSE) + 
  scale_fill_manual(name = "Survival", values = c("salmon", "orange", "darkturquoise")) + 
  labs(x = "Survival time in months", y = "Frequency") + 
  facet_wrap(~death_from_cancer)


genes = mb[, 30:518]
correlations = cor(genes, mb$overall_survival)

plot3 <- ggplot(as.data.frame(correlations), aes(correlations)) + 
  geom_histogram(bins = 30, fill = "darkgreen", color = "white", size = 0.1) + 
  labs(x = "Correlation with survival", y = "Number of genes")

mutgenes_int <- sapply(mutgenes, function(x) as.integer(factor(x)))
correlations_mut = cor(mutgenes_int, mb$overall_survival)

plot4 <- ggplot(as.data.frame(correlations_mut), aes(correlations_mut)) + 
  geom_histogram(bins = 30, fill = "darkgreen", color = "white", size = 0.1) + 
  labs(x = "Correlation with survival", y = "Number of mutated genes")

grid.arrange(plot3, plot4, ncol = 2)

############# NA FILLING #############

mean_neoplasm_histologic_grade <- mean(mb$neoplasm_histologic_grade, na.rm = TRUE)
mb$neoplasm_histologic_grade <- ifelse(is.na(mb$neoplasm_histologic_grade), mean_neoplasm_histologic_grade, mb$neoplasm_histologic_grade)

mean_mutation_count <- mean(mb$mutation_count, na.rm = TRUE)
mb$mutation_count <- ifelse(is.na(mb$mutation_count), mean_mutation_count, mb$mutation_count)

mean_tumor_size <- mean(mb$tumor_size, na.rm = TRUE)
mb$tumor_size <- ifelse(is.na(mb$tumor_size), mean_tumor_size, mb$tumor_size)

mean_tumor_stage <- mean(mb$tumor_stage, na.rm = TRUE)
mb$tumor_stage <- ifelse(is.na(mb$tumor_stage), mean_tumor_stage, mb$tumor_stage)

############ FACTORIZATION ############

char_vars <- sapply(mb, is.character)
mb[, char_vars] <- lapply(mb[, char_vars], as.factor)

# Conversion one-to-one of character columns into integer ones (instead of into factors)
# Slower and yields worse accuracy but also fewer non-zero parameters
# mb <- as.data.frame(lapply(mb, function(x) if (is.character(x)) as.integer(factor(x)) else x)) # slower and yields worse accuracy but also fewer non-zero parameters
# dim(mb) 

########### TRAIN & TEST SPLIT (CLASSIFICATION) ############

# Task: Predict the "overall_survival" binary var
predictors <- mb[, -which(names(mb)=="overall_survival")]
response <- mb$overall_survival

# Decide how to handle the "death_from_cancer" var ("Died of Disease", "Died of Other Causes", "Living")
# Here I just remove it
predictors <- predictors[, -which(names(predictors)=="death_from_cancer")]

set.seed(123)  # Set seed for reproducibility
train_indices <- sample(1:nrow(predictors), 0.8 * nrow(predictors))

# Sparse matrices handlers to speed up models training
train_data <- sparse.model.matrix(~., data = predictors[train_indices, ])
test_data <- sparse.model.matrix(~., data = predictors[-train_indices, ])

train_response <- response[train_indices]
test_response <- response[-train_indices]

########### SIMPLE LASSO ############

library(glmnet)

fit <- glmnet(x = train_data, y = train_response, nfolds=5, family = "binomial", alpha = 1)
cv_fit <- cv.glmnet(x = train_data, y = train_response, nfolds=5, family = "binomial", alpha = 1)
plot(cv_fit)

best_lambda <- cv_fit$lambda.min
coef.est<- as.matrix(coef(fit, s=best_lambda))
colnames(mb)[1:29] # clinical variables
rownames(subset(coef.est, coef.est !=0))[1:29]

predictions <- predict(cv_fit, newx = test_data, type="class", s=best_lambda)
accuracy <- mean(predictions==test_response)
print(accuracy) # 0.7637795

########### RELAXED LASSO ############ (BONUS)

fit_relaxed <- glmnet(x = train_data, y = train_response, nfolds=5, family = "binomial", alpha = 1, relax = TRUE)
cv_fit_relaxed <- cv.glmnet(x = train_data, y = train_response, nfolds=5, family = "binomial", alpha = 1, relax = TRUE, parallel = TRUE)
plot(cv_fit_relaxed)

cv_fit_relaxed$relaxed$nzero.min # 499 - gamma = 1 => Vanilla Lasso
cv_fit_relaxed$relaxed$nzero.1se # 10

best_lambda_relaxed <- cv_fit_relaxed$relaxed$lambda.1se
best_gamma_relaxed <- cv_fit_relaxed$relaxed$gamma.1se

coef.est_relaxed_1se <- as.matrix(coef(fit_relaxed, s = best_lambda_relaxed, gamma = best_gamma_relaxed))
onese_tencoeff <- rownames(subset(coef.est_relaxed_1se, coef.est_relaxed_1se != 0))[2:11]

predictions_relaxed <- predict(cv_fit_relaxed, newx = test_data, type = "class", s = best_lambda_relaxed, gamma=best_gamma_relaxed)
accuracy_relaxed <- mean(predictions_relaxed == test_response)
print(accuracy_relaxed) # 0.7559055

res1 <- as.data.frame(c(499,10,0.7559055), row.names = c("Non-zero coeff min", "Non-zero coeff 1SE", "Accuracy (1SE)"))
res2 <- data.frame("1SE non-zero coefficients:" = onese_tencoeff)

library(xtable)
print(xtable(res1, type = "latex"), file = "filename1.tex")
print(xtable(res2, type = "latex"), file = "filename2.tex")
