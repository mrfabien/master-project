library(readr)

setwd("/Users/fabienaugsburger/Documents/GitHub/master-project/case_study")

mean <- read.csv2("correlation_matrix_mean.csv",  header = TRUE, sep = ",")
sd <- read.csv2("correlation_matrix_sigma.csv",  header = TRUE, sep = ",")

mean <- as.matrix(mean, )
sd <- as.matrix(sd)