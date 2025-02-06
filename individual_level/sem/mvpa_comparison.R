rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(bruceR)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/mvpa_Facc_sc.xlsx"
data <- read_excel(excel_file_path, sheet=1)
# data[, c(2:19)] <- apply(data[, c(2:19)],  2, scale)


######## t1 ############
model <- ' 
            # latent variable definitions
            acc_memory =~ NA*macfb_A + mdrf_A + mewf_A   
            speed =~ NA*Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
            prediction_acc =~ t10s + t11s + t12s + t10c + t11c + t12c 
            
            # variance and covariance
            acc_memory ~~ 1*acc_memory
            speed ~~ 1*speed

            # regressions
            acc_memory ~ prediction_acc
            speed ~ prediction_acc '

fit <- sem(model, data = data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("t1")
lavaan_summary(fit)

######## t2 ############
model <- ' 
            # latent variable definitions
            acc_memory =~ NA*macfb_A + mdrf_A + mewf_A   
            speed =~ NA*Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
            prediction_acc =~ t20s + t21s + t22s + t20c + t21c + t22c
            
            # variance and covariance
            acc_memory ~~ 1*acc_memory
            speed ~~ 1*speed

            # regressions
            acc_memory ~ prediction_acc
            speed ~ prediction_acc '

fit <- sem(model, data = data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("t2")
lavaan_summary(fit)

######## t3 ############
model <- ' 
            # latent variable definitions
            acc_memory =~ NA*macfb_A + mdrf_A + mewf_A   
            speed =~ NA*Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
            prediction_acc =~ t30s + t31s + t32s + t30c + t31c + t32c
            
            # variance and covariance
            acc_memory ~~ 1*acc_memory
            speed ~~ 1*speed

            # regressions
            acc_memory ~ prediction_acc
            speed ~ prediction_acc '
fit <- sem(model, data = data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("t3")
lavaan_summary(fit)
