rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(bruceR)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/mvpa_Fsp_s_component.xlsx"
data <- read_excel(excel_file_path, sheet=1)
# data[, c(2:19)] <- apply(data[, c(2:19)],  2, scale)

######## t1 ############
model <- ' 
            # latent variable definitions
            FPA =~ msmfi_A + msmfu_A
            FMA =~ macfb_A + mdrf_A + mewf_A   
            speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
            prediction_acc =~ t10 + t11 + t12
            
            FPA ~ prediction_acc
            FMA ~ prediction_acc
            speed ~ prediction_acc '

fit <- sem(model, data = data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("t1")
lavaan_summary(fit)

######## t2 ############
model <- ' 
            # latent variable definitions
            FPA =~ msmfi_A + msmfu_A
            FMA =~ macfb_A + mdrf_A + mewf_A   
            speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
            prediction_acc =~ t20 + t21 + t22
            
            FPA ~ prediction_acc
            FMA ~ prediction_acc
            speed ~ prediction_acc '
fit <- sem(model, data = data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("t2")
lavaan_summary(fit)

######## t3 ############
model <- ' 
            # latent variable definitions
            FPA =~ msmfi_A + msmfu_A
            FMA =~ macfb_A + mdrf_A + mewf_A   
            speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
            prediction_acc =~ t30 + t31 + t32
            
            FPA ~ prediction_acc
            FMA ~ prediction_acc
            speed ~ prediction_acc '
fit <- sem(model, data = data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("t3")
lavaan_summary(fit)
