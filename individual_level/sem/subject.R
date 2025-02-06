rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(bruceR)
library(semTools)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/subject_Facc_c.xlsx"

data <- read_excel(excel_file_path, sheet=1)
# data[, c(2:7)] <- apply(data[, c(2:7)],  2, scale)


step0 <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step0_0 + step0_1 + step0_2
          
          # regressions
          FPA ~ step
          FMA ~ step
          face_speed ~ step '


step1 <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step1_0 + step1_1 + step1_2
          
          # regressions
          FPA ~ step
          FMA ~ step
          face_speed ~ step '


step2 <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step2_0 + step2_1 + step2_2
          
          # regressions
          FPA ~ step
          FMA ~ step
          face_speed ~ step '


step3 <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step3_0 + step3_1 + step3_2
          
          # regressions
          FPA ~ step
          FMA ~ step
          face_speed ~ step '


step4 <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step4_0 + step4_1 + step4_2
          
          # regressions
          FPA ~ step
          FMA ~ step
          face_speed ~ step '


fit0 <- sem(step0, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step0")
lavaan_summary(fit0)

fit1 <- sem(step1, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step1")
lavaan_summary(fit1)

fit <- sem(step2, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step2")
lavaan_summary(fit)

fit <- sem(step3, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step3")
lavaan_summary(fit)

fit <- sem(step4, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step4")
lavaan_summary(fit)