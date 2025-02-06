rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(bruceR)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/subject_Facc_c.xlsx"
data <- read_excel(excel_file_path, sheet=1)


step01 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step0_0 + step0_1 + step0_2 + step1_0 + step1_1 + step1_2
          
          # regressions
          FMA ~ step
          face_speed ~ step '


step12 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step1_0 + step1_1 + step1_2 + step2_0 + step2_1 + step2_2
          
          # regressions
          FMA ~ step
          face_speed ~ step '


step23 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step2_0 + step2_1 + step2_2 + step3_0 + step3_1 + step3_2
          
          # regressions
          FMA ~ step
          face_speed ~ step '


step34 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step3_0 + step3_1 + step3_2 + step4_0 + step4_1 + step4_2
          
          # regressions
          FMA ~ step
          face_speed ~ step '


fit0 <- sem(step01, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step01")
lavaan_summary(fit0)

fit1 <- sem(step12, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step12")
lavaan_summary(fit1)

fit <- sem(step23, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step23")
lavaan_summary(fit)

fit <- sem(step34, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step34")
lavaan_summary(fit)
