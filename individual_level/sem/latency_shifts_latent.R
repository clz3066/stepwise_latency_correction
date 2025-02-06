rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/latency_shifts_try/lag_erp.xlsx"
data <- read_excel(excel_file_path, sheet=1)

latency_shifts_Facc <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          
          FMA ~~ face_speed
          
          # regressions
          FMA ~ Facc_l
          face_speed ~ Facc_l '


latency_shifts_Fsp <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          
          FMA ~~ face_speed
          
          # regressions
          FMA ~ Fsp_l
          face_speed ~ Fsp_l '


fit <- sem(latency_shifts_Facc, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
# print(summary(fit, fit.measures = TRUE, standardized = TRUE))
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results[c(9, 10), c(1,2,3,4,7)])

fit <- sem(latency_shifts_Fsp, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results[c(9, 10), c(1,2,3,4,7)])



