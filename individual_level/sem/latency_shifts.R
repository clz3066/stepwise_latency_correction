rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/latency_shifts_try/real_mean_latency.xlsx"
data <- read_excel(excel_file_path, sheet=1)
data[, c(1:2)] <- apply(data[, c(1:2)],  2, scale)

latency_shifts_Facc <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
    
          # regressions
          FPA ~ Facc_l
          FMA ~ Facc_l
          face_speed ~ Facc_l '


latency_shifts_Fsp <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          
          # regressions
          FPA ~ Fsp_l
          FMA ~ Fsp_l
          face_speed ~ Fsp_l '


fit <- sem(latency_shifts_Facc, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
# print(summary(fit, fit.measures = TRUE, standardized = TRUE))
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results)
print(results[c(10, 11, 12), c(1,2,3,4,7)])

fit <- sem(latency_shifts_Fsp, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results[c(10, 11, 12), c(1,2,3,4,7)])


