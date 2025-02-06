rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/lags_variability.xlsx"
data <- read_excel(excel_file_path, sheet=2)

data[, c(2:9)] <- apply(data[, c(2:9)],  2, scale)


variability_Facc <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          
          # regressions
          FPA ~ Facc_variability
          FMA ~ Facc_variability
          face_speed ~ Facc_variability '


variability_Fsp <- ' 
          # latent variable definitions
          FPA =~ msmfu_A + msmfi_A
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          
          # regressions
          FPA ~ Fsp_variability
          FMA ~ Fsp_variability
          face_speed ~ Fsp_variability '


fit <- sem(variability_Facc, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results[c(10, 11, 12), c(1,2,3,4,7)])

fit <- sem(variability_Fsp, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results[c(10, 11, 12), c(1,2,3,4,7)])


