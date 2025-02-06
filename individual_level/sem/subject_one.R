rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)

excel_file_path <- "~/Documents/GitHub/latency_variability/individual_level/classify/subject_Fsp.xlsx"

data <- read_excel(excel_file_path, sheet=1)
# data[, c(2:7)] <- apply(data[, c(2:7)],  2, scale)


step0 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step0_0 
          
          # regressions
          FMA ~ step
          face_speed ~ step '


step1 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step1_0 
          
          # regressions
          FMA ~ step
          face_speed ~ step '


step2 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step2_0 
          
          # regressions
          FMA ~ step
          face_speed ~ step '


step3 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step3_0 
          
          # regressions
          FMA ~ step
          face_speed ~ step '


step4 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          step =~ step4_0 
          
          # regressions
          FMA ~ step
          face_speed ~ step '


fit <- sem(step0, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print(results)
print("step0")
print(results[c(9, 10), c(1,2,3,4,7)])

fit <- sem(step1, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print("step1")
print(results)
print(results[c(9, 10), c(1,2,3,4,7)])

fit <- sem(step2, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print("step2")
print(results[c(9, 10), c(1,2,3,4,7)])

fit <- sem(step3, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print("step3")
print(results[c(9, 10), c(1,2,3,4,7)])

fit <- sem(step4, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
semPaths(fit, what='std', nCharNodes=0, fade=FALSE, residuals=TRUE, intercepts=FALSE)
results <- standardizedSolution(fit)
print("step4")
print(results[c(9, 10), c(1,2,3,4,7)])
