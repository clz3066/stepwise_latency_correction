rm(list=ls())
library(readxl)
library(lavaan)
library(regsem)
library(semPlot)
library(bruceR)

excel_file_path <- "~/Documents/GitHub/latency_variability/results/individual_sem/subjects_combined.xlsx"
data <- read_excel(excel_file_path, sheet=1)

combined_step0 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          a_step =~ step0_0a + step0_1a + step0_2a
          s_step =~ step0_0s + step0_1s + step0_2s
          
          # variance
          a_step ~~ s_step
          FMA ~~ face_speed
          
          # regressions
          FMA ~ a_step + s_step
          face_speed ~ a_step + s_step '


combined_step1 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          a_step =~ step1_0a + step1_1a + step1_2a
          s_step =~ step1_0s + step1_1s + step1_2s
          
          # variance
          a_step ~~ s_step
          FMA ~~ face_speed
          
          # regressions
          FMA ~ a_step + s_step
          face_speed ~ a_step + s_step '


combined_step2 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          a_step =~ step2_0a + step2_1a + step2_2a
          s_step =~ step2_0s + step2_1s + step2_2s
          
          # variance
          a_step ~~ s_step
          FMA ~~ face_speed
          
          # regressions
          FMA ~ a_step + s_step
          face_speed ~ a_step + s_step '


combined_step3 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          a_step =~ step3_0a + step3_1a + step3_2a
          s_step =~ step3_0s + step3_1s + step3_2s
          
          # variance
          a_step ~~ s_step
          FMA ~~ face_speed
          
          # regressions
          FMA ~ a_step + s_step
          face_speed ~ a_step + s_step '


combined_step4 <- ' 
          # latent variable definitions
          FMA =~ macfb_A + mdrf_A + mewf_A    
          face_speed =~ Imdnmf_L + Imrsfb_L + Imverf_L + Immf_L
          a_step =~ step4_0a + step4_1a + step4_2a
          s_step =~ step4_0s + step4_1s + step4_2s
          
          # variance
          a_step ~~ s_step
          FMA ~~ face_speed
          
          # regressions
          FMA ~ a_step + s_step
          face_speed ~ a_step + s_step '

fit <- sem(combined_step0, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step0")
lavaan_summary(fit)

fit <- sem(combined_step1, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step1")
lavaan_summary(fit)

fit <- sem(combined_step2, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step2")
lavaan_summary(fit)

fit <- sem(combined_step3, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step3")
lavaan_summary(fit)

fit <- sem(combined_step4, data=data, estimator="ML", missing="ml", std.lv=1, meanstructure=TRUE)
print("step4")
lavaan_summary(fit)