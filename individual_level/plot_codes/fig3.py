import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import statsmodels.stats.api as sms
import mne
from matplotlib import cm
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, ttest_rel

set = './behaviors/ERPs/info.set'
info = mne.read_epochs_eeglab(set).info

data_path                           = '../../Dataset/Facc_Fsp/'
mm                                  = 1/25.4 # inch 和 毫米的转换
plt.rcParams['svg.fonttype']        = 'none'
plt.rcParams['font.family']         = 'sans-serif'   
plt.rcParams['font.sans-serif']     = 'Arial' 
plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'


def a_save_general_acc():

    excel_file = './results/individual_sem/subjects_combined.xlsx'

    Facc_df, Fsp_df = {}, {}
    for step_idx in range(5):
        df = pd.read_excel(excel_file)

        facc_scores = (df['step{}_0a'.format(step_idx)] + df['step{}_1a'.format(step_idx)] + df['step{}_2a'.format(step_idx)])/3
        Facc_df['subjects'] = df['subjects']
        Facc_df['step{}'.format(step_idx)] = facc_scores
        
        fsp_scores = (df['step{}_0s'.format(step_idx)] + df['step{}_1s'.format(step_idx)] + df['step{}_2s'.format(step_idx)])/3
        Fsp_df['subjects'] = df['subjects']
        Fsp_df['step{}'.format(step_idx)] = fsp_scores
        

    Facc_df = pd.DataFrame(Facc_df)
    Fsp_df = pd.DataFrame(Fsp_df)

    output_file = './results/individual_classification/general_scores_combined.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        Facc_df.to_excel(writer, index=False, sheet_name='Facc')
        Fsp_df.to_excel(writer, index=False, sheet_name='Fsp')


def c_plot_general_acc_distribution():
    fig, ax = plt.subplots(figsize=(80*mm, 50*mm))

    input_file = './results/individual_classification/general_scores.xlsx'

    Facc_df = pd.read_excel(input_file, sheet_name="Facc")
    Fsp_df = pd.read_excel(input_file, sheet_name="Fsp")
    columns_to_plot = ['Original', 'Step1', 'Step2', 'Step3', 'Step4', 'S Component'] 

    data = []
    for column in columns_to_plot:
        data.append(pd.DataFrame({
            'Value': Fsp_df[column],
            'Column': column,
            'Sheet': 'Fsp'
        }))
    for column in columns_to_plot:
        data.append(pd.DataFrame({
            'Value': Facc_df[column],
            'Column': column,
            'Sheet': 'Facc'
        }))

    df = pd.concat(data)

    # sns.violinplot(x='Column', y='Value', hue='Sheet', color='#a7413c', 
    #             data=df, split=True, inner='quartile')
    # sns.violinplot(data=df, x="class", y="age", hue="alive", split=True, gap=.1, inner="quart")
    fig = sns.boxplot(x='Column', y='Value', hue='Sheet', color='#a52a2a88', fill=True, gap=.2, fliersize=3, width=0.8,
                        data=df, ax=ax)
    plt.ylim(0.2, 1)
    plt.ylabel('classification accuracy', fontsize=8)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=6)

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    y_major_locator = MultipleLocator(0.2)
    plt.gca().yaxis.set_major_locator(y_major_locator)
    # plt.show()
    plt.savefig('./individual_level/paper_figures/fig3/c_subject_acc_distribution_separate.svg', bbox_inches='tight', transparent=True)
    plt.close()
# c_plot_general_acc_distribution()



def b_plot_reaction_acc():

    input_file = './results/individual_behaviors/reaction_time.xlsx'
    df_sheet1 = pd.read_excel(input_file, sheet_name="Sheet1")
    fig, ax = plt.subplots(figsize=(80*mm, 50*mm))
    
    x1 = df_sheet1['Fsp_PF'] - df_sheet1['Fsp_UPF']
    # x1 = df_sheet1['Fsp_variability']
    y1 = df_sheet1['Fsp_step0']
    r1, p1 = stats.spearmanr(x1, y1)
    txt1 = r'easy task: $\it{r}$ : %.2f $^{***}$' % (r1) 
    sns.regplot(x=x1, y=y1, ax=ax, robust=False, x_estimator=np.mean, truncate=True, scatter_kws={"color": "#d9dde7", "s": 8}, line_kws={"color": "#334c81"}, label=txt1)

    x2  = df_sheet1['Facc_PF'] - df_sheet1['Facc_UPF']
    # x2 = df_sheet1['Facc_variability']
    y2 = df_sheet1['Facc_step0']
    r2, p2 = stats.spearmanr(x2, y2)
    txt2 = r'difficult task: $\it{r}$ : %.2f $^{***}$' % (r2) 
    sns.regplot(x=x2, y=y2, ax=ax, robust=False, x_estimator=np.mean, truncate=True, scatter_kws={"color": "#eedad9", "s": 8}, line_kws={"color": "#a7413c"}, label=txt2)

    ax.set_xlim(-480, 105)
    ax.set_ylim(0.2, 1.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_xlabel('(primed - unprimed) reaction time difference (ms)', fontsize=8)
    ax.set_ylabel('classification accuracy', fontsize=8)
    ax.set_axisbelow(True)
    plt.legend(loc='upper right', fontsize=8)
    plt.show() 
    fig.savefig('./individual_level/paper_figures/fig3/b_rt_acc.svg', bbox_inches='tight', transparent=True)
    plt.close()
    

    # stat, p = stats.normaltest(df_sheet1['Fsp_PF'] - df_sheet1['Fsp_UPF'])
    # stat, p = stats.normaltest(df_sheet1['Facc_step0'])
    # print(p)
    # if p > 0.05:
    #     print('Sample looks Gaussian (fail to reject H0)')
    # else:
    #     print('Sample does not look Gaussian (reject H0)')
    # difficult task does not follow normal distribution, but easy task follows.
b_plot_reaction_acc()


def c_plot_general_acc_distribution():
    input_file = './results/individual_classification/general_scores.xlsx'

    Facc_df = pd.read_excel(input_file, sheet_name="Facc")
    Fsp_df = pd.read_excel(input_file, sheet_name="Fsp")
    columns_to_plot = ['Original', 'Step1', 'Step2', 'Step3', 'Step4', 'S Component'] 

    for idx, name in enumerate(columns_to_plot):
        Facc, Fsp = Facc_df[name], Fsp_df[name]
        Facc_further, Fsp_further = Facc_df[columns_to_plot[idx+1]], Fsp_df[columns_to_plot[idx+1]]

        t_test, p_value = ttest_rel(Facc_further, Facc)
        print(name, 'Facc', t_test, p_value)
        t_test, p_value = ttest_rel(Fsp_further, Fsp)
        print(name, 'Fsp', t_test, p_value)
        t_test, p_value = ttest_rel(Facc_further + Fsp_further,  Facc + Fsp)
        print(name, 'averaged', t_test, p_value)
        # t_test, p_value = ttest_rel(Facc, Fsp)
        # print(name, 'steps', t_test, p_value)




# def a_plot_general_acc():
#     input_file = './results/individual_classification/general_scores_combined.xlsx'

#     Facc_df = pd.read_excel(input_file, sheet_name="Facc")
#     Fsp_df = pd.read_excel(input_file, sheet_name="Fsp")
#     columns_to_plot = ['Original', 'Step1', 'Step2', 'Step3', 'Step4', 'S', 'R'] 
#     fig, ax = plt.subplots(figsize=(90*mm, 50*mm))

#     data = []
#     for column in columns_to_plot:
#         data.append(pd.DataFrame({
#             'Value': Fsp_df[column],
#             'Column': column,
#             'Sheet': 'Fsp'
#         }))
#     for column in columns_to_plot:
#         data.append(pd.DataFrame({
#             'Value': Facc_df[column],
#             'Column': column,
#             'Sheet': 'Facc'
#         }))

#     plot_data = pd.concat(data)

#     sns.barplot(data=plot_data, x='Column', y='Value', hue='Sheet', errorbar=('ci', 95), capsize=0.2, color='#a7413c', ax=ax)
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#     y_major_locator = MultipleLocator(0.1)
#     ax.yaxis.set_major_locator(y_major_locator)
#     ax.tick_params(axis='x', labelsize=8)
#     ax.tick_params(axis='y', labelsize=6)
#     ax.set_ylim(0.55, 0.85)
#     ax.set_ylabel('classification accuracy', fontsize=8)
    
#     plt.show()
#     fig.savefig('./individual_level/paper_figures/fig3/b_subject_acc_combined.svg', bbox_inches='tight', transparent=True)
#     plt.close()
