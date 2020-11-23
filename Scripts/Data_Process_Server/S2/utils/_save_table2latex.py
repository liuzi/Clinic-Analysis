import re 


def template1():
    return r'''\documentclass[preview]{{standalone}}
\usepackage{{booktabs}}
\begin{{document}}
%s
\end{{document}}'''

def template2():
    return r'''\documentclass{article}
\usepackage{amsfonts}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage[letterpaper,margin=0.5in]{geometry}
\usepackage{titlesec}
\titleformat*{\section}{\fontsize{12}{12}\bfseries}
\titleformat*{\subsection}{\fontsize{10}{10}\bfseries}
\begin{document}
%s
\end{document}'''


def each_drug_latex(dfs,drug_vars):
    section_drug_template=r'''\section{Drug rxnorm=%s, drug name=%s}'''
    sub_unique_section=r'''\subsection{Top 10 new diseases that are unique to each cluster}
-'''
    sub_section=r'''\subsection{Top 10 new diseases}
-'''

    section_drug=section_drug_template%(tuple(drug_vars[:2]))

    dfs_latex=list(map(lambda df:df.to_latex(index=False),dfs))

    dfs_latex_sum=('\n\n').join(
        [section_drug,sub_unique_section]+dfs_latex[:2]+[sub_section]+dfs_latex[2:])
    return dfs_latex_sum




# def plot_df_as_table_v2(dfs,save_path):
#     # plt.subplots(nrows,2,sharey=True)
#     filename = join(save_path,'out.tex')

#     template = \
#         r'''\documentclass[preview]{{standalone}}
# \usepackage{{booktabs}}
# \begin{{document}}
# %s
# \end{{document}}'''
#     print(template%("{}\n"*3))


#     df=dfs[0]
#     print(df.to_latex(index=False))  
#     table_latex=template.format(
#         *list(
#             map(lambda df:df.to_latex(index=False),dfs)
#         )
#     )
#     with open(filename, 'wb') as f:
#         f.write(bytes(table_latex,'UTF-8'))
    
#     subprocess.call(['pdflatex', filename])
#     subprocess.call(['convert', '-density', '300', pdffile, '-quality', '90', outname])

#     fig,axs = plt.subplots(nrows=1,ncols=2) # no visible frame
#     for (ax,df) in zip(axs,dfs):
#         ax.xaxis.set_visible(False)  # hide the x axis
#         ax.yaxis.set_visible(False)  # hide the y axis
#         plotting.table(ax, df)  # where df is your data frame

#     plt.savefig(join(save_path,'mytable.png'))


def save_df_as_latextable(rxnorm_dflists,save_path):
    # plt.subplots(nrows,2,sharey=True)
    # filename = join(save_path,'out.tex')

    template=template2()
    drug_latex_list=[]

    for (rxnorm,drug_name,dfs) in rxnorm_dflists:
        # print(dfs)
        drug_vars=[
            rxnorm,drug_name]
            # ("\"{}\"").format(rxnorm), 
            # ("\"{}\"").format(drug_name)]
        each_drug_latex(dfs,drug_vars)
        drug_latex_list.append(
            each_drug_latex(dfs,drug_vars))

    drug_latex=template%(('\n\n').join(drug_latex_list))
    with open(save_path, 'wb') as f:
        f.write(bytes(drug_latex,'UTF-8'))

    # print(template%("{}\n"*3))
    # print(section_drug.format(*section_vars))

    # print(
    #     ('\n\n').join(drug_latex_list)
    #     )


    # df=dfs[0]
    # print(df.to_latex(index=False))  
    # table_latex=template.format(
    #     *list(
    #         map(lambda df:df.to_latex(index=False),dfs)
    #     )
    # )
    # with open(filename, 'wb') as f:
    #     f.write(bytes(table_latex,'UTF-8'))
    
    # subprocess.call(['pdflatex', filename])
    # subprocess.call(['convert', '-density', '300', pdffile, '-quality', '90', outname])

    # fig,axs = plt.subplots(nrows=1,ncols=2) # no visible frame
    # for (ax,df) in zip(axs,dfs):
    #     ax.xaxis.set_visible(False)  # hide the x axis
    #     ax.yaxis.set_visible(False)  # hide the y axis
    #     plotting.table(ax, df)  # where df is your data frame

    # plt.savefig(join(save_path,'mytable.png'))


# if __name__ == '__main__':
#     plot_df_as_table([],"",["12341","dfge","that are uiqune to each cluster"])
    # print((*["12341","dfge","that are uiqune to each cluster"]))