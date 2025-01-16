import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#---------- p = 05
df05 = pd.read_csv("df.csv")
df05 = df05.sort_values(by=['method'])
fig, axes = plt.subplots(figsize=(15, 15))

ax = sns.boxplot(y="OoS",x = "method", data= df05)
plt.setp(ax, xlabel='')
plt.setp(ax, ylabel='')
plt.savefig("plots_results_bike_paper_model.pdf")


"""
fig, axes = plt.subplots(3, 3,figsize=(15, 15))

ax = sns.boxplot(y="OG", hue="method", data= df05[df05["T"]==100],ax = axes[0,0],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove() 
ax.set(ylim=(0, 5))
ax = sns.boxplot(y="OG", hue="method", data= df05[df05["T"]==1000],ax = axes[0,1],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 5)) 
ax = sns.boxplot(y="OG", hue="method", data= df05[df05["T"]==10000],ax = axes[0,2],palette = ['tab:blue' , 'tab:green','tab:red', 'tab:brown'])
ax.get_legend().remove()
ax.set(ylim=(0, 5))

plt.setp(axes[0,0], ylabel='p = 0.5')
plt.setp(axes[0,1], ylabel='')
plt.setp(axes[0,2], ylabel='')

plt.setp(axes[0,0], xlabel='N = 10^2')
plt.setp(axes[0,1], xlabel='N = 10^3')
plt.setp(axes[0,2], xlabel='N = 10^4')


ax.legend(loc='upper center', bbox_to_anchor=(-0.75, -0.1), fancybox=True, shadow=True, ncol=7)

#plt.savefig("plots_results_shipment.pdf")
plt.savefig("tito_plots_results_shipment.pdf")
"""