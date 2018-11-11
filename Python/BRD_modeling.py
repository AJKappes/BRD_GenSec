import numpy as np
import pandas as pd
import sympy as sym
import plotly.plotly as plt
import plotly
import plotly.io as pio
import plotly.graph_objs as go
import glob

### Parameter evaluation for theoretical model results ###
##########################################################


# cost functions based on calf type
def c_ms(r, q, t, alpha):
    return r * q * np.log(t + alpha)

def c_n(r, q, theta_h, theta_l, eta, t, alpha):
    return eta * r * q * theta_h * np.power(t, alpha) +\
           (1 - eta) * r * q * theta_l * np.power(t, alpha)

# parameters
r = 1
q = 1
theta = [.01, .5/5, 3/5, 5/5]
eta = [0.5, 0.9, 1]
alpha = 1.5
t = np.linspace(0, 31, 1000)

cost_ms = c_ms(r, q, t, alpha)

cost_n = [c_n(r, q, theta[1], theta[0], eta[0], t, alpha)]
i = 2
while i < len(theta):
    cost_n.append(c_n(r, q, theta[i], theta[0], eta[i-1], t, alpha))
    i += 1
    if i > len(theta):
        break

# cost mapping
trace_ms = go.Scatter(
    x=t,
    y=cost_ms,
    mode='lines',
    name='C[q_MS, r, t]'
)

trace_thetap1 = go.Scatter(
    x=t,
    y=cost_n[0],
    mode='lines',
    name='C[q_N, r, t] | eta = 0.5: morb scale 0'
)

trace_thetap6 = go.Scatter(
    x=t,
    y=cost_n[1],
    mode='lines',
    name='C[q_N, r, t] | eta = 0.9: morb scale 0.6'
)

trace_theta1 = go.Scatter(
    x=t,
    y=cost_n[2],
    mode='lines',
    name='C[q_N, r, t] | eta = 1: morb scale 1'
)

layout = go.Layout(yaxis=dict(range=[0, 10],
                              # showticklabels=False,
                              title='C[q_i, r, t]',
                              showgrid=False),
                   xaxis=dict(title='t',
                              showgrid=False),
                   font=dict(family='Liberation Serif'))
mapping = [trace_ms, trace_thetap1, trace_thetap6, trace_theta1]
figure = go.Figure(data=mapping, layout=layout)
plotly.offline.plot(figure, filename='cost_mapping.html')
pio.write_image(figure, '/home/akappes/Research/Agri/BRD_GenSec/LaTex/cost_mapping.pdf')

# integrate for costs over time
t = sym.Symbol('t')
cost_n_vals = [sym.integrate(c_n(r, q, theta[1], theta[0], eta[0], t, alpha), (t, 0, 26))]
i = 2
while i < len(theta):
    cost_n_vals.append(sym.integrate(c_n(r, q, theta[i], theta[0], eta[i-1], t, alpha), (t, 0, 26)))
    i += 1
    if i > len(theta):
        break

# https://www.ams.usda.gov/mnreports/lm_ct180.txt
# average dressed steer/heifer weight and steer/heifer dressed prices - all grades Choice
p_feeder = 144
feeder_weight = 800
p_dressed = np.mean([172.99, 173.26])
weight_dressed = np.array([775, 900, 1050])
cwt = 100

theory_rev = p_dressed * weight_dressed / cwt
theory_prof = [theory_rev[0] - cost_n_vals - p_feeder * feeder_weight / cwt]
i = 1
while i < len(theory_rev):
    theory_prof.append(theory_rev[i] - cost_n_vals - p_feeder * feeder_weight / cwt)
    i += 1
    if i > len(theory_rev):
        break
theory_prof = np.concatenate((theory_prof[0], theory_prof[1], theory_prof[2])).reshape(3,3)

weight_index = pd.DataFrame(np.zeros(len(weight_dressed)))
for i in range(len(weight_dressed)):
    weight_index.loc[i] = str(weight_dressed[i]) + 'lbs'

theory_prof_df = pd.DataFrame({'Morb_scale_0': theory_prof[:, 0],
                               'Morb_scale_3': theory_prof[:, 1],
                               'Morb_scale_5': theory_prof[:, 2]
                               }, index=weight_index).astype('float').round(2)

# comparison to genetic marker selected cost
cost_ms_val = sym.integrate(r * q * sym.log(t + alpha), (t, 0, 26)) # sympy integrate does not recognize np.log
theory_prof_ms = theory_rev - cost_ms_val

# combined calf type cost functions
q_ms = [2, 5, 8]
q_n = [2, 5, 8]

def c_ms_n(r, q_ms, q_n, theta, t):
    return r * np.divide(q_n, q_ms) * (np.log(theta * t + alpha) + theta * np.power(t, alpha))

eqprop_c = [c_ms_n(r, q_ms[1], q_n[1], theta[0], t)]
i = 1
while i < len(theta):
    eqprop_c.append(c_ms_n(r, q_ms[1], q_n[1], theta[i], t))
    i += 1
    if i > len(theta):
        break

msprop_c = [c_ms_n(r, q_ms[2], q_n[1], theta[0], t)]
i = 1
while i < len(theta):
    msprop_c.append(c_ms_n(r, q_ms[2], q_n[1], theta[i], t))
    i += 1
    if i > len(theta):
        break

nprop_c = [c_ms_n(r, q_ms[1], q_n[2], theta[0], t)]
i = 1
while i < len(theta):
    nprop_c.append(c_ms_n(r, q_ms[1], q_n[2], theta[i], t))
    i += 1
    if i > len(theta):
        break
# need to figure out if theoretical proportional costs are worth exploring

### Emprical analysis ###
#########################

def get_files(ext):
    return glob.glob('/home/akappes/Research/Agri/BRD_GenSec/' + ext)

csv_list = get_files('*.csv')
for i in range(len(csv_list)):
    print(i, csv_list[i])

df_dp = pd.read_csv(csv_list[0])
df_idsex = pd.read_csv(csv_list[1])
df_fp = pd.read_csv(csv_list[2])
df_brd = pd.read_csv(csv_list[3])
df_brd = pd.merge(df_brd, df_idsex, how='inner', on='ear_id')
#df_brd['sex'].unique() # reveals only steers, no heifers in trial


# convert hot weight to live weight using dressing percentage of 0.625
# typical dressing percentage is 0.6 to 0.65
dress_perc = 0.625
df_brd['live_weight'] = np.divide(df_brd['Hot Weight'], dress_perc)

# matching historic feeder and dressed prices based on date
# defined function is working better than pandas merge
fp_ml = df_fp['date'].unique().tolist()
for i in fp_ml:
    m_index = df_brd[df_brd['arr_date'] == i].index.tolist()
    df_brd.loc[m_index, 'feeder_price'] = np.array(df_fp[df_fp['date'] == i]['steer_p_8to9'])

dp_ml = df_dp['date'].unique().tolist()
for i in dp_ml:
    m_index = df_brd[df_brd['proc_date'] == i].index.tolist()
    df_brd.loc[m_index, 'dressed_price'] = np.array(df_dp[df_dp['date'] == i]['dress_price'])

# Draxxin treatment cost calculation
# 1.1ml/100lbs at $4.599 per ml
dosage_cost = 4.599 * 1.1
df_brd['dosage_cost'] = 0
case_list = df_brd[df_brd['ExpStatus'] == 'Case'].index.tolist()
control_list = df_brd[df_brd['ExpStatus'] == 'Control'].index.tolist()
df_brd.loc[case_list, 'dosage_cost'] = np.divide(dosage_cost * df_brd['PullWeight'], 100)

# month to integers for calculating days on feed
time_list = ['Jan-12', 'Feb-12', 'Mar-12', 'Apr-12', 'May-12', 'Jun-12',
             'Jul-12', 'Aug-12', 'Sep-12', 'Oct-12', 'Nov-12', 'Dec-12',
             'Jan-13', 'Feb-13', 'Mar-13', 'Apr-13', 'May-13', 'Jun-13',
             'Jul-13', 'Aug-13', 'Sep-13', 'Oct-13']
time_intgr = np.linspace(1, len(time_list), len(time_list))

arr_date_list = df_brd['arr_date'].unique().tolist()
proc_date_list = df_brd['proc_date'].unique().tolist()

for i in range(len(time_list)):
    arr_index = df_brd[df_brd['arr_date'] == time_list[i]].index.tolist()
    df_brd.loc[arr_index, 'arr_int'] = time_intgr[i]

for i in range(len(time_list)):
    proc_index = df_brd[df_brd['proc_date'] == time_list[i]].index.tolist()
    df_brd.loc[proc_index, 'proc_int'] = time_intgr[i]

df_brd['days_feed'] = np.multiply(df_brd['proc_int'] - df_brd['arr_int'], 30)

# including daily feed costs

# subset df for case (brd) and control
case_df = df_brd.loc[case_list]
control_df = df_brd.loc[control_list]

# morbidity scale rankings in case df
morb_5 = case_df[(case_df['SymTachypnea'] == 'Y') | (case_df['SymDyspnea'] == 'Y') &
                 (case_df['SymNasal'] == 'Excessive bilateral, mucopurulent') &
                 (case_df['SymRectTemp'] == '103 F and up')]

morb_3 = case_df[(~case_df.index.isin(morb_5.index)) & (case_df['SymNasal'] == 'Moderate bilateral, cloudy') |
                 (case_df['SymNasal'] == 'Small unilateral cloudy discharge') &
                 (case_df['SymRectTemp'] == '103 F and up')]

# lbs gain in different health rankings
def lbs_gain(df):
    df['lbs_gain'] = df['live_weight'] - 850

calf_health_dfs = [control_df, morb_3, morb_5]
for i in calf_health_dfs:
    lbs_gain(i)

# organize summary stats tables
n = len(case_df) + len(control_df)

def prop_fx(v, cat):
    return np.round(len(case_df[case_df[v] == cat]) / n, 3)

rtemp = pd.DataFrame({'Rectal Temp': np.array([prop_fx('SymRectTemp', '103 F and up'),
                                               1 - prop_fx('SymRectTemp', '103 F and up')])})

symnasal = pd.DataFrame({'Nasal Symptoms': np.array([prop_fx('SymNasal', 'Excessive bilateral, mucopurulent'),
                                                     prop_fx('SymNasal', 'Moderate bilateral, cloudy'),
                                                     prop_fx('SymNasal', 'Small unilateral cloudy discharge'),
                                                     (len(case_df[case_df['SymNasal'] == 'Normal']) +
                                                      len(control_df)) / n])})

resp = pd.DataFrame({'Respiratory Presentation': np.array([prop_fx('SymTachypnea', 'Y'),
                                                           prop_fx('SymDyspnea', 'Y')])})

m = len(rtemp) + len(symnasal) + len(resp)
def empty_df(df, dim):
    return pd.DataFrame({df.columns[0]: [''] * dim})

rtemp_df = pd.concat([rtemp, empty_df(rtemp, m - len(rtemp))]).reset_index().drop(columns='index')

symnasal_df = pd.concat([empty_df(symnasal, len(rtemp)), symnasal,
                         empty_df(symnasal, m - len(symnasal) - len(rtemp))]).reset_index().drop(columns='index')

resp_df = pd.concat([empty_df(resp, len(rtemp) + len(symnasal)), resp]).reset_index().drop(columns='index')

prop_idx = ['>= 103 F', '102-102.9 F', 'Excessive bilateral mucopurulent', 'Moderate bilateral cloudy discharge',
            'Small unilateral cloudy discharge', 'Normal', 'Tachypnic', 'Dyspnic']
prop_cat = pd.DataFrame({'Category': prop_idx})
prop_stat = pd.concat([prop_cat, rtemp_df, symnasal_df, resp_df], axis=1).set_index('Category')

morb_prop_df = pd.DataFrame({'Normal Respiratory Health': np.round(len(control_df) / n, 3),
                             'Morbidity Scale 3': np.round(len(morb_3) / n, 3),
                             'Morbidity Scale 0': np.round(len(morb_5) / n, 3)},
                            index=['Proportion'])

# additional days on feed to equate morb scale lb gain to normal resp health lb gain
def days_equiv(df):
    return (control_df['lbs_gain'].mean() - df['lbs_gain'].mean()) / np.mean(df['lbs_gain'] / df['days_feed'])

morb_3['adj_days_feed'] = morb_3['days_feed'] + days_equiv(morb_3)
morb_5['adj_days_feed'] = morb_5['days_feed'] + days_equiv(morb_5)

# dressed weight adjustment to equate normal resp health dressed weight
def weight_equiv(df):
    return df['Hot Weight'] + control_df['Hot Weight'].mean() - df['Hot Weight'].mean()

morb_3['adj_dressed_weight'] = weight_equiv(morb_3)
morb_5['adj_dressed_weight'] = weight_equiv(morb_5)

# production summary stats construction
def desc_stats(df, df_colname, data_col_name):
    return pd.DataFrame({df_colname: np.array([df[data_col_name].mean(),
                                               df[data_col_name].std(),
                                               df[data_col_name].min(),
                                               df[data_col_name].max()])})

norm_health_stats = np.round(pd.concat([desc_stats(control_df, 'Days on Feed', 'days_feed'),
                                        desc_stats(control_df, 'Dressed Weight', 'Hot Weight'),
                                        desc_stats(control_df, 'Lbs Gained', 'lbs_gain')],
                                       axis=1), 3)

morb3_health_stats = np.round(pd.concat([desc_stats(morb_3, 'Days on Feed', 'days_feed'),
                                         desc_stats(morb_3, 'Dressed Weight', 'Hot Weight'),
                                         desc_stats(morb_3, 'Lbs Gained', 'lbs_gain')],
                                        axis=1), 3)

morb5_health_stats = np.round(pd.concat([desc_stats(morb_5, 'Days on Feed', 'days_feed'),
                                         desc_stats(morb_5, 'Dressed Weight', 'Hot Weight'),
                                         desc_stats(morb_5, 'Lbs Gained', 'lbs_gain')],
                                        axis=1), 3)

stat_idx = ['Mean', 'Std. Dev', 'Min', 'Max']
stat_idx_df = pd.DataFrame({'Statistic': stat_idx * 3})
prod_stats = pd.concat([norm_health_stats, morb3_health_stats, morb5_health_stats]).reset_index()
prod_stats = pd.concat([stat_idx_df, prod_stats], axis=1).set_index('Statistic').drop(columns='index')

# prices for corn and hay computed as 2012-2013 national annual average
# prices sourced from https://www.nass.usda.gov/Quick_Stats/
# feed cost construction - 56lbs in corn bushel
corn_p = np.mean([6.15, 6.67]) / 56 # convert bushel price to per lb price
hay_p = np.mean([183, 184]) / 2000 # convert tone price to per lb price

# assume finishing stage diet ration proportions for grain and forage
# assume daily feed (lbs) consumed is 2.05% of live weight
grain_prop = 0.9
forage_prop = 0.1
perc_feed_cons = .0205

def total_feed_exp(df, days_feed):
    feed_lb_cons = df['live_weight'] * perc_feed_cons
    daily_feed_exp = feed_lb_cons * (grain_prop * corn_p + forage_prop * hay_p)
    return df[days_feed] * daily_feed_exp

total_feed_exp(control_df, 'days_feed')

for i in range(len(calf_health_dfs)):
    if i == 0:
        days_feed = 'days_feed'
    else:
        days_feed = 'adj_days_feed'

    calf_health_dfs[i]['total_feed_exp'] = total_feed_exp(calf_health_dfs[i], days_feed)

# cost and profit construction
def costs(df, indicate):
    if indicate == 'healthy':
        return df['feeder_price'] * 850 / 100 + df['total_feed_exp']
    elif indicate == 'morb':
        return df['feeder_price'] * 850 / 100 + df['total_feed_exp'] + df['dosage_cost']
    else:
        print('Specify costs as either "healthy" or "morb"')

def rev(df, indicate):
    if indicate == 'healthy':
        return df['dressed_price'] * df['Hot Weight'] / 100
    elif indicate == 'morb':
        return df['dressed_price'] * df['adj_dressed_weight'] / 100
    else:
        print('Specify calf class as either "healthy" or "morb"')

def profs(df, indicate):
    return rev(df, indicate) - costs(df, indicate)

calf_df = pd.DataFrame({'Healthy Calf': np.array([rev(control_df, 'healthy').mean(),
                                                  costs(control_df, 'healthy').mean(),
                                                  profs(control_df, 'healthy').mean()]),
                        'BRD Calf': np.array([np.mean([rev(morb_3, 'morb').mean(), rev(morb_5, 'morb').mean()]),
                                              np.mean([costs(morb_3, 'morb').mean(), costs(morb_5, 'morb').mean()]),
                                              np.mean([profs(morb_3, 'morb').mean(), profs(morb_5, 'morb').mean()])])
                        }).round(3)

calf_idx = pd.DataFrame({'calf': np.array(['Mean Per-Calf Revenue',
                                           'Mean Per-Calf Cost',
                                           'Mean Per-Calf Profit'])})

calf_df = pd.concat([calf_idx, calf_df], axis=1).set_index('calf')

# simulation to find proportion of morbidity scale cattle that result in loss
sub_control_df = control_df[['ExpStatus', 'dressed_price', 'Hot Weight', 'feeder_price',
                             'dosage_cost', 'total_feed_exp']]
sub_control_df = sub_control_df.rename(columns={'Hot Weight': 'dressed_weight'})

morb_df = pd.concat([morb_3, morb_5])
sub_morb_df = morb_df[['ExpStatus', 'dressed_price', 'adj_dressed_weight', 'feeder_price',
                       'dosage_cost', 'total_feed_exp']]
sub_morb_df = sub_morb_df.rename(columns={'adj_dressed_weight': 'dressed_weight'})

sim_df = pd.concat([sub_control_df, sub_morb_df])
sim_df = sim_df.dropna()

n = 100

sample = np.arange(0, 1000, 1).tolist()
sim_dict = {}
for i in sample:
    sim_dict[i] = sim_df.sample(n, replace=True)

# gamma probability of observing non marker selected
sim_brd_gamma = pd.DataFrame({'gamma': np.zeros(len(sample))})
for i in sample:
    sim_brd_gamma.loc[i] = len(sim_dict[i][sim_dict[i]['ExpStatus'] == 'Case']) / n

gamma = sim_brd_gamma['gamma'].mean()

def sim_prof(i, health):
    indicate = sim_dict[i][sim_dict[i]['ExpStatus'] == health]
    rev = indicate['dressed_price'] * indicate['dressed_weight'] / 100
    costs = indicate['feeder_price'] * 850 / 100 + indicate['total_feed_exp'] + indicate['dosage_cost']
    return rev - costs

repr_calf_profs = pd.DataFrame({'healthy': np.zeros(len(sample)),
                               'brd': np.zeros(len(sample))})
for i in sample:
    repr_calf_profs.loc[i, 'healthy'] = sim_prof(i, 'Control').mean()
    repr_calf_profs.loc[i, 'brd'] = sim_prof(i, 'Case').mean()

repr_calf_stats = pd.concat([desc_stats(repr_calf_profs, 'Healthy', 'healthy'),
                             desc_stats(repr_calf_profs, 'BRD', 'brd')], axis=1).round(3)

stat_idx_df = pd.DataFrame({'Statistic': stat_idx})
repr_calf_stats = pd.concat([stat_idx_df, repr_calf_stats], axis=1).set_index('Statistic')

repr_total_profs = pd.DataFrame({'healthy': np.zeros(len(sample)),
                                 'brd': np.zeros(len(sample))})

for i in sample:
    repr_total_profs.loc[i, 'healthy'] = sim_prof(i, 'Control').sum()
    repr_total_profs.loc[i, 'brd'] = sim_prof(i, 'Case').sum()

repr_total_profs['total'] = repr_total_profs['healthy'] + repr_total_profs['brd']
loss_prop = len(repr_total_profs[repr_total_profs['total'] < 0]) / len(sample)

repr_total_stats = pd.concat([desc_stats(repr_total_profs, 'Healthy', 'healthy'),
                              desc_stats(repr_total_profs, 'BRD', 'brd'),
                              desc_stats(repr_total_profs, 'Total Profits', 'total')], axis=1).round(3)

repr_total_stats = pd.concat([stat_idx_df, repr_total_stats], axis=1).set_index('Statistic')

loss_prop_df = pd.DataFrame({'Healthy': '',
                             'BRD': '',
                             'Total Profits': loss_prop}, index=['Pr(loss)'])

repr_total_stats = pd.concat([repr_total_stats, loss_prop_df])