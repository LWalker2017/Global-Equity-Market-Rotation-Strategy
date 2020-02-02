import pandas as pd 
import numpy as np 
import pickle 
import time 
import statsmodels.tsa.stattools
import statsmodels.graphics.tsaplots
import sys
import warnings
from gurobipy import * 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

DATA_PATH = '../data/'
warnings.filterwarnings('ignore')

# initialize backtest
Principal = 1000000 

# arima training days
arima_days = 30 

# model param
model_param_dict = {
    'theta': 0.5, # theta means the obj weights in alpha, i.e how importance is alpha in the obj
    'market_ub': 0.5, 
    'market_lb': 0, 
    'weight_ub': 0.05,  # buy 20 stocks at least
    'weight_lb': 0,  # buy 100 stocks at most
    'trans_cost': 0.01, 
    'bigM': 10, 
    'sector_ub': 0.3, 
    'sector_lb': 0.01
}

##### DATA PREPARATION 
stock_df = pd.read_csv(DATA_PATH + 'totalPrice.csv').rename(columns={'Unnamed: 0': 'Code'}).set_index('Code')
stock_df = stock_df.T.iloc[-arima_days-5:].T

# load in the backtest dataset
OpenPrice = pd.read_excel(DATA_PATH + 'openPrice_USD.xlsx',header=0,index_col=0)
ClosePrice = pd.read_excel(DATA_PATH + 'closePrice_USD.xlsx',header=0,index_col=0)
Benchmark = pd.read_excel(DATA_PATH + 'MSCI_World_NTR.xlsx',header=0)

# stock name arr 
with open(DATA_PATH + 'stock_name_arr.pkl', 'rb') as file:
    stock_name_arr = pickle.load(file)
    
# Covariance matrix 
with open(DATA_PATH + 'cov_mat.pkl', 'rb') as file:
    cov_mat_arr = pickle.load(file)

# market dict
with open(DATA_PATH + 'market_dict.pkl', 'rb') as file:
    market_dict = pickle.load(file)
    
# sector dict
with open(DATA_PATH + 'sector_dict.pkl', 'rb') as file:
    sector_dict = pickle.load(file) 
    
data_dict = {
    'stock_name_arr': stock_name_arr, 
    'cov_mat_arr': cov_mat_arr, 
    'market_dict': market_dict, 
    'sector_dict': sector_dict
}
del stock_name_arr 
del cov_mat_arr 
del market_dict 
del sector_dict

##### ARIMA MODEL 
#create a series for the 1-lag difference
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    statsmodels.graphics.tsaplots.plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    statsmodels.graphics.tsaplots.plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

def testStationarity(time_series):
    dftest = statsmodels.tsa.stattools.adfuller(time_series)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


def proper_model(data_ts, maxLag_p = 5, maxLag_q = 5):
    init_aic = sys.maxsize
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag_p):
        for q in np.arange(maxLag_q):
            try:
                model = statsmodels.tsa.arima_model.ARMA(data_ts, order=(p, q), freq = 'D')
                results_ARMA = model.fit(disp=-1, method='css')
                aic = results_ARMA.aic
                if aic < init_aic:
                    init_p = p
                    init_q = q
                    init_properModel = results_ARMA
                    init_aic = aic
            except:
                continue
            
    return init_properModel, init_p, init_q

def diff_to_stationary(ts):
    if(testStationarity(ts)['p-value'] <= 0.05):
        return ts,0
    else:
        ts_diff = ts.diff(1).dropna()
        num = 1
        while(testStationarity(ts_diff)['p-value'] > 0.05):
            ts_diff = ts_diff.diff(1).dropna()
            num += 1
        return ts_diff, num 
    
def arima_main(df):
    df = df.T 
    df.index = pd.to_datetime(df.index)
    cols = df.columns.to_list()
    df = df.reset_index(drop=False)
    df.rename(columns={'index':'DATE'}, inplace = True)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    def arima_predict(ric, training_days=arima_days):
        start = time.time()
        close_price = df[['DATE', ric]]
        helper = pd.DataFrame({'DATE': pd.date_range(close_price['DATE'].min(), close_price['DATE'].max())})
        close_price = pd.merge(close_price, helper, on='DATE', how='outer').sort_values('DATE')
        close_price[ric] = close_price[ric].interpolate(method='linear')   
        close_price.set_index(pd.to_datetime(close_price.DATE), inplace=True) # set the index to be the DATE
        close_price.sort_index(inplace=True)  # sort the dataframe by the newly created datetime index


        last_date = close_price.index.to_list()[-1] - timedelta(days=training_days)
        close_price = close_price[close_price.index >= last_date] 
        ts = close_price[ric]
        ts.index = pd.to_datetime(close_price.index)

        ts_diff, num_of_diff = diff_to_stationary(ts)

        inf_lst = proper_model(ts_diff)
        
        current = close_price.index.to_list()[-1]
        end_time = current
        day1 = current + timedelta(days = 1)
        day2 = current + timedelta(days = 2)
        day3 = current + timedelta(days = 3)
        day4 = current + timedelta(days = 4)
        day5 = current + timedelta(days = 5)

        try:
            predict_ts = inf_lst[0].predict(day1, day5, dynamic=True)
        except AttributeError:
            print('No appropriate model')
            return -1

        for i in range(num_of_diff):
            if(num_of_diff - i - 1 != 0):
                predict_ts[day1] = predict_ts[day1] + ts.diff(num_of_diff - i)[end_time]
            else:
                predict_ts[day1] = predict_ts[day1] + ts[end_time]
            predict_ts[day2] = predict_ts[day2] + predict_ts[day1]
            predict_ts[day3] = predict_ts[day3] + predict_ts[day2]
            predict_ts[day4] = predict_ts[day4] + predict_ts[day3]
            predict_ts[day5] = predict_ts[day5] + predict_ts[day4]
        return (predict_ts[day5] - ts[end_time]) / ts[end_time]
    
    pred_dict = {}
    for i in range(len(cols)):
        col = cols[i]
        if i % 100 == 0.:
            print(f'    The {i}th predictions ... ')
        pred_dict[col] = arima_predict(col)
        
    return pred_dict

##### OPTIMIZATION MODEL
class PortOpt:
    def __init__(self, data_dict, p_dict, init_port_=None):
        # params retrieve
        self.theta = p_dict['theta']
        self.market_ub = p_dict['market_ub']
        self.market_lb = p_dict['market_lb']
        self.weight_ub = p_dict['weight_ub']
        self.weight_lb = p_dict['weight_lb']
        self.sector_ub = p_dict['sector_ub']
        self.sector_lb = p_dict['sector_lb']
        self.trans_cost = p_dict['trans_cost']
        self.bigM = p_dict['bigM']
    
        # stock and market information retrieve
        self.stock_arr = data_dict['stock_name_arr']  # index of w
        self.stock_idx_arr = np.arange(0, len(self.stock_arr))
        
        self.alpha_dict = data_dict['alpha_dict']
        self.alpha_idx_dict = {}
        for idx in self.stock_idx_arr:
            key = self.stock_arr[idx]
            self.alpha_idx_dict[idx] = self.alpha_dict[key]
        self.cov_mat_arr = data_dict['cov_mat_arr']
        
        self.sector_dict = data_dict['sector_dict']
        self.sector_idx_dict = {}
        for key in self.sector_dict:
            arr = self.sector_dict[key]
            self.sector_idx_dict[key] = [np.where(self.stock_arr==i)[0][0] for i in arr]
            
        self.market_dict = data_dict['market_dict']
        self.market_idx_dict = {}
        for key in self.market_dict:
            arr = self.market_dict[key]
            self.market_idx_dict[key] = [np.where(self.stock_arr==i)[0][0] for i in arr]
        
        # port model 
        self.model = Model('Port Opt Model')
        self.init_port = init_port_
        self.var_dict = {} 
        self.__model_init()

    def optimize(self):
        self.model.optimize()
        
    def __model_init(self):
        self.__create_vars()
        self.__create_constrs()
        self.__create_obj()
        self.model.update()

    def __get_market_stock(self):  # TODO
        # return dict of arrays
        # keys of dict is the same as msci_dict
        pass  
        
    def __create_vars(self):
        # portfolio weights
#         self.var_dict['w'] = self.model.addVars(self.stock_idx_arr, vtype=GRB.CONTINUOUS, lb=0.0, name='w')
        self.var_dict['w'] = pd.Series(self.model.addVars(self.stock_idx_arr, name='w', lb=0.0, vtype=GRB.CONTINUOUS), index=self.stock_idx_arr)
        # portfolio change
        if self.init_port is None:
            pass 
        else:
            # portfolio change = last port - current port (only consider the sells part)
            self.var_dict['y'] = self.model.addVars(self.stock_idx_arr, vtype=GRB.CONTINUOUS, lb=0.0, name='y')

            # artificial var 
            self.var_dict['z'] = self.model.addVars(self.stock_idx_arr, vtype=GRB.BINARY, name='z')
        
        self.model.update()
        
    def __create_constrs(self):
        # 1 weights normalization
        self.model.addConstr(sum([self.var_dict['w'][i] for i in self.stock_idx_arr]) == 1, name='1_weights_normalization')
        
        # 2 limit weights per market
        for key in self.market_idx_dict:
            market_arr = self.market_idx_dict[key]
            self.model.addConstr((sum([self.var_dict['w'][i] for i in market_arr]) <= self.market_ub), name='2_1_weights_{}_market_ub'.format(key))
            self.model.addConstr((sum([self.var_dict['w'][i] for i in market_arr]) >= self.market_lb), name='2_1_weights_{}_market_lb'.format(key))
        
        # 3 limit weights per share 
        self.model.addConstrs((self.var_dict['w'][i] <= self.weight_ub for i in self.stock_idx_arr), name='3_weight_per_asset_ub')
        
        # 4 y = max{0, init_port - current port}
        if self.init_port is None:
            pass 
        else:
            self.model.addConstrs((self.init_port[i] - self.var_dict['w'][i] <= self.bigM * (1 - self.var_dict['z']) for i in self.stock_idx_arr), name='4c{}'.format(i))
            self.model.addConstrs((-self.var_dict['y'][i] + self.init_port[i] - self.var_dict['w'][i] <= self.bigM * self.var_dict['z'] for i in selff.stock_idx_arr), name='4b{}'.format(i))
            self.model.addConstrs((self.var_dict['y'][i] - self.init_port[i] + self.var_dict['w'][i] <= self.bigM * self.var_dict['z'] for i in selff.stock_idx_arr), name='4a{}'.format(i))

        # 5 limit weights per sector
        for key in self.sector_idx_dict:
            sector_arr = self.sector_idx_dict[key]
            self.model.addConstr((sum([self.var_dict['w'][i] for i in sector_arr]) <= self.sector_ub), name='5_weights_{}_sector_ub'.format(key))
            self.model.addConstr((sum([self.var_dict['w'][i] for i in sector_arr]) >= self.sector_lb), name='5_weights_{}_sector_lb'.format(key))
        
        self.model.update()

    def __create_obj(self):
        min_obj = self.cov_mat_arr.dot(self.var_dict['w']).dot(self.var_dict['w'])
        max_obj = np.array(list(data_dict['alpha_dict'].values())).dot(self.var_dict['w'])
        # transaction cost
        if self.init_port is None:
            cost = 0 
        else:
            cost = np.sum([self.var_dict['y'][i] * self.trans_cost for i in self.stock_idx_arr])

        obj = self.theta * (max_obj - cost) - (1 - self.theta) * min_obj
        
        self.model.setObjective(obj, GRB.MAXIMIZE)
        self.model.update()
    
    def get_results(self):
        temp_results_list = [self.var_dict['w'][i].x for i in self.stock_idx_arr]
        results_arr = []
        for i in temp_results_list:
            if i > 1e-10: 
                results_arr.append(i)
            else:
                results_arr.append(0)
        return self.stock_arr, np.array(results_arr)

###### OVERALL MODEL
def model_main():
    print('Start predicting the expected 5-day return ... ')
    start = time.time()
    alpha_dict = arima_main(stock_df)
    print('Prediction cost {} seconds'.format(time.time() - start))
    with open(DATA_PATH + 'alpha.pkl', 'wb') as file:
        pickle.dump(alpha_dict, file)
    data_dict['alpha_dict'] = alpha_dict
    print('Start Portfolio Model ... ')
    port = PortOpt(data_dict, model_param_dict)
    port.optimize()
    stock_arr, results_arr = port.get_results()
    return results_arr

###### BACKTEST
def MaxDrawdown(trading_days, return_list):
    '''return_list is a 1-Dimension list or np.array'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])  # 开始位置
    Max_Drawdown_Rate = (return_list[j] - return_list[i]) / (return_list[j])  # 最大回撤率
    plt.plot(trading_days, return_list)
    plt.title("Max Drawdown", color='k', size=15)
    plt.xlabel("Date", size=10)
    plt.ylabel("Net Worth", size=10)
    plt.plot([trading_days[i], trading_days[j]], [return_list[i], return_list[j]], 'o', color="r", markersize=10)
    plt.legend()
    plt.grid(True)
    plt.show()
    return Max_Drawdown_Rate

def SharpeRatio(return_list):
    '''SharpeRatio=[E(Rp)－Rf]/σp'''
    returns = (return_list[1:]-return_list[:-1])/return_list[:-1]   # 每日收益
    average_return = np.mean(returns)
    return_stdev = np.std(returns)
#  上面得出的夏普比率是以日为单位，我们需要将其年化
    AnnualRet = average_return*252               # 默认252个工作日
    AnnualVol = return_stdev*np.sqrt(252)        # 默认US Treasury Yields为1.5%
    sharpe_ratio = (AnnualRet-0.015) /AnnualVol  # 夏普比率
    return sharpe_ratio

Net_Asset_List = [Principal]
# build the initial investment pool
port_arr = model_main()
init_port = port_arr * Principal
range_list = OpenPrice.columns.to_list() # convert all the trading days to a list
open_price = OpenPrice[range_list[0]].values
holding_shares = init_port / open_price
close_price = ClosePrice[range_list[0]].values
temp = holding_shares * close_price
Principal = np.sum(temp) # sum up the net worth each day after the market close
print(f'Date {range_list[0]} has principal {Principal}')
last_port = temp / Principal
Net_Asset_List.append(Principal) # store the net worth of the day
print(Net_Asset_List)
# update training set
# Add the close_price, which is a 1-D np.array, to your dataset!
stock_df[range_list[0]] = ClosePrice[range_list[0]]
stock_df = stock_df.T.iloc[1:].T

# rolling over each day
# You could set range_list[1:] to range_list[1:20] for test purpose
for idx, date in enumerate(range_list[1:]):
    print(f'########## POSSESSING DATE: {date} ##########')
    # model results
    port_arr = model_main()
    # change of portfolio
    change_arr = port_arr - last_port  # buy and sell info stored in change_arr
    change_dollars = Principal * change_arr
    open_price = OpenPrice[date].values
    change_shares = change_dollars / open_price
    holding_shares += change_shares
    # calculate the net worth after the market closes
    close_price = ClosePrice[date].values
    temp = holding_shares * close_price
    Principal = np.sum(temp) # sum up the net worth each day after the market close
    print(f'Date {date} has principal {Principal}')
    last_port = temp / Principal
    Net_Asset_List.append(Principal) # store the net worth of the day
    print(Net_Asset_List)
    # update training set
    # Add the close_price, which is a 1-D np.array, to your dataset!
    stock_df[date] = ClosePrice[date]
    stock_df = stock_df.T.iloc[1:].T

# What I need here is just 'Net_Asset_List' calculated above
Date = Benchmark['Date'].values
MSCI_Returns = Benchmark['Benchmark Return'].values
print("The Max Drawdown of our quantitative strategy is: ", MaxDrawdown(Date, Net_Asset_List[1:]))
print("The Sharpe Ratio of our quantitative strategy is: ", SharpeRatio(Net_Asset_List))
Strat_Daily_Returns = (Net_Asset_List[1:]-Net_Asset_List[:-1])/Net_Asset_List[:-1]

# Plot the return curve
plt.plot(Date, MSCI_Returns, 'y-', linewidth=1.5, label="Benchmark")
plt.plot(Date, Strat_Daily_Returns, 'r-', linewidth=1.8, label="Strategy")
plt.legend()
plt.title("The Return Curve", color='k', size=15)
plt.xlabel("Date", size=10)
plt.ylabel("Returns", size=10)
plt.grid(True)
plt.show()