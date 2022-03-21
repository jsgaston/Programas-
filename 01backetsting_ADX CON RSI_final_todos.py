# IMPORTING PACKAGES

from operator import index
import numpy as np
from numpy import nan
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import requests
from math import floor, sqrt, ceil
from termcolor import colored as cl
import statistics
from binance.client import Client
from backtesting import Backtest, Strategy
from backtesting.lib import *

from backtesting.test import *

#import talib as ta
import time

import config
import datetime
from csv import DictWriter, writer

from numpy import matrixlib

tod = datetime.datetime.now()
d = datetime.timedelta(days = 300)
a = tod - d
#print("SINCE",a)
from datetime import datetime
now=datetime.now()
client = Client(config.API_KEY, config.API_SECRET, tld='com')
aa=a.strftime("%Y-%m-%d %H:%M:%S")

#print(aa)
#symbolo="BNBUSDT"   #input("Enter a stock ticker symbol: ")
idate=aa
#idate=input("Enter a since when u whant to test symbol: ")
rsi01=[]
dff=[]

#idate="2017-12-21"
#tickers='ATOMUSDT'
# EXTRACTING DATA
AA=[]
tickers = client.get_all_tickers()
AA=len(tickers)
for x in range(AA):

    for tick in tickers:
        if tick['symbol'][-4:] == 'USDT': #and tick['symbol'][-4:] != 'BUSD' and tick['symbol'][-3:] != 'BNB':
           
        #klines = client.get_historical_klines(tick['symbol'], Client.KLINE_INTERVAL_5MINUTE, "5 minute ago UTC")
        #if len(klines) != 1:
            #continue
            symbol = tick['symbol']
            print(symbol)

            df = []
            def get_historical_data(symbol, start_date):
                klines = client.get_historical_klines(tick['symbol'],client.KLINE_INTERVAL_30MINUTE,idate)
                
                
                    
                
                    

                
                #start = ["month_1"].values.astype('datetime64[D]')

                df = pd.DataFrame(klines,  columns=['Date',
                                                    'Open',
                                                    'High',
                                                    'Low',
                                                    'Close',
                                                    'Volume',
                                                    'Close time',
                                                    'Quote asset volume',
                                                    'Number of trades',
                                                    'Taker buy base asset volume',
                                                    'Taker buy quote asset volume',
                                                    'Ignore'])
                df = df.drop(df.columns[[6, 7, 8, 9, 10, 11]], axis=1)
                
                df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                df.set_index('Date', inplace=True, drop=True)

            
                df['open']   = df['Open'].astype(float)
                df['high']   = df['High'].astype(float)
                df['low']    = df['Low'].astype(float)
                df['close']  = df['Close'].astype(float)
                df['volume'] = df['Volume'].astype(float)
                
            #    raw_df = requests.get(api_url).json()
            #    df = pd.DataFrame(raw_df['values']).iloc[::-1].set_index('datetime').astype(float)
                df = df[df.index >= start_date]
                df.index = pd.to_datetime(df.index)
                    
                return df
            aapl = []
            
            aapl = get_historical_data(tick['symbol'], idate)
            if len(aapl)!=0:
                    
                #while len(aapl)!=0:
                #    continue
                #df=aapl
                aapl.tail()
                """
                # EXTRACTING STOCK DATA

                def get_historical_data(symbol, start_date):
                    api_key = 'YOUR API KEY'
                    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
                    raw_df = requests.get(api_url).json()
                    df = pd.DataFrame(raw_df['values']).iloc[::-1].set_index('datetime').astype(float)
                    df = df[df.index >= start_date]
                    df.index = pd.to_datetime(df.index)
                    return df

                aapl = get_historical_data('AAPL', idate)
                aapl.tail()
                """
                # ADX CALCULATION

                def get_adx(high, low, close, lookback):
                    plus_dm = high.diff()
                    minus_dm = low.diff()
                    plus_dm[plus_dm < 0] = 0
                    minus_dm[minus_dm > 0] = 0
                    tr1=[]
                    tr2=[]
                    tr3=[]
                    tr=[]
                    frames=[]
                    dx=[]
                    adx=[]
                    adx_smooth=[]


                    tr1 = pd.DataFrame(high - low)
                    tr2 = pd.DataFrame(abs(high - close.shift(1)))
                    tr3 = pd.DataFrame(abs(low - close.shift(1)))
                    frames = [tr1, tr2, tr3]
                    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
                    atr = tr.rolling(lookback).mean()
                    
                    plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
                    minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
                    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
                    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
                    adx_smooth = adx.ewm(alpha = 1/lookback).mean()
                    return plus_di, minus_di, adx_smooth

                aapl['plus_di'] = pd.DataFrame(get_adx(aapl['high'], aapl['low'], aapl['close'], 14)[0]).rename(columns = {0:'plus_di'})
                aapl['minus_di'] = pd.DataFrame(get_adx(aapl['high'], aapl['low'], aapl['close'], 14)[1]).rename(columns = {0:'minus_di'})
                aapl['adx'] = pd.DataFrame(get_adx(aapl['high'], aapl['low'], aapl['close'], 14)[2]).rename(columns = {0:'adx'})
                aapl = aapl.dropna()
                aapl.tail()

                # ADX PLOT
                """
                plot_data = aapl[aapl.index >= '2020-01-01']

                ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
                ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
                ax1.plot(plot_data['close'], linewidth = 2, color = '#ff9800')
                ax1.set_title('AAPL CLOSING PRICE')
                ax2.plot(plot_data['plus_di'], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)
                ax2.plot(plot_data['minus_di'], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)
                ax2.plot(plot_data['adx'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
                ax2.axhline(35, color = 'grey', linewidth = 2, linestyle = '--')
                ax2.legend()
                ax2.set_title('AAPL ADX 14')
                plt.show()
                """
                # RSI CALCULATION

                def get_rsi(close, lookback):
                    ret = []
                    ret = close.diff()
                    up = []
                    down = []
                    up_series=[]
                    down_series=[]
                                

                    for i in range(len(ret)):
                        if ret[i] < 0:
                            up.append(0)
                            down.append(ret[i])
                        else:
                            up.append(ret[i])
                            down.append(0)
                    
                    up_series = pd.Series(up)
                    down_series = pd.Series(down).abs()
                    up_ewm=[]
                    down_ewm = []
                    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
                    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
                    
                    rs = []
                    rs = up_ewm/down_ewm
                    rsi = []
                    rsi = 100 - (100 / (1 + rs))
                    rsi_df = []
                    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
                    rsi_df = rsi_df.dropna()
                    
                    return rsi_df[13:]

                aapl['rsi_14'] = get_rsi(aapl['close'], 14)
                aapl = aapl.dropna()
                aapl.tail()
                aapl['rsi_14']
                # RSI PLOT
                """
                plot_data = aapl[aapl.index >= '2020-01-01']

                ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
                ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
                ax1.plot(plot_data['close'], linewidth = 2.5)
                ax1.set_title('AAPL STOCK PRICES')
                ax2.plot(plot_data['rsi_14'], color = 'orange', linewidth = 2.5)
                ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
                ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
                ax2.set_title('AAPL RSI 14')
                plt.show()

                # RSI ADX PLOT

                plot_data = aapl[aapl.index >= '2020-01-01']

                ax1 = plt.subplot2grid((19,1), (0,0), rowspan = 5, colspan = 1)
                ax2 = plt.subplot2grid((19,1), (7,0), rowspan = 5, colspan = 1)
                ax3 = plt.subplot2grid((19,1), (14,0), rowspan = 5, colspan = 1)

                ax1.plot(plot_data['close'], linewidth = 2.5)
                ax1.set_title('AAPL STOCK PRICES')

                ax2.plot(plot_data['rsi_14'], color = 'orange', linewidth = 2.5)
                ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
                ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
                ax2.set_title('AAPL RSI 14')

                ax3.plot(plot_data['plus_di'], color = '#26a69a', label = '+ DI 14', linewidth = 3, alpha = 0.3)
                ax3.plot(plot_data['minus_di'], color = '#f44336', label = '- DI 14', linewidth = 3, alpha = 0.3)
                ax3.plot(plot_data['adx'], color = '#2196f3', label = 'ADX 14', linewidth = 3)
                ax3.axhline(35, color = 'grey', linewidth = 2, linestyle = '--')
                ax3.legend()
                ax3.set_title('AAPL ADX 14')
                plt.show()
                """
                # TRADING STRATEGY

                def adx_rsi_strategy(prices, adx, pdi, ndi, rsi):
                    buy_price = []
                    sell_price = []
                    adx_rsi_signal = []
                    
                    signal = None
                    signal = 0
                    adx_rsi_misenal = []
                    for i in range(len(prices)):
                        if adx[i] > 35 and pdi[i] < ndi[i] and rsi[i] < 50:
                            if signal != 1:
                                buy_price.append(prices[i])
                                sell_price.append(np.nan)
                                signal = 1
                                adx_rsi_signal.append(signal)
                                adx_rsi_misenal.append(signal)
                            else:
                                buy_price.append(np.nan)
                                sell_price.append(np.nan)
                                adx_rsi_signal.append(0)
                                
                                
                        elif adx[i] > 35 and pdi[i] > ndi[i] and rsi[i] > 50:
                            if signal != -1:
                                buy_price.append(np.nan)
                                sell_price.append(prices[i])
                                signal = -1
                                adx_rsi_signal.append(signal)
                                adx_rsi_misenal.append(signal)
                            else:
                                buy_price.append(np.nan)
                                sell_price.append(np.nan)
                                adx_rsi_signal.append(0)
                        else:
                            buy_price.append(np.nan)
                            sell_price.append(np.nan)
                            adx_rsi_signal.append(0)
                                        
                    return buy_price, sell_price, adx_rsi_signal, adx_rsi_misenal

                buy_price, sell_price, adx_rsi_signal, adx_rsi_misenal = adx_rsi_strategy(aapl['close'], aapl['adx'], aapl['plus_di'], aapl['minus_di'], aapl['rsi_14'])
                print(adx_rsi_misenal)
                print(adx_rsi_signal)
                
                """
                plt.plot(aapl['close'])
                plt.plot(aapl.index, buy_price, marker = '^', markersize = 10, color = 'green')
                plt.plot(aapl.index, sell_price, marker = 'v', markersize = 10, color = 'r')
                """
                ########################################################################################
                num_trades = []
                num_trades=len(adx_rsi_signal)
                #print("el num de intervalos es:",num_trades)
                numero_trades = []
                numero_trades=len(adx_rsi_misenal)

                #print(adx_rsi_misenal[aapl['date']])
                #print("el num de trades hechos es:",numero_trades)
                #print("aapl tiene estas col")
                #print(aapl)
                #buy_price=np.buy_price()
                #print(buy_price)
                #print(len(adx_rsi_misenal))
                #print(adx_rsi_misenal)
                buy_price001 = []
                buy_price001 = [x for x in buy_price if str(x) != 'nan']

                #print(len(buy_price001))
                #print(buy_price001)
                sell_price001 = []
                sell_price001 = [x for x in sell_price if str(x) != 'nan']

                #print(sell_price001)
                #print(len(sell_price001))
                ################################################################################################
                # POSITION
                
                frames = []
                strategy = []
                position = []
                
                #adx_rsi_signal=[]
                for i in range(len(adx_rsi_signal)):
                    if adx_rsi_signal[i] > 1:
                        position.append(0)
                    else:
                        position.append(1)
                        
                for i in range(len(aapl['close'])):
                    if adx_rsi_signal[i] == 1:
                        position[i] = 1
                    elif adx_rsi_signal[i] == -1:
                        position[i] = 0
                    else:
                        position[i] = position[i-1]
                
                pdi= []
                ndi = []
                close_price=[]
                

                adx = pd.DataFrame(aapl['adx'])
                pdi = pd.DataFrame(aapl['plus_di'])
                ndi = pd.DataFrame(aapl['minus_di'])
                rsi = pd.DataFrame(aapl['rsi_14']) 
                close_price = pd.DataFrame(aapl['close'])
                adx_rsi_signal = pd.DataFrame(adx_rsi_signal).rename(columns = {0:'adx_rsi_signal'}).set_index(aapl.index)
                position = pd.DataFrame(position).rename(columns = {0:'adx_rsi_position'}).set_index(aapl.index)

                frames = [close_price, adx, pdi, ndi, rsi, adx_rsi_signal, position]
                strategy = pd.concat(frames, join = 'inner', axis = 1)

                strategy.tail()
                #+strategy[45:50]

                # BACKTESTING
                aapl_ret = []
                aapl_ret = pd.DataFrame(np.diff(aapl['close'])).rename(columns = {0:'returns'})
                adx_rsi_strategy_ret = []

                for i in range(len(aapl_ret)):
                    returns = aapl_ret['returns'][i]*strategy['adx_rsi_position'][i]
                    adx_rsi_strategy_ret.append(returns)
                adx_rsi_strategy_ret_df = []
                adx_rsi_strategy_ret_df = pd.DataFrame(adx_rsi_strategy_ret).rename(columns = {0:'adx_rsi_returns'})
                aapl['close']= pd.DataFrame(aapl['close'])
                investment_value=None
                investment_value = 100000
                number_of_stocks = []
                #print(aapl)
                #print(aapl['close'])
                number_of_stocks = floor(investment_value/(aapl['close'][0]))

                adx_rsi_investment_ret = []
                returns=[]
                


                for i in range(len(adx_rsi_strategy_ret_df['adx_rsi_returns'])):
                    returns = number_of_stocks*adx_rsi_strategy_ret_df['adx_rsi_returns'][i]
                    adx_rsi_investment_ret.append(returns)


                adx_rsi_investment_ret_df = pd.DataFrame(adx_rsi_investment_ret).rename(columns = {0:'investment_returns'})

                
                total_investment_ret=[]
                profit_percentage=[]
                total_investment_ret = round(sum(adx_rsi_investment_ret_df['investment_returns']), 2)
                profit_percentage = floor((total_investment_ret/investment_value)*100)
                print(cl('Profit gained from the ADX RSI strategy by investing $1000k in AAPL : {}'.format(total_investment_ret), attrs = ['bold']))
                print(cl('Profit percentage of the ADX RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))

                # SPY ETF COMPARISON
                benchmark=[]
                spy=[]
                total_benchmark_investment_ret=[]
                benchmark_profit_percentage=[]
                number_of_stocks=[]
                def get_benchmark(start_date, investment_value):
                    spy = get_historical_data('SPY', start_date)['close']
                    benchmark = pd.DataFrame(np.diff(spy)).rename(columns = {0:'benchmark_returns'})
                    
                    
                    number_of_stocks = floor(investment_value/spy[0])
                    benchmark_investment_ret = []
                    
                    for i in range(len(benchmark['benchmark_returns'])):
                        returns = number_of_stocks*benchmark['benchmark_returns'][i]
                        benchmark_investment_ret.append(returns)

                    benchmark_investment_ret_df=[]
                    benchmark_investment_ret_df = pd.DataFrame(benchmark_investment_ret).rename(columns = {0:'investment_returns'})
                    return benchmark_investment_ret_df

                benchmark = get_benchmark(idate, 100000)
                investment_value = 100000
                total_benchmark_investment_ret = round(sum(benchmark['investment_returns']), 2)
                benchmark_profit_percentage = floor((total_benchmark_investment_ret/investment_value)*100)
                print(cl('Benchmark profit by investing $1000k : {}'.format(total_benchmark_investment_ret), attrs = ['bold']))
                print(cl('Benchmark Profit percentage : {}%'.format(benchmark_profit_percentage), attrs = ['bold']))
                print(cl('ADX RSI Strategy profit is {}% higher than the Benchmark Profit'.format(profit_percentage - benchmark_profit_percentage), attrs = ['bold']))

                ###########################################################################################
                            ##### RATIO SHARP
                # Simulate cumulative returns of 100 days
                if len(sell_price001)!=len(buy_price001):
                    if len(buy_price001)>len(sell_price001):
                        sell_price001.append(float(aapl['close'][-1]))
                    else:
                        buy_price001.append(float(aapl['close'][-1]))

                j=1
                inversionne=1
                diferencia=[]
                diferencia2=[]
                listados=[]
                #sell_price001=[]
                #buy_price001=[]
                element=None
                A=[]
                B=[]
                item1=None
                item2=None

                if (adx_rsi_misenal[0]>=1):
                        diferencia=list(-np.array(sell_price001)+np.array(buy_price001))
                        A=[element*1 for element in sell_price001]
                        B=[element*1 for element in buy_price001]
                        #print("primerB",A)
                        A.remove(A[0])
                        #print("quito el primer valor",A)
                        for (item1,item2) in zip(A,B):
                            diferencia2.append(item2-item1)
                else:
                        diferencia=list(np.array(sell_price001)-np.array(buy_price001))
                        A=[element*1 for element in sell_price001]
                        B=[element*1 for element in buy_price001]
                        #print("primerB",B)
                        B.remove(B[0])
                        #print("quito el primer valor",B)
                        for (item1,item2) in zip(B,A):
                            diferencia2.append(item2-item1)   
                        
                        
                #print(diferencia)




                #######################################################################################

                #print(len(diferencia))



                N=[]
                serieses=[]
                seriess=[]
                seriess2=[]
                maximisiomo=[]
                cumsuma=[]
                R=[]
                Rf=[]
                sigma=[]
                r=[]
                sr=[]
                #print(len(diferencia2))
                #print("diferencia2;",diferencia2)

                N = len(diferencia)+len(diferencia2)

                seriess=list(diferencia)
                seriess2=list(diferencia2)
                serieses=sum(seriess)+sum(seriess2)
                #print("serieses:",serieses)
                #print("seriess2:",seriess2)
                #print(adx_rsi_investment_ret_df['investment_returns'])
                #series=    pd.Series(adx_rsi_investment_ret_df['investment_returns'])
                maximisiomo=min(buy_price001 and sell_price001)
                cumsuma=serieses
                R = cumsuma
                Rf = ((1.0-0.036)*maximisiomo)



                #pd.DataFrame(np.random.normal(size=100)).cumsum()
                #print("R vale:",R)
                # Approach 1
                r = (R - Rf)
                #print("R-Rf vale:",r)
                # Approach 2
                #r = R.diff()
                sigma = statistics.stdev(seriess + seriess2)
                #print("sigma vale:",sigma)
                #sr = (r.mean()/r.std()) * np.sqrt(252)
                sr = (R - Rf)/sigma
                #print("el radio sharp es:",sr)

                ###############################################################################################
                ################ SQN
                numero_trades=None
                R_multiple=[]
                R_expetancy=[]
                st_dev=None
                sqn=None
                numero_trades=len(diferencia) + len(diferencia2)
                #print(num_trades)
                #R-multiple

                #print("el retorno es:",serieses)
                #print("el precio minimo vale: ",maximisiomo)
                #print("el valor de lo invertido es:",maximisiomo*len(diferencia))
                C=[element/maximisiomo for element in diferencia]
                D=[element/maximisiomo for element in diferencia2]
                
                R_multiple=(C+D)
                #print(R_multiple)
                #print("R_MULTIPLE:",R_multiple)
                #R_multiple=float(R_multiple)
                #num_trades=float(num_trades)
                R_expetancy=float((sum(R_multiple))/numero_trades)
                #print("R_expentancy es:",R_expetancy)
                #print(R_expetancy)
                st_dev=statistics.stdev(R_multiple)
                #print("la desviación estandard vale:",st_dev)
                sqn=((R_expetancy)/(st_dev))*sqrt(numero_trades)
                #print("El valor de SQN es:",sqn)
                #############################################################################
                #################ROI
                Lotess=[]
                suma_lotes=None
                beneficci=[]
                beneficciose=None
                roi=None
                Lotess=list((np.array(buy_price001)+np.array(sell_price001))/np.array(maximisiomo))

                #print(Lotess)
                suma_lotes=sum(Lotess)

                beneficci=list(np.array(diferencia))+list(np.array(diferencia2))
                #print(beneficci)
                beneficciose=sum(beneficci)


                #inversionnene=sum(diferencia*maximisiomo)
                roi=(beneficciose)*100/(suma_lotes)

                #######################################################################
                ###################### volatility
                prince003=[]
                period_std=[]
                std_003=None
                prices003 = pd.DataFrame(aapl['close'])

                prices003.sort_index(ascending=False, inplace=True)

                prices003['returns'] = (np.log(prices003.close / prices003.close.shift()))

                period_std = np.std(prices003.returns)
                #print(period_std)
                """
                print(period_std)
                = np.array(aapl['close'])
                dff['volatility']=aapl['close'].rolling(num_trades).std()
                print(dff['volatility']*100*sqrt(num_trades))
                """
                std_003= period_std *((num_trades)**0.5)
                

                ###########################################################################
                #################DRAWDOWN MDD
                #aaaaa=aapl(0,0)

                #print(aaaaa)
                aaa=None
                nowaaa=None
                diferencia_años=None
                periods_in_a_year01=None
                periods_in_a_year=None
                window004=None
                Roll_Max=[]
                Daily_Drawdown=[]
                Max_Daily_Drawdown=[]
                MDD =[]
                #aaa=None
                #diferencia_años=[]

                #aaaa=aaaaa.strftime("%Y")
                #print("el inicio de cuenta dias es: ",aaaa)
                aaa=a.strftime("%Y")
                #print("desde el año",aaa)
                nowaaa=now.strftime("%Y")
                ("hoy es",nowaaa)
                diferencia_años=int(nowaaa)-int(aaa)
                #print("inicio años closes",aaa)
                #print("año inicial",nowaaa)
                #print("diferencia de años",diferencia_años)
                periods_in_a_year01=int(num_trades)/diferencia_años
                periods_in_a_year=int(periods_in_a_year01)
                #print("periodos en un año",periods_in_a_year)
                window004=periods_in_a_year
                #print(periods_in_a_year)
                ###################################################
                
                ##################################################
                Roll_Max = aapl['close'].rolling(window004, min_periods=1).max()
                Daily_Drawdown = aapl['close']/Roll_Max -1.0

                Max_Daily_Drawdown = Daily_Drawdown.rolling(window004, min_periods=1).min()
                MDD = Max_Daily_Drawdown.min()*100
                ###########################################################################


                writer_object=None
                
                



                ##########################################################################
                print("##################################################################")
                #field_names = ['numero de intervalos', 'numero de trades', 'SQN', 'Sharpe', 'ROI', 'Volatilidad', 'MDD','beneficios en %',' beneficios','benchmark %', 'benchmark','diferencia en %' ]
                #dict = {'numero de intervalos', 'numero de trades', 'SQN', 'Sharpe', 'ROI', 'Volatilidad', 'MDD','beneficios en %',' beneficios','benchmark %', 'benchmark',"diferencia en %" }
                #toAdd=[str(tick['symbol']),str(num_trades),str( numero_trades), str(sqn), str(sr), str(roi), str(std_003), str(MDD), str(total_investment_ret), str(profit_percentage), str(total_benchmark_investment_ret), str(benchmark_profit_percentage), str(profit_percentage-benchmark_profit_percentage)]
                lista009=[(tick['symbol']),(num_trades),( numero_trades), (sqn), (sr), (roi), (std_003), (MDD), (total_investment_ret), (profit_percentage), (total_benchmark_investment_ret), (benchmark_profit_percentage), (profit_percentage-benchmark_profit_percentage)]
                
                with open('C:/Users/jsgas/OneDrive/Backtesting Binance/001.csv','a+',newline='') as write_obj:
                    csv_writer=writer(write_obj)
                    csv_writer.writerow(lista009)
                #writer_object = writer(f_objetct) 
                #writer_object.writerow(lista009)
                print(tick['symbol'])
                print("el numero de intervalos es: ",num_trades)
                print("el numero de trades es: ",numero_trades)
                print("el SQN es: ",sqn)
                print("el sharp vale: ",sr)
                print("el ROI es: ",roi)
                print("la volatilidad es: ",std_003)
                print("el MDD es: ",MDD)
            continue
        continue
        #print(tick['symbol'])