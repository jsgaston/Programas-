 # IMPORTING PACKAGES


from fileinput import close
from signal import signal
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import requests
from math import floor
from termcolor import colored as cl
import time
from datetime import datetime
import config
from binance.client import Client
from binance.enums import *
import talib
from talib import *
import asyncio
from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import telegram_send

now = datetime.now()


COMPRANDO=[None]
VENDIENDO=[None]
#position=[]
precio=[None]
symbolo=input("¿que simbolo?: ")
cantidad=input("¿que cantidad?: ")
leve=input("¿que leverage?: ")
client = Client(config.API_KEY, config.API_SECRET, tld='com')


ii=1
while ii!=0:
    print(symbolo)

    async def order_book(client, symbol):
        order_book = await client.get_order_book(symbol=symbolo)
        print(order_book)


    async def kline_listener(client):
        bm = BinanceSocketManager(client)
        symbol = symbolo
        res_count = 0
        async with bm.kline_socket(symbol=symbol) as stream:
            while True:
                res = await stream.recv()
                res_count += 1
                print(res)
                if res_count == 5:
                    res_count = 0
                    loop.call_soon(asyncio.create_task, order_book(client, symbol))


    #INTERVALOS: 1HOUR 1MINUTE

    klines = client.get_klines(symbol=symbolo,interval=Client.KLINE_INTERVAL_1DAY)


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

    ###########################################################

    ########################################################

    ###########################################################
    df = df.drop(df.columns[[ 6, 7, 8, 9, 10, 11]], axis=1)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True, drop=True)
    #data = data.drop(data.columns[0],axis=1)

    o=df['Open']   = df['Open'].astype(float)
    h=df['High']   = df['High'].astype(float)
    l=df['Low']    = df['Low'].astype(float)
    c=df['Close']  = df['Close'].astype(float)
    v=df['Volume'] = df['Volume'].astype(float)
       
    aapl=df
    """
    rATR = talib.ATR(h,l,c,14)
    #print("primer ATR")
    #print(rATR)
    v=rATR.shape
    r=v[0]-1
    datr1=rATR[r]
    print("atr1h es:", datr1)
    f=c.shape
    g=f[0]-1
    cc=c[g]
    CC = float(round((cc) , 5))
    print("el close es:",CC)
    #tpl=cc+((1.5)*datr1d)
    bsl=cc-((2)*datr1)
    ssl=cc+((2)*datr1)
    #sll=cc-((1.5)*datr1d)
    #TPL = float(round((tpl) , 4))
    #SLL = float(round((sll) , 4))    
    BSL = float(round((bsl) , 5))
    SSL = float(round((ssl) , 5))
    #print("TPL:",TPL)
    #print("SLL:",SLL)
    print("TPS:",BSL)
    print("SLS:",SSL)
    """




    # WILLIAMS %R CALCULATION





    def get_wr(high, low, close, lookback):
        highh = high.rolling(lookback).max() 
        lowl = low.rolling(lookback).min()
        wr = -100 * ((highh - close) / (highh - lowl))
        return wr

    aapl['wr_14'] = get_wr(aapl['High'], aapl['Low'], aapl['Close'], 14)
    aapl.tail()
    print(aapl)


    # MACD CALCULATION

    def get_macd(price, slow, fast, smooth):
        exp1 = price.ewm(span = fast, adjust = False).mean()
        exp2 = price.ewm(span = slow, adjust = False).mean()
        macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
        signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
        hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
        return macd, signal, hist

    aapl['macd'] = get_macd(aapl['Close'], 26, 12, 9)[0]
    aapl['macd_signal'] = get_macd(aapl['Close'], 26, 12, 9)[1]
    aapl['macd_hist'] = get_macd(aapl['Close'], 26, 12, 9)[2]
    aapl = aapl.dropna()
    aapl.tail()
    print(aapl)

    # TRADING STRATEGY

    def implement_wr_macd_strategy(prices, wr, macd, macd_signal):    
        buy_price = []
        sell_price = []
        wr_macd_signal = []
        signal = 0

        for i in range(len(wr)):
            if wr[i-1] > -50 and wr[i] < -50 and macd[i] > macd_signal[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    wr_macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    wr_macd_signal.append(0)
                    
            elif wr[i-1] < -50 and wr[i] > -50 and macd[i] < macd_signal[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    wr_macd_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    wr_macd_signal.append(0)
            
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                wr_macd_signal.append(0)
                
        return buy_price, sell_price, wr_macd_signal

    def placeBuyOrder():
        client.futures_change_leverage(symbol=symbolo,leverage=leve)
        buy_order=client.futures_create_order(
                                                    symbol=symbolo,
                                                    side='BUY',
                                                    type='MARKET',
                                                    quantity=cantidad,
                                                    #reduceOnly=True
                                                    )
        order_Buy_ID=buy_order['orderId']
        print(buy_order)
        precio=buy_order['price']
        print("Order ID: " + str(order_Buy_ID))
    """
    def placeBuySL():
        order_buysl=client.futures_create_order(
                                                symbol=symbolo,
                                                #quantity=cantidad,
                                                side='SELL',
                                                type='STOP_MARKET',
                                                stopPrice=BSL,
                                                closePosition=True
                                                #activationPrice=cc*1.01,
                                                #callbackRate=1,
                                                ##reduceOnly='True'
                                                )
        order_Buysl_ID=order_buysl['orderId']
        print(order_buysl)
        print("Or der IDsl: " + str(order_Buysl_ID))    
    """
    def placeSellOrder():
        client.futures_change_leverage(symbol=symbolo,leverage=leve)
        sell_order=client.futures_create_order(
                                        symbol=symbolo,
                                        side='SELL',
                                        type='MARKET',
                                        quantity=cantidad,
                                        #reduceOnly=True
                                        )

        order_Sell_ID=sell_order['orderId']
        print(sell_order)
        precio=sell_order['price']
        print("Order ID: " + str(order_Sell_ID))
    """
    def placeSellSL():
        order_sellsl=client.futures_create_order(
                                                symbol=symbolo,
                                                #quantity=cantidad,
                                                side='BUY',
                                                type='STOP_MARKET',
                                                stopPrice=SSL,
                                                closePosition=True
                                                #activationPrice=cc*1.01,
                                                #callbackRate=1,
                                                ##reduceOnly='True'
                                                )
        order_Sellsl_ID=order_sellsl['orderId']
        print(order_sellsl)
        print("Or der IDsl: " + str(order_Sellsl_ID))    
    """
    buy_price, sell_price, wr_macd_signal = implement_wr_macd_strategy(aapl['Close'], aapl['wr_14'], aapl['macd'], aapl['macd_signal'])

    print("la señal vale:",wr_macd_signal)
    # POSITION

    close_price = aapl['Close']

    if client.futures_position_information(symbol=symbolo)[-1]['positionAmt']!="0":
            if client.futures_position_information(symbol=symbolo)[-1]['positionAmt']!="0" and wr_macd_signal[-1]==-1 and wr_macd_signal[-2]==0 and COMPRANDO[0]==1:
                close_price=client.futures_create_order(symbol=symbolo,side='SELL', type="MARKET", quantity=cantidad, redueOnly='True')
                telegram_send.send(messages=["ORDEN DE CERRADA COMPRA WILL+MAC",symbolo])
                print("ORDEN DE CERRADA COMPRA WILL+MAC",symbolo)
                #if len(client.futures_get_open_orders(symbol=symbolo))!=0:
                    #client.futures_cancel_all_open_orders(symbol=symbolo)
            elif client.futures_position_information(symbol=symbolo)[-1]['positionAmt']!="0" and wr_macd_signal[-1]==1 and wr_macd_signal[-2]==0 and VENDIENDO[0]==1:
                close_price=client.futures_create_order(symbol=symbolo,side='BUY', type="MARKET", quantity=cantidad, redueOnly='True')
                telegram_send.send(messages=["ORDEN CERRADA DE VENTA WILL+MAC",symbolo])
                print("ORDEN CERRADA DE VENTA WILL+MAC",symbolo)
                #if len(client.futures_get_open_orders(symbol=symbolo))!=0:
                    #client.futures_cancel_all_open_orders(symbol=symbolo)
            else:
                print("na de na")

    else:
        if wr_macd_signal[-1] ==1 and client.futures_position_information(symbol=symbolo)[-1]['positionAmt']=="0":
            placeBuyOrder()
            #placeBuySL()
            COMPRANDO[0]=1
            telegram_send.send(messages=["ORDEN DE COMPRA WILL+MAC",symbolo])
        elif wr_macd_signal[-1] ==-1 and client.futures_position_information(symbol=symbolo)[-1]['positionAmt']=="0":
            placeSellOrder()
            #placeSellSL()
            VENDIENDO[0]=1
            telegram_send.send(messages=["ORDEN DE VENTA WILL+MAC",symbolo])
        elif client.futures_position_information(symbol=symbolo)[-1]['positionAmt']=="0":
            print("aún no")
        else:
            print("ya hay ordenes")



    
            
    

    wr = aapl['wr_14']
    macd_line = aapl['macd']
    signal_line = aapl['macd_signal']
    #wr_macd_signal = pd.DataFrame(wr_macd_signal).rename(columns = {0:'wr_macd_signal'}).set_index(aapl.index)
    #position = pd.DataFrame(position).rename(columns = {0:'wr_macd_position'}).set_index(aapl.index)

    #frames = [close_price, wr, macd_line, signal_line, wr_macd_signal, position]
    #strategy = pd.concat(frames, join = 'inner', axis = 1)

    #strategy.head()
    

ii=+1