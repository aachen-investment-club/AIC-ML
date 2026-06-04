from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class TransactionType(Enum): 
    BUY ="PURCHASE"
    SELL = "SALE"


class Currency(Enum): 
    #: TODO: add RMB, YEN
    USD = "USD"
    EUR = "EUR" 




class Security(BaseModel): 
    name : str
    ticker : str
    currency  : Currency

class Transaction(BaseModel): 
    type: TransactionType
    account: Currency
    portfolio: str
    date: str 
    currency: Currency
    shares: float
    security : Security
    


class TradeLog(BaseModel): 
    transactions: List
    version: str
    name: str

    
    def append_trade(
            self,
            type: TransactionType, 
            currency: Currency, 
            date: str, 
            shares: float, 
            security: Security
        ): 

        transaction = Transaction(
            type= type, 
            account= currency, 
            portfolio = self.name, 
            date= date, 
            currency= currency, 
            shares= shares, 
            security= security
        )
        self.transactions.append(transaction)




