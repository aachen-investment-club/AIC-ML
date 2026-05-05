from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class TradeLog(BaseModel): 
    transactions: List
    version: str


class TransactionType(Enum): 
    BUY ="PURCHASE"
    SELL = "SALE"



class Currency(Enum): 
    USD = "USD"

class Security(BaseModel): 
    name = str
    ticker = str
    currency  = Currency

class Transaction(BaseModel): 
    type: TransactionType
    account: Currency
    portfolio: str
    date: str 
    currency: Currency
    shares: float
    security : Security
    


