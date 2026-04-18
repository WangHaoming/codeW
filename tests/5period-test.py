import akshare as ak

df = ak.stock_zh_a_minute(
    symbol='sh000001', 
    period='5',
)

print(df)