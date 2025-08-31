import abc
from typing import Dict, Any
import MetaTrader5 as mt5

# ===============================
# Broker / Exchange Connector
# ===============================
class BrokerAPI(abc.ABC):
    """Abstract base class for broker integrations."""

    @abc.abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str, price: float = None):
        pass

    @abc.abstractmethod
    def get_balance(self) -> Dict[str, float]:
        pass

    @abc.abstractmethod
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        pass


class ExnessMT5Broker(BrokerAPI):
    """Exness broker connection via MetaTrader5."""

    def __init__(self, login: int, password: str, server: str):
        if not mt5.initialize(login=login, server=server, password=password):
            raise RuntimeError(f"MT5 initialize failed, error code: {mt5.last_error()}")

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "market", price: float = None,
                    sl_pips: int = 50, tp_pips: int = 100):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if side == "buy":
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            order_price = tick.ask
            sl = order_price - sl_pips * symbol_info.point
            tp = order_price + tp_pips * symbol_info.point
        elif side == "sell":
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            order_price = tick.bid
            sl = order_price + sl_pips * symbol_info.point
            tp = order_price - tp_pips * symbol_info.point
        else:
            raise ValueError("Invalid order side")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": quantity,
            "type": order_type_mt5,
            "price": order_price,
            "sl": round(sl, symbol_info.digits),
            "tp": round(tp, symbol_info.digits),
            "deviation": 20,
            "magic": 234000,
            "comment": "python exness trade bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order failed: {result}")
        else:
            print(f"✅ Order successful: {result}")
            print(f"   SL = {request['sl']} | TP = {request['tp']}")

    def get_balance(self):
        account_info = mt5.account_info()
        return {"balance": account_info.balance, "equity": account_info.equity}

    def get_market_data(self, symbol: str):
        if not mt5.symbol_select(symbol, True):
            raise ValueError(f"Symbol {symbol} not found or not visible in Market Watch.")
    
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Failed to get market data for {symbol}.")
    
    # Use last price if available, otherwise bid/ask
        price = tick.last
        if price == 0.0:
            price = tick.bid if tick.bid > 0 else tick.ask
    
        if price == 0.0:
            raise RuntimeError(f"Symbol {symbol} has no valid price data.")
    
        print(f"[Market Data] {symbol} price = {price}")  # 👈 Debug print so you can see the live price
        return {"symbol": symbol, "price": price}



# ===============================
# Strategy Interface
# ===============================
class Strategy(abc.ABC):
    """Abstract base class for strategies."""

    @abc.abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> str:
        pass

import pandas as pd

class MovingAverageStrategy(Strategy):
    """Real Moving Average Crossover Strategy."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50, timeframe=mt5.TIMEFRAME_M5, bars=200):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.timeframe = timeframe
        self.bars = bars

    def generate_signal(self, market_data: Dict[str, Any]) -> str:
        symbol = market_data["symbol"]

        # Get historical price data
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, self.bars)
        if rates is None or len(rates) < self.slow_period:
            print("[Strategy] Not enough data to calculate MAs.")
            return "hold"

        df = pd.DataFrame(rates)
        df['close'] = df['close'].astype(float)

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()

        # Get the last two values to detect crossover
        fast_prev, fast_curr = df['fast_ma'].iloc[-2], df['fast_ma'].iloc[-1]
        slow_prev, slow_curr = df['slow_ma'].iloc[-2], df['slow_ma'].iloc[-1]

        # Check crossover
        if fast_prev < slow_prev and fast_curr > slow_curr:
            return "buy"
        elif fast_prev > slow_prev and fast_curr < slow_curr:
            return "sell"
        else:
            return "hold"


# ===============================
# Risk Management
# ===============================
class RiskManager:
    def __init__(self, max_risk_per_trade: float = 0.01, min_lot: float = 0.01, max_lot: float = 1.0):
        self.max_risk_per_trade = max_risk_per_trade
        self.min_lot = min_lot
        self.max_lot = max_lot

    def position_size(self, balance: float, price: float) -> float:
        risk_amount = balance * self.max_risk_per_trade
        lots = risk_amount / (price * 100000)   # basic lot size formula for forex
        lots = max(self.min_lot, min(lots, self.max_lot))
        return round(lots, 2)



# ===============================
# Trading Engine
# ===============================
class TradingBot:
    def __init__(self, broker: BrokerAPI, strategy: Strategy, risk_manager: RiskManager):
        self.broker = broker
        self.strategy = strategy
        self.risk_manager = risk_manager

    def run(self, symbol: str):
        market_data = self.broker.get_market_data(symbol)
        signal = self.strategy.generate_signal(market_data)
        balance = self.broker.get_balance()["balance"]
        qty = self.risk_manager.position_size(balance, market_data["price"])

        if signal == "buy":
            self.broker.place_order(symbol, "buy", qty)
        elif signal == "sell":
            self.broker.place_order(symbol, "sell", qty)
        else:
            print("[Bot] Holding position, no action.")


# ===============================
# Example Run (replace with your Exness MT5 details)
# ===============================
if __name__ == "__main__":
    LOGIN = 211019157  # Replace with your Exness MT5 account login
    PASSWORD = "Jungle123."  # Replace with your Exness MT5 password
    SERVER = "Exness-MT5Trial9"  # Replace with your Exness MT5 server name

    broker = ExnessMT5Broker(LOGIN, PASSWORD, SERVER)
    strategy = MovingAverageStrategy(fast_period=20, slow_period=50, timeframe=mt5.TIMEFRAME_M5)

    risk_manager = RiskManager(max_risk_per_trade=0.02)

    bot = TradingBot(broker, strategy, risk_manager)
    bot.run("EURUSDm")

