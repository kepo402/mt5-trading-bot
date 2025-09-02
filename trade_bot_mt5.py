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
        
        
        import pandas as pd

class SmartStrategy(Strategy):
    """Improved strategy with MA crossover + RSI + ATR filters."""

    def __init__(self, fast_period=20, slow_period=50, rsi_period=14, atr_period=14,
                 timeframe=mt5.TIMEFRAME_M5, bars=500, atr_threshold=0.0005):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.timeframe = timeframe
        self.bars = bars
        self.atr_threshold = atr_threshold  # filter to avoid dead markets

    def generate_signal(self, market_data: Dict[str, Any]) -> str:
        symbol = market_data["symbol"]

        # Get historical data
        rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, self.bars)
        if rates is None or len(rates) < max(self.slow_period, self.rsi_period, self.atr_period):
            print("[Strategy] Not enough data for indicators.")
            return "hold"

        df = pd.DataFrame(rates)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        # Moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()

        # Latest values
        fast_prev, fast_curr = df['fast_ma'].iloc[-2], df['fast_ma'].iloc[-1]
        slow_prev, slow_curr = df['slow_ma'].iloc[-2], df['slow_ma'].iloc[-1]
        rsi_curr = df['rsi'].iloc[-1]
        atr_curr = df['atr'].iloc[-1]

        # ATR filter
        if atr_curr < self.atr_threshold:
            print("[Strategy] Market too quiet, skipping.")
            return "hold"

        # Buy condition
        if fast_prev < slow_prev and fast_curr > slow_curr and rsi_curr > 55:
            return "buy"

        # Sell condition
        elif fast_prev > slow_prev and fast_curr < slow_curr and rsi_curr < 45:
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
# Prop Firm Rules
# ===============================
from datetime import datetime
class PropFirmRules:
    def __init__(self, daily_loss_limit_pct=4.0, max_trades_per_day=3, max_overall_loss_pct=10.0):
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_trades_per_day = max_trades_per_day
        self.max_overall_loss_pct = max_overall_loss_pct

        self.starting_balance = None   # first balance when bot starts
        self.starting_equity_today = None
        self.trades_today = 0
        self.last_reset_date = None

    def reset_daily(self, current_equity):
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.starting_equity_today = current_equity
            self.trades_today = 0
            self.last_reset_date = today
            print(f"[PropRules] Daily reset: starting equity = {current_equity}")

    def can_trade(self, current_equity):
        # Set starting balance once
        if self.starting_balance is None:
            self.starting_balance = current_equity

        # Reset daily
        self.reset_daily(current_equity)

        # Check daily drawdown
        loss_pct = ((current_equity - self.starting_equity_today) / self.starting_equity_today) * 100
        if loss_pct <= -self.daily_loss_limit_pct:
            print(f"[PropRules] ❌ Daily loss limit hit ({loss_pct:.2f}%). No more trades today.")
            return False

        # Check overall drawdown
        overall_loss_pct = ((current_equity - self.starting_balance) / self.starting_balance) * 100
        if overall_loss_pct <= -self.max_overall_loss_pct:
            print(f"[PropRules] ❌ Overall loss limit hit ({overall_loss_pct:.2f}%). Trading stopped.")
            return False

        # Check trade count
        if self.trades_today >= self.max_trades_per_day:
            print("[PropRules] ❌ Max trades reached today. No more trades.")
            return False

        return True

    def register_trade(self):
        self.trades_today += 1
         
        
        
# ===============================
# Trading Engine
# ===============================
class TradingBot:
    def __init__(self, broker: BrokerAPI, strategy: Strategy, risk_manager: RiskManager, prop_rules: PropFirmRules):
        self.broker = broker
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.prop_rules = prop_rules

    def run(self, symbol: str):
        market_data = self.broker.get_market_data(symbol)
        signal = self.strategy.generate_signal(market_data)
        account = self.broker.get_balance()
        balance = account["equity"]
        qty = self.risk_manager.position_size(balance, market_data["price"])

        if self.prop_rules.can_trade(balance):
            if signal == "buy":
                self.broker.place_order(symbol, "buy", qty)
                self.prop_rules.register_trade()
            elif signal == "sell":
                self.broker.place_order(symbol, "sell", qty)
                self.prop_rules.register_trade()
            else:
                print("[Bot] Holding position, no action.")
        else:
            print("[Bot] Prop rules blocked trading today.")



# ===============================
# Example Run (replace with your Exness MT5 details)
# ===============================
if __name__ == "__main__":
    LOGIN = 211019157
    PASSWORD = "Jungle123."
    SERVER = "Exness-MT5Trial9"

    broker = ExnessMT5Broker(LOGIN, PASSWORD, SERVER)
    strategy = SmartStrategy(
    fast_period=20,
    slow_period=50,
    rsi_period=14,
    atr_period=14,
    timeframe=mt5.TIMEFRAME_M5,
    atr_threshold=0.0005   # adjust based on symbol volatility
)

    risk_manager = RiskManager(max_risk_per_trade=0.02)
    prop_rules = PropFirmRules(
    daily_loss_limit_pct=4.0,   # daily max loss
    max_trades_per_day=3,       # trades per day
    max_overall_loss_pct=10.0   # overall max loss (like prop firms)
)


    bot = TradingBot(broker, strategy, risk_manager, prop_rules)
    bot.run("EURUSDm")

