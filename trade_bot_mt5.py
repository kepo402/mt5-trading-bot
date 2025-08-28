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

    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = "market", price: float = None):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        tick = mt5.symbol_info_tick(symbol)
        if side == "buy":
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            order_price = tick.ask
        elif side == "sell":
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            order_price = tick.bid
        else:
            raise ValueError("Invalid order side")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": quantity,
            "type": order_type_mt5,
            "price": order_price,
            "deviation": 20,
            "magic": 234000,
            "comment": "python exness trade bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: {result}")
        else:
            print(f"Order successful: {result}")

    def get_balance(self):
        account_info = mt5.account_info()
        return {"balance": account_info.balance, "equity": account_info.equity}

    def get_market_data(self, symbol: str):
        tick = mt5.symbol_info_tick(symbol)
        return {"symbol": symbol, "price": tick.last}


# ===============================
# Strategy Interface
# ===============================
class Strategy(abc.ABC):
    """Abstract base class for strategies."""

    @abc.abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> str:
        pass


class MovingAverageStrategy(Strategy):
    """Simple placeholder strategy."""

    def generate_signal(self, market_data: Dict[str, Any]) -> str:
        price = market_data["price"]
        if price > 100:
            return "buy"
        elif price < 100:
            return "sell"
        return "hold"


# ===============================
# Risk Management
# ===============================
class RiskManager:
    def __init__(self, max_risk_per_trade: float = 0.01):
        self.max_risk_per_trade = max_risk_per_trade

    def position_size(self, balance: float, price: float) -> float:
        risk_amount = balance * self.max_risk_per_trade
        return round(risk_amount / price, 2)


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
    LOGIN = 12345678  # Replace with your Exness MT5 account login
    PASSWORD = "yourpassword"  # Replace with your Exness MT5 password
    SERVER = "Exness-MT5Trial"  # Replace with your Exness MT5 server name

    broker = ExnessMT5Broker(LOGIN, PASSWORD, SERVER)
    strategy = MovingAverageStrategy()
    risk_manager = RiskManager(max_risk_per_trade=0.02)

    bot = TradingBot(broker, strategy, risk_manager)
    bot.run("EURUSD")

