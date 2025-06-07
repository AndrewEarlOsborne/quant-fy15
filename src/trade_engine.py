import ccxt
import logging
from typing import Dict, Optional
from .config import config
from .data_manager import DataManager
from datetime import datetime

logger = logging.getLogger(__name__)

class TradingEngine:
    """Simple, low value trading agent"""
    
    def __init__(self):
        self.data_manager = DataManager()
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': config.exchange_api_key,
            'secret': config.exchange_secret,
            'sandbox': config.exchange_sandbox,
            'enableRateLimit': True,
        })
        
        self.symbol = 'ETH/USDT'
        self.min_order_size = 0.001  # Minimum ETH order size
    
    def get_current_price(self) -> float:
        """Get current ETH price"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price: {e}")
            return 0.0
    
    def get_balance(self) -> Dict[str, float]:
        """Get current balances"""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'ETH': balance.get('ETH', {}).get('free', 0),
                'USDT': balance.get('USDT', {}).get('free', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'ETH': 0, 'USDT': 0}
    
    def execute_trade(self, prediction: int, confidence: float) -> Optional[Dict]:
        """Execute trade based on prediction"""
        try:
            current_price = self.get_current_price()
            balances = self.get_balance()
            
            trade_data = {
                'prediction': prediction,
                'confidence': confidence,
                'price': current_price,
                'action': 'HOLD',
                'quantity': 0,
                'order_id': None,
                'profit_loss': 0
            }
            
            if prediction > 1 and confidence >= config.prediction_confidence_threshold:
                # Buy signal
                result = self._execute_buy(balances, current_price, confidence)
                if result:
                    trade_data.update(result)
                    
            elif prediction == 0 and balances['ETH'] > self.min_order_size:
                # Sell signal
                result = self._execute_sell(balances, current_price)
                if result:
                    trade_data.update(result)
            
            # Save trade record
            self.data_manager.save_trade_record(trade_data)
            
            return trade_data
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def _execute_buy(self, balances: Dict, price: float, confidence: float) -> Optional[Dict]:
        """Execute buy order"""
        try:
            # Calculate position size based on risk and confidence
            usdt_balance = balances['USDT']
            risk_amount = usdt_balance * config.risk_per_trade * confidence
            max_position = usdt_balance * config.max_position_size
            
            position_size = min(risk_amount, max_position)
            quantity = position_size / price
            
            if quantity < self.min_order_size:
                logger.warning(f"Buy quantity too small: {quantity}")
                return None
            
            logger.info(f"Executing BUY: {quantity:.6f} ETH at ${price:.2f}")
            
            if not config.exchange_sandbox:
                order = self.exchange.create_market_buy_order(self.symbol, quantity)
                order_id = order['id']
                actual_quantity = order.get('filled', quantity)
            else:
                # Simulated order for sandbox
                order_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                actual_quantity = quantity
            
            return {
                'action': 'BUY',
                'quantity': actual_quantity,
                'order_id': order_id
            }
            
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            return None
    
    # def _execute_sell(self, balances: Dict, price: float) -> Optional[Dict]:
    #     """Execute sell order"""
    #     try:
    #         eth_balance = balances['ETH']
            
    #         if eth_balance < self.min_order_size:
    #             logger.warning(f"ETH balance too low: {eth_balance}")
    #             return None
            
    #         logger.info(f"Executing SELL: {eth_balance:.6f} ETH at ${price:.2f}")
            
    #         if not config.exchange_sandbox:
    #             order = self.exchange.create_market_sell_order(self.symbol, eth_balance)
    #             order_id = order['id']
    #             actual_quantity = order.get('filled', eth_balance)
    #         else:
    #             # Simulated order for sandbox
    #             order_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    #             actual_quantity = eth_balance
            
    #         return {
    #             'action': 'SELL',
    #             'quantity': actual_quantity,
    #             'order_id': order_id
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Error executing sell order: {e}")
    #         return None