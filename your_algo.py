from base import Exchange, Trade, Order, Product, Msg, Rest
from typing import List, Dict

class PlayerAlgorithm:
    """
    Main trading algorithm class that implements the player's trading strategy.
    
    This class handles all trading decisions, order management, and market analysis
    for a single trading bot in the exchange simulation.
    """
    
    def __init__(self, products: List[Product]):
        """
        Initialize the trading algorithm with available products and basic configuration.
        
        Args:
            products: List of all tradeable products available in the current round
        """
        self.products = products        # A list of all tradeable products in the current round
        self.name = "PlayerAlgorithm"   # The name representing your bot in the trade logs  
        self.timestamp_num = 0
        self.bids = []
        self.asks = []
    
                 # Counter to track the number of timestamps completed

        # Initialize any other global variables you may need here
        # Examples: position tracking, risk management parameters, strategy state variables

    def send_messages(self, book: Dict[str, Dict[str, List[Rest]]]) -> List[Msg]:
        """
        Main trading logic method that analyzes the order book and generates trading decisions.
        
        This method is called on each trading cycle and should implement your core trading strategy.
        It analyzes the current market state and returns a list of messages to send to the exchange.
        
        Args:
            book: Complete order book structure organized as:
                  {ticker: {Bid: [resting_orders], Ask: [resting_orders]}}
                  - ticker: Product identifier (e.g., "UEC")
                  - Bid: List of resting buy orders (descending price order - highest bid first)
                  - Ask: List of resting sell orders (ascending price order - lowest ask first)
                  - resting_orders: List of Rest objects representing pending orders
        
        Returns:
            List[Msg]: List of messages to send to the exchange (orders, cancellations, etc.)
        """

        """
        This method returns a list of Msg objects you want to send to the exchange.
        There are two types of messages you may want to send: 

        1. Sending an order (Msg with "ORDER" type and Order object)
        2. Removing an order (Msg with "REMOVE" type and order ID)
        
        When sending an order, your Msg object should be created with:
        - First argument: the string "ORDER"
        - Second argument: an Order object with your desired trade parameters
        
        When removing an order, your Msg object should be created with:
        - First argument: the string "REMOVE" 
        - Second argument: the ID of the order you want to cancel

        To make sending and removing orders easier, we have provided the helper functions:
        - create_order(): Creates a new order message
        - remove_order(): Creates a cancel order message
        """

        messages = []
        
        # Example trading logic: Place a buy order on the first cycle
        # This is just a demonstration - replace with your actual strategy
        if self.timestamp_num == 0:
            # Place a buy order for 5 units of UEC at price 1005
            messages.append(self.create_order("UEC", 5, 1005, "Buy"))

        if book["UEC"]["Bids"]:
            self.bids.append(book["UEC"]["Bids"][0].price)
        else:
            self.bids.append(None)

        if book["UEC"]["Asks"]:
            self.asks.append(book["UEC"]["Asks"][0].price)
        else:
            self.asks.append(None)    
        
        self.timestamp_num += 1
        return messages

    def create_order(self, ticker: str, size: int, price: float, direction: str) -> Msg:
        """
        Creates a new order message for the exchange.
        
        This helper method constructs an Order object with the specified parameters
        and wraps it in a Msg object for transmission to the exchange.
        
        Args:
            ticker: Product symbol to trade (e.g., "UEC", "AAPL")
            size: Number of units to trade (positive integer)
            price: Price per unit (float)
            direction: Trade direction - "Buy" or "Sell"
        
        Returns:
            Msg: Message object containing the order for the exchange
        
        Note:
            Automatically assigns a unique order ID and increments the internal counter
        """
        order_idx = self.idx
        new_order = Order(ticker=ticker, price=price, size=size, order_id=order_idx,
                          agg_dir=direction, bot_name=self.name)
        new_message = Msg("ORDER", new_order)
        self.idx += 1
        return new_message

    def remove_order(self, order_idx):
        """
        Creates a message to cancel/remove an existing order.
        
        Args:
            order_idx: The unique ID of the order to cancel
        
        Returns:
            Msg: Message object requesting order cancellation
        """
        new_message = Msg("REMOVE", order_idx)
        return new_message

    def set_idx(self, idx: int) -> None:
        """
        Sets the starting order ID for the current trading session.
        
        Each order requires a unique integer ID to track it through the system.
        You are responsible for generating these IDs, ensuring they are unique
        and fall within the valid range provided by the exchange.
        
        Args:
            idx: Starting order ID for this trading session
        
        Note:
            - Valid ID range: [idx, idx + 999999]
            - Each order ID must be unique within this range
            - Recommended approach: start with `idx` and increment by 1 for each order
            - This function initializes `self.idx` to the starting ID for the session
        """
        self.idx = idx

    def process_trades(self, trades: List[Trade]) -> None:
        """
        Processes completed trades and updates internal state accordingly.
        
        This method is called after each bot turn with a list of all trades
        that occurred. Use this to:
        - Track your trading performance
        - Update position information
        - Analyze market activity
        - etc.
        
        Args:
            trades: List of Trade objects representing all completed trades in the most recent bot turn
        
        Note:
            - Bot names are anonymized except for your own
            - Order IDs are hidden except for your own
        """
        for trade in trades:
            # Check if this bot participated in the trade (either as aggressor or resting order)
            # This is just a demonstration - replace with your actual trade processing logic
            if trade.agg_bot == self.name or trade.rest_bot == self.name:
                print("You did a trade!")