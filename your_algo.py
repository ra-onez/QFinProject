"""
Trading bot implementation for the QFinProject simulation.

This module defines a ``PlayerAlgorithm`` class that attempts to act as a
simple market‑making strategy.  The bot places resting buy and sell orders
around the current midprice and adjusts its quoting behaviour as its
position changes.  Positions are tracked per product and used to skew the
bid/ask prices and reduce order sizes when approaching the position limit.

The exchange calls two methods during the simulation:

* ``send_messages`` – called on every turn with the current orderbook.
  The bot returns a list of messages (orders or cancellations) to send
  to the exchange.  This implementation cancels any outstanding resting
  orders before placing new quotes, then calculates a midprice and places
  symmetric bid/ask orders around it.
* ``process_trades`` – called after trades occur.  The bot can use this
  method to update internal state such as position and remove orders that
  were filled.

The default strategy defined here is meant to serve as a starting point
for competition participants.  Feel free to experiment with the quoted
spread, size, skew factor, and other parameters to optimise performance.
"""

from typing import List, Dict

from base import Exchange, Trade, Order, Product, Msg, Rest


class PlayerAlgorithm:
    """
    Market‑making trading algorithm for the QFinProject competition.

    This class implements a straightforward market‑making approach that
    continuously quotes both sides of the market.  It tracks its open
    orders and positions so that it can adjust quoting behaviour as
    inventory changes and avoid exceeding position limits.  The key
    parameters controlling the strategy are the base spread (distance
    between the bid and ask) and the skew factor (how much to bias
    quotes when long or short).  A small order size is used by default
    and is further reduced as the position approaches the allowed limit.
    """

    def __init__(self, products: List[Product]):
        """
        Initialise the trading algorithm with available products and
        configuration.  At construction time we set up dictionaries to
        hold our position and outstanding orders for each product.  We
        also record useful metadata such as the minimum price variance
        (tick size) and position limits from the ``Product`` objects.

        Args:
            products: List of all tradable products available in the
                current round.  Each ``Product`` provides metadata such
                as ``ticker``, ``pos_limit`` and ``mpv``.
        """
        self.products = products
        # Give the bot a descriptive name to appear in trade logs
        self.name: str = "MarketMakerBot"
        # Counter to track number of timesteps completed
        self.timestamp_num: int = 0
        # Set starting order id – this will be overwritten via set_idx
        self.idx: int = 0
        # Track open resting orders: order_id → (direction, size, price)
        self.open_orders: Dict[int, tuple] = {}
        # Track position per ticker: ticker → net position (long positive,
        # short negative)
        self.position: Dict[str, int] = {p.ticker: 0 for p in products}
        # Store position limits per ticker; if None, treat as unbounded
        self.pos_limits: Dict[str, float] = {
            p.ticker: (p.pos_limit if p.pos_limit is not None else float("inf"))
            for p in products
        }
        # Store tick sizes (minimum price variance) per ticker
        self.mpv: Dict[str, float] = {p.ticker: (p.mpv if p.mpv else 1.0) for p in products}

        # Parameters controlling quoting behaviour
        self.base_spread: float = 1.0  # total width between bid and ask
        self.skew_factor: float = 0.5  # controls how strongly to skew quotes based on position
        self.default_price: float = 1000.0  # fallback price if no market data exists
        self.default_order_size: int = 5  # base order size

    def set_idx(self, idx: int) -> None:
        """
        Set the starting order id for the current trading session.  Each
        message we send must carry a unique order id.  The exchange
        environment will call this before the first call to ``send_messages``.

        Args:
            idx: Starting integer for order IDs.  We will increment
                 ``self.idx`` after each order.
        """
        self.idx = idx

    def create_order(self, ticker: str, size: int, price: float, direction: str) -> Msg:
        """
        Construct a new order message wrapped in a ``Msg`` object.

        Args:
            ticker: Product symbol to trade (e.g. "UEC").
            size: Number of units to trade (positive integer).
            price: Price per unit (float).  Will be passed directly to the
                ``Order`` object.
            direction: Trade direction, either "Buy" or "Sell".

        Returns:
            Msg: Message object containing the new order for the exchange.
        """
        order_idx = self.idx
        new_order = Order(ticker=ticker, price=price, size=size,
                          order_id=order_idx, agg_dir=direction, bot_name=self.name)
        message = Msg("ORDER", new_order)
        self.idx += 1
        return message

    def remove_order(self, order_idx: int) -> Msg:
        """
        Construct a cancel message for an existing resting order.

        Args:
            order_idx: Unique ID of the order to cancel.

        Returns:
            Msg: Message requesting order cancellation.
        """
        return Msg("REMOVE", order_idx)

    def send_messages(self, book: Dict[str, Dict[str, List[Rest]]]) -> List[Msg]:
        """
        Main trading logic called on each turn.  The bot analyses the
        current orderbook, cancels any of its existing resting orders
        and places new quotes around the midprice.  It returns a list
        of messages to be sent to the exchange.

        Args:
            book: Complete order book structure.  It is a mapping from
                ticker → {"Bids": [Rest], "Asks": [Rest]} where each
                ``Rest`` object describes a resting order.

        Returns:
            List of ``Msg`` objects containing orders and/or cancellations.
        """
        messages: List[Msg] = []

        # First cancel any previously placed resting orders.  We do this
        # every turn to ensure our quotes are refreshed at the new prices.
        for order_id in list(self.open_orders.keys()):
            messages.append(self.remove_order(order_id))
            del self.open_orders[order_id]

        # For each product, compute a midprice and place symmetric bids and asks
        for product in self.products:
            ticker = product.ticker
            # Extract the best bid and ask from the book if available
            bids: List[Rest] = book.get(ticker, {}).get("Bids", [])
            asks: List[Rest] = book.get(ticker, {}).get("Asks", [])
            if bids and asks:
                best_bid = bids[0].price
                best_ask = asks[0].price
                mid = (best_bid + best_ask) / 2.0
            elif bids:
                # If only bids exist we set mid to the best bid
                mid = bids[0].price
            elif asks:
                # If only asks exist we set mid to the best ask
                mid = asks[0].price
            else:
                # If no market exists we fall back to a default price
                mid = self.default_price

            # Compute position‑based skew.  As our net position grows, we
            # bias quotes to encourage mean reversion towards flat.  When
            # long (positive), skew downward so that we sell at higher prices
            # and buy less aggressively; when short, skew upward.
            pos = self.position.get(ticker, 0)
            limit = self.pos_limits.get(ticker, float("inf"))
            # Avoid divide by zero; if no limit treat skew as zero
            skew = 0.0
            if limit and limit != float("inf"):
                skew = (pos / limit) * self.skew_factor

            # Calculate bid/ask quotes around the midprice
            half_spread = self.base_spread / 2.0
            buy_price = mid - half_spread - skew
            sell_price = mid + half_spread - skew

            # Round to tick size for this product
            tick = self.mpv.get(ticker, 1.0)
            buy_price = round(buy_price / tick) * tick
            sell_price = round(sell_price / tick) * tick

            # Determine order size.  Reduce size as we approach the position limit
            # to avoid incurring fines.  Base size is set via default_order_size.
            size = self.default_order_size
            # If near the boundary (e.g. above 80 % of limit) use smaller size
            if limit != float("inf") and limit > 0:
                if abs(pos) > 0.8 * limit:
                    size = max(1, size // 2)
                if abs(pos) > 0.9 * limit:
                    size = 1

            # Create buy order
            buy_msg = self.create_order(ticker, size, buy_price, "Buy")
            messages.append(buy_msg)
            # Record the new resting order
            self.open_orders[buy_msg.message.order_id] = ("Buy", size, buy_price)

            # Create sell order
            sell_msg = self.create_order(ticker, size, sell_price, "Sell")
            messages.append(sell_msg)
            self.open_orders[sell_msg.message.order_id] = ("Sell", size, sell_price)

        # Increment our internal timestamp counter
        self.timestamp_num += 1
        return messages

    def process_trades(self, trades: List[Trade]) -> None:
        """
        Process the list of completed trades returned by the exchange.  This
        method updates the bot's position and removes filled resting orders
        from the internal tracking dictionary.  It is called after
        ``send_messages`` when trades have occurred.

        Args:
            trades: List of ``Trade`` objects describing executed trades.
        """
        for trade in trades:
            ticker = trade.ticker
            size = trade.size
            # If we were the aggressor (sent the order that crossed the book)
            if trade.agg_bot == self.name:
                # Update position based on the direction of our aggressive order
                if trade.agg_dir == "Buy":
                    self.position[ticker] = self.position.get(ticker, 0) + size
                else:
                    self.position[ticker] = self.position.get(ticker, 0) - size
                # Remove from open orders if it still exists
                if trade.agg_order_id in self.open_orders:
                    del self.open_orders[trade.agg_order_id]

            # If we were the resting order
            if trade.rest_bot == self.name:
                # The side we executed is opposite of the aggressor direction
                # (e.g. if the aggressor bought, we sold)
                if trade.agg_dir == "Buy":
                    # We sold, so subtract position
                    self.position[ticker] = self.position.get(ticker, 0) - size
                else:
                    # We bought, so add position
                    self.position[ticker] = self.position.get(ticker, 0) + size
                if trade.rest_order_id in self.open_orders:
                    del self.open_orders[trade.rest_order_id]

        # Nothing to return; state updates are stored on the object
