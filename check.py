
import os
import pyotp
from dotenv import load_dotenv
from growwapi import GrowwAPI

load_dotenv()

api_key = os.environ["GROWW_TOTP_TOKEN"]
totp_secret = os.environ["GROWW_TOTP_SECRET"]

# Generate current TOTP
totp = pyotp.TOTP(totp_secret).now()

# Generate access token
access_token = GrowwAPI.get_access_token(
    api_key=api_key,
    totp=totp
)

# Initialize client
groww = GrowwAPI(access_token)

# Fetch available margin details
# margin = groww.get_available_margin_details()
# print("Available margin:", margin)

# # Fetch positions
# positions = groww.get_positions_for_user()
# print("Positions:", positions)

# Fetch holdings
holdings = groww.get_holdings_for_user()
print("Holdings:", type(holdings))
for i in holdings['holdings']:
    print(f"Stock: {i['trading_symbol']}, Quantity: {i['quantity']} ,Avg Price: {i['average_price']}")#) Current Price: {i['current_price']}")
