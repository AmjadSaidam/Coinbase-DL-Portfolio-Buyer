# usefull links 
# telegram bot API: https://core.telegram.org/bots/api

# telegram commands 
from typing import Final # used to set constant types
from telegram import Update # deals with commands, sends HTTP request to telegarm  
from telegram.ext import Application, CommandHandler, ContextTypes
# others
import coinbase_order_functions as cb_trade
import numpy as np 
import json 
import datetime as dt 

# HTTP API and bot name
TOKEN: Final = 'your_api_token'
BOT_USERNAME: Final = '@coinbase_portfolio_bot'

# input api details command
async def account_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ 
    Function established link with user account given supplied with api name and api secret key  
    inputs must be space seperated
    """
    # behaviour of keys split will always be the same 
    if len(context.args) != 8:
        update.message.reply_text('provided incorrect CDP keys')

    # process message 
    res = context.args 
    api_name = res[0]
    api_secret = " ".join(res[1:]) # join with spaces
    api_secret = api_secret.replace("\\n", "\n")
    api_secret = repr(api_secret) # returns string in one line with '\n'

    # get account 
    coinbase_account = cb_trade.CoinbaseTrader(api_key = api_name, api_secret = api_secret) # get link 
    context.user_data['cb_account'] = coinbase_account # store class object globally so it can be accessed by other functions 
    coinbase_account.login() # check if login was successfull

    await update.message.reply_text(f'Acount link status: {coinbase_account.authenticated}') # send message

# get accounts command 
async def get_account(update: Update, context: ContextTypes.DEFAULT_TYPE):
    coinbase_account = context.user_data.get('cb_account')
    if coinbase_account is not None: 
        try:
            user_accounts = coinbase_account.get_user_accounts() # get account data 
            context.user_data['user_accounts'] = user_accounts # save aas global variable 
            text = f'succeffully pulled data {context.user_data.get('user_accounts')}'
        except Exception as e:
            text = f'account link establisded but failed to load account information, error message: {e}'
    else: 
        text = 'keys are invalid'

    await update.message.reply_text(text)

# json file for saving/loading shut down state 
def save_json(name, data: dict):
    with open(name, mode = 'w') as f: # file object returned by open is f
        return json.dump(obj = data, fp = f)

def load_json(file: str):
    with open(file) as f:
        return json.load(fp = f)

# close all open orders and/or shut down strategy
shut_down_state_file_name = 'shut_down_state.json'
async def shut_down_reactivate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = context.args[0] 
    shut_down = True if arg == 'True' else False if arg == 'False' else None 
    coinbase_account = context.user_data.get('cb_account')
    portfolio_assets = list(coinbase_account.get_user_accounts().keys())
    portfolio_assets = [asset + '-GBP' for asset in portfolio_assets] # add suffix, check if invested and close out position
    
    if coinbase_account is not None and not (shut_down == None):
        try:
            json_dict = {f'status': shut_down} # json boolean type is lower case 
            save_json(name = shut_down_state_file_name, data = json_dict)
            if shut_down:
                order = coinbase_account.multi_asset_close(
                    portfolio_tickers = portfolio_assets, 
                    full_close = True) # fully close position out emidiatelly 
            else:
                pass
        except Exception as e:
            await update.message.reply_text(f'failed to update strategy state: {e}')
        
        # context.user_data['shutdown'] = shut_down # save so can be used by other async functions
        text = f'strategy shut down: {order}' if shut_down else 'strategy reactivated and sending orders to market' if shut_down == False else None
    else: 
        text = 'order failed to send, check keys'

    await update.message.reply_text(text)

# full close function

# error function 
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

# help command 
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    hellper_text = (
        """
        Bot Commands
        
        /start - gets list of all available commands.
        /link - establishes link to all coinbase (must prompt first to use other functions) advanced account. To correctley propmt, provide your CDP key name and private key serparted with a space, example: "link <APIname> <APIPrivateKey>". Ensure you do not enclose name and key with quotation marks.
        /getinvestments - function returns all user accounts, including your base account, and lists the value invested in each in terms of the asset currency.
        /shut_down - function that hauts and reactivates the strategy exacution (can be toggled on or off), example: "closeordersandshutdown <[False, True]>", the first command toggles the strategy on/off. The effect is not emmediate, the prompt will exacuate in the next scheduled run of sending orders, after model training.
        """
    )
    
    await update.message.reply_text(hellper_text)

# run all functions 
def run_telegram_bot(): # asynchronous function, if imported must await function (inside an event loop) using asyncio 
    print('Starting Bot!')
    app = Application.builder().token(TOKEN).build()

    # command handlers
    app.add_handler(CommandHandler('start', help_command))
    app.add_handler(CommandHandler('link', account_link))
    app.add_handler(CommandHandler('getinvestments', get_account))
    app.add_handler(CommandHandler('shutdown', shut_down_reactivate))

    # messages (bot does not have handdle response functionality)

    # errors 
    app.add_error_handler(error)

    # poll the bot
    print('Polling, waiting for user command')
    app.run_polling(poll_interval = 1) # checks for new user commands every 1 second 

# run bot seperately in different terminal
if __name__ == "__main__":
    run_telegram_bot() 