"""
telegarm bot for live startegy and account GET/POST commands, intended for per user hosting

bot commands exlcuiding shutdown will work if you query other bots, but for live-trading functionality bot must be cloned

their is no risk of collision depsite identical codebase, tokens, user_id and context.user_data ensure this

telegram bot API: https://core.telegram.org/bots/api
"""
import os
from dotenv import load_dotenv; load_dotenv()
from collections import defaultdict
import json 
from typing import Final # used to set constant types
# telegram commands 
from telegram import Update # deals with commands, sends HTTP request to telegarm  
from telegram.ext import Application, CommandHandler, ContextTypes
# .py
from live_trading.coinbase_order_functions import CoinbaseTrader as coin_trade
import live_trading.live_errors as coin_error

def _asset_universe():
    """lazy import, live_loop imports this module so a top-level import would be circular"""
    from live_trading.live_loop import _asset_universe as universe
    return universe()

# HTTP API and bot name
TOKEN: Final = os.environ.get('TELEGRAM_TOKEN')
OWNER_ID: Final = os.environ.get('TELEGRAM_OWNER_ID') # telegram user id allowed to toggle the shared live-strategy shut down state

# --- ACCOUNTS ---
# input api details command
async def link(update: Update, 
               context: ContextTypes.DEFAULT_TYPE):
    """Function established link with user account given supplied with (space seperated) api name and api secret key"""
    # behaviour of keys split will always be the same
    if len(context.args) != 8:
        await update.message.reply_text('provided incorrect CDP keys')
        return

    # process message 
    res = context.args 
    api_name = res[0]
    api_secret = " ".join(res[1:]) # join with spaces
    api_secret = api_secret.replace("\\n", "\n")
    api_secret = repr(api_secret) # returns string in one line with '\n'

    # get account
    coinbase_account: coin_trade = coin_trade(api_key = api_name, api_secret = api_secret) # get link
    try:
        coinbase_account.login() # check if login was successfull
    except coin_error.CoinLoginError as e:
        await update.message.reply_text(f'login error:\n{e}')
        return
    except Exception as e:
        await update.message.reply_text(f'unkown error:\n{e}')
        return

    # save coinbase class only once login succeeds, scoped per telegram user
    context.user_data['cb_account'] = coinbase_account
    await update.message.reply_text(f'account link status: {coinbase_account.authenticated}') # send message

# get accounts command 
async def get_account(update: Update, 
                      context: ContextTypes.DEFAULT_TYPE):
    """"""
    coinbase_account: coin_trade = context.user_data.get('cb_account')
    if coinbase_account is not None:
        try:
            user_accounts = coinbase_account.get_user_accounts() # get account data
            context.user_data['user_accounts'] = user_accounts # save per-user
            text = f"succeffully pulled data:\n{user_accounts}"
        except coin_error.CoinLoginError as e:
            text = f'login error:\n{e}'
        except Exception as e:
            text = f'unkown error:\n{e}'
    else:
        text = 'keys are invalid'

    await update.message.reply_text(text)

# --- METADATA --- 
async def get_asset_meta_data(update: Update,
                              context: ContextTypes.DEFAULT_TYPE):
    """gets given asset metadata, usage: "/get_asset_meta_data <BASE-QUOTE>" """
    coinbase_account: coin_trade = context.user_data.get('cb_account')
    if coinbase_account is None:
        await update.message.reply_text("account login not yet established, call '/link'")
        return
    if not context.args:
        await update.message.reply_text('missing argument, usage: "/get_asset_meta_data <BASE-QUOTE>"')
        return

    asset = context.args[0].upper() # asset
    keys = ['price', 'price_percentage_change_24h', 'volume_24h', 'low_24h', 'high_24h']
    res = {}
    try:
        # data
        asset_metadata_payload = coinbase_account.client.get_product(product_id = asset)
        for k in keys:
            asset_metadata = str(getattr(asset_metadata_payload, k))
            res[k] = float(asset_metadata.split('%')[0]) # drop '%' if present
        text = '\n'.join(f'{k}: {v}' for k, v in res.items())
    except Exception as e:
        text = f'{asset} query error:\n{e}'

    await update.message.reply_text(text)

async def get_portfolio_meta_data(update: Update, 
                                  context: ContextTypes.DEFAULT_TYPE):
    """gets portfolio metadata"""
    coinbase_account: coin_trade = context.user_data.get('cb_account')
    if coinbase_account is None:
        await update.message.reply_text("account login not yet established, call '/link'")
        return
    asset_portfolio = _asset_universe()

    res = defaultdict(dict)
    try:
        user_accounts = coinbase_account.get_user_accounts()
        total_value = coinbase_account.total_portfolio_value() # once, not per asset
        for a in asset_portfolio:
            res[a]['base_value'] = user_accounts.get(coinbase_account.get_base(a), 0.0) # accounts are keyed by BASE, e.g. 'BTC'
            res[a]['currency_value'] = coinbase_account.base_to_quote(user_accounts, a)
            res[a]['portfolio_weight'] = res[a]['currency_value'] / total_value if total_value else 0.0
        text = '\n'.join(f'{a}: {v}' for a, v in res.items())
    except Exception as e:
        text = f'query error:\n{e}'

    await update.message.reply_text(text)

# --- POLLING AND SHUTDOWN ---
# json file for saving/loading shut down state 
def save_json(name: str, 
              data: dict):
    """write json file to directory"""
    with open(name, mode = 'w') as f: # w = writing mode
        return json.dump(obj = data, fp = f)

def load_json(file: str):
    """read json file to directory"""
    with open(file) as f:
        return json.load(fp = f)

shut_down_state_file_name = 'shut_down_state.json' # persisted readable/writable variable 
async def shutdown(update: Update,
                   context: ContextTypes.DEFAULT_TYPE):
    """hault and resume live strategy"""
    if not context.args:
        await update.message.reply_text('missing argument, usage: "/shutdown <True/False>"')
        return

    arg = context.args[0]
    shut_down = True if (arg == 'True') else False if (arg == 'False') else None
    if shut_down is None:
        await update.message.reply_text('invalid argument, usage: "/shutdown <True/False>"')
        return

    # only the bot owner may toggle the shared live-strategy state file,
    # otherwise any telegram user messaging this bot could pause/resume the operator's strategy
    if OWNER_ID is None or str(update.effective_user.id) != OWNER_ID:
        await update.message.reply_text('not authorized to change the live strategy state')
        return

    # successful authorisation enables to write shut_down
    json_dict = {'status': shut_down} # json boolean type is lower case
    save_json(name = shut_down_state_file_name, data = json_dict) # write updated strategy on/off command

    await update.message.reply_text(f'strategy polling paused: {shut_down}')

async def close_all(update: Update,
                    context: ContextTypes.DEFAULT_TYPE):
    """liquidates all positions in portfolio"""
    coinbase_account: coin_trade = context.user_data.get('cb_account')
    if coinbase_account is None:
        await update.message.reply_text("account login not yet established, call '/link'")
        return

    confirmed = context.args == ['confirm', 'confirm']
    if not confirmed:
        await update.message.reply_text("invalid confermation message, type '/close_all confirm confirm' to liquidate portfolio")
        return

    try:
        asset_universe = _asset_universe()
        order = coinbase_account.multi_asset_close(portfolio_tickers = asset_universe, full_close = True)
        order_msg = f'positions closed: {order}'
    except coin_error.CoinLoginError as e:
        order_msg = f'loggin error:\n{e}'
    except coin_error.CoinOrderError as e:
        order_msg = f'order error:\n{e}'
    except Exception as e:
        order_msg = f'unkown error:\n{e}'

    await update.message.reply_text(order_msg)

#  --- ERROR HANDELLING ---
async def error(update: Update, 
                context: ContextTypes.DEFAULT_TYPE):
    """error handelling function"""
    print(f'Update {update} caused error {context.error}')

# --- BOT HELP ---
# help command 
async def help(update: Update):
    """function overview helper"""
    hellper_text = (
        """
        Bot Commands
        
        /help - gets list of all available commands.

        /link - establishes link to all coinbase (must prompt first to use other functions) advanced account. To correctley propmt, provide your CDP key name and private key serparted with a space, example: "link <APIname> <APIPrivateKey>". Ensure you do not enclose name and key with quotation marks.

        /get_account - function returns all user accounts, including your base account, and lists the value invested in each in terms of the asset currency.

        /get_asset_meta_data - function gets given asset metadata 

        /get_portfolio_meta_data - function gets portfolio specific metadata

        /shutdown - function that hauts and reactivates the strategy exacution (can be toggled on or off), example: "/shutdown <[False, True]>", the first command toggles the strategy on/off. The effect is not emmediate, the prompt will exacuate in the next scheduled run of sending orders, after model training. Restricted to the bot owner.

        /close_all - immediately liquidates all your linked positions back to your base currency, example: "/close_all confirm confirm".
        """
    )
    
    await update.message.reply_text(hellper_text)

# --- RUNS BOT ---
# run all functions 
def run_telegram_bot(): # asynchronous function, if imported must await function (inside an event loop) using asyncio 
    """runs telegran bot"""
    print('Starting Bot!')
    app = Application.builder().token(TOKEN).build()

    # command handlers
    app.add_handler(CommandHandler('help', help))
    app.add_handler(CommandHandler('link', link))
    app.add_handler(CommandHandler('get_account', get_account))
    app.add_handler(CommandHandler('get_asset_meta_data', get_asset_meta_data))
    app.add_handler(CommandHandler('get_portfolio_meta_data', get_portfolio_meta_data))
    app.add_handler(CommandHandler('shutdown', shutdown))
    app.add_handler(CommandHandler('close_all', close_all))

    # messages (bot does not have handdle response functionality)

    # errors 
    app.add_error_handler(error)

    # poll the bot
    print('Polling, waiting for user command')
    app.run_polling(poll_interval = 1) # checks for new user commands every 1 second 

# run bot seperately in different terminal
if __name__ == "__main__":
    run_telegram_bot() 