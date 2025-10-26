# Coinbase-Advanced Deep-Learning Trading Bot: A Deployable (End-to-end) Trading Strategy

![Coinbase Advanced - Trading Strategy Logs](/images/Coinbase_web_orders.png)

Looking to fully automate you’re cryptocurrency trading using state of the art machine-learning based methods. Well, this repository might be what you’re looking for. This repository provides all the requirements needed to fully automate a portfolio-based trading strategy that buys and sells cryptocurrency on the Coinbase exchange. The trading algorithm we use is inspired by the deep-learning based models in my MSc thesis [1] "Distribution-Driven Portfolio Optimisation using Deep-Learning". Importantly, we also include a multi-timeframe multi-hyperparameter model optimisation and a trading strategy fee-based backtest in the `backtest.ipynb` to give an idea of model performance and uncover optimal model settings, we can use to extract more excess return in Live markets. 

The rest of this README file will outline how to push this repository's contents to live markets and use the Telegram bot functionality for data extraction and direct strategy control. We will provide the steps for both local and remote deployment, using VS-code and Docker Desktop and VS Code, AWS, and Docker engine respectively. The latter is incase you want this strategy to run 24/7, 365.

I know the word demystify has become something of a buzzword (espically in the quantitative finance seen), largely inorder to increase content impressions. That said, I hope I can help "demystify" the world or algorithmic trading through this end-to-end pipeline from theory to construction to deployment. Cool ? Cool ! Let's get started.... 

## Disclaimers 

* Nothing contained in this repository or related content constitutes as financial advice.
* Crypotocurrency trading is very risky, and your capital is at risk.
* Do not Invest more than you are willing to loose.  
* Backtest results, are not indicative of future performance.
* Out of the box this strategy only works in Spot markets. (Future market compatibility in future updates).

## Getting Started:

Before we get started, It is importnat to note I am using macOS, so their will be a little variation in the terminal commands.

### Creating our Workspace
Create a new directory, from users, change into into directory, and check directory is empty  

```bash
mkdir ~/<directory_path>
cd ~/<directory_path>
ls -l
```
Clone this repository in your terminal.

```bash
git clone <this_repo_url>
```

### Defining Our Portfolio

For the stratgey to know what assets to buy and sell, you must edit this list `portfolio_assets = ['BTC-GBP', 'ETH-GBP', 'SOL-GBP']` in `market_algorithm.py`. 

### Setting up Coinbase Account and Downloading CDP API

You will require a Coinbase Advanced Account [2] and a valid ECDSA API key downloaded as a `.json` file. follow these steps ...

1. **Sign-up.** Create coinbase advanced account [2]

2. **Generate and Download your CDP key.** Navigate to the Coinbase Developer platform (CDP) $<$ "Dashboard" $<$ "API Keys" $<$ "Create API key". From here toggle "Advanced settings" and under the "Signature Algorithm" select the ECDSA option (very important or Coinbase Advanced will not receive our order messages). From here download you API key as a `.json` file into your created directory. 

3. **Fund your account.** Deposit your fiat account (GBP account) with fiat (You can deposit in other accounts but the market algorithm will only modify the holdings in this account if it is an element of your portfolio. If you have deposited in another account intending that to be your base account, sell all the holdings and transfer it to a GBP account). *Coinbase-Advanced has a minimum investment amount of £1 in all assets (UK), this is important and we will discuss more on this later*

### Telegram Bot 

![Telegram bot](/images/Telegram_bot.png)

To enable the telegram bot features, we need to download Telegram, note that documentation for telegram bot commands is provided in the telegram app. There are two options on how to use the bot.

* **Option 1.** The simpler and recommended option. Simply open Telegram and add the bot by typing `@coinbase_portfolio_bot` in the global search (which takes a few seconds). You can then connect your account by prompting the bot by calling `/link` and copy and paste your CDP key. *This method will only have full functionality if your trading strategy if running*

* **Option 2.** Duplicate the bot. By following [3] get an an API token, navigate to the `telegram_bot.py` file and copy and paste your HTTP API in `'TOKEN: Final = 'your_api_token'`. However, to use the bot we need to run it either locally or remotely (if you would like continuous bot functionality). This method is preffered if you would like to add functionality over the bot, otherwise you be creating a carbon copy of the original `@coinbase_portfolio_bot`.

Note, that no-one can prompt this bot and control your account (by e.g. closing all positions and shuting down your strategy by calling `/shutdown True`) unless they have your CDP API, so basically just do not share your API info, easy enough, fewwww !. At first you may think because I have logged in to my account my telegram bot is authenticated and polling telegram prompts from any user, however `context.user_data['cb_account'] = coinbase_account` is unique to each user, and therefore all orders which are methods of `RESTClient` will only triger on my account if your `context.user_data.get('cb_account)` is identical to mine, i.e. you must have my API key in-order to use bot functionality on my account.

## Running Locally Using Docker Desktop 

To ensure that all these files run on your machine, no matter what operating system you are using you need Docker Desktop [4]. The Dockerfile and `.yaml` file along with other files required to build images are provided in this repo, all we need to do now is just prompt in the VS-code terminal. 

*If you are using [**Option 1**](###-Telegram) then you will only need to build the image and conatiner for `market_algorithm.py`.*

For the telegram bot, `telegram_bot.py` (again skip if you have used [**Option 1**](###-Telegram)

```bash
docker build -t <image_name> .
dokcer run -d --restart always -p <host_port>:8000 --name <container_name> <image_name>
```

For the trading pipeline `market_algorithm.py`, the only differences are the volume mount `-v` so our container can modify our .db files, hots port (now 9000) and container name.

```bash
docker build -t <image_name> .
docker run -d --restart always -v "$(pwd):/app" -p <host_port>:8000 --name <container_name> <image_name>
```

To see the strategy in action, we can inspect the Docker logs by clicking on the containers in Docker Desktop. *Make sure to inspect the logs to double check the process is running without errors* 

## Running Remotelly using a Virtual-Private-Server (VPS): 

If you are reading this far down, odds are you probably want to run this strategy continuously. I mean the idea of a trading strategy running without intervention while you sleep is pretty cool ayy.

There are many different ways to deploy this strategy, we will cover two such methods, both using AWS. You will need an Amazon Web Services (AWS) account [5].

### AWS EC2 Windows Instance 

**1.** This the easiest of the provided options although is the most reasoruce-heavy/costy. We essentially replicate what we do locally on a Windows virtual machine. 

When creating your AWS EC2 instance [7] select a Windows Amazon Machine Image (AMI), and select a instance type with atleast 4 GiB Memory, these are the base system requirements for running Docker Desktop as of writing. When creating your instance you will be asked to create/re-use a key-pair, make sure to store this in a known location (preferably not your current directory), we will need this to decrypt our password that is required to log in as Administaror (defualt) on our virtual-machine.

If you’re using macOS, you will require Windows App [7] to run the `.rdp` file (which is the remote desktop).

**2.** Now we simply install VS code and docker desktop on our virtual-machine and repeat the steps in [Running Locally Using Docker Desktop](##-Running-Locally-Using-Docker-Desktop). 

### AWS EC2 Linux Ubuntu Distribution Instance

Start by selecting a Linux distribution, for example, Ubuntu. You can select the cheapest AMI, 1 GiB Memeory is all we need. The reason we can opt for a quarter of the memory is that we will be using Docker Engine. 

An added benefit of using this method is we can connect to our virtual machine via VS Code locally using SSH. Also note when creating this instance, you will be asked to create/re-use a key pair make sure you have the path to this key, we will need it later.

**1 Install Extensions.** Download the "Remote - SSH" Extension by Microsoft. 

**2 Create `.config` File.** Click the open remote window button in vs-code (bottom left) < "Connect to Host" < "Configure SSH Hosts" < `~/.ssh/config`. We need to modify this file by storing information about the server we are connecting to in order to connect to our machine. 

**4 Configure `.config`.** Navigate to AWS "Instances" < "Instances", select your instances and get your "Public IPv4 address". 

Then modify and save your config file as follows.

```bash 
Host <server_name>
    HostName <Public IPv4 address>
    User ubuntu
    Port 22
    IdentityFile <key_pair_path>
    StrictHostKeyChecking no
    ServerAliveInterval 60
```

To check if this works run `ssh <server_name>` in your terminal. If no errors print we have successfully connected to our virtual-machine.

We can continue in our local VS-code terminal, although it is probably easier to open a SSH host, to do this click "Open a Remote Window" < "Connect to Host" < "server_name". Now a VS Code pop up will open. 

**Install Docker Engine.** Run `docker --version`, the terminal will throw an error stating docker engine is not installed, just follow what the promot tells you to install. 

**5 Clone repositrory.** This step is identical to [Creating our Workspace](###-Creating-our-Workspace).

**6 Create Docker Images and Containers .** 

Now run the following command in the terminal. 

```bash
sudo docker build -t <image_name> .
sudo dokcer run -d --restart always -v .:/app -u "${UID}:${GID}" -p <host_port>:8000 --name <container_name> 
```

If you are using [**Option 2**](###-Telegram)), you again need to build two images and containers. After building the first container, edit the Dockerfile `CMD` by replacing `telegram_bot.py` with `market_algorithm.py`.

**Debug.** 

![Live Trading Logs](/images/Live_Strategy_logs.png)

Some useful Docker commands. 

```bash 
sudo docker ps # check images/containers 
sudo docker logs <container_name> # get script output for either container (especially useful for market_algorithm.py)
sudo docker rm -f <conatiner_name> # delete image and associated container 
sudo docker stop <conatiner_name> # stop container 
sudo docker start <conatiner_name> # start container
```

## Appendix

[1] - [My Thesis](https://github.com/AmjadSaidam/STAT0034-Research-Project-)

[2] - [Coinbase Create Account](https://login.coinbase.com/signup?action=signup&client_id=258660e1-9cfe-4202-9eda-d3beedb3e118&locale=en-gb&oauth_challenge=7de5c164-80d6-43d2-8a53-39cd7b935d3e)

[3] - [Create Telegram Bot](https://www.youtube.com/watch?v=vZtm1wuA2yc&t=1084s)

[4] - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

[5] - [Amazon Web Services (AWS)](https://aws.amazon.com/free/?trk=182ad523-8d86-462e-9a64-79a0d16df60c&sc_channel=ps&ef_id=CjwKCAjw6vHHBhBwEiwAq4zvA97XQWeb3PE17-nVysOi1uzCXlpyQKdL86bLKkwP0iIiU1vyQBCaYRoCq7AQAvD_BwE:G:s&s_kwcid=AL!4422!3!433803620870!e!!g!!aws%20account!9762827897!98496538463&gad_campaignid=9762827897&gbraid=0AAAAADjHtp9AVbE7LfnN5yGaWCk4xuH5j&gclid=CjwKCAjw6vHHBhBwEiwAq4zvA97XQWeb3PE17-nVysOi1uzCXlpyQKdL86bLKkwP0iIiU1vyQBCaYRoCq7AQAvD_BwE)

[6] - [Windows App](https://learn.microsoft.com/en-us/windows-app/overview)

[7] - [Creating an AWS Instance](https://www.youtube.com/watch?v=YH_DVenJHII&list=PLPhtRDeW6g6xaBMMUxZtJUpfIxmKcyU7g)