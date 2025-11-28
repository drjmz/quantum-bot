# ⚔️ Quantum Command: AI-Powered Twin Trading Bots

A high-frequency trading system comprising two distinct AI agents operating on the Base L2 network (Avantis Protocol).

![Dashboard Preview](https://imgur.com/wnsyIbC)
*(Replace this link with a screenshot of your actual dashboard)*

## 🤖 The Twin Engines

1.  **🛡️ Safe Bot (Quantum):**
    * **Strategy:** Conservative, trend-following, fib-based entries.
    * **Leverage:** 5x
    * **AI Logic:** Retrains hourly. Uses ML predictions for Stop Loss positioning.

2.  **🔥 Degen Bot (Turbo):**
    * **Strategy:** Aggressive momentum, liquidity wall scalping.
    * **Leverage:** 50x
    * **AI Logic:** "Full Send" mode. Ignores safety stops in favor of tight liquidation hunting.

## 🚀 Installation

### 1. Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* An Ethereum Wallet (Base Network) with ETH for gas/collateral.

### 2. Setup
Clone the repository and enter the directory:

```bash
git clone https://github.com/drjmz/quantum-bot.git
cd quantum-bot
```

Create your configuration file:

```bash
cp .env.example .env
```

**Edit the `.env` file** with your specific keys:
* `AVANTIS_PRIVATE_KEY`: Your wallet private key.
* `TELEGRAM_TOKEN`: Your bot token from @BotFather.
* `GOOGLE_API_KEY`: Your Gemini API key for AI analysis.

### 3. Launch
Start the engines:

```bash
docker-compose up --build -d
```

### 4. Monitor
* **Dashboard:** Open `http://localhost:8501` in your browser.
* **Logs:** Run `docker-compose logs -f` to see the AI brain in action.

## ⚠️ Disclaimer
**THIS SOFTWARE IS FOR EDUCATIONAL PURPOSES ONLY.**
Cryptocurrency trading, especially at 50x leverage, involves extreme risk. You can lose your entire balance in seconds. The developers are not responsible for financial losses. **Use `SIMULATION_MODE=True` first.**
