# Sui Governance Dashboard

A Streamlit dashboard for tracking governance voting events on the Sui blockchain for package `0x4eb9c090cd484778411c32894ec7b936793deaab69f114e9b47d07a58e8f5e5d`.

## Features

- Real-time monitoring of governance voting events
- Visualization of voting statistics and trends
- Validator participation tracking
- Quorum and approval progress monitoring
- Detailed validator activity tracking
- Automatic data updates every 5 minutes

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cetus-voting
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```bash
SUI_RPC_URL=https://fullnode.mainnet.sui.io
PACKAGE_ID=0x4eb9c090cd484778411c32894ec7b936793deaab69f114e9b47d07a58e8f5e5d
```

5. Run the dashboard:
```bash
streamlit run app.py
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .env
├── app.py                 # Main Streamlit application
└── utils/
    ├── __init__.py
    ├── sui_client.py     # Sui blockchain interaction
    ├── data_fetcher.py   # Event fetching and caching
    └── analytics.py      # Data analysis functions
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT 