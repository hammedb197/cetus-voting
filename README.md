# Cetus Recovery Vote Dashboard

A real-time dashboard for tracking the Cetus protocol upgrade vote progress. This dashboard monitors validator participation and voting power distribution for the proposal to return funds frozen from the Cetus hack.

## Features

- Real-time vote tracking
- Validator participation monitoring
- Voting power distribution analysis
- Auto-refresh every 5 minutes
- Detailed analytics and visualizations
- Caching system for efficient data retrieval

## Requirements

- Python 3.8+
- Streamlit
- pandas
- plotly
- pysui
- python-dotenv

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hammedb197/cetus-voting.git
cd cetus-voting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```env
SUI_RPC_URL=https://fullnode.mainnet.sui.io
PACKAGE_ID=your_package_id
```

## Usage

Run the dashboard:
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`.

## Project Structure

```
cetus-vote/
├── app.py                 # Main Streamlit application
├── utils/
│   ├── __init__.py       # Package initialization
│   ├── analytics.py      # Analytics and visualization logic
│   ├── constants.py      # Configuration constants
│   ├── data_fetcher.py   # Data fetching and caching
│   ├── sui_client.py     # Sui blockchain client wrapper
│   └── types.py          # Type definitions
├── .env                  # Environment variables
└── README.md            # This file
```

## Key Components

- **VotingAnalytics**: Handles data processing and visualization
- **GovernanceDataFetcher**: Manages data retrieval and caching
- **SuiClientWrapper**: Interfaces with the Sui blockchain

## Vote Criteria

- Voting Period: May 27th 1:00 PM PST - June 3rd 11:30 AM PST
- Early Completion Requirements:
  - Minimum 2 days of voting completed
  - >50% of total stake participated (excluding abstain)
  - "Yes" votes exceed sum of "No" votes and remaining non-participating stake
- Quorum: >50% of total stake must participate
- Approval: "Yes" votes must exceed "No" votes

## Cache Management

The dashboard implements a caching system with:
- 5-minute cache expiry
- 10MB size limit
- Automatic cleanup of old cache files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 