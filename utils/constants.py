"""Constants and configuration values for the Cetus Vote Dashboard."""

from datetime import datetime, timedelta
import os
import tempfile

# Cache directory configuration
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'cetus_vote_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache configuration
CACHE_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB
CACHE_AGE_LIMIT = timedelta(minutes=5)  # Match the refresh interval
CACHE_TTL = timedelta(minutes=5)

# Voting thresholds and timing
QUORUM_THRESHOLD = 0.50  # 50% of total stake must participate (excluding abstain)
VOTING_START = datetime(2024, 5, 27, 13, 0)  # May 27th, 1:00 PM PST
VOTING_END = datetime(2024, 6, 3, 11, 30)    # June 3rd, 11:30 AM PST
MIN_VOTING_DAYS = 2  # Minimum voting period before early completion
EARLY_END_THRESHOLD = 0.50  # 50% threshold for early completion

# RPC configuration
DEFAULT_RPC_URL = 'https://fullnode.mainnet.sui.io'
DEFAULT_GOVERNANCE_OBJECT_ID = '0x20f7aad455b839a7aec3be11143da7c7b6b481bfea89396424ea1eac02209e7a'
RPC_BATCH_LIMIT = 100
RPC_RETRY_ATTEMPTS = 3
RPC_RETRY_DELAY = 1  # seconds

# Vote types
VOTE_TYPES = ['Yes', 'No', 'Abstain']

# Colors for visualizations
VOTE_COLORS = {
    'Yes': '#00C853',    # Bright green
    'No': '#FF1744',     # Bright red
    'Abstain': '#78909C' # Blue-grey
}

PIE_CHART_COLORS = ['#2ecc71', '#e74c3c', '#95a5a6', '#bdc3c7']  # Yes, No, Abstain, Not Voted 