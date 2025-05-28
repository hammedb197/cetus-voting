"""Data fetcher for governance events with caching support."""

import os
import shutil
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from joblib import Memory
from datetime import datetime, timedelta
from functools import partial
from .sui_client import SuiClientWrapper
import logging

# Configure logger
logger = logging.getLogger('cetus_vote_dashboard.data_fetcher')

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache')
CACHE_SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB
CACHE_AGE_LIMIT = timedelta(minutes=5)  # Match the refresh interval

def clear_old_cache():
    """Clear cache files older than CACHE_AGE_LIMIT."""
    try:
        if not os.path.exists(CACHE_DIR):
            return
            
        now = datetime.now()
        cache_size = 0
        for root, _, files in os.walk(CACHE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                file_stat = os.stat(file_path)
                file_age = datetime.fromtimestamp(file_stat.st_mtime)
                cache_size += file_stat.st_size
                
                # Remove files older than age limit
                if now - file_age > CACHE_AGE_LIMIT:
                    try:
                        os.remove(file_path)
                        logger.debug(f"Removed old cache file: {file_path}")
                    except OSError as e:
                        logger.warning(f"Failed to remove cache file {file_path}: {e}")
                        
        # If cache size exceeds limit, clear all cache
        if cache_size > CACHE_SIZE_LIMIT:
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            logger.info(f"Cleared entire cache as size ({cache_size / 1024 / 1024:.2f} MB) exceeded limit ({CACHE_SIZE_LIMIT / 1024 / 1024:.2f} MB)")
            
    except Exception as e:
        logger.error(f"Error managing cache: {e}")

# Initialize cache with cleanup
clear_old_cache()
memory = Memory(CACHE_DIR, verbose=0)

# Constants for voting thresholds and timing
QUORUM_THRESHOLD = 0.50  # 50% of total stake must participate (excluding abstain)
VOTING_START = datetime(2024, 5, 27, 13, 0)  # May 27th, 1:00 PM PST
VOTING_END = datetime(2024, 6, 3, 11, 30)    # June 3rd, 11:30 AM PST
MIN_VOTING_DAYS = 2  # Minimum voting period before early completion
EARLY_END_THRESHOLD = 0.50  # 50% threshold for early completion

# Create a cached function outside the class
@memory.cache
def _cached_fetch_vote_events(
    client: SuiClientWrapper,
    start_time: datetime,
    end_time: datetime
) -> List[Dict[str, Any]]:
    """Cached implementation of fetch_vote_events."""
    events = []
    cursor = None
    
    # Convert timestamps to milliseconds
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    logger.debug(f"Fetching vote events from {start_time} to {end_time}")
    logger.debug(f"Timestamps in ms: {start_ms} to {end_ms}")
    
    while True:
        batch = client.get_events_by_module(
            module_name="governance_voting",
            event_type="ValidatorVoted",
            cursor=cursor,
            limit=100
        )
        
        logger.debug(f"Batch result: {batch}")
        
        if 'error' in batch or 'data' not in batch:
            logger.warning("No data in batch or error in response")
            break
            
        events.extend(batch['data'])
        logger.debug(f"Total events collected so far: {len(events)}")
        
        if not batch.get('hasNextPage'):
            logger.debug("No more pages")
            break
            
        cursor = batch.get('nextCursor')
        logger.debug(f"Next cursor: {cursor}")
        
    logger.info(f"Total events fetched: {len(events)}")
    return events

class GovernanceDataFetcher:
    """Handles fetching and caching of governance data."""
    
    def __init__(self) -> None:
        """Initialize the data fetcher with a Sui client."""
        self.client = SuiClientWrapper()
        self.cache_ttl = timedelta(minutes=5)
        clear_old_cache()  # Clear old cache on initialization
        
    def fetch_vote_events(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch voting events within a time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            
        Returns:
            List of voting events
        """
        return _cached_fetch_vote_events(self.client, start_time, end_time)
    
    def get_recent_votes(self, hours: int = 24) -> pd.DataFrame:
        """Get recent voting events.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            DataFrame containing recent votes
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        logger.info(f"Getting recent votes for the last {hours} hours")
        logger.debug(f"Time range: {start_time} to {end_time}")
        
        events = self.fetch_vote_events(start_time, end_time)
        
        if not events:
            logger.warning("No events found")
            return pd.DataFrame()
            
        df = pd.DataFrame(events)
        logger.info(f"Created DataFrame with {len(df)} rows")
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.debug("Converted timestamps to datetime")
        else:
            logger.warning("No timestamp column found in events")
        
        # Add validator names and update voting status
        voting_addresses = df['validator_address'].unique().tolist()
        self.client.update_voting_status(voting_addresses)
        df['validator_name'] = df['validator_address'].apply(self.client.get_validator_name)
        
        return df
    
    def analyze_voting_power(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze voting power distribution and impact.
        
        Args:
            df: DataFrame containing votes
            
        Returns:
            Dictionary containing voting power analysis
        """
        validator_map = self.client.get_validator_info_map()
        total_power = sum(int(info['voting_power']) for info in validator_map.values())
        
        # Calculate power by vote type
        vote_power = {'Yes': 0, 'No': 0, 'Abstain': 0}
        for _, row in df.iterrows():
            validator_info = validator_map.get(row['validator_address'], {})
            power = int(validator_info.get('voting_power', 0))
            vote_power[row['vote']] += power
        
        # Calculate total participating power (excluding abstain)
        participating_power = vote_power['Yes'] + vote_power['No']
        total_power_excl_abstain = total_power - vote_power['Abstain']
        
        # Calculate quorum and approval metrics
        quorum_reached = participating_power >= (total_power_excl_abstain * QUORUM_THRESHOLD)
        approval_reached = vote_power['Yes'] > vote_power['No'] if participating_power > 0 else False
        
        # Calculate time-based conditions
        now = datetime.utcnow()
        voting_duration = now - VOTING_START
        min_days_met = voting_duration >= timedelta(days=MIN_VOTING_DAYS)
        
        # Check if remaining votes could change outcome
        not_voted = [
            {
                'address': addr,
                'name': info['name'],
                'voting_power': int(info['voting_power']),
                'power_percentage': (int(info['voting_power']) / total_power * 100)
            }
            for addr, info in validator_map.items()
            if not info['has_voted']
        ]
        not_voted.sort(key=lambda x: x['voting_power'], reverse=True)
        remaining_power = sum(v['voting_power'] for v in not_voted)
        
        # Calculate if early completion is possible
        can_complete_early = (
            min_days_met and
            quorum_reached and
            approval_reached and
            remaining_power < abs(vote_power['Yes'] - vote_power['No'])
        )
        
        # Calculate voting status
        if now < VOTING_START:
            status = "Not Started"
        elif now > VOTING_END:
            status = "Ended"
        elif can_complete_early:
            status = "Can End Early"
        else:
            status = "In Progress"
        
        return {
            'total_power': total_power,
            'total_power_excl_abstain': total_power_excl_abstain,
            'vote_power': vote_power,
            'vote_power_percentage': {
                vote: (power / total_power_excl_abstain * 100) if total_power_excl_abstain > 0 else 0
                for vote, power in vote_power.items()
                if vote != 'Abstain'
            },
            'participating_power': participating_power,
            'quorum_threshold': total_power_excl_abstain * QUORUM_THRESHOLD,
            'quorum_reached': quorum_reached,
            'approval_reached': approval_reached,
            'remaining_power': remaining_power,
            'top_remaining_validators': not_voted[:5],  # Top 5 by voting power
            'voting_status': {
                'status': status,
                'start_time': VOTING_START,
                'end_time': VOTING_END,
                'min_days_met': min_days_met,
                'can_complete_early': can_complete_early,
                'time_remaining': (VOTING_END - now) if now < VOTING_END else timedelta(0)
            }
        }
    
    def get_voting_summary(self) -> Dict[str, Any]:
        """Get a summary of voting statistics.
        
        Returns:
            Dictionary containing voting statistics and validator information
        """
        # Get base statistics
        stats = self.client.get_voting_stats()
        validator_map = self.client.get_validator_info_map()
        
        # Get list of validators who haven't voted
        not_voted = [
            {
                'address': addr,
                'name': info['name'],
                'voting_power': info['voting_power']
            }
            for addr, info in validator_map.items()
            if not info['has_voted']
        ]
        
        # Sort non-voting validators by voting power (descending)
        not_voted.sort(key=lambda x: int(x['voting_power']), reverse=True)
        
        # Calculate total voting power and participation rate
        total_voting_power = sum(int(info['voting_power']) for info in validator_map.values())
        voted_power = sum(
            int(info['voting_power']) 
            for info in validator_map.values() 
            if info['has_voted']
        )
        
        # Calculate power concentration
        powers = [int(info['voting_power']) for info in validator_map.values()]
        power_concentration = self._calculate_power_concentration(powers)
        
        return {
            'total_validators': stats['total_validators'],
            'voted_count': stats['voted_count'],
            'not_voted_count': stats['not_voted_count'],
            'participation_rate': (stats['voted_count'] / stats['total_validators'] * 100) if stats['total_validators'] > 0 else 0,
            'voting_power_rate': (voted_power / total_voting_power * 100) if total_voting_power > 0 else 0,
            'total_voting_power': total_voting_power,
            'voted_power': voted_power,
            'not_voted_validators': not_voted,
            'power_concentration': power_concentration
        }
    
    def _calculate_power_concentration(self, powers: List[int]) -> Dict[str, float]:
        """Calculate voting power concentration metrics.
        
        Args:
            powers: List of voting powers
            
        Returns:
            Dictionary containing concentration metrics
        """
        if not powers:
            return {
                'gini_coefficient': 0.0,
                'top_10_percentage': 0.0,
                'herfindahl_index': 0.0
            }
            
        # Sort powers in ascending order for Gini coefficient
        powers = sorted(powers)
        total_power = sum(powers)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(powers)
        n = len(powers)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if n > 0 and cumsum[-1] > 0 else 0
        
        # Calculate top 10% concentration
        top_k = max(1, int(n * 0.1))  # At least 1 validator
        top_10_power = sum(sorted(powers, reverse=True)[:top_k])
        top_10_percentage = (top_10_power / total_power * 100) if total_power > 0 else 0
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        power_shares = [p / total_power for p in powers] if total_power > 0 else []
        hhi = sum(share * share for share in power_shares)
        
        return {
            'gini_coefficient': gini,
            'top_10_percentage': top_10_percentage,
            'herfindahl_index': hhi
        }
    
    def get_proposal_votes(self, proposal_id: int) -> pd.DataFrame:
        """Get all votes for a specific proposal.
        
        Args:
            proposal_id: ID of the proposal
            
        Returns:
            DataFrame containing votes for the proposal
        """
        # For now, we'll fetch all recent votes and filter
        # In a production system, we'd want to query specifically for the proposal
        df = self.get_recent_votes(hours=720)  # Last 30 days
        
        if df.empty:
            return df
            
        return df[df['proposal_id'] == proposal_id] 