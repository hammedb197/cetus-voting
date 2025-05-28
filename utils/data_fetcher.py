"""Data fetcher for governance events."""

import os
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .sui_client import SuiClientWrapper
from .types import VoteEvent, VotingPowerAnalysis, VotingSummary
from pysui.sui.sui_types.collections import EventID
from .constants import (
    QUORUM_THRESHOLD,
    VOTING_START,
    VOTING_END,
    MIN_VOTING_DAYS,
    EARLY_END_THRESHOLD,
    VOTE_TYPES
)
import logging

# Configure logger
logger = logging.getLogger('cetus_vote_dashboard.data_fetcher')

def _fetch_vote_events(
    client: SuiClientWrapper,
    start_time: datetime,
    end_time: datetime
) -> List[Dict[str, Any]]:
    """Real-time implementation of fetch_vote_events."""
    events = []
    cursor = None
    
    # Convert timestamps to milliseconds
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    logger.debug(f"Fetching vote events from {start_time} to {end_time}")
    logger.debug(f"Timestamps in ms: {start_ms} to {end_ms}")
    
    while True:
        try:
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
                
            # Get the next cursor
            next_cursor = batch.get('nextCursor')
            if next_cursor:
                try:
                    # Create EventID with the correct parameter order
                    if isinstance(next_cursor, dict):
                        event_seq = str(next_cursor.get('eventSeq', '0'))
                        tx_seq = str(next_cursor.get('txDigest', ''))
                        cursor = EventID(
                            event_seq=event_seq,  # First parameter: event sequence
                            tx_seq=tx_seq  # Second parameter: transaction sequence
                        )
                        logger.debug(f"Created cursor with event_seq: {event_seq}, tx_seq: {tx_seq}")
                    elif isinstance(next_cursor, str):
                        cursor = next_cursor
                    else:
                        cursor = next_cursor
                    logger.debug(f"Created cursor: {cursor}")
                except Exception as e:
                    logger.error(f"Error creating cursor: {str(e)}")
                    logger.error(f"Cursor data: {next_cursor}")
                    break
            else:
                break
                
        except Exception as e:
            logger.error(f"Error fetching events: {str(e)}")
            logger.error(f"Full error details: {str(e)}")
            break
    
    logger.info(f"Total events fetched: {len(events)}")
    return events

class GovernanceDataFetcher:
    """Handles fetching of governance data."""
    
    def __init__(self) -> None:
        """Initialize the data fetcher with a Sui client."""
        try:
            self.client = SuiClientWrapper()
            self._validator_info = None
            self._last_validator_update = None
        except ValueError as e:
            logger.error(f"Failed to initialize SuiClientWrapper: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing data fetcher: {str(e)}")
            raise
        
    def _get_validator_info(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get validator info, refreshing only when needed."""
        now = datetime.utcnow()
        if (force_refresh or 
            not self._validator_info or 
            not self._last_validator_update or 
            (now - self._last_validator_update) > timedelta(minutes=5)):
            
            self._validator_info = self.client.get_validator_info_map()
            self._last_validator_update = now
            
        return self._validator_info
        
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
        return _fetch_vote_events(self.client, start_time, end_time)
    
    def get_recent_votes(self, hours: Optional[int] = None) -> pd.DataFrame:
        """Get voting events.
        
        Args:
            hours: Optional number of hours to look back. If None, fetches all votes.
            
        Returns:
            DataFrame containing votes
        """
        if hours is not None:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
        else:
            # For all votes, use the voting period start and end times
            start_time = VOTING_START
            end_time = VOTING_END
        
        logger.info(f"Getting votes from {start_time} to {end_time}")
        logger.debug(f"Time range: {start_time} to {end_time}")
        
        events = self.fetch_vote_events(start_time, end_time)
        
        if not events:
            logger.warning("No events found")
            return pd.DataFrame()
            
        df = pd.DataFrame(events)
        logger.info(f"Created DataFrame with {len(df)} rows")
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            logger.debug("Converted timestamps to datetime")
        else:
            logger.warning("No timestamp column found in events")
        
        # Add validator names and update voting status
        voting_addresses = df['validator_address'].unique().tolist()
        validator_info = self._get_validator_info(force_refresh=True)
        
        # Update voting status in validator info
        for addr in validator_info:
            validator_info[addr]['has_voted'] = addr in voting_addresses
            
        # Add validator names to DataFrame
        df['validator_name'] = df['validator_address'].apply(
            lambda x: validator_info.get(x, {}).get('name', f"{x[:6]}...{x[-4:]}")
        )
        
        return df
    
    def analyze_voting_power(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze voting power distribution and impact.
        
        Args:
            df: DataFrame containing votes
            
        Returns:
            Dictionary containing voting power analysis
        """
        validator_info = self._get_validator_info()
        total_power = sum(int(info['voting_power']) for info in validator_info.values())
        
        # Calculate power by vote type
        vote_power = {'Yes': 0, 'No': 0, 'Abstain': 0}
        for _, row in df.iterrows():
            validator = validator_info.get(row['validator_address'], {})
            power = int(validator.get('voting_power', 0))
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
            for addr, info in validator_info.items()
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
        validator_info = self._get_validator_info()
        
        # Calculate statistics directly from validator info
        total_validators = len(validator_info)
        voted_count = sum(1 for info in validator_info.values() if info['has_voted'])
        
        # Get list of validators who haven't voted
        not_voted = [
            {
                'address': addr,
                'name': info['name'],
                'voting_power': info['voting_power']
            }
            for addr, info in validator_info.items()
            if not info['has_voted']
        ]
        
        # Sort non-voting validators by voting power (descending)
        not_voted.sort(key=lambda x: int(x['voting_power']), reverse=True)
        
        # Calculate total voting power and participation rate
        total_voting_power = sum(int(info['voting_power']) for info in validator_info.values())
        voted_power = sum(
            int(info['voting_power']) 
            for info in validator_info.values() 
            if info['has_voted']
        )
        
        # Calculate power concentration
        powers = [int(info['voting_power']) for info in validator_info.values()]
        power_concentration = self._calculate_power_concentration(powers)
        
        return {
            'total_validators': total_validators,
            'voted_count': voted_count,
            'not_voted_count': total_validators - voted_count,
            'participation_rate': (voted_count / total_validators * 100) if total_validators > 0 else 0,
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
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
 
 
 