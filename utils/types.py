"""Type definitions for the Cetus Vote Dashboard."""

from typing import TypedDict, List, Dict, Optional
from datetime import datetime, timedelta

class ValidatorInfo(TypedDict):
    """Type definition for validator information."""
    name: str
    description: str
    image_url: str
    project_url: str
    voting_power: str
    commission_rate: str
    staking_pool_id: str
    has_voted: bool

class VoteEvent(TypedDict):
    """Type definition for vote events."""
    validator_address: str
    vote: str
    timestamp: int
    transaction_module: str
    event_type: str
    event_id: str
    sender: str

class VotingStatus(TypedDict):
    """Type definition for voting status."""
    status: str
    start_time: datetime
    end_time: datetime
    min_days_met: bool
    can_complete_early: bool
    time_remaining: timedelta

class VotingPowerAnalysis(TypedDict):
    """Type definition for voting power analysis."""
    total_power: int
    total_power_excl_abstain: int
    vote_power: Dict[str, int]
    vote_power_percentage: Dict[str, float]
    participating_power: int
    quorum_threshold: float
    quorum_reached: bool
    approval_reached: bool
    remaining_power: int
    top_remaining_validators: List[Dict[str, str]]
    voting_status: VotingStatus

class VotingSummary(TypedDict):
    """Type definition for voting summary."""
    total_validators: int
    voted_count: int
    not_voted_count: int
    participation_rate: float
    voting_power_rate: float
    total_voting_power: int
    voted_power: int
    not_voted_validators: List[Dict[str, str]]
    power_concentration: Dict[str, float]

class RPCResponse(TypedDict):
    """Type definition for RPC responses."""
    jsonrpc: str
    id: int
    result: Optional[Dict]
    error: Optional[Dict[str, str]] 