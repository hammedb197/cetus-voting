"""Sui client wrapper for interacting with the Sui blockchain."""

import os
import json
import requests
import time
from typing import List, Dict, Any, Optional
from pysui.sui.sui_clients.sync_client import SuiClient
from pysui.sui.sui_types.address import SuiAddress
from pysui.sui.sui_types.collections import SuiMap
from pysui.sui.sui_config import SuiConfig
from pysui.sui.sui_types.event_filter import MoveEventModuleQuery
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from .types import ValidatorInfo, VoteEvent, RPCResponse
from .constants import (
    DEFAULT_RPC_URL,
    DEFAULT_GOVERNANCE_OBJECT_ID,
    RPC_BATCH_LIMIT,
    RPC_RETRY_ATTEMPTS,
    RPC_RETRY_DELAY,
    CACHE_TTL
)

# Configure logger
logger = logging.getLogger('cetus_vote_dashboard.sui_client')

load_dotenv()

class RPCError(Exception):
    """Custom exception for RPC errors."""
    pass

class SuiClientWrapper:
    """Wrapper for the Sui client to handle governance-related queries."""
    
    def __init__(self) -> None:
        """Initialize the Sui client with configuration from environment."""
        self.rpc_url = os.getenv('SUI_RPC_URL', DEFAULT_RPC_URL)
        self.package_id = os.getenv('PACKAGE_ID')
        logger.info(f"Initialized SuiClientWrapper with RPC URL: {self.rpc_url}")
        logger.info(f"Package ID: {self.package_id}")
        self.governance_object_id = DEFAULT_GOVERNANCE_OBJECT_ID
        config = SuiConfig.user_config(rpc_url=self.rpc_url)
        self.client = SuiClient(config)
        self._validator_info_cache: Dict[str, ValidatorInfo] = {}
        self._last_cache_update: Optional[datetime] = None
        self._cache_ttl = CACHE_TTL
        
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make a JSON-RPC call to the Sui network.
        
        Args:
            method: The RPC method name
            params: List of parameters for the method
            
        Returns:
            The JSON response from the RPC call
        """
        try:
            response = requests.post(
                self.rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params
                },
                headers={
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                logger.error(f"RPC error: {result['error']}")
                return {}
                
            return result.get("result", {})
            
        except Exception as e:
            logger.error(f"Error making RPC call: {str(e)}")
            return {}
        
    def get_validator_info_map(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get a mapping of validator addresses to their information.
        
        Args:
            force_refresh: Whether to force a refresh of the cache
            
        Returns:
            Dictionary mapping validator addresses to their information
        """
        now = datetime.utcnow()
        
        # Check if we need to refresh the cache
        if (force_refresh or 
            not self._validator_info_cache or 
            not self._last_cache_update or 
            now - self._last_cache_update > self._cache_ttl):
            
            logger.info("Fetching fresh validator information...")
            result = self._make_rpc_call("suix_getLatestSuiSystemState", [])
            
            # Reset cache
            self._validator_info_cache = {}
            
            # Update cache with new data
            active_validators = result.get('activeValidators', [])
            logger.debug(f"Active validators: {active_validators}")
            if active_validators:
                for validator in active_validators:
                    address = validator.get('suiAddress')
                    if address:
                        self._validator_info_cache[address] = {
                            'name': validator.get('name', 'Unknown'),
                            'description': validator.get('description', ''),
                            'image_url': validator.get('imageUrl', ''),
                            'project_url': validator.get('projectUrl', ''),
                            'voting_power': validator.get('votingPower', '0'),
                            'commission_rate': validator.get('commissionRate', '0'),
                            'staking_pool_id': validator.get('stakingPoolId', ''),
                            'has_voted': False  # Initialize voting status
                        }
            
            self._last_cache_update = now
            logger.info(f"Updated validator cache with {len(self._validator_info_cache)} entries")
        
        return self._validator_info_cache
    
    def get_validator_name(self, address: str) -> str:
        """Get a validator's name from their address.
        
        Args:
            address: The validator's address
            
        Returns:
            The validator's name or a shortened address if not found
        """
        validator_info = self.get_validator_info_map().get(address, {})
        if validator_info and validator_info.get('name'):
            return validator_info['name']
        
        # If no name found, return shortened address
        return f"{address[:6]}...{address[-4:]}"
    
    def update_voting_status(self, voting_addresses: List[str]) -> None:
        """Update the voting status of validators.
        
        Args:
            voting_addresses: List of validator addresses that have voted
        """
        validator_map = self.get_validator_info_map()
        for address in validator_map:
            validator_map[address]['has_voted'] = address in voting_addresses
    
    def get_voting_stats(self) -> Dict[str, int]:
        """Get statistics about validator voting.
        
        Returns:
            Dictionary containing voting statistics
        """
        validator_map = self.get_validator_info_map()
        total_validators = len(validator_map)
        voted_count = sum(1 for v in validator_map.values() if v['has_voted'])
        
        return {
            'total_validators': total_validators,
            'voted_count': voted_count,
            'not_voted_count': total_validators - voted_count
        }
    
    def get_events_by_module(
        self,
        module_name: str = "governance_voting",
        event_type: str = "ValidatorVoted",
        cursor: Optional[str] = None,
        limit: int = 100,
        descending: bool = True,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get voting events from the governance module.
        
        Args:
            module_name: Name of the module to query events from (default: governance_voting)
            event_type: Type of event to query (default: ValidatorVoted)
            cursor: Pagination cursor
            limit: Maximum number of events to return
            descending: Whether to return events in descending order
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            
        Returns:
            Dictionary containing events and pagination info
        """
        logger.debug(f"\nQuerying events with parameters:")
        logger.debug(f"Module: {module_name}")
        logger.debug(f"Event type: {event_type}")
        logger.debug(f"Package ID: {self.package_id}")
        
        # Create event filter query using MoveEventModuleQuery
        event_filter = MoveEventModuleQuery(
            module=module_name,
            package_id=self.package_id
        )
        
        logger.debug(f"\nEvent filter created: {event_filter}")
        
        # Get events using the proper query structure
        result = self.client.get_events(
            query=event_filter,
            cursor=cursor or "",
            limit=limit,
            descending_order=descending
        )
        
        logger.debug(f"\nQuery result status: {result.is_ok()}")
        if not result.is_ok():
            logger.error(f"Error: {result.result_string}")
            return {'error': result.result_string}
            
        logger.debug(f"Result data: {result.result_data}")
        
        # Transform the EventQueryEnvelope to a dictionary format
        if result.is_ok() and result.result_data:
            try:
                events_data = []
                # Access the data attribute directly from EventQueryEnvelope
                for event in result.result_data.data:
                    event_dict = {
                        'validator_address': event.parsed_json.get('validator_address'),
                        'vote': event.parsed_json.get('vote'),
                        'timestamp': event.timestamp_ms,
                        'transaction_module': event.transaction_module,
                        'event_type': event.event_type,
                        'event_id': event.event_id,
                        'sender': event.sender
                    }
                    events_data.append(event_dict)
                
                return {
                    'data': events_data,
                    'hasNextPage': result.result_data.has_next_page,
                    'nextCursor': result.result_data.next_cursor
                }
            except Exception as e:
                logger.error(f"Error processing events: {str(e)}")
                return {'error': f'Failed to process events: {str(e)}'}
            
        return {'error': 'No data available'}
    
    def get_validator_info(self, address: str) -> Dict[str, Any]:
        """Get information about a validator.
        
        Args:
            address: The validator's address
            
        Returns:
            Dictionary containing validator information
        """
        result = self.client.get_validator(SuiAddress(address))
        return result.result_data if result.is_ok() else {'error': result.result_string}
    
    def get_proposal_info(self, proposal_id: int) -> Dict[str, Any]:
        """Get information about a specific proposal.
        
        Args:
            proposal_id: The ID of the proposal
            
        Returns:
            Dictionary containing proposal information
        """
        try:
            # Get the governance object
            result = self.client.get_object(self.governance_object_id)
            
            if not result.is_ok():
                return {
                    "error": f"Failed to fetch governance object: {result.result_string}",
                    "proposal_id": proposal_id
                }
            
            gov_object = result.result_data
            if not gov_object or 'content' not in gov_object:
                return {
                    "error": "Governance object not found or invalid",
                    "proposal_id": proposal_id
                }
            
            # Extract proposal information from the governance object
            fields = gov_object['content']['fields']
            proposals = fields.get('proposals', [])
            
            # Find the specific proposal
            proposal = next(
                (p for p in proposals if p.get('id') == proposal_id),
                None
            )
            
            if not proposal:
                return {
                    "error": "Proposal not found",
                    "proposal_id": proposal_id
                }
            
            # Get all votes for this proposal using proper event query
            votes_query = MoveEventModuleQuery(
                module="governance_voting",
                package_id=self.package_id
            )
            
            votes_result = self.client.get_events(
                query=votes_query,
                cursor="",
                limit=1000,  # Increase if needed
                descending_order=True
            )
            
            # Count votes
            vote_counts = {"Yes": 0, "No": 0, "Abstain": 0}
            voters = set()
            
            if votes_result.is_ok():
                for event in votes_result.result_data.get('data', []):
                    if 'parsedJson' in event:
                        vote = event['parsedJson'].get('vote')
                        voter = event['parsedJson'].get('validator_address')
                        if vote in vote_counts and voter not in voters:
                            vote_counts[vote] += 1
                            voters.add(voter)
            
            return {
                "proposal_id": proposal_id,
                "title": proposal.get('title', ''),
                "description": proposal.get('description', ''),
                "status": proposal.get('status', 'Unknown'),
                "created_at": proposal.get('created_at', ''),
                "voting_end_time": proposal.get('voting_end_time', ''),
                "proposer": proposal.get('proposer', ''),
                "votes": {
                    "yes": vote_counts["Yes"],
                    "no": vote_counts["No"],
                    "abstain": vote_counts["Abstain"],
                    "total_voters": len(voters)
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to fetch proposal: {str(e)}",
                "proposal_id": proposal_id
            }
            
    # def subscribe_to_votes(self, handler) -> None:
        # """Subscribe to voting events in real-time.
        # 
        # Args:
            # handler: Callback function to handle new vote events
            # 
        # Note: This requires an async client, which we'll implement if needed
        # """
        # raise NotImplementedError(
            # "Real-time vote subscription requires async client implementation"
        # ) 