"""Sui client wrapper for interacting with the Sui blockchain."""

import os
import json
import requests
import time
from typing import List, Dict, Any, Optional
from pysui.sui.sui_clients.sync_client import SuiClient
from pysui.sui.sui_types.address import SuiAddress
from pysui.sui.sui_types.collections import SuiMap, EventID
from pysui.sui.sui_config import SuiConfig
from pysui.sui.sui_types.event_filter import MoveEventModuleQuery
from datetime import datetime, timedelta, UTC
from dotenv import load_dotenv
import logging
from .types import ValidatorInfo, VoteEvent, RPCResponse
from .constants import (
    DEFAULT_RPC_URL,
    DEFAULT_GOVERNANCE_OBJECT_ID,
    RPC_BATCH_LIMIT,
    RPC_RETRY_ATTEMPTS,
    RPC_RETRY_DELAY
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
        self.package_id = os.getenv('PACKAGE_ID', DEFAULT_GOVERNANCE_OBJECT_ID)
        
        logger.info(f"Initialized SuiClientWrapper with RPC URL: {self.rpc_url}")
        logger.info(f"Package ID: {self.package_id}")
        self.governance_object_id = DEFAULT_GOVERNANCE_OBJECT_ID
        
        try:
            config = SuiConfig.user_config(rpc_url=self.rpc_url)
            self.client = SuiClient(config)
            # Test the connection
            result = self._make_rpc_call("suix_getLatestSuiSystemState", [])
            if not result:
                logger.error("Failed to connect to Sui RPC endpoint")
            else:
                logger.info("Successfully connected to Sui RPC endpoint")
        except Exception as e:
            logger.error(f"Error initializing Sui client: {str(e)}")
            # Don't raise the error, let the app continue with degraded functionality
        
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make a JSON-RPC call to the Sui network.
        
        Args:
            method: The RPC method name
            params: List of parameters for the method
            
        Returns:
            The JSON response from the RPC call
        """
        for attempt in range(RPC_RETRY_ATTEMPTS):
            try:
                # Validate params to ensure no null values
                validated_params = ['' if param is None else param for param in params]
                
                logger.info(f"Making RPC call to {self.rpc_url} (attempt {attempt + 1}/{RPC_RETRY_ATTEMPTS})")
                logger.info(f"Method: {method}")
                logger.info(f"Params: {validated_params}")
                
                response = requests.post(
                    self.rpc_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": validated_params
                    },
                    headers={
                        "Content-Type": "application/json"
                    },
                    timeout=10  # Add timeout
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"RPC response status: {response.status_code}")
                logger.debug(f"RPC response: {result}")
                
                if "error" in result:
                    error_msg = result.get('error', {})
                    logger.error(f"RPC error: {error_msg}")
                    logger.error(f"Method: {method}")
                    logger.error(f"Params: {validated_params}")
                    if attempt < RPC_RETRY_ATTEMPTS - 1:
                        time.sleep(RPC_RETRY_DELAY * (attempt + 1))  # Exponential backoff
                        continue
                    return {}
                    
                return result.get("result", {})
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error making RPC call: {str(e)}")
                logger.error(f"URL: {self.rpc_url}")
                logger.error(f"Method: {method}")
                if attempt < RPC_RETRY_ATTEMPTS - 1:
                    time.sleep(RPC_RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
                return {}
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in RPC response: {str(e)}")
                logger.error(f"Response text: {response.text if 'response' in locals() else 'No response'}")
                if attempt < RPC_RETRY_ATTEMPTS - 1:
                    time.sleep(RPC_RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
                return {}
            except Exception as e:
                logger.error(f"Unexpected error in RPC call: {str(e)}")
                logger.error(f"Method: {method}")
                logger.error(f"Params: {params}")
                if attempt < RPC_RETRY_ATTEMPTS - 1:
                    time.sleep(RPC_RETRY_DELAY * (attempt + 1))  # Exponential backoff
                    continue
                return {}
        
        return {}
        
    def get_validator_info_map(self) -> Dict[str, Dict[str, Any]]:
        """Get a mapping of validator addresses to their information.
            
        Returns:
            Dictionary mapping validator addresses to their information
        """
        logger.info("Fetching validator information...")
        try:
            result = self._make_rpc_call("suix_getLatestSuiSystemState", [])
            if not result:
                logger.error("Failed to get system state - empty response")
                return {}

            logger.info(f"System state response: {result.keys()}")
            validator_info = {}
            
            # Get validator data
            active_validators = result.get('activeValidators', [])
            logger.info(f"Active validators found: {len(active_validators)}")
            
            if not active_validators:
                logger.error("No active validators found in system state")
                logger.error(f"System state content: {result}")
                return {}
                
            for validator in active_validators:
                if not isinstance(validator, dict):
                    logger.warning(f"Invalid validator data type: {type(validator)}")
                    logger.warning(f"Validator data: {validator}")
                    continue
                    
                address = validator.get('suiAddress')
                if not address:
                    logger.warning("Validator missing address")
                    logger.warning(f"Validator data: {validator}")
                    continue
                    
                voting_power = validator.get('votingPower', '0')
                logger.debug(f"Processing validator {address} with power {voting_power}")
                
                validator_info[address] = {
                    'name': validator.get('name', 'Unknown'),
                    'description': validator.get('description', ''),
                    'image_url': validator.get('imageUrl', ''),
                    'project_url': validator.get('projectUrl', ''),
                    'voting_power': voting_power,
                    'commission_rate': validator.get('commissionRate', '0'),
                    'staking_pool_id': validator.get('stakingPoolId', ''),
                    'has_voted': False  # Initialize voting status
                }
            
            logger.info(f"Retrieved {len(validator_info)} validator entries")
            logger.info(f"Total voting power: {sum(int(info.get('voting_power', 0)) for info in validator_info.values())}")
            return validator_info
            
        except Exception as e:
            logger.error(f"Error fetching validator info: {str(e)}")
            logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
            logger.error(f"Result content: {result if 'result' in locals() else 'No result'}")
            return {}
    
    def get_validator_name(self, address: str) -> str:
        """Get a validator's name from their address.
        
        Args:
            address: The validator's address
            
        Returns:
            The validator's name or a shortened address if not found
        """
        # If no name found, return shortened address
        if not address:
            return "Unknown"
            
        # Return shortened address format if not found
        return f"{address[:6]}...{address[-4:]}"
    
    def update_voting_status(self, voting_addresses: List[str]) -> None:
        """Update the voting status of validators.
        
        Args:
            voting_addresses: List of validator addresses that have voted
        """
        validator_map = self._make_rpc_call("suix_getLatestSuiSystemState", [])
        active_validators = validator_map.get('activeValidators', [])
        
        for validator in active_validators:
            address = validator.get('suiAddress')
            if address:
                validator['has_voted'] = address in voting_addresses
    
    def get_voting_stats(self) -> Dict[str, int]:
        """Get statistics about validator voting.
        
        Returns:
            Dictionary containing voting statistics
        """
        validator_map = self._make_rpc_call("suix_getLatestSuiSystemState", [])
        active_validators = validator_map.get('activeValidators', [])
        total_validators = len(active_validators)
        voted_count = sum(1 for v in active_validators if v.get('has_voted', False))
        
        return {
            'total_validators': total_validators,
            'voted_count': voted_count,
            'not_voted_count': total_validators - voted_count
        }
    
    def get_events_by_module(
        self,
        module_name: str = "governance_voting",
        event_type: str = "ValidatorVoted",
        cursor: Optional[Any] = None,
        limit: int = 100,
        descending: bool = True,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get voting events from the governance module."""
        logger.debug(f"\nQuerying events with parameters:")
        logger.debug(f"Module: {module_name}")
        logger.debug(f"Event type: {event_type}")
        logger.debug(f"Package ID: {self.package_id}")
        logger.debug(f"Cursor type: {type(cursor)}")
        logger.debug(f"Cursor value: {cursor}")
        
        # Create event filter query using MoveEventModuleQuery
        event_filter = MoveEventModuleQuery(
            module=module_name,
            package_id=self.package_id
        )
        
        logger.debug(f"\nEvent filter created: {event_filter}")
        
        try:
            # Get events using the proper query structure
            result = self.client.get_events(
                query=event_filter,
                cursor=cursor if cursor is not None else "",
                limit=limit,
                descending_order=descending
            )
            
            logger.debug(f"\nQuery result status: {result.is_ok()}")
            if not result.is_ok():
                logger.error(f"Error: {result.result_string}")
                return {'error': result.result_string}
            
            # Add detailed logging of the result structure
            logger.debug("Result data structure:")
            logger.debug(f"Result type: {type(result.result_data)}")
            if result.result_data and hasattr(result.result_data, 'data'):
                logger.debug(f"First event type: {type(result.result_data.data[0]) if result.result_data.data else 'No events'}")
                if result.result_data.data:
                    logger.debug(f"First event structure: {vars(result.result_data.data[0])}")
            
            # Transform the EventQueryEnvelope to a dictionary format
            if result.is_ok() and result.result_data:
                try:
                    events_data = []
                    for event in result.result_data.data:
                        # Log the event structure
                        logger.debug(f"Processing event: {vars(event)}")
                        logger.debug(f"Event ID type: {type(event.event_id)}")
                        logger.debug(f"Event ID structure: {vars(event.event_id) if hasattr(event.event_id, '__dict__') else event.event_id}")
                        
                        # Get and validate vote value
                        vote = event.parsed_json.get('vote')
                        if vote not in ['Yes', 'No', 'Abstain']:
                            logger.warning(f"Invalid vote value: {vote}, event: {vars(event)}")
                            continue
                        
                        event_dict = {
                            'validator_address': event.parsed_json.get('validator_address'),
                            'vote': vote,
                            'timestamp': event.timestamp_ms,
                            'transaction_module': event.transaction_module,
                            'event_type': event.event_type,
                            'event_id': event.event_id,  # Store the raw event_id for now
                            'sender': event.sender
                        }
                        events_data.append(event_dict)
                    
                    return {
                        'data': events_data,
                        'hasNextPage': result.result_data.has_next_page,
                        'nextCursor': result.result_data.next_cursor  # Store the raw cursor
                    }
                except Exception as e:
                    logger.error(f"Error processing events: {str(e)}")
                    logger.error(f"Event being processed: {vars(event) if 'event' in locals() else 'No event'}")
                    return {'error': f'Failed to process events: {str(e)}'}
        except Exception as e:
            logger.error(f"Error in get_events: {str(e)}")
            logger.error(f"Parameters: module={module_name}, event_type={event_type}, cursor={cursor}, limit={limit}")
            return {'error': f'Failed to get events: {str(e)}'}
            
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