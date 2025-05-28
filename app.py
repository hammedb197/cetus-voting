"""Streamlit dashboard for Sui governance monitoring."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import GovernanceDataFetcher, VotingAnalytics
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cetus_vote_dashboard')

# Page config
st.set_page_config(
    page_title="Cetus Recovery Vote Dashboard",
    page_icon="🗳️",
    layout="wide"
)

# Add auto-refresh
st.markdown("""
    <meta http-equiv="refresh" content="300">
    <style>
        footer {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# Initialize data fetcher in session state
if 'data_fetcher' not in st.session_state:
    logger.info("Initializing data fetcher")
    st.session_state.data_fetcher = GovernanceDataFetcher()

# Initialize data cache in session state if not present
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(minutes=6)
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'voting_summary' not in st.session_state:
    st.session_state.voting_summary = {}
if 'voting_power_analysis' not in st.session_state:
    st.session_state.voting_power_analysis = {}

# Update data every 5 minutes
if (datetime.now() - st.session_state.last_update) > timedelta(minutes=5):
    try:
        logger.info("Fetching new data")
        st.session_state.df = st.session_state.data_fetcher.get_recent_votes()
        st.session_state.voting_summary = st.session_state.data_fetcher.get_voting_summary()
        if not st.session_state.df.empty:
            st.session_state.voting_power_analysis = st.session_state.data_fetcher.analyze_voting_power(st.session_state.df)
        st.session_state.last_update = datetime.now()
        logger.info("Data update completed successfully")
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        st.error(f"Error updating data: {str(e)}")

# Get data from session state
df = st.session_state.df
voting_summary = st.session_state.voting_summary
voting_power_analysis = st.session_state.voting_power_analysis

# Title and description
st.title("🗳️ Cetus Recovery Vote Dashboard")
st.markdown("""
### Proposal: Protocol Upgrade to Return Hacked Funds

This dashboard tracks the voting progress for the protocol upgrade proposal to return the funds frozen from the Cetus hack.
The funds will be transferred to a 4-of-6 multisig wallet controlled by:
- Cetus (2 keys)
- Sui Foundation (2 keys)
- OtterSec (2 keys)

**Key Information:**
- Voting Period: May 27th 1:00 PM PST - June 3rd 11:30 AM PST
- Early Completion Criteria:
  - Minimum 2 days of voting completed
  - More than 50% of total stake participated in Yes/No votes (excluding abstain)
  - "Yes" votes exceed the sum of "No" votes and remaining non-participating stake
- Quorum Requirement: >50% of total stake must participate with Yes/No votes (excluding abstain)
- Approval Requirement: "Yes" votes must exceed "No" votes
""")

# Voting Status Banner
status = voting_power_analysis['voting_status']
status_color = {
    "Not Started": "blue",
    "In Progress": "orange",
    "Can End Early": "green",
    "Ended": "gray"
}[status['status']]

# Main content
if df.empty and status['status'] == "Not Started":
    st.info("Voting has not started yet. Check back after May 27th 1:00 PM PST.")
elif df.empty:
    st.warning("No voting events found in the selected time range.")
else:
    # Time-based Vote Distribution
    st.markdown("### 📊 Voting Timeline")
    
    # Create time-based visualization if we have data
    if not df.empty:
        # View selector
        view_type = st.radio(
            "Select Time View",
            ["Hourly", "Daily"],
            horizontal=True
        )
        
        # Get time-based data and create visualization
        cumulative_votes, formatted_dates = VotingAnalytics.prepare_time_based_data(df, view_type)
        fig = VotingAnalytics.create_timeline_figure(cumulative_votes, formatted_dates)
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Add key statistics
    stats_cols = st.columns(5)
    
    with stats_cols[0]:
        total_validators = len(df['validator_address'].unique())
        total_possible_validators = len(voting_summary['not_voted_validators']) + total_validators
        st.metric(
            "Validator Votes",
            f"{total_validators} of {total_possible_validators}",
            help="Number of validators who have voted out of total validators"
        )
    
    with stats_cols[1]:
        yes_votes = len(df[df['vote'] == 'Yes'])
        st.metric(
            "Yes Votes",
            yes_votes,
            help="Total number of Yes votes"
        )
    
    with stats_cols[2]:
        no_votes = len(df[df['vote'] == 'No'])
        st.metric(
            "No Votes",
            no_votes,
            help="Total number of No votes"
        )
    
    with stats_cols[3]:
        abstain_votes = len(df[df['vote'] == 'Abstain'])
        st.metric(
            "Abstain",
            abstain_votes,
            help="Total number of Abstain votes"
        )
    
    with stats_cols[4]:
        not_voted = len(voting_summary['not_voted_validators'])
        st.metric(
            "Have Not Voted",
            not_voted,
            help="Number of validators who haven't voted yet"
        )
    
    # Voting Progress
    st.markdown("### 📊 Voting Progress")
    progress_cols = st.columns(2)
    
    with progress_cols[0]:
        # Quorum Progress
        st.markdown("#### Quorum Progress (>50% Required)")
        quorum_pct = (voting_power_analysis['participating_power'] / 
                     voting_power_analysis['total_power_excl_abstain'] * 100)
        st.progress(min(quorum_pct / 100, 1.0))
        st.markdown(f"Current: {quorum_pct:.1f}% of total stake participating")
        
        if voting_power_analysis['quorum_reached']:
            st.success("✅ Quorum requirement met")
        else:
            st.warning(f"⏳ Need {voting_power_analysis['quorum_threshold'] - voting_power_analysis['participating_power']:,} more voting power for quorum")
    
    with progress_cols[1]:
        # Approval Progress
        st.markdown("#### Approval Progress")
        yes_power = voting_power_analysis['vote_power']['Yes']
        no_power = voting_power_analysis['vote_power']['No']
        total_yes_no = yes_power + no_power
        
        if total_yes_no > 0:
            yes_pct = (yes_power / total_yes_no * 100)
            st.progress(min(yes_pct / 100, 1.0))
            st.markdown(f"Yes: {yes_pct:.1f}% vs No: {100-yes_pct:.1f}% of voted stake")
            
            if yes_power > no_power:
                st.success("✅ Currently passing")
            else:
                st.error("❌ Currently not passing")
        else:
            st.info("No Yes/No votes cast yet")
    
    # Voting Power Distribution
    st.markdown("### 📈 Vote Distribution & Impact")
    dist_cols = st.columns([2, 1])
    
    with dist_cols[0]:
        # Create power distribution visualization
        power_data = voting_power_analysis['vote_power']
        remaining_power = voting_power_analysis['remaining_power']
        fig, yes_percent, no_percent = VotingAnalytics.create_power_distribution_figure(
            power_data,
            remaining_power
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_cols[1]:
        st.markdown("#### Vote Impact Analysis")
        
        # Get and display metrics
        metrics = VotingAnalytics.calculate_vote_metrics(power_data, remaining_power)
        for label, (value, help_text) in metrics.items():
            st.metric(
                label,
                VotingAnalytics.format_power(value),
                help=help_text
            )
        
        # Simple outcome status
        if remaining_power > abs(power_data['Yes'] - power_data['No']):
            st.warning("⚠️ Remaining votes could change the current outcome")
        else:
            st.success("✅ Current outcome is final")
    
    # Voting Activity Section
    st.markdown("### 📊 Validator Activity")
    
    # Create two columns
    vote_cols = st.columns(2)
    
    # Recent Votes Column
    with vote_cols[0]:
        st.markdown("#### Recent Votes")
        if not df.empty:
            vote_df = VotingAnalytics.prepare_validator_activity(df)
            st.dataframe(
                vote_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time": st.column_config.TextColumn(
                        "Time",
                        width="medium",
                        help="When the vote was cast"
                    ),
                    "Validator": st.column_config.TextColumn(
                        "Validator",
                        width="medium",
                        help="Name of the validator"
                    ),
                    "Address": st.column_config.TextColumn(
                        "Address",
                        width="large",
                        help="Validator's address"
                    ),
                    "Vote": st.column_config.TextColumn(
                        "Vote",
                        width="small",
                        help="The validator's vote"
                    )
                }
            )
        else:
            st.info("No votes cast yet")
    
    # Non-Voting Validators Column
    with vote_cols[1]:
        st.markdown("#### Non-Voted Validators")
        if voting_summary['not_voted_validators']:
            display_df = VotingAnalytics.prepare_non_voted_validators(
                voting_summary['not_voted_validators'],
                voting_summary['total_voting_power']
            )
            if display_df is not None:
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Validator": st.column_config.TextColumn(
                            "Validator",
                            width="medium",
                            help="Validator name"
                        ),
                        "Address": st.column_config.TextColumn(
                            "Address",
                            width="large",
                            help="Validator address"
                        ),
                        "Voting Power": st.column_config.NumberColumn(
                            "Voting Power",
                            help="Absolute voting power"
                        ),
                        "Power %": st.column_config.TextColumn(
                            "Power %",
                            help="Percentage of total voting power"
                        )
                    }
                )
        else:
            st.success("All validators have voted! 🎉")

# Footer
st.markdown("---")
last_update_time = st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div style='text-align: center'>
    Page refreshes automatically every 5 minutes | Last updated: {last_update_time}<br>
    <small>
        For more information, see the <a href="https://blog.sui.io/cetus-incident-response-onchain-community-vote/" target="_blank">official Sui Foundation announcement</a> |
        View source code on <a href="https://github.com/hammedb197/cetus-voting" target="_blank">GitHub</a>
    </small>
</div>
""", unsafe_allow_html=True) 