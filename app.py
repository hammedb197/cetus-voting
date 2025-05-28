"""Streamlit dashboard for Sui governance monitoring."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from datetime import datetime, timedelta
from utils import GovernanceDataFetcher, GovernanceAnalytics
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
        st.session_state.df = st.session_state.data_fetcher.get_recent_votes(hours=720)
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
- Early Completion: Possible after 2 days if outcome cannot change
- Quorum Requirement: >50% of total stake (excluding abstain)
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
        df_time = df.copy()
        df_time['date'] = pd.to_datetime(df_time['timestamp'])
        
        # View selector
        view_type = st.radio(
            "Select Time View",
            ["Hourly", "Daily"],
            horizontal=True
        )
        
        # Sort the data by timestamp first
        df_time = df_time.sort_values('date')
        
        # Create time-based aggregation
        freq = 'h' if view_type == "Hourly" else 'D'
        
        # Group by time and vote, then count
        vote_counts = df_time.groupby([
            pd.Grouper(key='date', freq=freq),
            'vote'
        ]).size().reset_index(name='count')
        
        # Pivot the data
        vote_pivot = vote_counts.pivot(
            index='date',
            columns='vote',
            values='count'
        ).fillna(0)
        
        # Calculate cumulative sums
        cumulative_votes = vote_pivot.cumsum()
        
        # Ensure all vote types exist
        for vote_type in ['Yes', 'No', 'Abstain']:
            if vote_type not in cumulative_votes.columns:
                cumulative_votes[vote_type] = 0
        
        # Create the timeline chart
        fig = go.Figure()
        
        # Enhanced color scheme
        colors = {
            'Yes': '#00C853',    # Bright green
            'No': '#FF1744',     # Bright red
            'Abstain': '#78909C' # Blue-grey
        }
        
        # Format dates for display
        date_format = "%m-%d %H:%M" if view_type == "Hourly" else "%m-%d"
        formatted_dates = cumulative_votes.index.strftime(date_format)
        
        # Add lines for cumulative votes
        for vote_type in ['Yes', 'No', 'Abstain']:
            fig.add_trace(go.Scatter(
                name=vote_type,
                x=formatted_dates,
                y=cumulative_votes[vote_type],
                mode='lines',
                line=dict(
                    color=colors[vote_type],
                    width=3,
                    shape='spline',
                    smoothing=0.3
                ),
                hovertemplate=(
                    f"<b>{vote_type}</b><br>" +
                    "Time: %{x}<br>" +
                    "Votes: %{y}<br>" +
                    "<extra></extra>"
                )
            ))
        
        # Update layout with improved styling
        fig.update_layout(
            title='voting breakdown',
            xaxis_title=None,
            yaxis_title="Number of Votes",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                showline=False,
                linecolor='rgba(128, 128, 128, 0.2)',
                linewidth=1,
                tickfont=dict(size=12),
                tickangle=-45,
                type='category',  # Use category type for formatted dates
                categoryorder='array',
                categoryarray=formatted_dates
            ),
            yaxis=dict(
                gridwidth=1,
                showline=True,
                tickfont=dict(size=12),
                title_text="Number of Votes",
                rangemode='nonnegative'
            ),
            template="plotly_dark",  # Use dark theme
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Add key statistics
    stats_cols = st.columns(5)
    
    with stats_cols[0]:
        total_validators = len(df_time['validator_address'].unique())
        total_possible_validators = len(voting_summary['not_voted_validators']) + total_validators
        st.metric(
            "Validator Votes",
            f"{total_validators} of {total_possible_validators}",
            help="Number of validators who have voted out of total validators"
        )
    
    with stats_cols[1]:
        yes_votes = len(df_time[df_time['vote'] == 'Yes'])
        st.metric(
            "Yes Votes",
            yes_votes,
            help="Total number of Yes votes"
        )
    
    with stats_cols[2]:
        no_votes = len(df_time[df_time['vote'] == 'No'])
        st.metric(
            "No Votes",
            no_votes,
            help="Total number of No votes"
        )
    
    with stats_cols[3]:
        abstain_votes = len(df_time[df_time['vote'] == 'Abstain'])
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
        # Create two subplots: donut chart and bar chart
        fig = go.Figure()
        
        # Donut chart for overall distribution
        power_data = voting_power_analysis['vote_power']
        power_labels = ['Yes', 'No', 'Abstain', 'Not Voted']
        power_values = [
            power_data['Yes'],
            power_data['No'],
            power_data['Abstain'],
            voting_power_analysis['remaining_power']
        ]
        
        # Calculate percentages for the title
        total_power = sum(power_values)
        yes_percent = (power_data['Yes'] / total_power * 100) if total_power > 0 else 0
        no_percent = (power_data['No'] / total_power * 100) if total_power > 0 else 0
        
        fig.add_trace(go.Pie(
            labels=power_labels,
            values=power_values,
            hole=.4,
            textinfo='label+percent',
            marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6', '#bdc3c7']),
            domain=dict(x=[0, 1], y=[0, 1]),
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"Current Distribution (Yes: {yes_percent:.1f}% | No: {no_percent:.1f}%)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with dist_cols[1]:
        st.markdown("#### Vote Impact Analysis")
        
        # Calculate impact metrics
        total_cast = power_data['Yes'] + power_data['No']
        remaining = voting_power_analysis['remaining_power']
        
        # Format large numbers
        def format_power(value):
            if value >= 1_000_000:
                return f"{value/1_000_000:.1f}M"
            elif value >= 1_000:
                return f"{value/1_000:.1f}K"
            return str(value)
        
        # Show current state
        current_metrics = {
            "Yes Power": (power_data['Yes'], "Total voting power supporting the proposal"),
            "No Power": (power_data['No'], "Total voting power against the proposal"),
            "Remaining": (remaining, "Voting power yet to be cast"),
            "Difference": (abs(power_data['Yes'] - power_data['No']), "Absolute difference between Yes and No votes")
        }
        
        for label, (value, help_text) in current_metrics.items():
            st.metric(
                label,
                format_power(value),
                help=help_text
            )
        
        # Simple outcome status
        if remaining > abs(power_data['Yes'] - power_data['No']):
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
            # Prepare vote data
            vote_df = df[['timestamp', 'validator_name', 'validator_address', 'vote']].copy()
            vote_df['timestamp'] = pd.to_datetime(vote_df['timestamp'])
            vote_df = vote_df.sort_values('timestamp', ascending=False)
            
            # Rename and format columns
            vote_df.columns = ['Time', 'Validator', 'Address', 'Vote']
            vote_df['Time'] = vote_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Show the dataframe with custom formatting
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
            not_voted_df = pd.DataFrame(voting_summary['not_voted_validators'])
            
            # Ensure voting_power is numeric before calculations
            not_voted_df['voting_power'] = pd.to_numeric(not_voted_df['voting_power'])
            
            # Calculate power percentage
            not_voted_df['power_percentage'] = (
                not_voted_df['voting_power'] / voting_summary['total_voting_power'] * 100
            ).round(2)
            
            # Format the DataFrame
            display_df = not_voted_df.copy()
            display_df = display_df.rename(columns={
                'name': 'Validator',
                'address': 'Address',
                'voting_power': 'Voting Power',
                'power_percentage': 'Power %'
            })
            
            # Format the Power % column
            display_df['Power %'] = display_df['Power %'].astype(str) + '%'
            
            # Sort by voting power descending
            display_df = display_df.sort_values('Voting Power', ascending=False)
            
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
    Data updates every 5 minutes | Last updated: {last_update_time}<br>
    <small>For more information, see the <a href="https://blog.sui.io/cetus-incident-response-onchain-community-vote/" target="_blank">official Sui Foundation announcement</a></small>
</div>
""", unsafe_allow_html=True) 